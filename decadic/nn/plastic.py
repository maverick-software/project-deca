"""Growable, sparse, plastic MLP block — the substrate for A/B/C neuroplasticity.

A drop-in replacement for the two-layer ``nn.Sequential`` MLPs in the cognitive
stack that adds three opt-in mechanisms while keeping the block's *external*
input/output dimensions fixed (so it never breaks the body action space or the
State-Bus head contracts):

- A (Hebbian plasticity): a fast plastic trace added to each weight, gated by a
  neuromodulator (pleasure - pain). ``alpha == 0`` -> identical to the base MLP.
- B (dynamic sparse training): per-connection masks with prune/grow rewiring.
  ``density == 1.0`` -> all connections active -> parity.
- C (neuron growth): the hidden layer is allocated up to a ceiling but most
  neurons start dormant; growth wakes dormant neurons (function-preserving:
  outgoing weights start at zero so the block's output is unchanged on wake).

With all three neutral (alpha 0, density 1.0, every hidden neuron awake) the
forward pass is numerically identical to ``Linear -> GELU -> Dropout -> Linear``
(optionally preceded by a LayerNorm), guaranteeing parity with the baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

HEBB_CLIP = 5.0


@dataclass
class PlasticityFlags:
    """Build-time switches for the A/B/C neuroplasticity subsystems (all off = parity)."""

    plastic: bool = False
    alpha: float = 0.0
    sparse: bool = False
    density: float = 1.0
    growth: bool = False
    hidden_ceiling: int = 0
    max_neurons: int = 0

    @property
    def any_enabled(self) -> bool:
        return bool(self.plastic or self.sparse or self.growth)

    def ceiling_for(self, hidden: int) -> int:
        if not self.growth:
            return hidden
        return max(hidden, int(self.hidden_ceiling), int(self.max_neurons))

    @classmethod
    def from_env(cls) -> "PlasticityFlags":
        from decadic import config as _cfg

        return cls(
            plastic=_cfg.plasticity_enabled(),
            alpha=_cfg.plasticity_alpha(),
            sparse=_cfg.sparse_enabled(),
            density=_cfg.sparse_density(),
            growth=_cfg.growth_enabled(),
            hidden_ceiling=_cfg.growable_hidden_ceiling(),
            max_neurons=_cfg.max_neurons(),
        )


@dataclass
class PlasticityRuntimeState:
    """Per-bundle live state for the plasticity controllers (reset with the bundle)."""

    pc_ema: float | None = None
    cycles_since_rewire: int = 0
    cycles_since_growth: int = 0
    rewire_events: int = 0
    growth_events: int = 0
    frozen: bool = False
    max_neurons: int = 0
    density: float = 1.0
    structural_version: int = 0

    @classmethod
    def from_flags(cls, flags: "PlasticityFlags") -> "PlasticityRuntimeState":
        return cls(max_neurons=int(flags.max_neurons), density=float(flags.density))


class PlasticSparseGrowableMLP(nn.Module):
    """Two-layer MLP with Hebbian plasticity, sparse masks, and dormant-neuron growth."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_active: int,
        *,
        hidden_ceiling: int | None = None,
        dropout: float = 0.0,
        pre_layernorm: bool = False,
        plastic: bool = False,
        plastic_alpha: float = 0.0,
        sparse: bool = False,
        density: float = 1.0,
        growth: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        hidden_active = int(hidden_active)
        ceiling = int(hidden_ceiling) if (growth and hidden_ceiling) else hidden_active
        self.hidden_ceiling = max(hidden_active, ceiling)
        self.plastic = bool(plastic)
        self.sparse = bool(sparse)
        self.growth = bool(growth)
        self.density = float(density)

        H, I, O = self.hidden_ceiling, self.in_features, self.out_features
        self.pre_ln = nn.LayerNorm(I) if pre_layernorm else None

        self.l1_weight = nn.Parameter(torch.empty(H, I))
        self.l1_bias = nn.Parameter(torch.zeros(H))
        self.l2_weight = nn.Parameter(torch.empty(O, H))
        self.l2_bias = nn.Parameter(torch.zeros(O))
        nn.init.kaiming_uniform_(self.l1_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.l2_weight, a=math.sqrt(5))
        self._reset_bias(self.l1_weight, self.l1_bias)
        self._reset_bias(self.l2_weight, self.l2_bias)

        self.dropout = nn.Dropout(dropout)
        # Plastic overlay magnitude is a learnable coefficient (differentiable
        # plasticity); fixed at 0 and frozen when plasticity is disabled.
        self.alpha = nn.Parameter(
            torch.tensor(float(plastic_alpha) if self.plastic else 0.0),
            requires_grad=self.plastic,
        )

        # Non-trained state (saved with the module): Hebbian traces, connection
        # masks, and the per-hidden-neuron awake gate.
        self.register_buffer("hebb1", torch.zeros(H, I))
        self.register_buffer("hebb2", torch.zeros(O, H))
        self.register_buffer("mask1", torch.ones(H, I))
        self.register_buffer("mask2", torch.ones(O, H))
        self.register_buffer("awake", torch.zeros(H))
        self.awake[:hidden_active] = 1.0

        if self.sparse and self.density < 1.0:
            self._seed_sparse_masks()
        self._apply_awake_to_masks()
        self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        with torch.no_grad():
            self._zero_dormant_and_pruned()

    @staticmethod
    def _reset_bias(weight: torch.Tensor, bias: torch.Tensor) -> None:
        fan_in = weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(bias, -bound, bound)

    # --- B: sparse mask seeding -------------------------------------------------
    def _seed_sparse_masks(self) -> None:
        for mask in (self.mask1, self.mask2):
            keep = torch.rand_like(mask) < self.density
            mask.copy_(keep.float())

    def _apply_awake_to_masks(self) -> None:
        self.mask1.mul_(self.awake.unsqueeze(1))
        self.mask2.mul_(self.awake.unsqueeze(0))

    def _zero_dormant_and_pruned(self) -> None:
        """Force pruned/dormant weights to exactly zero (no Adam drift)."""
        gate1 = self.mask1 * self.awake.unsqueeze(1)
        gate2 = self.mask2 * self.awake.unsqueeze(0)
        self.l1_weight.mul_(gate1)
        self.l1_bias.mul_(self.awake)
        self.l2_weight.mul_(gate2)

    def enforce_masks(self) -> None:
        with torch.no_grad():
            self._zero_dormant_and_pruned()

    # --- forward ----------------------------------------------------------------
    def _eff_w1(self) -> torch.Tensor:
        w = self.l1_weight
        if self.plastic:
            w = w + self.alpha * self.hebb1
        return w * self.mask1

    def _eff_w2(self) -> torch.Tensor:
        w = self.l2_weight
        if self.plastic:
            w = w + self.alpha * self.hebb2
        return w * self.mask2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_ln is not None:
            x = self.pre_ln(x)
        h_pre = F.linear(x, self._eff_w1(), self.l1_bias * self.awake)
        h = F.gelu(h_pre) * self.awake
        h = self.dropout(h)
        y = F.linear(h, self._eff_w2(), self.l2_bias)
        if self.plastic:
            self._cache = (x.detach(), h.detach(), y.detach())
        return y

    # --- A: Hebbian update ------------------------------------------------------
    def hebbian_update(self, modulation: float, eta: float) -> None:
        if not self.plastic or self._cache is None:
            return
        x, h, y = self._cache
        with torch.no_grad():
            pre1, post1 = x.mean(0), h.mean(0)
            pre2, post2 = h.mean(0), y.mean(0)
            self.hebb1.mul_(1.0 - eta).add_(eta * modulation * torch.outer(post1, pre1))
            self.hebb2.mul_(1.0 - eta).add_(eta * modulation * torch.outer(post2, pre2))
            self.hebb1.mul_(self.mask1).clamp_(-HEBB_CLIP, HEBB_CLIP)
            self.hebb2.mul_(self.mask2).clamp_(-HEBB_CLIP, HEBB_CLIP)

    def reset_plastic_trace(self) -> None:
        with torch.no_grad():
            self.hebb1.zero_()
            self.hebb2.zero_()

    # --- B: rewire (prune weakest active, grow highest-gradient inactive) -------
    def rewire(self, fraction: float) -> int:
        if not self.sparse:
            return 0
        changed = 0
        for w, m in ((self.l1_weight, self.mask1), (self.l2_weight, self.mask2)):
            changed += self._rewire_matrix(w, m, fraction)
        with torch.no_grad():
            self._zero_dormant_and_pruned()
        return changed

    def _rewire_matrix(self, weight: nn.Parameter, mask: torch.Tensor, fraction: float) -> int:
        with torch.no_grad():
            allowed = self._allowed(mask)
            active = (mask > 0) & allowed
            n_active = int(active.sum().item())
            k = int(n_active * fraction)
            if k <= 0:
                return 0
            wmag = (weight.detach().abs()) * active.float()
            wmag[~active] = float("inf")
            prune_idx = torch.topk(wmag.flatten(), k, largest=False).indices
            inactive = (mask == 0) & allowed
            grad = weight.grad
            if grad is None:
                score = torch.rand_like(weight)
            else:
                score = grad.detach().abs()
            score = score * inactive.float()
            score[~inactive] = float("-inf")
            n_grow = min(k, int(inactive.sum().item()))
            if n_grow <= 0:
                return 0
            grow_idx = torch.topk(score.flatten(), n_grow, largest=True).indices
            flat_mask = mask.flatten()
            flat_w = weight.data.flatten()
            flat_mask[prune_idx[:n_grow]] = 0.0
            flat_w[prune_idx[:n_grow]] = 0.0
            flat_mask[grow_idx] = 1.0
            bound = 1.0 / math.sqrt(max(1, weight.shape[1]))
            flat_w[grow_idx] = torch.empty(n_grow, device=weight.device).uniform_(-bound, bound)
            return n_grow

    def _allowed(self, mask: torch.Tensor) -> torch.Tensor:
        """Connections touching only awake neurons (rewiring never targets dormant)."""
        if mask is self.mask1:
            return (self.awake.unsqueeze(1) > 0).expand_as(mask)
        return (self.awake.unsqueeze(0) > 0).expand_as(mask)

    # --- C: growth (wake dormant neurons, function-preserving) ------------------
    def grow(self, n: int) -> list[int]:
        if not self.growth or n <= 0:
            return []
        dormant = (self.awake == 0).nonzero(as_tuple=False).flatten().tolist()
        woken = dormant[: int(n)]
        with torch.no_grad():
            for idx in woken:
                self.awake[idx] = 1.0
                # Function-preserving: outgoing weights start at zero so the
                # block output is unchanged the moment the neuron wakes.
                self.l2_weight.data[:, idx] = 0.0
                self.hebb1[idx, :] = 0.0
                self.hebb2[:, idx] = 0.0
                bound = 1.0 / math.sqrt(max(1, self.in_features))
                self.l1_weight.data[idx, :].uniform_(-bound, bound)
                self.l1_bias.data[idx] = 0.0
                if self.sparse and self.density < 1.0:
                    self.mask1[idx, :] = (torch.rand(self.in_features, device=self.awake.device) < self.density).float()
                    self.mask2[:, idx] = (torch.rand(self.out_features, device=self.awake.device) < self.density).float()
                else:
                    self.mask1[idx, :] = 1.0
                    self.mask2[:, idx] = 1.0
            self._zero_dormant_and_pruned()
        return woken

    def sleep(self, n: int) -> list[int]:
        if n <= 0:
            return []
        awake_idx = (self.awake > 0).nonzero(as_tuple=False).flatten().tolist()
        # Keep at least one neuron awake; sleep the highest indices first.
        sleepable = awake_idx[1:] if len(awake_idx) > 1 else []
        victims = sleepable[-int(n):] if sleepable else []
        with torch.no_grad():
            for idx in victims:
                self.awake[idx] = 0.0
            self._zero_dormant_and_pruned()
        return victims

    def set_awake_ceiling(self, n: int) -> list[int]:
        n = max(1, min(self.hidden_ceiling, int(n)))
        cur = self.awake_count()
        if n > cur:
            return self.grow(n - cur)
        if n < cur:
            return self.sleep(cur - n)
        return []

    def reseed_density(self, density: float) -> bool:
        """Re-seed connection masks to a new target density over the awake region."""
        density = min(1.0, max(0.01, float(density)))
        if abs(density - self.density) < 1e-6:
            return False
        self.density = density
        with torch.no_grad():
            if density >= 1.0:
                self.mask1.fill_(1.0)
                self.mask2.fill_(1.0)
            else:
                self.mask1.copy_((torch.rand_like(self.mask1) < density).float())
                self.mask2.copy_((torch.rand_like(self.mask2) < density).float())
            self._apply_awake_to_masks()
            self._zero_dormant_and_pruned()
        return True

    # --- telemetry / introspection ---------------------------------------------
    def awake_count(self) -> int:
        return int(self.awake.sum().item())

    def active_connections(self) -> int:
        with torch.no_grad():
            g1 = (self.mask1 * self.awake.unsqueeze(1)) > 0
            g2 = (self.mask2 * self.awake.unsqueeze(0)) > 0
            return int(g1.sum().item() + g2.sum().item())

    def structural_params(self) -> list[nn.Parameter]:
        """Params whose optimizer state must be reset after a structural change."""
        return [self.l1_weight, self.l1_bias, self.l2_weight, self.l2_bias]

    def arch_meta(self) -> dict[str, Any]:
        return {
            "in": self.in_features,
            "out": self.out_features,
            "hidden_ceiling": self.hidden_ceiling,
            "hidden_active": self.awake_count(),
            "pre_layernorm": self.pre_ln is not None,
            "plastic": self.plastic,
            "sparse": self.sparse,
            "growth": self.growth,
            "density": self.density,
        }
