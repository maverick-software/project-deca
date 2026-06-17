"""Live loss-landscape probe (filter-normalized 2D slice of the brain's weights).

Visualizes the agent's real training objective as a surface ``L(a, b) = loss(
theta* + a*d1 + b*d2)`` over a 2D grid of two directions in weight space, with the
current weights ``theta*`` at the center. The directions are *filter-normalized*
random Gaussian directions (Li et al., 2018, "Visualizing the Loss Landscape of
Neural Nets"): each weight filter's direction is rescaled to the norm of the
corresponding weight filter, which removes the scale-invariance artifact that makes
raw random directions misleading.

Design notes:
- The probe owns its OWN throwaway clone of the live stack, so evaluating the grid
  never perturbs the live agent (or the consolidator's training clone).
- It reuses ``consolidator.replay_batch_loss`` so the plotted surface is the exact
  objective the agent trains on (predictive coding + forward models), not a proxy.
- The recurrent buffers are zeroed before every grid-point evaluation, so the loss
  is a pure function of (weights, batch, directions) and the surface does not depend
  on evaluation order.
- Random directions are generated once from a fixed seed (persisted on the probe) so
  the surface is comparable across refreshes; they are re-normalized against the
  current ``theta*`` each refresh, so the axis scale tracks the live weights.
"""

from __future__ import annotations

from typing import Any

import torch

from decadic.consolidation.consolidator import replay_batch_loss
from decadic.nn.neural_stack import NeuralCognitiveStack

# Safety cap so a misconfigured grid can never spin up a runaway number of
# forward passes (grid*grid*batch). 41x41 is already a very fine surface.
MAX_GRID = 41


class LossLandscapeProbe:
    """Owns a clone of the live stack and evaluates filter-normalized loss slices."""

    def __init__(self, bundle, *, seed: int | None = None) -> None:
        self.bundle = bundle
        self.device = bundle.device
        self.clone = self._clone_stack()
        self._seed = 0 if seed is None else int(seed)
        # Raw (un-normalized) Gaussian directions, generated once and reused so the
        # random basis is stable across refreshes. name -> CPU tensor.
        self._raw: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None

    # --- cloning ------------------------------------------------------------

    def _clone_stack(self) -> NeuralCognitiveStack:
        stack = NeuralCognitiveStack(
            self.bundle.cfg, self.bundle.flags, self.bundle.faculties
        ).to(self.device)
        stack.load_state_dict(self.bundle.stack.state_dict())
        return stack

    def _filter_params(self) -> list[tuple[str, torch.Tensor]]:
        """The perturbed parameters: weight matrices (dim>=2), the network's filters.

        1-D parameters (biases, norm gains) are held fixed, matching Li et al. and
        the codebase's "connections = params with dim>=2" convention.
        """
        return [(n, p) for n, p in self.clone.named_parameters() if p.dim() >= 2]

    # --- directions ---------------------------------------------------------

    @staticmethod
    def _filter_normalize(d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Rescale each filter (row, dim 0) of ``d`` to the norm of ``w``'s filter."""
        df = d.reshape(d.shape[0], -1)
        wf = w.reshape(w.shape[0], -1)
        scale = wf.norm(dim=1, keepdim=True) / (df.norm(dim=1, keepdim=True) + 1e-10)
        return (df * scale).reshape_as(d)

    def _ensure_raw_dirs(self, params: list[tuple[str, torch.Tensor]]) -> None:
        if self._raw is not None:
            return
        gen = torch.Generator().manual_seed(self._seed)
        d1 = {n: torch.randn(p.shape, generator=gen) for n, p in params}
        d2 = {n: torch.randn(p.shape, generator=gen) for n, p in params}
        self._raw = (d1, d2)

    def _normalized_dirs(
        self, theta: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        assert self._raw is not None
        raw1, raw2 = self._raw
        d1 = {n: self._filter_normalize(raw1[n].to(self.device), theta[n]) for n in theta}
        d2 = {n: self._filter_normalize(raw2[n].to(self.device), theta[n]) for n in theta}
        return d1, d2

    # --- grid evaluation ----------------------------------------------------

    @torch.no_grad()
    def compute(
        self,
        batch: list,
        *,
        grid: int = 15,
        span: float = 1.0,
        cycle: int = 0,
    ) -> dict[str, Any] | None:
        """Evaluate the filter-normalized loss surface. None if no batch / unstable.

        Returns a JSON-shaped dict: ``alphas``/``betas`` (grid coords), ``z`` (grid x
        grid losses, row-major over alpha then beta), ``center_loss`` (loss at
        theta*), plus ``grid``/``span``/``batch``/``cycle``/``preset`` metadata.
        """
        if not batch:
            return None
        grid = max(3, min(int(grid), MAX_GRID))
        span = float(abs(span)) or 1.0

        # Snapshot theta* from the live stack (it moves as the agent learns).
        self.clone.load_state_dict(self.bundle.stack.state_dict())
        self.clone.eval()
        params = self._filter_params()
        theta = {n: p.detach().clone() for n, p in params}
        self._ensure_raw_dirs(params)
        d1, d2 = self._normalized_dirs(theta)

        def eval_at(a: float, b: float) -> float:
            for n, p in params:
                p.copy_(theta[n] + a * d1[n] + b * d2[n])
            # Reset transient context so the loss depends only on weights + batch.
            self.clone.reset_recurrent_state()
            val = replay_batch_loss(self.clone, batch, self.device)
            return float(val.detach().cpu().item())

        center_loss = eval_at(0.0, 0.0)
        if not _finite(center_loss):
            self._restore(params, theta)
            return None

        coords = torch.linspace(-span, span, grid).tolist()
        z: list[list[float]] = []
        for a in coords:
            row: list[float] = []
            for b in coords:
                v = eval_at(a, b)
                row.append(v if _finite(v) else float("nan"))
            z.append(row)

        self._restore(params, theta)
        finite = [v for row in z for v in row if _finite(v)]
        return {
            "alphas": coords,
            "betas": coords,
            "z": z,
            "center_loss": center_loss,
            "z_min": min(finite) if finite else 0.0,
            "z_max": max(finite) if finite else 0.0,
            "grid": grid,
            "span": span,
            "batch": len(batch),
            "cycle": int(cycle),
            "preset": getattr(self.bundle, "preset", None),
        }

    def _restore(
        self, params: list[tuple[str, torch.Tensor]], theta: dict[str, torch.Tensor]
    ) -> None:
        for n, p in params:
            p.copy_(theta[n])
        self.clone.reset_recurrent_state()


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
