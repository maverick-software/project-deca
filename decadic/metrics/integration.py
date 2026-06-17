"""Perturbational integration proxy (PCI / Phi-style) over the cognitive stack.

Falsification track for the self-model program. A phase that adds a feedback
pathway must *prove* it raises integration, not just relabel outputs. This
module drives a :class:`~decadic.nn.neural_stack.NeuralCognitiveStack` with a
fixed synthetic percept, injects a single bounded perturbation (a "pulse") into
that percept, and scores how widely and how durably the pulse spreads across the
per-stage activations versus an unperturbed baseline.

The spread is summarized by the Lempel-Ziv complexity of the binarized
deviation matrix -- the core of the Perturbational Complexity Index (PCI).
Higher PCI == a perturbation that is both *differentiated* (many stages respond
differently) and *integrated* (the response persists across cycles via the
recurrent / self-feedback loops). A closed self-model loop should therefore
raise PCI relative to the same stack with the loop severed.

Notes
-----
- This probe *mutates and then resets* the stack's transient recurrent buffers,
  so call it on a freshly built or dedicated stack (as the tests do), never on a
  bundle's live stack mid-cycle.
- The probe runs under ``stack.eval()`` and ``torch.no_grad()`` for determinism
  (dropout off, no graph), restoring the prior training mode on exit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

# The canonical self-state vector the Phase-1 spine feeds back: state-of-mind
# (A), narrative (C), and metacognition (E) concatenated. Kept here so the probe
# and the live pipeline agree on what "the self-report" is.
SELF_STATE_KEYS = ("state_mind", "narrative", "metacognition")


def self_state_vector(out: dict) -> torch.Tensor:
    """Detached A||C||E vector from a stack ``forward`` output."""
    return torch.cat([out[k] for k in SELF_STATE_KEYS], dim=-1).detach()


def lz76_complexity(sequence) -> int:
    """Lempel-Ziv (1976) production complexity of a binary sequence.

    Accepts a string of ``"0"``/``"1"`` or any iterable of truthy/falsy values.
    Implements the Kaspar-Schuster parsing used throughout the PCI literature.
    """
    if isinstance(sequence, str):
        s = sequence
    else:
        s = "".join("1" if x else "0" for x in sequence)
    n = len(s)
    if n <= 1:
        return n
    i = 0
    c = 1
    ell = 1
    k = 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[ell + k - 1]:
            k += 1
            if ell + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == ell:
                c += 1
                ell += k_max
                if ell + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


@dataclass
class IntegrationResult:
    """Outcome of a perturbational-complexity probe."""

    pci: float  # LZ complexity normalized to [0, ~1]: lz * log2(n) / n
    lz: int  # raw Lempel-Ziv production count
    n_bits: int  # size of the binarized deviation matrix
    active_fraction: float  # fraction of deviation bits set (spread x persistence)
    persistence: float  # fraction of post-pulse cycles with any deviation
    stage_spread: float  # fraction of stages deviating at the final cycle
    self_feedback: bool  # whether the probe drove the self-state spine

    def as_dict(self) -> dict:
        return asdict(self)


def _fixed_inputs(stack, seed: int):
    cfg = stack.cfg
    dev = stack.ingress.weight.device
    dt = stack.ingress.weight.dtype
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    z0 = torch.randn(1, cfg.d_model, generator=g).to(device=dev, dtype=dt)
    ep = torch.rand(1, 4, generator=g).to(device=dev, dtype=dt)
    mem = torch.randn(1, cfg.memory_context_dim, generator=g).to(device=dev, dtype=dt)
    return z0, ep, mem


def _stage_activation_vector(out: dict) -> np.ndarray:
    """Flatten the per-stage 32-bin |activation| profiles into one vector."""
    rows = [m["activations"] for m in out["stage_metrics"]]
    return np.asarray([x for row in rows for x in row], dtype=np.float64)


def _run_trajectory(
    stack,
    z0: torch.Tensor,
    ep: torch.Tensor,
    mem: torch.Tensor,
    *,
    cycles: int,
    pulse: torch.Tensor | None,
    pulse_cycle: int,
    use_self: bool,
) -> np.ndarray:
    """Run ``cycles`` forwards from a clean recurrent state; return [cycles, F]."""
    stack.reset_recurrent_state()
    self_prev: torch.Tensor | None = None
    rows: list[np.ndarray] = []
    for c in range(cycles):
        z0_c = z0
        # Single impulse: injected only at pulse_cycle. Any persistence beyond
        # that cycle comes from recurrence + the self-feedback loop, which is
        # exactly the integration we want to measure.
        if pulse is not None and c == pulse_cycle:
            z0_c = z0 + pulse
        kwargs = {"self_prev": self_prev} if use_self else {}
        out = stack(z0_c, ep, mem, **kwargs)
        rows.append(_stage_activation_vector(out))
        if use_self:
            self_prev = self_state_vector(out)
    return np.stack(rows, axis=0)


def perturbational_complexity(
    stack,
    *,
    cycles: int = 12,
    pulse_cycle: int = 2,
    pulse_scale: float = 1.0,
    seed: int = 0,
    tau: float = 0.10,
) -> IntegrationResult:
    """Score how a single percept pulse spreads across the stack's stages.

    The stack is driven for ``cycles`` forwards twice from a clean recurrent
    state -- once unperturbed, once with a bounded pulse added to ``z0`` at
    ``pulse_cycle`` -- and the post-pulse deviation matrix is binarized
    (deviation > ``tau`` x mean-baseline-activation) and scored by normalized
    Lempel-Ziv complexity. If the stack exposes ``has_self_model_feedback`` the
    probe also closes the A||C||E spine each cycle, so the on-vs-off comparison
    reflects the real feedback pathway.
    """
    was_training = bool(stack.training)
    stack.eval()
    try:
        with torch.no_grad():
            z0, ep, mem = _fixed_inputs(stack, seed)
            use_self = bool(getattr(stack, "has_self_model_feedback", False))
            base = _run_trajectory(
                stack, z0, ep, mem, cycles=cycles, pulse=None,
                pulse_cycle=pulse_cycle, use_self=use_self,
            )
            g = torch.Generator(device="cpu").manual_seed(int(seed) + 9973)
            pulse = (torch.randn(1, stack.cfg.d_model, generator=g) * float(pulse_scale))
            pulse = pulse.to(device=z0.device, dtype=z0.dtype)
            pert = _run_trajectory(
                stack, z0, ep, mem, cycles=cycles, pulse=pulse,
                pulse_cycle=pulse_cycle, use_self=use_self,
            )
    finally:
        stack.train(was_training)
        stack.reset_recurrent_state()

    post = slice(pulse_cycle, cycles)
    b = base[post]
    p = pert[post]
    dev = np.abs(p - b)
    scale = float(np.mean(np.abs(b)) + 1e-6)
    bits = (dev > tau * scale).astype(np.uint8)
    flat = bits.flatten(order="C")
    n = int(flat.size)
    lz = lz76_complexity(flat.tolist())
    pci = (lz * math.log2(n) / n) if n > 1 else 0.0
    active_fraction = float(bits.mean()) if n else 0.0
    persistence = float((bits.sum(axis=1) > 0).mean()) if bits.size else 0.0
    stage_spread = 0.0
    if bits.shape[1] % 32 == 0 and bits.shape[1] > 0:
        last = bits[-1].reshape(-1, 32)
        stage_spread = float((last.sum(axis=1) > 0).mean())
    return IntegrationResult(
        pci=pci,
        lz=lz,
        n_bits=n,
        active_fraction=active_fraction,
        persistence=persistence,
        stage_spread=stage_spread,
        self_feedback=use_self,
    )


def integration_delta(
    off_stack, on_stack, **kwargs
) -> tuple[IntegrationResult, IntegrationResult, float]:
    """Convenience: (off_result, on_result, pci_on - pci_off) for ablations."""
    off = perturbational_complexity(off_stack, **kwargs)
    on = perturbational_complexity(on_stack, **kwargs)
    return off, on, on.pci - off.pci
