"""WS-EXPAND E9 — discrete abstraction bottleneck (FSQ, no learned codebook).

Finite Scalar Quantization: project a latent onto a few dimensions, bound each
with tanh, and round each to a small fixed grid. The "codebook" is the implicit
product grid — nothing is learned as a dictionary, so the classic VQ failure
modes (codebook collapse, straight-through covariate shift, dead codes) are
structurally absent (the evidence-review reason FSQ was chosen over a growing
learned codebook).

Deca usage: the stack projects a DETACHED z5 into FSQ_DIMS dimensions each
cycle; the quantized code is the cycle's symbol; a next-code head
self-supervises one-step symbol dynamics. Gradients never reach the shared
trunk (behavior byte-identical); grounding accrues on the side as codes
co-occur with the existing ``predicts_*`` beliefs. Expression/comprehension
(the E12 language loop) stays gated on adaptive others existing.
"""

from __future__ import annotations

from typing import Any

# Per-dimension quantization levels. Product = implicit codebook size:
# 8*6*5*5*4 = 4800 codes over 5 dims — small, composable, fully utilizable.
FSQ_LEVELS: tuple[int, ...] = (8, 6, 5, 5, 4)
FSQ_DIMS = len(FSQ_LEVELS)


def fsq_quantize(x: "Any") -> "tuple[Any, Any]":
    """(quantized_vector, code_index) for a [B, FSQ_DIMS] tensor.

    Each bounded dim rounds to its grid; the straight-through estimator keeps
    the projection trainable by the next-code loss. Pure function of x.
    """
    import torch

    if x.shape[-1] != FSQ_DIMS:
        raise ValueError(f"expected [..., {FSQ_DIMS}], got {tuple(x.shape)}")
    z = torch.tanh(x)  # [-1, 1] per dim (saturates to EXACTLY +/-1 in fp32)
    qs = []
    idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    mult = 1
    for d, levels in enumerate(FSQ_LEVELS):
        half = (levels - 1) / 2.0
        scaled = z[:, d] * half
        if levels % 2 == 0:
            # Even level count: the symmetric grid lives on HALF-integers
            # (e.g. L=4 -> {-1.5,-0.5,0.5,1.5}). Plain rounding lands on 5
            # integer points and lets a saturated tanh (exactly 1.0) round to
            # 2/1.5 = 1.333 OFF the grid — caught by the quantizer test.
            rounded = torch.round(scaled - 0.5) + 0.5
        else:
            rounded = torch.round(scaled)
        rounded = torch.clamp(rounded, -half, half)  # never off-grid, even at tanh=+/-1
        # Straight-through: forward uses the grid point, backward the smooth path.
        q = (rounded - scaled).detach() + scaled
        qs.append(q / half)
        # (rounded + half) is exactly integer-valued for both parities.
        idx = idx + torch.round(rounded + half).long().clamp(0, levels - 1) * mult
        mult *= levels
    return torch.stack(qs, dim=-1), idx


def fsq_code_to_vector(code: int) -> list[float]:
    """WS-SYM 3.3: inverse of ``fsq_quantize``'s index -> the code's canonical
    normalized grid-point vector (length FSQ_DIMS, each in [-1, 1]).

    Recall returns a code *index*; to feed a recalled/own code back through the
    symbol ingress (which expects a FSQ_DIMS vector) we decode it to the exact
    quantized vector ``fsq_quantize`` would have produced. Pure function; the
    round-trip fsq_quantize(x) -> idx -> fsq_code_to_vector(idx) == quantized(x)
    is pinned by tests.
    """
    c = int(code)
    out: list[float] = []
    mult = 1
    for levels in FSQ_LEVELS:
        half = (levels - 1) / 2.0
        level_index = (c // mult) % levels
        rounded = level_index - half
        out.append(rounded / half if half > 0 else 0.0)
        mult *= levels
    return out


class CodeUsage:
    """Rolling code-utilization telemetry (distinct codes over a window)."""

    def __init__(self, window: int = 512) -> None:
        self.window = max(8, int(window))
        self._recent: list[int] = []
        self.total = 0

    def note(self, code: int) -> None:
        self._recent.append(int(code))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self.total += 1

    def telemetry(self) -> dict[str, int | float]:
        distinct = len(set(self._recent))
        return {
            "symbol_codes_seen": self.total,
            "symbol_distinct_recent": distinct,
            "symbol_utilization": round(distinct / max(1, len(self._recent)), 6),
        }
