"""Stage 4 — Risk-utility evaluation & curiosity (stub)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s4_risk_utility")
    z += 0.05 * latent_seed(ctx, "s3_heuristic")
    nudge_latent(z, 0.02)
    risk_proxy = float(1.0 / (1.0 + ctx.viability.value / 25.0))
    return trace(4, "risk_utility", risk_proxy=risk_proxy)
