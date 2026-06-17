"""Stage 5 — Pre-normative conclusion development (stub)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s5_pre_norm")
    z += 0.07 * latent_seed(ctx, "s4_risk_utility")
    nudge_latent(z, 0.02)
    ctx.state_bus.narrative_emb[: min(len(ctx.state_bus.narrative_emb), len(z))] += 0.01 * z[
        : len(ctx.state_bus.narrative_emb)
    ]
    return trace(5, "pre_normative", narrative_touch=float(ctx.state_bus.narrative_emb.mean()))
