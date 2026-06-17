"""Stage 6 — Emotional / physiological experience (stub)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s6_emotion")
    z += 0.05 * latent_seed(ctx, "s5_pre_norm")
    nudge_latent(z, 0.03)
    emo = ctx.state_bus.emotion_physio
    emo[2 : min(len(emo), 2 + len(z))] += 0.01 * z[: min(len(z), len(emo) - 2)]
    return trace(
        6,
        "emotion_physio",
        pain=ctx.state_bus.pain_scalar,
        pleasure=ctx.state_bus.pleasure_scalar,
    )
