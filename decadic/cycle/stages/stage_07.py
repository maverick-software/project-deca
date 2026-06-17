"""Stage 7 — Reprioritization & update state of mind (stub)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s7_reprioritize")
    z += 0.06 * latent_seed(ctx, "s6_emotion")
    nudge_latent(z, 0.02)
    ctx.state_bus.state_of_mind[: min(len(ctx.state_bus.state_of_mind), len(z))] += 0.01 * z[
        : len(ctx.state_bus.state_of_mind)
    ]
    # Priority shifts slightly toward avoidance when pain dominates
    delta = float(ctx.state_bus.pain_scalar - ctx.state_bus.pleasure_scalar)
    ctx.state_bus.priority_scalar = float(
        max(0.0, min(1.0, ctx.state_bus.priority_scalar + 0.02 * delta))
    )
    ctx.state_bus.priority_label = "avoid" if delta > 0.15 else "explore"
    return trace(7, "reprioritize", priority_label=ctx.state_bus.priority_label)
