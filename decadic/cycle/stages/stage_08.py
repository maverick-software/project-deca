"""Stage 8 — Strategy formation (stub — emit latent policy hint)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s8_strategy")
    z += 0.06 * latent_seed(ctx, "s7_reprioritize")
    nudge_latent(z, 0.02)
    ctx.state_bus.metacognition[: min(len(ctx.state_bus.metacognition), len(z))] += 0.01 * z[
        : len(ctx.state_bus.metacognition)
    ]
    return trace(8, "strategy", hint=float(z.mean()))
