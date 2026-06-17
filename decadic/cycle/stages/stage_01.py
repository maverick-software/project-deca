"""Stage 1 — Sensory perception (stub): reads maintained perceptual state."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    seed = latent_seed(ctx, "s1_perception")
    fused = ctx.perceptual.fused_stub_emb.astype("float32", copy=False)
    seed[: min(len(seed), len(fused))] += 0.1 * fused[: len(seed)]
    nudge_latent(seed, 0.01)
    return trace(1, "sensory_perception", fused_dim=len(fused))
