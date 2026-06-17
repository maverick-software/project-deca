"""Stage 2 — Experience framing & multisensory integration (stub)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, nudge_latent, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s2_integration")
    z += 0.05 * latent_seed(ctx, "s1_perception")
    nudge_latent(z, 0.02)
    return trace(2, "experience_framing", modality_mix=float(z.mean()))
