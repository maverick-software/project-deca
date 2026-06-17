"""Stage 9 — Behavioral response (stub action selection)."""

from __future__ import annotations

from decadic.cycle.stages._helpers import latent_seed, trace
from decadic.cycle.types import CycleContext, StageTrace


def run(ctx: CycleContext) -> StageTrace:
    z = latent_seed(ctx, "s8_strategy")
    mean_z = float(z.mean())
    if ctx.state_bus.priority_label == "avoid":
        action_type = "move"
        direction = [-0.7, 0.0, -0.7]
        speed = 1.0
    else:
        action_type = "move"
        direction = [0.7, 0.0, 0.7]
        speed = 1.0
    action = {
        "type": action_type,
        "parameters": {"direction": direction, "speed": speed, "hint": mean_z},
    }
    predicted = {
        "embedding": z[:16].tolist(),
        "expected_position": ctx.perceptual.proprio_position or [0.0, 0.0, 0.0],
    }
    ctx.latents["action"] = action
    ctx.latents["predicted_outcome"] = predicted
    return trace(9, "behavioral_response", action_type=action_type)
