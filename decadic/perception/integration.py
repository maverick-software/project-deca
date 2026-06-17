"""Perceptual integration (Phase 1 stub → Phase 4 world model).

Separated into ``decadic.perception`` so the architectural boundary matches the plan:
streaming observations converge here before the Decadic Cycle samples stabilized state at stage 1.
"""

from __future__ import annotations

from typing import Any

from decadic.state.perceptual_state import PerceptualState


class PerceptualIntegrator:
    """Maintained perceptual synthesis driven by each inbound observation."""

    def integrate(self, perceptual: PerceptualState, observation: dict[str, Any]) -> None:
        perceptual.integrate_observation(observation)
