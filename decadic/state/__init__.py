"""State package exports.

Keep these imports lazy. ``decadic.config`` imports ``decadic.state.body_map``
for dimension constants during startup; eager package-level imports pull in
working memory, which imports config again and creates a circular import.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PerceptualState", "StateBus", "ViabilityState"]


def __getattr__(name: str) -> Any:
    if name == "PerceptualState":
        from decadic.state.perceptual_state import PerceptualState

        return PerceptualState
    if name == "StateBus":
        from decadic.state.state_bus import StateBus

        return StateBus
    if name == "ViabilityState":
        from decadic.state.viability import ViabilityState

        return ViabilityState
    raise AttributeError(name)
