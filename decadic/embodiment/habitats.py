"""Habitat descriptors, crowd config accessors, and zone-clamp math.

A *habitat* is a small circular zone on the floor that one scripted NPC is
confined to. The learner discovers these zones passively as it wanders (it has
motor babble, no spatial-curiosity drive), so the habitats are placed in-view
around the spawn origin, well inside the arena fence.

Everything here is pure (no MuJoCo, no I/O) so it is cheap to unit test. Config
accessors mirror the ``decadic/config.py`` env-var pattern.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Behavior labels assigned per habitat (consumed by npc_controller dispatch).
BEHAVIOR_FORAGE = "forage"  # seek/eat/drink in-zone (the parent also delivers)
BEHAVIOR_WANDER = "wander"  # stroll between random in-zone waypoints
BEHAVIOR_SIT = "sit"  # hold a seated pose
BEHAVIOR_SIT_STAND = "sit_stand"  # cycle between sitting and standing
BEHAVIOR_COMMUNICATE = "communicate"  # face a partner and gesture
BEHAVIOR_IDLE = "idle"  # stand still at the zone center

VALID_BEHAVIORS: tuple[str, ...] = (
    BEHAVIOR_FORAGE,
    BEHAVIOR_WANDER,
    BEHAVIOR_SIT,
    BEHAVIOR_SIT_STAND,
    BEHAVIOR_COMMUNICATE,
    BEHAVIOR_IDLE,
)


@dataclass(frozen=True)
class Habitat:
    """One NPC's confined zone and what it does there.

    ``center`` is the zone's floor XY; ``radius`` bounds both the NPC's roaming
    and where its co-located food/water spawn. ``face`` is an optional point a
    ``communicate`` NPC turns toward (its conversation partner). ``food`` /
    ``water`` are how many respawning consumables to scatter in the zone.
    """

    name: str
    center: tuple[float, float]
    radius: float
    behavior: str
    is_parent: bool = False
    face: tuple[float, float] | None = None
    food: int = 2
    water: int = 1


# Eight habitats arranged around the spawn origin, all within FENCE_RADIUS (18)
# and roughly in view, so a wandering learner crosses them. Exactly one is the
# parent (it provisions the learner); the rest are ambient demonstrators showing
# varied survival-relevant behaviors. The two communicate zones face each other.
_CHAT_A = (-3.0, -9.0)
_CHAT_B = (-6.5, -7.0)

DEFAULT_HABITATS: tuple[Habitat, ...] = (
    Habitat("parent_grove", (0.0, 8.0), 2.4, BEHAVIOR_FORAGE, is_parent=True, food=3, water=2),
    Habitat("sitting_rocks", (8.0, 6.0), 2.0, BEHAVIOR_SIT, food=1, water=1),
    Habitat("exercise_yard", (10.0, -2.0), 2.2, BEHAVIOR_SIT_STAND, food=1, water=1),
    Habitat("wander_field", (6.0, -8.0), 2.6, BEHAVIOR_WANDER, food=2, water=1),
    Habitat("chat_circle_a", _CHAT_A, 1.8, BEHAVIOR_COMMUNICATE, face=_CHAT_B, food=1, water=1),
    Habitat("chat_circle_b", _CHAT_B, 1.8, BEHAVIOR_COMMUNICATE, face=_CHAT_A, food=1, water=1),
    Habitat("rest_knoll", (-9.5, 1.0), 2.0, BEHAVIOR_IDLE, food=1, water=1),
    Habitat("forage_meadow", (-7.0, 8.0), 2.6, BEHAVIOR_FORAGE, food=3, water=2),
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def crowd_size() -> int:
    """How many NPCs/habitats to instantiate (clamped to the descriptor table)."""
    n = _env_int("DECADIC_CROWD_SIZE", len(DEFAULT_HABITATS))
    return max(1, min(len(DEFAULT_HABITATS), n))


def habitat_radius() -> float:
    """Optional global override for every zone radius (<=0 keeps per-zone radii)."""
    return _env_float("DECADIC_HABITAT_RADIUS", 0.0)


def parent_need_threshold() -> float:
    """Normalized reservoir level (0..1) at/below which the parent provisions."""
    return min(1.0, max(0.0, _env_float("DECADIC_PARENT_NEED_THRESHOLD", 0.45)))


def parent_fade_per_offer() -> float:
    """Per-offer decay of the effective threshold (help fades as the child grows)."""
    return min(1.0, max(0.0, _env_float("DECADIC_PARENT_FADE_PER_OFFER", 0.96)))


def parent_threshold_floor() -> float:
    """Floor the faded threshold never drops below (the parent never abandons)."""
    return min(1.0, max(0.0, _env_float("DECADIC_PARENT_THRESHOLD_FLOOR", 0.12)))


def parent_refractory_s() -> float:
    """Minimum sim-seconds between successive parental offers (a refractory)."""
    return max(0.0, _env_float("DECADIC_PARENT_REFRACTORY_S", 20.0))


def crowd_lod_distance() -> float:
    """Beyond this distance (m) from the learner an NPC is held static (LOD)."""
    return max(0.0, _env_float("DECADIC_CROWD_LOD_DISTANCE", 16.0))


def active_habitats(n: int | None = None) -> list[Habitat]:
    """The first ``n`` habitats (default ``crowd_size()``), with radius override."""
    count = crowd_size() if n is None else max(1, min(len(DEFAULT_HABITATS), n))
    override = habitat_radius()
    out: list[Habitat] = []
    for hab in DEFAULT_HABITATS[:count]:
        if override > 0.0:
            hab = Habitat(
                hab.name, hab.center, override, hab.behavior, hab.is_parent,
                hab.face, hab.food, hab.water,
            )
        out.append(hab)
    return out


def clamp_to_zone(
    x: float, y: float, center: tuple[float, float], radius: float
) -> tuple[float, float]:
    """Pull ``(x, y)`` back inside the disc of ``radius`` around ``center``."""
    dx, dy = x - center[0], y - center[1]
    dist = math.hypot(dx, dy)
    if dist <= radius or dist == 0.0:
        return (x, y)
    scale = radius / dist
    return (center[0] + dx * scale, center[1] + dy * scale)


def parent_effective_threshold(offers: int) -> float:
    """Need threshold after ``offers`` past deliveries (faded, floored)."""
    base = parent_need_threshold()
    faded = base * (parent_fade_per_offer() ** max(0, int(offers)))
    return max(parent_threshold_floor(), faded)
