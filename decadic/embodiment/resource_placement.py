"""Pure resource-placement math for per-life anti-camping randomization.

Given a set of consumable names and a seeded RNG, choose fresh XY positions so
the *location* of relief (food/water) is never memorizable across lives -- only
the SKILL of seeking-and-reaching transfers, never "sit at the spot I know."

No MuJoCo / I/O here so it is cheap to unit test; the body adapter
(``scripts/mujoco_decadic_adapter.py``) applies the returned positions to
``model.body_pos`` and re-shows the props. Config accessors live in
``decadic/config.py`` (``randomize_resources_enabled`` etc.); this module only
does the geometry.

Two modes:
- ``arena``: scatter uniformly (by area) across the arena disc, kept at least
  ``min_dist`` from the spawn origin so the agent never spawns on top of relief.
- ``zone``: keep each resource inside a randomly chosen habitat zone (preserves
  the "resources live in habitats" flavor while still randomizing which/where).
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def _point_in_disc(
    rng: random.Random, radius: float, *, min_r: float = 0.0
) -> tuple[float, float]:
    """Uniform-by-area sample in the annulus ``[min_r, radius]`` about the origin."""
    radius = max(0.0, float(radius))
    min_r = max(0.0, min(float(min_r), radius))
    r = math.sqrt(rng.uniform(min_r * min_r, radius * radius))
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return (r * math.cos(theta), r * math.sin(theta))


def _clamp_to_arena(x: float, y: float, usable: float) -> tuple[float, float]:
    """Pull ``(x, y)`` back onto the usable disc of radius ``usable``."""
    d = math.hypot(x, y)
    if d <= usable or d == 0.0:
        return (x, y)
    s = usable / d
    return (x * s, y * s)


def scatter_positions(
    names: Sequence[str],
    rng: random.Random,
    *,
    fence_radius: float,
    min_dist: float = 0.0,
    margin: float = 0.0,
    mode: str = "arena",
    zones: "Sequence[tuple[float, float, float]] | None" = None,
) -> dict[str, tuple[float, float]]:
    """Return ``{name: (x, y)}`` fresh positions for every consumable in ``names``.

    ``fence_radius`` is the arena half-size the body must stay inside; ``margin``
    keeps resources that far inside the fence. ``min_dist`` (arena mode) keeps
    each resource at least that far from the origin. ``zones`` is a list of
    ``(cx, cy, radius)`` habitat discs, required only for ``zone`` mode.
    """
    usable = max(0.5, float(fence_radius) - max(0.0, float(margin)))
    # Never let min_dist crowd out the whole usable disc.
    min_dist = max(0.0, min(float(min_dist), usable - 0.5))
    out: dict[str, tuple[float, float]] = {}

    if mode == "zone" and zones:
        for name in names:
            cx, cy, zr = zones[rng.randrange(len(zones))]
            ox, oy = _point_in_disc(rng, max(0.3, float(zr)))
            x, y = _clamp_to_arena(cx + ox, cy + oy, usable)
            out[name] = (round(x, 3), round(y, 3))
        return out

    # arena (default): uniform over the usable disc, kept >= min_dist from origin.
    for name in names:
        x, y = _point_in_disc(rng, usable, min_r=min_dist)
        out[name] = (round(x, 3), round(y, 3))
    return out
