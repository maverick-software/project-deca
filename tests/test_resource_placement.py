"""Per-life resource randomization (anti-camping): determinism, bounds, per-life change."""

import math
import random

from decadic.embodiment.resource_placement import scatter_positions

_NAMES = ["prop_food_s1", "prop_water_w1", "prop_food_s2"]


def test_scatter_is_deterministic_for_a_fixed_seed():
    a = scatter_positions(_NAMES, random.Random(7), fence_radius=18.0, min_dist=3.0, margin=1.5)
    b = scatter_positions(_NAMES, random.Random(7), fence_radius=18.0, min_dist=3.0, margin=1.5)
    assert a == b
    assert set(a.keys()) == set(_NAMES)


def test_arena_respects_min_dist_and_fence():
    usable = 18.0 - 1.5
    out = scatter_positions(_NAMES, random.Random(123), fence_radius=18.0, min_dist=3.0, margin=1.5)
    for _name, (x, y) in out.items():
        d = math.hypot(x, y)
        assert d >= 3.0 - 1e-6, d  # never spawn relief on top of the origin
        assert d <= usable + 1e-6, d  # never outside the fence margin


def test_different_seeds_move_resources():
    a = scatter_positions(_NAMES, random.Random(1), fence_radius=18.0, min_dist=3.0, margin=1.5)
    b = scatter_positions(_NAMES, random.Random(2), fence_radius=18.0, min_dist=3.0, margin=1.5)
    assert a != b  # a new life scatters resources to new spots


def test_zone_mode_keeps_resources_inside_arena():
    zones = [(0.0, 8.0, 2.4), (8.0, 6.0, 2.0), (-7.0, -7.0, 2.2)]
    out = scatter_positions(
        _NAMES, random.Random(9), fence_radius=18.0, min_dist=3.0, margin=1.5,
        mode="zone", zones=zones,
    )
    usable = 18.0 - 1.5
    for _name, (x, y) in out.items():
        assert math.hypot(x, y) <= usable + 1e-6


def test_min_dist_cannot_crowd_out_the_disc():
    # Absurd min_dist is clamped so a placement is always returned.
    out = scatter_positions(["a"], random.Random(0), fence_radius=4.0, min_dist=100.0, margin=0.5)
    assert "a" in out and len(out) == 1
