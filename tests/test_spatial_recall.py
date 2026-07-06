"""WS-FORAGE M4 — spatial recall: remembered target -> egocentric bearing.

Pure/torch-free. Covers the allocentric->egocentric transform on known geometry,
the need-conditioned graph query, the landmark fallback, and the defensive
degrade-to-None behaviour that keeps M4 a safe no-op.
"""

import math

from decadic.state.spatial_recall import (
    egocentric_bearing,
    resolve_goal_target,
)


# --------------------------------------------------------- bearing transform


def test_bearing_dead_ahead():
    c, s, d = egocentric_bearing([0, 0, 1], 0.0, [5, 0, 0], max_dist=10.0)
    assert c > 0.99 and abs(s) < 1e-6  # ahead
    assert d == 0.5  # 5 m of 10 m


def test_bearing_to_the_left_and_right():
    # Agent facing +x. Target at +y is to its LEFT (sin>0).
    c, s, _ = egocentric_bearing([0, 0, 1], 0.0, [0, 5, 0])
    assert abs(c) < 1e-6 and s > 0.99
    # Target at -y is to its RIGHT (sin<0).
    c2, s2, _ = egocentric_bearing([0, 0, 1], 0.0, [0, -5, 0])
    assert abs(c2) < 1e-6 and s2 < -0.99


def test_bearing_behind():
    c, s, _ = egocentric_bearing([0, 0, 1], 0.0, [-5, 0, 0])
    assert c < -0.99 and abs(s) < 1e-6  # behind


def test_bearing_accounts_for_heading():
    # Agent facing +y (yaw=pi/2); a target at world +x is to its RIGHT.
    c, s, _ = egocentric_bearing([0, 0, 1], math.pi / 2, [5, 0, 0])
    assert abs(c) < 1e-6 and s < -0.99


def test_bearing_distance_clamps():
    _, _, d = egocentric_bearing([0, 0, 1], 0.0, [100, 0, 0], max_dist=10.0)
    assert d == 1.0  # clamped


# --------------------------------------------------------- graph query


class _FakeGraph:
    def __init__(self, beliefs=None, nodes=None, edges=None):
        self._beliefs = beliefs or {}
        self._nodes = nodes or {}
        self._edges = edges or {}


def test_resolve_picks_highest_scoring_relief_entity():
    g = _FakeGraph(
        beliefs={
            ("ent-1", "predicts_hydration_relief"): {"confidence": 0.4, "mean": 1.0},
            ("ent-2", "predicts_hydration_relief"): {"confidence": 0.9, "mean": 1.0},
            ("ent-3", "predicts_energy_relief"): {"confidence": 0.99, "mean": 1.0},
        },
        nodes={
            "ent-1": {"position": [1.0, 1.0, 0.1]},
            "ent-2": {"position": [3.0, 0.0, 0.1]},
            "ent-3": {"position": [9.0, 9.0, 0.1]},
        },
    )
    tid, pos = resolve_goal_target("hydration", g)
    assert tid == "ent-2" and pos == [3.0, 0.0, 0.1]  # highest conf*mean for hydration
    # Different need selects a different entity.
    assert resolve_goal_target("energy", g)[0] == "ent-3"


def test_resolve_returns_none_without_target_or_graph():
    assert resolve_goal_target("hydration", None) is None
    assert resolve_goal_target(None, _FakeGraph()) is None
    assert resolve_goal_target("hydration", _FakeGraph()) is None  # no beliefs
    # Integrity has no relief belief in this scene -> no target.
    assert resolve_goal_target("integrity", _FakeGraph()) is None


def test_landmark_fallback_when_target_has_no_position():
    # The relief entity has no stored position; a spatially-related neighbour does.
    g = _FakeGraph(
        beliefs={("water", "predicts_hydration_relief"): {"confidence": 0.9, "mean": 1.0}},
        nodes={"water": {"position": None}, "house": {"position": [4.0, 2.0, 0.0]}},
        edges={("water", "house", "scene_near"): {"weight": 0.8}},
    )
    tid, pos = resolve_goal_target("hydration", g)
    assert tid == "water" and pos == [4.0, 2.0, 0.0]  # borrowed the landmark's position


def test_resolve_survives_malformed_graph():
    # A belief pointing at a missing node, and a concurrent-ish odd structure:
    # must return None, never raise.
    g = _FakeGraph(
        beliefs={("ghost", "predicts_hydration_relief"): {"confidence": 0.9, "mean": 1.0}},
        nodes={},  # 'ghost' absent
    )
    assert resolve_goal_target("hydration", g) is None
