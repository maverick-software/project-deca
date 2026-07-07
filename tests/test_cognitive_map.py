"""WS-EXPAND E1: cognitive map — pose estimation, experiential graph, planner.

Pins the E1 guardrails: pose blends toward observation and dead-reckons without
it; landmark correction bounds drift; breadcrumb hops carry MEASURED (not
straight-line) costs; the planner NEVER reroutes without stall evidence (the
straight-line default is byte-parity); A* routes through walked space when the
direct way is evidenced-blocked; everything degrades to None, never raises.
"""

import math

import pytest

from decadic.state.cognitive_map import CognitiveMap


def _cmap(**kw):
    defaults = dict(
        breadcrumb_m=1.0,
        connect_radius_m=3.0,
        max_nodes=64,
        stall_cycles=5,
        min_progress_m=0.1,
        block_threshold=2,
        pose_blend=0.5,
    )
    defaults.update(kw)
    return CognitiveMap(**defaults)


def _obs(x, y, yaw=0.0):
    return {"position": [x, y, 0.0], "orientation": [0.0, 0.0, yaw]}


def _walk(m, pts, yaw=0.0):
    for x, y in pts:
        m.update_pose(_obs(x, y, yaw))


# ---------------------------------------------------------------- pose (E1.1/2)


def test_first_fix_snaps_then_blends():
    m = _cmap(pose_blend=0.5)
    assert m.pose() is None
    m.update_pose(_obs(2.0, 3.0, 0.5))
    x, y, yaw = m.pose()
    assert (x, y, yaw) == pytest.approx((2.0, 3.0, 0.5))  # first fix snaps
    m.update_pose(_obs(4.0, 3.0, 0.5))
    x, y, _ = m.pose()
    assert x == pytest.approx(3.0)  # blend 0.5 toward the observation
    assert y == pytest.approx(3.0)


def test_dead_reckoning_integrates_velocity_and_landmark_corrects():
    m = _cmap(pose_blend=0.5)
    m.update_pose(_obs(0.0, 0.0, 0.0))
    # Position drops out; velocity keeps the pose moving (E1.1 dead-reckoning).
    for _ in range(10):
        m.update_pose({"velocity": [1.0, 0.0], "orientation": [0.0, 0.0, 0.0]}, dt=0.1)
    x, y, _ = m.pose()
    assert x == pytest.approx(1.0) and y == pytest.approx(0.0)
    assert m.dead_reckon_cycles == 10
    # Inject drift, then re-sight a known landmark 1 m dead ahead (E1.2): the
    # implied self position pulls the estimate back toward truth.
    m._x += 0.8  # simulated integration drift
    lm_world = (2.0, 0.0)  # true self is at (1, 0); landmark 1 m ahead
    before = abs(m.pose()[0] - 1.0)
    for _ in range(6):
        m.correct_from_landmark(lm_world, (1.0, 0.0))
    after = abs(m.pose()[0] - 1.0)
    assert after < before * 0.2  # drift collapsed
    assert m.landmark_corrections == 6


def test_pose_survives_malformed_input():
    m = _cmap()
    m.update_pose(None)
    m.update_pose({"position": [float("nan")]})
    m.update_pose({"position": "junk"})
    m.update_pose(_obs(1.0, 1.0))
    x, y, _ = m.pose()
    assert math.isfinite(x) and math.isfinite(y)


# ------------------------------------------------------------ graph (E1.4)


def test_breadcrumbs_drop_on_measured_travel_with_costs():
    # pose_blend=1.0 -> the pose tracks the observation exactly, so measured
    # travel equals the walked path (deterministic crumb spacing).
    m = _cmap(breadcrumb_m=1.0, pose_blend=1.0)
    # Walk a straight 5 m line in 0.5 m steps -> crumbs at 1,2,3,4,5 m.
    _walk(m, [(i * 0.5, 0.0) for i in range(11)])
    t = m.telemetry()
    assert t["cmap_nodes"] == 5
    assert t["cmap_edges"] == 4
    assert all(c == pytest.approx(1.0, abs=0.01) for c in m._edges.values())


def test_edge_cost_is_measured_path_not_displacement():
    m = _cmap(breadcrumb_m=4.0, pose_blend=1.0)
    # Hairpin pacing: out 2 m, back, out, back (8 m of measured travel, net
    # displacement ~0). Crumbs drop at 4 m and 8 m of MEASURED travel — both
    # near x=0 — so the hop cost must be ~4 m, not the ~0 m euclid.
    pts = [(i * 0.2, 0.0) for i in range(11)]  # out to 2.0
    pts += [(2.0 - i * 0.2, 0.0) for i in range(1, 11)]  # back to 0.0
    pts += [(i * 0.2, 0.0) for i in range(1, 11)]  # out to 2.0
    pts += [(2.0 - i * 0.2, 0.0) for i in range(1, 11)]  # back to 0.0
    _walk(m, pts)
    assert len(m._edges) >= 1
    cost = max(m._edges.values())
    assert cost >= 3.9  # measured (~4 m per crumb interval), NOT euclid (~0 m)


def test_node_budget_evicts_oldest():
    m = _cmap(breadcrumb_m=1.0, max_nodes=8)
    _walk(m, [(i * 1.0, 0.0) for i in range(40)])
    assert m.telemetry()["cmap_nodes"] <= 8


# ------------------------------------------------- stall + planner (E1.5)


def test_no_reroute_without_stall_evidence():
    m = _cmap()
    _walk(m, [(i * 0.5, 0.0) for i in range(20)])
    m.note_landmark("water", [9.0, 0.0])
    # Approaching normally: never blocked, planner stays silent (parity).
    for d in (8.0, 7.0, 6.0, 5.0):
        m.note_pursuit("water", d)
    assert m.is_blocked("water") is False
    assert m.plan_next_waypoint((0.0, 0.0), (9.0, 0.0), "water") is None


def test_stall_strikes_accumulate_then_gate_opens():
    # Strikes land on every stall_cycles-th no-progress note: 5th and 10th.
    m = _cmap(stall_cycles=5, block_threshold=2)
    for _ in range(4):
        m.note_pursuit("water", 5.0)
    assert m.is_blocked("water") is False  # 4 stalled notes: no strike yet
    m.note_pursuit("water", 5.0)  # 5th note -> strike 1
    assert m.is_blocked("water") is False  # 1 strike < threshold 2
    for _ in range(5):
        m.note_pursuit("water", 5.0)  # 10th note -> strike 2
    assert m.is_blocked("water") is True


def test_progress_resets_the_stall_clock_and_arrival_clears():
    m = _cmap(stall_cycles=5, block_threshold=1, min_progress_m=0.1)
    for _ in range(4):
        m.note_pursuit("water", 5.0)
    m.note_pursuit("water", 4.5)  # real approach -> clock resets
    for _ in range(4):
        m.note_pursuit("water", 4.5)
    assert m.is_blocked("water") is False
    # Force a block, then arrive: the record clears (the route works).
    for _ in range(10):
        m.note_pursuit("water", 4.5)
    assert m.is_blocked("water") is True
    m.note_pursuit("water", 0.5)  # inside arrival radius
    assert m.is_blocked("water") is False


def test_reroute_returns_first_hop_of_walked_detour():
    # World: agent at (0,0); water at (10,0); direct east is blocked (the agent
    # stalled there twice). The agent HAS walked an L-shaped detour: north to
    # (0,5), east to (10,5), south to (10,0). The planner must return a first
    # hop on the northward leg, not the straight-line east.
    m = _cmap(breadcrumb_m=1.0, stall_cycles=5, block_threshold=2, connect_radius_m=3.0)
    _walk(m, [(0.0, i * 0.5) for i in range(11)])  # north leg to (0,5)
    _walk(m, [(i * 0.5, 5.0) for i in range(1, 21)])  # east leg to (10,5)
    _walk(m, [(10.0, 5.0 - i * 0.5) for i in range(1, 11)])  # south leg to (10,0)
    m.note_landmark("water", [10.0, 0.0])
    for _ in range(12):  # two stall strikes against direct pursuit
        m.note_pursuit("water", 10.0)
    assert m.is_blocked("water") is True
    wp = m.plan_next_waypoint((0.0, 0.0), (10.0, 0.0), "water")
    assert wp is not None
    wx, wy = wp
    # First hop heads NORTH along the walked detour (not east toward the block).
    assert wy > 0.4
    assert abs(wx) < 1.5
    assert m.telemetry()["cmap_reroutes"] == 1


def test_planner_fails_safe_when_graph_cannot_anchor():
    m = _cmap(block_threshold=1, stall_cycles=5)
    for _ in range(6):
        m.note_pursuit("far", 50.0)
    assert m.is_blocked("far") is True
    # No nodes at all -> None (straight line), never an exception.
    assert m.plan_next_waypoint((0.0, 0.0), (50.0, 0.0), "far") is None
    # Nodes exist but the target is beyond connect radius -> still None.
    _walk(m, [(i * 0.5, 0.0) for i in range(8)])
    assert m.plan_next_waypoint((0.0, 0.0), (50.0, 0.0), "far") is None


def test_astar_prefers_cheap_measured_route():
    m = _cmap()
    # Hand-build a diamond: s -> a -> g cheap, s -> b -> g expensive.
    m._nodes = {"s": (0.0, 0.0), "a": (1.0, 1.0), "b": (1.0, -1.0), "g": (2.0, 0.0)}
    m._edges = {
        ("a", "s"): 1.0,
        ("a", "g"): 1.0,
        ("b", "s"): 0.5,
        ("b", "g"): 9.0,  # measured: the southern way was painful
    }
    path = m._astar("s", "g")
    assert path == ["a", "g"]


def test_telemetry_is_finite_and_complete():
    m = _cmap()
    _walk(m, [(i * 0.5, 0.0) for i in range(5)])
    t = m.telemetry()
    for k in (
        "cmap_nodes",
        "cmap_edges",
        "cmap_pose_updates",
        "cmap_dead_reckon_cycles",
        "cmap_landmark_corrections",
        "cmap_stall_events",
        "cmap_reroutes",
        "cmap_blocked_targets",
    ):
        assert k in t and math.isfinite(float(t[k]))
