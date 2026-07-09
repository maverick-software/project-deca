"""WS-DEPTH A1: the arena companion vs the E10 adaptivity gate (closed loop).

The companion's design claim, tested end to end: scripted (constant-velocity
patrol) movement must NOT activate an other-model; adaptive (jittered pursuit)
movement MUST. Time is monkeypatched so the kinematics are deterministic.
"""

import pytest

pytest.importorskip("mujoco", reason="adapter imports mujoco")

from decadic.state.other_agents import OtherAgentRegistry


def _make(monkeypatch, mode):
    import scripts.mujoco_decadic_adapter as A

    # _Companion imports time INSIDE its methods, so patch the stdlib module
    # attribute itself (the in-method import resolves to the same object).
    t = {"now": 100.0}
    monkeypatch.setattr("time.monotonic", lambda: t["now"])
    c = A._Companion(mode=mode)
    return c, t


def _drive(c, t, registry, steps=80, dt=0.25, agent=(0.0, 0.0, 0.9)):
    for _ in range(steps):
        t["now"] += dt
        c.step(list(agent))
        registry.ingest([c.entity(list(agent))])


def test_scripted_companion_keeps_gate_closed(monkeypatch):
    c, t = _make(monkeypatch, "scripted")
    reg = OtherAgentRegistry(err_threshold=0.05, warmup_obs=8, ema_alpha=0.3, max_tracks=4)
    _drive(c, t, reg)
    tel = reg.telemetry()
    assert tel["other_tracks"] == 1
    assert tel["other_models_active"] == 0  # ballistic prior wins on patrol legs


def test_adaptive_companion_opens_gate(monkeypatch):
    c, t = _make(monkeypatch, "adaptive")
    reg = OtherAgentRegistry(err_threshold=0.05, warmup_obs=8, ema_alpha=0.3, max_tracks=4)
    _drive(c, t, reg)
    assert reg.telemetry()["other_models_active"] == 1  # prior defeated
    assert reg.dominant_adaptive() == "companion"


def test_companion_entity_carries_pose(monkeypatch):
    c, t = _make(monkeypatch, "adaptive")
    _drive(c, t, OtherAgentRegistry(err_threshold=1e9, warmup_obs=999, ema_alpha=0.1, max_tracks=1), steps=10)
    e = c.entity([0.0, 0.0, 0.9])
    assert e["id"] == "companion" and e["kind"] == "agent"
    assert len(e["pose_joints"]) == 8  # E10.4's pose signal
    assert all(-1.0 <= j <= 1.0 for j in e["pose_joints"])
    assert len(e["position"]) == 3 and len(e["relative"]) == 3
