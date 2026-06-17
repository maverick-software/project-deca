"""Temporal-integration window: commit a bound "now" (Phase 3).

Covers the window module (pass-through off-branch, frame-cap and time-based
close, mean bind, reopen) and that the pipeline holds the committed moment and
threads a live per-agent setting.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_zero_window_is_passthrough():
    from decadic.cycle.integration_window import IntegrationWindow

    w = IntegrationWindow(window_ms=0.0)
    a = np.arange(8, dtype=np.float32)
    res = w.push(a, now_s=0.0)
    assert res.closed is True
    assert res.committed is not None
    assert np.allclose(res.committed, a)


def test_closes_on_frame_cap_and_binds_mean():
    from decadic.cycle.integration_window import IntegrationWindow

    w = IntegrationWindow(window_ms=100000.0, max_frames=2)
    a = np.ones(4, dtype=np.float32)
    b = np.full(4, 3.0, dtype=np.float32)
    r1 = w.push(a, now_s=0.0)
    assert r1.committed is None and r1.buffered == 1 and r1.closed is False
    r2 = w.push(b, now_s=0.001)  # frame cap, not time
    assert r2.closed is True
    assert np.allclose(r2.committed, (a + b) / 2.0)


def test_closes_on_time():
    from decadic.cycle.integration_window import IntegrationWindow

    w = IntegrationWindow(window_ms=400.0, max_frames=100)
    a = np.ones(4, dtype=np.float32)
    assert w.push(a, now_s=0.0).committed is None
    # 500 ms later -> window elapsed past 400 ms -> commit.
    assert w.push(a, now_s=0.5).committed is not None


def test_reopens_after_commit():
    from decadic.cycle.integration_window import IntegrationWindow

    w = IntegrationWindow(window_ms=100000.0, max_frames=2)
    a = np.ones(4, dtype=np.float32)
    w.push(a, now_s=0.0)
    w.push(a, now_s=0.001)  # closes
    r3 = w.push(a, now_s=0.002)  # fresh window
    assert r3.committed is None and r3.buffered == 1


# --- Pipeline + live toggle --------------------------------------------------


def _build_ctx(*, window_ms):
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    return CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        homeostasis=None,
        integration_window_ms=window_ms,
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _tiny_bundle(monkeypatch, name):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    from decadic.nn.bundle import NeuralBundle

    b = NeuralBundle.try_build(name)
    assert b is not None
    return b


def test_pipeline_holds_committed_now(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_INTEGRATION_WINDOW_MAX_FRAMES", "2")
    from decadic.cycle.neural_pipeline import run_neural_cycle

    b = _tiny_bundle(monkeypatch, "iw")
    # First cycle: window open, no commit yet -> acts on live percept.
    run_neural_cycle(_build_ctx(window_ms=100000.0), b)
    # Reuse the same bundle so the window persists across cycles.
    ctx2 = _build_ctx(window_ms=100000.0)
    run_neural_cycle(ctx2, b)
    ws = ctx2.latents.get("integration_window")
    assert isinstance(ws, dict) and ws.get("enabled") is True
    # Frame cap = 2 -> the second push closes the window and commits a "now".
    assert ws.get("committed") is True


def test_configure_window_is_live(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "iw-live",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    before = agent.neural
    assert agent.capacity_config()["integration_window_ms"] == 0.0  # conftest pins 0

    cfg = agent.configure(integration_window_ms=300.0)
    assert agent.neural is before  # live setting: no rebuild
    assert cfg["integration_window_ms"] == 300.0
    assert agent.integration_window_ms == 300.0
