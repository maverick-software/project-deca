"""Plasticity event telemetry.

Two layers are covered:
  1. The per-cycle edge flags surfaced by ``apply_plasticity_step`` (rewired /
     connections_rewired / grew / neurons_woken / froze), exercised end-to-end
     through ``run_neural_cycle`` with rewire + growth forced to fire.
  2. The runtime log emission (``_apply_cycle_diagnostics``), which turns those
     edge flags into structured ``plasticity_rewire`` / ``plasticity_growth`` /
     ``plasticity_frozen`` / ``plasticity_snapshot`` log lines. Driven directly
     with crafted diagnostics so it needs neither torch nor the async loop.
"""

import logging

import pytest


def _ctx(bus, perc, via, epi):
    from decadic.cycle.types import CycleContext

    return CycleContext(
        state_bus=bus,
        perceptual=perc,
        viability=via,
        episodic=epi,
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _build_forced(monkeypatch):
    """A tiny plastic bundle with rewire + growth forced to fire every cycle."""
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_PLASTICITY_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_ENABLED", "1")
    monkeypatch.setenv("DECADIC_GROWTH_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_DENSITY", "0.5")
    monkeypatch.setenv("DECADIC_SPARSE_REWIRE_INTERVAL", "1")
    monkeypatch.setenv("DECADIC_SPARSE_REWIRE_FRACTION", "0.2")
    monkeypatch.setenv("DECADIC_GROWTH_INTERVAL", "1")
    monkeypatch.setenv("DECADIC_GROWTH_STEP", "8")
    monkeypatch.setenv("DECADIC_GROWTH_PCLOSS_THRESHOLD", "0")
    # Governance gates off (0 = disabled): this fixture FORCES growth to fire.
    monkeypatch.setenv("DECADIC_GROWTH_MIN_PROGRESS", "0")
    monkeypatch.setenv("DECADIC_GROWTH_MIN_GAIN", "0")
    monkeypatch.setenv("DECADIC_MAX_NEURONS", "160")
    monkeypatch.setenv("DECADIC_GROWABLE_HIDDEN_CEILING", "256")

    from decadic.nn.bundle import NeuralBundle

    return NeuralBundle.try_build("plasticity-logging")


def _fresh_state():
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    return StateBus(), PerceptualState(), ViabilityState(), EpisodicStore(None)


# --- 1. Edge flags out of the cycle -----------------------------------------


def test_cycle_emits_rewire_and_growth_edge_flags(monkeypatch):
    pytest.importorskip("torch")
    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build_forced(monkeypatch)
    assert bundle is not None and bundle.stack.has_plastic
    bus, perc, via, epi = _fresh_state()

    rewired = grew = False
    conn = woken = 0
    for _ in range(3):
        diag = run_neural_cycle(_ctx(bus, perc, via, epi), bundle)["_diagnostics"]
        # The flags are always present once the stack is plastic.
        assert {"rewired", "connections_rewired", "grew", "neurons_woken", "froze"} <= diag.keys()
        if diag["rewired"]:
            rewired = True
            conn = max(conn, int(diag["connections_rewired"]))
        if diag["grew"]:
            grew = True
            woken = max(woken, int(diag["neurons_woken"]))

    assert rewired and conn > 0, "forced rewire should report connections_rewired > 0"
    assert grew and woken > 0, "forced growth should report neurons_woken > 0"


def test_froze_edge_flag_on_instability(monkeypatch):
    pytest.importorskip("torch")
    import torch

    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build_forced(monkeypatch)
    bus, perc, via, epi = _fresh_state()
    run_neural_cycle(_ctx(bus, perc, via, epi), bundle)
    # Poison a plastic weight so the post-step guard freezes plasticity this cycle.
    with torch.no_grad():
        bundle.stack.plastic_blocks()[0].l1_weight[0, 0] = float("nan")
    diag = run_neural_cycle(_ctx(bus, perc, via, epi), bundle)["_diagnostics"]
    assert diag["froze"] is True
    assert diag["plasticity_frozen"] is True


# --- 2. Runtime log emission -------------------------------------------------


def _runtime(monkeypatch, agent_id: str, cycle: int):
    """An AgentRuntime with no torch bundle (use_neural off) at a known cycle."""
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    from decadic.agents.runtime import AgentRuntime

    logging.getLogger("decadic.agents.runtime").propagate = True
    rt = AgentRuntime(agent_id)
    rt.state_bus.cycle_index = cycle
    return rt


def test_runtime_logs_rewire_and_growth(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-LOGTEST", 777)
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(
            {
                "rewired": True,
                "connections_rewired": 12,
                "active_connections": 100,
                "sparse_density": 0.5,
                "rewire_events": 1,
                "grew": True,
                "neurons_woken": 8,
                "awake_neurons": 40,
                "allocated_neurons": 64,
                "growth_events": 1,
                "plasticity_frozen": False,
            }
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "plasticity_rewire" in m and "agent_id=agent-LOGTEST" in m and "cycle=777" in m
        for m in msgs
    )
    assert any("plasticity_growth" in m and "neurons_woken=8" in m for m in msgs)


def test_runtime_logs_freeze_warning(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-FREEZE", 42)
    with caplog.at_level(logging.WARNING, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(
            {"froze": True, "rewire_events": 3, "growth_events": 2, "plasticity_frozen": True}
        )
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "plasticity_frozen" in m and "agent_id=agent-FREEZE" in m and "cycle=42" in m
        for m in warns
    )


def test_runtime_no_event_logs_when_quiet(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-QUIET", 5)
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(
            {
                "rewired": False,
                "grew": False,
                "froze": False,
                "plasticity_frozen": False,
                "awake_neurons": 32,
            }
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("plasticity_rewire" in m or "plasticity_growth" in m for m in msgs)
    assert not any("plasticity_snapshot" in m for m in msgs)  # snapshot off by default


def test_runtime_snapshot_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("DECADIC_PLASTICITY_LOG_EVERY", "10")
    rt = _runtime(monkeypatch, "agent-SNAP", 20)  # 20 % 10 == 0
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(
            {
                "awake_neurons": 40,
                "allocated_neurons": 64,
                "active_connections": 100,
                "sparse_density": 0.5,
                "rewire_events": 0,
                "growth_events": 0,
                "plasticity_frozen": False,  # presence marks a plastic stack
            }
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "plasticity_snapshot" in m and "agent_id=agent-SNAP" in m and "cycle=20" in m
        for m in msgs
    )
