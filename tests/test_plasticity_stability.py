"""Stability — long A+B+C runs stay finite; the instability guard recovers NaNs."""

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


def _build(monkeypatch, **extra):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_PLASTICITY_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_ENABLED", "1")
    monkeypatch.setenv("DECADIC_GROWTH_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_DENSITY", "0.5")
    monkeypatch.setenv("DECADIC_SPARSE_REWIRE_INTERVAL", "5")
    monkeypatch.setenv("DECADIC_GROWTH_INTERVAL", "5")
    monkeypatch.setenv("DECADIC_GROWTH_STEP", "8")
    monkeypatch.setenv("DECADIC_GROWTH_PCLOSS_THRESHOLD", "0")
    monkeypatch.setenv("DECADIC_MAX_NEURONS", "160")
    monkeypatch.setenv("DECADIC_GROWABLE_HIDDEN_CEILING", "256")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)

    from decadic.nn.bundle import NeuralBundle

    return NeuralBundle.try_build("stability")


def _fresh_state():
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    return StateBus(), PerceptualState(), ViabilityState(), EpisodicStore(None)


def test_long_run_all_features_stays_finite(monkeypatch):
    pytest.importorskip("torch")
    import math

    import torch

    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build(monkeypatch)
    bus, perc, via, epi = _fresh_state()
    awake0 = bundle.stack.awake_neurons()
    for _ in range(40):
        out = run_neural_cycle(_ctx(bus, perc, via, epi), bundle)
        assert math.isfinite(out["_diagnostics"]["neural_pc_loss"])
    assert all(torch.isfinite(p).all() for p in bundle.stack.parameters())
    assert math.isfinite(via.value)
    assert bus.cycle_index == 40
    # Growth actually woke neurons over the run.
    assert bundle.stack.awake_neurons() > awake0
    assert bundle.plasticity_state.frozen is False


def test_nan_injection_trips_guard_and_recovers(monkeypatch):
    pytest.importorskip("torch")
    import torch

    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build(monkeypatch)
    bus, perc, via, epi = _fresh_state()
    for _ in range(3):
        run_neural_cycle(_ctx(bus, perc, via, epi), bundle)

    # Poison a plastic weight; the post-step guard must catch it.
    with torch.no_grad():
        bundle.stack.plastic_blocks()[0].l1_weight[0, 0] = float("nan")

    run_neural_cycle(_ctx(bus, perc, via, epi), bundle)
    assert bundle.plasticity_state.frozen is True
    # Recovery: everything finite again, and cycling continues.
    assert all(torch.isfinite(p).all() for p in bundle.stack.parameters())
    out = run_neural_cycle(_ctx(bus, perc, via, epi), bundle)
    assert all(torch.isfinite(p).all() for p in bundle.stack.parameters())
    assert bus.cycle_index == 5
    assert out["action"]["type"] == "motor"
