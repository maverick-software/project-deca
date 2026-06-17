import pytest


def test_neural_cycle_smoke(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    bundle = NeuralBundle.try_build("unit-neural")
    assert bundle is not None

    bus = StateBus()
    percept = PerceptualState()
    via = ViabilityState()
    episodic = EpisodicStore(None)
    ctx = CycleContext(
        state_bus=bus,
        perceptual=percept,
        viability=via,
        episodic=episodic,
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )
    out = run_neural_cycle(ctx, bundle)
    assert out["action"]["type"] == "motor"
    params = out["action"]["parameters"]
    assert len(params["ctrl"]) == bundle.cfg.n_actuators
    assert 0.0 <= params["assist_gain"] <= 1.0
    assert "_diagnostics" in out
    assert out["_diagnostics"]["neural_pc_loss"] >= 0.0
    assert bus.cycle_index >= 1
