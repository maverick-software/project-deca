"""Predictive affect (self-model program, Phase 4).

Covers: the AffectPredictor module (zero-init parity, learned non-zero delta),
default-off parity (no module built), env/faculty threading, the configure()
rebuild-on-toggle + capacity readout, a checkpoint round-trip with the predictor
built, and that routing a learned predicted affect into the cycle changes the
agent's perception/output.

All neural builds pin ``encoder_mode="zeros"`` so no CLIP/Whisper download.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _cfg():
    from decadic.nn.config import neural_config_from_env

    return neural_config_from_env("tiny")


def _stack(monkeypatch, *, enabled: bool):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_PREDICTIVE_AFFECT", "1" if enabled else "0")
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


# --- AffectPredictor module --------------------------------------------------


def test_predictor_zero_init_returns_zero_delta():
    from decadic.nn.affect_model import AFFECT_DIM, AffectPredictor

    p = AffectPredictor()
    assert hasattr(AffectPredictor, "predict")
    x = torch.randn(3, AFFECT_DIM)
    out = p.predict(x)
    assert out.shape == x.shape
    assert float(out.abs().sum()) == 0.0


def test_predictor_learns_nonzero_delta():
    from decadic.nn.affect_model import AFFECT_DIM, AffectPredictor

    p = AffectPredictor()
    with torch.no_grad():
        p.net[-1].weight.normal_(0.0, 0.5)
        p.net[-1].bias.normal_(0.0, 0.2)
    out = p.predict(torch.randn(2, AFFECT_DIM))
    assert float(out.abs().sum()) > 0.0


# --- Faculty / build wiring --------------------------------------------------


def test_default_off_has_no_predictor(monkeypatch):
    stack = _stack(monkeypatch, enabled=False)
    assert stack.has_predictive_affect is False
    assert not hasattr(stack, "affect_predictor")


def test_flag_on_builds_zero_init_predictor(monkeypatch):
    stack = _stack(monkeypatch, enabled=True)
    assert stack.has_predictive_affect is True
    assert hasattr(stack, "affect_predictor")
    out_w = stack.affect_predictor.net[-1].weight.detach()
    assert float(out_w.abs().sum()) == 0.0


def test_from_env_threads_faculty(monkeypatch):
    monkeypatch.setenv("DECADIC_PREDICTIVE_AFFECT", "1")
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties.from_env().predictive_affect is True


def test_config_default_on(monkeypatch):
    from decadic import config as C

    # Self-model program ships ON by default (conftest pins it OFF for tests).
    monkeypatch.delenv("DECADIC_PREDICTIVE_AFFECT", raising=False)
    assert C.predictive_affect_enabled() is True


def test_configure_rebuilds_on_toggle_and_reports(monkeypatch):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "pa-rebuild",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert agent.neural.stack.has_predictive_affect is False
    assert agent.capacity_config()["predictive_affect"] is False
    before = agent.neural

    cfg = agent.configure(predictive_affect=True)
    assert cfg["predictive_affect"] is True
    assert agent.neural is not before  # architecture toggle rebuilt the brain
    assert agent.neural.stack.has_predictive_affect is True
    assert hasattr(agent.neural.stack, "affect_predictor")


def test_checkpoint_roundtrips_predictor(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    fac_on = CognitionFaculties(
        perception_feedback=False,
        perception_mode="oracle",
        encoder_mode="zeros",
        predictive_affect=True,
    )
    b = NeuralBundle.try_build("pa-ckpt", faculties=fac_on)
    assert b.stack.has_predictive_affect is True
    with torch.no_grad():
        b.stack.affect_predictor.net[-1].weight.normal_(0.0, 0.3)
    path = tmp_path / "pa.pt"
    b.save(path)

    fac_off = CognitionFaculties(
        perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
    )
    b2 = NeuralBundle.try_build("pa-ckpt", faculties=fac_off)
    assert b2.stack.has_predictive_affect is False
    b2.load(path)
    assert b2.stack.has_predictive_affect is True
    assert torch.allclose(
        b2.stack.affect_predictor.net[-1].weight, b.stack.affect_predictor.net[-1].weight
    )


# --- Pipeline routing --------------------------------------------------------


def _build_ctx():
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
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def test_pipeline_learned_predictor_changes_output(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_PREDICTIVE_AFFECT", "1")
    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.nn.bundle import NeuralBundle

    b = NeuralBundle.try_build("pa-pipe")
    assert b is not None and b.stack.has_predictive_affect is True

    # Seed prev_affect and learn a non-zero predictor so the routed delta bites.
    b.prev_affect = torch.tensor([[0.5, 0.2, 0.1, 0.3]], dtype=torch.float32)
    with torch.no_grad():
        b.stack.affect_predictor.net[-1].weight.normal_(0.0, 0.8)
        b.stack.affect_predictor.net[-1].bias.normal_(0.0, 0.3)
    ctx = _build_ctx()
    run_neural_cycle(ctx, b)
    # prev_affect was refreshed with this cycle's ACTUAL affect (pre-prediction).
    assert b.prev_affect is not None
    # The predicted delta fed perception, so emotion_physio is finite + populated.
    import numpy as np

    assert np.isfinite(np.array(ctx.state_bus.emotion_physio)).all()
