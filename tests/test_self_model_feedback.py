"""Self-state feedback spine (self-model program, Phase 1).

Covers: default-off parity (no module, byte-identical state_dict), zero-init
parity when the flag is on (feeding self_prev is a no-op until learning moves the
projection), that a learned spine actually changes the output, env/faculty
threading, the configure() rebuild-on-toggle + capacity readout, and a checkpoint
round-trip with the spine built.

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
    monkeypatch.setenv("DECADIC_SELF_MODEL_FEEDBACK", "1" if enabled else "0")
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


def _inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(1, cfg.d_model, generator=g)
    ep = torch.rand(1, 4, generator=g)
    mem = torch.randn(1, cfg.memory_context_dim, generator=g)
    self_dim = cfg.state_mind_out + cfg.narrative_out + cfg.metacog_out
    sp = torch.randn(1, self_dim, generator=g)
    return z0, ep, mem, sp


def test_default_off_has_no_spine_module(monkeypatch):
    stack = _stack(monkeypatch, enabled=False)
    assert stack.has_self_model_feedback is False
    assert not hasattr(stack, "self_ingress")
    # Capability marker still present (so the harness knows the feature exists).
    assert stack._supports_self_model_feedback is True
    # self_prev is ignored entirely when off (parity).
    z0, ep, mem, sp = _inputs(_cfg())
    stack.eval()
    with torch.no_grad():
        stack.reset_recurrent_state()
        a = stack(z0, ep, mem, self_prev=sp)["narrative"]
        stack.reset_recurrent_state()
        b = stack(z0, ep, mem, self_prev=None)["narrative"]
    assert torch.equal(a, b)


def test_flag_on_builds_zero_init_spine(monkeypatch):
    stack = _stack(monkeypatch, enabled=True)
    assert stack.has_self_model_feedback is True
    assert hasattr(stack, "self_ingress")
    assert float(stack.self_ingress.weight.detach().abs().sum()) == 0.0
    assert float(stack.self_ingress.bias.detach().abs().sum()) == 0.0


def test_zero_init_is_byte_identical(monkeypatch):
    """With the zero-init projection, feeding self_prev must not change anything."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    stack.eval()
    z0, ep, mem, sp = _inputs(cfg)
    with torch.no_grad():
        stack.reset_recurrent_state()
        out_fed = stack(z0, ep, mem, self_prev=sp)
        stack.reset_recurrent_state()
        out_none = stack(z0, ep, mem, self_prev=None)
    for k in ("narrative", "state_mind", "metacognition", "z5", "motor_u"):
        assert torch.equal(out_fed[k], out_none[k]), k


def test_learned_spine_changes_output(monkeypatch):
    """Once the projection moves off zero, the fed-back self-state matters."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    stack.eval()
    with torch.no_grad():
        stack.self_ingress.weight.normal_(0.0, 0.3)
        stack.self_ingress.bias.normal_(0.0, 0.1)
    z0, ep, mem, sp = _inputs(cfg)
    with torch.no_grad():
        stack.reset_recurrent_state()
        fed = stack(z0, ep, mem, self_prev=sp)["narrative"]
        stack.reset_recurrent_state()
        none = stack(z0, ep, mem, self_prev=None)["narrative"]
    assert not torch.allclose(fed, none, atol=1e-6)


def test_from_env_threads_faculty(monkeypatch):
    monkeypatch.setenv("DECADIC_SELF_MODEL_FEEDBACK", "1")
    from decadic.nn.faculties import CognitionFaculties

    fac = CognitionFaculties.from_env()
    assert fac.self_model_feedback is True


def test_config_default_on(monkeypatch):
    from decadic import config as C

    # Self-model program ships ON by default (conftest pins it OFF for tests).
    monkeypatch.delenv("DECADIC_SELF_MODEL_FEEDBACK", raising=False)
    assert C.self_model_feedback_enabled() is True


def test_configure_rebuilds_on_toggle_and_reports(monkeypatch):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "smf-rebuild",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert agent.neural.stack.has_self_model_feedback is False
    assert agent.capacity_config()["self_model_feedback"] is False
    before = agent.neural

    cfg = agent.configure(self_model_feedback=True)
    assert cfg["self_model_feedback"] is True
    assert agent.neural is not before  # architecture toggle rebuilt the brain
    assert agent.neural.stack.has_self_model_feedback is True
    assert hasattr(agent.neural.stack, "self_ingress")


def test_checkpoint_roundtrips_spine(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    fac_on = CognitionFaculties(
        perception_feedback=False,
        perception_mode="oracle",
        encoder_mode="zeros",
        self_model_feedback=True,
    )
    b = NeuralBundle.try_build("smf-ckpt", faculties=fac_on)
    assert b.stack.has_self_model_feedback is True
    with torch.no_grad():
        b.stack.self_ingress.weight.normal_(0.0, 0.2)
    path = tmp_path / "smf.pt"
    b.save(path)

    # A fresh bundle built WITHOUT the spine must rebuild to the saved arch on load.
    fac_off = CognitionFaculties(
        perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
    )
    b2 = NeuralBundle.try_build("smf-ckpt", faculties=fac_off)
    assert b2.stack.has_self_model_feedback is False
    b2.load(path)
    assert b2.stack.has_self_model_feedback is True
    assert torch.allclose(b2.stack.self_ingress.weight, b.stack.self_ingress.weight)
