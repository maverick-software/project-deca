"""Core cognitive faculties: inherent-on defaults + per-agent build threading.

Covers that the faculties (perception-feedback loop, discovered perception, hf
encoder) and the A/B/C neuroplasticity subsystems default ON; that they are
threaded per-agent through the brain build (not read from the process env at
build time); that toggling an architecture faculty via configure() rebuilds the
brain while an observation toggle applies live; and that a checkpoint round-trips
the faculties (load rebuilds to the saved architecture).

All neural builds here pin ``encoder_mode="zeros"`` so the slot/feedback modules
are exercised structurally without downloading CLIP/Whisper.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest


def test_inherent_defaults_are_on(monkeypatch):
    from decadic import config as C
    from decadic.nn.faculties import CognitionFaculties

    # Dataclass defaults: the faculties are inherent, so on out of the box.
    fac = CognitionFaculties()
    assert fac.perception_feedback is True
    assert fac.perception_mode == "discovered"
    assert fac.encoder_mode == "hf"
    assert fac.discovered is True

    # Config-level defaults (clear the baseline-pinning autouse env first).
    for k in (
        "DECADIC_PERCEPTION_FEEDBACK_ENABLED",
        "DECADIC_PERCEPTION_MODE",
        "DECADIC_ENCODER_MODE",
        "DECADIC_PLASTICITY_ENABLED",
        "DECADIC_SPARSE_ENABLED",
        "DECADIC_GROWTH_ENABLED",
    ):
        monkeypatch.delenv(k, raising=False)
    assert C.perception_feedback_enabled() is True
    assert C.perception_mode() == "discovered"
    assert C.encoder_mode() == "hf"
    assert C.plasticity_enabled() is True
    assert C.sparse_enabled() is True
    assert C.growth_enabled() is True

    fac2 = CognitionFaculties.from_env()
    assert fac2.perception_feedback is True
    assert fac2.perception_mode == "discovered"
    assert fac2.encoder_mode == "hf"


def test_faculties_validation_and_serialization():
    from decadic.nn.faculties import CognitionFaculties

    fac = CognitionFaculties(perception_feedback=1, perception_mode="OracLe", encoder_mode="bogus")
    assert fac.perception_feedback is True
    assert fac.perception_mode == "oracle"
    assert fac.encoder_mode == "hf"  # unknown -> default
    # asdict round-trips back to an equal object (used by save/load).
    assert CognitionFaculties(**asdict(fac)) == fac


def test_try_build_threads_faculties(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    off = NeuralBundle.try_build(
        "fac-off",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert off is not None
    assert off.stack.has_perception_feedback is False
    assert off.stack.has_slots is False
    assert off.encoders.mode == "zeros"

    on = NeuralBundle.try_build(
        "fac-on",
        faculties=CognitionFaculties(
            perception_feedback=True, perception_mode="discovered", encoder_mode="zeros"
        ),
    )
    assert on.stack.has_perception_feedback is True
    assert on.stack.has_slots is True
    assert hasattr(on.stack, "top_down") and hasattr(on.stack, "precision_gate")


def test_configure_rebuilds_on_faculty_toggle(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "fac-rebuild",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert agent.neural is not None
    assert agent.neural.stack.has_perception_feedback is False
    before = agent.neural

    cfg = agent.configure(perception_feedback=True)
    assert cfg["perception_feedback"] is True
    assert agent.neural is not before  # architecture toggle rebuilt the brain
    assert agent.neural.stack.has_perception_feedback is True

    # Switching to discovered builds the slot/agency modules.
    agent.configure(perception_mode="discovered")
    assert agent.faculties.perception_mode == "discovered"
    assert agent.neural.stack.has_slots is True


def test_configure_observation_toggle_is_live(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "fac-live",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    before = agent.neural
    cfg = agent.configure(cognition_trace=False, probe_capture=True)
    assert agent.neural is before  # observation toggles do NOT rebuild
    assert agent.cognition_trace is False
    assert agent.probe_capture is True
    assert cfg["cognition_trace"] is False
    assert cfg["probe_capture"] is True


def test_configure_episodic_async_toggle_is_live(monkeypatch):
    """The write-behind episodic toggle flips live (no rebuild) and is reported back.

    conftest pins DECADIC_EPISODIC_ASYNC=0, so the agent is born synchronous; turning
    it on starts the worker, turning it back off drains+stops it. The store object is
    never swapped, and the brain is never rebuilt.
    """
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "fac-episodic",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    before = agent.neural
    store = agent.episodic
    assert agent.capacity_config()["episodic_async"] is False  # born sync (test pin)

    cfg_on = agent.configure(episodic_async=True)
    assert agent.neural is before  # live toggle: no rebuild
    assert agent.episodic is store  # store object is never swapped
    assert cfg_on["episodic_async"] is True
    assert agent.episodic.async_enabled is True

    cfg_off = agent.configure(episodic_async=False)
    assert cfg_off["episodic_async"] is False
    assert agent.episodic.async_enabled is False


def test_configure_ltm_async_toggle_is_live(monkeypatch):
    """The write-behind LTM toggle flips live (no rebuild) and is reported back.

    conftest pins DECADIC_LTM_ASYNC=0, so the agent is born synchronous; turning it on
    starts the consolidation worker, turning it back off drains+stops it. The graph
    object is never swapped, and the brain is never rebuilt.
    """
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "fac-ltm",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    before = agent.neural
    graph = agent.ltm_graph
    assert graph is not None
    assert agent.capacity_config()["ltm_async"] is False  # born sync (test pin)

    cfg_on = agent.configure(ltm_async=True)
    assert agent.neural is before  # live toggle: no rebuild
    assert agent.ltm_graph is graph  # graph object is never swapped
    assert cfg_on["ltm_async"] is True
    assert agent.ltm_graph.async_enabled is True

    cfg_off = agent.configure(ltm_async=False)
    assert cfg_off["ltm_async"] is False
    assert agent.ltm_graph.async_enabled is False


def test_checkpoint_roundtrips_faculties(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    saved = NeuralBundle.try_build(
        "ckpt-fac",
        faculties=CognitionFaculties(
            perception_feedback=True, perception_mode="discovered", encoder_mode="zeros"
        ),
    )
    assert saved.stack.has_perception_feedback is True and saved.stack.has_slots is True
    path = tmp_path / "brain.pt"
    saved.save(path)

    fresh = NeuralBundle.try_build(
        "ckpt-fac",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert fresh.stack.has_perception_feedback is False
    fresh.load(path)
    # load() rebuilds the stack to the checkpoint's faculties before restoring.
    assert fresh.faculties.perception_feedback is True
    assert fresh.faculties.perception_mode == "discovered"
    assert fresh.stack.has_perception_feedback is True
    assert fresh.stack.has_slots is True
