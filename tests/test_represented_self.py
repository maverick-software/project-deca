"""Represented self (self-model program, Phase 5).

Covers the RepresentedSelf builder (embedding shape/bounds, node content, controls
edges), the zero-init ingress faculty (default-off parity, build + checkpoint),
env/faculty threading, configure rebuild-on-toggle, and that a live cycle writes
the self onto the egocentric node, binds controls edges, and feeds the embedding.

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
    monkeypatch.setenv("DECADIC_REPRESENTED_SELF", "1" if enabled else "0")
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


# --- RepresentedSelf builder -------------------------------------------------


class _Slot:
    def __init__(self, kind, agency):
        self.kind = kind
        self.agency = agency


class _WM:
    def __init__(self, slots):
        self.slots = slots


class _Homeo:
    hydration = 80.0
    energy = 60.0
    integrity = 90.0


def test_embedding_shape_bounds_and_capability():
    import numpy as np

    from decadic.state.self_model import REPSELF_DIM, build_represented_self

    wm = _WM({"a": _Slot("self_part", 0.8), "b": _Slot("self_part", 0.4), "c": _Slot("unknown", 0.0)})
    rs = build_represented_self(
        viability=60.0,
        homeostasis=_Homeo(),
        pain=0.2,
        pleasure=0.1,
        priority=0.5,
        working_memory=wm,
    )
    emb = rs.embedding()
    assert emb.shape == (REPSELF_DIM,)
    assert np.isfinite(emb).all()
    assert (emb >= 0.0).all() and (emb <= 1.0).all()
    assert rs.n_parts == 2
    assert rs.capability == pytest.approx(0.6, abs=1e-6)  # mean(0.8, 0.4)


def test_node_content_and_controls_edges():
    from decadic.state.self_model import build_represented_self

    wm = _WM({"hand": _Slot("self_part", 0.9), "rock": _Slot("unknown", 0.0)})
    rs = build_represented_self(
        viability=100.0, homeostasis=None, pain=0.0, pleasure=0.0, priority=0.0, working_memory=wm
    )
    content = rs.node_content()
    assert set(content) >= {"intero", "affect", "capability", "n_parts"}
    nodes = [
        {"role": "entity", "id": "hand", "kind": "self_part", "agency": 0.9},
        {"role": "entity", "id": "rock", "kind": "unknown"},
    ]
    edges = rs.semantic_edges("self", nodes)
    assert len(edges) == 1
    assert edges[0]["kind"] == "controls" and edges[0]["target"] == "hand"


# --- Faculty / build wiring --------------------------------------------------


def test_default_off_has_no_ingress(monkeypatch):
    stack = _stack(monkeypatch, enabled=False)
    assert stack.has_represented_self is False
    assert not hasattr(stack, "repself_ingress")


def test_flag_on_builds_zero_init_ingress(monkeypatch):
    stack = _stack(monkeypatch, enabled=True)
    assert stack.has_represented_self is True
    assert hasattr(stack, "repself_ingress")
    assert float(stack.repself_ingress.weight.detach().abs().sum()) == 0.0
    assert float(stack.repself_ingress.bias.detach().abs().sum()) == 0.0


def test_zero_init_is_byte_identical(monkeypatch):
    from decadic.state.self_model import REPSELF_DIM

    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    stack.eval()
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, cfg.d_model, generator=g)
    ep = torch.rand(1, 4, generator=g)
    mem = torch.randn(1, cfg.memory_context_dim, generator=g)
    rp = torch.randn(1, REPSELF_DIM, generator=g)
    with torch.no_grad():
        stack.reset_recurrent_state()
        fed = stack(z0, ep, mem, repself_prev=rp)["narrative"]
        stack.reset_recurrent_state()
        none = stack(z0, ep, mem, repself_prev=None)["narrative"]
    assert torch.equal(fed, none)


def test_from_env_threads_faculty(monkeypatch):
    monkeypatch.setenv("DECADIC_REPRESENTED_SELF", "1")
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties.from_env().represented_self is True


def test_config_default_on(monkeypatch):
    from decadic import config as C

    # Self-model program ships ON by default (conftest pins it OFF for tests).
    monkeypatch.delenv("DECADIC_REPRESENTED_SELF", raising=False)
    assert C.represented_self_enabled() is True


def test_configure_rebuilds_on_toggle_and_reports(monkeypatch):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "rs-rebuild",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    assert agent.neural.stack.has_represented_self is False
    assert agent.capacity_config()["represented_self"] is False
    before = agent.neural

    cfg = agent.configure(represented_self=True)
    assert cfg["represented_self"] is True
    assert agent.neural is not before
    assert agent.neural.stack.has_represented_self is True
    assert hasattr(agent.neural.stack, "repself_ingress")


def test_checkpoint_roundtrips_ingress(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    fac_on = CognitionFaculties(
        perception_feedback=False,
        perception_mode="oracle",
        encoder_mode="zeros",
        represented_self=True,
    )
    b = NeuralBundle.try_build("rs-ckpt", faculties=fac_on)
    assert b.stack.has_represented_self is True
    with torch.no_grad():
        b.stack.repself_ingress.weight.normal_(0.0, 0.2)
    path = tmp_path / "rs.pt"
    b.save(path)

    fac_off = CognitionFaculties(
        perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
    )
    b2 = NeuralBundle.try_build("rs-ckpt", faculties=fac_off)
    assert b2.stack.has_represented_self is False
    b2.load(path)
    assert b2.stack.has_represented_self is True
    assert torch.allclose(b2.stack.repself_ingress.weight, b.stack.repself_ingress.weight)


# --- Pipeline routing --------------------------------------------------------


def test_pipeline_enriches_self_node_and_feeds_repself(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_REPRESENTED_SELF", "1")
    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    b = NeuralBundle.try_build("rs-pipe")
    assert b is not None and b.stack.has_represented_self is True

    perceptual = PerceptualState()
    # Seed an egocentric graph with a self node + a learned body part.
    perceptual.egocentric_nodes = [
        {"role": "self", "id": "self"},
        {"role": "entity", "id": "hand", "kind": "self_part", "agency": 0.9},
    ]
    perceptual.egocentric_edges = []
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=perceptual,
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
    run_neural_cycle(ctx, b)

    self_node = next(n for n in perceptual.egocentric_nodes if n.get("role") == "self")
    assert "self_model" in self_node
    assert "intero" in self_node["self_model"] and "capability" in self_node["self_model"]
    # A "controls" edge was bound to the body part.
    assert any(e.get("kind") == "controls" for e in perceptual.egocentric_edges)
    # The self-node embedding was fed back for the next cycle.
    assert b.prev_repself is not None
