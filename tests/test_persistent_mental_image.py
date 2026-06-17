"""Persistent mental image: does a stateful world model emerge from the input feed?

These are *integration* tests over the real neural cycle (``run_neural_cycle``),
complementary to the unit-level gate math in ``test_perception_feedback.py`` and
the isolated store behavior in ``test_working_memory.py``. They answer the
project's founding question -- "is the agent building a persistent mental image
of its world?" -- by pinning the three load-bearing properties such an image must
have:

  1. State *accumulates across cycles* and is carried forward, not rebuilt each
     tick: the recurrent state of mind (GRU/LSTM buffers), the one-step
     world-model memory (last latent + last action), and the persisting scene
     latent (the "image in the mind" in working memory).
  2. Under occlusion (no reliable senses) perception is *filled in* from that
     persistent history via the precision-gated top-down loop: with the gate
     closed the percept is reconstructed from memory and is invariant to the
     missing input; with the gate open the senses pass straight through.
  3. The transient image is *separable from the learned weights*: a recurrent
     reset clears the moment-to-moment image without touching the model.
"""

import pytest

torch = pytest.importorskip("torch")

# Synthetic proprioception width (matches the humanoid feed used elsewhere).
N_JOINTS = 34


def _obs(i: int, *, blank: bool = False) -> dict:
    """One body observation. ``blank`` = total sensory loss (occlusion)."""
    if blank:
        return {
            "timestamp": f"t{i}",
            "proprioception": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "mujoco_humanoid:active_inference",
                "joints": [0.0] * N_JOINTS,
                "contacts": [0.0, 0.0, 0.0, 0.0],
            },
            "events": [],
        }
    # A structured, mildly time-varying body percept (the "persistent input feed"
    # the mental image is built from).
    return {
        "timestamp": f"t{i}",
        "proprioception": {
            "position": [0.1 * i, 0.0, 1.2],
            "orientation": [0.0, 0.0, 0.05 * i],
            "velocity": [0.05, 0.0, 0.0],
            "current_action": "mujoco_humanoid:active_inference",
            "joints": [0.2 * ((j + i) % 5) - 0.4 for j in range(N_JOINTS)],
            "contacts": [120.0, 110.0, 0.0, 0.0],
        },
        "events": [],
    }


def _build_bundle(monkeypatch):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    # The mental-image loop is the subject under test -> force it on (the baseline
    # autouse fixture pins it off so the parity tests keep their semantics).
    monkeypatch.setenv("DECADIC_PERCEPTION_FEEDBACK_ENABLED", "1")
    from decadic.nn.bundle import NeuralBundle

    bundle = NeuralBundle.try_build("test-mental-image")
    assert bundle is not None
    assert bundle.stack.has_perception_feedback is True
    return bundle


def _new_state():
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    homeo = Homeostasis()
    return {
        "state_bus": StateBus(),
        "perceptual": PerceptualState(),
        "viability": ViabilityState(value=homeo.viability),
        "episodic": EpisodicStore(None),
        "homeostasis": homeo,
    }


def _run(bundle, st, obs):
    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext

    ctx = CycleContext(
        state_bus=st["state_bus"],
        perceptual=st["perceptual"],
        viability=st["viability"],
        episodic=st["episodic"],
        homeostasis=st["homeostasis"],
        last_observation=obs,
        pending_observations=[obs],
    )
    return run_neural_cycle(ctx, bundle)


def test_persistent_image_accumulates_from_the_input_feed(monkeypatch):
    """A streaming feed leaves persistent state behind: recurrent mind, last-latent
    world-model memory, and the persisting scene latent (the image in the mind)."""
    bundle = _build_bundle(monkeypatch)
    st = _new_state()

    # Before any input: the recurrent state of mind is the zero prior, and there
    # is no world-model memory or scene image yet.
    assert float(bundle.stack.lstm_h.abs().sum()) == 0.0
    assert bundle.prev_state is None
    assert st["perceptual"].working_memory.scene_latent is None

    out = None
    for i in range(6):
        out = _run(bundle, st, _obs(i))

    diag = out["_diagnostics"]
    # The top-down mental-image loop ran and reported its precision gate + PE.
    assert diag["perception_feedback"] is True
    gate = diag["precision_gate_mean"]
    assert gate is not None and 0.0 < gate < 1.0
    assert diag["perceptual_pred_error"] is not None

    # 1) Recurrent state of mind accumulated (carried across cycles, not zeroed).
    assert float(bundle.stack.lstm_h.abs().sum()) > 0.0
    # 2) One-step world-model memory carried forward (last latent + last action).
    assert bundle.prev_state is not None
    assert bundle.prev_state.shape[-1] == bundle.cfg.d_model
    assert bundle.prev_motor is not None
    # 3) The persisting "scene latent" -- the latent image in the mind.
    scene = st["perceptual"].working_memory.scene_latent
    assert scene is not None and len(scene) > 0
    assert all(v == v for v in scene)  # finite (no NaN)


def test_perception_fills_in_from_memory_under_occlusion(monkeypatch):
    """With the senses occluded, perception is reconstructed from the persistent
    mental image (history-only) and is invariant to the missing input; with the
    senses trusted, the same two frames pass straight through and differ."""
    bundle = _build_bundle(monkeypatch)
    st = _new_state()
    # Build a real history from the streaming feed -- this is what the image is
    # made of: the last latent, the recurrent state, and the persisted scene.
    for i in range(8):
        _run(bundle, st, _obs(i))

    scene_list = st["perceptual"].working_memory.scene_latent
    scene_t = (
        torch.as_tensor(scene_list, dtype=torch.float32).unsqueeze(0) if scene_list else None
    )
    intero = torch.tensor([[0.1, 0.0, 0.9]])

    def hk():
        # Identical accumulated history for every probe, so the ONLY thing that
        # can vary the percept is the current sensory input z0_bu.
        return dict(
            prev_z5=bundle.prev_state,
            lstm_h=bundle.stack.lstm_h,
            mem=None,
            scene=scene_t,
            intero=intero,
        )

    with torch.no_grad():
        z0_struct = bundle.stack.ingress(bundle.encoders(_obs(99)))
        z0_blank = torch.zeros_like(z0_struct)

        # Senses trusted (gate = 1): pure bottom-up -> the percept IS the current
        # input, so a structured frame and an occluded frame clearly differ.
        bundle.stack.precision_gate.weight.zero_()
        bundle.stack.precision_gate.bias.fill_(50.0)  # sigmoid(50) == 1
        eff_struct_open, _, gate_open = bundle.stack.top_down_perceive(z0_struct, **hk())
        eff_blank_open, _, _ = bundle.stack.top_down_perceive(z0_blank, **hk())
        assert torch.allclose(gate_open, torch.ones_like(gate_open))
        assert torch.allclose(eff_struct_open, z0_struct, atol=1e-5)
        assert torch.allclose(eff_blank_open, z0_blank, atol=1e-5)
        assert not torch.allclose(eff_struct_open, eff_blank_open, atol=1e-4)

        # Senses unreliable (gate = 0, e.g. occlusion / darkness): the percept is
        # reconstructed from the persistent mental image (depends only on history)
        # -> it is INVARIANT to whether the senses are present or blank.
        bundle.stack.precision_gate.bias.fill_(-50.0)  # sigmoid(-50) == 0
        eff_struct_closed, _, gate_closed = bundle.stack.top_down_perceive(z0_struct, **hk())
        eff_blank_closed, hat_blank, _ = bundle.stack.top_down_perceive(z0_blank, **hk())
        assert torch.allclose(gate_closed, torch.zeros_like(gate_closed))
        # Filled in from memory: same percept with or without the senses.
        assert torch.allclose(eff_struct_closed, eff_blank_closed, atol=1e-6)
        # And that filled-in percept IS the history-driven top-down prediction.
        assert torch.allclose(eff_blank_closed, hat_blank, atol=1e-6)
        # The image carries real content (the feed actually trained a non-trivial
        # prediction -- it is not the zero-init prior).
        assert float(hat_blank.abs().sum()) > 0.0


def test_recurrent_reset_clears_image_but_not_the_model(monkeypatch):
    """The moment-to-moment image (recurrent buffers) is separable from the learned
    world model (weights): the NaN-firewall reset clears the former, not the latter."""
    bundle = _build_bundle(monkeypatch)
    st = _new_state()
    for i in range(4):
        _run(bundle, st, _obs(i))

    # The transient image is populated...
    assert float(bundle.stack.lstm_h.abs().sum()) > 0.0
    w_before = bundle.stack.ingress.weight.detach().clone()
    td_before = bundle.stack.top_down[0].weight.detach().clone()

    bundle.stack.reset_recurrent_state()

    # ...the recurrent buffers are zeroed (the moment-to-moment image is cleared)...
    assert float(bundle.stack.lstm_h.abs().sum()) == 0.0
    assert float(bundle.stack.gru_h.abs().sum()) == 0.0
    assert float(bundle.stack.lstm_c.abs().sum()) == 0.0
    # ...but the learned model weights are untouched.
    assert torch.equal(bundle.stack.ingress.weight, w_before)
    assert torch.equal(bundle.stack.top_down[0].weight, td_before)
