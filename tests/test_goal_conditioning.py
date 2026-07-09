"""WS-FORAGE M3 — goal-conditioned policy.

The goal vector layout is frozen; the stack's goal ingress is zero-init so the
agent is byte-identical at birth and the capability emerges only as the ingress
trains. These tests pin the layout, the birth-identity (parity), and that the
conditioning becomes content-sensitive once the ingress has a non-zero weight.
"""

import pytest

from decadic.nn.goal_conditioning import (
    DEFICIT_IDX,
    GOAL_LABELS,
    GOAL_VEC_DIM,
    TARGET_MASK_IDX,
    encode_goal,
)


# --------------------------------------------------------------- layout (pure)


def test_goal_labels_match_state_source():
    # The nn-local copy must not drift from the authoritative state definition.
    from decadic.state.goal_lifecycle import GOAL_LABELS as STATE_LABELS

    assert GOAL_LABELS == STATE_LABELS


def test_encode_goal_layout_frozen():
    assert GOAL_VEC_DIM == 16  # 8 -> 12 (E1.3 pose) -> 16 (E5.1 threat)
    # No goal -> all zeros (no conditioning).
    assert encode_goal(None, 0.5) == [0.0] * GOAL_VEC_DIM
    # A latched need -> one-hot at its index + deficit; M4 fields stay zero/off.
    v = encode_goal("energy", 0.7)
    assert len(v) == GOAL_VEC_DIM
    assert v[GOAL_LABELS.index("energy")] == 1.0
    assert sum(v[0:3]) == 1.0  # exactly one need active
    assert v[DEFICIT_IDX] == pytest.approx(0.7)
    assert v[4:8] == [0.0, 0.0, 0.0, 0.0]  # bearing/mask off in M3
    assert v[TARGET_MASK_IDX] == 0.0
    assert v[8:12] == [0.0, 0.0, 0.0, 0.0]  # E1.3 pose slots off by default
    assert v[12:16] == [0.0, 0.0, 0.0, 0.0]  # E5.1 threat slots off by default


def test_e51_threat_slots_populate_and_scale():
    # Full-strength threat dead ahead at half distance.
    v = encode_goal(
        "energy", 0.1, threat_cos=1.0, threat_sin=0.0, threat_prox=0.5, threat_scale=1.0
    )
    assert v[12] == pytest.approx(1.0) and v[13] == pytest.approx(0.0)
    assert v[14] == pytest.approx(0.5)
    assert v[15] == pytest.approx(1.0)  # threat mask on
    # Urgency override (extinction-lite): scale 0 silences every threat slot —
    # a starving agent re-tests rather than starving behind a stale belief.
    v0 = encode_goal(
        "energy", 0.9, threat_cos=1.0, threat_sin=0.0, threat_prox=0.5, threat_scale=0.0
    )
    assert v0[12:16] == [0.0, 0.0, 0.0, 0.0]
    # Partial scale attenuates proportionally.
    vh = encode_goal(
        "energy", 0.4, threat_cos=1.0, threat_sin=0.0, threat_prox=0.5, threat_scale=0.5
    )
    assert vh[12] == pytest.approx(0.5) and vh[14] == pytest.approx(0.25)
    assert vh[15] == pytest.approx(0.5)


def test_e13_positional_slots_populate_when_supplied():
    import math

    v = encode_goal("energy", 0.2, pos_nx=0.25, pos_ny=-0.5, yaw=math.pi / 2)
    assert v[8] == pytest.approx(0.25) and v[9] == pytest.approx(-0.5)
    assert v[10] == pytest.approx(1.0)  # sin(pi/2)
    assert v[11] == pytest.approx(0.0, abs=1e-9)  # cos(pi/2)
    # Out-of-range normalized position clamps; partial pose stays all-zero.
    v2 = encode_goal("energy", 0.2, pos_nx=5.0, pos_ny=-5.0, yaw=0.0)
    assert v2[8] == 1.0 and v2[9] == -1.0 and v2[11] == 1.0
    v3 = encode_goal("energy", 0.2, pos_nx=0.5, pos_ny=None, yaw=0.0)
    assert v3[8:12] == [0.0, 0.0, 0.0, 0.0]


def test_encode_goal_clamps_and_survives_bad_input():
    assert encode_goal("hydration", 5.0)[DEFICIT_IDX] == 1.0  # clamped to 1
    assert encode_goal("hydration", -2.0)[DEFICIT_IDX] == 0.0  # clamped to 0
    assert encode_goal("hydration", float("nan"))[DEFICIT_IDX] == 0.0  # NaN -> 0
    # Unknown label: no one-hot, but still finite and the right width.
    v = encode_goal("mystery", 0.4)
    assert len(v) == GOAL_VEC_DIM and sum(v[0:3]) == 0.0


def test_m4_bearing_fields_populate_when_supplied():
    v = encode_goal("hydration", 0.3, bearing_cos=1.0, bearing_sin=0.0, distance=0.5)
    assert v[4] == pytest.approx(1.0) and v[5] == pytest.approx(0.0)
    assert v[6] == pytest.approx(0.5)
    assert v[TARGET_MASK_IDX] == 1.0  # target mask on


# --------------------------------------------------------- stack wiring (torch)


def _tiny_stack(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def _stack_inputs(stack):
    import torch

    # episodic_proxy is the fixed 4-d salience proxy (stack.epi_proj = Linear(4, d)).
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    return z0, ep


def test_stack_has_zero_init_goal_ingress(monkeypatch):
    import torch

    stack = _tiny_stack(monkeypatch)
    assert stack.has_goal_conditioning is True
    assert stack.goal_ingress.in_features == GOAL_VEC_DIM
    assert torch.count_nonzero(stack.goal_ingress.weight) == 0
    assert torch.count_nonzero(stack.goal_ingress.bias) == 0


def _motor_from_state(stack, snapshot, z0, ep, goal_vec):
    # Restore the full state (recurrent buffers advance in place each forward) so
    # the ONLY difference between calls is the goal vector.
    import torch

    stack.load_state_dict(snapshot)
    with torch.no_grad():
        return stack(z0, ep, goal_vec=goal_vec)["motor_u"].clone()


def test_goal_vec_is_noop_at_birth(monkeypatch):
    # Zero-init ingress -> motor output is identical with or without a goal
    # vector until the ingress trains (birth-identity, house rule G2).
    import copy

    import torch

    stack = _tiny_stack(monkeypatch)
    z0, ep = _stack_inputs(stack)
    snap = copy.deepcopy(stack.state_dict())
    gv = torch.tensor(encode_goal("hydration", 0.9))

    out_none = _motor_from_state(stack, snap, z0, ep, None)
    out_goal = _motor_from_state(stack, snap, z0, ep, gv)
    assert torch.allclose(out_none, out_goal, atol=1e-6)


def test_goal_vec_becomes_content_sensitive_once_trained(monkeypatch):
    # Give the ingress a non-zero weight (simulating learning); now different
    # goals drive different motor output -- the pathway is real, gated only by
    # experience at birth.
    import copy

    import torch

    stack = _tiny_stack(monkeypatch)
    with torch.no_grad():
        stack.goal_ingress.weight.normal_(0.0, 0.3)
    z0, ep = _stack_inputs(stack)
    snap = copy.deepcopy(stack.state_dict())

    out_h = _motor_from_state(stack, snap, z0, ep, torch.tensor(encode_goal("hydration", 0.9)))
    out_e = _motor_from_state(stack, snap, z0, ep, torch.tensor(encode_goal("energy", 0.9)))
    assert not torch.allclose(out_h, out_e, atol=1e-5)
