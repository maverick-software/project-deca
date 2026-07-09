"""WS-DEPTH: metacog calibration (D1), percept refinement (P1) + top-down cap
(P2), unified self-candidate (D2), k-round deliberation (D3), ignition
prediction (D4).

Pins: zero-init/birth-ramp parity for every new lane, the percept-key
invariance guardrail (refinement is exactly identity at birth), the top-down
hard cap, calibration math (ECE), the SELF_VEC frozen layout and its
urgency-priced ramped salience, and trainability of the new heads.
"""

import math

import pytest

from decadic.nn.attention_schema import GATE_REASONS, SCHEMA_VEC_DIM, encode_ws_target
from decadic.nn.metacog_cal import CalibrationTracker
from decadic.state.self_vec import (
    SELF_VEC_DIM,
    build_self_vec,
    candidate_salience,
    pack_candidate,
)


# ---------------------------------------------------------------- D1 tracker


def test_calibration_tracker_scores_reliability():
    t = CalibrationTracker(window=128, bins=4)
    # A perfectly calibrated predictor: p=0.8 events happen 80% of the time.
    for i in range(100):
        t.note(pred_err=1.0, realized_err=1.0, p_improve=0.8, improved=(i % 10) < 8)
    tel = t.telemetry()
    assert tel["metacog_err_mae"] == 0.0
    assert tel["metacog_calibration"] < 0.05  # near-zero ECE
    # An overconfident predictor: p=0.9 events happen 10% of the time.
    t2 = CalibrationTracker(window=128, bins=4)
    for i in range(100):
        t2.note(pred_err=0.0, realized_err=2.0, p_improve=0.9, improved=(i % 10) == 0)
    tel2 = t2.telemetry()
    assert tel2["metacog_err_mae"] == pytest.approx(2.0)
    assert tel2["metacog_calibration"] > 0.5  # badly miscalibrated
    t2.note(float("nan"), 1.0, 0.5, True)  # junk never raises


# ------------------------------------------------------------- D2 self-vector


def test_self_vec_frozen_layout_and_bounds():
    assert SELF_VEC_DIM == 16
    v = build_self_vec(
        metacog=[0.5] * 24,
        pain=0.3,
        pleasure=0.1,
        urgency=0.4,
        viability=60.0,
        schema_pred=[0.9] * 9,
        cal_next_err=0.2,
        cal_p_improve=0.7,
        rest_active=True,
    )
    assert len(v) == SELF_VEC_DIM
    assert v[0:4] == pytest.approx([0.5] * 4)
    assert v[4] == pytest.approx(0.3) and v[6] == pytest.approx(0.4)
    assert v[7] == pytest.approx(0.6)
    assert v[8:12] == pytest.approx([0.9] * 4)
    assert v[13] == pytest.approx(0.7) and v[14] == 1.0
    # Defensive: junk inputs -> finite zeros, right width.
    v0 = build_self_vec(
        metacog=None, pain="x", pleasure=None, urgency=-2, viability=1e9
    )
    assert len(v0) == SELF_VEC_DIM and all(math.isfinite(x) for x in v0)
    assert v0[6] == 0.0 and v0[7] == 1.0  # clamped


def test_self_candidate_salience_birth_ramp():
    # Cycle 0 -> EXACTLY zero (the self cannot win an ignition at birth).
    assert candidate_salience(1.0, 0, gain=0.5, ramp_cycles=1000) == 0.0
    half = candidate_salience(1.0, 500, gain=0.5, ramp_cycles=1000)
    full = candidate_salience(1.0, 1000, gain=0.5, ramp_cycles=1000)
    assert half == pytest.approx(0.25) and full == pytest.approx(0.5)
    assert candidate_salience(0.0, 5000, gain=0.5, ramp_cycles=1000) == 0.0  # calm self is quiet
    assert candidate_salience(1.0, 10**9, gain=0.5, ramp_cycles=1000) == 0.5  # ramp saturates


def test_pack_candidate_deterministic_and_sized():
    v = build_self_vec(metacog=[1.0], pain=0, pleasure=0, urgency=1, viability=50)
    a = pack_candidate(v, 64)
    b = pack_candidate(v, 64)
    assert a.shape == (64,) and (a == b).all()


# ------------------------------------------------------------- D4 ws targets


def test_encode_ws_target():
    assert encode_ws_target(None) is None
    assert encode_ws_target({"enabled": False}) is None
    assert encode_ws_target({"enabled": True, "ignited": True, "share": 0.7}) == (1.0, 0.7)
    assert encode_ws_target({"enabled": True, "ignited": False, "share": "junk"}) == (0.0, 0.0)
    assert SCHEMA_VEC_DIM == 4 + len(GATE_REASONS)  # D4 grew the layout to 9


# ------------------------------------------------- torch: parity + training


def _tiny_stack(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    torch.manual_seed(1234)  # deterministic init (the WS-IND lesson)
    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def test_percept_refine_identity_at_birth_and_trainable(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    z0 = torch.randn(1, stack.cfg.d_model)
    with torch.no_grad():
        out = stack.refine_percept(z0, 3)
    assert torch.equal(out, z0)  # EXACT identity: the invariance guardrail
    # Trainable: the percept forward model drives the refiner off identity.
    gen = torch.Generator().manual_seed(5)
    params = [
        p
        for n, p in stack.named_parameters()
        if n.startswith("refine_") or n.startswith("percept_fwd_")
    ]
    opt = torch.optim.Adam(params, lr=5e-3)
    mix = torch.randn(stack.cfg.n_actuators, stack.cfg.d_model, generator=gen) * 0.1
    first = None
    for _ in range(200):
        z = torch.randn(8, stack.cfg.d_model, generator=gen)
        u = torch.rand(8, stack.cfg.n_actuators, generator=gen) * 2 - 1
        nxt = 0.9 * z + u @ mix  # a predictable world
        opt.zero_grad()
        pred = stack.percept_forward(stack.refine_percept(z, 2), u)
        l = torch.nn.functional.mse_loss(pred, nxt)
        if first is None:
            first = float(l.item())
        l.backward()
        opt.step()
    assert float(l.item()) < first * 0.5  # learning, through the refiner


def test_metacog_head_zero_init_and_trainable(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    z5 = torch.randn(4, stack.cfg.d_model)
    with torch.no_grad():
        raw = stack.metacog_calibrate(z5)
    assert raw.shape == (4, 2) and torch.count_nonzero(raw) == 0
    # Trainable toward a latent-dependent error signal.
    gen = torch.Generator().manual_seed(6)
    params = [p for n, p in stack.named_parameters() if n.startswith("metacog_cal")]
    opt = torch.optim.Adam(params, lr=1e-2)
    w = torch.randn(stack.cfg.d_model, generator=gen) * 0.1
    for _ in range(300):
        z = torch.randn(16, stack.cfg.d_model, generator=gen)
        target = torch.nn.functional.softplus(z @ w)  # "how wrong I'm about to be"
        opt.zero_grad()
        pred = torch.nn.functional.softplus(stack.metacog_calibrate(z)[:, 0])
        l = torch.nn.functional.mse_loss(pred, target)
        l.backward()
        opt.step()
    assert float(l.item()) < 0.1


def test_topdown_cap_enforced(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_PERCEPTION_FEEDBACK_ENABLED", "1")
    monkeypatch.setenv("DECADIC_PERCEPT_TOPDOWN_CAP", "0.6")
    stack = _tiny_stack(monkeypatch)
    if not getattr(stack, "has_perception_feedback", False):
        pytest.skip("perception-feedback faculty not built in this preset path")
    with torch.no_grad():
        # Force the precision gate as far toward top-down as it can go.
        stack.precision_gate.weight.zero_()
        stack.precision_gate.bias.fill_(-50.0)  # sigmoid -> ~0 (all top-down)
        z0_bu = torch.randn(1, stack.cfg.d_model)
        _, _, gate = stack.top_down_perceive(
            z0_bu,
            prev_z5=torch.randn(1, stack.cfg.d_model),
            lstm_h=stack.lstm_h,
            mem=torch.zeros(1, stack.cfg.memory_context_dim),
            scene=None,
            intero=torch.zeros(1, 3),
        )
    # Bottom-up weight can never fall below 1 - cap: the world stays in.
    assert float(gate.min()) >= 1.0 - 0.6 - 1e-6


def test_draft_rounds_parity_at_birth(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    stack = _tiny_stack(monkeypatch)
    z0 = torch.randn(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    snap = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap)
    with torch.no_grad():
        plain = stack(z0, ep)["motor_u"].clone()
    # Three-round D3 semantics at birth (zero-init draft ingress): identical.
    stack.load_state_dict(snap)
    rec = stack.snapshot_recurrent_state()
    dv = None
    with torch.no_grad():
        for _ in range(2):
            d = stack(z0, ep, draft_vec=dv)
            stack.restore_recurrent_state(rec)
            dv = torch.cat([d["z5"].detach(), d["motor_u"].detach()], dim=-1)
        final = stack(z0, ep, draft_vec=dv)["motor_u"].clone()
    assert torch.allclose(plain, final, atol=1e-6)