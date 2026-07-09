"""WS-EXPAND E10.3/E10.4 wiring: other-vector policy ingress + inverse dynamics.

Pins the frozen other-vector layout and its solo/all-zero guarantee, the
zero-init ingress parity, the dominant-adaptive selection, and that the
inverse-dynamics head actually recovers actions from proprio transitions on a
synthetic body (the labeling model imitation-from-observation needs).
"""

import pytest

from decadic.state.other_agents import (
    OTHER_VEC_DIM,
    OtherAgentRegistry,
    encode_other_vec,
)


def _reg(**kw):
    defaults = dict(err_threshold=0.05, warmup_obs=5, ema_alpha=0.5, max_tracks=8)
    defaults.update(kw)
    return OtherAgentRegistry(**defaults)


def _ent(eid, x, y):
    return {"entity_id": eid, "position": [x, y, 0.0]}


def _make_adaptive(reg, eid, seed=9):
    import random

    rng = random.Random(seed)
    for _ in range(30):
        reg.ingest([_ent(eid, rng.uniform(0, 2), rng.uniform(0, 2))])


# --------------------------------------------------------------- other vector


def test_other_vec_all_zero_when_solo_or_missing():
    assert encode_other_vec(None, [0, 0, 0], 0.0, max_dist=10.0) == [0.0] * OTHER_VEC_DIM
    reg = _reg()
    reg.ingest([_ent("rock", 3.0, 3.0)])  # a prop, never adaptive
    for _ in range(30):
        reg.ingest([_ent("rock", 3.0, 3.0)])
    v = encode_other_vec(reg, [0.0, 0.0, 0.0], 0.0, max_dist=10.0)
    assert v == [0.0] * OTHER_VEC_DIM  # presence stays off: solo parity
    assert encode_other_vec(reg, None, 0.0, max_dist=10.0) == [0.0] * OTHER_VEC_DIM


def test_other_vec_populates_for_adaptive_other():
    reg = _reg()
    _make_adaptive(reg, "agentB")
    v = encode_other_vec(reg, [0.0, 0.0, 0.0], 0.0, max_dist=10.0)
    assert len(v) == OTHER_VEC_DIM
    assert v[0] == 1.0  # presence
    assert abs(v[1]) <= 1.0 and abs(v[2]) <= 1.0  # bearing on the unit circle
    assert 0.0 <= v[3] <= 1.0  # normalized distance
    assert 0.0 < v[7] <= 1.0  # adaptivity strength


def test_dominant_adaptive_picks_hardest_to_predict():
    reg = _reg()
    _make_adaptive(reg, "mild", seed=1)
    _make_adaptive(reg, "wild", seed=2)
    # Make "wild" defeat the prior harder with a violent jump streak.
    for i in range(10):
        reg.ingest([_ent("wild", (i % 2) * 8.0, (i % 3) * 8.0), _ent("mild", 1.0, 1.0)])
    assert reg.dominant_adaptive() == "wild"


# ------------------------------------------------------------ torch: the heads


def _tiny_stack(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def test_other_ingress_zero_init_parity_then_real(monkeypatch):
    torch = pytest.importorskip("torch")
    import copy

    stack = _tiny_stack(monkeypatch)
    assert torch.count_nonzero(stack.other_ingress.weight) == 0
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    ov = torch.rand(OTHER_VEC_DIM)
    snap = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap)
    with torch.no_grad():
        base = stack(z0, ep)["motor_u"].clone()
    stack.load_state_dict(snap)
    with torch.no_grad():
        fed = stack(z0, ep, other_vec=ov)["motor_u"].clone()
    assert torch.allclose(base, fed, atol=1e-7)  # zero-init: parity at birth
    with torch.no_grad():
        stack.other_ingress.weight.normal_(0.0, 0.3)
    snap2 = copy.deepcopy(stack.state_dict())
    stack.load_state_dict(snap2)
    with torch.no_grad():
        a = stack(z0, ep, other_vec=ov)["motor_u"].clone()
    stack.load_state_dict(snap2)
    with torch.no_grad():
        b = stack(z0, ep, other_vec=torch.zeros(OTHER_VEC_DIM))["motor_u"].clone()
    assert not torch.allclose(a, b, atol=1e-6)  # once trained, presence matters


def test_inverse_dynamics_recovers_actions(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    fdim, n_act = stack.cfg.forward_pred_dim, stack.cfg.n_actuators
    gen = torch.Generator().manual_seed(21)
    # Synthetic body: next proprio = proprio + B @ action (a fixed linear
    # response). The inverse head must recover the action from the transition.
    B = torch.randn(n_act, fdim, generator=gen) * 0.3
    params = [p for n, p in stack.named_parameters() if n.startswith("inv_l")]
    opt = torch.optim.Adam(params, lr=5e-3)
    for _ in range(400):
        s = torch.randn(8, fdim, generator=gen)
        a = torch.rand(8, n_act, generator=gen) * 2 - 1
        s_next = s + a @ B
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(stack.inverse_action(s, s_next), a)
        loss.backward()
        opt.step()
    s = torch.randn(32, fdim, generator=gen)
    a = torch.rand(32, n_act, generator=gen) * 2 - 1
    with torch.no_grad():
        err = (stack.inverse_action(s, s + a @ B) - a).abs().mean()
    # Decisively better than the null model (predicting zero: mean |a| = 0.5).
    assert float(err) < 0.25