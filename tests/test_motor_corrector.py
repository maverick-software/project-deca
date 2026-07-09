"""WS-EXPAND E3: motor corrector (E3.1) + phase generator (E3.2/E3.3).

Pins the birth-identity of both heads (zero-init -> forward output unchanged
whether or not the inputs are supplied), that each pathway is REAL once weights
exist, the aperiodic escape (gate 0.0 == no contribution, phase held), the
bounded correction, and that the feedback-error-learning supervision actually
trains the corrector toward a tracking-error target.
"""

import copy

import pytest

torch = pytest.importorskip("torch")


def _tiny_stack(monkeypatch):
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_MOTOR_CORRECTOR", "1")
    monkeypatch.setenv("DECADIC_CPG", "1")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    stack.eval()
    return stack


def _inputs(stack):
    z0 = torch.zeros(1, stack.cfg.d_model)
    ep = torch.zeros(1, 4)
    pv = torch.rand(1, stack.cfg.forward_pred_dim) * 0.5
    return z0, ep, pv


def _motor(stack, snapshot, z0, ep, **kw):
    stack.load_state_dict(snapshot)  # restores cpg_phase buffer too
    with torch.no_grad():
        return stack(z0, ep, **kw)["motor_u"].clone()


# ------------------------------------------------------------- birth identity


def test_heads_are_zero_init(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    assert stack.has_motor_corrector is True and stack.has_cpg is True
    assert torch.count_nonzero(stack.motor_corrector_l2.weight) == 0
    assert torch.count_nonzero(stack.motor_corrector_l2.bias) == 0
    assert torch.count_nonzero(stack.cpg_head.weight) == 0
    assert torch.count_nonzero(stack.cpg_head.bias) == 0
    assert torch.count_nonzero(stack.cpg_phase) == 0


def test_corrector_is_noop_at_birth(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    z0, ep, pv = _inputs(stack)
    snap = copy.deepcopy(stack.state_dict())
    base = _motor(stack, snap, z0, ep)
    with_pv = _motor(stack, snap, z0, ep, proprio_vec=pv)
    assert torch.allclose(base, with_pv, atol=1e-7)


def test_cpg_is_silent_at_birth(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    z0, ep, _ = _inputs(stack)
    snap = copy.deepcopy(stack.state_dict())
    base = _motor(stack, snap, z0, ep)
    gated = _motor(stack, snap, z0, ep, cpg_gate=1.0)
    assert torch.allclose(base, gated, atol=1e-7)  # c is zero-init -> no rhythm


# ------------------------------------------------------- pathways become real


def test_corrector_becomes_active_and_bounded(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    with torch.no_grad():
        stack.motor_corrector_l2.weight.normal_(0.0, 0.5)
    z0, ep, pv = _inputs(stack)
    snap = copy.deepcopy(stack.state_dict())
    base = _motor(stack, snap, z0, ep)
    with_pv = _motor(stack, snap, z0, ep, proprio_vec=pv)
    assert not torch.allclose(base, with_pv, atol=1e-6)
    from decadic import config as C

    # Additive correction is tanh-bounded then gain-scaled.
    assert float((with_pv - base).abs().max()) <= C.motor_corrector_gain() + 1e-6


def test_cpg_contributes_once_amplitude_opens_and_escape_silences(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    with torch.no_grad():
        stack.cpg_head.bias[: stack.cfg.n_actuators].fill_(1.0)  # open amplitude
        stack.cpg_phase.fill_(1.0)  # off the sin(0) zero-crossing
    z0, ep, _ = _inputs(stack)
    snap = copy.deepcopy(stack.state_dict())
    base = _motor(stack, snap, z0, ep)  # cpg_gate None -> feature untouched
    on = _motor(stack, snap, z0, ep, cpg_gate=1.0)
    escaped = _motor(stack, snap, z0, ep, cpg_gate=0.0)  # E3.3 aperiodic escape
    assert not torch.allclose(base, on, atol=1e-6)
    assert torch.allclose(base, escaped, atol=1e-7)


def test_phase_advances_only_when_gated_on(monkeypatch):
    stack = _tiny_stack(monkeypatch)
    z0, ep, _ = _inputs(stack)
    with torch.no_grad():
        stack(z0, ep, cpg_gate=1.0)
    assert float(stack.cpg_phase.abs().sum()) > 0.0  # advanced
    phase_after = stack.cpg_phase.clone()
    with torch.no_grad():
        stack(z0, ep, cpg_gate=0.0)  # escape: phase holds
        stack(z0, ep)  # feature untouched: phase holds
    assert torch.equal(stack.cpg_phase, phase_after)
    assert float(stack.cpg_phase.max()) < 6.2832  # wrapped, never unbounded


# --------------------------------------------------- FEL supervision (E3.1)


def test_motor_correction_bounded():
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    stack = NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())
    with torch.no_grad():
        stack.motor_corrector_l2.weight.normal_(0.0, 5.0)  # hostile weights
        out = stack.motor_correction(
            torch.rand(4, stack.cfg.n_actuators) * 2 - 1,
            torch.rand(4, stack.cfg.forward_pred_dim) * 10,
        )
    assert float(out.abs().max()) <= 1.0  # tanh bound holds regardless


def test_fel_supervision_trains_the_corrector(monkeypatch):
    # A persistent tracking gap (command 0.4 high on joint 0) supervised the
    # E3.1 way -- clamp(k * (prev_cmd - realized)) as MSE target -- must pull
    # the corrector output on that joint toward the compensating value.
    stack = _tiny_stack(monkeypatch)
    n_act, fdim = stack.cfg.n_actuators, stack.cfg.forward_pred_dim
    prev_u = torch.zeros(1, n_act)
    prev_u[0, 0] = 0.4
    prev_pv = torch.full((1, fdim), 0.2)
    target = torch.zeros(1, n_act)
    target[0, 0] = 0.5 * 0.4  # fel_k * tracking error on joint 0
    params = [
        p
        for n, p in stack.named_parameters()
        if n.startswith("motor_corrector_")
    ]
    opt = torch.optim.Adam(params, lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            stack.motor_correction(prev_u, prev_pv), target
        )
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
    assert float(loss.item()) < first * 0.05  # supervision drives it down
    with torch.no_grad():
        out = stack.motor_correction(prev_u, prev_pv)
    assert out[0, 0].item() == pytest.approx(0.2, abs=0.05)  # learned the gap