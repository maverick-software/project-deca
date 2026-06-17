"""Successor-features head + value shaping + imagined replay.

Covers naive-start parity (psi == 0 at birth so value contributes nothing and the
agent is byte-identical to the pre-value baseline), the SF head's TD-target shape
and learnability, the deficit-gated value composition, the 0->max shaping-weight
ramp, and the bounded imagined-rollout SF loss.
"""

import pytest

torch = pytest.importorskip("torch")

from decadic.config import INTERO_PRED_DIM
from decadic.consolidation.imagination import imagined_sf_loss
from decadic.consolidation.replay_buffer import Transition
from decadic.nn.config import neural_config_from_env
from decadic.nn.frozen_encoders import intero_preference_weights
from decadic.nn.neural_stack import NeuralCognitiveStack


def _cfg():
    return neural_config_from_env("tiny")


def _stack(monkeypatch):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    return NeuralCognitiveStack(_cfg())


def test_sf_head_built_and_in_state_dict(monkeypatch):
    stack = _stack(monkeypatch)
    assert stack.has_successor_model is True
    assert hasattr(stack, "sf_head")
    keys = set(stack.state_dict().keys())
    assert {"sf_head.l1.weight", "sf_head.l1.bias", "sf_head.l2.weight", "sf_head.l2.bias"} <= keys


def test_sf_zero_init_parity(monkeypatch):
    """psi == 0 at birth, so value == 0 and behavior is identical to the baseline."""
    cfg = _cfg()
    stack = _stack(monkeypatch)
    state = torch.randn(1, cfg.d_model)
    u = torch.randn(1, cfg.n_actuators)
    psi = stack.successor_predict(state, u)
    assert psi.shape == (1, INTERO_PRED_DIM)
    assert torch.allclose(psi, torch.zeros_like(psi))
    # Deficit-gated value is therefore exactly zero at birth.
    w = torch.as_tensor([intero_preference_weights(INTERO_PRED_DIM)])
    deficit = torch.tensor([[0.5, 0.2, 0.0]])
    value = ((w * deficit) * psi).sum()
    assert float(value.item()) == 0.0


def test_sf_head_learns_a_target(monkeypatch):
    cfg = _cfg()
    stack = _stack(monkeypatch)
    state = torch.randn(1, cfg.d_model)
    u = torch.randn(1, cfg.n_actuators)
    target = torch.tensor([[0.2, -0.1, 0.05]])
    opt = torch.optim.Adam(stack.sf_head.parameters(), lr=1e-2)

    def err():
        return float(torch.nn.functional.mse_loss(stack.successor_predict(state, u), target).item())

    first = err()
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(stack.successor_predict(state, u), target)
        loss.backward()
        opt.step()
    assert err() < first * 0.25


def test_detach_params_blocks_sf_weight_grad_but_keeps_input_grad(monkeypatch):
    cfg = _cfg()
    stack = _stack(monkeypatch)
    # Train the head a little so weights are non-zero (otherwise grad is trivially 0).
    state = torch.randn(1, cfg.d_model)
    u0 = torch.randn(1, cfg.n_actuators)
    opt = torch.optim.Adam(stack.sf_head.parameters(), lr=1e-1)
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.mse_loss(
            stack.successor_predict(state, u0), torch.tensor([[0.3, 0.0, 0.0]])
        ).backward()
        opt.step()
    u = torch.randn(1, cfg.n_actuators, requires_grad=True)
    stack.zero_grad(set_to_none=True)
    psi = stack.successor_predict(state, u, detach_params=True)
    psi.sum().backward()
    assert stack.sf_head.l2.weight.grad is None  # SF weights frozen for the policy term
    assert u.grad is not None  # gradient still flows to the action


def test_value_weight_ramps_from_zero(monkeypatch):
    monkeypatch.setenv("DECADIC_SF_VALUE_WEIGHT", "0.3")
    monkeypatch.setenv("DECADIC_SF_VALUE_RAMP_CYCLES", "1000")
    from decadic import config as C

    assert C.sf_value_weight_for_cycle(0) == 0.0
    assert abs(C.sf_value_weight_for_cycle(500) - 0.15) < 1e-9
    assert abs(C.sf_value_weight_for_cycle(1000) - 0.3) < 1e-9
    assert abs(C.sf_value_weight_for_cycle(5000) - 0.3) < 1e-9  # clamped at max


# --- imagined replay --------------------------------------------------------


def _drive_transition(cfg):
    return Transition(
        z0=None, ep=None, mem=None,
        prev_state=torch.randn(1, cfg.d_model),
        prev_motor=torch.randn(1, cfg.n_actuators),
        proprio_target=None,
        drive_on=True,
        prev_intero=torch.tensor([[0.3, 0.5, 0.9]]),
    )


def test_imagined_loss_is_finite_and_backprops(monkeypatch):
    cfg = _cfg()
    stack = _stack(monkeypatch)
    batch = [_drive_transition(cfg), _drive_transition(cfg)]
    loss = imagined_sf_loss(stack, batch, gamma=0.97, horizon=5, device=torch.device("cpu"))
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert stack.sf_head.l2.weight.grad is not None


def test_imagined_loss_noop_without_drive(monkeypatch):
    stack = _stack(monkeypatch)
    empty = [Transition(z0=None, ep=None, mem=None, prev_state=None, prev_motor=None,
                        proprio_target=None, drive_on=False)]
    assert imagined_sf_loss(stack, empty, gamma=0.97, horizon=5, device=torch.device("cpu")) is None
