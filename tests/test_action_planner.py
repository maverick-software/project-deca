"""WS-EXPAND E1.6: online rollout action selection.

Pins the planner guardrails: bias is bounded and gain-scaled; candidate zero is
the policy's own action and winning yields no bias (parity when the policy is
already best); noise is deterministic per cycle and leaves the global RNG
untouched; non-finite model output disables the plan; nothing ever raises.
Uses a stub world model so the scoring landscape is controlled.
"""

import pytest

torch = pytest.importorskip("torch")

from decadic.nn.action_planner import plan_action_bias

N_ACT = 4
N_INT = 3


class _RewardingStack:
    """Intero model that pays for positive action dims; psi echoes the action."""

    def forward_predict_intero(self, z5, u, cur):
        return cur + 0.05 * u[:, :N_INT]

    def successor_predict(self, z5, u, detach_params=True):
        return 0.1 * u[:, :N_INT]


class _PunishingStack:
    """Any deviation from the base action scores worse (base always wins)."""

    def forward_predict_intero(self, z5, u, cur):
        return cur - 0.05 * u.abs()[:, :N_INT]

    def successor_predict(self, z5, u, detach_params=True):
        return -0.1 * u.abs()[:, :N_INT]


class _NaNStack:
    def forward_predict_intero(self, z5, u, cur):
        return cur * float("nan")

    def successor_predict(self, z5, u, detach_params=True):
        return u[:, :N_INT]


def _inputs():
    z5 = torch.zeros(1, 8)
    motor_u = torch.zeros(1, N_ACT)
    intero = torch.full((1, N_INT), 0.5)
    w = torch.ones(1, N_INT)
    return z5, motor_u, intero, w


def _plan(stack, cycle=100, **kw):
    z5, motor_u, intero, w = _inputs()
    defaults = dict(
        k=8, horizon=3, gamma=0.99, sigma=0.5, bias_gain=0.5, bias_max=0.25, cycle=cycle
    )
    defaults.update(kw)
    return plan_action_bias(stack, z5, motor_u, intero, w, **defaults)


def test_bias_is_bounded_and_gain_scaled():
    out = _plan(_RewardingStack())
    assert out is not None
    delta, diag = out
    assert delta.shape == (1, N_ACT)
    # |delta| <= bias_gain * bias_max, elementwise.
    assert float(delta.abs().max()) <= 0.5 * 0.25 + 1e-9
    assert diag["planner_best_gain"] > 0.0
    assert diag["planner_candidates"] == 9  # k + the policy's own action


def test_no_bias_when_policy_action_already_best():
    assert _plan(_PunishingStack()) is None


def test_deterministic_per_cycle_and_varies_across_cycles():
    a1 = _plan(_RewardingStack(), cycle=7)
    a2 = _plan(_RewardingStack(), cycle=7)
    b = _plan(_RewardingStack(), cycle=8)
    assert a1 is not None and a2 is not None and b is not None
    assert torch.equal(a1[0], a2[0])  # same cycle -> identical plan
    assert not torch.equal(a1[0], b[0])  # different cycle -> different noise


def test_global_rng_untouched():
    state_before = torch.random.get_rng_state()
    _plan(_RewardingStack())
    assert torch.equal(state_before, torch.random.get_rng_state())


def test_disabled_and_failure_paths_return_none():
    assert _plan(_RewardingStack(), k=0) is None
    assert _plan(_RewardingStack(), bias_gain=0.0) is None
    assert _plan(_RewardingStack(), bias_max=0.0) is None
    assert _plan(_NaNStack()) is None
    # A stack whose methods raise must be swallowed, not propagated.

    class _Boom:
        def forward_predict_intero(self, *a, **k):
            raise RuntimeError("model died")

        def successor_predict(self, *a, **k):
            raise RuntimeError("model died")

    assert _plan(_Boom()) is None


def test_inputs_not_mutated():
    z5, motor_u, intero, w = _inputs()
    z5c, mc, ic, wc = z5.clone(), motor_u.clone(), intero.clone(), w.clone()
    plan_action_bias(
        _RewardingStack(), z5, motor_u, intero, w,
        k=4, horizon=2, gamma=0.99, sigma=0.5, bias_gain=0.5, bias_max=0.25, cycle=3,
    )
    assert torch.equal(z5, z5c) and torch.equal(motor_u, mc)
    assert torch.equal(intero, ic) and torch.equal(w, wc)


def test_bias_points_toward_higher_scoring_actions():
    out = _plan(_RewardingStack(), k=32, sigma=0.6)
    assert out is not None
    delta, _ = out
    # The rewarding stub pays for POSITIVE action dims that feed intero (the
    # first N_INT dims); the winning perturbation must push net-positive there.
    assert float(delta[0, :N_INT].sum()) > 0.0
