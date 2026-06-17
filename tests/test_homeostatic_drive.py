"""Homeostatic drive reduction: innate thirst pain + interoceptive world model.

Covers Part 1 (deprivation pain is monotonic and bounded, and folds into affect),
Part 2 (the interoceptive vectors, the always-built forward model's shapes, and the
detached-params gradient path that lets the policy plan through a frozen world
model), and the always-on drive (intero head built unconditionally; active when a
body streams reservoirs).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _cfg():
    from decadic.nn.config import neural_config_from_env

    return neural_config_from_env("tiny")


def _stack(monkeypatch):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.delenv("DECADIC_HOMEOSTATIC_DRIVE_ENABLED", raising=False)
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


# --- Part 1: innate thirst -> pain ------------------------------------------


def test_drive_pain_zero_when_full():
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    res = Homeostasis(hydration=100.0, energy=100.0, integrity=100.0)
    assert interoceptive_drive_pain(res, comfort=100.0, gain=1.0) == 0.0


def test_drive_pain_has_no_dead_zone():
    """Any dip below full registers a little pain (the comfort dead zone is gone)."""
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    res = Homeostasis(hydration=90.0, energy=100.0, integrity=100.0)
    # deficit 0.1, convex (squared) -> 0.01: faint but non-zero.
    assert interoceptive_drive_pain(res, comfort=100.0, gain=1.0) == pytest.approx(0.01)


def test_drive_pain_is_convex():
    """Doubling the deficit more-than-doubles the pain (slight need stays resistible)."""
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    mild = Homeostasis(hydration=90.0, energy=100.0, integrity=100.0)  # .1^2 = .01
    severe = Homeostasis(hydration=80.0, energy=100.0, integrity=100.0)  # .2^2 = .04
    p_mild = interoceptive_drive_pain(mild, comfort=100.0, gain=1.0)
    p_severe = interoceptive_drive_pain(severe, comfort=100.0, gain=1.0)
    assert p_severe > 2.0 * p_mild


def test_drive_pain_compounds_across_reservoirs():
    """Simultaneous needs add up: thirst AND hunger hurt more than either alone."""
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    one = Homeostasis(hydration=70.0, energy=100.0, integrity=100.0)  # .3^2 = .09
    both = Homeostasis(hydration=70.0, energy=70.0, integrity=100.0)  # .09 + .09 = .18
    p_one = interoceptive_drive_pain(one, comfort=100.0, gain=1.0)
    p_both = interoceptive_drive_pain(both, comfort=100.0, gain=1.0)
    assert p_both == pytest.approx(2.0 * p_one)


def test_drive_pain_monotonic_in_deficit():
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    mild = Homeostasis(hydration=60.0, energy=100.0, integrity=100.0)
    severe = Homeostasis(hydration=20.0, energy=100.0, integrity=100.0)
    p_mild = interoceptive_drive_pain(mild, comfort=100.0, gain=1.0)
    p_severe = interoceptive_drive_pain(severe, comfort=100.0, gain=1.0)
    assert 0.0 < p_mild < p_severe


def test_drive_pain_exponent_controls_convexity():
    """exponent == 1 is linear; > 1 softens mild deprivation (convex)."""
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    res = Homeostasis(hydration=50.0, energy=100.0, integrity=100.0)  # deficit 0.5
    linear = interoceptive_drive_pain(res, comfort=100.0, gain=1.0, exponent=1.0)
    convex = interoceptive_drive_pain(res, comfort=100.0, gain=1.0, exponent=2.0)
    assert linear == pytest.approx(0.5)
    assert convex == pytest.approx(0.25)
    assert convex < linear


def test_drive_pain_bounded_and_gain_scaled():
    from decadic.state.viability import Homeostasis, interoceptive_drive_pain

    empty = Homeostasis(hydration=0.0, energy=0.0, integrity=0.0)
    # Summed deficits (1+1+1) already saturate; gain pushes further but clamps to 1.
    assert interoceptive_drive_pain(empty, comfort=100.0, gain=5.0) == 1.0
    # gain == 0 disables the drive.
    assert interoceptive_drive_pain(empty, comfort=100.0, gain=0.0) == 0.0


# --- Part 2: interoceptive vectors ------------------------------------------


def test_controllable_intero_vector_normalizes():
    from decadic.config import INTERO_PRED_DIM
    from decadic.nn.frozen_encoders import controllable_intero_vector
    from decadic.state.viability import Homeostasis

    res = Homeostasis(hydration=50.0, energy=75.0, integrity=100.0)
    v = controllable_intero_vector(res, INTERO_PRED_DIM)
    assert v == pytest.approx([0.5, 0.75, 1.0])


def test_controllable_intero_vector_none_is_full():
    """Missing homeostasis (stub/test) reads as full reservoirs -> zero drive."""
    from decadic.config import INTERO_PRED_DIM
    from decadic.nn.frozen_encoders import controllable_intero_vector

    assert controllable_intero_vector(None, INTERO_PRED_DIM) == pytest.approx([1.0, 1.0, 1.0])


def test_preferred_intero_is_full_setpoint():
    from decadic.config import INTERO_PRED_DIM
    from decadic.nn.frozen_encoders import intero_preference_weights, preferred_intero_vector

    assert preferred_intero_vector(INTERO_PRED_DIM) == pytest.approx([1.0, 1.0, 1.0])
    # Equal weights so the most-deprived reservoir dominates on its own.
    w = intero_preference_weights(INTERO_PRED_DIM)
    assert w == pytest.approx([1.0, 1.0, 1.0])


# --- Part 2: interoceptive world model (neural stack) -----------------------


def test_intero_head_always_built(monkeypatch):
    stack = _stack(monkeypatch)
    assert stack.has_intero_model is True
    assert hasattr(stack, "fwd_intero_l1")
    assert hasattr(stack, "fwd_intero_l2")
    keys = set(stack.state_dict().keys())
    assert {
        "fwd_intero_l1.weight",
        "fwd_intero_l1.bias",
        "fwd_intero_l2.weight",
        "fwd_intero_l2.bias",
    } <= keys


def test_forward_predict_intero_shapes(monkeypatch):
    from decadic.config import INTERO_PRED_DIM

    cfg = _cfg()
    stack = _stack(monkeypatch)
    assert stack.has_intero_model is True
    state = torch.randn(1, cfg.d_model)
    u = torch.randn(1, cfg.n_actuators)
    intero_cur = torch.tensor([[0.4, 0.6, 0.9]])
    out = stack.forward_predict_intero(state, u, intero_cur)
    assert out.shape == (1, INTERO_PRED_DIM)
    assert torch.isfinite(out).all()


def test_intero_world_model_learns_transition(monkeypatch):
    """The forward model fits a realized (state, action, reservoirs) -> next-reservoirs."""
    cfg = _cfg()
    stack = _stack(monkeypatch)
    torch.manual_seed(0)
    state = torch.randn(1, cfg.d_model)
    u = torch.randn(1, cfg.n_actuators)
    intero_cur = torch.tensor([[0.3, 0.5, 0.8]])
    target = torch.tensor([[0.35, 0.5, 0.8]])  # hydration ticked up
    opt = torch.optim.Adam(
        list(stack.fwd_intero_l1.parameters()) + list(stack.fwd_intero_l2.parameters()), lr=1e-2
    )

    def err() -> float:
        pred = stack.forward_predict_intero(state, u, intero_cur)
        return float(torch.nn.functional.mse_loss(pred, target).item())

    first = err()
    for _ in range(200):
        opt.zero_grad()
        pred = stack.forward_predict_intero(state, u, intero_cur)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
    assert err() < first * 0.5


def test_detached_params_route_gradient_to_policy_not_world_model(monkeypatch):
    """l_pref_intero with detach_params=True trains the action, never the world model.

    The motor command (a leaf standing in for the policy output) must receive
    gradient, while the world-model weights must not (frozen during planning) -
    the same anti-hallucination guarantee as the proprio preferred-state loss.
    """
    cfg = _cfg()
    stack = _stack(monkeypatch)
    state = torch.randn(1, cfg.d_model)
    u = torch.randn(1, cfg.n_actuators, requires_grad=True)
    intero_cur = torch.tensor([[0.2, 0.5, 0.9]])
    preferred = torch.ones(1, 3)
    pred = stack.forward_predict_intero(state, u, intero_cur, detach_params=True)
    loss = (pred - preferred).pow(2).mean()
    loss.backward()
    assert u.grad is not None and float(u.grad.abs().sum()) > 0.0
    assert stack.fwd_intero_l1.weight.grad is None
    assert stack.fwd_intero_l2.weight.grad is None


# --- Always-on drive in the neural cycle (no feature flag) ------------------


def test_drive_on_by_default_with_depleted_reservoir(monkeypatch):
    """Homeostatic drive is the root motivation: active without any env flag."""
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.delenv("DECADIC_HOMEOSTATIC_DRIVE_ENABLED", raising=False)

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    bundle = NeuralBundle.try_build("unit-drive")
    assert bundle is not None
    assert bundle.stack.has_intero_model is True

    homeo = Homeostasis(hydration=30.0, energy=100.0, integrity=100.0)
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(value=homeo.viability),
        episodic=EpisodicStore(None),
        homeostasis=homeo,
        last_observation={
            "proprioception": {
                "position": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )
    out = run_neural_cycle(ctx, bundle)
    diag = out["_diagnostics"]
    assert diag["homeostatic_drive"] is True
    assert diag["intero_drive"] is not None
    assert float(diag["intero_drive"]) > 0.0
    assert "preferred_state_error" not in diag


def _darkroom_obs(i: int) -> dict:
    # A perfectly static body: a "still" policy could predict this world exactly,
    # which is the dark-room temptation. The unmet need must override it.
    return {
        "timestamp": f"t{i}",
        "proprioception": {
            "position": [0.0, 0.0, 1.2],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "mujoco_humanoid:active_inference",
            "joints": [0.0 for _ in range(34)],
            "contacts": [120.0, 110.0, 0.0, 0.0],
        },
        "events": [],
    }


def test_unmet_need_keeps_exploring_and_drive_falls_on_refill(monkeypatch):
    """Anti-dark-room verification (Unit B).

    While a reservoir is depleted: the drive registers (``intero_drive`` > 0) and
    exploration stays alive (``motor_babble_sigma`` floored above zero, NOT decayed
    to zero on a clock), so the agent keeps acting even though a still policy could
    predict the static world perfectly. Refilling the reservoir drops the drive the
    policy is pulled toward -- the relief signal that, with the severity-weighted
    drive-reduction loss, teaches action over quiescence.
    """
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_MOTOR_BABBLE_FLOOR", "0.05")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    bundle = NeuralBundle.try_build("unit-darkroom")
    assert bundle is not None
    bus, percept, ep = StateBus(), PerceptualState(), EpisodicStore(None)
    homeo = Homeostasis(hydration=10.0, energy=100.0, integrity=100.0)  # severe thirst
    via = ViabilityState(value=homeo.viability)

    out = None
    for i in range(6):
        ctx = CycleContext(
            state_bus=bus,
            perceptual=percept,
            viability=via,
            episodic=ep,
            homeostasis=homeo,
            last_observation=_darkroom_obs(i),
            pending_observations=[_darkroom_obs(i)],
        )
        out = run_neural_cycle(ctx, bundle)

    diag = out["_diagnostics"]
    assert diag["intero_drive"] is not None and float(diag["intero_drive"]) > 0.0
    # Exploration stays alive (floored) instead of freezing into the dark room.
    assert float(diag["motor_babble_sigma"]) >= 0.05
    # The agent is acting on the world (a realized transition was buffered).
    assert bundle.prev_state is not None
    drive_thirsty = float(diag["intero_drive"])

    # Drink: the reservoir recovers, so the felt drive must fall.
    homeo.hydration = 95.0
    via.value = homeo.viability
    for i in range(6, 9):
        ctx = CycleContext(
            state_bus=bus,
            perceptual=percept,
            viability=via,
            episodic=ep,
            homeostasis=homeo,
            last_observation=_darkroom_obs(i),
            pending_observations=[_darkroom_obs(i)],
        )
        out = run_neural_cycle(ctx, bundle)
    assert float(out["_diagnostics"]["intero_drive"]) < drive_thirsty
