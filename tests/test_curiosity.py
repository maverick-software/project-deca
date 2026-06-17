"""Need-gated curiosity drive: pure epistemic math + end-to-end cycle wiring.

Covers the plan's four requirements:
- parity when off (the diagnostics curiosity channels are absent and the
  priority is never "investigate" with the flag disabled -> baseline behavior);
- the drive rises with sustained forward-model learning progress;
- it is suppressed under threat (high pain) or deprivation (low viability);
- when enabled it enters element B / the pleasure affect and the priority label.
"""

import pytest

from decadic.state.curiosity import (
    CuriosityState,
    compute_curiosity,
    curiosity_signal,
    epistemic_opportunity,
    learning_progress,
    permission,
    survival_urgency,
)

torch = pytest.importorskip("torch")


# --- Part 1: learning progress (reward the *fall* of prediction error) -------


def test_learning_progress_zero_when_too_short():
    assert learning_progress([1.0, 0.5, 0.2]) == 0.0  # < 4 samples


def test_learning_progress_zero_when_flat():
    assert learning_progress([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) == 0.0


def test_learning_progress_zero_when_rising():
    # Error climbing -> the relative decrease is negative -> clamped to 0.
    assert learning_progress([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) == 0.0


def test_learning_progress_positive_when_falling():
    assert learning_progress([1.0, 0.8, 0.6, 0.4, 0.2, 0.1]) > 0.0


def test_learning_progress_bounded_to_unit():
    lp = learning_progress([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    assert lp == pytest.approx(1.0)


def test_learning_progress_zero_when_already_mastered():
    # Older half already near-perfect prediction -> nothing left to learn here.
    assert learning_progress([1e-9, 1e-9, 1e-9, 1e-9]) == 0.0


def test_learning_progress_ignores_non_finite():
    assert learning_progress([float("nan"), 0.8, 0.6, 0.4, 0.2, 0.1]) >= 0.0


# --- Part 1: survival urgency + permission (the need-gate) -------------------


def test_urgency_zero_when_safe_and_full():
    assert survival_urgency(pain=0.0, viability=100.0) == 0.0


def test_urgency_rises_with_pain():
    assert survival_urgency(pain=0.6, viability=100.0) == pytest.approx(0.6)


def test_urgency_rises_as_viability_falls():
    assert survival_urgency(pain=0.0, viability=40.0) == pytest.approx(0.6)


def test_urgency_is_max_of_pain_and_deficit():
    assert survival_urgency(pain=0.3, viability=40.0) == pytest.approx(0.6)


def test_permission_full_when_safe():
    assert permission(0.0, sharpness=2.0) == pytest.approx(1.0)


def test_permission_zero_when_fully_urgent():
    assert permission(1.0, sharpness=2.0) == pytest.approx(0.0)


def test_permission_sharper_with_higher_exponent():
    # Higher sharpness suppresses curiosity faster at the same urgency.
    assert permission(0.5, sharpness=3.0) < permission(0.5, sharpness=1.0)


# --- Part 1: opportunity + the (pain, pleasure) signal -----------------------


def test_opportunity_dominated_by_progress():
    assert epistemic_opportunity(0.8, 0.0) == pytest.approx(0.8)


def test_opportunity_has_small_error_floor():
    # A flat-but-wrong state still invites a small probe (floor < the prog term).
    opp = epistemic_opportunity(0.0, 1.0)
    assert 0.0 < opp < 0.25


def test_curiosity_signal_never_produces_pain():
    pain, pleasure = curiosity_signal(
        learning_progress=0.8, fwd_error=0.5, pain=0.0, viability=100.0
    )
    assert pain == 0.0
    assert pleasure > 0.0


def test_curiosity_signal_rises_with_progress():
    _, low = curiosity_signal(learning_progress=0.0, fwd_error=0.5, pain=0.0, viability=100.0)
    _, high = curiosity_signal(learning_progress=0.8, fwd_error=0.5, pain=0.0, viability=100.0)
    assert high > low


def test_curiosity_signal_suppressed_under_pain():
    _, safe = curiosity_signal(learning_progress=0.8, fwd_error=0.5, pain=0.0, viability=100.0)
    _, hurt = curiosity_signal(learning_progress=0.8, fwd_error=0.5, pain=0.9, viability=100.0)
    assert hurt < safe
    assert hurt < 0.05


def test_curiosity_signal_suppressed_under_low_viability():
    _, safe = curiosity_signal(learning_progress=0.8, fwd_error=0.5, pain=0.0, viability=100.0)
    _, starving = curiosity_signal(learning_progress=0.8, fwd_error=0.5, pain=0.0, viability=10.0)
    assert starving < safe
    assert starving < 0.05


def test_curiosity_signal_scaled_by_gain():
    _, g1 = curiosity_signal(
        learning_progress=0.5, fwd_error=0.0, pain=0.0, viability=100.0, gain=1.0
    )
    _, g2 = curiosity_signal(
        learning_progress=0.5, fwd_error=0.0, pain=0.0, viability=100.0, gain=2.0
    )
    assert g2 == pytest.approx(2.0 * g1)


# --- Part 1: stateful compute + the investigate decision ---------------------


def test_state_history_is_bounded_to_window():
    st = CuriosityState(window=4, history=[])
    for v in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        st.observe(v, window=4)
    assert len(st.history) == 4
    assert st.history == [0.8, 0.7, 0.6, 0.5]


def test_compute_investigate_when_safe_and_learning():
    st = CuriosityState(window=8, history=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2])
    out = compute_curiosity(st, fwd_error=0.1, pain=0.0, viability=100.0)
    assert out.learning_progress > 0.0
    assert out.drive > 0.0
    assert out.pleasure > 0.0
    assert out.pain == 0.0
    assert out.investigate is True


def test_compute_no_investigate_when_threatened():
    st = CuriosityState(window=8, history=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2])
    out = compute_curiosity(st, fwd_error=0.1, pain=0.95, viability=100.0)
    assert out.investigate is False
    assert out.drive < 0.05


def test_compute_floor_drives_babble_but_not_investigate():
    # Flat-but-wrong: no learning progress -> a small babble drive (error floor)
    # but the priority must NOT be hijacked into "investigate".
    st = CuriosityState(window=8, history=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    out = compute_curiosity(st, fwd_error=0.5, pain=0.0, viability=100.0)
    assert out.learning_progress == 0.0
    assert out.drive > 0.0
    assert out.investigate is False


# --- Part 2: end-to-end wiring through run_neural_cycle ----------------------

_STEEP_FALL = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2]  # sustained learning progress


def _body_obs(i: int) -> dict:
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


def _run_cycle(monkeypatch, *, enabled: bool, viability_value: float, seed_history=None):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_CURIOSITY_ENABLED", "1" if enabled else "0")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    torch.manual_seed(0)
    bundle = NeuralBundle.try_build("unit-curiosity")
    assert bundle is not None
    if seed_history is not None:
        bundle._curiosity = CuriosityState(window=8, history=list(seed_history))
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(value=viability_value),
        episodic=EpisodicStore(None),
        homeostasis=Homeostasis(hydration=100.0, energy=100.0, integrity=100.0),
        last_observation=_body_obs(0),
        pending_observations=[_body_obs(0)],
    )
    out = run_neural_cycle(ctx, bundle)
    return ctx, out["_diagnostics"]


def test_parity_when_off(monkeypatch):
    ctx, diag = _run_cycle(monkeypatch, enabled=False, viability_value=100.0, seed_history=_STEEP_FALL)
    assert diag["curiosity_drive"] is None
    assert diag["curiosity_pleasure"] is None
    assert diag["curiosity_learning_progress"] is None
    assert ctx.state_bus.priority_label != "investigate"


def test_enabled_enters_affect_and_telemetry(monkeypatch):
    ctx, diag = _run_cycle(monkeypatch, enabled=True, viability_value=100.0, seed_history=_STEEP_FALL)
    assert diag["curiosity_learning_progress"] > 0.0
    assert diag["curiosity_drive"] > 0.0
    assert diag["curiosity_pleasure"] > 0.0
    # Folded into the pleasure-side affect scalar (element B path).
    assert ctx.state_bus.pleasure_scalar > 0.0


def test_enabled_priority_becomes_investigate(monkeypatch):
    ctx, diag = _run_cycle(monkeypatch, enabled=True, viability_value=100.0, seed_history=_STEEP_FALL)
    assert diag["curiosity_learning_progress"] > 0.0
    # A safe, learning agent investigates; curiosity never overrides genuine avoid.
    if ctx.state_bus.priority_label != "avoid":
        assert ctx.state_bus.priority_label == "investigate"


def test_suppressed_under_low_viability_in_cycle(monkeypatch):
    _, safe = _run_cycle(monkeypatch, enabled=True, viability_value=100.0, seed_history=_STEEP_FALL)
    _, starving = _run_cycle(monkeypatch, enabled=True, viability_value=8.0, seed_history=_STEEP_FALL)
    assert starving["curiosity_drive"] < safe["curiosity_drive"]
    assert starving["curiosity_drive"] < 0.05
