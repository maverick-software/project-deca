"""Real cycle affect: PE-stub removal + intrinsic drive-reduction reward.

These cover the two live placeholders replaced in the neural cognitive cycle:

* ``config.pe_stub_weight()`` gates the legacy cycle-counter PE oscillation. The
  production default is 0.0 (the predictive-coding loss is the genuine surprise);
  with the weight at 0 the cycle's ``prediction_error_delta`` is exactly
  ``-viability_pe_scale() * neural_pc_loss``.
* ``viability.drive_reduction_reward`` is the intrinsic homeostatic-relief reward
  (the positive complement to ``interoceptive_drive_pain``): phasic pleasure
  proportional to the per-cycle *reduction* in drive pressure, bounded to [0, 1],
  grounded only in the agent's own reservoirs.

The autouse baseline (tests/conftest.py) pins the feature OFF and the PE stub at
its legacy 0.25, so the suite stays byte-identical; tests here flip the flags in
the body to exercise the real-affect path, and one test asserts the OFF path is
inert.
"""

import asyncio

import pytest

from decadic.state.viability import Homeostasis, drive_reduction_reward, interoceptive_drive_pain


# --- Unit: drive-reduction reward -------------------------------------------


def test_reward_fires_only_on_falling_drive():
    # Drive dropped (relief) -> positive, scaled by the gain.
    assert drive_reduction_reward(0.5, 0.2, gain=1.0) == pytest.approx(0.3)
    assert drive_reduction_reward(0.5, 0.2, gain=2.0) == pytest.approx(0.6)


def test_reward_zero_when_drive_rises_or_flat():
    # Rising drive is already felt as tonic pain; the relief reward stays 0.
    assert drive_reduction_reward(0.2, 0.5) == 0.0
    assert drive_reduction_reward(0.4, 0.4) == 0.0


def test_reward_is_bounded_to_unit_interval():
    # A huge drop and/or gain cannot push the felt relief above 1.0.
    assert drive_reduction_reward(1.0, 0.0, gain=10.0) == 1.0
    assert 0.0 <= drive_reduction_reward(0.9, 0.1, gain=3.0) <= 1.0


def test_reward_zero_gain_is_silent():
    assert drive_reduction_reward(1.0, 0.0, gain=0.0) == 0.0


def test_reward_is_relief_complement_of_drive_pain():
    """Refilling a deprived reservoir lowers drive pain and yields relief == the drop."""
    deprived = Homeostasis(hydration=60.0, energy=60.0, integrity=60.0)
    full = Homeostasis(hydration=100.0, energy=100.0, integrity=100.0)
    p_before = interoceptive_drive_pain(deprived, comfort=100.0, gain=1.0)
    p_after = interoceptive_drive_pain(full, comfort=100.0, gain=1.0)
    assert p_before > p_after  # relief direction
    assert drive_reduction_reward(p_before, p_after, gain=1.0) == pytest.approx(p_before - p_after)


# --- Unit: pe_stub_weight config knob ---------------------------------------


def test_pe_stub_weight_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("DECADIC_PE_STUB_WEIGHT", raising=False)
    from decadic import config as C

    assert C.pe_stub_weight() == 0.0


def test_pe_stub_weight_reads_env(monkeypatch):
    monkeypatch.setenv("DECADIC_PE_STUB_WEIGHT", "0.25")
    from decadic import config as C

    assert C.pe_stub_weight() == pytest.approx(0.25)


# --- Checkpoint persistence (relief is continuous across restarts) ----------


def test_prev_drive_pressure_round_trips_through_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    from decadic.agents.runtime import AgentRuntime

    async def go():
        src = AgentRuntime("ckpt-src")
        src.state_bus.prev_drive_pressure = 0.42
        payload = src.checkpoint_payload()
        assert payload["state_bus"]["prev_drive_pressure"] == pytest.approx(0.42)

        dst = AgentRuntime("ckpt-dst")
        assert dst.state_bus.prev_drive_pressure == 0.0
        dst.apply_checkpoint_payload(payload)
        assert dst.state_bus.prev_drive_pressure == pytest.approx(0.42)
        await src.stop()
        await dst.stop()

    asyncio.run(go())


# --- End-to-end neural cycle ------------------------------------------------


def _build_ctx(homeostasis=None, last_obs=None):
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    return CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        homeostasis=homeostasis,
        last_observation=last_obs
        or {
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _tiny_bundle(monkeypatch, name):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    from decadic.nn.bundle import NeuralBundle

    bundle = NeuralBundle.try_build(name)
    assert bundle is not None
    return bundle


def test_pe_delta_is_pure_pc_loss_at_zero_weight(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_PE_STUB_WEIGHT", "0")
    bundle = _tiny_bundle(monkeypatch, "unit-pe-zero")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.nn.config import viability_pe_scale

    ctx = _build_ctx()
    diag = run_neural_cycle(ctx, bundle)["_diagnostics"]

    pc = float(diag["neural_pc_loss"])
    assert diag["prediction_error_delta"] == pytest.approx(-viability_pe_scale() * pc)
    # The legacy alias mirrors the real key exactly.
    assert diag["stub_prediction_error_delta"] == pytest.approx(diag["prediction_error_delta"])


def test_drive_reduction_reward_drives_felt_relief(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_DRIVE_REWARD_ENABLED", "1")
    monkeypatch.setenv("DECADIC_PE_STUB_WEIGHT", "0")
    bundle = _tiny_bundle(monkeypatch, "unit-drive-reward")

    from decadic.cycle.neural_pipeline import run_neural_cycle

    # Start deprived: drive pressure > 0. First cycle records the baseline (no
    # relief yet, since prev pressure starts at 0 -> drive only "rose").
    homeo = Homeostasis(hydration=70.0, energy=70.0, integrity=70.0)
    ctx = _build_ctx(homeostasis=homeo)
    diag1 = run_neural_cycle(ctx, bundle)["_diagnostics"]
    assert diag1["drive_reward_delta"] == 0.0
    baseline_pressure = ctx.state_bus.prev_drive_pressure
    assert baseline_pressure > 0.0

    # Reservoirs refilled -> drive falls to ~0 -> phasic relief is felt.
    homeo.hydration = homeo.energy = homeo.integrity = 100.0
    diag2 = run_neural_cycle(ctx, bundle)["_diagnostics"]
    assert diag2["drive_reward_delta"] > 0.0
    assert ctx.state_bus.pleasure_scalar >= min(1.0, diag2["drive_reward_delta"])
    assert ctx.state_bus.prev_drive_pressure == pytest.approx(0.0, abs=1e-6)
    # Bounded affect.
    assert 0.0 <= ctx.state_bus.pleasure_scalar <= 1.0


def test_disabled_path_never_touches_prev_drive_pressure(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_DRIVE_REWARD_ENABLED", "0")  # the conftest baseline
    bundle = _tiny_bundle(monkeypatch, "unit-drive-off")

    from decadic.cycle.neural_pipeline import run_neural_cycle

    homeo = Homeostasis(hydration=70.0, energy=70.0, integrity=70.0)
    ctx = _build_ctx(homeostasis=homeo)
    run_neural_cycle(ctx, bundle)
    homeo.hydration = homeo.energy = homeo.integrity = 100.0
    diag2 = run_neural_cycle(ctx, bundle)["_diagnostics"]
    # The legacy placeholder reward path leaves the new state untouched.
    assert ctx.state_bus.prev_drive_pressure == 0.0
    assert diag2["drive_reward_delta"] == diag2["stub_reward_delta"]
