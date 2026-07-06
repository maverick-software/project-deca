"""Growth governance (2026-07-05) — growth must be EARNED, not scheduled.

The 1-h embodied soak (bodydiag_kuzu_20260705_094611) grew 8 times against a
pc-loss EMA that never moved off 0.91: in an open world absolute loss does not
converge, so the old level-triggered gate fired every interval forever. The
governed gate requires (a) a progress baseline (never grow on the first
check), (b) learning to have STALLED at current capacity, and (c) the previous
growth event to have PAID. Setting either governance config to 0 disables
that gate (the forced-growth escape hatch used by the mechanics tests).

These tests drive ``apply_plasticity_step`` with a pinned pc-EMA (setting
``st.pc_ema = X`` then feeding ``pc_loss=X`` keeps the EMA exactly at X), so
each scenario is deterministic.
"""

import pytest


def _ctx(bus, perc, via, epi):
    from decadic.cycle.types import CycleContext

    return CycleContext(
        state_bus=bus,
        perceptual=perc,
        viability=via,
        episodic=epi,
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _build(monkeypatch, **extra):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_PLASTICITY_ENABLED", "0")
    monkeypatch.setenv("DECADIC_SPARSE_ENABLED", "0")
    monkeypatch.setenv("DECADIC_GROWTH_ENABLED", "1")
    monkeypatch.setenv("DECADIC_GROWTH_INTERVAL", "1")  # every call is a check
    monkeypatch.setenv("DECADIC_GROWTH_STEP", "8")
    monkeypatch.setenv("DECADIC_GROWTH_PCLOSS_THRESHOLD", "0.5")
    monkeypatch.setenv("DECADIC_GROWTH_MIN_PROGRESS", "0.01")  # governance ON
    monkeypatch.setenv("DECADIC_GROWTH_MIN_GAIN", "0.02")
    monkeypatch.setenv("DECADIC_MAX_NEURONS", "160")
    monkeypatch.setenv("DECADIC_GROWABLE_HIDDEN_CEILING", "256")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)

    from decadic.nn.bundle import NeuralBundle

    return NeuralBundle.try_build("growth-governance")


def _fresh_state():
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    return StateBus(), PerceptualState(), ViabilityState(), EpisodicStore(None)


def _step_at_ema(ctx, bundle, ema: float) -> dict:
    """One governed eligibility check with the pc-EMA pinned exactly at ema."""
    from decadic.cycle.neural_pipeline import apply_plasticity_step

    st = bundle.plasticity_state
    st.pc_ema = float(ema)
    return apply_plasticity_step(ctx, bundle, pc_loss=float(ema), modulation=0.0)


def test_first_check_never_grows_blind(monkeypatch):
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    d = _step_at_ema(ctx, bundle, 2.0)
    assert d["growth_events"] == 0
    assert d["growth_blocked_reason"] == "no_progress_baseline"


def test_flat_high_loss_grows_once_then_blocks_unpaid(monkeypatch):
    """THE SOAK SCENARIO: high loss that never improves. Old gate: one growth
    event per interval, forever (8/hour observed). Governed gate: exactly one
    -- capacity gets one honest try, and when it does not pay, growth stops."""
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    awake0 = bundle.stack.awake_neurons()

    d = _step_at_ema(ctx, bundle, 2.0)  # baseline check
    assert d["growth_events"] == 0
    d = _step_at_ema(ctx, bundle, 2.0)  # stalled + never grown -> grows
    assert d["growth_events"] == 1
    assert bundle.stack.awake_neurons() > awake0
    assert d["growth_pc_ema_at_last_growth"] == pytest.approx(2.0)

    for _ in range(6):  # loss stays flat: the growth did not pay
        d = _step_at_ema(ctx, bundle, 2.0)
    assert d["growth_events"] == 1  # NOT 7
    assert d["growth_blocked_reason"] == "last_growth_unpaid"


def test_still_learning_blocks_growth(monkeypatch):
    """Loss falling at current capacity: capacity is not the bottleneck."""
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    _step_at_ema(ctx, bundle, 2.0)  # baseline
    d = _step_at_ema(ctx, bundle, 1.5)  # 25% improvement >> min_progress
    assert d["growth_events"] == 0
    assert d["growth_blocked_reason"] == "still_learning_at_capacity"


def test_growth_rearms_after_paid_improvement(monkeypatch):
    """A growth event that PAYS (EMA falls past min_gain) re-arms the gate:
    the next genuine stall may grow again."""
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    _step_at_ema(ctx, bundle, 2.0)  # baseline
    d = _step_at_ema(ctx, bundle, 2.0)  # first growth at ema=2.0
    assert d["growth_events"] == 1
    d = _step_at_ema(ctx, bundle, 1.9)  # paid (1.9 < 2.0*0.98) but improving
    assert d["growth_events"] == 1
    assert d["growth_blocked_reason"] == "still_learning_at_capacity"
    d = _step_at_ema(ctx, bundle, 1.9)  # stalled again + last growth paid
    assert d["growth_events"] == 2


def test_below_threshold_blocks(monkeypatch):
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    d = _step_at_ema(ctx, bundle, 0.4)  # under the 0.5 threshold
    assert d["growth_events"] == 0
    assert d["growth_blocked_reason"] == "pc_loss_below_threshold"


def test_governance_zero_disables_gates(monkeypatch):
    """The escape hatch: MIN_PROGRESS=0 and MIN_GAIN=0 restore the old
    level-triggered behavior (used by the growth-mechanics fixtures)."""
    pytest.importorskip("torch")
    bundle = _build(
        monkeypatch,
        DECADIC_GROWTH_MIN_PROGRESS="0",
        DECADIC_GROWTH_MIN_GAIN="0",
    )
    ctx = _ctx(*_fresh_state())
    d = _step_at_ema(ctx, bundle, 2.0)
    assert d["growth_events"] == 1  # grows on the very first check
    d = _step_at_ema(ctx, bundle, 2.0)
    assert d["growth_events"] == 2  # and keeps growing while above threshold


def test_governance_state_survives_checkpoint_roundtrip(monkeypatch, tmp_path):
    """pc_ema_at_last_growth rides the plasticity-state bundle payload, so a
    restored agent cannot be tricked into re-growing for free."""
    pytest.importorskip("torch")
    bundle = _build(monkeypatch)
    ctx = _ctx(*_fresh_state())
    _step_at_ema(ctx, bundle, 2.0)
    _step_at_ema(ctx, bundle, 2.0)  # grows; records ema-at-growth
    assert bundle.plasticity_state.pc_ema_at_last_growth == pytest.approx(2.0)
    assert bundle.plasticity_state.growth_events == 1

    ckpt = tmp_path / "gov.pt"
    bundle.save(ckpt)

    fresh = _build(monkeypatch)
    fresh.load(ckpt)
    st = fresh.plasticity_state
    assert st.pc_ema_at_last_growth == pytest.approx(2.0)
    assert st.growth_events == 1
