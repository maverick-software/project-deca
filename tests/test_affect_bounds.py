"""Affect scalars stay bounded in [0,1] (regression for the pain runaway).

The phasic pain/pleasure scalars are felt intensities in [0,1], not running sums.
A previous leaky integrator (``0.98*prev + new``) had no clamp and weighted the new
term at full strength, so with the scalar fed back into the network it diverged to
astronomical values (e.g. 6.5e+24) within a session. These tests pin the bound.
"""

from __future__ import annotations

import math

import pytest

from decadic.state.viability import ema_affect


# ---------------------------------------------------------------------------
# ema_affect: true clamped EMA
# ---------------------------------------------------------------------------


def test_ema_affect_stays_in_unit_interval():
    assert ema_affect(0.0, 0.0) == 0.0
    assert ema_affect(0.5, 0.5) == pytest.approx(0.5)
    # A huge new term is bounded, never accumulated.
    assert ema_affect(0.0, 1e9) <= 1.0
    assert ema_affect(1.0, 1e9) <= 1.0
    # Negative input (and any drift below 0) is floored.
    assert ema_affect(0.0, -5.0) == 0.0
    # An out-of-range prior is pulled back into range.
    assert 0.0 <= ema_affect(1e9, 0.0) <= 1.0


def test_ema_affect_weights_new_term_by_one_minus_retain():
    # One step from 0 with new=1.0 moves exactly (1-retain), not the full new value.
    assert ema_affect(0.0, 1.0, retain=0.98) == pytest.approx(0.02)
    assert ema_affect(0.0, 1.0, retain=0.95) == pytest.approx(0.05)


def test_ema_affect_converges_to_new_not_fifty_times_new():
    # Feeding a constant target converges to that target (a real EMA), whereas the
    # old leaky integrator (0.98*prev + new) would settle at new/(1-0.98) = 50*new.
    x = 0.0
    for _ in range(2000):
        x = ema_affect(x, 0.5)
    assert x == pytest.approx(0.5, abs=1e-3)


def test_ema_affect_monotonic_in_new():
    prev = 0.3
    assert ema_affect(prev, 0.2) <= ema_affect(prev, 0.8)


# ---------------------------------------------------------------------------
# End-to-end: pain_scalar never escapes [0,1] across many cycles
# ---------------------------------------------------------------------------


def test_pain_scalar_stays_bounded_over_many_cycles(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    bundle = NeuralBundle.try_build("unit-affect")
    assert bundle is not None

    bus = StateBus()
    ctx = CycleContext(
        state_bus=bus,
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )

    # The old leaky integrator left [0,1] within a handful of cycles (settling at
    # ~50x its input, then diverging via the network feedback). The EMA + clamp
    # keeps every cycle in range.
    for _ in range(150):
        run_neural_cycle(ctx, bundle)
        assert math.isfinite(bus.pain_scalar)
        assert math.isfinite(bus.pleasure_scalar)
        assert 0.0 <= bus.pain_scalar <= 1.0
        assert 0.0 <= bus.pleasure_scalar <= 1.0
