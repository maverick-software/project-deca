"""NaN firewall: a non-finite forward pass must not destroy the brain or lock the body.

A NaN that reaches the persistent recurrent buffers (gru_h/lstm_h/lstm_c) used to
re-poison every subsequent forward pass, so the motor output stayed zero forever
(the body locked into the joint-midpoint pose). The firewall detects the
non-finite cycle, skips the weight update, and zeroes only the transient
recurrent state so the next cycle recovers - with learned weights untouched.
"""

import math

import pytest


def _build_bundle(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")

    from decadic.nn.bundle import NeuralBundle

    bundle = NeuralBundle.try_build("unit-nan-firewall")
    assert bundle is not None
    return bundle


def _make_ctx(bundle):
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
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _ctrl(out):
    return out["action"]["parameters"]["ctrl"]


def test_nan_firewall_recovers_without_destroying_brain(monkeypatch):
    torch = pytest.importorskip("torch")
    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build_bundle(monkeypatch)

    # 1) A normal cycle: finite motor command, no recovery triggered.
    ctx = _make_ctx(bundle)
    bus = ctx.state_bus
    out = run_neural_cycle(ctx, bundle)
    assert all(math.isfinite(x) for x in _ctrl(out))
    assert out["_diagnostics"]["nan_recovery"] is False

    # Snapshot a learned weight AFTER the normal step, just before the NaN.
    weight_before = bundle.stack.lstm_cell.weight_hh.detach().clone()

    # 2) Poison the persistent recurrent state and run a cycle.
    with torch.no_grad():
        bundle.stack.lstm_h.fill_(float("nan"))
    ctx2 = _make_ctx(bundle)
    ctx2.state_bus = bus  # keep the same cycle counter / state bus
    out2 = run_neural_cycle(ctx2, bundle)

    # The firewall fired: it is reported, the emitted command is still finite,
    # and the recurrent buffers are clean again.
    assert out2["_diagnostics"]["nan_recovery"] is True
    assert all(math.isfinite(x) for x in _ctrl(out2))
    assert torch.isfinite(bundle.stack.lstm_h).all()
    assert torch.isfinite(bundle.stack.lstm_c).all()
    assert torch.isfinite(bundle.stack.gru_h).all()

    # The brain is not destroyed: the update was skipped, so the learned weight
    # is unchanged and finite (no nan_to_num-to-zero, no rollback needed).
    assert torch.equal(bundle.stack.lstm_cell.weight_hh.detach(), weight_before)
    assert torch.isfinite(bundle.stack.lstm_cell.weight_hh.detach()).all()

    # Cross-cycle transition buffers were dropped so they can't re-poison.
    assert bundle.prev_state is None
    assert bundle.prev_motor is None

    # Plasticity must not be permanently frozen by a transient NaN.
    assert bundle.plasticity_state is None or not bundle.plasticity_state.frozen

    # 3) The very next cycle runs normally again (full recovery, no lingering lock).
    ctx3 = _make_ctx(bundle)
    ctx3.state_bus = bus
    out3 = run_neural_cycle(ctx3, bundle)
    assert out3["_diagnostics"]["nan_recovery"] is False
    assert all(math.isfinite(x) for x in _ctrl(out3))


def test_clean_run_never_triggers_firewall(monkeypatch):
    pytest.importorskip("torch")
    from decadic.cycle.neural_pipeline import run_neural_cycle

    bundle = _build_bundle(monkeypatch)
    ctx = _make_ctx(bundle)
    bus = ctx.state_bus
    for _ in range(5):
        c = _make_ctx(bundle)
        c.state_bus = bus
        out = run_neural_cycle(c, bundle)
        # The firewall is a behavioral no-op on finite cycles.
        assert out["_diagnostics"]["nan_recovery"] is False
        assert all(math.isfinite(x) for x in _ctrl(out))
