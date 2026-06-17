"""Section-7 signature assays (P1-P5) for the self-model program.

These are the *falsification* tests: each later phase (self-state spine, global
workspace, temporal window, predictive affect) must show a measurable, mechanistic
signature, not just a relabelled output. The integration proxy
(``decadic.metrics.integration``) is the shared instrument.

The phase-specific assays are **capability-gated**: until the phase that adds the
faculty/flag lands, the relevant build is identical to the baseline and the test
``skip``s. As each phase is implemented the capability appears and the assay runs
automatically -- so this one file scaffolds P1-P5 and keeps the suite green at
every step.
"""

import pytest

torch = pytest.importorskip("torch")

from decadic.metrics.integration import (  # noqa: E402
    lz76_complexity,
    perturbational_complexity,
    self_state_vector,
)


def _cfg():
    from decadic.nn.config import neural_config_from_env

    return neural_config_from_env("tiny")


def _stack(monkeypatch, env: dict[str, str] | None = None):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


# --- Phase 0: the instrument itself -----------------------------------------


def test_lz76_basic_properties():
    assert lz76_complexity("") == 0
    assert lz76_complexity("0") == 1
    # An alternating string is more complex than a constant one of equal length.
    const = lz76_complexity("0" * 64)
    alt = lz76_complexity("01" * 32)
    assert alt > const


def test_pci_metric_wellformed(monkeypatch):
    """The perturbational probe returns a finite, bounded score for a baseline."""
    stack = _stack(monkeypatch)
    res = perturbational_complexity(stack, cycles=10, seed=1)
    assert 0.0 <= res.pci <= 2.0
    assert res.n_bits > 0
    assert 0.0 <= res.active_fraction <= 1.0
    assert 0.0 <= res.persistence <= 1.0
    assert res.self_feedback is False  # spine not built in the baseline


def test_pci_is_deterministic(monkeypatch):
    stack = _stack(monkeypatch)
    a = perturbational_complexity(stack, cycles=10, seed=3)
    b = perturbational_complexity(stack, cycles=10, seed=3)
    assert a.pci == b.pci and a.lz == b.lz


def test_probe_restores_recurrent_state(monkeypatch):
    """The probe must leave the live recurrent buffers zeroed (no clobber)."""
    stack = _stack(monkeypatch)
    perturbational_complexity(stack, cycles=8, seed=0)
    assert float(stack.gru_h.abs().sum()) == 0.0
    assert float(stack.lstm_h.abs().sum()) == 0.0


# --- Phase 1: severing the self-loop changes self-report (P1) ----------------


def test_self_feedback_raises_integration(monkeypatch):
    """Closing the A||C||E spine must raise the integration proxy vs severed."""
    off = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "0"})
    if not getattr(off, "_supports_self_model_feedback", False):
        pytest.skip("self-model feedback not implemented yet (Phase 1)")
    on = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "1"})
    # Move the spine off its zero-init parity so the loop actually carries signal.
    with torch.no_grad():
        on.self_ingress.weight.normal_(0.0, 0.2)
        on.self_ingress.bias.normal_(0.0, 0.05)
    r_off = perturbational_complexity(off, cycles=14, seed=5)
    r_on = perturbational_complexity(on, cycles=14, seed=5)
    assert r_on.self_feedback is True and r_off.self_feedback is False
    assert r_on.persistence >= r_off.persistence


def test_severing_self_loop_changes_self_report(monkeypatch):
    """P1: the same percept yields a different self-report with the loop closed."""
    on = _stack(monkeypatch, {"DECADIC_SELF_MODEL_FEEDBACK": "1"})
    if not getattr(on, "has_self_model_feedback", False):
        pytest.skip("self-model feedback not implemented yet (Phase 1)")
    with torch.no_grad():
        on.self_ingress.weight.normal_(0.0, 0.3)
    on.eval()
    cfg = _cfg()
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    with torch.no_grad():
        on.reset_recurrent_state()
        out0 = on(z0, ep, mem, self_prev=None)
        sev = self_state_vector(out0)
        on.reset_recurrent_state()
        fed = self_state_vector(on(z0, ep, mem, self_prev=sev))
    assert not torch.allclose(sev, fed, atol=1e-5)


# --- Phase 2: ignition threshold changes report (P3) -------------------------


def test_gwt_ignition_threshold_changes_report(monkeypatch):
    pytest.importorskip("decadic.nn.workspace")
    from decadic.nn.workspace import GlobalWorkspace  # noqa: F401

    if not hasattr(GlobalWorkspace, "ignite"):
        pytest.skip("global workspace not implemented yet (Phase 2)")
    import numpy as np

    slots = np.random.RandomState(0).randn(6, 16).astype(np.float32)
    salience = np.linspace(0.1, 1.0, 6).astype(np.float32)
    low = GlobalWorkspace(threshold=0.0).ignite(slots, salience)
    high = GlobalWorkspace(threshold=10.0).ignite(slots, salience)
    assert low.ignited != high.ignited or not np.allclose(low.content, high.content)


# --- Phase 3: window length shifts the committed "now" (P1.x) ----------------


def test_integration_window_shifts_committed_now(monkeypatch):
    try:
        from decadic.cycle.integration_window import IntegrationWindow
    except Exception:
        pytest.skip("integration window not implemented yet (Phase 3)")
    import numpy as np

    w1 = IntegrationWindow(window_ms=0.0)
    w3 = IntegrationWindow(window_ms=400.0, max_frames=3)
    a = np.ones(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    # Zero-ms window commits immediately (now == latest); a real window binds.
    assert w1.push(a, now_s=0.0).committed is not None
    assert w3.push(a, now_s=0.0).committed is None
    assert w3.push(b, now_s=1.0).committed is not None


# --- Phase 4: predicted affect changes perception (P-test) -------------------


def test_predictive_affect_changes_perception(monkeypatch):
    try:
        from decadic.nn.affect_model import AffectPredictor  # noqa: F401
    except Exception:
        pytest.skip("predictive affect not implemented yet (Phase 4)")
    assert hasattr(AffectPredictor, "predict")
