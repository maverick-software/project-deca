"""WS5-M0: gate decision log + shadow deliberation tap.

M0.1 -- GateDecisionLog: buffered JSONL, log-and-continue on IO failure,
        off-by-default config.
M0.2 -- stage4_shadow: fresh risk_mlp(z3) computed beside a stage4_override,
        no_grad diagnostics only; the forward's live outputs must be
        bit-identical with the tap on or off.
"""

import json

import pytest

from decadic.cycle.attention_gate import (
    GateDecisionLog,
    gate_log_enabled,
    gate_shadow_rate,
    shadow_sampled,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------- M0.1


def test_gate_log_config_defaults(monkeypatch):
    monkeypatch.delenv("DECADIC_GATE_LOG", raising=False)
    assert gate_log_enabled() is False  # zero new IO on existing runs
    monkeypatch.setenv("DECADIC_GATE_LOG", "1")
    assert gate_log_enabled() is True

    monkeypatch.delenv("DECADIC_GATE_SHADOW_RATE", raising=False)
    assert gate_shadow_rate() == pytest.approx(0.05)
    monkeypatch.setenv("DECADIC_GATE_SHADOW_RATE", "7")  # clamped
    assert gate_shadow_rate() == 1.0
    monkeypatch.setenv("DECADIC_GATE_SHADOW_RATE", "junk")  # tolerated
    assert gate_shadow_rate() == pytest.approx(0.05)


def test_shadow_sampling_deterministic_and_near_rate():
    assert shadow_sampled(1234, 0.0) is False
    assert shadow_sampled(1234, 1.0) is True
    # Deterministic: same cycle, same answer, no RNG state involved.
    assert shadow_sampled(777, 0.05) == shadow_sampled(777, 0.05)
    # Over many cycles the hit rate lands near the configured rate.
    hits = sum(1 for c in range(20_000) if shadow_sampled(c, 0.05))
    assert 0.03 < hits / 20_000 < 0.07


def test_gate_decision_log_buffers_flushes_and_survives(tmp_path):
    path = tmp_path / "gate" / "decisions.jsonl"
    log = GateDecisionLog(path)
    for i in range(GateDecisionLog.FLUSH_EVERY + 5):
        log.log({"cycle": i, "escalate": i % 2})
    # Auto-flush fired at FLUSH_EVERY; the tail is still buffered.
    on_disk = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(on_disk) >= GateDecisionLog.FLUSH_EVERY
    log.close()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == GateDecisionLog.FLUSH_EVERY + 5
    assert rows[3] == {"cycle": 3, "escalate": 1}

    # A malformed row is dropped, never raised.
    log2 = GateDecisionLog(tmp_path / "gate2.jsonl")
    log2.log({"bad": object()})
    log2.log({"ok": 1})
    log2.close()
    rows2 = (tmp_path / "gate2.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows2) == 1 and json.loads(rows2[0]) == {"ok": 1}


def test_gate_decision_log_io_failure_disables_quietly(tmp_path):
    # Parent "directory" is a FILE -> mkdir and appends fail -> sink disables
    # itself without raising (the cognitive loop never pays for telemetry).
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    log = GateDecisionLog(blocker / "sub" / "decisions.jsonl")
    for i in range(100):
        log.log({"cycle": i})
    log.flush()
    log.close()  # no exception = pass
    assert log._failed is True


# --------------------------------------------------------------------- M0.2


def _tiny_stack(monkeypatch):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(neural_config_from_env("tiny")), neural_config_from_env(
        "tiny"
    )


def test_stage4_shadow_is_pure_diagnostics(monkeypatch):
    """Live outputs bit-identical with the shadow tap on vs off; shadow keys
    carry a fresh deliberation that can disagree with the substituted one."""
    stack, cfg = _tiny_stack(monkeypatch)
    stack.eval()
    torch.manual_seed(11)
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)

    with torch.no_grad():
        # Reference deliberation supplies realistic override shapes.
        stack.reset_recurrent_state()
        base = stack(z0, ep, mem)
        override = (base["z4"].detach() * 0.5, base["risk_logit"].detach() * 0.5)

        stack.reset_recurrent_state()
        off = stack(z0, ep, mem, stage4_override=override, stage4_shadow=False)
        stack.reset_recurrent_state()
        on = stack(z0, ep, mem, stage4_override=override, stage4_shadow=True)

    assert "shadow_z4" not in off
    assert "shadow_z4" in on and "shadow_risk_logit" in on
    assert on["shadow_z4"].shape == on["z4"].shape

    # Bit-identical live path: every shared tensor output matches exactly.
    for k, v in off.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, on[k]), f"shadow tap perturbed output {k!r}"

    # The shadow is FRESH deliberation: with a halved override it must
    # disagree with the substituted z4 (that divergence IS the regret signal).
    assert not torch.allclose(on["shadow_z4"], on["z4"])
    # And it matches what un-overridden stage 4 would have produced.
    assert torch.allclose(on["shadow_z4"], base["z4"], atol=1e-5)


def test_stage4_shadow_requires_override(monkeypatch):
    """Shadow on an escalated (no-override) cycle is a no-op: fresh stage 4
    already ran; the counterfactual there is computed by the caller."""
    stack, cfg = _tiny_stack(monkeypatch)
    stack.eval()
    torch.manual_seed(12)
    with torch.no_grad():
        stack.reset_recurrent_state()
        out = stack(
            torch.randn(1, cfg.d_model),
            torch.rand(1, 4),
            torch.randn(1, cfg.memory_context_dim),
            stage4_shadow=True,
        )
    assert "shadow_z4" not in out
