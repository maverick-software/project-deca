import math

import pytest


def test_loss_canary_jump_ratio_fires_during_warmup(monkeypatch):
    from decadic.cycle.objective_health import ObjectiveHealthCanary

    monkeypatch.setenv("DECADIC_LOSS_CANARY_WARMUP_CYCLES", "20")
    monkeypatch.setenv("DECADIC_LOSS_CANARY_HARD_JUMP_RATIO", "25")
    c = ObjectiveHealthCanary()
    states = [
        c.update(total_loss=2.47, pc_loss=2.47, forward_finite=True).state,
        c.update(total_loss=13.0, pc_loss=13.0, forward_finite=True).state,
        c.update(total_loss=461.0, pc_loss=461.0, forward_finite=True).state,
    ]
    assert states[0] == "warming"
    assert states[1] == "warming"
    assert states[2] == "diverging"
    assert c.last_report.optimizer_action == "skipped"


def test_loss_canary_nonfinite_fires_on_first_cycle():
    from decadic.cycle.objective_health import ObjectiveHealthCanary

    c = ObjectiveHealthCanary()
    report = c.update(total_loss=math.inf, pc_loss=1.0, forward_finite=True)
    assert report.state == "diverging"
    assert report.reason == "nonfinite"
    assert report.optimizer_action == "skipped"


def test_loss_canary_ema_waits_for_warmup(monkeypatch):
    from decadic.cycle.objective_health import ObjectiveHealthCanary

    monkeypatch.setenv("DECADIC_LOSS_CANARY_WARMUP_CYCLES", "5")
    monkeypatch.setenv("DECADIC_LOSS_CANARY_HARD_JUMP_RATIO", "1000")
    monkeypatch.setenv("DECADIC_LOSS_CANARY_WARN_PCEMA", "3")
    c = ObjectiveHealthCanary()
    early = [c.update(total_loss=4.0, pc_loss=4.0, forward_finite=True) for _ in range(4)]
    assert all(r.state == "warming" for r in early)
    late = c.update(total_loss=4.0, pc_loss=4.0, forward_finite=True)
    assert late.state == "warning"
    assert late.optimizer_action == "scaled"
    assert late.step_scale == pytest.approx(0.25)

