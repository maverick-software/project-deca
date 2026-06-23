import json

import pytest


def _sample(cycle: int, **metrics):
    from decadic.evaluation.types import EvalSample

    base = {
        "cycles_completed": cycle,
        "loss_canary_state": "healthy",
        "loss_dominant_fraction": 0.2,
        "loss_canary_jump_ratio": 1.0,
        "nan_recovery_events": 0,
        "plasticity_frozen": False,
        "consume_events": 0,
        "net_energy_return": 0.0,
        "energy": 100.0,
        "teacher_override_fraction": 0.0,
    }
    base.update(metrics)
    return EvalSample(cycle=cycle, t_s=float(cycle), metrics=base)


def test_metric_trend_handles_rising_flat_and_nonfinite():
    from decadic.evaluation.metrics import trend_for

    samples = [_sample(0, x=1), _sample(1, x=2), _sample(2, x="bad"), _sample(3, x=4)]
    tr = trend_for(samples, "x")
    assert tr.count == 3
    assert tr.first == 1
    assert tr.last == 4
    assert tr.delta == 3
    assert tr.slope > 0
    assert tr.nonfinite == 1


def test_gate_evaluator_reports_pass_and_fail():
    from decadic.evaluation.metrics import evaluate_gate
    from decadic.evaluation.types import MetricGate

    samples = [_sample(0, consume_events=0), _sample(1, consume_events=2)]
    ok = evaluate_gate(
        MetricGate(name="consume", metric="consume_events", mode="delta", op=">=", threshold=1),
        samples,
    )
    bad = evaluate_gate(
        MetricGate(name="consume", metric="consume_events", mode="delta", op=">=", threshold=5),
        samples,
    )
    assert ok["satisfied"] is True
    assert bad["satisfied"] is False
    assert bad["reason"] == "gate_failed"


def test_build_report_passes_resource_like_stream():
    from decadic.evaluation.runner import build_report
    from decadic.evaluation.types import EvalSpec, MetricGate

    spec = EvalSpec(
        scenario="resource",
        gates=[
            MetricGate(name="consume", metric="consume_events", mode="delta", op=">=", threshold=1),
            MetricGate(name="net", metric="net_energy_return", mode="final", op=">=", threshold=0),
        ],
    )
    samples = [
        _sample(0, consume_events=0, net_energy_return=-0.1),
        _sample(1, consume_events=1, net_energy_return=0.2),
    ]
    report = build_report(spec=spec, samples=samples, agent_id="a")
    assert report.status == "pass"
    assert report.failures == []
    assert report.behavior["consume_events"]["delta"] == 1


def test_build_report_fails_dominant_loss_and_canary():
    from decadic.evaluation.runner import build_report
    from decadic.evaluation.types import EvalSpec

    samples = [
        _sample(i, loss_dominant_fraction=0.95, loss_canary_state="diverging" if i == 0 else "healthy")
        for i in range(20)
    ]
    report = build_report(spec=EvalSpec(scenario="bad"), samples=samples)
    assert report.status == "fail"
    assert any("dominant loss" in f for f in report.failures)
    assert any("loss canary" in f for f in report.failures)


def test_dojo_status_is_included():
    from decadic.evaluation.runner import build_report
    from decadic.evaluation.types import EvalSample, EvalSpec

    sample = _sample(1)
    sample.dojo = {"state": "running", "skill_id": "stand_and_recover"}
    report = build_report(spec=EvalSpec(scenario="dojo"), samples=[sample])
    assert report.behavior["dojo_latest"]["skill_id"] == "stand_and_recover"


def test_probe_bank_parser(tmp_path):
    from decadic.evaluation.probes import summarize_probe_bank

    path = tmp_path / "probes.json"
    path.write_text(
        json.dumps(
            {
                "targets": {
                    "energy": {
                        "kind": "regression",
                        "best_latent": "z5",
                        "per_latent": {"z5": {"score": 0.42, "n": 12}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_probe_bank(path)
    assert summary["available"] is True
    assert summary["targets"]["energy"]["score"] == pytest.approx(0.42)


def test_holdout_split_never_samples_holdout_for_train():
    from decadic.consolidation.replay_buffer import Transition
    from decadic.evaluation.holdout import HoldoutReplaySplit

    split = HoldoutReplaySplit(20, holdout_fraction=1.0, seed=0)
    transitions = [Transition(None, None, None, None, None, None, salience=float(i + 1)) for i in range(5)]
    for t in transitions:
        assert split.push(t) == "holdout"
    assert split.sample_train(4) == []
    assert split.sample_holdout(4)

