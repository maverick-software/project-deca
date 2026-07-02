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


def test_gate_evaluator_identifies_discovery_namespace_mismatch():
    from decadic.evaluation.metrics import evaluate_gate
    from decadic.evaluation.types import EvalSample, MetricGate

    sample = EvalSample(
        cycle=1,
        t_s=0.0,
        metrics={},
        discovery={"discovery_health": {"object_files": 3}},
    )
    bad = evaluate_gate(
        MetricGate(name="objects", metric="object_files", mode="max", op=">=", threshold=2),
        [sample],
    )
    assert bad["satisfied"] is False
    assert bad["reason"] == "metric_present_in_discovery_not_metrics"


def test_normalize_eval_metrics_flattens_discovery_for_gates():
    from decadic.evaluation.metrics import evaluate_gate
    from decadic.evaluation.sampling import normalize_eval_metrics
    from decadic.evaluation.types import EvalSample, MetricGate

    discovery = {
        "discovery_health": {
            "object_files": 3,
            "active_proposals": 5,
            "stable_tracked_objects": 4,
            "centroid_spread": 0.12,
            "collapsed": False,
            "flow_confidence": 0.2,
            "looming_count": 1,
            "stuff_count": 2,
            "body_candidate_count": 1,
        },
        "ltm_consolidation": {"status": "accepted"},
    }
    sample = EvalSample(
        cycle=1,
        t_s=0.0,
        metrics=normalize_eval_metrics({}, discovery),
        discovery=discovery,
    )
    ok = evaluate_gate(
        MetricGate(name="objects", metric="object_files", mode="max", op=">=", threshold=2),
        [sample],
    )
    assert ok["satisfied"] is True
    assert sample.metrics["perception_collapsed"] == 0.0
    assert sample.metrics["ltm_write_accepted"] == 1.0


def test_eval_window_uses_relative_observed_cycles():
    from decadic.evaluation.sampling import eval_window, target_end_cycle

    samples = [_sample(3900), _sample(5400)]
    assert target_end_cycle(3900, 1500) == 5400
    assert eval_window(samples, 1500) == {
        "target_cycles": 1500,
        "start_cycle": 3900,
        "end_cycle": 5400,
        "observed_cycles": 1500,
    }


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
    assert report.eval_window["observed_cycles"] == 1


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
