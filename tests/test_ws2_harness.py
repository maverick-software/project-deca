"""WS2 harness unit tests: sampler, rollups, soak gates, report generation."""

import json
import subprocess
import sys
from pathlib import Path

from decadic.metrics.harness import (
    HarnessSampler,
    evaluate_soak_gates,
    load_samples,
    rollup,
    write_rollup_csv,
)

ROOT = Path(__file__).resolve().parents[1]


# -- fixtures ----------------------------------------------------------------

def _make_samples(n: int = 240, *, pc_start: float = 2.0, pc_end: float = 0.2,
                  rate_hz: float = 10.0, t0: float = 999_960.0) -> list[dict]:
    # t0 is minute-aligned (999_960 % 60 == 0) so bucket counts are exact.
    """n samples at 2s intervals with linearly falling pc loss and steady cycles."""
    rows = []
    for i in range(n):
        frac = i / max(1, n - 1)
        rows.append(
            {
                "ts": t0 + i * 2.0,
                "cycles_completed": int(rate_hz * i * 2.0),
                "neural_pc_loss_last": pc_start + (pc_end - pc_start) * frac,
                "loss_total": 2.2 + (0.3 - 2.2) * frac,
                "viability": 90.0 - 5.0 * frac,
                "nan_recovery_events": 0,
                "growth_events": 2 if frac > 0.5 else 0,
                "priority_label": "explore" if i % 3 else "investigate",
                "a_norm": 1.0 + 0.01 * i,
            }
        )
    return rows


def _fake_run_dir(tmp_path: Path, samples: list[dict]) -> Path:
    run = tmp_path / "soak_test"
    run.mkdir()
    with (run / "harness_samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    (run / "manifest.json").write_text(
        json.dumps({"git_sha": "abc1234", "hours": 0.1, "env": {"DECADIC_CONSOLIDATION_ENABLED": "1"}}),
        encoding="utf-8",
    )
    verdict = evaluate_soak_gates(samples, stall_events=0)
    (run / "gates.json").write_text(json.dumps(verdict), encoding="utf-8")
    (run / "soak_summary.json").write_text(
        json.dumps({"result": "completed", "stall_events": 0, "n_samples": len(samples)}),
        encoding="utf-8",
    )
    return run


# -- sampler ---------------------------------------------------------------

def test_sampler_flattens_metrics_and_state(tmp_path):
    def fake_http(url: str):
        if url.endswith("/metrics"):
            return {"metrics": {"cycles_completed": 42, "neural_pc_loss_last": 0.5,
                                "priority_label": "explore", "paused": False,
                                "bad": float("nan")}}
        if url.endswith("/state"):
            return {"payload": {"state_bus": {"state_of_mind": [3.0, 4.0],
                                              "pain_scalar": 0.25}}}
        return None

    sampler = HarnessSampler(
        "http://x", "aid", tmp_path / "s.jsonl", collect_gpu=False, http_get=fake_http
    )
    row = sampler.write_sample()
    assert row is not None
    assert row["cycles_completed"] == 42
    assert row["priority_label"] == "explore"
    assert row["paused"] == 0  # bool -> int
    assert row["bad"] is None and row["nonfinite_metric_seen"] == 1
    assert abs(row["a_norm"] - 5.0) < 1e-9  # l2([3,4])
    assert row["state_pain_scalar"] == 0.25
    # second sample computes drift
    row2 = sampler.write_sample()
    assert row2["a_norm_drift"] == 0.0
    assert len(load_samples(tmp_path / "s.jsonl")) == 2


def test_sampler_survives_http_failure(tmp_path):
    sampler = HarnessSampler(
        "http://x", "aid", tmp_path / "s.jsonl", collect_gpu=False, http_get=lambda url: None
    )
    assert sampler.write_sample() is None
    assert sampler.errors == 1


# -- rollups ---------------------------------------------------------------

def test_rollup_buckets_and_cycle_rate(tmp_path):
    samples = _make_samples(n=120, rate_hz=10.0)  # 4 minutes of 2s samples
    rows = rollup(samples, bucket_seconds=60)
    assert len(rows) == 4
    # steady 10 Hz -> derived rate close to 10 for buckets after the first
    rates = [r["cycle_rate_hz"] for r in rows if "cycle_rate_hz" in r]
    assert rates and all(abs(r - 10.0) < 0.5 for r in rates)
    assert rows[0]["n_samples"] == 30
    assert "neural_pc_loss_last__mean" in rows[0]
    assert rows[0]["priority_label__mode"] == "explore"
    path = write_rollup_csv(rows, tmp_path / "r.csv")
    assert Path(path).stat().st_size > 0


def test_rollup_empty():
    assert rollup([]) == []


# -- gates --------------------------------------------------------------------

def test_gates_pass_on_healthy_run():
    verdict = evaluate_soak_gates(_make_samples(), stall_events=0)
    assert verdict["passed"] is True
    names = {g["name"]: g for g in verdict["gates"]}
    assert names["no stalls"]["ok"] is True
    assert names["pc loss decreases"]["ok"] is True
    assert names["no NaN recoveries"]["ok"] is True


def test_gates_fail_on_stall_and_rising_loss():
    samples = _make_samples(pc_start=0.2, pc_end=2.0)  # rising loss
    verdict = evaluate_soak_gates(samples, stall_events=1)
    assert verdict["passed"] is False
    names = {g["name"]: g for g in verdict["gates"]}
    assert names["no stalls"]["ok"] is False
    assert names["pc loss decreases"]["ok"] is False


def test_gates_nan_events_detected():
    samples = _make_samples()
    samples[-1]["nan_recovery_events"] = 3
    verdict = evaluate_soak_gates(samples, stall_events=0)
    names = {g["name"]: g for g in verdict["gates"]}
    assert names["no NaN recoveries"]["ok"] is False


# -- report generator ------------------------------------------------------------

def test_single_run_report_generation(tmp_path):
    run = _fake_run_dir(tmp_path, _make_samples())
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_run_report.py"), str(run)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    report = (run / "report.md").read_text(encoding="utf-8")
    for heading in ("## Verdict", "## 1. Stability", "## 2. Learning",
                    "## 3. State coherence", "## 4. Memory", "## 5. Distinctness"):
        assert heading in report
    assert "PASS" in report


def test_comparison_report_generation(tmp_path):
    run_a = _fake_run_dir(tmp_path, _make_samples())
    run_b_samples = _make_samples(pc_start=2.0, pc_end=0.5)
    run_b = tmp_path / "soak_b"
    run_b.mkdir()
    with (run_b / "harness_samples.jsonl").open("w", encoding="utf-8") as f:
        for s in run_b_samples:
            f.write(json.dumps(s) + "\n")
    (run_b / "manifest.json").write_text(
        json.dumps({"git_sha": "abc1234", "env": {"DECADIC_CONSOLIDATION_ENABLED": "0"}}),
        encoding="utf-8",
    )
    (run_b / "soak_summary.json").write_text(json.dumps({"result": "completed"}), encoding="utf-8")
    (run_b / "gates.json").write_text(
        json.dumps(evaluate_soak_gates(run_b_samples, stall_events=0)), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_run_report.py"),
         "--compare", str(run_a), str(run_b)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    compare_files = list(run_a.glob("compare_*.md"))
    assert compare_files, "comparison report not written"
    text = compare_files[0].read_text(encoding="utf-8")
    assert "neural_pc_loss_last" in text and "Overlaid trends" in text
