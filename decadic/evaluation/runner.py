"""Build training evaluation reports from sampled runtime telemetry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from decadic.evaluation.metrics import evaluate_gate, metric_values, trend_for
from decadic.evaluation.probes import summarize_probe_bank
from decadic.evaluation.sampling import eval_window
from decadic.evaluation.types import EvalReport, EvalSample, EvalSpec


SCENARIO_DIR = Path(__file__).resolve().parents[2] / "docs" / "eval_scenarios"


def load_eval_spec(name_or_path: str | Path) -> EvalSpec:
    p = Path(name_or_path)
    if not p.exists():
        p = SCENARIO_DIR / f"{name_or_path}.json"
    with p.open(encoding="utf-8") as f:
        return EvalSpec.from_dict(json.load(f))


def write_samples_jsonl(samples: list[EvalSample], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
    return str(p)


def _fraction(samples: list[EvalSample], pred) -> float:
    if not samples:
        return 0.0
    return sum(1 for s in samples if pred(s)) / len(samples)


def _last_metric(samples: list[EvalSample], key: str, default: Any = None) -> Any:
    for sample in reversed(samples):
        if key in sample.metrics:
            return sample.metrics.get(key)
    return default


def _health(samples: list[EvalSample], failures: list[str]) -> dict[str, Any]:
    diverging_frac = _fraction(samples, lambda s: s.metrics.get("loss_canary_state") == "diverging")
    dominant_ok_frac = _fraction(
        samples,
        lambda s: float(s.metrics.get("loss_dominant_fraction", 0.0) or 0.0) <= 0.7,
    )
    nan_tr = trend_for(samples, "nan_recovery_events")
    jump_vals = metric_values(samples[20:] if len(samples) > 20 else samples, "loss_canary_jump_ratio")
    max_jump = max(jump_vals) if jump_vals else 1.0
    frozen_final = bool(_last_metric(samples, "plasticity_frozen", False))
    freeze_count = int(float(_last_metric(samples, "plasticity_freeze_count", 0) or 0))
    thaw_count = int(float(_last_metric(samples, "plasticity_thaw_count", 0) or 0))
    if diverging_frac > 0.01:
        failures.append(f"loss canary diverged for {diverging_frac:.1%} of samples")
    if int(nan_tr.last or 0) > 0:
        failures.append(f"nan recovery events occurred: {int(nan_tr.last or 0)}")
    if frozen_final and thaw_count < freeze_count:
        failures.append("plasticity remained frozen at end of eval")
    if dominant_ok_frac < 0.9:
        failures.append(f"dominant loss fraction healthy for only {dominant_ok_frac:.1%} of samples")
    if max_jump >= 25.0:
        failures.append(f"loss jump ratio exceeded hard threshold after startup: {max_jump:.2f}")
    return {
        "samples": len(samples),
        "canary_diverging_fraction": diverging_frac,
        "dominant_loss_ok_fraction": dominant_ok_frac,
        "nan_recovery_events": int(nan_tr.last or 0),
        "plasticity_frozen_final": frozen_final,
        "plasticity_freeze_count": freeze_count,
        "plasticity_thaw_count": thaw_count,
        "max_loss_jump_after_startup": max_jump,
        "loss_total": trend_for(samples, "loss_total").to_dict(),
        "neural_pc_loss": trend_for(samples, "neural_pc_loss_last").to_dict(),
    }


def _behavior(samples: list[EvalSample]) -> dict[str, Any]:
    keys = (
        "cycles_completed",
        "energy",
        "hydration",
        "integrity",
        "consume_events",
        "resource_relief_events",
        "net_energy_return",
        "distance_traveled",
        "fall_rate",
        "teacher_override_fraction",
        "root_height",
        "torso_tilt",
    )
    out = {k: trend_for(samples, k).to_dict() for k in keys}
    latest_dojo = next((s.dojo for s in reversed(samples) if s.dojo), None)
    if latest_dojo:
        out["dojo_latest"] = latest_dojo
        out["caregiver_latest"] = {
            "enabled": bool(latest_dojo.get("caregiver_enabled", False)),
            "status": latest_dojo.get("caregiver_status"),
            "kind": latest_dojo.get("caregiver_kind"),
            "missing_parent": bool(latest_dojo.get("caregiver_missing_parent", False)),
            "delivery_count": latest_dojo.get("caregiver_delivery_count"),
            "request_kind": latest_dojo.get("caregiver_request_kind"),
        }
    return out


def _perception(samples: list[EvalSample]) -> dict[str, Any]:
    latest = next((s.discovery for s in reversed(samples) if s.discovery), None)
    metrics = {
        "object_files": trend_for(samples, "object_files").to_dict(),
        "discovered_objects": trend_for(samples, "discovered_objects").to_dict(),
        "ltm_property_beliefs": trend_for(samples, "ltm_property_beliefs").to_dict(),
    }
    if latest:
        metrics["latest_discovery"] = latest
        metrics["latest_discovery_health"] = latest.get("discovery_health")
    return metrics


def _learning(samples: list[EvalSample]) -> dict[str, Any]:
    return {
        "forward_model_error": trend_for(samples, "forward_model_error").to_dict(),
        "tactile_pred_error": trend_for(samples, "tactile_pred_error").to_dict(),
        "effort_pred_error": trend_for(samples, "effort_pred_error").to_dict(),
        "intero_pred_error": trend_for(samples, "intero_pred_error").to_dict(),
        "consolidator_loss": trend_for(samples, "consolidator_loss").to_dict(),
    }


def build_report(
    *,
    spec: EvalSpec,
    samples: list[EvalSample],
    agent_id: str | None = None,
    samples_path: str = "",
    probe_bank: str | Path | None = None,
    baseline_report: dict[str, Any] | None = None,
) -> EvalReport:
    failures: list[str] = []
    gate_reports = [evaluate_gate(g, samples) for g in spec.gates]
    for gate in gate_reports:
        if not gate["satisfied"]:
            failures.append(f"gate failed: {gate['name']} ({gate['reason']})")
    health = _health(samples, failures)
    behavior = _behavior(samples)
    latest_dojo = behavior.get("dojo_latest")
    if isinstance(latest_dojo, dict):
        reason = str(latest_dojo.get("failure_reason") or "")
        if reason == "setup_failed_missing_parent":
            failures.append("setup failed: missing caregiver parent (setup_failed_missing_parent)")
    behavior["gates"] = gate_reports
    report = EvalReport(
        scenario=spec.scenario,
        status="pass",
        agent_id=agent_id,
        seeds=spec.seeds,
        health=health,
        mechanical={
            "available": False,
            "reason": "mechanical module-gradient checks are covered by in-process tests in v1",
        },
        learning=_learning(samples),
        perception=_perception(samples),
        probes=summarize_probe_bank(probe_bank),
        behavior=behavior,
        baseline_comparison=baseline_report or {},
        eval_window=eval_window(samples, spec.cycles),
        failures=failures,
        samples_path=samples_path,
    )
    if failures:
        report.status = "fail"
    return report


def report_path_for(scenario: str, out_dir: str | Path = "reports") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / f"training_eval_{scenario}_{stamp}.json"
