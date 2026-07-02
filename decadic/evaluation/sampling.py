"""Shared helpers for turning live telemetry into eval-ready samples."""

from __future__ import annotations

from typing import Any


DISCOVERY_SCALAR_KEYS = (
    "object_files",
    "active_proposals",
    "stable_tracked_objects",
    "centroid_spread",
    "flow_confidence",
    "looming_count",
    "stuff_count",
    "body_candidate_count",
)


def normalize_eval_metrics(
    metrics: dict[str, Any],
    discovery: dict[str, Any] | None = None,
    dojo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy scalar discovery/dojo diagnostics into the gate metric namespace."""

    out = dict(metrics)
    health = (discovery or {}).get("discovery_health")
    if isinstance(health, dict):
        for key in DISCOVERY_SCALAR_KEYS:
            if key in health:
                out[key] = health.get(key)
        if "collapsed" in health:
            out["perception_collapsed"] = 1.0 if health.get("collapsed") else 0.0
        if "ltm_write" in health:
            out["ltm_write_status"] = str(health.get("ltm_write") or "")

    if "object_files" not in out and isinstance(discovery, dict):
        files = discovery.get("object_files")
        if isinstance(files, list):
            out["object_files"] = len(files)

    ltm = (discovery or {}).get("ltm_consolidation")
    if isinstance(ltm, dict):
        status = str(ltm.get("status") or "")
        out["ltm_write_status"] = status
        out["ltm_write_accepted"] = 1.0 if status in {"accepted", "promoted_entity"} else 0.0
    elif "ltm_write_accepted" not in out:
        status = str(out.get("ltm_write_status") or "")
        out["ltm_write_accepted"] = 1.0 if status in {"accepted", "promoted_entity"} else 0.0

    if isinstance(dojo, dict):
        for key in (
            "caregiver_enabled",
            "caregiver_missing_parent",
            "caregiver_pending",
            "caregiver_delivery_count",
        ):
            if key in dojo and key not in out:
                value = dojo.get(key)
                if isinstance(value, bool):
                    out[key] = 1.0 if value else 0.0
                else:
                    out[key] = value
        for key in ("caregiver_status", "caregiver_kind", "caregiver_request_kind"):
            if key in dojo and key not in out:
                out[key] = dojo.get(key)

    return out


def eval_window(samples: list[Any], target_cycles: int) -> dict[str, Any]:
    """Describe an eval as a relative observation window."""

    if not samples:
        return {
            "target_cycles": int(target_cycles),
            "start_cycle": None,
            "end_cycle": None,
            "observed_cycles": 0,
        }
    start = int(getattr(samples[0], "cycle", 0) or 0)
    end = int(getattr(samples[-1], "cycle", start) or start)
    return {
        "target_cycles": int(target_cycles),
        "start_cycle": start,
        "end_cycle": end,
        "observed_cycles": max(0, end - start),
    }


def target_end_cycle(start_cycle: int, target_cycles: int) -> int:
    return int(start_cycle) + max(1, int(target_cycles))
