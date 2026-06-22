"""Stable body-map sensor contract for localized touch, effort, fatigue, and pain."""

from __future__ import annotations

from typing import Any


BODY_PARTS: tuple[str, ...] = (
    "torso",
    "head",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_thigh",
    "left_shin",
    "left_foot",
    "right_thigh",
    "right_shin",
    "right_foot",
)

BODY_MAP_FIELDS: tuple[str, ...] = (
    "contact_load",
    "effort",
    "work",
    "strain",
    "fatigue",
    "pain",
)

BODY_MAP_VECTOR_DIM = len(BODY_PARTS) * len(BODY_MAP_FIELDS)
EFFORT_VECTOR_FIELDS: tuple[str, ...] = (
    "effort_total",
    "work_total",
    "strain_total",
    "fatigue_total",
    "pain_total",
)
EFFORT_VECTOR_DIM = BODY_MAP_VECTOR_DIM + len(EFFORT_VECTOR_FIELDS)
BODY_PAIN_VECTOR_DIM = len(BODY_PARTS)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x


def _coerce_list(raw: Any, n: int) -> list[float]:
    out = [0.0] * n
    if isinstance(raw, dict):
        for i, part in enumerate(BODY_PARTS[:n]):
            out[i] = _finite_float(raw.get(part, 0.0))
        return out
    if isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw[:n]):
            out[i] = _finite_float(v)
    return out


def normalize_body_map(raw: Any) -> dict[str, Any]:
    """Return a deterministic, zero-filled body-map dict."""
    src = raw if isinstance(raw, dict) else {}
    n = len(BODY_PARTS)
    out: dict[str, Any] = {"parts": list(BODY_PARTS)}
    for field in BODY_MAP_FIELDS:
        out[field] = _coerce_list(src.get(field), n)
    return out


def normalize_effort(raw: Any) -> dict[str, Any]:
    """Return aggregate effort telemetry with finite totals and safe arrays."""
    src = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {
        "actuator_effort": _coerce_list(src.get("actuator_effort"), 0),
        "actuator_work": _coerce_list(src.get("actuator_work"), 0),
        "joint_strain": _coerce_list(src.get("joint_strain"), 0),
        "joint_fatigue": _coerce_list(src.get("joint_fatigue"), 0),
        "support_effort": _finite_float(src.get("support_effort", 0.0)),
    }
    for field in EFFORT_VECTOR_FIELDS:
        out[field] = _finite_float(src.get(field, 0.0))
    return out


def flatten_body_map(body_map: Any) -> list[float]:
    bm = normalize_body_map(body_map)
    vals: list[float] = []
    for field in BODY_MAP_FIELDS:
        vals.extend(float(x) for x in bm[field])
    return vals


def effort_vector(body_map: Any, effort: Any) -> list[float]:
    vals = flatten_body_map(body_map)
    eff = normalize_effort(effort)
    vals.extend(float(eff[field]) for field in EFFORT_VECTOR_FIELDS)
    return vals


def body_pain_vector(body_map: Any) -> list[float]:
    bm = normalize_body_map(body_map)
    return [float(x) for x in bm["pain"]]


def most_pained_part(body_map: Any) -> tuple[str, float]:
    bm = normalize_body_map(body_map)
    pain = bm["pain"]
    if not pain:
        return "", 0.0
    idx = max(range(len(pain)), key=lambda i: float(pain[i]))
    val = float(pain[idx])
    return (BODY_PARTS[idx] if val > 0.0 else "", val)
