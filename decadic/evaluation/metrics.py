"""Metric trend and gate scoring helpers for eval reports."""

from __future__ import annotations

import math
from typing import Any

from decadic.evaluation.types import EvalSample, MetricGate, MetricTrend


def _as_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metric_values(samples: list[EvalSample], metric: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        v = _as_float(sample.metrics.get(metric))
        if v is not None:
            values.append(v)
    return values


def trend_for(samples: list[EvalSample], metric: str) -> MetricTrend:
    xs: list[float] = []
    ys: list[float] = []
    nonfinite = 0
    for idx, sample in enumerate(samples):
        raw = sample.metrics.get(metric)
        v = _as_float(raw)
        if v is None:
            if raw is not None:
                nonfinite += 1
            continue
        xs.append(float(idx))
        ys.append(v)
    if not ys:
        return MetricTrend(metric=metric, nonfinite=nonfinite)
    first = ys[0]
    last = ys[-1]
    slope = 0.0
    if len(ys) >= 2:
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom > 1e-12:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    return MetricTrend(
        metric=metric,
        count=len(ys),
        first=first,
        last=last,
        delta=last - first,
        min=min(ys),
        max=max(ys),
        slope=slope,
        nonfinite=nonfinite,
    )


def compare(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op in ("==", "="):
        return abs(value - threshold) < 1e-9
    raise ValueError(f"unsupported gate op {op!r}")


def evaluate_gate(gate: MetricGate, samples: list[EvalSample]) -> dict[str, Any]:
    vals = metric_values(samples, gate.metric)
    tr = trend_for(samples, gate.metric)
    if not vals:
        return {
            "name": gate.name,
            "metric": gate.metric,
            "satisfied": False,
            "reason": "missing_metric",
            "trend": tr.to_dict(),
        }
    if gate.mode == "final":
        value = vals[-1]
        ok = compare(value, gate.op, gate.threshold)
    elif gate.mode == "delta":
        value = float(tr.delta or 0.0)
        ok = compare(value, gate.op, gate.threshold)
    elif gate.mode == "max":
        value = max(vals)
        ok = compare(value, gate.op, gate.threshold)
    elif gate.mode == "min":
        value = min(vals)
        ok = compare(value, gate.op, gate.threshold)
    elif gate.mode in ("fraction<=", "fraction>="):
        matches = sum(1 for v in vals if compare(v, gate.op, gate.threshold))
        value = matches / max(1, len(vals))
        ok = value >= gate.fraction
    else:
        raise ValueError(f"unsupported gate mode {gate.mode!r}")
    return {
        "name": gate.name,
        "metric": gate.metric,
        "mode": gate.mode,
        "op": gate.op,
        "threshold": gate.threshold,
        "value": value,
        "satisfied": bool(ok),
        "reason": "ok" if ok else "gate_failed",
        "trend": tr.to_dict(),
    }

