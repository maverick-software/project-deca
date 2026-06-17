"""Pure, side-effect-free gate logic for the walking curriculum.

A gate is a list of :class:`Criterion`, each comparing a rolling-window summary
of an eval-only metric against a threshold. Nothing here reads or writes agent
state, the network, or the loss - it only interprets samples the supervisor has
already collected, so it is trivially unit-testable and provably observation-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Comparator = Literal["<=", ">=", "trend>="]

_EPS = 1e-9


@dataclass(frozen=True)
class Criterion:
    """One promotion condition over a rolling window of metric samples.

    - ``<=``: the window mean of ``key`` must be at most ``threshold`` (lower is
      better, e.g. prediction error or fall-rate).
    - ``>=``: the window mean of ``key`` must be at least ``threshold`` (higher is
      better, e.g. ROM earned or gait regularity).
    - ``trend>=``: the change across the window (last - first) of ``key`` must be
      at least ``threshold`` (e.g. distance travelled or consume-count rising).
    """

    key: str
    comparator: Comparator
    threshold: float
    label: str
    unit: str = ""


@dataclass
class CriterionResult:
    label: str
    key: str
    comparator: Comparator
    threshold: float
    value: float
    satisfied: bool
    progress: float
    unit: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "key": self.key,
            "comparator": self.comparator,
            "threshold": round(self.threshold, 6),
            "value": round(self.value, 6),
            "satisfied": self.satisfied,
            "progress": round(self.progress, 4),
            "unit": self.unit,
        }


@dataclass
class GateResult:
    satisfied: bool
    progress: float
    samples: int
    enough_samples: bool
    criteria: list[CriterionResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "satisfied": self.satisfied,
            "progress": round(self.progress, 4),
            "samples": self.samples,
            "enough_samples": self.enough_samples,
            "criteria": [c.as_dict() for c in self.criteria],
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def mean_metric(window: list[dict], key: str) -> float:
    """Mean of ``key`` over the window, ignoring missing/non-numeric samples."""
    vals = [
        float(s[key])
        for s in window
        if key in s and isinstance(s[key], (int, float)) and not isinstance(s[key], bool)
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def delta_metric(window: list[dict], key: str) -> float:
    """Change in ``key`` across the window (last numeric - first numeric)."""
    vals = [
        float(s[key])
        for s in window
        if key in s and isinstance(s[key], (int, float)) and not isinstance(s[key], bool)
    ]
    if len(vals) < 2:
        return 0.0
    return vals[-1] - vals[0]


def evaluate_criterion(criterion: Criterion, window: list[dict]) -> CriterionResult:
    """Score one criterion against the window; progress is a UI hint in [0, 1]."""
    c = criterion
    if c.comparator == "<=":
        value = mean_metric(window, c.key)
        satisfied = value <= c.threshold
        # Lower is better: full bar once at/under threshold, fades as value grows.
        progress = _clamp01(c.threshold / max(value, _EPS)) if c.threshold > 0 else (
            1.0 if satisfied else 0.0
        )
    elif c.comparator == ">=":
        value = mean_metric(window, c.key)
        satisfied = value >= c.threshold
        progress = _clamp01(value / c.threshold) if c.threshold > 0 else (
            1.0 if satisfied else 0.0
        )
    else:  # trend>=
        value = delta_metric(window, c.key)
        satisfied = value >= c.threshold
        if c.threshold > 0:
            progress = _clamp01(value / c.threshold)
        else:
            progress = 1.0 if satisfied else 0.0
    return CriterionResult(
        label=c.label,
        key=c.key,
        comparator=c.comparator,
        threshold=c.threshold,
        value=value,
        satisfied=satisfied,
        progress=progress,
        unit=c.unit,
    )


def evaluate_gate(
    criteria: list[Criterion], window: list[dict], *, min_samples: int
) -> GateResult:
    """Evaluate every criterion; the gate opens only with enough samples and all met."""
    results = [evaluate_criterion(c, window) for c in criteria]
    enough = len(window) >= max(1, min_samples)
    all_met = all(r.satisfied for r in results) if results else True
    progress = (sum(r.progress for r in results) / len(results)) if results else 1.0
    return GateResult(
        satisfied=bool(enough and all_met),
        progress=progress,
        samples=len(window),
        enough_samples=enough,
        criteria=results,
    )
