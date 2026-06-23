"""Serializable types for training evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MetricGate:
    name: str
    metric: str
    op: str
    threshold: float
    mode: str = "final"  # final | delta | max | min | fraction<= | fraction>=
    fraction: float = 0.9

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetricGate":
        return cls(
            name=str(raw.get("name") or raw.get("metric") or "gate"),
            metric=str(raw["metric"]),
            op=str(raw.get("op", ">=")),
            threshold=float(raw.get("threshold", 0.0)),
            mode=str(raw.get("mode", "final")),
            fraction=float(raw.get("fraction", 0.9)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSpec:
    scenario: str
    description: str = ""
    cycles: int = 500
    seeds: list[int] = field(default_factory=lambda: [1])
    agent_preset: str | None = None
    dojo_skill_id: str | None = None
    baseline: str | None = None
    poll_interval_s: float = 1.0
    timeout_s: float = 600.0
    gates: list[MetricGate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalSpec":
        return cls(
            scenario=str(raw["scenario"]),
            description=str(raw.get("description", "")),
            cycles=int(raw.get("cycles", 500)),
            seeds=[int(x) for x in raw.get("seeds", [1])],
            agent_preset=raw.get("agent_preset"),
            dojo_skill_id=raw.get("dojo_skill_id"),
            baseline=raw.get("baseline"),
            poll_interval_s=float(raw.get("poll_interval_s", 1.0)),
            timeout_s=float(raw.get("timeout_s", 600.0)),
            gates=[MetricGate.from_dict(g) for g in raw.get("gates", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gates"] = [g.to_dict() for g in self.gates]
        return d


@dataclass
class EvalSample:
    cycle: int
    t_s: float
    metrics: dict[str, Any]
    discovery: dict[str, Any] | None = None
    dojo: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricTrend:
    metric: str
    count: int = 0
    first: float | None = None
    last: float | None = None
    delta: float | None = None
    min: float | None = None
    max: float | None = None
    slope: float = 0.0
    nonfinite: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    scenario: str
    status: str
    agent_id: str | None
    seeds: list[int]
    health: dict[str, Any] = field(default_factory=dict)
    mechanical: dict[str, Any] = field(default_factory=dict)
    learning: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    samples_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

