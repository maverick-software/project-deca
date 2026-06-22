"""Typed Skill Dojo primitives.

The dojo surrounds the Decadic loop with curricula, teacher hints, replay
metadata, and gates. These types are intentionally small and serializable so new
skills can be added without touching the cognitive cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from decadic.training.gates import Criterion


@dataclass(frozen=True)
class SkillGate:
    """Named gate wrapper for status/reporting."""

    criteria: tuple[Criterion, ...]
    min_samples: int = 8


@dataclass(frozen=True)
class PeriodicBodyCommand:
    """Body/world command repeated while a phase is active."""

    command: str
    period_s: float


@dataclass(frozen=True)
class TeacherAdaptation:
    """Closed-loop assist-as-needed teacher policy for one phase."""

    enabled: bool = True
    min_weight: float = 0.0
    max_weight: float = 0.0
    rise_rate: float = 0.5
    fade_rate: float = 0.08
    danger_thresholds: dict[str, float] = field(default_factory=dict)
    stability_thresholds: dict[str, float] = field(default_factory=dict)
    stable_dwell_s: float = 3.0
    unstable_dwell_s: float = 0.0
    zero_required_for_graduation: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "rise_rate": self.rise_rate,
            "fade_rate": self.fade_rate,
            "danger_thresholds": dict(self.danger_thresholds),
            "stability_thresholds": dict(self.stability_thresholds),
            "stable_dwell_s": self.stable_dwell_s,
            "unstable_dwell_s": self.unstable_dwell_s,
            "zero_required_for_graduation": self.zero_required_for_graduation,
        }


@dataclass(frozen=True)
class SkillPhase:
    """One dojo phase: optional config, body commands, teacher weight, and gates."""

    index: int
    name: str
    description: str
    teacher_weight: float
    config: dict[str, Any] = field(default_factory=dict)
    body_commands: tuple[str, ...] = ()
    periodic_body_commands: tuple[PeriodicBodyCommand, ...] = ()
    gate: SkillGate = field(default_factory=lambda: SkillGate(tuple()))
    failure_gate: SkillGate = field(default_factory=lambda: SkillGate(tuple(), min_samples=1))
    reset_commands: tuple[str, ...] = ()
    timeout_s: float = 0.0
    max_attempts: int = 3
    auto_retry: bool = True
    min_dwell_s: float = 20.0
    demote_on_death: bool = False
    is_terminal: bool = False
    teacher_adaptation: TeacherAdaptation | None = None


@dataclass(frozen=True)
class SkillSpec:
    """Reusable skill definition."""

    skill_id: str
    version: str
    name: str
    description: str
    target_behavior: str
    teacher: str
    phases: tuple[SkillPhase, ...]
    required_sensors: tuple[str, ...] = ("proprioception", "contacts")
    checkpoint_on_graduate: bool = True
    caregiver_enabled: bool = False
    caregiver_threshold: float = 80.0
    warnings: tuple[str, ...] = ()

    def as_dict(self, *, source: str = "builtin") -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "target_behavior": self.target_behavior,
            "teacher": self.teacher,
            "source": source,
            "builtin": source == "builtin",
            "required_sensors": list(self.required_sensors),
            "checkpoint_on_graduate": self.checkpoint_on_graduate,
            "caregiver_enabled": self.caregiver_enabled,
            "caregiver_threshold": self.caregiver_threshold,
            "warnings": list(self.warnings),
            "phases": [
                {
                    "index": p.index,
                    "name": p.name,
                    "description": p.description,
                    "teacher_weight": p.teacher_weight,
                    "teacher_adaptation": (
                        p.teacher_adaptation.as_dict()
                        if p.teacher_adaptation is not None
                        else None
                    ),
                    "config": dict(p.config),
                    "body_commands": list(p.body_commands),
                    "periodic_body_commands": [
                        {"command": c.command, "period_s": c.period_s}
                        for c in p.periodic_body_commands
                    ],
                    "min_dwell_s": p.min_dwell_s,
                    "timeout_s": p.timeout_s,
                    "max_attempts": p.max_attempts,
                    "auto_retry": p.auto_retry,
                    "reset_commands": list(p.reset_commands),
                    "min_samples": p.gate.min_samples,
                    "demote_on_death": p.demote_on_death,
                    "is_terminal": p.is_terminal,
                    "criteria": [
                        {
                            "key": c.key,
                            "comparator": c.comparator,
                            "threshold": c.threshold,
                            "label": c.label,
                            "unit": c.unit,
                        }
                        for c in p.gate.criteria
                    ],
                    "failure_criteria": [
                        {
                            "key": c.key,
                            "comparator": c.comparator,
                            "threshold": c.threshold,
                            "label": c.label,
                            "unit": c.unit,
                        }
                        for c in p.failure_gate.criteria
                    ],
                    "failure_min_samples": p.failure_gate.min_samples,
                }
                for p in self.phases
            ],
        }

    def upload_dict(self) -> dict[str, Any]:
        """Schema written for uploaded skills (no derived source/builtin fields)."""
        d = self.as_dict(source="uploaded")
        d.pop("source", None)
        d.pop("builtin", None)
        return d


@dataclass
class SkillRunStatus:
    """Serializable state of the active dojo run."""

    state: str = "stopped"
    agent_id: str | None = None
    skill_id: str | None = None
    phase_index: int | None = None
    phase_name: str | None = None
    teacher_weight: float = 0.0
    teacher_assist: float = 0.0
    teacher_origin: str = "self"
    assist_reason: str = ""
    samples: int = 0
    graduated: bool = False
    gate: dict[str, Any] | None = None
    report_path: str | None = None
    error: str | None = None


@dataclass
class DemoRecord:
    """A compact dojo episode sample for reports and later dataset export."""

    skill_id: str
    phase_index: int
    origin: str
    metrics: dict[str, float]
    teacher_motor: list[float] | None = None
    teacher_weight: float = 0.0
    assist_reason: str = ""
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "phase_index": self.phase_index,
            "origin": self.origin,
            "metrics": dict(self.metrics),
            "teacher_motor": list(self.teacher_motor) if self.teacher_motor else None,
            "teacher_weight": self.teacher_weight,
            "assist_reason": self.assist_reason,
            "success": self.success,
            "timestamp": self.timestamp,
        }


class TeacherPolicy(Protocol):
    """Protocol for skill teachers that provide consolidation-only hints."""

    name: str

    def motor_target(self, *, n_actuators: int, metrics: dict[str, float]) -> list[float]:
        """Return normalized motor targets in [-1, 1]."""
        ...
