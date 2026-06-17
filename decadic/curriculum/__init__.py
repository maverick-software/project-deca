"""Walking curriculum: an observation-only developmental trainer.

The curriculum shapes the world and reads gates; it never adds a term to the
loss. See :mod:`decadic.curriculum.supervisor` for the runtime state machine,
:mod:`decadic.curriculum.phases` for the phase table, and
:mod:`decadic.curriculum.gates` for the pure gate logic.
"""

from __future__ import annotations

from decadic.curriculum.gates import (
    Criterion,
    CriterionResult,
    GateResult,
    evaluate_criterion,
    evaluate_gate,
)
from decadic.curriculum.phases import (
    CURRICULUM_SCENE,
    Phase,
    PhaseConfig,
    SatisfierPolicy,
    affective_phase,
    build_phases,
    default_phases,
)
from decadic.curriculum.supervisor import CurriculumError, CurriculumSupervisor

__all__ = [
    "Criterion",
    "CriterionResult",
    "GateResult",
    "evaluate_criterion",
    "evaluate_gate",
    "CURRICULUM_SCENE",
    "Phase",
    "PhaseConfig",
    "SatisfierPolicy",
    "affective_phase",
    "build_phases",
    "default_phases",
    "CurriculumError",
    "CurriculumSupervisor",
]
