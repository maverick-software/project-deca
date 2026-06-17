"""Developmental phase table for the walking curriculum.

Each phase only (a) sets live config knobs that *reweight* the existing self-
supervised objective, (b) chooses a satisfier-placement policy, and (c) declares
the observational gate that promotes the agent to the next phase. The world is a
single fixed scene throughout; no phase ever adds a reward or a loss term. The
brace ROM curriculum (welded -> earned range of motion) progresses automatically
from the body's own per-joint prediction error, so phases need no body restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decadic.curriculum.gates import Criterion


@dataclass
class PhaseConfig:
    """Live ``AgentRuntime.configure`` knobs to apply on entering a phase.

    Every field is optional; ``None`` leaves the agent's current value untouched
    (and, for the active-inference weights, falls back to the process-env default
    inside the cycle - exact parity).
    """

    viability_mode: str | None = None
    metabolic_compression: float | None = None
    ai_intero_pref_weight: float | None = None
    drive_priority_gain: float | None = None
    motor_babble_sigma: float | None = None

    def to_configure_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.viability_mode is not None:
            out["viability_mode"] = self.viability_mode
        if self.metabolic_compression is not None:
            out["metabolic_compression"] = self.metabolic_compression
        if self.ai_intero_pref_weight is not None:
            out["ai_intero_pref_weight"] = self.ai_intero_pref_weight
        if self.drive_priority_gain is not None:
            out["drive_priority_gain"] = self.drive_priority_gain
        if self.motor_babble_sigma is not None:
            out["motor_babble_sigma"] = self.motor_babble_sigma
        return out


@dataclass
class SatisfierPolicy:
    """How (and how often) the 'parent' places food/water a step ahead.

    Placement uses the body command ``give_<resource>_near`` (the same path the
    dashboard 'Give' button uses): it drops an UNLABELED prop a step away, so the
    agent must perceive and walk to it - the self-learned act->relief loop stays
    intact. Disabled phases place nothing.
    """

    enabled: bool = False
    resources: tuple[str, ...] = ("food", "water")
    period_s: float = 20.0
    mode: str = "near"


@dataclass
class Phase:
    index: int
    name: str
    description: str
    config: PhaseConfig
    satisfier: SatisfierPolicy
    promote_criteria: list[Criterion]
    min_dwell_s: float = 30.0
    min_samples: int = 12
    demote_on_death: bool = True
    is_terminal: bool = False


def default_phases() -> list[Phase]:
    """Fresh copy of the default 4-phase curriculum (0 self-model -> 3 forage)."""
    return [
        Phase(
            index=0,
            name="Self-modeling",
            description=(
                "Welded body, no drive. The brain learns the proprioceptive and "
                "tactile forward models; the brace ratchet starts unlocking ROM as "
                "per-joint prediction error falls."
            ),
            config=PhaseConfig(
                viability_mode="immortal",
                metabolic_compression=1.0,
                motor_babble_sigma=0.15,
            ),
            satisfier=SatisfierPolicy(enabled=False),
            promote_criteria=[
                Criterion("forward_model_error", "<=", 0.05, "World-model PE low"),
                Criterion("tactile_pred_error", "<=", 0.05, "Tactile PE low"),
                Criterion("rom_mean", ">=", 0.05, "ROM unlocking"),
            ],
            min_dwell_s=30.0,
            min_samples=12,
            demote_on_death=False,
        ),
        Phase(
            index=1,
            name="Postural control",
            description=(
                "Gentle metabolism switches on. With partly-free joints the body "
                "must hold itself up; the world model learns balance dynamics while "
                "ROM keeps widening and falls stay rare."
            ),
            config=PhaseConfig(
                viability_mode="metabolic",
                metabolic_compression=1.0,
            ),
            satisfier=SatisfierPolicy(enabled=False),
            promote_criteria=[
                Criterion("fall_rate", "<=", 0.1, "Stays upright"),
                Criterion("rom_mean", ">=", 0.25, "ROM earned"),
                Criterion("forward_model_error", "<=", 0.06, "World-model PE low"),
            ],
            min_dwell_s=45.0,
            min_samples=15,
            demote_on_death=True,
        ),
        Phase(
            index=2,
            name="Locomotion onset",
            description=(
                "Drive rises and a satisfier is placed a step ahead. The drive-"
                "reduction pull plus the displaced satisfier make a step the "
                "predicted path to relief; the agent reaches and consumes."
            ),
            config=PhaseConfig(
                viability_mode="metabolic",
                metabolic_compression=2.0,
                drive_priority_gain=3.0,
                motor_babble_sigma=0.25,
            ),
            satisfier=SatisfierPolicy(
                enabled=True, resources=("food", "water"), period_s=20.0
            ),
            promote_criteria=[
                Criterion("consume_events", "trend>=", 1.0, "Reaches & consumes"),
                Criterion("viability", "trend>=", 0.0, "Net non-negative viability"),
                Criterion("distance_traveled", "trend>=", 0.5, "Moves toward goal", "m"),
            ],
            min_dwell_s=60.0,
            min_samples=15,
            demote_on_death=True,
        ),
        Phase(
            index=3,
            name="Sustained gait / forage",
            description=(
                "Satisfiers come slower (effectively farther), so the agent must "
                "walk multiple steps to survive. Sustained foraging, rising travel, "
                "and a regular left/right gait are the graduation signs."
            ),
            config=PhaseConfig(
                viability_mode="metabolic",
                metabolic_compression=2.0,
            ),
            satisfier=SatisfierPolicy(
                enabled=True, resources=("food", "water"), period_s=60.0
            ),
            promote_criteria=[
                Criterion("consume_events", "trend>=", 3.0, "Sustained foraging"),
                Criterion("viability", ">=", 50.0, "Surviving comfortably"),
                Criterion("distance_traveled", "trend>=", 2.0, "Walks to forage", "m"),
                Criterion("gait_regularity", ">=", 0.3, "Regular gait"),
            ],
            min_dwell_s=120.0,
            min_samples=20,
            demote_on_death=True,
            is_terminal=True,
        ),
    ]


def affective_phase() -> Phase:
    """Optional stretch phase 4: a woken threat forces urgent locomotion.

    Requires a bear in the scene (and a body 'wake bear' toggle). Appended only
    when the operator opts in, so the default run terminates at phase 3.
    """
    return Phase(
        index=4,
        name="Affective gait",
        description=(
            "A threat is woken: the agent must flee (urgent locomotion away) and "
            "still forage. Distinct gait under threat vs. foraging is the goal."
        ),
        config=PhaseConfig(
            viability_mode="metabolic",
            metabolic_compression=2.0,
            drive_priority_gain=4.0,
            motor_babble_sigma=0.25,
        ),
        satisfier=SatisfierPolicy(enabled=True, resources=("food", "water"), period_s=45.0),
        promote_criteria=[
            Criterion("distance_traveled", "trend>=", 3.0, "Flees / forages", "m"),
            Criterion("viability", ">=", 40.0, "Survives the threat"),
        ],
        min_dwell_s=120.0,
        min_samples=20,
        demote_on_death=True,
        is_terminal=True,
    )


def _apply_overrides(phase: Phase, ov: dict[str, Any]) -> None:
    """Shallow-merge a JSON override block onto a phase (config + scalars only)."""
    cfg = ov.get("config")
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if hasattr(phase.config, k):
                setattr(phase.config, k, v)
    sat = ov.get("satisfier")
    if isinstance(sat, dict):
        for k, v in sat.items():
            if hasattr(phase.satisfier, k):
                setattr(phase.satisfier, k, tuple(v) if k == "resources" else v)
    for k in ("min_dwell_s", "min_samples", "demote_on_death"):
        if k in ov:
            setattr(phase, k, ov[k])


def build_phases(
    *, include_affective: bool = False, overrides: dict[str, Any] | None = None
) -> list[Phase]:
    """Default phase list, optionally with the affective stretch phase + JSON tuning.

    ``overrides`` maps a phase name (case-insensitive) to a block of
    ``{config, satisfier, min_dwell_s, ...}`` values that are shallow-merged onto
    the defaults, so thresholds/knobs can be tuned without code edits.
    """
    phases = default_phases()
    if include_affective:
        phases[-1].is_terminal = False
        phases.append(affective_phase())
    if overrides:
        by_name = {p.name.lower(): p for p in phases}
        for name, ov in overrides.items():
            target = by_name.get(str(name).lower())
            if target is not None and isinstance(ov, dict):
                _apply_overrides(target, ov)
    return phases


# Fixed-scene elements the curriculum runs in (calm homeostasis world: shelter +
# food + water). The body composes this once; phases never change the scene.
CURRICULUM_SCENE: tuple[str, ...] = ("house", "food", "water")
