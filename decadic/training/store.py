"""Persistent uploaded Skill Dojo skill store."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from decadic.training.gates import Criterion
from decadic.training.skills import SKILLS
from decadic.training.teachers import TEACHERS
from decadic.training.types import (
    PeriodicBodyCommand,
    SkillGate,
    SkillPhase,
    SkillSpec,
    TeacherAdaptation,
)

VALID_COMPARATORS = {"<=", ">=", "trend>="}
VALID_CONFIG_KEYS = {
    "viability_mode",
    "metabolic_compression",
    "ai_intero_pref_weight",
    "drive_priority_gain",
    "motor_babble_sigma",
}
BODY_COMMAND_RE = re.compile(
    r"^(recenter|perturb:small|set_stance:[a-zA-Z0-9_]+)$"
)
PERIODIC_BODY_COMMAND_RE = re.compile(r"^(perturb:small|give_food_near|give_water_near)$")
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
LEGACY_SCAFFOLD_COMMANDS = {"braces_on", "braces_off", "hold_on", "hold_off", "reset_braces"}
PACKAGED_SKILL_DIR = Path(__file__).resolve().parents[2] / "docs" / "dojo_skills"


class SkillValidationError(ValueError):
    """Uploaded skill does not match the allowed SkillSpec subset."""


class UploadedSkillStore:
    """Filesystem-backed uploaded skill library."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                spec = self._load_path(path)
            except Exception:
                continue
            out.append(spec.as_dict(source="uploaded"))
        return out

    def get(self, skill_id: str) -> SkillSpec | None:
        sid = _sid(skill_id)
        path = self._path(sid)
        if not path.is_file():
            return None
        return self._load_path(path)

    def get_any(self, skill_id: str) -> SkillSpec | None:
        sid = _sid(skill_id)
        return SKILLS.get(sid) or _packaged_skill(sid) or self.get(sid)

    def get_any_dict(self, skill_id: str) -> dict[str, Any] | None:
        sid = _sid(skill_id)
        builtin = SKILLS.get(sid)
        if builtin is not None:
            return builtin.as_dict(source="builtin")
        packaged = _packaged_skill(sid)
        if packaged is not None:
            return packaged.as_dict(source="builtin")
        uploaded = self.get(sid)
        return uploaded.as_dict(source="uploaded") if uploaded is not None else None

    def list_all(self) -> list[dict[str, Any]]:
        packaged = _packaged_skills()
        protected = set(SKILLS) | {s.skill_id for s in packaged}
        uploaded = [s for s in self.list() if str(s.get("skill_id", "")) not in protected]
        return (
            [s.as_dict(source="builtin") for s in SKILLS.values()]
            + [s.as_dict(source="builtin") for s in packaged]
            + uploaded
        )

    def save(self, raw: dict[str, Any]) -> dict[str, Any]:
        spec = parse_skill(raw)
        if spec.skill_id in SKILLS or _packaged_skill(spec.skill_id) is not None:
            raise SkillValidationError(f"{spec.skill_id!r} is a built-in skill and cannot be replaced")
        path = self._path(spec.skill_id)
        path.write_text(json.dumps(spec.upload_dict(), indent=2), encoding="utf-8")
        return spec.as_dict(source="uploaded")

    def delete(self, skill_id: str) -> bool:
        sid = _sid(skill_id)
        if sid in SKILLS or _packaged_skill(sid) is not None:
            raise SkillValidationError("built-in skills cannot be deleted")
        path = self._path(sid)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path(self, skill_id: str) -> Path:
        return self.base_dir / f"{_sid(skill_id)}.json"

    def _load_path(self, path: Path) -> SkillSpec:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SkillValidationError("skill file must contain a JSON object")
        migrated, removed = _strip_legacy_scaffold_commands(raw)
        if removed:
            warning = (
                "Removed legacy manual scaffold command(s): "
                + ", ".join(sorted(set(removed)))
            )
            warnings = list(migrated.get("warnings") or [])
            if warning not in warnings:
                warnings.append(warning)
            migrated["warnings"] = warnings
            path.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
            raw = migrated
        return parse_skill(raw)


def parse_skill(raw: dict[str, Any]) -> SkillSpec:
    sid = _required_str(raw, "skill_id").strip().lower()
    if not SKILL_ID_RE.match(sid):
        raise SkillValidationError("skill_id must be lowercase snake_case, 3-64 chars")
    teacher = _required_str(raw, "teacher")
    if teacher not in TEACHERS:
        raise SkillValidationError(f"unknown teacher {teacher!r}; choose one of {sorted(TEACHERS)}")
    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        raise SkillValidationError("phases must be a non-empty list")
    phases: list[SkillPhase] = []
    for expected, item in enumerate(phases_raw):
        if not isinstance(item, dict):
            raise SkillValidationError("each phase must be an object")
        idx = int(item.get("index", expected))
        if idx != expected:
            raise SkillValidationError("phase indexes must be contiguous starting at 0")
        config_raw = item.get("config") or {}
        if not isinstance(config_raw, dict):
            raise SkillValidationError("phase.config must be an object")
        config = {str(k): v for k, v in config_raw.items() if str(k) in VALID_CONFIG_KEYS}
        bad_cfg = sorted(str(k) for k in config_raw if str(k) not in VALID_CONFIG_KEYS)
        if bad_cfg:
            raise SkillValidationError(f"unsupported config key(s): {', '.join(bad_cfg)}")
        commands = item.get("body_commands") or []
        if not isinstance(commands, list):
            raise SkillValidationError("phase.body_commands must be a list")
        body_commands = tuple(_validate_command(str(c)) for c in commands)
        periodic_raw = item.get("periodic_body_commands") or []
        if not isinstance(periodic_raw, list):
            raise SkillValidationError("phase.periodic_body_commands must be a list")
        periodic_body_commands = tuple(_periodic_command(c) for c in periodic_raw)
        gate_raw = item.get("gate") or {}
        if not isinstance(gate_raw, dict):
            raise SkillValidationError("phase.gate must be an object")
        criteria_raw = gate_raw.get("criteria", item.get("criteria", []))
        if not isinstance(criteria_raw, list):
            raise SkillValidationError("gate.criteria must be a list")
        criteria = tuple(_criterion(c) for c in criteria_raw)
        failure_raw = item.get("failure_criteria", gate_raw.get("failure_criteria", []))
        if not isinstance(failure_raw, list):
            raise SkillValidationError("failure_criteria must be a list")
        failure_criteria = tuple(_criterion(c) for c in failure_raw)
        reset_raw = item.get("reset_commands", item.get("body_commands", []))
        if not isinstance(reset_raw, list):
            raise SkillValidationError("phase.reset_commands must be a list")
        reset_commands = tuple(_validate_command(str(c)) for c in reset_raw)
        teacher_weight = max(0.0, float(item.get("teacher_weight", 0.0)))
        teacher_adaptation = _teacher_adaptation(
            item.get("teacher_adaptation"),
            teacher_weight=teacher_weight,
            is_terminal=bool(item.get("is_terminal", False)),
        )
        phases.append(
            SkillPhase(
                index=idx,
                name=_required_str(item, "name"),
                description=str(item.get("description", "")),
                teacher_weight=teacher_weight,
                config=config,
                body_commands=body_commands,
                periodic_body_commands=periodic_body_commands,
                gate=SkillGate(criteria=criteria, min_samples=max(1, int(gate_raw.get("min_samples", item.get("min_samples", 8))))),
                failure_gate=SkillGate(
                    criteria=failure_criteria,
                    min_samples=max(1, int(item.get("failure_min_samples", gate_raw.get("failure_min_samples", 1)))),
                ),
                reset_commands=reset_commands,
                timeout_s=max(0.0, float(item.get("timeout_s", 0.0))),
                max_attempts=max(1, int(item.get("max_attempts", 3))),
                auto_retry=bool(item.get("auto_retry", True)),
                min_dwell_s=max(0.0, float(item.get("min_dwell_s", 20.0))),
                demote_on_death=bool(item.get("demote_on_death", False)),
                is_terminal=bool(item.get("is_terminal", False)),
                teacher_adaptation=teacher_adaptation,
            )
        )
    warnings_raw = raw.get("warnings", [])
    warnings = warnings_raw if isinstance(warnings_raw, list) else []
    return SkillSpec(
        skill_id=sid,
        version=str(raw.get("version", "1.0")),
        name=_required_str(raw, "name"),
        description=str(raw.get("description", "")),
        target_behavior=str(raw.get("target_behavior", "")),
        teacher=teacher,
        required_sensors=tuple(str(x) for x in raw.get("required_sensors", ["proprioception", "contacts"])),
        checkpoint_on_graduate=bool(raw.get("checkpoint_on_graduate", True)),
        warnings=tuple(str(x) for x in warnings if str(x).strip()),
        phases=tuple(phases),
    )


def _strip_legacy_scaffold_commands(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a copy with old brace/hold commands removed from saved skills."""
    removed: list[str] = []
    out = json.loads(json.dumps(raw))
    phases = out.get("phases")
    if not isinstance(phases, list):
        return out, removed
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        for key in ("body_commands", "reset_commands"):
            commands = phase.get(key)
            if not isinstance(commands, list):
                continue
            kept: list[Any] = []
            for cmd in commands:
                if str(cmd) in LEGACY_SCAFFOLD_COMMANDS:
                    removed.append(str(cmd))
                else:
                    kept.append(cmd)
            phase[key] = kept
        periodic = phase.get("periodic_body_commands")
        if isinstance(periodic, list):
            kept_periodic: list[Any] = []
            for item in periodic:
                command = item.get("command") if isinstance(item, dict) else None
                if str(command) in LEGACY_SCAFFOLD_COMMANDS:
                    removed.append(str(command))
                else:
                    kept_periodic.append(item)
            phase["periodic_body_commands"] = kept_periodic
    return out, removed


def _packaged_skills() -> list[SkillSpec]:
    out: list[SkillSpec] = []
    if not PACKAGED_SKILL_DIR.is_dir():
        return out
    for path in sorted(PACKAGED_SKILL_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out.append(parse_skill(raw))
        except Exception:
            continue
    return out


def _packaged_skill(skill_id: str) -> SkillSpec | None:
    sid = _sid(skill_id)
    path = PACKAGED_SKILL_DIR / f"{sid}.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return parse_skill(raw) if isinstance(raw, dict) else None
    except Exception:
        return None


def _criterion(raw: Any) -> Criterion:
    if not isinstance(raw, dict):
        raise SkillValidationError("criterion must be an object")
    comp = str(raw.get("comparator", ""))
    if comp not in VALID_COMPARATORS:
        raise SkillValidationError(f"invalid comparator {comp!r}")
    return Criterion(
        _required_str(raw, "key"),
        comp,  # type: ignore[arg-type]
        float(raw.get("threshold", 0.0)),
        _required_str(raw, "label"),
        str(raw.get("unit", "")),
    )


def _teacher_adaptation(raw: Any, *, teacher_weight: float, is_terminal: bool) -> TeacherAdaptation:
    if raw is None:
        return TeacherAdaptation(
            enabled=not is_terminal and teacher_weight > 0.0,
            min_weight=0.0,
            max_weight=0.0 if is_terminal else max(0.0, teacher_weight),
        )
    if not isinstance(raw, dict):
        raise SkillValidationError("phase.teacher_adaptation must be an object")
    danger = raw.get("danger_thresholds", {})
    stable = raw.get("stability_thresholds", {})
    if not isinstance(danger, dict):
        raise SkillValidationError("teacher_adaptation.danger_thresholds must be an object")
    if not isinstance(stable, dict):
        raise SkillValidationError("teacher_adaptation.stability_thresholds must be an object")
    max_weight = 0.0 if is_terminal else _clamp01(float(raw.get("max_weight", teacher_weight)))
    min_weight = min(max_weight, _clamp01(float(raw.get("min_weight", 0.0))))
    return TeacherAdaptation(
        enabled=bool(raw.get("enabled", not is_terminal and max_weight > 0.0)) and not is_terminal,
        min_weight=min_weight,
        max_weight=max_weight,
        rise_rate=max(0.0, float(raw.get("rise_rate", 0.5))),
        fade_rate=max(0.0, float(raw.get("fade_rate", 0.08))),
        danger_thresholds=_float_dict(danger),
        stability_thresholds=_float_dict(stable),
        stable_dwell_s=max(0.0, float(raw.get("stable_dwell_s", 3.0))),
        unstable_dwell_s=max(0.0, float(raw.get("unstable_dwell_s", 0.0))),
        zero_required_for_graduation=bool(raw.get("zero_required_for_graduation", True)),
    )


def _float_dict(raw: dict[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise SkillValidationError(f"teacher threshold {key!r} must be numeric") from exc
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_command(cmd: str) -> str:
    if not BODY_COMMAND_RE.match(cmd):
        raise SkillValidationError(f"unsupported body command {cmd!r}")
    return cmd


def _periodic_command(raw: Any) -> PeriodicBodyCommand:
    if not isinstance(raw, dict):
        raise SkillValidationError("periodic body command must be an object")
    command = str(raw.get("command", "")).strip()
    if not PERIODIC_BODY_COMMAND_RE.match(command):
        raise SkillValidationError(f"unsupported periodic body command {command!r}")
    period_s = max(0.25, float(raw.get("period_s", 5.0)))
    return PeriodicBodyCommand(command=command, period_s=period_s)


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillValidationError(f"{key} is required")
    return value.strip()


def _sid(skill_id: str) -> str:
    return str(skill_id).strip().lower()
