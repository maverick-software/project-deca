"""Skill Dojo supervisor: generalized skill curricula around the Decadic loop."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from decadic.training.gates import CriterionResult, GateResult, evaluate_criterion, evaluate_gate
from decadic.training.skills import get_skill
from decadic.training.teachers import get_teacher
from decadic.training.types import DemoRecord, SkillSpec, TeacherAdaptation

if TYPE_CHECKING:
    from decadic.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

WINDOW_MAX = 240
DEFAULT_POLL_S = 1.0
DEFAULT_CAREGIVER_THRESHOLD = 80.0
DEFAULT_CAREGIVER_REFRACTORY_S = 15.0

SAMPLE_KEYS = (
    "viability",
    "hydration",
    "energy",
    "integrity",
    "forward_model_error",
    "tactile_pred_error",
    "fall_rate",
    "rom_mean",
    "brace_engaged",
    "root_height",
    "torso_tilt",
    "stance_phase",
    "movement_hold",
    "braces_enabled",
    "foot_load_l",
    "foot_load_r",
    "hand_load_l",
    "hand_load_r",
    "motor_activity_rms",
    "teacher_motor_agreement",
    "teacher_support_active",
    "teacher_support_force",
    "teacher_support_torque",
    "teacher_drop_m",
    "teacher_target_drop_m",
    "teacher_height_error_m",
    "teacher_vertical_velocity",
    "teacher_override_fraction",
    "consume_events",
    "distance_traveled",
    "net_displacement",
    "gait_regularity",
    "object_files",
    "centroid_spread",
    "stable_tracked_objects",
    "perception_collapsed",
    "ltm_write_accepted",
    "flow_confidence",
    "looming_count",
    "stuff_count",
    "body_candidate_count",
    "scene_dynamics_error",
    "scene_dynamics_unstable",
    "scene_dynamics_matches",
    "caregiver_parent_present",
    "caregiver_missing_parent",
    "caregiver_delivery_count",
    "caregiver_kind",
)


class SkillDojoError(RuntimeError):
    """Invalid dojo action."""


class SkillDojoSupervisor:
    """Runs one named skill curriculum for one agent."""

    def __init__(
        self,
        registry: "AgentRegistry",
        *,
        backups_dir: Path,
        log_dir: Path | None = None,
        skill_loader: Callable[[str], SkillSpec | None] | None = None,
    ) -> None:
        self._registry = registry
        self._backups_dir = Path(backups_dir)
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._skill_loader = skill_loader or get_skill
        self._poll_s = max(0.01, float(os.environ.get("DECADIC_DOJO_POLL_S", str(DEFAULT_POLL_S))))
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._reset()

    def _reset(self) -> None:
        self._state = "stopped"
        self._agent_id: str | None = None
        self._skill: SkillSpec | None = None
        self._phase_idx = 0
        self._phase_started = 0.0
        self._started_at = 0.0
        self._window: deque[dict[str, float]] = deque(maxlen=WINDOW_MAX)
        self._last_gate: GateResult | None = None
        self._last_failure: CriterionResult | None = None
        self._attempt_index = 1
        self._attempt_started = 0.0
        self._attempt_failures = 0
        self._last_attempt_outcome: str | None = None
        self._failure_reason: str | None = None
        self._paused_started = 0.0
        self._auto_retry_override: bool | None = None
        self._max_attempts_override: int | None = None
        self._timeout_multiplier = 1.0
        self._history: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._error: str | None = None
        self._report_path: Path | None = None
        self._last_body_cmd: dict[str, float] = {}
        self._manual_scaffold_active = False
        self._teacher_assist = 0.0
        self._teacher_min = 0.0
        self._teacher_max = 0.0
        self._teacher_rise_rate = 0.0
        self._teacher_fade_rate = 0.0
        self._teacher_stable_dwell_s = 0.0
        self._teacher_unstable_dwell_s = 0.0
        self._teacher_last_update = 0.0
        self._assist_reason = "idle"
        self._teacher_origin = "self"
        self._objective_confidence = 0.0
        self._confidence_reason = "idle"
        self._confidence_dwell_s = 0.0
        self._caregiver_enabled = False
        self._caregiver_threshold = DEFAULT_CAREGIVER_THRESHOLD
        self._caregiver_refractory_s = max(
            0.0,
            float(
                os.environ.get(
                    "DECADIC_DOJO_CAREGIVER_REFRACTORY_S",
                    str(DEFAULT_CAREGIVER_REFRACTORY_S),
                )
            ),
        )
        self._caregiver_last_request_at = 0.0
        self._caregiver_last_offer_cycle: int | None = None
        self._caregiver_last_offer_item: str | None = None
        self._caregiver_delivery_count = 0
        self._caregiver_status = "disabled"
        self._caregiver_need = "none"
        self._caregiver_trigger_reservoir: str | None = None
        self._caregiver_request_kind: str | None = None
        self._caregiver_missing_parent = False
        self._caregiver_pending = False

    async def start(
        self,
        agent_id: str,
        skill_id: str,
        *,
        auto_retry: bool | None = None,
        max_attempts: int | None = None,
        timeout_multiplier: float | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            if self.is_running():
                raise SkillDojoError("A skill dojo run is already active; stop it first.")
            agent = self._registry.get(agent_id)
            if agent is None:
                raise SkillDojoError(f"Unknown agent {agent_id!r}")
            skill = self._skill_loader(skill_id)
            if skill is None:
                raise SkillDojoError(f"Unknown skill {skill_id!r}")
            self._reset()
            self._agent_id = agent_id
            self._skill = skill
            self._caregiver_enabled = bool(getattr(skill, "caregiver_enabled", False))
            self._caregiver_threshold = max(
                1.0,
                min(
                    100.0,
                    float(getattr(skill, "caregiver_threshold", DEFAULT_CAREGIVER_THRESHOLD) or DEFAULT_CAREGIVER_THRESHOLD),
                ),
            )
            self._state = "running"
            self._started_at = time.time()
            self._auto_retry_override = auto_retry
            self._max_attempts_override = max(1, int(max_attempts)) if max_attempts is not None else None
            self._timeout_multiplier = max(0.1, float(timeout_multiplier)) if timeout_multiplier is not None else 1.0
            self._begin_attempt()
            self._open_report_path()
            await self._apply_phase(reason="start")
            self._task = asyncio.create_task(self._loop(), name=f"skill-dojo-{agent_id}")
            logger.info("skill_dojo_start agent_id=%s skill_id=%s", agent_id, skill.skill_id)
            return self.status()

    def pause(self) -> dict[str, Any]:
        if not self.is_running():
            raise SkillDojoError("No running skill dojo to pause.")
        self._state = "paused"
        self._paused_started = time.monotonic()
        self._clear_agent_training()
        return self.status()

    async def resume(self) -> dict[str, Any]:
        if self._task is None or self._task.done():
            raise SkillDojoError("No skill dojo to resume.")
        if self._paused_started:
            paused_for = max(0.0, time.monotonic() - self._paused_started)
            self._attempt_started += paused_for
            self._phase_started += paused_for
            self._paused_started = 0.0
        self._state = "running"
        await self._apply_phase(reason="resume")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            task = self._task
            self._task = None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._clear_agent_training()
            self._write_report(event="stopped")
            status = self.status()
            self._state = "stopped"
            return {**status, "state": "stopped"}

    async def set_phase(self, index: int) -> dict[str, Any]:
        async with self._lock:
            if self._skill is None:
                raise SkillDojoError("No skill dojo run is active.")
            self._phase_idx = max(0, min(len(self._skill.phases) - 1, int(index)))
            self._window.clear()
            self._last_body_cmd.clear()
            self._reset_phase_attempts()
            self._begin_attempt()
            await self._apply_phase(reason="manual")
            return self.status()

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        phase = self._phase()
        now = time.monotonic()
        timeout_s = self._phase_timeout_s(phase) if phase else 0.0
        elapsed = max(0.0, now - self._attempt_started) if self._attempt_started else 0.0
        last = self._window[-1] if self._window else {}
        return {
            "state": self._state,
            "running": self.is_running() and self._state in ("running", "paused"),
            "paused": self._state == "paused",
            "agent_id": self._agent_id,
            "skill_id": self._skill.skill_id if self._skill else None,
            "skill_name": self._skill.name if self._skill else None,
            "phase_index": phase.index if phase else None,
            "phase_name": phase.name if phase else None,
            "phase_description": phase.description if phase else None,
            "phase_count": len(self._skill.phases) if self._skill else 0,
            "teacher_weight": float(phase.teacher_weight) if phase else 0.0,
            "teacher_assist": round(self._teacher_assist, 4),
            "teacher_min": round(self._teacher_min, 4),
            "teacher_max": round(self._teacher_max, 4),
            "teacher_rise_rate": round(self._teacher_rise_rate, 4),
            "teacher_fade_rate": round(self._teacher_fade_rate, 4),
            "stable_dwell_s": round(self._teacher_stable_dwell_s, 3),
            "unstable_dwell_s": round(self._teacher_unstable_dwell_s, 3),
            "assist_reason": self._assist_reason,
            "teacher_origin": self._teacher_origin,
            "objective_confidence": round(self._objective_confidence, 4),
            "confidence_reason": self._confidence_reason,
            "confidence_dwell_s": round(self._confidence_dwell_s, 3),
            "teacher_live": bool(self._teacher_assist > 0.0 and phase is not None and not getattr(phase, "is_terminal", False)),
            "teacher_support_active": bool(float(last.get("teacher_support_active", 0.0) or 0.0) >= 0.5),
            "teacher_support_force": round(float(last.get("teacher_support_force", 0.0) or 0.0), 4),
            "teacher_support_torque": round(float(last.get("teacher_support_torque", 0.0) or 0.0), 4),
            "teacher_drop_m": round(float(last.get("teacher_drop_m", 0.0) or 0.0), 4),
            "teacher_target_drop_m": round(float(last.get("teacher_target_drop_m", 0.25) or 0.25), 4),
            "teacher_height_error_m": round(float(last.get("teacher_height_error_m", 0.0) or 0.0), 4),
            "teacher_vertical_velocity": round(float(last.get("teacher_vertical_velocity", 0.0) or 0.0), 4),
            "teacher_support_mode": str(last.get("teacher_support_mode", "off") or "off"),
            "caregiver_enabled": bool(self._caregiver_enabled),
            "caregiver_status": self._caregiver_status,
            "caregiver_kind": str(last.get("caregiver_kind", "") or ""),
            "caregiver_need": self._caregiver_need,
            "caregiver_threshold": round(float(self._caregiver_threshold), 3),
            "caregiver_trigger_reservoir": self._caregiver_trigger_reservoir,
            "caregiver_request_kind": self._caregiver_request_kind,
            "caregiver_last_offer_cycle": self._caregiver_last_offer_cycle,
            "caregiver_last_offer_item": self._caregiver_last_offer_item,
            "caregiver_missing_parent": bool(self._caregiver_missing_parent),
            "caregiver_refractory_s": round(float(self._caregiver_refractory_s), 3),
            "caregiver_delivery_count": int(self._caregiver_delivery_count),
            "caregiver_pending": bool(self._caregiver_pending),
            "hydration": round(float(last.get("hydration", 100.0) or 100.0), 4),
            "energy": round(float(last.get("energy", 100.0) or 100.0), 4),
            "integrity": round(float(last.get("integrity", 100.0) or 100.0), 4),
            "samples": len(self._window),
            "gate": self._last_gate.as_dict() if self._last_gate else None,
            "failure": self._last_failure.as_dict() if self._last_failure else None,
            "attempt_index": self._attempt_index,
            "attempt_failures": self._attempt_failures,
            "attempt_elapsed_s": round(elapsed, 3),
            "attempt_timeout_s": timeout_s,
            "max_attempts": self._phase_max_attempts(phase) if phase else 0,
            "auto_retry": self._phase_auto_retry(phase) if phase else False,
            "last_attempt_outcome": self._last_attempt_outcome,
            "failure_reason": self._failure_reason,
            "manual_scaffold_active": self._manual_scaffold_active,
            "history": list(self._history[-20:]),
            "report_path": str(self._report_path) if self._report_path else None,
            "started_at": self._started_at or None,
            "poll_interval_s": self._poll_s,
            "error": self._error,
        }

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_s)
                if self._state != "running":
                    continue
                agent = self._agent()
                if agent is None:
                    self._state = "error"
                    self._error = "bound agent disappeared"
                    self._clear_agent_training()
                    return
                phase = self._phase()
                if phase is None:
                    continue
                sample = await self._sample(agent)
                self._window.append(sample)
                self._manual_scaffold_active = self._sample_manual_scaffold_active(sample)
                self._update_caregiver(agent, sample)
                self._update_teacher_assist(agent, phase, sample)
                self._record_sample(sample)
                if await self._handle_death(agent, phase):
                    continue
                await self._periodic_body_commands(agent)
                failure = self._evaluate_failure(phase)
                if failure is not None:
                    self._last_failure = failure
                    if await self._close_attempt(agent, "failed", failure.label):
                        continue
                    return
                if self._attempt_timed_out(phase):
                    if await self._close_attempt(agent, "timeout", "phase attempt timeout"):
                        continue
                    return
                gate = evaluate_gate(
                    list(phase.gate.criteria), list(self._window), min_samples=phase.gate.min_samples
                )
                self._last_gate = gate
                dwell_ok = (time.monotonic() - self._phase_started) >= phase.min_dwell_s
                teacher_ok = self._teacher_allows_success(phase)
                support_ok = float(sample.get("teacher_support_active", 0.0) or 0.0) < 0.5
                caregiver_ok = self._caregiver_allows_success(phase)
                confidence_ok = self._objective_confidence >= 1.0
                if (
                    gate.satisfied
                    and dwell_ok
                    and confidence_ok
                    and support_ok
                    and caregiver_ok
                    and not self._manual_scaffold_active
                    and teacher_ok
                ):
                    self._last_attempt_outcome = "success"
                    self._failure_reason = None
                    self._history.append(
                        {
                            "phase": phase.index,
                            "name": phase.name,
                            "event": "attempt_success",
                            "attempt": self._attempt_index,
                            "at": _utc(),
                        }
                    )
                    if phase.is_terminal:
                        await self._graduate(agent)
                        return
                    else:
                        await self._promote()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("skill_dojo_loop_failed agent_id=%s", self._agent_id)
            self._state = "error"
            self._error = "loop exception (see server log)"
            self._clear_agent_training()

    async def _promote(self) -> None:
        phase = self._phase()
        if phase is not None:
            self._history.append({"phase": phase.index, "name": phase.name, "event": "promoted", "at": _utc()})
        assert self._skill is not None
        self._phase_idx = min(self._phase_idx + 1, len(self._skill.phases) - 1)
        self._window.clear()
        self._last_body_cmd.clear()
        self._last_failure = None
        self._reset_phase_attempts()
        self._begin_attempt()
        await self._apply_phase(reason="promote")

    async def _graduate(self, agent: Any) -> None:
        self._state = "graduated"
        self._clear_agent_training()
        phase = self._phase()
        if phase is not None:
            self._history.append({"phase": phase.index, "name": phase.name, "event": "graduated", "at": _utc()})
        if self._skill and self._skill.checkpoint_on_graduate:
            await self._checkpoint(agent, "graduated")
        self._write_report(event="graduated")

    async def _apply_phase(self, *, reason: str) -> None:
        agent = self._agent()
        phase = self._phase()
        if agent is None or phase is None or self._skill is None:
            return
        if phase.config:
            async with agent.lock:
                agent.configure(**phase.config)
        self._init_teacher_for_phase(phase)
        self._write_teacher_training(agent, phase, {})
        for cmd in phase.body_commands:
            agent.queue_body_command(cmd)
            self._last_body_cmd[cmd] = time.monotonic()
        for cmd in phase.periodic_body_commands:
            self._last_body_cmd[cmd.command] = 0.0
        logger.info(
            "skill_dojo_phase agent_id=%s skill_id=%s phase=%s reason=%s teacher_weight=%.3f",
            self._agent_id,
            self._skill.skill_id,
            phase.index,
            reason,
            phase.teacher_weight,
        )

    def _reset_phase_attempts(self) -> None:
        self._attempt_index = 1
        self._attempt_failures = 0
        self._last_attempt_outcome = None
        self._failure_reason = None

    def _begin_attempt(self) -> None:
        now = time.monotonic()
        self._phase_started = now
        self._attempt_started = now
        self._last_gate = None
        self._last_failure = None
        self._manual_scaffold_active = False
        self._teacher_stable_dwell_s = 0.0
        self._teacher_unstable_dwell_s = 0.0
        self._teacher_last_update = now
        self._objective_confidence = 0.0
        self._confidence_reason = "attempt start"
        self._confidence_dwell_s = 0.0

    def _phase_timeout_s(self, phase: Any) -> float:
        if phase is None:
            return 0.0
        timeout = float(getattr(phase, "timeout_s", 0.0) or 0.0)
        return round(timeout * self._timeout_multiplier, 3) if timeout > 0 else 0.0

    def _phase_max_attempts(self, phase: Any) -> int:
        if phase is None:
            return 0
        if self._max_attempts_override is not None:
            return self._max_attempts_override
        return max(1, int(getattr(phase, "max_attempts", 3) or 3))

    def _phase_auto_retry(self, phase: Any) -> bool:
        if phase is None:
            return False
        if self._auto_retry_override is not None:
            return bool(self._auto_retry_override)
        return bool(getattr(phase, "auto_retry", True))

    def _attempt_timed_out(self, phase: Any) -> bool:
        timeout = self._phase_timeout_s(phase)
        return bool(timeout > 0 and (time.monotonic() - self._attempt_started) >= timeout)

    def _evaluate_failure(self, phase: Any) -> CriterionResult | None:
        criteria = list(getattr(getattr(phase, "failure_gate", None), "criteria", ()) or ())
        if not criteria:
            return None
        min_samples = max(1, int(getattr(getattr(phase, "failure_gate", None), "min_samples", 1) or 1))
        if len(self._window) < min_samples:
            return None
        window = list(self._window)
        for criterion in criteria:
            result = evaluate_criterion(criterion, window)
            if result.satisfied:
                return result
        return None

    async def _close_attempt(self, agent: Any, outcome: str, reason: str) -> bool:
        phase = self._phase()
        if phase is None:
            return False
        self._attempt_failures += 1
        self._last_attempt_outcome = outcome
        self._failure_reason = reason
        self._history.append(
            {
                "phase": phase.index,
                "name": phase.name,
                "event": outcome,
                "reason": reason,
                "attempt": self._attempt_index,
                "failures": self._attempt_failures,
                "at": _utc(),
            }
        )
        if not self._phase_auto_retry(phase) or self._attempt_failures >= self._phase_max_attempts(phase):
            self._state = "failed"
            self._clear_agent_training()
            self._write_report(event="failed")
            return False
        self._attempt_index += 1
        self._window.clear()
        self._last_body_cmd.clear()
        self._begin_attempt()
        await self._reset_attempt_body(agent, phase)
        await self._apply_phase(reason=outcome)
        self._write_report(event=outcome)
        return True

    async def _reset_attempt_body(self, agent: Any, phase: Any) -> None:
        commands = tuple(getattr(phase, "reset_commands", ()) or getattr(phase, "body_commands", ()) or ())
        for cmd in commands:
            agent.queue_body_command(cmd)

    async def _periodic_body_commands(self, agent: Any) -> None:
        phase = self._phase()
        if phase is None:
            return
        now = time.monotonic()
        for cmd in phase.periodic_body_commands:
            if now - self._last_body_cmd.get(cmd.command, 0.0) >= cmd.period_s:
                if agent.queue_body_command(cmd.command):
                    self._last_body_cmd[cmd.command] = now

    def _update_caregiver(self, agent: Any, sample: dict[str, Any]) -> None:
        if not self._caregiver_enabled:
            self._caregiver_status = "disabled"
            self._caregiver_need = "none"
            self._caregiver_trigger_reservoir = None
            self._caregiver_request_kind = None
            self._caregiver_missing_parent = False
            self._caregiver_pending = False
            return

        delivery_count = int(float(sample.get("caregiver_delivery_count", self._caregiver_delivery_count) or 0.0))
        if delivery_count > self._caregiver_delivery_count:
            self._caregiver_delivery_count = delivery_count
            self._caregiver_last_offer_cycle = _agent_cycle(agent)
            item = str(sample.get("caregiver_last_offer_item", "") or "")
            self._caregiver_last_offer_item = item or self._caregiver_last_offer_item
            self._caregiver_pending = False
            self._caregiver_status = "delivered"

        parent_present = sample.get("caregiver_parent_present")
        explicit_missing = sample.get("caregiver_missing_parent")
        if parent_present is None and explicit_missing is not None and float(explicit_missing or 0.0) >= 0.5:
            parent_present = 0.0
        if parent_present is not None and float(parent_present or 0.0) < 0.5:
            self._caregiver_missing_parent = True
            self._caregiver_status = "caregiver_missing_parent"
            self._caregiver_pending = False
            return
        if parent_present is not None:
            self._caregiver_missing_parent = False

        reservoirs = {
            "hydration": float(sample.get("hydration", 100.0) or 100.0),
            "energy": float(sample.get("energy", 100.0) or 100.0),
            "integrity": float(sample.get("integrity", 100.0) or 100.0),
        }
        reservoir, value = min(reservoirs.items(), key=lambda kv: kv[1])
        if value >= self._caregiver_threshold:
            self._caregiver_need = "none"
            self._caregiver_trigger_reservoir = None
            self._caregiver_request_kind = None
            self._caregiver_pending = False
            if self._caregiver_status not in {"delivered", "caregiver_missing_parent"}:
                self._caregiver_status = "monitoring"
            return

        kind = {"hydration": "water", "energy": "food", "integrity": "care"}[reservoir]
        self._caregiver_need = reservoir
        self._caregiver_trigger_reservoir = reservoir
        self._caregiver_request_kind = kind
        body_status = str(sample.get("caregiver_status", "") or "")
        if self._caregiver_pending and body_status in {"requested", "delivering"}:
            self._caregiver_status = body_status
            return
        now = time.monotonic()
        if now - self._caregiver_last_request_at < self._caregiver_refractory_s:
            self._caregiver_status = "refractory"
            self._caregiver_pending = True
            return
        queued = bool(agent.queue_body_command("parent_enable"))
        queued = bool(agent.queue_body_command(f"parent_request:{kind}")) or queued
        if queued:
            self._caregiver_last_request_at = now
            self._caregiver_pending = True
            self._caregiver_status = "requested"
            self._history.append(
                {
                    "phase": self._phase_idx,
                    "name": self._phase().name if self._phase() else "",
                    "event": "caregiver_request",
                    "need": reservoir,
                    "request": kind,
                    "value": round(value, 4),
                    "at": _utc(),
                }
            )
        else:
            self._caregiver_status = "request_queue_failed"

    def _caregiver_allows_success(self, phase: Any) -> bool:
        if not self._caregiver_enabled:
            return True
        if self._caregiver_missing_parent:
            return False
        if not getattr(phase, "is_terminal", False):
            return True
        return not self._caregiver_pending and self._caregiver_need == "none"

    async def _handle_death(self, agent: Any, phase: Any) -> bool:
        if getattr(agent, "status", None) != "dead":
            return False
        self._history.append({"phase": phase.index, "name": phase.name, "event": "died", "at": _utc()})
        if hasattr(agent, "revive"):
            try:
                maybe = agent.revive()
                if hasattr(maybe, "__await__"):
                    await maybe
            except Exception:
                logger.exception("skill_dojo_revive_failed agent_id=%s", self._agent_id)
        if not bool(getattr(phase, "demote_on_death", False)):
            return await self._close_attempt(agent, "failed", "agent died")
        target = max(0, self._phase_idx - 1)
        if target != self._phase_idx:
            self._history.append({"phase": target, "name": self._skill.phases[target].name, "event": "demoted", "at": _utc()})
        self._phase_idx = target
        self._window.clear()
        self._last_body_cmd.clear()
        self._reset_phase_attempts()
        self._begin_attempt()
        await self._apply_phase(reason="death")
        return True

    async def _sample(self, agent: Any) -> dict[str, Any]:
        async with agent.lock:
            m = dict(agent.metrics)
            viability = float(getattr(agent.viability, "value", 0.0))
            perc = getattr(agent, "perceptual", None)
            health = getattr(perc, "discovery_health", None) if perc is not None else None
            ltm = getattr(perc, "ltm_consolidation", None) if perc is not None else None
            scene_prediction = getattr(perc, "scene_prediction", None) if perc is not None else None
        sample = {"viability": viability}
        for key in SAMPLE_KEYS:
            if key == "viability":
                continue
            value = m.get(key, 0.0)
            try:
                sample[key] = float(value)
            except (TypeError, ValueError):
                sample[key] = 0.0
        sample["teacher_support_mode"] = str(m.get("teacher_support_mode", "off") or "off")
        sample["caregiver_status"] = str(m.get("caregiver_status", "") or "")
        sample["caregiver_kind"] = str(m.get("caregiver_kind", "") or "")
        sample["caregiver_last_offer_item"] = str(m.get("caregiver_last_offer_item", "") or "")
        sample["caregiver_request_kind"] = str(m.get("caregiver_request_kind", "") or "")
        if "caregiver_parent_present" not in m:
            sample["caregiver_parent_present"] = None
        if "caregiver_missing_parent" not in m:
            sample["caregiver_missing_parent"] = None
        if isinstance(health, dict):
            sample["object_files"] = float(health.get("object_files", 0.0) or 0.0)
            sample["centroid_spread"] = float(health.get("centroid_spread", 0.0) or 0.0)
            sample["stable_tracked_objects"] = float(
                health.get("stable_tracked_objects", 0.0) or 0.0
            )
            sample["perception_collapsed"] = 1.0 if health.get("collapsed") else 0.0
            sample["flow_confidence"] = float(health.get("flow_confidence", 0.0) or 0.0)
            sample["looming_count"] = float(health.get("looming_count", 0.0) or 0.0)
            sample["stuff_count"] = float(health.get("stuff_count", 0.0) or 0.0)
            sample["body_candidate_count"] = float(health.get("body_candidate_count", 0.0) or 0.0)
        if isinstance(ltm, dict):
            sample["ltm_write_accepted"] = 1.0 if ltm.get("status") == "accepted" else 0.0
        if isinstance(scene_prediction, dict):
            sample["scene_dynamics_error"] = float(scene_prediction.get("error", 0.0) or 0.0)
            sample["scene_dynamics_unstable"] = float(scene_prediction.get("unstable_count", 0.0) or 0.0)
            sample["scene_dynamics_matches"] = float(scene_prediction.get("reidentified_count", 0.0) or 0.0)
        return sample

    def _sample_manual_scaffold_active(self, sample: dict[str, Any]) -> bool:
        return bool(
            float(sample.get("braces_enabled", 0.0) or 0.0) >= 0.5
            or float(sample.get("movement_hold", 0.0) or 0.0) >= 0.5
        )

    def _phase_adaptation(self, phase: Any) -> TeacherAdaptation:
        policy = getattr(phase, "teacher_adaptation", None)
        if isinstance(policy, TeacherAdaptation):
            if getattr(phase, "is_terminal", False):
                return TeacherAdaptation(enabled=False, min_weight=0.0, max_weight=0.0)
            return policy
        max_weight = 0.0 if getattr(phase, "is_terminal", False) else _clamp01(float(getattr(phase, "teacher_weight", 0.0) or 0.0))
        return TeacherAdaptation(
            enabled=max_weight > 0.0,
            min_weight=0.0,
            max_weight=max_weight,
            danger_thresholds={
                "root_height_min": 1.0,
                "torso_tilt_max": 0.6,
                "fall_rate_max": 0.2,
            },
            stability_thresholds={
                "root_height_min": 1.05,
                "torso_tilt_max": 0.35,
                "fall_rate_max": 0.08,
            },
        )

    def _init_teacher_for_phase(self, phase: Any) -> None:
        policy = self._phase_adaptation(phase)
        self._teacher_min = _clamp01(policy.min_weight)
        self._teacher_max = max(self._teacher_min, _clamp01(policy.max_weight))
        self._teacher_rise_rate = max(0.0, policy.rise_rate)
        self._teacher_fade_rate = max(0.0, policy.fade_rate)
        if not policy.enabled or getattr(phase, "is_terminal", False):
            self._teacher_assist = 0.0
            self._assist_reason = "autonomous evaluation"
        else:
            initial = float(getattr(phase, "teacher_weight", 0.0) or 0.0)
            self._teacher_assist = max(self._teacher_min, min(self._teacher_max, initial))
            self._assist_reason = "phase start"
        self._teacher_origin = self._assist_origin(self._teacher_assist)
        self._teacher_stable_dwell_s = 0.0
        self._teacher_unstable_dwell_s = 0.0
        self._teacher_last_update = time.monotonic()

    def _update_teacher_assist(self, agent: Any, phase: Any, sample: dict[str, Any]) -> None:
        policy = self._phase_adaptation(phase)
        now = time.monotonic()
        dt = max(0.0, now - (self._teacher_last_update or now))
        self._teacher_last_update = now
        if not policy.enabled or getattr(phase, "is_terminal", False):
            self._teacher_assist = 0.0
            self._teacher_stable_dwell_s = 0.0
            self._teacher_unstable_dwell_s = 0.0
            stable = self._teacher_stable(policy, sample)
            self._objective_confidence = self._compute_objective_confidence(policy, phase, sample, 0.0, stable)
            self._confidence_reason = "autonomous evaluation"
            self._assist_reason = "autonomous evaluation"
            self._teacher_origin = "self"
            self._write_teacher_training(agent, phase, sample)
            return

        demand, reason = self._teacher_demand(policy, sample)
        stable = self._teacher_stable(policy, sample)
        self._objective_confidence = self._compute_objective_confidence(policy, phase, sample, demand, stable)
        confidence_ready = self._objective_confidence >= 1.0
        if demand > 0.0:
            self._teacher_unstable_dwell_s += dt
            self._teacher_stable_dwell_s = 0.0
            self._confidence_dwell_s = 0.0
            if self._teacher_unstable_dwell_s >= policy.unstable_dwell_s:
                target = max(
                    self._teacher_assist,
                    self._teacher_min + (self._teacher_max - self._teacher_min) * max(0.25, demand),
                )
                self._teacher_assist = min(
                    self._teacher_max,
                    self._teacher_assist + self._teacher_rise_rate * dt,
                    target,
                )
            self._assist_reason = reason
        elif stable and confidence_ready:
            self._teacher_stable_dwell_s += dt
            self._teacher_unstable_dwell_s = 0.0
            self._confidence_dwell_s += dt
            if self._teacher_stable_dwell_s >= policy.stable_dwell_s:
                self._teacher_assist = max(
                    self._teacher_min,
                    self._teacher_assist - self._teacher_fade_rate * dt,
                )
                self._assist_reason = "stable, fading"
            else:
                self._assist_reason = "confidence dwell"
        else:
            self._teacher_stable_dwell_s = 0.0
            self._teacher_unstable_dwell_s = 0.0
            self._confidence_dwell_s = 0.0
            self._assist_reason = "confidence building" if stable else "monitoring"
        if self._teacher_assist <= 1e-4:
            self._teacher_assist = 0.0
        self._teacher_origin = self._assist_origin(self._teacher_assist)
        self._write_teacher_training(agent, phase, sample)

    def _compute_objective_confidence(
        self,
        policy: TeacherAdaptation,
        phase: Any,
        sample: dict[str, Any],
        demand: float,
        stable: bool,
    ) -> float:
        gate = evaluate_gate(
            list(getattr(getattr(phase, "gate", None), "criteria", ()) or ()),
            list(self._window),
            min_samples=max(1, int(getattr(getattr(phase, "gate", None), "min_samples", 1) or 1)),
        )
        sample_progress = min(1.0, len(self._window) / max(1, gate.samples if gate.enough_samples else gate.samples + 1))
        if getattr(phase, "gate", None) is not None:
            sample_progress = min(1.0, len(self._window) / max(1, phase.gate.min_samples))
        agreement = _clamp01(float(sample.get("teacher_motor_agreement", 1.0) or 1.0))
        scaffold_ok = 0.0 if self._manual_scaffold_active else 1.0
        danger_ok = max(0.0, 1.0 - _clamp01(demand))
        stable_ok = 1.0 if stable else 0.0
        confidence = min(
            _clamp01(gate.progress),
            sample_progress,
            agreement,
            scaffold_ok,
            danger_ok,
            stable_ok,
        )
        if not gate.satisfied:
            self._confidence_reason = "gate not satisfied"
        elif not stable_ok:
            self._confidence_reason = "posture not stable"
        elif agreement < 0.95:
            self._confidence_reason = "student differs from teacher"
        elif self._manual_scaffold_active:
            self._confidence_reason = "manual scaffold active"
        else:
            confidence = 1.0
            self._confidence_reason = "objective held"
        return _clamp01(confidence)

    def _teacher_demand(self, policy: TeacherAdaptation, sample: dict[str, Any]) -> tuple[float, str]:
        thresholds = policy.danger_thresholds
        demand = 0.0
        reason = ""
        for key in ("forward_model_error", "tactile_pred_error"):
            value = float(sample.get(key, 0.0) or 0.0)
            if not math.isfinite(value):
                return 1.0, f"{key} non-finite"
        if "root_height_min" in thresholds:
            value = float(sample.get("root_height", 0.0) or 0.0)
            limit = thresholds["root_height_min"]
            if value < limit:
                score = min(1.0, (limit - value) / max(0.1, abs(limit) * 0.35))
                demand, reason = _max_reason(demand, reason, score, "root height dropping")
        drop = float(sample.get("teacher_drop_m", 0.0) or 0.0)
        target_drop = max(0.01, float(sample.get("teacher_target_drop_m", 0.25) or 0.25))
        vertical_v = float(sample.get("teacher_vertical_velocity", 0.0) or 0.0)
        if drop > 0.0:
            release_drop = max(0.0, target_drop - 0.10)
            emergency_drop = target_drop + 0.05
            if drop >= emergency_drop or vertical_v > 0.8:
                score = max(0.8, min(1.0, (drop - release_drop) / max(0.01, emergency_drop - release_drop)))
                demand, reason = _max_reason(demand, reason, score, "catching drop")
            elif drop > target_drop:
                score = min(0.9, 0.5 + (drop - target_drop) / max(0.01, emergency_drop - target_drop) * 0.4)
                demand, reason = _max_reason(demand, reason, score, "recovering height")
            elif drop >= release_drop:
                score = min(0.55, (drop - release_drop) / max(0.01, target_drop - release_drop) * 0.55)
                demand, reason = _max_reason(demand, reason, score, "holding height band")
        if vertical_v > 0.25:
            score = min(0.85, vertical_v / 1.2)
            demand, reason = _max_reason(demand, reason, score, "catching downward velocity")
        if "torso_tilt_max" in thresholds:
            value = abs(float(sample.get("torso_tilt", 0.0) or 0.0))
            limit = thresholds["torso_tilt_max"]
            if value > limit:
                score = min(1.0, (value - limit) / max(0.1, abs(limit)))
                demand, reason = _max_reason(demand, reason, score, "torso tilt high")
        if "fall_rate_max" in thresholds:
            value = float(sample.get("fall_rate", 0.0) or 0.0)
            limit = thresholds["fall_rate_max"]
            if value > limit:
                score = min(1.0, (value - limit) / max(0.1, 1.0 - min(0.99, limit)))
                demand, reason = _max_reason(demand, reason, score, "fall rate high")
        for key in ("forward_model_error", "tactile_pred_error"):
            limit_key = f"{key}_max"
            if limit_key in thresholds:
                value = float(sample.get(key, 0.0) or 0.0)
                limit = thresholds[limit_key]
                if value > limit:
                    score = min(1.0, (value - limit) / max(0.1, abs(limit)))
                    demand, reason = _max_reason(demand, reason, score, f"{key} high")
        if "stance_phase_delta_min" in thresholds and len(self._window) >= 4:
            recent = list(self._window)[-4:]
            delta = float(recent[-1].get("stance_phase", 0.0) - recent[0].get("stance_phase", 0.0))
            phase_now = float(recent[-1].get("stance_phase", 0.0) or 0.0)
            if phase_now < 0.95 and delta < thresholds["stance_phase_delta_min"]:
                demand, reason = _max_reason(demand, reason, 0.75, "motion stalled")
        if "total_foot_load_min" in thresholds:
            total = float(sample.get("foot_load_l", 0.0) or 0.0) + float(sample.get("foot_load_r", 0.0) or 0.0)
            limit = thresholds["total_foot_load_min"]
            if total < limit:
                score = min(1.0, (limit - total) / max(0.1, abs(limit)))
                demand, reason = _max_reason(demand, reason, score, "feet unloading")
        return demand, reason or "stable"

    def _teacher_stable(self, policy: TeacherAdaptation, sample: dict[str, Any]) -> bool:
        thresholds = policy.stability_thresholds
        if not thresholds:
            return True
        if "root_height_min" in thresholds and float(sample.get("root_height", 0.0) or 0.0) < thresholds["root_height_min"]:
            return False
        target_drop = float(sample.get("teacher_target_drop_m", 0.0) or 0.0)
        if target_drop > 0.0 and float(sample.get("teacher_drop_m", 0.0) or 0.0) > target_drop:
            return False
        if float(sample.get("teacher_support_active", 0.0) or 0.0) >= 0.5:
            total_load = float(sample.get("foot_load_l", 0.0) or 0.0) + float(sample.get("foot_load_r", 0.0) or 0.0)
            if total_load < 0.02:
                return False
        if "torso_tilt_max" in thresholds and abs(float(sample.get("torso_tilt", 0.0) or 0.0)) > thresholds["torso_tilt_max"]:
            return False
        if "fall_rate_max" in thresholds and float(sample.get("fall_rate", 0.0) or 0.0) > thresholds["fall_rate_max"]:
            return False
        for key in ("forward_model_error", "tactile_pred_error"):
            limit_key = f"{key}_max"
            if limit_key in thresholds:
                value = float(sample.get(key, 0.0) or 0.0)
                if not math.isfinite(value) or value > thresholds[limit_key]:
                    return False
        if "stance_phase_min" in thresholds and float(sample.get("stance_phase", 0.0) or 0.0) < thresholds["stance_phase_min"]:
            return False
        if "total_foot_load_min" in thresholds:
            total = float(sample.get("foot_load_l", 0.0) or 0.0) + float(sample.get("foot_load_r", 0.0) or 0.0)
            if total < thresholds["total_foot_load_min"]:
                return False
        return True

    def _teacher_allows_success(self, phase: Any) -> bool:
        if getattr(phase, "is_terminal", False):
            return self._teacher_assist <= 1e-4
        return True

    def _assist_origin(self, assist: float) -> str:
        if assist <= 1e-4:
            return "self"
        if assist >= 0.75:
            return "demo"
        return "dagger"

    def _write_teacher_training(self, agent: Any, phase: Any, metrics: dict[str, Any]) -> None:
        if self._skill is None:
            return
        teacher = get_teacher(self._skill.teacher)
        bundle = getattr(agent, "neural", None)
        cfg = getattr(bundle, "cfg", None)
        n_act = int(getattr(cfg, "n_actuators", 21) or 21)
        assist = round(_clamp01(self._teacher_assist), 4)
        origin = self._assist_origin(assist)
        self._teacher_origin = origin
        setattr(
            agent,
            "dojo_training",
            {
                "skill_id": self._skill.skill_id,
                "origin": origin,
                "demo_weight": assist,
                "expert_motor": teacher.motor_target(n_actuators=n_act, metrics=metrics),
                "assist_reason": self._assist_reason,
                "objective_confidence": self._objective_confidence,
                "confidence_reason": self._confidence_reason,
                "teacher_live": assist > 0.0,
            },
        )
        try:
            agent.metrics["teacher_override_fraction"] = assist
        except Exception:
            pass

    def _record_sample(self, sample: dict[str, Any]) -> None:
        skill = self._skill
        phase = self._phase()
        if skill is None or phase is None:
            return
        teacher = get_teacher(skill.teacher)
        rec = DemoRecord(
            skill_id=skill.skill_id,
            phase_index=phase.index,
            origin=self._teacher_origin,
            metrics=sample,
            teacher_motor=(
                teacher.motor_target(n_actuators=21, metrics=sample)
                if self._teacher_assist > 0
                else None
            ),
            teacher_weight=self._teacher_assist,
            assist_reason=self._assist_reason,
            success=bool(self._last_gate.satisfied) if self._last_gate is not None else False,
        )
        d = rec.as_dict()
        d["objective_confidence"] = self._objective_confidence
        d["confidence_reason"] = self._confidence_reason
        self._records.append(d)
        if len(self._records) > 1000:
            self._records = self._records[-1000:]

    async def _checkpoint(self, agent: Any, reason: str) -> None:
        self._backups_dir.mkdir(parents=True, exist_ok=True)
        path = self._backups_dir / f"agent_{self._agent_id}_{self._skill.skill_id}_dojo_checkpoint.json"
        async with agent.lock:
            payload = agent.checkpoint_payload()
            brain = agent.save_brain(self._backups_dir)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("skill_dojo_checkpoint agent_id=%s reason=%s path=%s brain=%s", self._agent_id, reason, path, brain)

    def _write_report(self, *, event: str) -> None:
        if self._report_path is None:
            return
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            self._report_path.write_text(
                json.dumps(
                    {
                        "event": event,
                        "written_at": _utc(),
                        "status": self.status(),
                        "manual_scaffold_active": self._manual_scaffold_active,
                        "records": self._records[-250:],
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("skill_dojo_report_failed agent_id=%s", self._agent_id)

    def _open_report_path(self) -> None:
        if self._log_dir is None or self._agent_id is None or self._skill is None:
            return
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._report_path = self._log_dir / f"skill_dojo_{self._skill.skill_id}_{self._agent_id}_{stamp}.json"

    def _agent(self) -> Any:
        return self._registry.get(self._agent_id) if self._agent_id else None

    def _phase(self):
        if self._skill is None:
            return None
        return self._skill.phases[self._phase_idx]

    def _clear_agent_training(self) -> None:
        agent = self._agent()
        if agent is not None:
            setattr(agent, "dojo_training", None)
            try:
                agent.metrics["teacher_override_fraction"] = 0.0
            except Exception:
                pass


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _agent_cycle(agent: Any) -> int | None:
    bus = getattr(agent, "state_bus", None)
    cycle = getattr(bus, "cycle_index", None)
    try:
        return int(cycle)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _max_reason(current: float, reason: str, candidate: float, candidate_reason: str) -> tuple[float, str]:
    if candidate > current:
        return candidate, candidate_reason
    return current, reason
