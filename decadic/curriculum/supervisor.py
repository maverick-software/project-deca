"""Server-side walking-curriculum supervisor (single-slot, in-process).

A ``CurriculumSupervisor`` drives ONE learner agent through the developmental
phase table. It is the 'parent that shapes the world and reads gates': every
tick it samples eval-only metrics, places satisfiers on a cadence, and - when a
phase's observational gate opens after a minimum dwell - applies the next phase's
live config and checkpoints the brain.

Faithfulness invariant: the supervisor only ever READS metrics/state, calls
``AgentRuntime.configure`` (which reweights the existing objective), queues body
commands (world shaping), and checkpoints. It NEVER calls the cognitive cycle and
never adds a term to the loss. See tests/test_curriculum_faithfulness.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from decadic.curriculum.gates import GateResult, evaluate_gate
from decadic.curriculum.phases import Phase, build_phases

if TYPE_CHECKING:  # avoid an import cycle (registry -> runtime -> ...)
    from decadic.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

WINDOW_MAX = 240
DEFAULT_POLL_S = 2.0

# Metric keys the gates read; the sample is built from exactly these so the
# faithfulness test can assert the supervisor consumes telemetry only.
SAMPLE_KEYS = (
    "viability",
    "forward_model_error",
    "tactile_pred_error",
    "rom_mean",
    "brace_engaged",
    "fall_rate",
    "gait_regularity",
    "distance_traveled",
    "net_displacement",
    "consume_events",
    "neural_pc_loss_last",
)


class CurriculumError(RuntimeError):
    """Invalid curriculum action (e.g. starting while one already runs)."""


class CurriculumSupervisor:
    """Owns at most one running curriculum bound to one agent."""

    def __init__(
        self,
        registry: "AgentRegistry",
        *,
        backups_dir: Path,
        log_dir: Path | None = None,
    ) -> None:
        self._registry = registry
        self._backups_dir = Path(backups_dir)
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._poll_s = max(
            0.25, float(os.environ.get("DECADIC_CURRICULUM_POLL_S", str(DEFAULT_POLL_S)))
        )
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self._agent_id: str | None = None
        self._phases: list[Phase] = []
        self._phase_idx: int = 0
        self._state: str = "stopped"  # stopped|running|paused|error
        self._graduated: bool = False
        self._window: deque[dict[str, float]] = deque(maxlen=WINDOW_MAX)
        self._phase_started: float = 0.0
        self._started_at: float = 0.0
        self._last_give: dict[str, float] = {}
        self._history: list[dict[str, Any]] = []
        self._last_gate: GateResult | None = None
        self._error: str | None = None
        self._log_path: Path | None = None

    # --- lifecycle ---------------------------------------------------------

    async def start(
        self,
        agent_id: str,
        *,
        include_affective: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            if self.is_running():
                raise CurriculumError(
                    "A curriculum is already running; stop it first."
                )
            agent = self._registry.get(agent_id)
            if agent is None:
                raise CurriculumError(f"Unknown agent {agent_id!r}")
            self._reset_run_state()
            self._agent_id = agent_id
            self._phases = build_phases(
                include_affective=include_affective, overrides=overrides
            )
            self._phase_idx = 0
            self._state = "running"
            self._started_at = time.time()
            self._phase_started = time.monotonic()
            self._open_log()
            self._log(f"curriculum start agent={agent_id} phases={len(self._phases)}")
            await self._apply_phase(self._phases[0], reason="start")
            self._task = asyncio.create_task(
                self._loop(), name=f"decadic-curriculum-{agent_id}"
            )
            return self._status_dict()

    def pause(self) -> dict[str, Any]:
        if not self.is_running():
            raise CurriculumError("No running curriculum to pause.")
        self._state = "paused"
        self._log("curriculum paused")
        return self._status_dict()

    def resume(self) -> dict[str, Any]:
        if self._task is None or self._task.done():
            raise CurriculumError("No curriculum to resume.")
        self._state = "running"
        self._log("curriculum resumed")
        return self._status_dict()

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
            self._log("curriculum stopped")
            self._close_log()
            status = self._status_dict()
            self._state = "stopped"
            self._graduated = False
            return {**status, "state": "stopped"}

    async def set_phase(self, index: int) -> dict[str, Any]:
        """Manual override: jump to a phase and apply its config (experiments)."""
        async with self._lock:
            if not self._phases:
                raise CurriculumError("No curriculum running.")
            idx = max(0, min(len(self._phases) - 1, int(index)))
            agent = self._current_agent()
            if agent is None:
                raise CurriculumError("Bound agent is gone.")
            self._phase_idx = idx
            self._graduated = False
            self._window.clear()
            self._phase_started = time.monotonic()
            self._last_give.clear()
            await self._apply_phase(self._phases[idx], reason="manual")
            return self._status_dict()

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        return self._status_dict()

    # --- core loop ---------------------------------------------------------

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_s)
                if self._state != "running":
                    continue
                agent = self._current_agent()
                if agent is None:
                    self._state = "error"
                    self._error = "bound agent disappeared"
                    self._log("curriculum error: bound agent disappeared")
                    return
                sample = await self._sample(agent)
                self._window.append(sample)
                phase = self._phases[self._phase_idx]

                # Death: revive and (optionally) step back a phase to retry.
                if agent.status == "dead":
                    await self._on_death(agent, phase)
                    continue

                await self._maybe_give(agent, phase)

                gate = evaluate_gate(
                    phase.promote_criteria, list(self._window), min_samples=phase.min_samples
                )
                self._last_gate = gate
                dwell_ok = (time.monotonic() - self._phase_started) >= phase.min_dwell_s
                if gate.satisfied and dwell_ok:
                    if phase.is_terminal:
                        if not self._graduated:
                            self._graduated = True
                            self._record_history(phase, "graduated")
                            self._log(f"curriculum graduated at phase {phase.index} ({phase.name})")
                            await self._checkpoint(agent, "graduated")
                    else:
                        await self._promote(agent)
        except asyncio.CancelledError:
            raise
        except Exception:  # never let the loop die silently
            logger.exception("curriculum_loop_failed agent_id=%s", self._agent_id)
            self._state = "error"
            self._error = "loop exception (see server log)"

    async def _promote(self, agent: Any) -> None:
        prev = self._phases[self._phase_idx]
        await self._checkpoint(agent, f"promote_from_{prev.index}")
        self._record_history(prev, "promoted")
        self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
        self._window.clear()
        self._phase_started = time.monotonic()
        self._last_give.clear()
        nxt = self._phases[self._phase_idx]
        self._log(f"curriculum promote -> phase {nxt.index} ({nxt.name})")
        await self._apply_phase(nxt, reason="promote")

    async def _on_death(self, agent: Any, phase: Phase) -> None:
        self._log(f"curriculum agent died in phase {phase.index} ({phase.name})")
        try:
            agent.revive()
        except Exception:
            logger.exception("curriculum_revive_failed agent_id=%s", self._agent_id)
        self._window.clear()
        self._phase_started = time.monotonic()
        if phase.demote_on_death and self._phase_idx > 0:
            self._record_history(phase, "demoted")
            self._phase_idx -= 1
            self._graduated = False
            target = self._phases[self._phase_idx]
            self._log(f"curriculum demote -> phase {target.index} ({target.name})")
            await self._apply_phase(target, reason="demote")

    # --- agent interaction (read / configure / world / checkpoint only) ----

    def _current_agent(self) -> Any:
        if self._agent_id is None:
            return None
        return self._registry.get(self._agent_id)

    async def _sample(self, agent: Any) -> dict[str, float]:
        async with agent.lock:
            m = dict(agent.metrics)
            viability = float(agent.viability.value)
        sample: dict[str, float] = {"viability": viability}
        for key in SAMPLE_KEYS:
            if key == "viability":
                continue
            v = m.get(key, 0.0)
            try:
                sample[key] = float(v)
            except (TypeError, ValueError):
                sample[key] = 0.0
        return sample

    async def _apply_phase(self, phase: Phase, *, reason: str) -> None:
        agent = self._current_agent()
        if agent is None:
            return
        kwargs = phase.config.to_configure_kwargs()
        if not kwargs:
            return
        try:
            async with agent.lock:
                agent.configure(**kwargs)
            self._log(f"phase {phase.index} config applied ({reason}): {kwargs}")
        except Exception:
            logger.exception("curriculum_configure_failed agent_id=%s", self._agent_id)

    async def _maybe_give(self, agent: Any, phase: Phase) -> None:
        pol = phase.satisfier
        if not pol.enabled or not agent.has_body():
            return
        now = time.monotonic()
        for res in pol.resources:
            last = self._last_give.get(res, 0.0)
            if now - last >= pol.period_s:
                if agent.queue_body_command(f"give_{res}_{pol.mode}"):
                    self._last_give[res] = now
                    self._log(f"placed {res} ({pol.mode}) in phase {phase.index}")

    async def _checkpoint(self, agent: Any, reason: str) -> None:
        try:
            self._backups_dir.mkdir(parents=True, exist_ok=True)
            path = self._backups_dir / f"agent_{self._agent_id}_checkpoint.json"
            async with agent.lock:
                payload = agent.checkpoint_payload()
                brain = agent.save_brain(self._backups_dir)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._log(f"checkpoint ({reason}): {path.name} brain={brain}")
        except Exception:
            logger.exception("curriculum_checkpoint_failed agent_id=%s", self._agent_id)

    # --- reporting / logging ----------------------------------------------

    def _record_history(self, phase: Phase, event: str) -> None:
        self._history.append(
            {
                "phase": phase.index,
                "name": phase.name,
                "event": event,
                "at": datetime.now(UTC).isoformat(),
                "cycles": None,
            }
        )

    def _status_dict(self) -> dict[str, Any]:
        phase = self._phases[self._phase_idx] if self._phases else None
        gate = self._last_gate.as_dict() if self._last_gate is not None else None
        dwell_s = (
            round(time.monotonic() - self._phase_started, 1)
            if self._phase_started and self._state in ("running", "paused")
            else 0.0
        )
        return {
            "state": self._state,
            "running": self.is_running(),
            "paused": self._state == "paused",
            "graduated": self._graduated,
            "agent_id": self._agent_id,
            "phase_index": phase.index if phase else None,
            "phase_name": phase.name if phase else None,
            "phase_description": phase.description if phase else None,
            "phase_count": len(self._phases),
            "is_terminal": phase.is_terminal if phase else False,
            "min_dwell_s": phase.min_dwell_s if phase else None,
            "dwell_s": dwell_s,
            "window_size": len(self._window),
            "satisfier": {
                "enabled": phase.satisfier.enabled if phase else False,
                "resources": list(phase.satisfier.resources) if phase else [],
                "period_s": phase.satisfier.period_s if phase else None,
            },
            "gate": gate,
            "history": list(self._history[-20:]),
            "started_at": self._started_at or None,
            "poll_interval_s": self._poll_s,
            "error": self._error,
            "log_path": str(self._log_path) if self._log_path else None,
        }

    def _open_log(self) -> None:
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self._log_dir / f"curriculum_{self._agent_id}.log"
        except Exception:
            self._log_path = None

    def _close_log(self) -> None:
        self._log_path = None

    def _log(self, message: str) -> None:
        logger.info("curriculum agent_id=%s %s", self._agent_id, message)
        if self._log_path is None:
            return
        try:
            stamp = datetime.now(UTC).isoformat()
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {message}\n")
        except Exception:
            pass
