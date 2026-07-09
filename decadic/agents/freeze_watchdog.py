"""WS-FREEZE — silent-hang watchdog.

The cognitive loop runs its heavy work synchronously under the runtime lock, so
a blocking call (a wedged background thread holding the graph RLock, a dead
kuzu flusher, a CUDA stall) freezes the whole event loop with no traceback:
cycles simply stop and ``commit_lag`` climbs with wall-clock. Runs to date
have shown exactly this at ~22-24k cycles / ~100 min.

This watchdog is READ-ONLY and LOCK-FREE by construction: subsystems write
plain monotonic-timestamp heartbeats (GIL-atomic single assignments), and a
daemon thread only reads them. It holds no lock and mutates no shared state, so
it can neither cause nor worsen a freeze -- the worst it can do is print.

When cognition goes stale it emits one structured ``FREEZE_REPORT`` naming each
hypothesis's live state, then a full ``faulthandler`` all-thread stack dump. The
report is designed to attribute the freeze to any, several, or none of the
candidates:

  H1  write-behind worker holding the graph RLock (``wb_in_job`` + hold age)
  H2  kuzu flusher dead / stuck (``flusher_alive`` + last-batch age + backlog)
  H3  neural/other (cognition ``phase`` + the actual stack in the dump)
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def watchdog_enabled() -> bool:
    return _env_flag("DECADIC_WATCHDOG", "1")


def _stall_s() -> float:
    # Long enough never to fire on a slow-but-live cycle, short enough to catch
    # a real hang fast.
    return max(1.0, _env_float("DECADIC_WATCHDOG_STALL_S", 20.0))


def _check_s() -> float:
    return max(0.1, _env_float("DECADIC_WATCHDOG_CHECK_S", 5.0))


def _repeat_s() -> float:
    return max(1.0, _env_float("DECADIC_WATCHDOG_REPEAT_S", 60.0))


class FreezeWatchdog:
    """Polls a heartbeat ``probe`` on a daemon thread; dumps stacks on a stall.

    ``probe`` returns a snapshot dict::

        {
          "now": <monotonic>,
          "cognition": {"hb_cycle_s", "phase", "cycle_index"},
          "write_behind": {"in_job", "job_start_s", "jobs_completed",
                           "last_worker_ms", "queue_size"},
          "flusher": {"alive", "last_batch_s", "backlog"},
        }

    ``dump`` and ``clock`` are injectable so ``check_once`` is unit-testable
    without a real agent or a real hang.
    """

    def __init__(
        self,
        probe: Callable[[], dict[str, Any]],
        *,
        agent_id: str = "",
        stall_s: float | None = None,
        check_s: float | None = None,
        repeat_s: float | None = None,
        dump: Callable[[], None] | None = None,
    ) -> None:
        self._probe = probe
        self._agent_id = agent_id
        self._stall_s = stall_s if stall_s is not None else _stall_s()
        self._check_s = check_s if check_s is not None else _check_s()
        self._repeat_s = repeat_s if repeat_s is not None else _repeat_s()
        self._dump = dump if dump is not None else self._default_dump
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_report_s = 0.0
        self.reports = 0  # count (telemetry/testing)
        self.last_report: str | None = None

    @staticmethod
    def _default_dump() -> None:
        # All-thread stacks to stderr -> captured in server.err.log.
        faulthandler.dump_traceback(all_threads=True)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"freeze-watchdog-{self._agent_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._check_s)
            if self._stop.is_set():
                break
            try:
                self.check_once()
            except Exception:  # pragma: no cover - watchdog must never raise
                logger.debug("freeze watchdog check failed", exc_info=True)

    def check_once(self) -> dict[str, Any] | None:
        """One poll. Returns the report dict if a freeze was reported, else None.

        Split out from the thread loop so tests can drive it deterministically.
        """
        snap = self._probe() or {}
        cog = snap.get("cognition") or {}
        hb_cycle_s = cog.get("hb_cycle_s")
        if hb_cycle_s is None:
            return None
        now = float(snap.get("now", time.monotonic()))
        age = now - float(hb_cycle_s)
        if age < self._stall_s:
            return None
        # Throttle re-emits while still stalled.
        if self.reports > 0 and (now - self._last_report_s) < self._repeat_s:
            return None
        self._last_report_s = now
        self.reports += 1
        # A dead agent idling is EXPECTED (mortality), not a silent hang -- label
        # it clearly and skip the stack dump. (In soak tests revive keeps it
        # alive, so this only fires on a real death in a normal run.)
        if str(cog.get("status", "")).lower() == "dead":
            line = (
                f"agent={self._agent_id} state=AGENT_DEAD (viability=0, expected "
                f"idle -- NOT a hang) cycle={cog.get('cycle_index', '?')} "
                f"idle={age:.1f}s"
            )
            self.last_report = line
            logger.error("FREEZE_REPORT %s", line)
            return {"line": line, "snapshot": snap, "stalled_s": age, "dead": True}
        line = self._build_report(snap, age)
        self.last_report = line
        logger.error("FREEZE_REPORT %s", line)
        try:
            self._dump()
        except Exception:  # pragma: no cover - dumping must never raise
            logger.debug("faulthandler dump failed", exc_info=True)
        return {"line": line, "snapshot": snap, "stalled_s": age}

    def _build_report(self, snap: dict[str, Any], age: float) -> str:
        now = float(snap.get("now", time.monotonic()))
        cog = snap.get("cognition") or {}
        wb = snap.get("write_behind") or {}
        fl = snap.get("flusher") or {}

        def _age(v: Any) -> str:
            return "-" if v is None else f"{now - float(v):.1f}s"

        wb_in_job = bool(wb.get("in_job"))
        parts = [
            f"agent={self._agent_id}",
            f"cognition_stalled={age:.1f}s",
            f"phase={cog.get('phase', '?')}",
            f"cycle={cog.get('cycle_index', '?')}",
            # H1 -- write-behind holding the graph RLock
            f"H1_wb_in_job={wb_in_job}",
            f"H1_wb_lock_held={_age(wb.get('job_start_s')) if wb_in_job else '-'}",
            f"wb_jobs={wb.get('jobs_completed', '?')}",
            f"wb_last_ms={wb.get('last_worker_ms', '?')}",
            f"wb_queue={wb.get('queue_size', '?')}",
            # H2 -- kuzu flusher dead / stuck
            f"H2_flusher_alive={bool(fl.get('alive'))}",
            f"flusher_last_batch={_age(fl.get('last_batch_s'))}",
            f"flusher_backlog={fl.get('backlog', '?')}",
            # H3 -- neural/other is read from the phase + the stack dump below
            "H3_see=stack_dump",
        ]
        return " ".join(parts)
