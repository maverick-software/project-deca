"""Serial cognition plus lossless prefetch support.

The public module name is kept for compatibility with the earlier
``stage_pipeline`` mode, but the implementation is now a producer/consumer
pipeline:

- every observation becomes a ``DecadicSession``;
- producer workers predecode/prefetch it and fold its perceptual evidence into
  the scene model in arrival order;
- the serial Decadic cycle deep-processes one prepared observation at a time.

No fake Decadic stages are computed here. StateBus, action, optimizer, replay,
episodic memory, and LTM mutation remain owned by the runtime commit path.
"""

from __future__ import annotations

import asyncio
import copy
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


SESSION_STATUSES = {
    "received",
    "prefetching",
    "prefetched",
    "folding",
    "folded",
    "ready",
    "deep_processing",
    "deep_processed",
    "coalesced",
    "overflow",
    "consolidation",
    "failed",
}

READY_COALESCE_POLICIES = {"freshest", "oldest"}

_FORBIDDEN_KEYS = {
    "label",
    "semantic_label",
    "class",
    "class_name",
    "sim_class",
    "oracle_label",
    "task_label",
    "reward_label",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in copy.deepcopy(value).items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {
            k: _jsonable(v)
            for k, v in value.items()
            if not str(k).startswith("_") and str(k) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
            if not str(k).startswith("_") and str(k) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach"):
        return "<tensor>"
    return str(value)


def _o1_select_enabled() -> bool:
    """WS-ATTN: O(1) two-lane commit selection (urgent lane pre-empts FIFO)
    instead of the O(n) arbiter scan over the whole ready queue every cycle.
    The scan cost grew with an unbounded backlog into a feedback loop that
    slowed cognition (4.14->3.29 cyc/s) and left the agent deep-processing
    22-min-stale perception. OFF restores the arbiter scan for A/B parity."""
    return os.environ.get("DECADIC_PIPELINE_O1_SELECT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _observation_salience(observation: Any) -> tuple[float, bool]:
    """Fold-time salience tag: (salience_scalar, is_urgent), computed ONCE per
    session at fold. Derived from the observation's events/intensity -- the
    SAME signal the old arbiter scanned for, but evaluated once instead of
    every cycle. Prediction-error-adaptive salience is a later layer; this
    preserves urgency-preemption semantics exactly (urgent iff events present),
    so the O(1) path is a behavioral no-op vs the scan."""
    if not isinstance(observation, dict):
        return 0.0, False
    events = observation.get("events")
    if not (isinstance(events, list) and events):
        return 0.0, False
    intensity = 0.0
    for ev in events:
        if isinstance(ev, dict):
            try:
                intensity = max(intensity, float(ev.get("intensity", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
    return max(0.05, intensity), True


# --- WS-ATTN salience-priority hierarchy knobs (all default to today's behavior)
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _rich_salience_enabled() -> bool:
    """WS-ATTN 2.0: score fold salience from perception-organ motion/looming/
    flow (available at fold) in addition to events/intensity. Default ON; set
    DECADIC_SALIENCE_RICH=0 for events-only parity with the O(1)-select tag."""
    return _env_flag("DECADIC_SALIENCE_RICH", "1")


def _priority_select_enabled() -> bool:
    """WS-ATTN 3.0: deliberation picks the highest priority = salience x
    recency-decay (O(cap) over the small ready set) instead of FIFO. Urgent
    (threat/homeostatic) still pre-empts absolutely. Default ON; set
    DECADIC_PIPELINE_PRIORITY_SELECT=0 for FIFO O(1) parity."""
    return _env_flag("DECADIC_PIPELINE_PRIORITY_SELECT", "1")


def _overflow_enabled() -> bool:
    """WS-ATTN 4.0: T1 evictions cascade into a bounded priority overflow (T2)
    and then a consolidation queue (T3) instead of the 24-slot debug ring.
    Default ON; set DECADIC_PIPELINE_OVERFLOW=0 for drop-to-ring parity."""
    return _env_flag("DECADIC_PIPELINE_OVERFLOW", "1")


def _overflow_cap() -> int:
    return max(1, _env_int("DECADIC_PIPELINE_OVERFLOW_CAP", 100))


def _consolidation_cap() -> int:
    return max(1, _env_int("DECADIC_PIPELINE_CONSOLIDATION_CAP", 1000))


def _recency_tau_s() -> float:
    """Recency half-life (seconds) for priority decay. A surprising percept
    from long ago loses priority to a fresh one -- keeps the agent present."""
    return max(1.0, _env_float("DECADIC_PIPELINE_RECENCY_TAU_S", 30.0))


def _salience_features(observation: Any, organ_diag: Any) -> tuple[float, bool]:
    """WS-ATTN 2.0 richer fold-time salience: max of the events/intensity tag
    and a motion-surprise proxy from the perception organ diagnostics
    (`looming_count`, `local_motion_max`, `flow_confidence` -- all computed at
    fold, `perception/organ.py`). Urgency is unchanged (events present).
    Returns (salience in [0,1], urgent)."""
    base, urgent = _observation_salience(observation)
    if not _rich_salience_enabled() or not isinstance(organ_diag, dict):
        return base, urgent

    def _g(k: str) -> float:
        try:
            return float(organ_diag.get(k, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    looming = min(1.0, _g("looming_count") / 3.0)
    motion = min(1.0, _g("local_motion_max"))
    flow = min(1.0, _g("flow_confidence"))
    motion_sal = min(1.0, 0.5 * looming + 0.3 * motion + 0.2 * flow)
    return max(base, motion_sal), urgent


def _priority(sess: "DecadicSession", now: float) -> float:
    """Priority = salience x recency-decay. Recency uses fold wall-time so a
    stale-but-salient percept yields to a fresh one (the staleness trap in new
    clothes). Urgent sessions are handled by a separate absolute lane, not here."""
    import math

    ref = sess.folded_s if sess.folded_s is not None else sess.created_s
    age = max(0.0, now - float(ref))
    return float(sess.salience) * math.exp(-age / _recency_tau_s())


@dataclass(frozen=True)
class StageCandidate:
    """Compatibility shell for older diagnostics/tests.

    Real Decadic stage candidates are not produced by this module.
    """

    stage: int
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0
    urgency: float = 0.0
    confidence: float = 0.0
    prediction_error: float | None = None
    started_s: float = 0.0
    finished_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": int(self.stage),
            "name": self.name,
            "payload": _jsonable(self.payload),
            "salience": round(float(self.salience), 6),
            "urgency": round(float(self.urgency), 6),
            "confidence": round(float(self.confidence), 6),
            "prediction_error": self.prediction_error,
            "latency_ms": round(max(0.0, self.finished_s - self.started_s) * 1000.0, 4),
            "error": self.error,
        }


@dataclass
class DecadicSession:
    session_id: str
    frame_seq: int
    timestamp: str | None
    observation: dict[str, Any]
    observation_snapshot: Any
    snapshots: Any
    status: str = "received"
    created_s: float = field(default_factory=time.perf_counter)
    updated_s: float = field(default_factory=time.perf_counter)
    prefetched_s: float | None = None
    folded_s: float | None = None
    ready_s: float | None = None
    selected_s: float | None = None
    deep_processed_s: float | None = None
    failure_reason: str | None = None
    action_type: str | None = None
    # WS-ATTN: fold-time salience tag (computed ONCE at fold, not re-scanned
    # every cycle). ``urgent`` drives O(1) lane pre-emption; ``salience`` is a
    # scalar for future prediction-error-adaptive triage + the rest-pressure
    # signal.
    salience: float = 0.0
    urgent: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        frame_seq: int,
        observation: dict[str, Any],
        snapshots: dict[str, Any] | None = None,
    ) -> "DecadicSession":
        obs = copy.deepcopy(observation)
        return cls(
            session_id=f"sess-{frame_seq:08d}-{uuid.uuid4().hex[:8]}",
            frame_seq=int(frame_seq),
            timestamp=str(obs.get("timestamp")) if obs.get("timestamp") is not None else None,
            observation=obs,
            observation_snapshot=_freeze(obs),
            snapshots=_freeze(snapshots or {}),
        )

    @property
    def current_stage(self) -> int:
        # Compatibility for the old dashboard timeline: 1=prefetch, 2=folded,
        # 10=deep processed.
        if self.status in ("received", "prefetching", "prefetched"):
            return 1
        if self.status in ("folding", "folded", "ready"):
            return 2
        return 10

    @property
    def stage_results(self) -> dict[int, StageCandidate]:
        # Compatibility field. The corrected pipeline does not produce fake
        # cognition-stage outputs.
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_seq": self.frame_seq,
            "timestamp": self.timestamp,
            "current_stage": self.current_stage,
            "status": self.status,
            "age_ms": round((time.perf_counter() - self.created_s) * 1000.0, 2),
            "failure_reason": self.failure_reason,
            "snapshots": _jsonable(self.snapshots),
            "timings_ms": {k: round(float(v), 4) for k, v in self.timings_ms.items()},
            "action_type": self.action_type,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_seq": self.frame_seq,
            "timestamp": self.timestamp,
            "current_stage": self.current_stage,
            "status": self.status,
            "age_ms": round((time.perf_counter() - self.created_s) * 1000.0, 2),
            "failure_reason": self.failure_reason,
            "timings_ms": {k: round(float(v), 4) for k, v in self.timings_ms.items()},
            "action_type": self.action_type,
        }


@dataclass(frozen=True)
class CommitBundle:
    session_id: str
    frame_seq: int
    arbitration_reason: str
    salience: float = 0.0
    urgency: float = 0.0
    action_candidate: dict[str, Any] | None = None
    workspace_candidate: dict[str, Any] | None = None
    memory_candidate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_seq": self.frame_seq,
            "arbitration_reason": self.arbitration_reason,
            "salience": round(float(self.salience), 6),
            "urgency": round(float(self.urgency), 6),
            "action_candidate": _jsonable(self.action_candidate),
            "workspace_candidate": _jsonable(self.workspace_candidate),
            "memory_candidate": _jsonable(self.memory_candidate),
        }


class DecadicCommitArbiter:
    """FIFO selector, with urgent frames allowed to pre-empt."""

    def select(self, sessions: list[DecadicSession]) -> tuple[DecadicSession | None, str]:
        if not sessions:
            return None, "none_ready"
        urgent = []
        for sess in sessions:
            events = sess.observation.get("events") if isinstance(sess.observation, dict) else None
            if isinstance(events, list) and events:
                urgent.append(sess)
        if urgent:
            return min(urgent, key=lambda s: int(s.frame_seq)), "urgent"
        return min(sessions, key=lambda s: int(s.frame_seq)), "fifo"


class SerialPrefetchSupervisor:
    """Session manager for lossless perception folding + serial cognition."""

    def __init__(self, *, capacity: int = 10, coalesce_policy: str = "freshest") -> None:
        self.capacity = max(1, int(capacity))
        self.coalesce_policy = coalesce_policy if coalesce_policy in READY_COALESCE_POLICIES else "freshest"
        self.sessions: dict[str, DecadicSession] = {}
        self.by_seq: dict[int, DecadicSession] = {}
        self.ready: OrderedDict[int, DecadicSession] = OrderedDict()
        # WS-ATTN: urgent lane -- a subset index of ``ready`` (same session
        # objects) kept in fold order, so O(1) pre-emption never scans.
        self._urgent: OrderedDict[int, DecadicSession] = OrderedDict()
        self.last_select_reason = "none_ready"
        # WS-ATTN tiers 2/3: bounded priority overflow (T2) and the
        # consolidation queue (T3 intake) drained during rest. Only used when
        # DECADIC_PIPELINE_OVERFLOW is on; otherwise evictions drop as before.
        self._overflow: dict[int, DecadicSession] = {}
        self._consolidation_q: list[DecadicSession] = []
        self._overflow_spilled = 0  # -> consolidation queue (telemetry)
        self._consolidation_dropped = 0  # lowest-priority lost at T3 cap
        self._consolidation_drained = 0  # handed to rest for consolidation
        self.recent: list[dict[str, Any]] = []
        self.recent_full: list[dict[str, Any]] = []
        self.next_frame_seq = 0
        self.frames_received = 0
        self.frames_prefetched = 0
        self.frames_folded = 0
        self.frames_deep_processed = 0
        self.coalesced_sessions = 0
        self.failed_count = 0
        self.information_loss = 0
        self.producer_active_s = 0.0
        self.prefetch_backpressure_events = 0
        self.prefetch_backpressure_s = 0.0
        # WS-ATTN Phase 0: cap-leak diagnostics. Static reading could not
        # explain how ``ready`` reached ~10k while capacity=10 and coalesce
        # should trim every pop. These catch it in the act:
        #   pop_calls        -- times pop_commit_candidate ran (should ~= cycles)
        #   coalesce_calls   -- times the overflow trim actually FIRED (>capacity)
        #   max_ready_depth  -- high-water mark of the ready queue during the run
        self._pop_calls = 0
        self._coalesce_calls = 0
        self._max_ready_depth = 0
        self.started_s = time.perf_counter()
        self.last_selected: CommitBundle | None = None
        self.last_fold_s: float | None = None
        self.last_deep_processed_s: float | None = None
        self.last_decode_on_consume_ms: float | None = None
        self.last_consume_wait_ms: float | None = None
        self._lock = asyncio.Lock()
        self.arbiter = DecadicCommitArbiter()

    def set_capacity(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))

    def set_coalesce_policy(self, policy: str) -> None:
        self.coalesce_policy = policy if policy in READY_COALESCE_POLICIES else "freshest"

    def start(self) -> None:
        return

    async def stop(self) -> None:
        return

    def clear(self) -> None:
        self.sessions.clear()
        self.by_seq.clear()
        self.ready.clear()
        self._urgent.clear()
        self._overflow.clear()
        self._consolidation_q.clear()
        self._overflow_spilled = 0
        self._consolidation_dropped = 0
        self._consolidation_drained = 0
        self.recent.clear()
        self.recent_full.clear()
        self.next_frame_seq = 0
        self.frames_received = 0
        self.frames_prefetched = 0
        self.frames_folded = 0
        self.frames_deep_processed = 0
        self.coalesced_sessions = 0
        self.failed_count = 0
        self.information_loss = 0
        self.producer_active_s = 0.0
        self.prefetch_backpressure_events = 0
        self.prefetch_backpressure_s = 0.0
        self._pop_calls = 0
        self._coalesce_calls = 0
        self._max_ready_depth = 0
        self.started_s = time.perf_counter()
        self.last_selected = None
        self.last_fold_s = None
        self.last_deep_processed_s = None
        self.last_decode_on_consume_ms = None
        self.last_consume_wait_ms = None

    async def enqueue_observation(
        self,
        observation: dict[str, Any],
        *,
        snapshots: dict[str, Any] | None = None,
    ) -> DecadicSession:
        async with self._lock:
            self.next_frame_seq += 1
            sess = DecadicSession.create(
                frame_seq=self.next_frame_seq,
                observation=observation,
                snapshots=snapshots,
            )
            self.sessions[sess.session_id] = sess
            self.by_seq[sess.frame_seq] = sess
            self.frames_received += 1
            return sess

    async def mark_prefetching(self, frame_seq: int) -> None:
        async with self._lock:
            sess = self.by_seq.get(int(frame_seq))
            if sess is not None:
                sess.status = "prefetching"
                sess.updated_s = time.perf_counter()

    async def mark_prefetched(self, frame_seq: int, *, elapsed_s: float = 0.0) -> None:
        async with self._lock:
            sess = self.by_seq.get(int(frame_seq))
            if sess is None:
                return
            sess.status = "prefetched"
            sess.prefetched_s = time.perf_counter()
            sess.updated_s = sess.prefetched_s
            sess.timings_ms["prefetch"] = max(0.0, float(elapsed_s) * 1000.0)
            self.producer_active_s += max(0.0, float(elapsed_s))
            self.frames_prefetched += 1

    async def record_prefetch_backpressure(self, *, elapsed_s: float) -> None:
        elapsed_s = max(0.0, float(elapsed_s))
        if elapsed_s <= 0.0:
            return
        async with self._lock:
            self.prefetch_backpressure_events += 1
            self.prefetch_backpressure_s += elapsed_s

    async def mark_folding(self, frame_seq: int) -> None:
        async with self._lock:
            sess = self.by_seq.get(int(frame_seq))
            if sess is not None:
                sess.status = "folding"
                sess.updated_s = time.perf_counter()

    async def mark_folded(
        self,
        frame_seq: int,
        *,
        elapsed_s: float = 0.0,
        organ_diag: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            sess = self.by_seq.get(int(frame_seq))
            if sess is None:
                return
            now = time.perf_counter()
            sess.status = "ready"
            sess.folded_s = now
            sess.ready_s = now
            sess.updated_s = now
            sess.timings_ms["fold"] = max(0.0, float(elapsed_s) * 1000.0)
            self.frames_folded += 1
            self.last_fold_s = now
            # WS-ATTN: tag salience ONCE here (not re-derived every cycle).
            # organ_diag (motion/looming/flow, computed at fold) enriches the
            # score when DECADIC_SALIENCE_RICH is on; else events-only (parity).
            sal, urgent = _salience_features(sess.observation, organ_diag)
            sess.salience = sal
            sess.urgent = urgent
            self.ready[sess.frame_seq] = sess
            if urgent:
                self._urgent[sess.frame_seq] = sess

    async def mark_failed(self, frame_seq: int, reason: str) -> None:
        async with self._lock:
            sess = self.by_seq.get(int(frame_seq))
            if sess is None:
                self.information_loss += 1
                return
            sess.status = "failed"
            sess.failure_reason = reason
            sess.updated_s = time.perf_counter()
            self.failed_count += 1
            self.information_loss += 1
            self._forget(sess)

    def _coalesce_ready_overflow_locked(self) -> None:
        if len(self.ready) <= self.capacity:
            return
        self._coalesce_calls += 1  # WS-ATTN Phase 0: overflow trim FIRED
        use_priority = _overflow_enabled() or _priority_select_enabled()
        to_overflow = _overflow_enabled()
        while len(self.ready) > self.capacity:
            if use_priority:
                # Keep the highest-priority in T1: evict the LOWEST-priority.
                now = time.perf_counter()
                fs = min(self.ready, key=lambda k: _priority(self.ready[k], now))
                skipped = self.ready.pop(fs)
            elif self.coalesce_policy == "oldest":
                _, skipped = self.ready.popitem(last=True)
            else:
                _, skipped = self.ready.popitem(last=False)
            self._urgent.pop(skipped.frame_seq, None)
            self.coalesced_sessions += 1
            skipped.updated_s = time.perf_counter()
            if to_overflow:
                # WS-ATTN T2: not dropped -- staged in the priority overflow,
                # may be promoted back to T1 or consolidated during rest.
                self._overflow_push_locked(skipped)
            else:
                skipped.status = "coalesced"
                self._remember(skipped)
                self.sessions.pop(skipped.session_id, None)
                self.by_seq.pop(skipped.frame_seq, None)

    # ---- WS-ATTN tiers 2/3 (priority overflow + consolidation) --------------
    def _overflow_push_locked(self, sess: DecadicSession) -> None:
        sess.status = "overflow"
        self._overflow[sess.frame_seq] = sess
        cap = _overflow_cap()
        while len(self._overflow) > cap:
            now = time.perf_counter()
            fs = min(self._overflow, key=lambda k: _priority(self._overflow[k], now))
            spilled = self._overflow.pop(fs)
            self._overflow_spilled += 1
            self._consolidation_push_locked(spilled)

    def _consolidation_push_locked(self, sess: DecadicSession) -> None:
        sess.status = "consolidation"
        self._consolidation_q.append(sess)
        ccap = _consolidation_cap()
        if len(self._consolidation_q) > ccap:
            # Genuine, priority-ordered forgetting: keep the top ccap, drop the
            # least important (the "you don't remember lunch" tier).
            now = time.perf_counter()
            self._consolidation_q.sort(key=lambda s: _priority(s, now), reverse=True)
            for d in self._consolidation_q[ccap:]:
                self._consolidation_dropped += 1
                self._remember(d)
                self.sessions.pop(d.session_id, None)
                self.by_seq.pop(d.frame_seq, None)
            del self._consolidation_q[ccap:]

    def _promote_from_overflow_locked(self) -> None:
        """When T1 has spare deliberation capacity (consumer caught up), pull
        the highest-priority percepts back from T2 for a second chance."""
        if not _overflow_enabled() or not self._overflow:
            return
        now = time.perf_counter()
        while len(self.ready) < self.capacity and self._overflow:
            fs = max(self._overflow, key=lambda k: _priority(self._overflow[k], now))
            sess = self._overflow.pop(fs)
            sess.status = "ready"
            sess.ready_s = time.perf_counter()
            self.ready[fs] = sess
            if sess.urgent:
                self._urgent[fs] = sess

    def drain_consolidation(self, n: int = 64) -> list[DecadicSession]:
        """WS-ATTN T3: hand the highest-priority un-deliberated percepts to the
        caller (rest consolidation). Removes them from the pipeline.

        Sync + lock-free by design: the runtime calls this from the (sync)
        rest-scheduling step inside the single-threaded asyncio loop; with no
        ``await`` inside it runs atomically w.r.t. every other coroutine, so
        the asyncio ``_lock`` is unnecessary (and can't be taken from sync code)."""
        if not self._consolidation_q:
            return []
        now = time.perf_counter()
        self._consolidation_q.sort(key=lambda s: _priority(s, now), reverse=True)
        out = self._consolidation_q[: max(1, int(n))]
        del self._consolidation_q[: max(1, int(n))]
        for s in out:
            self._consolidation_drained += 1
            self.sessions.pop(s.session_id, None)
            self.by_seq.pop(s.frame_seq, None)
        return out

    def pressure(self) -> float:
        """WS-ATTN 6.1: normalized backpressure across the tiers (0..~3).
        Sustained high pressure is the homeostatic signal to rest."""
        t1 = len(self.ready) / max(1, self.capacity)
        t2 = len(self._overflow) / max(1, _overflow_cap())
        t3 = len(self._consolidation_q) / max(1, _consolidation_cap())
        return float(t1 + t2 + t3)

    async def pop_commit_candidate(self) -> tuple[DecadicSession | None, CommitBundle | None]:
        async with self._lock:
            self._pop_calls += 1  # WS-ATTN Phase 0: should track cycle count
            if len(self.ready) > self._max_ready_depth:
                self._max_ready_depth = len(self.ready)  # high-water pre-trim
            self._coalesce_ready_overflow_locked()
            self._promote_from_overflow_locked()  # T2->T1 if spare capacity
            if not self.ready:
                return None, None
            if self._urgent:
                # Threat/homeostatic always pre-empts (non-habituating).
                selected = next(iter(self._urgent.values()))
                reason = "urgent"
            elif _priority_select_enabled():
                # WS-ATTN 3.0: highest priority = salience x recency-decay,
                # O(cap) over the small (<=capacity) ready set.
                now = time.perf_counter()
                fs = max(self.ready, key=lambda k: _priority(self.ready[k], now))
                selected = self.ready[fs]
                reason = "priority"
            elif _o1_select_enabled():
                # O(1): fold order == frame_seq order, so the head IS the
                # min-frame_seq the arbiter would scan for. FIFO.
                selected = next(iter(self.ready.values()))
                reason = "fifo"
            else:
                selected, reason = self.arbiter.select(list(self.ready.values()))
                if selected is None:
                    return None, None
            self.last_select_reason = reason
            self.ready.pop(selected.frame_seq, None)
            self._urgent.pop(selected.frame_seq, None)
            selected.status = "deep_processing"
            selected.selected_s = time.perf_counter()
            selected.updated_s = selected.selected_s
            if selected.ready_s is not None:
                self.last_consume_wait_ms = max(0.0, (selected.selected_s - selected.ready_s) * 1000.0)
                selected.timings_ms["consume_wait"] = self.last_consume_wait_ms
            self.last_decode_on_consume_ms = float(selected.timings_ms.get("prefetch", 0.0))
            bundle = CommitBundle(
                session_id=selected.session_id,
                frame_seq=selected.frame_seq,
                arbitration_reason=reason,
            )
            self.last_selected = bundle
            return selected, bundle

    async def mark_committed(self, session_id: str, *, action_type: str | None = None) -> None:
        async with self._lock:
            sess = self.sessions.pop(session_id, None)
            if sess is None:
                return
            self.by_seq.pop(sess.frame_seq, None)
            now = time.perf_counter()
            sess.status = "deep_processed"
            sess.deep_processed_s = now
            sess.updated_s = now
            sess.action_type = action_type
            if sess.selected_s is not None:
                sess.timings_ms["deep_process"] = max(0.0, (now - sess.selected_s) * 1000.0)
            self.frames_deep_processed += 1
            self.last_deep_processed_s = now
            self._remember(sess)

    def _remember(self, sess: DecadicSession) -> None:
        self.recent.append(sess.to_summary_dict())
        self.recent = self.recent[-24:]
        self.recent_full.append(sess.to_dict())
        self.recent_full = self.recent_full[-24:]

    def _forget(self, sess: DecadicSession) -> None:
        self.sessions.pop(sess.session_id, None)
        self.by_seq.pop(sess.frame_seq, None)
        self.ready.pop(sess.frame_seq, None)
        self._urgent.pop(sess.frame_seq, None)
        self._overflow.pop(sess.frame_seq, None)
        self._remember(sess)

    def metrics(self) -> dict[str, Any]:
        elapsed = max(1e-6, time.perf_counter() - self.started_s)
        active = len(self.sessions)
        received = max(1, self.frames_received)
        now = time.perf_counter()
        unfolded_ages = [
            max(0.0, (now - s.created_s) * 1000.0)
            for s in self.sessions.values()
            if s.status not in ("folded", "ready", "deep_processing", "deep_processed", "coalesced")
        ]
        return {
            "active_sessions": active,
            "ready_sessions": len(self.ready),
            "committed_sessions": self.frames_deep_processed,
            "committed_sessions_per_s": self.frames_deep_processed / elapsed,
            "dropped_sessions": 0,
            "stale_sessions": 0,
            "failed_sessions": self.failed_count,
            "coalesced_sessions": self.coalesced_sessions,
            "information_loss": self.information_loss,
            "frames_received": self.frames_received,
            "frames_prefetched": self.frames_prefetched,
            "frames_folded": self.frames_folded,
            "frames_deep_processed": self.frames_deep_processed,
            "producer_overlap_ratio": min(1.0, self.producer_active_s / elapsed),
            "decode_on_consume_ms": self.last_decode_on_consume_ms,
            "consume_wait_ms": self.last_consume_wait_ms,
            "ready_queue_depth": len(self.ready),
            "urgent_queue_depth": len(self._urgent),
            "last_select_reason": self.last_select_reason,
            # WS-ATTN Phase 0 cap-leak diagnostics:
            "ready_capacity": int(self.capacity),
            "ready_pop_calls": int(self._pop_calls),
            "ready_coalesce_calls": int(self._coalesce_calls),
            "ready_max_depth": int(self._max_ready_depth),
            # WS-ATTN tiers 2/3 + pressure:
            "overflow_depth": len(self._overflow),
            "consolidation_depth": len(self._consolidation_q),
            "overflow_spilled": int(self._overflow_spilled),
            "consolidation_dropped": int(self._consolidation_dropped),
            "consolidation_drained": int(self._consolidation_drained),
            "attn_pressure": float(self.pressure()),
            "ready_coalesce_policy": self.coalesce_policy,
            "prefetch_backpressure_events": self.prefetch_backpressure_events,
            "prefetch_backpressure_ms": self.prefetch_backpressure_s * 1000.0,
            "oldest_unfolded_age_ms": max(unfolded_ages) if unfolded_ages else 0.0,
            "stage_queue_depths": {"prefetch": active - len(self.ready), "ready": len(self.ready)},
            "stage_inflight": {"prefetch": max(0, active - len(self.ready)), "commit": 0},
            "stage_latency_ms": {},
            "commit_lag_ms": (
                (time.perf_counter() - self.last_deep_processed_s) * 1000.0
                if self.last_deep_processed_s is not None
                else None
            ),
            "fold_lag_ms": (
                (time.perf_counter() - self.last_fold_s) * 1000.0
                if self.last_fold_s is not None
                else None
            ),
            "selected_session": self.last_selected.to_dict() if self.last_selected else None,
            "recent_sessions": list(reversed(self.recent[-12:])),
            "fold_ratio": self.frames_folded / received,
        }

    def debug_sessions(self, limit: int = 24) -> list[dict[str, Any]]:
        limit = max(1, min(100, int(limit)))
        return list(reversed(self.recent_full[-limit:]))


# Compatibility names used by the previous implementation and public API.
DecadicStagePipelineSupervisor = SerialPrefetchSupervisor
