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

    async def mark_folded(self, frame_seq: int, *, elapsed_s: float = 0.0) -> None:
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
            self.ready[sess.frame_seq] = sess

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
        while len(self.ready) > self.capacity:
            if self.coalesce_policy == "oldest":
                _, skipped = self.ready.popitem(last=True)
            else:
                _, skipped = self.ready.popitem(last=False)
            skipped.status = "coalesced"
            skipped.updated_s = time.perf_counter()
            self.coalesced_sessions += 1
            self._remember(skipped)
            self.sessions.pop(skipped.session_id, None)
            self.by_seq.pop(skipped.frame_seq, None)

    async def pop_commit_candidate(self) -> tuple[DecadicSession | None, CommitBundle | None]:
        async with self._lock:
            if not self.ready:
                return None, None
            self._coalesce_ready_overflow_locked()
            selected, reason = self.arbiter.select(list(self.ready.values()))
            if selected is None:
                return None, None
            self.ready.pop(selected.frame_seq, None)
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
