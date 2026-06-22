"""Write-behind long-term graph: move the per-cycle WM->LTM consolidation off the hot path.

Stage 10 calls ``ltm_graph.consolidate(active_slots, ...)`` once per cycle inside the
agent lock. That consolidation mutates the in-memory graph *and* persists each touched
node/edge to SQLite -- ending in ``conn.commit()`` (an fsync) that blocks the cognitive
cycle. When async mode is ON this wrapper hands the whole consolidation to a single
background worker thread so the cycle returns immediately; every read (``match`` /
``snapshot`` / ``counts``) is inherited unchanged from :class:`LongTermGraph` and runs
against the same thread-safe, lock-guarded state.

Mirrors :class:`~decadic.memory.write_behind.WriteBehindEpisodicStore`:

- This graph is **always** used by the runtime so the mode can be flipped live from the
  dashboard (Agent Settings) without recreating the graph or migrating its SQLite
  connection. ``set_async(False)`` drains pending jobs and stops the worker, after which
  ``consolidate`` runs synchronously -- byte-identical to a bare :class:`LongTermGraph`.
- The worker is started lazily, so an agent that never enables async never spawns a
  thread (the test baseline, pinned OFF, has zero overhead).

Correctness:
- No consolidation is ever lost: if the bounded queue is full, ``consolidate`` flushes
  the backlog (preserving order) and applies the job synchronously (backpressure).
- The live working-memory slots are mutated every cycle, so each job snapshots exactly
  the fields :meth:`LongTermGraph.consolidate` reads (appearance/seen_count/kind/
  position/affect/id) into immutable carriers at enqueue time -- the worker never reads
  a slot object that has since changed.
- A single FIFO worker preserves cycle order (node id coining, ``last_cycle``, edge
  counts are all order-sensitive); the worker runs ``consolidate`` under the graph's own
  lock, so reads next cycle never see a torn graph.
- ``clear`` / ``backup_to`` / ``restore_from`` ``flush()`` first, so a snapshot or wipe
  never races a pending consolidation.
- Visibility: while async is ON a node discovered this cycle becomes matchable once the
  worker drains it (sub-ms to a few ms later). Stage-3 re-identification is associative
  and reads the graph on the *next* cycle, so this lag is immaterial.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decadic.memory.semantic_graph import DEFAULT_MATCH_THRESHOLD, LongTermGraph

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass(frozen=True)
class _SlotSnapshot:
    """Immutable copy of the slot fields ``LongTermGraph.consolidate`` consumes.

    Captured on the cycle thread at enqueue time so the background worker is never
    exposed to a working-memory slot that has been refreshed/decayed since.
    """

    entity_id: str
    kind: str
    position: list[float] | None
    affective_weight: float
    seen_count: int
    appearance: list[float] | None
    confidence: float
    kind_hint: str
    property_evidence: dict[str, Any]


def _snapshot_slot(s: Any) -> _SlotSnapshot:
    pos = getattr(s, "position", None)
    app = getattr(s, "appearance", None)
    return _SlotSnapshot(
        entity_id=str(getattr(s, "entity_id", "")),
        kind=str(getattr(s, "kind", "unknown")),
        position=list(pos) if pos is not None else None,
        affective_weight=float(getattr(s, "affective_weight", 0.0) or 0.0),
        seen_count=int(getattr(s, "seen_count", 0) or 0),
        # consolidate's ``if not app`` guard requires a list/None (never a raw
        # array), which MemorySlot.appearance already is; copy to decouple.
        appearance=list(app) if app else None,
        confidence=float(getattr(s, "confidence", 1.0) or 0.0),
        kind_hint=str(getattr(s, "kind_hint", "object")),
        property_evidence=dict(getattr(s, "property_evidence", {}) or {}),
    )


class WriteBehindLongTermGraph(LongTermGraph):
    """:class:`LongTermGraph` whose ``consolidate`` can run on a background thread."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        max_queue: int = 4096,
        enabled: bool = True,
    ) -> None:
        super().__init__(db_path, match_threshold=match_threshold)
        self._max_queue = max(1, int(max_queue))
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._async_enabled = False
        if enabled:
            self.set_async(True)

    @property
    def async_enabled(self) -> bool:
        return self._async_enabled

    def set_async(self, enabled: bool) -> None:
        """Turn background consolidation on/off (drains + stops the worker when off)."""
        enabled = bool(enabled)
        if enabled == self._async_enabled:
            return
        if enabled:
            self._queue = queue.Queue(maxsize=self._max_queue)
            self._async_enabled = True
            self._worker = threading.Thread(
                target=self._drain_loop,
                args=(self._queue,),
                name="ltm-write-behind",
                daemon=True,
            )
            self._worker.start()
        else:
            # Flip to synchronous first so any subsequent consolidate runs inline,
            # then drain the backlog (in order) and retire the worker.
            self._async_enabled = False
            old_queue, old_worker = self._queue, self._worker
            self._queue, self._worker = None, None
            if old_queue is not None:
                old_queue.join()
                old_queue.put(_SENTINEL)
            if old_worker is not None:
                old_worker.join(timeout=2.0)

    def consolidate(
        self,
        slots: Any,
        affect: dict[str, float] | None = None,
        *,
        cycle: int = 0,
        min_seen: int = 2,
        property_update: bool = True,
        relationship_update: bool = True,
    ) -> list[str]:
        """Enqueue consolidation for the background worker when async; else inline.

        Returns the consolidated node ids synchronously; an empty list when deferred
        (stage 10 ignores the return value, so the deferral is invisible to callers).
        """
        q = self._queue if self._async_enabled else None
        if q is None:
            return super().consolidate(
                slots,
                affect,
                cycle=cycle,
                min_seen=min_seen,
                property_update=property_update,
                relationship_update=relationship_update,
            )
        job = (
            [_snapshot_slot(s) for s in slots],
            dict(affect) if affect else None,
            int(cycle),
            int(min_seen),
            bool(property_update),
            bool(relationship_update),
        )
        try:
            q.put_nowait(job)
        except queue.Full:
            # Never drop a consolidation and never reorder: drain the backlog first,
            # then apply this one inline. Backpressure is the safety valve, not the
            # expected path.
            logger.warning("ltm write-behind queue full; consolidating synchronously")
            self.flush()
            return super().consolidate(
                slots,
                affect,
                cycle=cycle,
                min_seen=min_seen,
                property_update=property_update,
                relationship_update=relationship_update,
            )
        return []

    def _drain_loop(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                if item is _SENTINEL:
                    return
                snaps, affect, cycle, min_seen, property_update, relationship_update = item
                try:
                    super().consolidate(
                        snaps,
                        affect,
                        cycle=cycle,
                        min_seen=min_seen,
                        property_update=property_update,
                        relationship_update=relationship_update,
                    )
                except Exception:  # pragma: no cover - persistence must not kill worker
                    logger.exception("ltm write-behind consolidate failed")
            finally:
                q.task_done()

    def flush(self) -> None:
        """Block until all queued consolidations have been applied (no-op when sync)."""
        q = self._queue
        if q is not None:
            q.join()

    def clear(self) -> None:
        self.flush()
        super().clear()

    def backup_to(self, path: Path) -> None:
        self.flush()
        super().backup_to(path)

    def restore_from(self, path: Path) -> None:
        self.flush()
        super().restore_from(path)

    def close(self) -> None:
        """Drain and stop the worker; the graph stays usable in synchronous mode."""
        self.set_async(False)
