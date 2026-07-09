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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decadic import config as C
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
    entity_role: str
    precision: float
    provisional: bool
    evidence_count: float
    contradiction_pressure: float
    event_links: list[str]
    relationship_links: list[str]
    scene_entity_id: str | None
    property_evidence: dict[str, Any]
    attention_focused: bool = False  # WS-SYM 2.2: joint-attention gate for binding


@dataclass(frozen=True)
class LtmConsolidationJob:
    slots: list[_SlotSnapshot]
    all_slots: list[_SlotSnapshot]
    events: list[dict[str, Any]]
    scene_relationships: list[dict[str, Any]]
    cycle: int
    min_seen: int
    property_update: bool
    relationship_update: bool
    symbol_code: int | None = None  # WS-SYM 2.2: FSQ code to bind this cycle


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
        entity_role=str(getattr(s, "entity_role", "compact_entity")),
        precision=float(getattr(s, "precision", getattr(s, "confidence", 0.0)) or 0.0),
        provisional=bool(getattr(s, "provisional", True)),
        evidence_count=float(getattr(s, "evidence_count", 0.0) or 0.0),
        contradiction_pressure=float(getattr(s, "contradiction_pressure", 0.0) or 0.0),
        event_links=list(getattr(s, "event_links", []) or []),
        relationship_links=list(getattr(s, "relationship_links", []) or []),
        scene_entity_id=(
            str(getattr(s, "scene_entity_id", ""))
            if getattr(s, "scene_entity_id", None)
            else None
        ),
        property_evidence=dict(getattr(s, "property_evidence", {}) or {}),
        attention_focused=bool(getattr(s, "attention_focused", False)),
    )


def _safe_dict_list(items: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(items or [])[: limit or 10_000]:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


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
        self._jobs_enqueued = 0
        self._jobs_completed = 0
        self._sync_fallbacks = 0
        self._semantic_jobs_skipped_by_interval = 0
        self._last_worker_ms = 0.0
        # WS-FREEZE: write-behind heartbeat. _wb_in_job is True only while the
        # worker holds the graph RLock for a consolidation job -- if the
        # cognitive loop freezes while this stays True, that's hypothesis H1.
        self._wb_in_job = False
        self._wb_job_start_s: float | None = None
        self._wb_hb_s = time.monotonic()
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
            self._jobs_enqueued += 1
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

    def enqueue_consolidation_job(
        self,
        slots: Any,
        *,
        all_slots: Any | None = None,
        events: list[dict[str, Any]] | None = None,
        scene_relationships: list[dict[str, Any]] | None = None,
        cycle: int = 0,
        min_seen: int = 2,
        property_update: bool = True,
        relationship_update: bool = True,
        symbol_code: int | None = None,
    ) -> dict[str, Any]:
        """Queue the full Stage-10 LTM job, including semantic evidence.

        Stage 10 should call this method instead of calling ``consolidate``,
        ``bump_edge`` and ``record_semantic_evidence`` itself. The job snapshots
        all live WM data immediately; the worker never reads mutable slots.
        """
        job = LtmConsolidationJob(
            slots=[_snapshot_slot(s) for s in slots],
            all_slots=[_snapshot_slot(s) for s in (all_slots if all_slots is not None else slots)],
            events=_safe_dict_list(events, limit=64),
            scene_relationships=_safe_dict_list(scene_relationships),
            cycle=int(cycle),
            min_seen=int(min_seen),
            property_update=bool(property_update),
            relationship_update=bool(relationship_update),
            symbol_code=(int(symbol_code) if symbol_code is not None else None),
        )
        q = self._queue if self._async_enabled else None
        if q is None:
            self._sync_fallbacks += 1
            return self._apply_consolidation_job(job)
        try:
            q.put_nowait(job)
            self._jobs_enqueued += 1
        except queue.Full:
            logger.warning("ltm consolidation queue full; flushing and applying synchronously")
            self.flush()
            self._sync_fallbacks += 1
            return self._apply_consolidation_job(job)
        return {
            "status": "queued_consolidation",
            "queued": True,
            "accepted_ids": [],
            "identity_refresh": False,
            "property_update": bool(property_update and job.slots),
            "relationship_update": bool(relationship_update and job.slots),
            "relationship_updates_skipped": 0 if relationship_update else 1,
            "semantic_update": {},
            **self.cached_belief_stats(),
        }

    def _apply_consolidation_job(self, job: LtmConsolidationJob) -> dict[str, Any]:
        started = time.perf_counter()
        self._wb_job_start_s = time.monotonic()  # WS-FREEZE: RLock hold begins
        self._wb_in_job = True
        with self.write_batch():
            ids = super().consolidate(
                job.slots,
                cycle=job.cycle,
                min_seen=job.min_seen,
                property_update=job.property_update,
                relationship_update=job.relationship_update,
            )
            scene_edges = 0
            if job.relationship_update and ids:
                scene_to_ltm: dict[str, str] = {}
                for slot, node_id in zip(job.slots, ids):
                    if slot.scene_entity_id:
                        scene_to_ltm[str(slot.scene_entity_id)] = str(node_id)
                rels = sorted(
                    job.scene_relationships,
                    key=lambda r: float(r.get("confidence", 0.0) or 0.0),
                    reverse=True,
                )[: C.ltm_scene_edge_max_per_job()]
                for rel in rels:
                    src = scene_to_ltm.get(str(rel.get("src")))
                    dst = scene_to_ltm.get(str(rel.get("dst")))
                    kind = str(rel.get("kind", "scene_relation"))
                    if src and dst and src != dst:
                        super().bump_edge(
                            src,
                            dst,
                            kind=f"scene_{kind}",
                            weight=float(rel.get("confidence", 1.0) or 1.0),
                            cycle=job.cycle,
                        )
                        scene_edges += 1
            semantic_update: dict[str, Any] = {}
            semantic_allowed = (
                job.cycle % C.ltm_semantic_evidence_interval() == 0
                or bool(job.events)
            )
            if job.all_slots and semantic_allowed:
                semantic_update = super().record_semantic_evidence(
                    job.all_slots,
                    events=job.events,
                    scene_relationships=job.scene_relationships,
                    cycle=job.cycle,
                    promoted_ids=list(ids),
                    symbol_code=job.symbol_code,
                )
            elif job.all_slots:
                self._semantic_jobs_skipped_by_interval += 1
            retention = super().prune_retention(cycle=job.cycle)
            stats = super().belief_stats()
        self._wb_in_job = False  # WS-FREEZE: RLock released
        self._wb_hb_s = time.monotonic()
        self._jobs_completed += 1
        self._last_worker_ms = (time.perf_counter() - started) * 1000.0
        status = "promoted_entity" if ids else "recorded_provisional_evidence"
        if not ids and semantic_update.get("values", 0):
            status = "updated_value"
        elif not ids and semantic_update.get("conclusions", 0):
            status = "formed_conclusion"
        elif not ids and semantic_update.get("correlations", 0):
            status = "strengthened_correlation"
        elif not ids and semantic_update.get("relationships", 0):
            status = "strengthened_relationship"
        return {
            "status": status,
            "queued": False,
            "accepted_ids": list(ids),
            "identity_refresh": bool(ids),
            "property_update": bool(job.property_update and ids),
            "relationship_update": bool(job.relationship_update and ids),
            "relationship_updates_skipped": 0 if job.relationship_update else 1,
            "scene_edges": scene_edges,
            "semantic_update": semantic_update,
            "retention": retention,
            **stats,
        }

    def record_semantic_evidence(
        self,
        slots: Any,
        *,
        events: list[dict[str, Any]] | None = None,
        scene_relationships: list[dict[str, Any]] | None = None,
        cycle: int = 0,
        promoted_ids: list[str] | None = None,
        symbol_code: int | None = None,
    ) -> dict[str, Any]:
        """Record provisional semantic evidence from immutable slot snapshots.

        This path is intentionally synchronous and lightweight: it updates the
        in-memory semantic graph immediately so the dashboard and subsequent
        cycles can see provisional entities from moment one. SQLite writes still
        happen through the graph lock, preserving the same serialization rules.
        """
        snaps = [_snapshot_slot(s) for s in slots]
        return super().record_semantic_evidence(
            snaps,
            events=events,
            scene_relationships=scene_relationships,
            cycle=cycle,
            promoted_ids=promoted_ids,
            symbol_code=symbol_code,
        )

    def _drain_loop(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                if item is _SENTINEL:
                    return
                try:
                    if isinstance(item, LtmConsolidationJob):
                        self._apply_consolidation_job(item)
                    else:
                        started = time.perf_counter()
                        snaps, affect, cycle, min_seen, property_update, relationship_update = item
                        self._wb_job_start_s = time.monotonic()  # WS-FREEZE
                        self._wb_in_job = True
                        with self.write_batch():
                            super().consolidate(
                                snaps,
                                affect,
                                cycle=cycle,
                                min_seen=min_seen,
                                property_update=property_update,
                                relationship_update=relationship_update,
                            )
                            super().prune_retention(cycle=cycle)
                        self._wb_in_job = False  # WS-FREEZE
                        self._wb_hb_s = time.monotonic()
                        self._jobs_completed += 1
                        self._last_worker_ms = (time.perf_counter() - started) * 1000.0
                except Exception:  # pragma: no cover - persistence must not kill worker
                    self._wb_in_job = False  # WS-FREEZE: never leak a held-flag on error
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

    def runtime_metrics(self) -> dict[str, Any]:
        q = self._queue
        match_stats = self.match_cache_stats()
        return {
            "ltm_consolidation_queue_depth": int(q.qsize()) if q is not None else 0,
            "ltm_consolidation_queue_max": int(self._max_queue),
            "ltm_consolidation_worker_ms": float(self._last_worker_ms),
            "ltm_consolidation_jobs_enqueued": int(self._jobs_enqueued),
            "ltm_consolidation_jobs_completed": int(self._jobs_completed),
            "ltm_consolidation_sync_fallbacks": int(self._sync_fallbacks),
            "ltm_semantic_jobs_skipped_by_interval": int(
                self._semantic_jobs_skipped_by_interval
            ),
            "ltm_match_ms": float(match_stats.get("last_ms", 0.0)),
            "ltm_match_cache_size": int(match_stats.get("size", 0)),
            "ltm_match_cache_hits": int(match_stats.get("hits", 0)),
            "ltm_match_cache_misses": int(match_stats.get("misses", 0)),
            "ltm_match_cache_enabled": bool(match_stats.get("enabled", False)),
            "ltm_write_batch_size_last": 1,
            # WS-SYM 3.2: symbol recall telemetry (counters live on the base
            # graph; record_semantic_evidence runs there via super()).
            "symbol_recall_queries": int(getattr(self, "_symbol_recall_queries", 0)),
            "symbol_recall_hits": int(getattr(self, "_symbol_recall_hits", 0)),
            "symbol_recall_hit_rate": (
                float(getattr(self, "_symbol_recall_hits", 0))
                / float(getattr(self, "_symbol_recall_queries", 0))
                if getattr(self, "_symbol_recall_queries", 0)
                else 0.0
            ),
            # WS-SYM 5.0: drift proxy -- rate an entity's top-evidence code flips.
            "symbol_binding_updates": int(getattr(self, "_symbol_binding_updates", 0)),
            "symbol_binding_flips": int(getattr(self, "_symbol_binding_flips", 0)),
            "symbol_binding_churn": (
                float(getattr(self, "_symbol_binding_flips", 0))
                / float(getattr(self, "_symbol_binding_updates", 0))
                if getattr(self, "_symbol_binding_updates", 0)
                else 0.0
            ),
            **self.persistence_metrics(),
        }

    def close(self) -> None:
        """Drain and stop the worker; the graph stays usable in synchronous mode."""
        self.set_async(False)
