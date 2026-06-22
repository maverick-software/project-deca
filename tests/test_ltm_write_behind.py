"""Write-behind long-term graph: async consolidation with lossless, consistent reads.

Stage 10 hands the WM->LTM consolidation (in-memory upsert + SQLite commit) to a
background worker, but: the graph state after a ``flush`` must match the synchronous
graph exactly (parity), nothing may be lost under backpressure, cycle order must be
preserved, ``clear`` must flush before wiping, and a node must be re-identifiable once
drained. All CPU-only and fast (no GPU / no HF).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
from decadic.memory.semantic_graph import LongTermGraph


@dataclass
class _Slot:
    """Minimal MemorySlot stand-in carrying the fields consolidate reads."""

    entity_id: str
    kind: str
    appearance: list[float]
    seen_count: int = 3
    affective_weight: float = 0.0
    position: list[float] | None = field(default=None)


def _slots(seed: int) -> list[_Slot]:
    # Stable, distinct appearance fingerprints so the same entity re-identifies
    # across cycles (no new node coined) -> deterministic node/edge counts.
    return [
        _Slot("rock", "rock", [1.0, 0.0, 0.0, 0.25]),
        _Slot("tree", "tree", [0.0, 1.0, 0.0, 0.25]),
    ]


def _consolidate_runs(graph) -> None:
    for cycle in range(6):
        graph.consolidate(_slots(cycle), cycle=cycle, min_seen=2)


def test_async_counts_match_sync_after_flush(tmp_path):
    sync = LongTermGraph(tmp_path / "sync.db")
    _consolidate_runs(sync)

    graph = WriteBehindLongTermGraph(tmp_path / "async.db", enabled=True)
    try:
        assert graph.async_enabled is True
        _consolidate_runs(graph)
        graph.flush()
        assert graph.counts() == sync.counts()  # same nodes + edges (parity)
        # Re-identification works once drained: the rock's fingerprint matches its node.
        assert graph.match([1.0, 0.0, 0.0, 0.25]) is not None
    finally:
        graph.close()


def test_async_persists_after_flush_and_reopen(tmp_path):
    db = tmp_path / "g.db"
    graph = WriteBehindLongTermGraph(db, enabled=True)
    _consolidate_runs(graph)
    graph.flush()
    expected = graph.counts()
    graph.close()
    # A fresh (synchronous) graph over the same file must load the same state -> the
    # background commits really hit disk; nothing buffered-and-lost.
    reopened = LongTermGraph(db)
    assert reopened.counts() == expected
    assert expected[0] >= 2  # at least the two distinct entities were committed


def test_clear_flushes_then_wipes(tmp_path):
    graph = WriteBehindLongTermGraph(tmp_path / "g.db", enabled=True)
    try:
        _consolidate_runs(graph)
        graph.clear()  # must flush the backlog first, then wipe
        assert graph.counts() == (0, 0)
    finally:
        graph.close()


def test_backpressure_never_drops_and_keeps_order(tmp_path):
    # A tiny queue forces the order-preserving synchronous fallback.
    sync = LongTermGraph(tmp_path / "sync.db")
    _consolidate_runs(sync)
    graph = WriteBehindLongTermGraph(tmp_path / "async.db", max_queue=1, enabled=True)
    try:
        _consolidate_runs(graph)
        graph.flush()
        assert graph.counts() == sync.counts()  # identical despite backpressure
    finally:
        graph.close()


def test_born_disabled_has_no_worker_and_consolidates_synchronously(tmp_path):
    graph = WriteBehindLongTermGraph(tmp_path / "g.db", enabled=False)
    try:
        assert graph.async_enabled is False
        assert graph._worker is None  # no thread spawned when async is off
        graph.consolidate(_slots(0), cycle=0, min_seen=2)
        # Synchronous: immediately queryable without a flush.
        assert graph.counts()[0] == 2
    finally:
        graph.close()


def test_live_toggle_off_on_preserves_state(tmp_path):
    sync = LongTermGraph(tmp_path / "sync.db")
    _consolidate_runs(sync)

    graph = WriteBehindLongTermGraph(tmp_path / "async.db", enabled=True)
    try:
        graph.consolidate(_slots(0), cycle=0, min_seen=2)
        graph.set_async(False)  # drains the backlog, retires the worker
        assert graph.async_enabled is False
        assert graph._worker is None
        graph.consolidate(_slots(1), cycle=1, min_seen=2)  # now synchronous
        graph.set_async(True)  # restart the worker
        assert graph.async_enabled is True
        for cycle in range(2, 6):
            graph.consolidate(_slots(cycle), cycle=cycle, min_seen=2)
        graph.flush()
        assert graph.counts() == sync.counts()  # nothing lost across the toggles
    finally:
        graph.close()


def test_full_consolidation_job_updates_semantics_and_scene_edges(tmp_path):
    slots = _slots(0)
    for idx, slot in enumerate(slots):
        slot.position = [float(idx), 0.0, 0.0]
        slot.scene_entity_id = f"scene-{idx}"  # type: ignore[attr-defined]
        slot.property_evidence = {"compactness": 0.7}  # type: ignore[attr-defined]
    graph = WriteBehindLongTermGraph(tmp_path / "async.db", enabled=True)
    try:
        report = graph.enqueue_consolidation_job(
            slots,
            all_slots=slots,
            events=[{"type": "contact", "intensity": 0.5}],
            scene_relationships=[
                {"src": "scene-0", "dst": "scene-1", "kind": "near", "confidence": 0.9}
            ],
            cycle=4,
            min_seen=2,
        )
        assert report["status"] == "queued_consolidation"
        graph.flush()
        snap = graph.snapshot()
        assert snap["total_nodes"] == 2
        assert any(e["kind"] == "scene_near" for e in snap["edges"])
        stats = graph.belief_stats()
        assert stats["total_property_beliefs"] == 2
        assert stats["semantic_entities"] >= 2
        assert graph.runtime_metrics()["ltm_consolidation_jobs_completed"] >= 1
    finally:
        graph.close()


def test_semantic_evidence_interval_throttles_low_salience_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_LTM_SEMANTIC_EVIDENCE_INTERVAL", "10")
    graph = WriteBehindLongTermGraph(tmp_path / "async.db", enabled=True)
    try:
        graph.enqueue_consolidation_job(
            [],
            all_slots=_slots(0),
            events=[],
            scene_relationships=[],
            cycle=3,
            min_seen=2,
        )
        graph.flush()
        assert graph.belief_stats()["semantic_entities"] == 0
        assert graph.runtime_metrics()["ltm_semantic_jobs_skipped_by_interval"] == 1
    finally:
        graph.close()
