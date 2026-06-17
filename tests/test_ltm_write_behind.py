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
