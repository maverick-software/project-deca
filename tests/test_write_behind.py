"""Write-behind episodic store: async persistence with consistent, lossless reads.

The cycle hands the SQLite write to a background worker, but reads must still be
correct after a ``flush``, search must rank the same as the synchronous store, no
record may be dropped under backpressure, and ``clear`` must flush before wiping.
All CPU-only and fast (no GPU / no HF).
"""

from __future__ import annotations

import numpy as np

from decadic.memory.embeddings import EMBEDDING_DIM
from decadic.memory.episodic_store import EpisodicRecord
from decadic.memory.write_behind import WriteBehindEpisodicStore


def _rec(i: int) -> EpisodicRecord:
    emb = list(np.random.RandomState(i).randn(EMBEDDING_DIM).astype(float))
    return EpisodicRecord(cycle_index=i, summary={"i": i}, salience=0.9, embedding=emb)


def test_write_behind_persists_after_flush(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "ep.db")
    try:
        for i in range(25):
            store.append(_rec(i))
        store.flush()
        recent = store.recent(limit=5)
        assert len(recent) == 5
        assert recent[0]["cycle_index"] == 24  # newest first, written in order
    finally:
        store.close()


def test_write_behind_search_ranks_exact_match_first(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "ep.db")
    try:
        for i in range(30):
            store.append(_rec(i))
        store.flush()
        query = np.random.RandomState(7).randn(EMBEDDING_DIM)
        hits = store.search_similar(query, top_k=3)
        assert hits
        assert hits[0]["cycle_index"] == 7  # same seed -> cosine ~1.0
    finally:
        store.close()


def test_write_behind_clear_flushes_then_wipes(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "ep.db")
    try:
        for i in range(10):
            store.append(_rec(i))
        store.clear()
        assert store.recent() == []
    finally:
        store.close()


def test_write_behind_backpressure_never_drops(tmp_path):
    # A tiny queue forces the synchronous fallback; every record must still land.
    store = WriteBehindEpisodicStore(tmp_path / "ep.db", max_queue=2)
    try:
        for i in range(50):
            store.append(_rec(i))
        store.flush()
        assert len(store.recent(limit=200)) == 50
    finally:
        store.close()


def test_born_disabled_has_no_worker_and_writes_synchronously(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "ep.db", enabled=False)
    try:
        assert store.async_enabled is False
        assert store._worker is None  # no thread spawned when async is off
        store.append(_rec(0))
        # Synchronous write -> immediately queryable without flush.
        assert len(store.recent()) == 1
    finally:
        store.close()


def test_live_toggle_off_on_preserves_records(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "ep.db", enabled=True)
    try:
        for i in range(10):
            store.append(_rec(i))
        store.set_async(False)  # drains the backlog, retires the worker
        assert store.async_enabled is False
        assert store._worker is None
        assert len(store.recent(limit=100)) == 10  # nothing lost on the way down
        for i in range(10, 20):  # now synchronous
            store.append(_rec(i))
        assert len(store.recent(limit=100)) == 20
        store.set_async(True)  # restart the worker
        assert store.async_enabled is True
        for i in range(20, 30):
            store.append(_rec(i))
        store.flush()
        assert len(store.recent(limit=100)) == 30
    finally:
        store.close()
