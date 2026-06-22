from __future__ import annotations

import json
import sqlite3

import numpy as np

from decadic.memory.embeddings import EMBEDDING_DIM
from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore
from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
from decadic.memory.semantic_graph import LongTermGraph
from decadic.memory.write_behind import WriteBehindEpisodicStore
from decadic.state.working_memory import MemorySlot


def _embedding(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(float).tolist()


def test_sqlite_connection_uses_wal_and_normal_sync(tmp_path):
    store = EpisodicStore(tmp_path / "episodes.sqlite")
    assert store._conn is not None
    journal = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    sync = int(store._conn.execute("PRAGMA synchronous").fetchone()[0])
    assert str(journal).lower() == "wal"
    assert sync == 1  # NORMAL


def test_episodic_writes_blob_and_reads_legacy_json(tmp_path):
    db = tmp_path / "episodes.sqlite"
    store = EpisodicStore(db)
    store.append(EpisodicRecord(cycle_index=1, summary={"a": 1}, salience=0.5, embedding=_embedding(1)))
    assert store._conn is not None
    emb_json, emb_blob = store._conn.execute(
        "SELECT embedding_json, embedding_blob FROM episodes WHERE cycle_index = 1"
    ).fetchone()
    assert emb_json is None
    assert isinstance(emb_blob, bytes)

    legacy = sqlite3.connect(str(tmp_path / "legacy.sqlite"))
    try:
        legacy.execute(
            """
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_index INTEGER NOT NULL,
                salience REAL NOT NULL,
                summary_json TEXT NOT NULL,
                embedding_json TEXT
            )
            """
        )
        legacy.execute(
            "INSERT INTO episodes (cycle_index, salience, summary_json, embedding_json) VALUES (?, ?, ?, ?)",
            (2, 0.9, json.dumps({"legacy": True}), json.dumps(_embedding(2))),
        )
        legacy.commit()
    finally:
        legacy.close()
    other = EpisodicStore(tmp_path / "restored.sqlite")
    other.restore_from(tmp_path / "legacy.sqlite")
    rows = other.recent(limit=10)
    assert rows[0]["cycle_index"] == 2
    assert len(rows[0]["embedding"]) == EMBEDDING_DIM


def test_episodic_write_behind_batches_commits(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_EPISODIC_WRITE_BATCH_SIZE", "8")
    monkeypatch.setenv("DECADIC_EPISODIC_WRITE_BATCH_MS", "1")
    store = WriteBehindEpisodicStore(tmp_path / "episodes.sqlite", enabled=True)
    try:
        before = store.persistence_metrics()["sqlite_batch_commit_count"]
        for i in range(5):
            store.append(
                EpisodicRecord(cycle_index=i, summary={"i": i}, salience=0.1, embedding=_embedding(i))
            )
        store.flush()
        metrics = store.persistence_metrics()
        assert metrics["episodic_db_rows"] == 5
        assert metrics["sqlite_batch_commit_count"] == before + 1
        assert metrics["episodic_write_batch_size_last"] == 5
    finally:
        store.close()


def test_episodic_db_retention_keeps_recent_and_salient(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RECENT_CAP", "3")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_SALIENT_CAP", "1")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_INTERVAL_WRITES", "1")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_BATCH", "20")
    store = EpisodicStore(tmp_path / "episodes.sqlite")
    store.append(EpisodicRecord(cycle_index=1, summary={"old": "salient"}, salience=0.99, embedding=_embedding(99)))
    for i in range(2, 12):
        store.append(EpisodicRecord(cycle_index=i, summary={"i": i}, salience=0.01, embedding=_embedding(i)))
    cycles = sorted(row["cycle_index"] for row in store.recent(limit=20))
    assert cycles == [1, 9, 10, 11]
    assert store.persistence_metrics()["episodic_db_pruned_rows"] > 0


def test_ltm_job_batches_to_one_commit_and_uses_blob(tmp_path):
    graph = WriteBehindLongTermGraph(tmp_path / "ltm.sqlite", enabled=False)
    slot = MemorySlot(
        entity_id="obj",
        appearance=[1.0, 0.0],
        seen_count=3,
        confidence=1.0,
        precision=1.0,
        property_evidence={"compactness": 0.7},
    )
    before = graph.persistence_metrics()["sqlite_batch_commit_count"]
    graph.enqueue_consolidation_job([slot], all_slots=[slot], events=[{"type": "contact", "intensity": 0.5}], cycle=4)
    metrics = graph.persistence_metrics()
    assert metrics["sqlite_batch_commit_count"] == before + 1
    assert graph._conn is not None
    appearance_json, appearance_blob = graph._conn.execute(
        "SELECT appearance_json, appearance_blob FROM nodes LIMIT 1"
    ).fetchone()
    assert appearance_json is None
    assert isinstance(appearance_blob, bytes)


def test_ltm_retention_keeps_belief_nodes_and_promoted_values(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_LTM_MAX_NODES", "1")
    monkeypatch.setenv("DECADIC_LTM_PRUNE_STALE_CYCLES", "0")
    monkeypatch.setenv("DECADIC_LTM_PRUNE_BATCH", "10")
    monkeypatch.setenv("DECADIC_LTM_MAX_SEMANTIC_RECORDS", "1")
    graph = LongTermGraph(tmp_path / "ltm.sqlite")
    protected = graph.upsert_node([1.0, 0.0], cycle=1)
    graph.upsert_property_beliefs(protected, {"compactness": 0.8}, cycle=1)
    stale = graph.upsert_node([0.0, 1.0], cycle=1)
    with graph._lock:
        graph._nodes[stale]["salience"] = 0.0
        graph._nodes[stale]["seen_count"] = 1
        graph._semantic["value"]["value-keep"] = {
            "id": "value-keep",
            "payload": {"context": "risk_context"},
            "evidence_count": 5.0,
            "confidence": 1.0,
            "first_cycle": 1,
            "last_cycle": 1,
            "promoted": True,
        }
        graph._semantic["entity"]["entity-drop"] = {
            "id": "entity-drop",
            "payload": {},
            "evidence_count": 0.1,
            "confidence": 0.0,
            "first_cycle": 1,
            "last_cycle": 1,
            "promoted": False,
        }
    pruned = graph.prune_retention(cycle=10)
    assert pruned["nodes"] == 1
    assert protected in graph._nodes
    assert stale not in graph._nodes
    assert "value-keep" in graph._semantic["value"]
