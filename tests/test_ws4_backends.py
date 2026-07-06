"""WS4 backend seam + LanceDB parity tests (M0.2 layout, M0.3 factory, M1.2, M3.1).

The LanceDB-dependent tests are skipped automatically when the ``lancedb``
extra is not installed, so the default CPU suite stays zero-configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from decadic.memory import factory
from decadic.memory.embeddings import (
    EMBEDDING_DIM,
    PERCEPT_KEY_DIM,
    PERCEPT_KEY_SLICE,
    episode_embedding_from_cycle,
    perceptual_key,
)
from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore
from decadic.state.state_bus import StateBus


# --------------------------------------------------------------------------
# M0.2 -- embedding layout freeze
# --------------------------------------------------------------------------


def test_embedding_layout_frozen():
    assert EMBEDDING_DIM == 80
    assert PERCEPT_KEY_DIM == 16
    assert PERCEPT_KEY_SLICE == slice(64, 80)

    rng = np.random.default_rng(7)
    sb = StateBus()
    sb.narrative_emb = rng.standard_normal(48).astype(np.float32)
    sb.emotion_physio = rng.standard_normal(32).astype(np.float32)
    sb.metacognition = rng.standard_normal(24).astype(np.float32)
    z5 = rng.standard_normal(16).astype(np.float32)
    percept = rng.standard_normal(37).astype(np.float32)

    emb = episode_embedding_from_cycle(sb, z5, percept)
    assert emb.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(
        emb[PERCEPT_KEY_SLICE], perceptual_key(percept), rtol=0.0, atol=1e-6
    )
    # No percept -> the perceptual tail is exactly zeros (the parity guarantee).
    emb0 = episode_embedding_from_cycle(sb, z5, None)
    assert np.all(emb0[PERCEPT_KEY_SLICE] == 0.0)


# --------------------------------------------------------------------------
# M0.3 -- backend factory
# --------------------------------------------------------------------------


def test_factory_defaults_are_lancedb_kuzu(monkeypatch):
    """WS4-M5 cutover: with no env vars set the defaults are lancedb + kuzu."""
    pytest.importorskip("lancedb")
    pytest.importorskip("kuzu")
    monkeypatch.delenv("DECADIC_MEMORY_BACKEND", raising=False)
    monkeypatch.delenv("DECADIC_GRAPH_BACKEND", raising=False)
    from decadic.memory.kuzu_graph import (
        KuzuLongTermGraph,
        WriteBehindKuzuLongTermGraph,
    )
    from decadic.memory.lancedb_store import LanceEpisodicStore

    store = factory.make_episodic_store(None)
    try:
        assert type(store) is LanceEpisodicStore
    finally:
        store.close()
    graph = factory.make_semantic_graph(None)
    try:
        assert type(graph) is KuzuLongTermGraph
    finally:
        graph.close()

    rt_store = factory.make_runtime_episodic_store(None, enabled=False)
    try:
        assert type(rt_store) is LanceEpisodicStore
        assert rt_store.async_enabled is False
    finally:
        rt_store.close()
    rt_graph = factory.make_runtime_ltm_graph(None, enabled=False)
    try:
        assert isinstance(rt_graph, WriteBehindKuzuLongTermGraph)
    finally:
        rt_graph.close()


def test_factory_sqlite_legacy(monkeypatch):
    """sqlite stays fully supported as the explicit legacy backend value."""
    monkeypatch.setenv("DECADIC_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("DECADIC_GRAPH_BACKEND", "sqlite")
    store = factory.make_episodic_store(None)
    assert type(store) is EpisodicStore
    from decadic.memory.semantic_graph import LongTermGraph

    graph = factory.make_semantic_graph(None)
    assert type(graph) is LongTermGraph

    from decadic.memory.ltm_write_behind import WriteBehindLongTermGraph
    from decadic.memory.write_behind import WriteBehindEpisodicStore

    rt_store = factory.make_runtime_episodic_store(None, enabled=False)
    assert type(rt_store) is WriteBehindEpisodicStore
    rt_graph = factory.make_runtime_ltm_graph(None, enabled=False)
    assert type(rt_graph) is WriteBehindLongTermGraph


def test_factory_lancedb_backend(monkeypatch):
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_MEMORY_BACKEND", "lancedb")
    from decadic.memory.lancedb_store import LanceEpisodicStore

    store = factory.make_episodic_store(None)
    try:
        assert isinstance(store, LanceEpisodicStore)
    finally:
        store.close()
    rt_store = factory.make_runtime_episodic_store(None, enabled=True)
    try:
        assert isinstance(rt_store, LanceEpisodicStore)
        assert rt_store.async_enabled is True
    finally:
        rt_store.close()


def test_factory_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("DECADIC_MEMORY_BACKEND", "duckdb")
    with pytest.raises(ValueError, match="DECADIC_MEMORY_BACKEND"):
        factory.make_episodic_store(None)
    monkeypatch.setenv("DECADIC_GRAPH_BACKEND", "neo4j")
    with pytest.raises(ValueError, match="DECADIC_GRAPH_BACKEND"):
        factory.make_semantic_graph(None)


def test_factory_kuzu_backend(monkeypatch):
    # WS4-M2 replaced the reserved NotImplementedError with the real backend.
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_GRAPH_BACKEND", "kuzu")
    from decadic.memory.kuzu_graph import (
        KuzuLongTermGraph,
        WriteBehindKuzuLongTermGraph,
    )

    graph = factory.make_semantic_graph(None)
    try:
        assert type(graph) is KuzuLongTermGraph
    finally:
        graph.close()
    rt_graph = factory.make_runtime_ltm_graph(None, enabled=False)
    try:
        assert isinstance(rt_graph, WriteBehindKuzuLongTermGraph)
        assert rt_graph.async_enabled is False
    finally:
        rt_graph.close()


# --------------------------------------------------------------------------
# M3.1 -- percept-key search on the sqlite backend (the WS3 novelty rationale)
# --------------------------------------------------------------------------


def test_sqlite_percept_search_restores_novelty_range():
    """Loop-like episodes: internal state drifts, percept key repeats.

    Full-vector similarity is dragged down by the drifting internal 64-d part
    (the WS3 finding: novelty ~ blind), while percept-only similarity correctly
    reports "seen this before" at ~1.0.
    """
    store = EpisodicStore(None)
    key = np.zeros(PERCEPT_KEY_DIM, dtype=np.float32)
    key[0] = 1.0
    for i in range(60):
        internal = np.zeros(64, dtype=np.float32)
        internal[i % 64] = 1.0  # drifting, mutually orthogonal internal states
        emb = np.concatenate([internal, key])
        store.append(
            EpisodicRecord(
                cycle_index=i,
                summary={"i": i},
                salience=0.5,
                embedding=[float(x) for x in emb],
            )
        )

    q_internal = np.zeros(64, dtype=np.float32)
    q_internal[63] = 1.0  # a never-stored internal direction
    q = np.concatenate([q_internal, key])

    full_hits = store.search_similar(q, top_k=5)
    percept_hits = store.search_similar_percept(key, top_k=5)

    assert len(full_hits) == 5 and len(percept_hits) == 5
    assert store.last_best_percept_similarity is not None
    assert store.last_best_percept_similarity > 0.99  # repeat found
    assert store.last_best_similarity is not None
    assert store.last_best_similarity < 0.75  # internal drift swamps the key
    assert (
        store.last_best_percept_similarity
        > store.last_best_similarity + 0.2
    )
    for hit in percept_hits:
        assert set(hit) >= {"cycle_index", "salience", "summary", "embedding", "similarity"}
        assert hit["similarity"] > 0.99


def test_sqlite_percept_search_empty_store():
    store = EpisodicStore(None)
    assert store.search_similar_percept(np.ones(PERCEPT_KEY_DIM)) == []
    assert store.last_best_percept_similarity is None


def test_sqlite_percept_search_no_cache_fallback(monkeypatch):
    monkeypatch.setenv("DECADIC_EPISODIC_RECALL_CACHE_ENABLED", "0")
    store = EpisodicStore(None)
    rng = np.random.default_rng(3)
    for i in range(20):
        emb = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        store.append(
            EpisodicRecord(
                cycle_index=i, summary={}, salience=0.5, embedding=emb.tolist()
            )
        )
    key = rng.standard_normal(PERCEPT_KEY_DIM).astype(np.float32)
    hits = store.search_similar_percept(key, top_k=3)
    assert len(hits) == 3
    assert store.last_best_percept_similarity == pytest.approx(
        hits[0]["similarity"], abs=1e-6
    )


# --------------------------------------------------------------------------
# M1.2 -- sqlite vs lancedb parity
# --------------------------------------------------------------------------


def _synthetic_records(n: int = 200, seed: int = 123) -> list[EpisodicRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        internal = rng.standard_normal(64).astype(np.float32)
        key = rng.standard_normal(PERCEPT_KEY_DIM).astype(np.float32)
        key /= max(1e-8, float(np.linalg.norm(key)))  # distinct unit percept keys
        emb = np.concatenate([internal, key]).astype(np.float32)
        records.append(
            EpisodicRecord(
                cycle_index=i,
                summary={"i": i, "tag": f"ep{i}"},
                salience=float(rng.uniform(0.0, 1.0)),
                embedding=[float(x) for x in emb],
            )
        )
    return records


def _make_pair(tmp_path):
    from decadic.memory.lancedb_store import LanceEpisodicStore

    sq = EpisodicStore(tmp_path / "parity.sqlite")
    lz = LanceEpisodicStore(tmp_path / "parity_lance")
    return sq, lz


def _cycles(hits):
    return [int(h["cycle_index"]) for h in hits]


def test_backend_parity_search(tmp_path):
    pytest.importorskip("lancedb")
    sq, lz = _make_pair(tmp_path)
    try:
        records = _synthetic_records(200)
        for r in records:
            sq.append(r)
            lz.append(r)
        lz.flush()

        rng = np.random.default_rng(99)
        for trial in range(10):
            q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            hits_sq = sq.search_similar(q, top_k=5)
            hits_lz = lz.search_similar(q, top_k=5)
            assert _cycles(hits_sq) == _cycles(hits_lz)
            assert sq.last_best_similarity == pytest.approx(
                lz.last_best_similarity, abs=1e-5
            )
            for a, b in zip(hits_sq, hits_lz):
                assert a["similarity"] == pytest.approx(b["similarity"], abs=1e-5)
                assert a["summary"] == b["summary"]

        # Filter parity: min_salience + exclude_cycle.
        q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        base_sq = sq.search_similar(q, top_k=5)
        exclude = int(base_sq[0]["cycle_index"])
        hits_sq = sq.search_similar(q, top_k=5, min_salience=0.5, exclude_cycle=exclude)
        hits_lz = lz.search_similar(q, top_k=5, min_salience=0.5, exclude_cycle=exclude)
        assert _cycles(hits_sq) == _cycles(hits_lz)
        assert exclude not in _cycles(hits_lz)
        assert all(h["salience"] >= 0.5 for h in hits_lz)

        # Percept-key parity: exact repeats and random keys.
        for i in (0, 57, 199):
            key = np.asarray(records[i].embedding, dtype=np.float32)[PERCEPT_KEY_SLICE]
            hits_sq = sq.search_similar_percept(key, top_k=5)
            hits_lz = lz.search_similar_percept(key, top_k=5)
            assert _cycles(hits_sq) == _cycles(hits_lz)
            assert _cycles(hits_sq)[0] == i  # the repeat is found
            assert sq.last_best_percept_similarity == pytest.approx(
                lz.last_best_percept_similarity, abs=1e-5
            )
            assert lz.last_best_percept_similarity == pytest.approx(1.0, abs=1e-4)
        for _ in range(5):
            key = rng.standard_normal(PERCEPT_KEY_DIM).astype(np.float32)
            hits_sq = sq.search_similar_percept(key, top_k=5)
            hits_lz = lz.search_similar_percept(key, top_k=5)
            assert _cycles(hits_sq) == _cycles(hits_lz)
            assert sq.last_best_percept_similarity == pytest.approx(
                lz.last_best_percept_similarity, abs=1e-5
            )

        # retrieval_context_vector parity (mean-pool of the same top hits).
        q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        v_sq = sq.retrieval_context_vector(q, 32, top_k=5)
        v_lz = lz.retrieval_context_vector(q, 32, top_k=5)
        np.testing.assert_allclose(v_sq, v_lz, atol=1e-5)
    finally:
        lz.close()


def test_backend_parity_prune(tmp_path, monkeypatch):
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "1")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RECENT_CAP", "50")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_SALIENT_CAP", "25")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_INTERVAL_WRITES", "10")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_BATCH", "1000")
    sq, lz = _make_pair(tmp_path)
    try:
        for r in _synthetic_records(200, seed=5):
            sq.append(r)
            lz.append(r)
        lz.flush()

        rows_sq = sq.persistence_metrics()["episodic_db_rows"]
        rows_lz = lz.persistence_metrics()["episodic_db_rows"]
        assert rows_sq == rows_lz
        assert rows_lz < 200  # pruning actually happened
        surviving_sq = {r["cycle_index"] for r in sq.recent(limit=500)}
        surviving_lz = {r["cycle_index"] for r in lz.recent(limit=500)}
        assert surviving_sq == surviving_lz
    finally:
        lz.close()


def test_backend_parity_backup_restore(tmp_path):
    pytest.importorskip("lancedb")
    sq, lz = _make_pair(tmp_path)
    try:
        records = _synthetic_records(120, seed=11)
        for r in records:
            sq.append(r)
            lz.append(r)
        lz.flush()

        q = np.random.default_rng(2).standard_normal(EMBEDDING_DIM).astype(np.float32)
        baseline_sq = _cycles(sq.search_similar(q, top_k=5))
        baseline_lz = _cycles(lz.search_similar(q, top_k=5))
        assert baseline_sq == baseline_lz

        sq.backup_to(tmp_path / "bak.sqlite")
        lz.backup_to(tmp_path / "bak_lance")

        # Mutate: episodes engineered to dominate any query direction.
        for j in range(20):
            emb = (q / max(1e-8, float(np.linalg.norm(q)))).astype(np.float32)
            sq.append(
                EpisodicRecord(cycle_index=1000 + j, summary={}, salience=0.9,
                               embedding=emb.tolist())
            )
            lz.append(
                EpisodicRecord(cycle_index=1000 + j, summary={}, salience=0.9,
                               embedding=emb.tolist())
            )
        lz.flush()
        assert _cycles(lz.search_similar(q, top_k=5)) != baseline_lz

        sq.restore_from(tmp_path / "bak.sqlite")
        lz.restore_from(tmp_path / "bak_lance")
        assert _cycles(sq.search_similar(q, top_k=5)) == baseline_sq
        assert _cycles(lz.search_similar(q, top_k=5)) == baseline_lz
        assert (
            sq.persistence_metrics()["episodic_db_rows"]
            == lz.persistence_metrics()["episodic_db_rows"]
            == 120
        )
    finally:
        lz.close()


def test_lancedb_ephemeral_mode_cleanup():
    pytest.importorskip("lancedb")
    from decadic.memory.lancedb_store import LanceEpisodicStore

    store = LanceEpisodicStore(None)
    records = _synthetic_records(10, seed=42)
    for r in records:
        store.append(r)
    hits = store.search_similar(
        np.asarray(records[3].embedding, dtype=np.float32), top_k=1
    )
    assert _cycles(hits) == [3]
    assert store.last_best_similarity == pytest.approx(1.0, abs=1e-5)
    assert store.recent(limit=3)[0]["cycle_index"] == 9
    tmp_dir = store._dir
    assert tmp_dir is None or tmp_dir.exists()
    store.close()
    assert store._dir is None  # temp storage cleaned up


# --------------------------------------------------------------------------
# M5 -- LanceDB full-mirror L1 recall cache (the cutover's performance layer)
# --------------------------------------------------------------------------


def _lance_store(tmp_path, name):
    from decadic.memory.lancedb_store import LanceEpisodicStore

    return LanceEpisodicStore(tmp_path / name)


def test_lance_mirror_write_through_before_flush(tmp_path, monkeypatch):
    """Appends are searchable via the mirror immediately, before any flush."""
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "0")
    store = _lance_store(tmp_path, "wt_lance")
    try:
        store.set_async(True)  # micro-batching on: appends stay pending
        records = _synthetic_records(20, seed=31)
        for r in records[:10]:
            store.append(r)
        assert store._pending  # nothing flushed yet -- this is the point
        q = np.asarray(records[3].embedding, dtype=np.float32)
        assert _cycles(store.search_similar(q, top_k=1)) == [3]
        assert store.last_best_similarity == pytest.approx(1.0, abs=1e-5)
        # Mirror is live now; further appends write through, still unflushed.
        for r in records[10:]:
            store.append(r)
        assert store._pending
        q = np.asarray(records[17].embedding, dtype=np.float32)
        assert _cycles(store.search_similar(q, top_k=1)) == [17]
        key = np.asarray(records[17].embedding, dtype=np.float32)[PERCEPT_KEY_SLICE]
        assert _cycles(store.search_similar_percept(key, top_k=1)) == [17]
        stats = store.recall_cache_stats()
        assert stats["enabled"] is True
        assert stats["size"] == 20
        assert stats["hits"] >= 3
        assert stats["misses"] == 0
    finally:
        store.close()


# --------------------------------------------------------------------------
# WS4B -- graph writes off the critical path (kuzu off-lock flusher)
# --------------------------------------------------------------------------


def test_ws4b_m01_dual_connection_probe(tmp_path):
    """M0.1 GROUND-TRUTH PROBE: two connections on one kuzu Database, one
    writing from a thread while the other reads. PASS = the dedicated-write-
    connection upgrade is available; SKIP(recorded) = the io-lock fallback
    (which WS4B ships with) remains mandatory. Either outcome is a finding."""
    kuzu = pytest.importorskip("kuzu")
    import threading

    db = kuzu.Database(str(tmp_path / "probe_kuzu"))
    c1 = kuzu.Connection(db)
    c2 = kuzu.Connection(db)
    c1.execute("CREATE NODE TABLE T(id INT64, PRIMARY KEY(id))")
    errs: list[Exception] = []

    def _writer() -> None:
        try:
            for i in range(50):
                c2.execute("CREATE (:T {id: $i})", {"i": i})
        except Exception as exc:  # noqa: BLE001 - the probe records, not raises
            errs.append(exc)

    t = threading.Thread(target=_writer)
    t.start()
    read_errs: list[Exception] = []
    for _ in range(20):
        try:
            r = c1.execute("MATCH (t:T) RETURN count(t)")
            r.get_next()
        except Exception as exc:  # noqa: BLE001
            read_errs.append(exc)
    t.join(timeout=30)
    for c in (c1, c2):
        try:
            c.close()
        except Exception:
            pass
    if errs or read_errs:
        pytest.skip(
            f"M0.1 finding: dual-connection NOT safe on this kuzu build "
            f"(write={errs[:1]} read={read_errs[:1]}); io-lock fallback stands"
        )


def _kuzu_graph(tmp_path, name):
    from decadic.memory.kuzu_graph import KuzuLongTermGraph

    return KuzuLongTermGraph(tmp_path / name)


def _rand_apps(n, seed=61):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 16)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_ws4b_offlock_flush_drain_and_reopen(tmp_path, monkeypatch):
    """Writes land through the flusher thread; drain() is a true barrier;
    reopen sees everything (durability equivalence with the inline path)."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "3")  # force several batches
    apps = _rand_apps(10)
    g = _kuzu_graph(tmp_path, "offlock_kuzu")
    ids = [g.upsert_node(apps[i], kind="npc", cycle=i) for i in range(10)]
    assert len(set(ids)) == 10
    assert g.drain(timeout=15) is True
    m = g.persistence_metrics()
    assert m["graph_flush_queue_depth"] == 0
    assert m["graph_flush_error_batches"] == 0
    # Resolve time under the lock must be tiny -- the whole point of WS4B.
    assert m["graph_flush_lock_ms"] < 50.0
    # Dedicated write connection engaged (M0.1 probe passed on this box):
    # readers of the shared connection never queue behind a batch.
    assert m["graph_dedicated_write_conn"] is True
    g.close()

    g2 = _kuzu_graph(tmp_path, "offlock_kuzu")
    try:
        assert len(g2._nodes) == 10
    finally:
        g2.close()


def test_ws4b_flush_failure_replay_converges(tmp_path, monkeypatch):
    """Failure drill (PRD criterion 4): an injected batch failure rolls back
    and replays per-op; kuzu converges to what memory claims; later flushes
    are healthy; the live in-memory graph is never touched."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")  # manual flush timing
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")  # no age-trigger races
    apps = _rand_apps(6, seed=62)
    g = _kuzu_graph(tmp_path, "replay_kuzu")
    for i in range(5):
        g.upsert_node(apps[i], kind="npc", cycle=i)
    mem_node_ids = set(g._nodes)
    g._test_fail_next_batch = True
    assert g.drain(timeout=15) is True
    m = g.persistence_metrics()
    assert m["graph_flush_error_batches"] == 1
    assert set(g._nodes) == mem_node_ids  # memory untouched by the failure path

    # Subsequent writes flush cleanly.
    g.upsert_node(apps[5], kind="npc", cycle=99)
    assert g.drain(timeout=15) is True
    assert g.persistence_metrics()["graph_flush_error_batches"] == 1
    g.close()

    g2 = _kuzu_graph(tmp_path, "replay_kuzu")
    try:
        assert len(g2._nodes) == 6  # replay + follow-up both durable
    finally:
        g2.close()


def test_ws4b_m34_group_stmts_pure():
    """M3.4 grouping is a pure list walk: runs of the same multirow-capable
    template coalesce into one UNWIND statement; global order is preserved;
    deletes and paramless statements never group; singletons pass through."""
    pytest.importorskip("kuzu")
    from decadic.memory.kuzu_graph import (
        _Q_MULTIROW,
        _Q_NODE_CREATE,
        _Q_NODE_DEL,
        _Q_NODE_SET,
        _Q_SEM_CREATE,
        _group_stmts,
    )

    p = lambda i: {"id": f"n{i}"}  # noqa: E731 - shape is irrelevant to grouping
    stmts = [
        (_Q_NODE_CREATE, p(0)),
        (_Q_NODE_CREATE, p(1)),
        (_Q_NODE_CREATE, p(2)),  # run of 3 -> 1 UNWIND
        (_Q_NODE_DEL, p(3)),  # never grouped
        (_Q_NODE_CREATE, p(4)),  # singleton after the break -> passthrough
        (_Q_SEM_CREATE, {"pk": "a"}),
        (_Q_SEM_CREATE, {"pk": "b"}),  # run of 2 -> 1 UNWIND
        (_Q_NODE_SET, p(5)),
    ]
    out = _group_stmts(stmts)
    assert [q for q, _ in out] == [
        _Q_MULTIROW[_Q_NODE_CREATE],
        _Q_NODE_DEL,
        _Q_NODE_CREATE,
        _Q_MULTIROW[_Q_SEM_CREATE],
        _Q_NODE_SET,
    ]
    assert [r["id"] for r in out[0][1]["rows"]] == ["n0", "n1", "n2"]
    assert out[1][1] == p(3)  # delete param untouched
    assert [r["pk"] for r in out[3][1]["rows"]] == ["a", "b"]
    # Row order inside a run == original statement order (last-wins semantics
    # of the pending dict survive grouping).
    assert _group_stmts([]) == []


def test_ws4b_m34_multirow_flush_durable_and_compressed(tmp_path, monkeypatch):
    """M3.4 end-to-end: an insert-heavy batch executes as ONE multi-row
    statement (telemetry proves the compression), lands durably, and a
    follow-up update-heavy batch compresses the same way. Parity arm:
    DECADIC_KUZU_MULTIROW=0 produces the identical durable state."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")  # manual flush timing
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    apps = _rand_apps(12, seed=64)

    g = _kuzu_graph(tmp_path, "multirow_kuzu")
    ids = [g.upsert_node(apps[i], kind="npc", cycle=i) for i in range(12)]
    assert len(set(ids)) == 12
    assert g.drain(timeout=15) is True
    m = g.persistence_metrics()
    assert m["graph_flush_error_batches"] == 0
    assert m["graph_flush_rows"] == 12
    assert m["graph_flush_stmts"] < m["graph_flush_rows"]  # the M3.4 win
    assert m["graph_flush_stmts"] == 1  # one uninterrupted CREATE run

    # Update-heavy follow-up: same keys -> SET run -> one statement again.
    for i in range(12):
        g.upsert_node(apps[i], kind="npc", cycle=100 + i)
    assert g.drain(timeout=15) is True
    m2 = g.persistence_metrics()
    assert m2["graph_flush_error_batches"] == 0
    assert m2["graph_flush_stmts"] == 1
    g.close()

    g2 = _kuzu_graph(tmp_path, "multirow_kuzu")
    try:
        assert len(g2._nodes) == 12  # durable through the UNWIND path
    finally:
        g2.close()

    # Parity arm: grouping off -> same durable result, one statement per row.
    monkeypatch.setenv("DECADIC_KUZU_MULTIROW", "0")
    h = _kuzu_graph(tmp_path, "multirow_off_kuzu")
    for i in range(12):
        h.upsert_node(apps[i], kind="npc", cycle=i)
    assert h.drain(timeout=15) is True
    hm = h.persistence_metrics()
    assert hm["graph_flush_error_batches"] == 0
    assert hm["graph_flush_stmts"] == hm["graph_flush_rows"] == 12
    h.close()
    h2 = _kuzu_graph(tmp_path, "multirow_off_kuzu")
    try:
        assert len(h2._nodes) == 12
    finally:
        h2.close()


def test_ws4b_m34_multirow_edges_create_and_set(tmp_path, monkeypatch):
    """Regression for the edge-SET alias collision (found by the embodied
    soak): the UNWIND alias must not shadow the relationship variable `r` in
    ``MATCH (a)-[r:RELATES]->(b) ... SET r.*``. Exercises a CREATE run (new
    edges) then a SET run (re-bumping the SAME edges) -- both must flush as
    grouped UNWIND statements with ZERO error batches, and the SET values must
    land durably. Node-only tests never touch this path."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")  # manual flush timing
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    apps = _rand_apps(6, seed=66)
    g = _kuzu_graph(tmp_path, "multirow_edges_kuzu")
    ids = [g.upsert_node(apps[i], kind="npc", cycle=i) for i in range(6)]
    assert g.drain(timeout=15) is True

    # CREATE run: 5 fresh edges in one batch -> one grouped edge-CREATE.
    for i in range(5):
        g.bump_edge(ids[i], ids[i + 1], kind="near", weight=1.0, cycle=i)
    assert g.drain(timeout=15) is True
    m = g.persistence_metrics()
    assert m["graph_flush_error_batches"] == 0, "edge CREATE grouping must not fall to replay"

    # SET run: re-bump the SAME 5 edges -> one grouped edge-SET (the shape
    # that failed with the `r` alias).
    for i in range(5):
        g.bump_edge(ids[i], ids[i + 1], kind="near", weight=3.0, cycle=100 + i)
    assert g.drain(timeout=15) is True
    m2 = g.persistence_metrics()
    assert m2["graph_flush_error_batches"] == 0, "edge SET grouping must not shadow the REL var"
    assert m2["graph_flush_stmts"] < m2["graph_flush_rows"]  # actually grouped
    g.close()

    # Durability: the SET run's mutations survive. bump_edge caps weight at
    # 1.0 but increments count (1 -> 2) and advances last_cycle to the SET
    # value (100+i), so those are the fields that prove the grouped SET landed.
    g2 = _kuzu_graph(tmp_path, "multirow_edges_kuzu")
    try:
        edges = g2._edges
        assert len(edges) == 5, "all edges must reload"
        assert all(e["count"] == 2 for e in edges.values()), "SET must have re-bumped every edge"
        assert all(e["last_cycle"] >= 100 for e in edges.values()), "SET cycle must be durable"
    finally:
        g2.close()


def test_ws4b_m34_multirow_failure_replay_still_converges(tmp_path, monkeypatch):
    """The failure drill survives grouping: an injected failure on a grouped
    batch rolls back and replays from OPS (statement-shape agnostic)."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    apps = _rand_apps(5, seed=65)
    g = _kuzu_graph(tmp_path, "multirow_replay_kuzu")
    for i in range(5):
        g.upsert_node(apps[i], kind="npc", cycle=i)
    mem_node_ids = set(g._nodes)
    g._test_fail_next_batch = True
    assert g.drain(timeout=15) is True
    m = g.persistence_metrics()
    assert m["graph_flush_error_batches"] == 1
    assert set(g._nodes) == mem_node_ids
    g.close()
    g2 = _kuzu_graph(tmp_path, "multirow_replay_kuzu")
    try:
        assert len(g2._nodes) == 5  # per-op replay converged
    finally:
        g2.close()


def test_ws4b_inline_mode_still_works(tmp_path, monkeypatch):
    """DECADIC_KUZU_OFFLOCK_FLUSH=0 restores the 07-04 inline path (A/B arm)."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "0")
    apps = _rand_apps(4, seed=63)
    g = _kuzu_graph(tmp_path, "inline_kuzu")
    for i in range(4):
        g.upsert_node(apps[i], kind="npc", cycle=i)
    assert g.drain(timeout=15) is True
    g.close()
    g2 = _kuzu_graph(tmp_path, "inline_kuzu")
    try:
        assert len(g2._nodes) == 4
    finally:
        g2.close()


def test_percept_recency_exclusion_parity(tmp_path, monkeypatch):
    """WS3 recency horizon: exclude_cycle_after behaves identically on both
    backends and restores 'familiar = seen BEFORE the recent past'.

    Layout: cycles 0..9 carry distinct random keys; cycles 100..109 repeat
    those same keys (a second patrol lap). Without the horizon a lap-repeat
    query trivially matches its fresh twin; with the horizon at 50 it must
    match the OLD lap (still similarity ~1 -> ambient stays calm), and with
    the horizon below everything it returns empty (None side channel ->
    fully novel downstream).
    """
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "0")
    from decadic.memory.lancedb_store import LanceEpisodicStore

    rng = np.random.default_rng(23)
    keys = rng.normal(size=(10, 16)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)

    def _rec(cycle: int, key: np.ndarray) -> EpisodicRecord:
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb[PERCEPT_KEY_SLICE] = key
        return EpisodicRecord(
            cycle_index=cycle, summary={"c": cycle}, salience=0.9, embedding=emb
        )

    sq = EpisodicStore(tmp_path / "recency.sqlite")
    lz = LanceEpisodicStore(tmp_path / "recency_lance")
    try:
        for store in (sq, lz):
            for i in range(10):
                store.append(_rec(i, keys[i]))
            for i in range(10):
                store.append(_rec(100 + i, keys[i]))
            if hasattr(store, "flush"):  # lance micro-batch; sqlite commits inline
                store.flush()

        for store in (sq, lz):
            # No horizon: the fresh twin (cycle 103) is a legitimate match.
            hits = store.search_similar_percept(keys[3], top_k=1)
            assert hits and hits[0]["cycle_index"] in (3, 103)
            assert store.last_best_percept_similarity == pytest.approx(1.0, abs=1e-5)

            # Horizon at 50: the recent lap is invisible; the OLD sighting
            # still answers -- a repeated loop stays familiar.
            hits = store.search_similar_percept(keys[3], top_k=1, exclude_cycle_after=50)
            assert hits and hits[0]["cycle_index"] == 3
            assert store.last_best_percept_similarity == pytest.approx(1.0, abs=1e-5)

            # A never-seen key under the horizon: best match must be weak.
            novel = rng.normal(size=16).astype(np.float32)
            novel /= np.linalg.norm(novel)
            store.search_similar_percept(novel, top_k=1, exclude_cycle_after=50)
            assert store.last_best_percept_similarity is not None
            assert store.last_best_percept_similarity < 0.999

            # Horizon below everything: no eligible rows -> empty + None
            # (reads as fully novel downstream).
            hits = store.search_similar_percept(keys[3], top_k=1, exclude_cycle_after=0)
            assert hits == []
            assert store.last_best_percept_similarity is None

        # Cross-backend parity on the horizon-constrained ranking.
        for q in keys:
            a = sq.search_similar_percept(q, top_k=3, exclude_cycle_after=50)
            b = lz.search_similar_percept(q, top_k=3, exclude_cycle_after=50)
            assert [h["cycle_index"] for h in a] == [h["cycle_index"] for h in b]
    finally:
        lz.close()  # sqlite EpisodicStore has no close(); GC handles it


def test_lance_nan_embedding_sanitized_at_boundary(tmp_path, monkeypatch):
    """NaN/inf embeddings must not fail the lance flush (sqlite parity).

    The sqlite backend silently persisted NaN embeddings (raw blob) and its
    cosine path simply never ranked them. Lance VALIDATES vectors on add() and
    raises "Vector column contains NaN values" -- one bad episode from the
    cognition side became a failed flush on the caller's thread
    (test_api_dashboard::test_list_agents under the flipped defaults).
    _row_from_record now zeroes non-finite values, so the row lands, the
    mirror and disk stay identical, and the zero vector never wins a search.
    """
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "0")
    good = _synthetic_records(3, seed=17)
    q = np.asarray(good[1].embedding, dtype=np.float32)

    store = _lance_store(tmp_path, "nan_lance")
    try:
        for r in good:
            store.append(r)
        bad = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
        bad[0] = np.inf
        bad[1] = -np.inf
        store.append(
            EpisodicRecord(
                cycle_index=99, summary={"nan": True}, salience=0.5, embedding=bad
            )
        )
        store.flush()  # the original failure point: must not raise
        # The sanitized row is stored but can never win a search.
        assert _cycles(store.search_similar(q, top_k=1)) == [1]
        assert store.recall_cache_stats()["size"] == 4
    finally:
        store.close()

    # Reopen: the committed corpus (including the sanitized row) bulk-loads
    # cleanly and searches identically -- mirror and durability layer agree.
    store2 = _lance_store(tmp_path, "nan_lance")
    try:
        assert _cycles(store2.search_similar(q, top_k=1)) == [1]
        assert store2.recall_cache_stats()["size"] == 4
    finally:
        store2.close()


def test_lance_mirror_reopen_rebuilds_from_disk(tmp_path, monkeypatch):
    """append + flush + close, reopen: the mirror bulk-loads the full corpus."""
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "0")
    records = _synthetic_records(50, seed=33)
    store = _lance_store(tmp_path, "reopen_lance")
    try:
        for r in records:
            store.append(r)
        store.flush()
    finally:
        store.close()

    store2 = _lance_store(tmp_path, "reopen_lance")
    try:
        q = np.asarray(records[7].embedding, dtype=np.float32)
        assert _cycles(store2.search_similar(q, top_k=1)) == [7]
        assert store2.last_best_similarity == pytest.approx(1.0, abs=1e-5)
        key = np.asarray(records[7].embedding, dtype=np.float32)[PERCEPT_KEY_SLICE]
        assert _cycles(store2.search_similar_percept(key, top_k=1)) == [7]
        stats = store2.recall_cache_stats()
        assert stats["enabled"] is True
        assert stats["size"] == 50  # every committed row mirrored
        assert stats["hits"] >= 2 and stats["misses"] == 0
    finally:
        store2.close()


def test_lance_mirror_prune_invalidation(tmp_path, monkeypatch):
    """Pruned victims disappear from mirror-served search results."""
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "1")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RECENT_CAP", "20")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_SALIENT_CAP", "10")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_INTERVAL_WRITES", "10")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_PRUNE_BATCH", "1000")
    store = _lance_store(tmp_path, "prune_lance")
    try:
        records = _synthetic_records(100, seed=35)
        # Warm the mirror first so pruning exercises the id-mask delete path
        # (not just a later rebuild).
        assert store.search_similar(np.ones(EMBEDDING_DIM, np.float32), top_k=1) == []
        for r in records:
            store.append(r)
        surviving = {r["cycle_index"] for r in store.recent(limit=500)}
        assert len(surviving) < 100  # pruning actually happened
        pruned = sorted(set(range(100)) - surviving)
        assert pruned
        for cycle in pruned[:5]:
            q = np.asarray(records[cycle].embedding, dtype=np.float32)
            assert cycle not in _cycles(store.search_similar(q, top_k=3))
        stats = store.recall_cache_stats()
        assert stats["enabled"] is True
        assert stats["size"] == len(surviving)  # mirror tracks the survivors
    finally:
        store.close()


def test_lance_mirror_equals_scan_path(tmp_path, monkeypatch):
    """Mirror-served results EXACTLY match the lance-scan fallback path.

    The scan store's mirror cap is monkeypatched to 0, so it takes the
    brute-force ``table.search`` + exact-cosine-rerank path on identical data.
    """
    pytest.importorskip("lancedb")
    monkeypatch.setenv("DECADIC_EPISODIC_DB_RETENTION_ENABLED", "0")
    records = _synthetic_records(300, seed=37)
    mirror = _lance_store(tmp_path, "eq_mirror")
    scan = _lance_store(tmp_path, "eq_scan")
    monkeypatch.setattr(scan, "_mirror_cap_rows", lambda: 0)  # force lance path
    try:
        for r in records:
            mirror.append(r)
            scan.append(r)
        mirror.flush()
        scan.flush()

        rng = np.random.default_rng(101)
        for _ in range(10):
            q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            hm = mirror.search_similar(q, top_k=5, min_salience=0.3)
            hs = scan.search_similar(q, top_k=5, min_salience=0.3)
            assert _cycles(hm) == _cycles(hs)
            for a, b in zip(hm, hs):
                assert a["similarity"] == pytest.approx(b["similarity"], abs=1e-6)
            assert mirror.last_best_similarity == pytest.approx(
                scan.last_best_similarity, abs=1e-6
            )

        # exclude_cycle applied identically (as a pre-top-k mask).
        q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        exclude = int(mirror.search_similar(q, top_k=1)[0]["cycle_index"])
        hm = mirror.search_similar(q, top_k=5, exclude_cycle=exclude)
        hs = scan.search_similar(q, top_k=5, exclude_cycle=exclude)
        assert _cycles(hm) == _cycles(hs)
        assert exclude not in _cycles(hm)

        # Percept-key mirror equality too.
        for _ in range(10):
            key = rng.standard_normal(PERCEPT_KEY_DIM).astype(np.float32)
            hm = mirror.search_similar_percept(key, top_k=5)
            hs = scan.search_similar_percept(key, top_k=5)
            assert _cycles(hm) == _cycles(hs)
            for a, b in zip(hm, hs):
                assert a["similarity"] == pytest.approx(b["similarity"], abs=1e-6)
            assert mirror.last_best_percept_similarity == pytest.approx(
                scan.last_best_percept_similarity, abs=1e-6
            )

        # Each store took the path this test believes it took.
        m_stats = mirror.recall_cache_stats()
        s_stats = scan.recall_cache_stats()
        assert m_stats["enabled"] is True and m_stats["misses"] == 0
        assert m_stats["hits"] >= 22
        assert s_stats["enabled"] is False and s_stats["size"] == 0
        assert s_stats["misses"] >= 21 and s_stats["hits"] == 0
    finally:
        mirror.close()
        scan.close()


# --------------------------------------------------------------------------
# M2 -- sqlite vs kuzu semantic-graph parity
# --------------------------------------------------------------------------

APPEARANCE_DIM = 16  # production appearance-fingerprint dim (perception/organ.py)


def _graph_pair(tmp_path):
    pytest.importorskip("kuzu")
    from decadic.memory.kuzu_graph import KuzuLongTermGraph
    from decadic.memory.semantic_graph import LongTermGraph

    sq = LongTermGraph(tmp_path / "graph.sqlite")
    kz = KuzuLongTermGraph(tmp_path / "graph_kuzu")
    return sq, kz


def _appearance_stream(n=60, seed=17):
    """Observation stream: ~half revisits (tiny noise, well inside the 0.6
    cosine threshold), ~half novel random unit directions."""
    rng = np.random.default_rng(seed)
    protos: list[np.ndarray] = []
    stream: list[np.ndarray] = []
    for _ in range(n):
        if protos and rng.uniform() < 0.5:
            base = protos[int(rng.integers(len(protos)))]
            vec = base + rng.normal(0.0, 0.02, size=APPEARANCE_DIM).astype(np.float32)
        else:
            vec = rng.normal(size=APPEARANCE_DIM).astype(np.float32)
            protos.append(vec)
        vec = vec / max(1e-8, float(np.linalg.norm(vec)))
        stream.append(vec.astype(np.float32))
    return stream


def _run_identity_stream(sq, kz, stream):
    match_sq, match_kz, ids_sq, ids_kz = [], [], [], []
    for cycle, vec in enumerate(stream):
        match_sq.append(sq.match(vec))
        match_kz.append(kz.match(vec))
        ids_sq.append(sq.upsert_node(vec, kind="obj", cycle=cycle))
        ids_kz.append(kz.upsert_node(vec, kind="obj", cycle=cycle))
    return match_sq, match_kz, ids_sq, ids_kz


def test_kuzu_identity_parity(tmp_path):
    sq, kz = _graph_pair(tmp_path)
    try:
        match_sq, match_kz, ids_sq, ids_kz = _run_identity_stream(
            sq, kz, _appearance_stream(60)
        )
        assert match_sq == match_kz  # identical match-decision sequence
        assert ids_sq == ids_kz  # identical node-id assignment sequence
        assert sq.counts() == kz.counts()
    finally:
        kz.close()


def test_kuzu_identity_parity_no_match_cache(tmp_path, monkeypatch):
    """Cache disabled: sqlite linear scan vs kuzu vector-index (or its guarded
    linear fallback) must produce the same decisions."""
    monkeypatch.setenv("DECADIC_LTM_MATCH_CACHE_ENABLED", "0")
    sq, kz = _graph_pair(tmp_path)
    try:
        match_sq, match_kz, ids_sq, ids_kz = _run_identity_stream(
            sq, kz, _appearance_stream(40, seed=23)
        )
        assert match_sq == match_kz
        assert ids_sq == ids_kz
        assert sq.counts() == kz.counts()
    finally:
        kz.close()


def test_kuzu_belief_parity(tmp_path):
    sq, kz = _graph_pair(tmp_path)
    try:
        vec = np.ones(APPEARANCE_DIM, dtype=np.float32)
        n_sq = sq.upsert_node(vec, kind="obj", cycle=0)
        n_kz = kz.upsert_node(vec, kind="obj", cycle=0)
        assert n_sq == n_kz
        rng = np.random.default_rng(5)
        for cycle in range(1, 12):
            ev = {
                "size_proxy": float(0.40 + 0.01 * rng.uniform()),
                "edge_strength": float(rng.uniform(0.20, 0.25)),
            }
            w = float(rng.uniform(0.3, 1.0))
            u_sq = sq.upsert_property_beliefs(n_sq, ev, cycle=cycle, evidence_weight=w)
            u_kz = kz.upsert_property_beliefs(n_kz, ev, cycle=cycle, evidence_weight=w)
            assert u_sq == u_kz == 2
        # Contradiction: mean jump >= CONTRADICTION_DELTA after >= 5 evidence.
        contradiction = {"size_proxy": 1.0}
        sq.upsert_property_beliefs(n_sq, contradiction, cycle=20, evidence_weight=1.0)
        kz.upsert_property_beliefs(n_kz, contradiction, cycle=20, evidence_weight=1.0)

        bs_sq, bs_kz = sq.belief_stats(), kz.belief_stats()
        assert bs_sq["total_property_beliefs"] == bs_kz["total_property_beliefs"] == 2
        assert bs_sq["unstable_property_count"] == bs_kz["unstable_property_count"] == 1
        assert bs_sq["avg_property_confidence"] == pytest.approx(
            bs_kz["avg_property_confidence"], abs=1e-6
        )
        pb_sq = sq.snapshot()["nodes"][0]["property_beliefs"]
        pb_kz = kz.snapshot()["nodes"][0]["property_beliefs"]
        assert len(pb_sq) == len(pb_kz) == 2
        for a, b in zip(pb_sq, pb_kz):
            assert a["property_key"] == b["property_key"]
            assert a["mean"] == pytest.approx(b["mean"], abs=1e-6)
            assert a["variance"] == pytest.approx(b["variance"], abs=1e-6)
            assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-6)
            assert a["evidence_count"] == pytest.approx(b["evidence_count"], abs=1e-6)
            assert a["unstable"] == b["unstable"]
    finally:
        kz.close()


def test_kuzu_edge_parity(tmp_path):
    sq, kz = _graph_pair(tmp_path)
    try:
        ids = []
        for cycle, vec in enumerate(_appearance_stream(20, seed=3)):
            nid_sq = sq.upsert_node(vec, kind="obj", cycle=cycle)
            nid_kz = kz.upsert_node(vec, kind="obj", cycle=cycle)
            assert nid_sq == nid_kz
            ids.append(nid_sq)
        uniq = list(dict.fromkeys(ids))
        assert len(uniq) >= 2
        for rep in range(3):  # repeated co-presence accrues weight/count
            for a, b in zip(uniq, uniq[1:]):
                sq.bump_edge(a, b, cycle=100 + rep)
                kz.bump_edge(a, b, cycle=100 + rep)
        sq.bump_edge(uniq[0], uniq[1], kind="scene_near", weight=0.7, cycle=200)
        kz.bump_edge(uniq[0], uniq[1], kind="scene_near", weight=0.7, cycle=200)
        assert sq.counts() == kz.counts()
        edges_sq = {
            (e["source"], e["target"], e["kind"]): (e["weight"], e["count"], e["last_cycle"])
            for e in sq.snapshot(limit=0)["edges"]
        }
        edges_kz = {
            (e["source"], e["target"], e["kind"]): (e["weight"], e["count"], e["last_cycle"])
            for e in kz.snapshot(limit=0)["edges"]
        }
        assert edges_sq == edges_kz
    finally:
        kz.close()


def test_kuzu_snapshot_shape(tmp_path):
    sq, kz = _graph_pair(tmp_path)
    try:
        ids = []
        for cycle, vec in enumerate(_appearance_stream(10, seed=41)):
            ids.append(sq.upsert_node(vec, kind="obj", cycle=cycle))
            kz.upsert_node(vec, kind="obj", cycle=cycle)
        uniq = list(dict.fromkeys(ids))
        sq.bump_edge(uniq[0], uniq[1], cycle=50)
        kz.bump_edge(uniq[0], uniq[1], cycle=50)
        sq.upsert_property_beliefs(uniq[0], {"size_proxy": 0.5}, cycle=51)
        kz.upsert_property_beliefs(uniq[0], {"size_proxy": 0.5}, cycle=51)

        snap_sq, snap_kz = sq.snapshot(), kz.snapshot()
        assert set(snap_sq) == set(snap_kz)  # same top-level keys
        assert set(snap_sq["nodes"][0]) == set(snap_kz["nodes"][0])  # per-node keys
        assert snap_sq["edges"] and set(snap_sq["edges"][0]) == set(snap_kz["edges"][0])
        assert set(snap_sq["semantic"]) == set(snap_kz["semantic"])
        # The dashboard-facing scalars agree too (same underlying model).
        for key in ("total_nodes", "total_edges", "total_property_beliefs"):
            assert snap_sq[key] == snap_kz[key]
    finally:
        kz.close()


def test_kuzu_backup_restore_roundtrip(tmp_path):
    pytest.importorskip("kuzu")
    from decadic.memory.kuzu_graph import KuzuLongTermGraph

    kz = KuzuLongTermGraph(tmp_path / "graph_kuzu")
    try:
        ids = []
        for cycle, vec in enumerate(_appearance_stream(30, seed=9)):
            ids.append(kz.upsert_node(vec, kind="obj", cycle=cycle))
        uniq = list(dict.fromkeys(ids))
        kz.bump_edge(uniq[0], uniq[1], cycle=40)
        kz.upsert_property_beliefs(uniq[0], {"size_proxy": 0.5}, cycle=41)
        baseline_counts = kz.counts()
        baseline_snapshot = kz.snapshot()

        kz.backup_to(tmp_path / "bak_kuzu")

        # Mutate with basis-vector appearances (orthogonal-ish: guaranteed new nodes).
        for j in range(8):
            vec = np.zeros(APPEARANCE_DIM, dtype=np.float32)
            vec[j] = 1.0
            kz.upsert_node(vec, kind="obj", cycle=99)
        assert kz.counts()[0] > baseline_counts[0]

        kz.restore_from(tmp_path / "bak_kuzu")
        assert kz.counts() == baseline_counts
        assert kz.snapshot() == baseline_snapshot

        # Persistence across close/reopen (exercises the kuzu load path).
        kz.close()
        kz2 = KuzuLongTermGraph(tmp_path / "graph_kuzu")
        try:
            assert kz2.counts() == baseline_counts
            assert kz2.snapshot() == baseline_snapshot
        finally:
            kz2.close()
    finally:
        kz.close()  # idempotent


def test_kuzu_write_behind_sync_consolidation(tmp_path):
    """The runtime wrapper's synchronous path works end-to-end on kuzu."""
    pytest.importorskip("kuzu")
    from types import SimpleNamespace

    from decadic.memory.kuzu_graph import WriteBehindKuzuLongTermGraph

    graph = WriteBehindKuzuLongTermGraph(tmp_path / "rt_kuzu", enabled=False)
    try:
        slot = SimpleNamespace(
            entity_id="wm-1",
            kind="object",
            position=None,
            affective_weight=0.0,
            seen_count=3,
            appearance=[float(x) for x in np.eye(APPEARANCE_DIM, dtype=np.float32)[0]],
            confidence=0.9,
            kind_hint="object",
            entity_role="compact_entity",
            precision=0.9,
            provisional=False,
            evidence_count=3.0,
            contradiction_pressure=0.0,
            event_links=[],
            relationship_links=[],
            scene_entity_id=None,
            property_evidence={"size_proxy": 0.4},
        )
        report = graph.enqueue_consolidation_job([slot], cycle=5, min_seen=2)
        assert report["queued"] is False
        assert report["accepted_ids"] == ["ent-00001"]
        assert graph.counts() == (1, 0)
        metrics = graph.runtime_metrics()
        assert metrics["backend"] == "kuzu"
        assert "ltm_consolidation_queue_depth" in metrics
    finally:
        graph.close()
