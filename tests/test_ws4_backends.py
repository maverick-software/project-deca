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


def test_factory_default_is_sqlite(monkeypatch):
    monkeypatch.delenv("DECADIC_MEMORY_BACKEND", raising=False)
    monkeypatch.delenv("DECADIC_GRAPH_BACKEND", raising=False)
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
