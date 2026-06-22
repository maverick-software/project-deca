from __future__ import annotations

import numpy as np
import pytest

from decadic.memory.embeddings import EMBEDDING_DIM
from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore
from decadic.memory.write_behind import WriteBehindEpisodicStore


def _embedding(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    return list(rng.randn(EMBEDDING_DIM).astype(float))


def _record(cycle: int, *, salience: float = 0.5, seed: int | None = None) -> EpisodicRecord:
    return EpisodicRecord(
        cycle_index=cycle,
        summary={"cycle": cycle},
        salience=salience,
        embedding=_embedding(seed if seed is not None else cycle),
    )


def test_hot_recall_cache_avoids_json_decode(monkeypatch):
    store = EpisodicStore(None)
    for i in range(8):
        store.append(_record(i, salience=0.6))

    import decadic.memory.episodic_store as episodic_module

    def fail_json_loads(*_args, **_kwargs):
        raise AssertionError("json.loads must not run on hot RAM-cache recall")

    monkeypatch.setattr(episodic_module.json, "loads", fail_json_loads)
    hits = store.search_similar(np.asarray(_embedding(3), dtype=np.float32), top_k=3)
    assert hits
    assert hits[0]["cycle_index"] == 3
    stats = store.recall_cache_stats()
    assert stats["hits"] >= 1


def test_sql_fallback_is_salience_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADIC_EPISODIC_RECALL_CACHE_ENABLED", "0")
    monkeypatch.setenv("DECADIC_EPISODIC_RECALL_SQL_FALLBACK_CAP", "7")
    store = EpisodicStore(tmp_path / "episodes.sqlite")
    for i in range(20):
        store.append(_record(i, salience=float(i) / 20.0))

    traces: list[str] = []
    assert store._conn is not None
    store._conn.set_trace_callback(traces.append)
    hits = store.search_similar(np.asarray(_embedding(19), dtype=np.float32), top_k=3)
    assert hits
    recall_sql = " ".join(traces)
    assert "ORDER BY salience DESC" in recall_sql
    assert "LIMIT 7" in recall_sql


def test_recall_cache_retains_salient_old_memory(monkeypatch):
    monkeypatch.setenv("DECADIC_EPISODIC_RECALL_RECENT_CAP", "3")
    monkeypatch.setenv("DECADIC_EPISODIC_RECALL_SALIENT_CAP", "2")
    store = EpisodicStore(None)
    store.append(_record(1, salience=0.99, seed=1234))
    for i in range(2, 18):
        store.append(_record(i, salience=0.05, seed=i))

    hits = store.search_similar(np.asarray(_embedding(1234), dtype=np.float32), top_k=1)
    assert hits
    assert hits[0]["cycle_index"] == 1
    assert store.recall_cache_stats()["size"] <= 5


def test_write_behind_recall_visible_before_flush(tmp_path):
    store = WriteBehindEpisodicStore(tmp_path / "episodes.sqlite", enabled=True)
    try:
        store.append(_record(42, salience=0.9, seed=4242))
        hits = store.search_similar(np.asarray(_embedding(4242), dtype=np.float32), top_k=1)
        assert hits
        assert hits[0]["cycle_index"] == 42
    finally:
        store.close()


def test_neural_cycle_uses_cached_memory_context(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_REQUIRE_CUDA", "0")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    bundle = NeuralBundle.try_build("unit-cached-memory")
    assert bundle is not None
    episodic = EpisodicStore(None)

    def fail_recall(*_args, **_kwargs):
        raise AssertionError("episodic recall must not run when cached context is supplied")

    monkeypatch.setattr(episodic, "retrieval_context_vector", fail_recall)
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=episodic,
        cached_memory_context=[0.0] * bundle.cfg.memory_context_dim,
        last_observation={"proprioception": {"position": [0.0, 0.0, 0.0]}},
    )
    out = run_neural_cycle(ctx, bundle)
    assert out["_diagnostics"]["memory_recall_on_critical_path"] is False

