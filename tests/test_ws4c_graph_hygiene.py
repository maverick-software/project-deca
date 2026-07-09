"""WS4C M2 relational hygiene (2026-07-07).

Pins the three fixes from reports/ws4c_m1_verdict.md:

- M2.3 keyed events: one aggregate semantic event record per event_class
  (evidence accumulates) instead of a fresh anonymous id per instance
  (194k rows / 6 h in the diagnosed run).
- M2.1 edge retention + degree cap: stale scene-class edges retire; a degree
  cap backstop bounds the relation set; deletions mirror into kuzu via the
  new del_edge op (before WS4C no RELATES delete path existed at all).
- M2.2 refresh != rewrite: an upsert whose payload equals the last staged
  payload (modulo volatile bookkeeping fields) stages NOTHING -- with a
  staleness ceiling so last_cycle in kuzu never fossilizes.

Params are pinned via env; tests set them explicitly rather than relying on
defaults, per house discipline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from decadic.memory.semantic_graph import LongTermGraph


def _rand_apps(n, seed=61):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 16)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _kuzu_graph(tmp_path, name):
    from decadic.memory.kuzu_graph import KuzuLongTermGraph

    return KuzuLongTermGraph(tmp_path / name)


def _slot(sid: str) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=sid,
        confidence=0.9,
        precision=0.9,
        entity_role="prop",
        provisional=False,
        seen_count=3,
        property_evidence={},
        affective_weight=0.0,
    )


# ---------------------------------------------------------------------------
# M2.3 -- keyed events
# ---------------------------------------------------------------------------
def test_event_records_keyed_by_class(monkeypatch):
    """Same event class across cycles -> ONE record, evidence accumulates."""
    monkeypatch.setenv("DECADIC_LTM_EVENT_KEYED", "1")
    g = LongTermGraph(None)
    ev = {"intensity": 0.5, "kind": "collision"}
    for cycle in range(1, 6):
        g.record_semantic_evidence([_slot("e1")], events=[dict(ev)], cycle=cycle)
    bucket = g._semantic["event"]
    assert len(bucket) == 1, f"expected one keyed record, got {sorted(bucket)}"
    rec = next(iter(bucket.values()))
    assert rec["id"].startswith("event:")
    assert rec["evidence_count"] > 1.0  # accumulated, not re-created
    assert rec["last_cycle"] == 5


def test_event_records_legacy_per_instance(monkeypatch):
    """Flag off restores the anonymous per-instance behavior."""
    monkeypatch.setenv("DECADIC_LTM_EVENT_KEYED", "0")
    g = LongTermGraph(None)
    ev = {"intensity": 0.5, "kind": "collision"}
    for cycle in range(1, 6):
        g.record_semantic_evidence([_slot("e1")], events=[dict(ev)], cycle=cycle)
    assert len(g._semantic["event"]) == 5  # one fresh id per instance


# ---------------------------------------------------------------------------
# M2.1 -- edge retention + degree cap
# ---------------------------------------------------------------------------
def _pin_edge_retention(monkeypatch, *, stale=100, cap=4):
    monkeypatch.setenv("DECADIC_LTM_RETENTION_ENABLED", "1")
    monkeypatch.setenv("DECADIC_LTM_EDGE_RETENTION_ENABLED", "1")
    monkeypatch.setenv("DECADIC_LTM_EDGE_STALE_CYCLES", str(stale))
    monkeypatch.setenv("DECADIC_LTM_EDGE_DEGREE_CAP", str(cap))
    monkeypatch.setenv("DECADIC_LTM_EDGE_PRUNE_PREFIXES", "scene_")


def test_edge_retention_retires_stale_scene_edges(monkeypatch):
    """Scene-class edges unconfirmed for N cycles retire; co_occurrence and
    fresh scene edges survive."""
    _pin_edge_retention(monkeypatch, stale=100, cap=99)
    g = LongTermGraph(None)
    apps = _rand_apps(3, seed=5)
    a = g.upsert_node(apps[0], kind="obj", cycle=1)
    b = g.upsert_node(apps[1], kind="obj", cycle=1)
    c = g.upsert_node(apps[2], kind="obj", cycle=1)
    g.bump_edge(a, b, kind="scene_near", weight=0.5, cycle=10)  # goes stale
    g.bump_edge(a, c, kind="scene_near", weight=0.5, cycle=490)  # fresh
    g.bump_edge(b, c, kind="co_occurrence", weight=0.5, cycle=10)  # exempt kind
    out = g.prune_retention(cycle=500)
    assert out["edges"] == 1
    kinds = {(k[0], k[1], k[2]) for k in g._edges}
    assert (a, b, "scene_near") not in kinds
    assert (a, c, "scene_near") in kinds
    assert (b, c, "co_occurrence") in kinds


def test_edge_degree_cap_keeps_top_k(monkeypatch):
    """Backstop: only the top-K prunable edges per node (by weight, then
    recency) survive, even when none are stale."""
    _pin_edge_retention(monkeypatch, stale=10_000, cap=3)
    g = LongTermGraph(None)
    apps = _rand_apps(8, seed=6)
    hub = g.upsert_node(apps[0], kind="obj", cycle=1)
    others = [g.upsert_node(apps[i], kind="obj", cycle=1) for i in range(1, 8)]
    for i, nid in enumerate(others):
        g.bump_edge(hub, nid, kind="scene_near", weight=0.1 * (i + 1), cycle=10 + i)
    assert len(g._edges) == 7
    out = g.prune_retention(cycle=20)
    assert out["edges"] == 4  # 7 - cap(3)
    survivors = {k for k in g._edges}
    weights = sorted(float(g._edges[k]["weight"]) for k in survivors)
    assert len(survivors) == 3
    assert weights == pytest.approx([0.5, 0.6, 0.7])  # top-3 by weight


def test_edge_retention_disabled_is_parity(monkeypatch):
    monkeypatch.setenv("DECADIC_LTM_RETENTION_ENABLED", "1")
    monkeypatch.setenv("DECADIC_LTM_EDGE_RETENTION_ENABLED", "0")
    g = LongTermGraph(None)
    apps = _rand_apps(2, seed=7)
    a = g.upsert_node(apps[0], kind="obj", cycle=1)
    b = g.upsert_node(apps[1], kind="obj", cycle=1)
    g.bump_edge(a, b, kind="scene_near", weight=0.5, cycle=1)
    out = g.prune_retention(cycle=100_000)
    assert out["edges"] == 0 and len(g._edges) == 1


def test_kuzu_edge_retention_mirrors_delete(tmp_path, monkeypatch):
    """The kuzu subclass stages del_edge for retired edges; reopen proves the
    RELATES row is gone (durable delete), and the key can be re-created."""
    pytest.importorskip("kuzu")
    _pin_edge_retention(monkeypatch, stale=100, cap=99)
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    apps = _rand_apps(3, seed=8)
    g = _kuzu_graph(tmp_path, "edge_ret_kuzu")
    a = g.upsert_node(apps[0], kind="obj", cycle=1)
    b = g.upsert_node(apps[1], kind="obj", cycle=1)
    c = g.upsert_node(apps[2], kind="obj", cycle=1)
    g.bump_edge(a, b, kind="scene_near", weight=0.5, cycle=10)  # will retire
    g.bump_edge(a, c, kind="scene_near", weight=0.5, cycle=490)  # survives
    out = g.prune_retention(cycle=500)
    assert out["edges"] == 1
    m = g.persistence_metrics()
    assert m.get("graph_writes_staged_del_edge", 0) == 1
    assert g.drain(timeout=15) is True
    g.close()
    g2 = _kuzu_graph(tmp_path, "edge_ret_kuzu")
    try:
        keys = set(g2._edges)
        assert (a, b, "scene_near") not in keys
        assert (a, c, "scene_near") in keys
        # retired key re-creates cleanly (gate forgot it)
        g2.bump_edge(a, b, kind="scene_near", weight=0.9, cycle=600)
        assert g2.drain(timeout=15) is True
    finally:
        g2.close()


# ---------------------------------------------------------------------------
# M2.2 -- refresh != rewrite (compare-before-stage)
# ---------------------------------------------------------------------------
def test_skip_unchanged_drops_identical_payloads(tmp_path, monkeypatch):
    """Re-persisting an edge whose non-volatile payload is unchanged stages
    and parks NOTHING; a real weight change passes the gate again."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_SKIP_UNCHANGED", "1")
    monkeypatch.setenv("DECADIC_KUZU_REFRESH_MAX_CYCLES", "1000")
    monkeypatch.setenv("DECADIC_KUZU_WRITE_MIN_CYCLES", "0")  # isolate the skip
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    g = _kuzu_graph(tmp_path, "skip_kuzu")
    # drive _persist_edge directly: same src/dst/kind/weight, only volatile
    # fields (count, last_cycle) advance -- the scene-refresh signature.
    base = {"src": "a", "dst": "b", "kind": "scene_near", "weight": 0.5}
    g._persist_edge({**base, "count": 1, "last_cycle": 1})
    for c in range(2, 30):
        g._persist_edge({**base, "count": c, "last_cycle": c})
    m = g.persistence_metrics()
    assert m["graph_writes_skipped_unchanged"] == 28
    assert m["graph_writes_deferred"] == 0
    assert m["graph_deferred_depth"] == 0
    # a REAL change passes the gate
    g._persist_edge({**base, "weight": 0.9, "count": 30, "last_cycle": 30})
    m = g.persistence_metrics()
    assert m["graph_writes_staged_edge_scene_near"] == 2
    g.close()


def test_skip_unchanged_staleness_ceiling_forces_refresh(tmp_path, monkeypatch):
    """Unchanged payloads still re-stage once the last staging is older than
    DECADIC_KUZU_REFRESH_MAX_CYCLES, so kuzu's last_cycle tracks reality."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_SKIP_UNCHANGED", "1")
    monkeypatch.setenv("DECADIC_KUZU_REFRESH_MAX_CYCLES", "50")
    monkeypatch.setenv("DECADIC_KUZU_WRITE_MIN_CYCLES", "0")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    g = _kuzu_graph(tmp_path, "skip_ceiling_kuzu")
    base = {"src": "a", "dst": "b", "kind": "scene_near", "weight": 0.5}
    g._persist_edge({**base, "count": 1, "last_cycle": 1})
    g._persist_edge({**base, "count": 2, "last_cycle": 10})  # skipped (young)
    g._persist_edge({**base, "count": 3, "last_cycle": 60})  # forced (stale)
    m = g.persistence_metrics()
    assert m["graph_writes_skipped_unchanged"] == 1
    assert m["graph_writes_staged_edge_scene_near"] == 2
    g.close()


def test_skip_unchanged_newest_state_survives_drain(tmp_path, monkeypatch):
    """Ordering hazard pinned: change staged -> change PARKED (throttle) ->
    attempt equal to the DURABLE state. The equal attempt must obsolete the
    parked intermediate (pop it), so drain does not regress the final state."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_SKIP_UNCHANGED", "1")
    monkeypatch.setenv("DECADIC_KUZU_REFRESH_MAX_CYCLES", "1000")
    monkeypatch.setenv("DECADIC_KUZU_WRITE_MIN_CYCLES", "25")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    g = _kuzu_graph(tmp_path, "skip_drain_kuzu")
    apps = _rand_apps(2, seed=9)
    a = g.upsert_node(apps[0], kind="obj", cycle=1)
    b = g.upsert_node(apps[1], kind="obj", cycle=1)
    base = {"src": a, "dst": b, "kind": "scene_near"}
    g._persist_edge({**base, "weight": 0.5, "count": 1, "last_cycle": 1})  # staged
    g._persist_edge({**base, "weight": 0.7, "count": 2, "last_cycle": 2})  # parked
    assert g.persistence_metrics()["graph_deferred_depth"] == 1
    g._persist_edge({**base, "weight": 0.5, "count": 3, "last_cycle": 3})  # == durable
    assert g.persistence_metrics()["graph_deferred_depth"] == 0  # parked obsoleted
    assert g.drain(timeout=15) is True
    g.close()
    g2 = _kuzu_graph(tmp_path, "skip_drain_kuzu")
    try:
        e = g2._edges[(a, b, "scene_near")]
        assert float(e["weight"]) == pytest.approx(0.5)  # final state, not 0.7
    finally:
        g2.close()


# ---------------------------------------------------------------------------
# M3 -- flusher catch-up (adaptive merge, pressure telemetry, shedding)
# ---------------------------------------------------------------------------
def test_flusher_multi_batch_drain_barrier_exact(tmp_path, monkeypatch):
    """M3.1: the flusher may pop and merge several batches per wake; the drain
    barrier accounting must remain EXACT and the merged data durable."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_OFFLOCK_FLUSH", "1")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "2")  # many small batches
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_MERGE_MAX", "4")
    monkeypatch.setenv("DECADIC_KUZU_WRITE_MIN_CYCLES", "0")
    monkeypatch.setenv("DECADIC_KUZU_SKIP_UNCHANGED", "0")
    apps = _rand_apps(12, seed=10)
    g = _kuzu_graph(tmp_path, "merge_kuzu")
    ids = [g.upsert_node(apps[i], kind="obj", cycle=i) for i in range(12)]
    assert g.drain(timeout=15) is True
    with g._flush_cv:
        assert g._batches_done == g._batches_enqueued  # exact after merge
    m = g.persistence_metrics()
    assert m["graph_flush_error_batches"] == 0
    assert m["graph_write_pressure"] >= 0.0  # telemetry present and sane
    assert m["graph_drain_capacity_rows_per_s"] > 0.0
    g.close()
    g2 = _kuzu_graph(tmp_path, "merge_kuzu")
    try:
        assert len(g2._nodes) == len(set(ids))
    finally:
        g2.close()


def test_maybe_shed_tiers_and_exemptions(tmp_path, monkeypatch):
    """M3.2: over the ceiling, edge SETs shed; belief SETs only past twice the
    ceiling; CREATEs/deletes/nodes are never shed. 0 disables."""
    pytest.importorskip("kuzu")
    from decadic.memory.kuzu_graph import (
        _Q_BELIEF_SET,
        _Q_EDGE_CREATE,
        _Q_EDGE_DEL,
        _Q_EDGE_SET,
        _Q_NODE_SET,
    )

    monkeypatch.setenv("DECADIC_KUZU_SHED_PRESSURE", "1.5")
    g = _kuzu_graph(tmp_path, "shed_kuzu")
    stmts = [
        (_Q_EDGE_SET, {"src": "a", "dst": "b", "kind": "scene_near"}),
        (_Q_EDGE_CREATE, {"src": "a", "dst": "c", "kind": "scene_near"}),
        (_Q_BELIEF_SET, {"pk": "a\x1fcolor"}),
        (_Q_NODE_SET, {"id": "a"}),
        (_Q_EDGE_DEL, {"src": "a", "dst": "d", "kind": "scene_near"}),
    ]
    ops = [
        ("edge", stmts[0][1]),
        ("edge", stmts[1][1]),
        ("belief", stmts[2][1]),
        ("node", stmts[3][1]),
        ("del_edge", stmts[4][1]),
    ]
    # below ceiling: nothing shed
    g._arrival_rows_ema, g._drain_capacity_ema = 100.0, 100.0  # pressure 1.0
    s, o = g._maybe_shed(list(stmts), list(ops))
    assert len(s) == 5
    # tier 1 (pressure 2.0): edge SET shed; belief SET survives
    g._arrival_rows_ema = 200.0
    s, o = g._maybe_shed(list(stmts), list(ops))
    assert [q for q, _ in s] == [_Q_EDGE_CREATE, _Q_BELIEF_SET, _Q_NODE_SET, _Q_EDGE_DEL]
    assert g._writes_shed_total == 1
    # tier 2 (pressure 4.0 > 2x ceiling): belief SET shed too; never the rest
    g._arrival_rows_ema = 400.0
    s, o = g._maybe_shed(list(stmts), list(ops))
    assert [q for q, _ in s] == [_Q_EDGE_CREATE, _Q_NODE_SET, _Q_EDGE_DEL]
    # disabled
    monkeypatch.setenv("DECADIC_KUZU_SHED_PRESSURE", "0")
    s, o = g._maybe_shed(list(stmts), list(ops))
    assert len(s) == 5
    g.close()


def test_skip_unchanged_disabled_is_parity(tmp_path, monkeypatch):
    """DECADIC_KUZU_SKIP_UNCHANGED=0 restores stage/defer for every attempt."""
    pytest.importorskip("kuzu")
    monkeypatch.setenv("DECADIC_KUZU_SKIP_UNCHANGED", "0")
    monkeypatch.setenv("DECADIC_KUZU_WRITE_MIN_CYCLES", "0")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_OPS", "999")
    monkeypatch.setenv("DECADIC_KUZU_FLUSH_S", "999")
    g = _kuzu_graph(tmp_path, "skip_off_kuzu")
    base = {"src": "a", "dst": "b", "kind": "scene_near", "weight": 0.5}
    for c in range(1, 11):
        g._persist_edge({**base, "count": c, "last_cycle": c})
    m = g.persistence_metrics()
    assert m["graph_writes_skipped_unchanged"] == 0
    assert m["graph_writes_staged_edge_scene_near"] == 10
    g.close()
