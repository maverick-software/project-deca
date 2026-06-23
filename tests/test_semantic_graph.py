"""Long-term knowledge graph: the persistent, unbounded relational memory.

Covers the LongTermGraph store in isolation (upsert / cosine re-identification /
unbounded growth / edges / consolidation / SQLite persistence) and its wiring
into working memory (reinstatement rebinds to the long-term id; the no-LTM path
stays byte-identical; the long-term graph grows while working memory stays
bounded; the feature is on by default).
"""

from __future__ import annotations

from decadic import config
from decadic.memory.semantic_graph import LongTermGraph
from decadic.state.working_memory import MemorySlot, WorkingMemory

WM_CAP = 12  # the bounded "now" capacity the long-term graph must grow past


def _prop(appearance, *, idx=0, uv=(0.5, 0.5), rel=(1.0, 0.0, 0.0), presence=0.9):
    return {
        "idx": idx,
        "appearance": list(appearance),
        "uv": list(uv),
        "relative": list(rel),
        "presence": presence,
    }


# --- store in isolation ----------------------------------------------------


def test_upsert_then_match_reidentifies():
    g = LongTermGraph()
    a = [1.0, 0.0, 0.0, 0.0]
    id1 = g.upsert_node(a, cycle=1)
    assert id1 == "ent-00001"
    # A near-identical sighting EMA-updates the *same* node (no duplicate).
    id2 = g.upsert_node([1.0, 0.02, 0.0, 0.0], cycle=2)
    assert id2 == id1
    assert g.counts() == (1, 0)
    # match re-identifies a similar appearance and rejects an orthogonal one.
    assert g.match([0.95, 0.0, 0.0, 0.0]) == id1
    assert g.match([0.0, 1.0, 0.0, 0.0]) is None


def test_match_threshold_respected():
    g = LongTermGraph(match_threshold=0.99)
    a = g.upsert_node([1.0, 0.0], cycle=1)
    # 0.99 threshold: a 0.92-cosine sighting is NOT the same object.
    assert g.match([1.0, 0.45]) is None
    assert g.match([1.0, 0.001]) == a


def test_growth_is_unbounded():
    g = LongTermGraph()
    n = WM_CAP * 4  # far past the bounded working-memory capacity
    for i in range(n):
        v = [0.0] * 64
        v[i] = 1.0  # orthogonal one-hot -> every object is distinct
        g.upsert_node(v, cycle=i)
    assert g.counts()[0] == n


def test_edges_bump_and_snapshot_totals():
    g = LongTermGraph()
    a = g.upsert_node([1.0, 0.0, 0.0], cycle=1)
    b = g.upsert_node([0.0, 1.0, 0.0], cycle=1)
    g.bump_edge(a, b, cycle=1)
    g.bump_edge(b, a, cycle=2)  # undirected co-occurrence -> same edge
    snap = g.snapshot()
    assert snap["total_nodes"] == 2
    assert snap["total_edges"] == 1
    assert {snap["edges"][0]["source"], snap["edges"][0]["target"]} == {a, b}
    # snapshot nodes carry a deterministic appearance hue and a degree count.
    hues = {nd["id"]: nd["appearance_hash"] for nd in snap["nodes"]}
    assert all(0 <= h < 360 for h in hues.values())
    assert all(nd["degree"] == 1 for nd in snap["nodes"])


def test_snapshot_reports_render_window_and_parallel_edge_density():
    g = LongTermGraph()
    a = g.upsert_node([1.0, 0.0, 0.0], cycle=1)
    b = g.upsert_node([0.0, 1.0, 0.0], cycle=1)
    c = g.upsert_node([0.0, 0.0, 1.0], cycle=1)
    g.bump_edge(a, b, kind="co_occurrence", cycle=1)
    g.bump_edge(a, b, kind="scene_near", cycle=1)
    g.bump_edge(a, c, kind="scene_left_of", cycle=1)

    capped = g.snapshot(limit=2)
    assert capped["total_nodes"] == 3
    assert capped["total_edges"] == 3
    assert capped["rendered_nodes"] == 2
    assert capped["truncated_nodes"] is True
    assert capped["truncated_edges"] is True
    assert capped["edge_kind_counts"]["co_occurrence"] == 1
    assert capped["edge_kind_counts"]["scene_near"] == 1
    assert max(capped["edge_pair_counts"].values()) == 2

    full = g.snapshot(limit=0)
    assert full["rendered_nodes"] == 3
    assert full["rendered_edges"] == 3
    assert full["truncated_nodes"] is False
    assert full["truncated_edges"] is False
    assert all("count" in e and "last_cycle" in e for e in full["edges"])


def test_consolidate_commits_stable_and_links_copresent():
    g = LongTermGraph()
    stable_a = MemorySlot(entity_id="obj-0", appearance=[1.0, 0.0, 0.0], seen_count=3)
    stable_b = MemorySlot(entity_id="obj-1", appearance=[0.0, 1.0, 0.0], seen_count=3)
    fresh = MemorySlot(entity_id="obj-2", appearance=[0.0, 0.0, 1.0], seen_count=1)
    ids = g.consolidate([stable_a, stable_b, fresh], cycle=5, min_seen=2)
    assert len(ids) == 2  # the once-seen slot is below the stability gate
    assert g.counts() == (2, 1)  # one co-occurrence edge between the two stable nodes


def test_oracle_slots_without_appearance_are_skipped():
    g = LongTermGraph()
    oracle_like = MemorySlot(entity_id="bear", kind="animal", seen_count=9, appearance=None)
    assert g.consolidate([oracle_like], cycle=1, min_seen=2) == []
    assert g.counts() == (0, 0)


def test_persistence_round_trip(tmp_path):
    db = tmp_path / "graph.sqlite"
    g = LongTermGraph(db)
    a = g.upsert_node([1.0, 0.0, 0.0], cycle=1)
    b = g.upsert_node([0.0, 1.0, 0.0], cycle=1)
    g.bump_edge(a, b, cycle=1)
    # Re-open the same file: nodes, edges, appearances and the id counter survive.
    g2 = LongTermGraph(db)
    assert g2.counts() == (2, 1)
    assert g2.match([1.0, 0.0, 0.0]) == a
    assert g2.upsert_node([0.0, 0.0, 1.0], cycle=2) == "ent-00003"


def test_backup_restore_between_stores(tmp_path):
    src = LongTermGraph()  # in-memory
    a = src.upsert_node([1.0, 0.0, 0.0], cycle=1)
    src.upsert_node([0.0, 1.0, 0.0], cycle=1)
    snap_path = tmp_path / "snap.sqlite"
    src.backup_to(snap_path)
    dst = LongTermGraph()
    dst.restore_from(snap_path)
    assert dst.counts()[0] == 2
    assert dst.match([1.0, 0.0, 0.0]) == a


def test_restore_missing_file_is_noop(tmp_path):
    g = LongTermGraph()
    g.upsert_node([1.0, 0.0], cycle=1)
    g.restore_from(tmp_path / "does_not_exist.sqlite")  # must not raise
    assert g.counts()[0] == 1


def test_clear_wipes_graph():
    g = LongTermGraph()
    g.upsert_node([1.0, 0.0], cycle=1)
    g.clear()
    assert g.counts() == (0, 0)


# --- wiring into working memory --------------------------------------------


def test_reidentify_rebinds_to_long_term_id():
    ltm = LongTermGraph()
    app = [1.0, 0.0, 0.0, 0.0]
    ent = ltm.upsert_node(app, cycle=1)  # the object is already known long-term
    wm = WorkingMemory(capacity=WM_CAP, decay=0.9)
    wm.integrate_discovered([_prop(app, uv=(0.2, 0.2))], reidentify=ltm.match)
    # Reinstatement: the slot adopts the persistent ent id instead of a fresh obj id.
    assert ent in wm.slots
    assert not any(k.startswith("obj-") for k in wm.slots)


def test_no_reidentify_path_is_unchanged():
    wm = WorkingMemory(capacity=WM_CAP, decay=0.9)
    wm.integrate_discovered([_prop([1.0, 0.0, 0.0, 0.0], uv=(0.2, 0.2))])
    # Oracle / no-LTM parity: coin the same anonymous id as before.
    assert set(wm.slots) == {"obj-0000"}


def test_long_term_graph_grows_while_working_memory_stays_bounded():
    ltm = LongTermGraph()
    for i in range(WM_CAP * 3):
        v = [0.0] * 64
        v[i] = 1.0
        slot = MemorySlot(entity_id=f"obj-{i}", appearance=v, seen_count=2)
        ltm.consolidate([slot], cycle=i, min_seen=2)
    assert ltm.counts()[0] == WM_CAP * 3  # unbounded long-term growth

    wm = WorkingMemory(capacity=4, decay=0.9, min_salience=0.01)
    for i in range(20):
        v = [0.0] * 64
        v[i] = 1.0
        wm.integrate_discovered([_prop(v, rel=(float(i), 0.0, 0.0))])
        assert len(wm.slots) <= 4  # the "now" buffer never exceeds capacity


def test_ltm_graph_enabled_by_default(monkeypatch):
    monkeypatch.delenv("DECADIC_LTM_GRAPH", raising=False)
    assert config.ltm_graph_enabled() is True
    monkeypatch.setenv("DECADIC_LTM_GRAPH", "0")
    assert config.ltm_graph_enabled() is False


def test_semantic_evidence_records_provisional_framework_graph():
    g = LongTermGraph()
    slot = MemorySlot(
        entity_id="obj-0",
        appearance=[1.0, 0.0, 0.0],
        seen_count=1,
        confidence=0.08,
        kind_hint="stuff",
        entity_role="extended_entity",
        precision=0.08,
        provisional=True,
        property_evidence={"shape_extent": 0.9, "floor_label": 1.0},
    )

    counts = g.record_semantic_evidence(
        [slot],
        events=[{"type": "collision", "intensity": 0.8}],
        scene_relationships=[{"src": "obj-0", "dst": "self", "kind": "near", "confidence": 0.7}],
        cycle=1,
    )

    assert counts["entities"] == 1
    assert counts["events"] == 1
    assert counts["relationships"] >= 2
    snap = g.snapshot()
    assert snap["total_nodes"] == 0
    assert snap["semantic"]["entities"] == 1
    assert snap["semantic"]["events"] == 1
    payload = next(iter(g._semantic["entity"].values()))["payload"]
    assert "floor_label" not in payload.get("property_keys", [])


def test_repeated_semantic_patterns_form_conclusions_and_values():
    g = LongTermGraph()
    slot = MemorySlot(
        entity_id="obj-0",
        appearance=[1.0, 0.0, 0.0],
        seen_count=4,
        confidence=0.6,
        precision=0.6,
        provisional=False,
    )
    for cycle in (1, 2, 3):
        g.record_semantic_evidence([slot], events=[{"type": "collision", "intensity": 0.8}], cycle=cycle)

    stats = g.semantic_stats()
    assert stats["correlations"] >= 1
    assert stats["conclusions"] >= 1
    assert stats["values"] >= 1


def test_ltm_match_cache_matches_bounded_candidate_window(monkeypatch):
    monkeypatch.setenv("DECADIC_LTM_MATCH_RECENT_CAP", "4")
    monkeypatch.setenv("DECADIC_LTM_MATCH_SALIENT_CAP", "4")
    g = LongTermGraph()
    for i in range(20):
        vec = [0.0] * 20
        vec[i] = 1.0
        slot = MemorySlot(
            entity_id=f"obj-{i}",
            appearance=vec,
            seen_count=3,
            confidence=1.0,
            precision=1.0,
        )
        g.consolidate([slot], cycle=i, min_seen=2)

    assert g.match([0.0] * 19 + [1.0]) is not None
    stats = g.match_cache_stats()
    assert stats["enabled"] is True
    assert stats["size"] <= 8
    assert stats["hits"] >= 1
