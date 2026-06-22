from decadic.state.body_map import (
    BODY_PARTS,
    BODY_MAP_VECTOR_DIM,
    EFFORT_VECTOR_DIM,
    body_pain_vector,
    effort_vector,
    flatten_body_map,
    most_pained_part,
    normalize_body_map,
    normalize_effort,
)
from decadic.memory.semantic_graph import LongTermGraph


def test_body_map_normalizes_missing_fields():
    bm = normalize_body_map({"pain": {"left_hand": 0.4}, "effort": [0.1]})
    assert bm["parts"] == list(BODY_PARTS)
    assert bm["pain"][BODY_PARTS.index("left_hand")] == 0.4
    assert bm["effort"][0] == 0.1
    assert len(flatten_body_map(bm)) == BODY_MAP_VECTOR_DIM
    assert body_pain_vector(bm)[BODY_PARTS.index("left_hand")] == 0.4
    assert most_pained_part(bm) == ("left_hand", 0.4)


def test_effort_vector_includes_aggregates():
    bm = normalize_body_map({"fatigue": {"right_foot": 0.3}})
    eff = normalize_effort({"effort_total": 0.2, "work_total": 0.1})
    vec = effort_vector(bm, eff)
    assert len(vec) == EFFORT_VECTOR_DIM
    assert vec[-5] == 0.2
    assert vec[-4] == 0.1


def test_ltm_allows_body_pain_belief_but_rejects_external_labels():
    g = LongTermGraph()
    nid = g.upsert_node([1.0, 0.0], cycle=1)
    updated = g.upsert_property_beliefs(
        nid,
        {
            "predicts_left_hand_pain": 0.7,
            "food_label": 1.0,
        },
        cycle=2,
    )
    snap = g.snapshot()
    beliefs = snap["nodes"][0]["property_beliefs"]
    assert updated == 1
    assert beliefs[0]["property_key"] == "predicts_left_hand_pain"
