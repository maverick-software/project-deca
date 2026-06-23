from decadic.perception.scene_workspace import SceneWorkspace


def _obj(idx: int, *, uv=None, rel=None, confidence=0.9, kind_hint="object", **extra):
    return {
        "object_id": f"obj-{idx:04d}",
        "idx": idx,
        "centroid_uv": uv or [0.2 + idx * 0.1, 0.5],
        "relative": rel or [float(idx), 0.0, 1.0],
        "appearance": [float(idx + 1), 0.0, 0.0],
        "motion": [0.0, 0.0],
        "confidence": confidence,
        "presence": confidence,
        "kind_hint": kind_hint,
        "property_evidence": {"roundness": 0.8, "food_label": 1.0, "water": 1.0},
        **extra,
    }


def test_scene_workspace_rejects_semantic_property_evidence():
    ws = SceneWorkspace()
    ws.update([_obj(0)], focus_capacity=4)
    ent = ws.snapshot()["entities"][0]
    assert ent["property_evidence"]["roundness"] == 0.8
    assert "food_label" not in ent["property_evidence"]
    assert "water" not in ent["property_evidence"]


def test_scene_workspace_persists_occluded_entities_then_expires():
    ws = SceneWorkspace(ttl_cycles=2)
    ws.update([_obj(0)], focus_capacity=4)
    eid = ws.snapshot()["entities"][0]["entity_id"]

    ws.update([], focus_capacity=4)
    snap = ws.snapshot()
    ent = snap["entities"][0]
    assert ent["entity_id"] == eid
    assert ent["occluded"] is True
    assert snap["occluded_count"] == 1

    ws.update([], focus_capacity=4)
    assert ws.snapshot()["entity_count"] == 1
    ws.update([], focus_capacity=4)
    assert ws.snapshot()["entity_count"] == 0


def test_scene_workspace_admits_extended_entities_and_prioritizes_looming():
    ws = SceneWorkspace()
    ws.update(
        [
            _obj(0, confidence=0.9, kind_hint="stuff"),
            _obj(1, confidence=0.5),
            _obj(2, confidence=0.7, looming=0.8),
        ],
        focus_capacity=1,
    )
    snap = ws.snapshot()
    focused = snap["focus_ids"]
    assert len(focused) == 1
    assert any(e["kind_hint"] == "stuff" and e["entity_role"] == "extended_entity" for e in snap["entities"])
    ent = next(e for e in snap["entities"] if e["entity_id"] == focused[0])
    assert ent["object_id"] == "obj-0002"


def test_scene_workspace_builds_anonymous_spatial_relations():
    ws = SceneWorkspace()
    ws.update(
        [
            _obj(0, uv=[0.2, 0.5], rel=[0.0, 0.0, 1.0]),
            _obj(1, uv=[0.7, 0.5], rel=[0.5, 0.0, 1.0]),
        ],
        focus_capacity=4,
    )
    kinds = {r["kind"] for r in ws.snapshot()["relations"]}
    assert "co_visible" in kinds
    assert "near" in kinds
    assert "left_of" in kinds or "right_of" in kinds


def test_scene_workspace_can_hold_more_entities_than_focus_cache():
    ws = SceneWorkspace()
    ws.update([_obj(i, uv=[0.05 + i * 0.035, 0.5]) for i in range(12)], focus_capacity=7, entity_capacity=32)
    snap = ws.snapshot()
    assert snap["entity_count"] == 12
    assert len(snap["focus_ids"]) == 7


def test_drive_attention_prefers_anonymous_energy_relief_when_energy_low():
    ws = SceneWorkspace()
    ws.update(
        [
            _obj(0, property_evidence={"predicts_energy_relief": 0.9, "roundness": 0.4}),
            _obj(1, property_evidence={"roundness": 0.9}),
        ],
        focus_capacity=1,
        attention_context={
            "energy_deficit": 0.9,
            "hydration_deficit": 0.0,
            "integrity_deficit": 0.0,
            "pain": 0.0,
            "priority": "explore",
        },
    )
    focused = ws.snapshot()["focus_ids"][0]
    ent = next(e for e in ws.snapshot()["entities"] if e["entity_id"] == focused)
    assert ent["object_id"] == "obj-0000"
    assert ent["attention_reasons"]["relief"] > 0.0
    assert "predicts_energy_relief" in ent["property_evidence"]


def test_drive_attention_prefers_anonymous_threat_when_integrity_low():
    ws = SceneWorkspace()
    ws.update(
        [
            _obj(0, property_evidence={"roundness": 0.9}),
            _obj(1, property_evidence={"predicts_integrity_loss": 0.8}),
        ],
        focus_capacity=1,
        attention_context={
            "energy_deficit": 0.0,
            "hydration_deficit": 0.0,
            "integrity_deficit": 0.75,
            "pain": 0.2,
            "priority": "avoid",
        },
    )
    focused = ws.snapshot()["focus_ids"][0]
    ent = next(e for e in ws.snapshot()["entities"] if e["entity_id"] == focused)
    assert ent["object_id"] == "obj-0001"
    assert ent["attention_reasons"]["threat"] > 0.0
