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


def test_scene_workspace_focus_excludes_stuff_and_prioritizes_looming():
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
    ent = next(e for e in snap["entities"] if e["entity_id"] == focused[0])
    assert ent["object_id"] == "obj-0002"
    assert all(e["kind_hint"] != "stuff" for e in snap["entities"] if e["entity_id"] in focused)


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

