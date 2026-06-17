"""Working memory: assimilation / accommodation / eviction / decay (logical-layer Part B)."""

import math

from decadic.state.working_memory import WorkingMemory


def _ent(eid, x=0.0):
    return {"id": eid, "role": "entity", "kind": "box", "position": [x, 0.0, 0.0]}


def test_assimilate_and_accommodate():
    wm = WorkingMemory(capacity=8, decay=0.9)
    wm.integrate([_ent("a"), _ent("b")], {})
    assert {s.entity_id for s in wm.active_slots()} == {"a", "b"}
    # re-seeing 'a' refreshes it; 'c' is new
    wm.integrate([_ent("a", 2.0), _ent("c")], {})
    a = wm.slots["a"]
    assert a.salience == 1.0
    assert a.seen_count == 2
    assert a.position == [2.0, 0.0, 0.0]
    assert "c" in wm.slots


def test_object_permanence_then_fade():
    wm = WorkingMemory(capacity=8, decay=0.5, min_salience=0.05)
    wm.integrate([_ent("ghost")], {})
    # not seen again: salience halves each cycle but the slot persists for a while
    wm.integrate([], {})
    assert wm.slots["ghost"].salience == 0.5
    for _ in range(6):
        wm.integrate([], {})
    # eventually it decays below min_salience and is forgotten
    assert "ghost" not in wm.slots


def test_capacity_evicts_lowest_salience():
    wm = WorkingMemory(capacity=2, decay=0.9)
    wm.integrate([_ent("old")], {})
    for _ in range(3):
        wm.integrate([], {})  # 'old' decays
    wm.integrate([_ent("x"), _ent("y")], {})  # two fresh, high-salience entities
    assert len(wm.slots) == 2
    assert "old" not in wm.slots
    assert {"x", "y"} <= set(wm.slots)


def test_attention_vector_reflects_affect_and_salience():
    wm = WorkingMemory(capacity=8, decay=0.9)
    wm.integrate([_ent("a"), _ent("b")], {"a": 2.0, "b": -2.0})
    vec = wm.attention_vector(16)
    assert len(vec) == 16
    assert any(abs(v) > 0 for v in vec)
    # empty memory → zero vector
    assert WorkingMemory().attention_vector(8) == [0.0] * 8


def test_snapshot_marks_in_view():
    wm = WorkingMemory(capacity=8, decay=0.9)
    wm.integrate([_ent("seen")], {})
    wm.integrate([], {})  # advance a cycle without seeing it
    snap = wm.snapshot()
    slot = next(s for s in snap["slots"] if s["entity_id"] == "seen")
    assert slot["in_view"] is False


def test_snapshot_includes_render_fields():
    wm = WorkingMemory(capacity=8, decay=0.9)
    wm.integrate([_ent("a", 1.0)], {"a": -0.5})
    slot = wm.snapshot()["slots"][0]
    # the snapshot is self-sufficient for the Mind's Eye render
    for key in ("position", "heading", "audio_intensity", "last_event", "affective_weight"):
        assert key in slot
    assert slot["position"] == [1.0, 0.0, 0.0]
    assert slot["affective_weight"] == -0.5


def test_event_binding_sets_audio_and_decays():
    wm = WorkingMemory(capacity=8, decay=0.9)
    wm.integrate([_ent("bear")], {}, events=[{"type": "threat_near", "intensity": 0.8, "source": "bear"}])
    slot = wm.slots["bear"]
    assert slot.audio_intensity == 0.8
    assert slot.last_event == "threat_near"
    # next cycle with no event: audio fades but the last_event label persists
    wm.integrate([_ent("bear")], {}, events=[])
    assert wm.slots["bear"].audio_intensity < 0.8
    assert wm.slots["bear"].last_event == "threat_near"


def test_event_for_unknown_entity_is_ignored():
    wm = WorkingMemory(capacity=8, decay=0.9)
    # a sensor-named collision (no matching entity slot) must not crash or leak
    wm.integrate([_ent("box")], {}, events=[{"type": "collision", "intensity": 0.9, "source": "touch_right_foot"}])
    assert wm.slots["box"].audio_intensity == 0.0
    assert wm.slots["box"].last_event is None


def test_deposit_scene_ema():
    wm = WorkingMemory(capacity=8, decay=0.9, scene_alpha=0.5)
    wm.deposit_scene([2.0, 4.0])
    # first deposit seeds the latent verbatim
    assert wm.scene_latent == [2.0, 4.0]
    wm.deposit_scene([0.0, 0.0])
    # EMA: (1-0.5)*old + 0.5*new
    assert wm.scene_latent == [1.0, 2.0]
    # rms of [1, 2] = sqrt(2.5)
    assert wm.scene_rms() is not None
    assert abs(wm.scene_rms() - math.sqrt(2.5)) < 1e-9


def test_deposit_scene_dim_change_reseeds():
    wm = WorkingMemory(capacity=8, decay=0.9, scene_alpha=0.5)
    wm.deposit_scene([1.0, 1.0])
    wm.deposit_scene([3.0, 3.0, 3.0])  # encoder dim changed → reseed, no blend
    assert wm.scene_latent == [3.0, 3.0, 3.0]


def test_attention_vector_includes_scene_component():
    wm = WorkingMemory(capacity=8, decay=0.9, scene_blend=0.5)
    # scene latent alone (no slots) drives the attention vector
    wm.deposit_scene([1.0] * 64)
    vec = wm.attention_vector(8)
    assert len(vec) == 8
    assert all(abs(v - math.tanh(1.0)) < 1e-9 for v in vec)
    # with slots present, the scene is mixed in at scene_blend weight
    wm.integrate([_ent("a")], {})
    mixed = wm.attention_vector(8)
    entity_only = WorkingMemory(capacity=8, decay=0.9)
    entity_only.integrate([_ent("a")], {})
    ent_vec = entity_only.attention_vector(8)
    expected = [0.5 * e + 0.5 * math.tanh(1.0) for e in ent_vec]
    assert all(abs(m - x) < 1e-9 for m, x in zip(mixed, expected))


def test_snapshot_exposes_scene_latent():
    wm = WorkingMemory(capacity=8, decay=0.9)
    snap = wm.snapshot()
    assert snap["scene_latent_rms"] is None
    assert snap["scene_preview"] is None
    wm.deposit_scene([0.5] * 128)
    snap = wm.snapshot()
    assert snap["scene_latent_rms"] == 0.5
    preview = snap["scene_preview"]
    assert isinstance(preview, list) and len(preview) == 32
    assert all(abs(v - round(math.tanh(0.5), 4)) < 1e-9 for v in preview)


def test_heading_from_position_history():
    wm = WorkingMemory(capacity=8, decay=0.9)
    # entity walking in +y: heading should approach pi/2
    for y in (0.0, 1.0, 2.0, 3.0):
        wm.integrate([{"id": "m", "role": "entity", "kind": "box", "position": [0.0, y, 0.0]}], {})
    h = wm.slots["m"].heading()
    assert h is not None
    assert abs(h - (math.pi / 2)) < 1e-6
