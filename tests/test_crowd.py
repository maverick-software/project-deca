"""Scripted NPC crowd: habitats, behaviors, XML, and the MuJoCo controller.

The crowd is a "village" of 8 collisionless, kinematically-animated NPCs, each
confined to its own habitat running a per-zone behavior, with co-located
respawning resources and one parent that provisions the learner on a need
threshold. These tests pin the invariants the rest of the system relies on:

- adding the crowd must not change the agent's 21-actuator / 42-value joint
  contract,
- every NPC stays inside its habitat,
- every NPC is perceivable as a ``kind:"npc"`` entity,
- NPC geoms are collisionless (render but never push the learner),
- an NPC eating never credits the *agent's* viability,
- habitat resources register + respawn and are net-additive,
- the parent provisions on a low reservoir (threshold), not on a fixed timer,
- the per-zone forage FSM advances and emits its distinct event types.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter_crowd", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter_crowd"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Pure tests (no MuJoCo) ------------------------------------------------


def test_active_habitats_has_one_parent():
    from decadic.embodiment.habitats import active_habitats

    habs = active_habitats()
    assert len(habs) == 8
    assert sum(1 for h in habs if h.is_parent) == 1


def test_crowd_size_env_override(monkeypatch):
    from decadic.embodiment.habitats import active_habitats, crowd_size

    monkeypatch.setenv("DECADIC_CROWD_SIZE", "3")
    assert crowd_size() == 3
    assert len(active_habitats()) == 3


def test_clamp_to_zone_pulls_inside():
    from decadic.embodiment.habitats import clamp_to_zone

    x, y = clamp_to_zone(10.0, 0.0, (0.0, 0.0), 2.0)
    assert math.isclose(math.hypot(x, y), 2.0, rel_tol=1e-6)
    # A point already inside is unchanged.
    assert clamp_to_zone(0.5, 0.5, (0.0, 0.0), 2.0) == (0.5, 0.5)


def test_behavior_poses():
    from decadic.embodiment import npc_behaviors as B

    # Walking legs swing in antiphase.
    pose = B.walk_pose(0.5)
    assert pose["r_hip"] * pose["l_hip"] <= 0.0 or abs(pose["r_hip"] - pose["l_hip"]) > 0.1
    # Sitting flexes hips and knees well past standing.
    assert B.sit_pose()["r_hip"] < B.stand_pose()["r_hip"] - 0.5
    # The sit<->stand blend stays in [0, 1].
    for t in (0.0, 1.0, 3.0, 5.5, 9.9):
        assert 0.0 <= B.sit_stand_blend(t) <= 1.0
    # lerp at the endpoints returns the endpoints.
    a, b = B.stand_pose(), B.sit_pose()
    assert B.lerp_pose(a, b, 0.0)["r_hip"] == pytest.approx(a["r_hip"])
    assert B.lerp_pose(a, b, 1.0)["r_hip"] == pytest.approx(b["r_hip"])


def test_parent_threshold_fades_and_floors(monkeypatch):
    from decadic.embodiment.habitats import parent_effective_threshold

    monkeypatch.setenv("DECADIC_PARENT_NEED_THRESHOLD", "0.5")
    monkeypatch.setenv("DECADIC_PARENT_FADE_PER_OFFER", "0.5")
    monkeypatch.setenv("DECADIC_PARENT_THRESHOLD_FLOOR", "0.1")
    assert parent_effective_threshold(0) == pytest.approx(0.5)
    assert parent_effective_threshold(1) == pytest.approx(0.25)
    # Fades monotonically but never below the floor.
    assert parent_effective_threshold(50) == pytest.approx(0.1)


def test_crowd_xml_emits_all_npcs_and_resources():
    from decadic.embodiment.npc_xml import crowd_scene_xml

    asset = Path(__file__).resolve().parents[1] / "assets" / "humanoid_body.xml"
    xml = crowd_scene_xml(asset)
    for i in range(8):
        assert f"npc{i}_torso" in xml
    assert "prop_food_gift_c" in xml and "prop_water_gift_c" in xml
    assert "prop_food_h0_1" in xml  # co-located habitat resources
    assert "prop_zone_0" in xml  # habitat marker


def test_element_lists_in_sync():
    """The supervisor's element set must match the adapter's selectable list."""
    from decadic.api.environment import VALID_ELEMENTS

    mod = _load_adapter_module()
    assert set(mod.SELECTABLE_ELEMENTS) == set(VALID_ELEMENTS)
    assert "crowd" in VALID_ELEMENTS


def test_village_builtin_preset_resolves_to_valid_elements():
    """The seeded 'village' preset still resolves to known world elements."""
    from decadic.api.environment import VALID_ELEMENTS
    from decadic.api.presets.store import BUILTIN_PRESETS

    by_id = {p["id"]: p for p in BUILTIN_PRESETS}
    assert "village" in by_id
    village = by_id["village"]
    assert "crowd" in village["elements"]
    for el in village["elements"]:
        assert el in VALID_ELEMENTS


# --- MuJoCo-backed tests ---------------------------------------------------


def _crowd_sim(mod):
    return mod.HumanoidSim(vision=False, view=False, elements=["crowd", "house"])


def test_crowd_preserves_agent_actuator_contract():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        assert sim.model.nu == 21
        assert len(sim.hinge_qpos_adr) == 21
        assert sim.crowd is not None
        assert len(sim.crowd.npcs) == 8
        snap = sim.snapshot()
        assert len(snap.joints) == 42  # agent's own hinges only (21 x 2)
    finally:
        sim.close()


def test_crowd_all_perceived_as_npc():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        snap = sim.snapshot()
        ids = {p["id"] for p in snap.props if p["kind"] == "npc"}
        assert ids == {f"npc{i}" for i in range(8)}
    finally:
        sim.close()


def test_crowd_geoms_are_collisionless():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        mj = sim._mj
        for g in range(sim.model.ngeom):
            bid = int(sim.model.geom_bodyid[g])
            name = mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith("npc"):
                assert int(sim.model.geom_contype[g]) == 0
                assert int(sim.model.geom_conaffinity[g]) == 0
    finally:
        sim.close()


def test_crowd_zone_confinement():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        for i in range(150):
            sim.step(0.02)
            sim.scene_events(i)
        for npc in sim.crowd.npcs:
            p = sim.data.xpos[npc.torso_body]
            cx, cy = npc.habitat.center
            d = math.hypot(float(p[0]) - cx, float(p[1]) - cy)
            assert d <= npc.habitat.radius + 1.0, f"{npc.entity_id} left its zone ({d:.2f})"
    finally:
        sim.close()


def test_crowd_resources_register_and_respawn():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        # Net-additive habitat resources are registered as consumables.
        assert any(n.startswith("prop_food_h") for n in sim.food_bodies)
        assert any(n.startswith("prop_water_h") for n in sim.water_bodies)
        assert len(sim.food_bodies) >= 8
        name = next(n for n in sim.food_bodies if n.startswith("prop_food_h"))
        sim._consume(name)
        assert name in sim.eaten
        sim._respawn(name)
        assert name not in sim.eaten
    finally:
        sim.close()


def test_crowd_credit_isolation():
    """A crowd NPC eating emits npc_eat (ignored by the agent's reservoirs)."""
    pytest.importorskip("mujoco")
    from decadic.state.viability import classify_events

    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        mj = sim._mj
        forager = next(
            n for n in sim.crowd.npcs if n.habitat.behavior == "forage" and not n.is_parent
        )
        # Park the forager on a live food morsel inside its own zone.
        food = next(
            n for n in sim.food_bodies
            if n.startswith("prop_food_h") and _in_zone(sim, n, forager)
        )
        fp = sim.data.xpos[sim.food_bodies[food]]
        _place(sim, forager, float(fp[0]), float(fp[1]))
        evs = sim.crowd.events(0, float(sim.data.time))
        assert any(e["type"] == "npc_eat" for e in evs)
        out = classify_events(evs, threshold=0.0)
        assert out["energy_gain"] == 0.0 and out["hydration_gain"] == 0.0
    finally:
        sim.close()


def test_crowd_forage_fsm_transition():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        forager = next(
            n for n in sim.crowd.npcs if n.habitat.behavior == "forage" and not n.is_parent
        )
        forager.phase = "seek_food"
        food = next(
            n for n in sim.food_bodies
            if n.startswith("prop_food_h") and _in_zone(sim, n, forager)
        )
        fp = sim.data.xpos[sim.food_bodies[food]]
        _place(sim, forager, float(fp[0]), float(fp[1]))
        sim.crowd.events(0, float(sim.data.time))
        assert forager.phase == "seek_water"  # advanced after eating
    finally:
        sim.close()


def test_crowd_parent_delivers_on_need_not_timer():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        parent = next(n for n in sim.crowd.npcs if n.is_parent)
        # Park the parent at its zone center (away from morsels) so foraging
        # doesn't fire; make the refractory satisfied.
        cx, cy = parent.habitat.center
        _place(sim, parent, cx, cy)
        parent.phase = "seek_food"
        parent.next_deliver = 0.0

        # Sated agent -> no provisioning even though the refractory has elapsed.
        sim.crowd.set_reservoirs({"hydration": 0.95, "energy": 0.95, "integrity": 1.0})
        sim.crowd.events(1, 5.0)
        assert parent.phase == "seek_food"

        # Missing reservoir telemetry -> no automatic timer provisioning.
        sim.crowd.set_reservoirs(None)
        sim.crowd.events(2, 5.5)
        assert parent.phase == "seek_food"

        # Deprived agent -> the parent leaves to fetch a gift (need threshold).
        sim.crowd.set_reservoirs({"hydration": 0.2, "energy": 0.9, "integrity": 1.0})
        _place(sim, parent, cx, cy)  # re-park (events may have nudged nothing, be safe)
        sim.crowd.events(3, 6.0)
        assert parent.phase == "pickup"
    finally:
        sim.close()


def test_crowd_parent_pickup_and_offer():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        parent = next(n for n in sim.crowd.npcs if n.is_parent)
        parent.item = "food"
        parent.phase = "pickup"
        # Reach a food source in zone -> pick the gift up and carry it.
        food = next(
            n for n in sim.food_bodies
            if n.startswith("prop_food_h") and _in_zone(sim, n, parent)
        )
        fp = sim.data.xpos[sim.food_bodies[food]]
        _place(sim, parent, float(fp[0]), float(fp[1]))
        sim.crowd.events(0, float(sim.data.time))
        assert parent.carry is True and parent.phase == "deliver"

        # Arrive at the drop point (zone edge toward the agent) -> offer + drop.
        dx, dy = sim.crowd._drop_point(parent)
        _place(sim, parent, dx, dy)
        evs = sim.crowd.events(1, float(sim.data.time))
        offer = next((e for e in evs if e["type"] == "offer"), None)
        assert offer is not None
        assert offer["item"] == "food"
        assert parent.phase == "seek_food" and parent.carry is False
        assert parent.offers == 1
        assert sim.crowd.last_offer_item == "food"
        assert sim.crowd.delivery_count == 1
    finally:
        sim.close()


def test_crowd_parent_counts_as_caregiver_and_accepts_request():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = _crowd_sim(mod)
    try:
        snap = sim.snapshot()
        assert snap.caregiver_parent_present is True
        assert snap.caregiver_kind == "crowd_parent"

        assert sim._request_parent("water") is True
        parent = next(n for n in sim.crowd.npcs if n.is_parent)
        assert parent.item == "water"
        assert parent.phase == "pickup"
        assert sim.crowd.requested_item == "water"
        assert sim._caregiver_status() == "requested"
        assert sim._caregiver_pending_request() is True
    finally:
        sim.close()


# --- helpers ---------------------------------------------------------------


def _place(sim, npc, x: float, y: float) -> None:
    """Teleport an NPC's root to (x, y) and refresh world positions."""
    qa = npc.root_qadr
    sim.data.qpos[qa : qa + 3] = (x, y, 1.30)
    npc.x, npc.y = x, y
    sim._mj.mj_forward(sim.model, sim.data)


def _in_zone(sim, name: str, npc) -> bool:
    p = sim.data.xpos[sim.food_bodies.get(name) or sim.water_bodies[name]]
    cx, cy = npc.habitat.center
    return math.hypot(float(p[0]) - cx, float(p[1]) - cy) <= npc.habitat.radius + 0.5
