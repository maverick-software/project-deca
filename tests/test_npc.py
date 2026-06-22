"""Parent NPC: model parity, observability, scripted FSM, and reward isolation.

The parent is a second humanoid driven by applied forces (no actuators). These
tests pin the invariants the rest of the system relies on:
- adding the parent must not change the agent's 21-actuator / 42-value joint
  contract,
- the parent is perceivable as a ``kind:"npc"`` entity,
- the parent's own eating/drinking must never credit the *agent's* viability,
- the forage loop and the periodic fetch -> pickup -> deliver state machine
  advance and emit their distinct event types, dropping the gift far from the
  agent (so the agent must move to it).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter_npc", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter_npc"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Pure tests (no MuJoCo) ------------------------------------------------


def test_classify_events_ignores_npc_consumption():
    """The parent eating/drinking/offering does not feed the agent's reservoirs."""
    from decadic.state.viability import classify_events

    out = classify_events(
        [
            {"type": "npc_eat", "intensity": 1.0, "source": "npc"},
            {"type": "npc_drink", "intensity": 1.0, "source": "npc"},
            {"type": "offer", "intensity": 1.0, "source": "npc"},
        ],
        threshold=0.0,
    )
    assert out["energy_gain"] == 0.0
    assert out["hydration_gain"] == 0.0
    assert out["integrity_damage"] == 0.0
    assert out["stress"] == 0.0

    # The agent eating the offered gift (a normal "food" event) still credits it.
    fed = classify_events([{"type": "food", "intensity": 1.0, "source": "prop_food_gift"}], 0.0)
    assert fed["energy_gain"] > 0.0


def test_offer_builds_positive_social_affect():
    from decadic.state.world_graph import update_entity_affect

    affect = update_entity_affect({}, [{"type": "offer", "intensity": 1.0, "source": "npc"}])
    assert affect.get("npc", 0.0) > 0.0


def test_build_observation_includes_npc_entity():
    mod = _load_adapter_module()
    snap = mod.BodySnapshot(
        position=[0.0, 0.0, 1.4],
        orientation=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        joints=[0.0] * 34,
        contacts={"touch_right_foot": 1.0},
        props=[{"id": "npc", "kind": "npc", "position": [1.0, 2.0, 1.4]}],
    )
    obs = mod.build_body_observation(snap)
    ents = obs["world_state"]["entities"]
    npc = next((e for e in ents if e["id"] == "npc"), None)
    assert npc is not None
    assert npc["kind"] == "npc"
    assert npc["relative"] == [1.0, 2.0, 0.0]


# --- MuJoCo-backed tests ---------------------------------------------------


def test_npc_preserves_agent_actuator_contract():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["house", "food", "water", "npc"])
    try:
        # Parent adds bodies/joints but no actuators: the brain's contract holds.
        assert sim.model.nu == 21
        assert len(sim.hinge_qpos_adr) == 21
        assert sim.npc_torso is not None
        # Both movable gifts exist (food + water) but add no actuators.
        assert "prop_water_gift" in sim.water_bodies
        assert sim._gift_addr["food"][0] >= 0
        assert sim._gift_addr["water"][0] >= 0
        # The agent's proprioceptive joint vector tracks its hinges (21 x 2).
        snap = sim.snapshot()
        assert len(snap.joints) == 42
        # The parent is reported as a perceivable entity.
        assert any(p["id"] == "npc" and p["kind"] == "npc" for p in snap.props)
    finally:
        sim.close()


def test_npc_fsm_eats_drinks_and_offers():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["food", "water", "npc"])
    try:
        mj = sim._mj

        def place_npc(x: float, y: float) -> None:
            qa = sim.npc_root_qadr
            sim.data.qpos[qa : qa + 3] = (x, y, mod.STAND_ROOT_HEIGHT)
            mj.mj_forward(sim.model, sim.data)

        # 1) Park the parent on a real food morsel -> npc_eat, advance to water.
        food_name = next(n for n in sim.food_bodies if "gift" not in n)
        fp = sim.data.xpos[sim.food_bodies[food_name]]
        place_npc(float(fp[0]), float(fp[1]))
        evs = sim.scene_events(0)
        assert any(e["type"] == "npc_eat" for e in evs)
        assert sim._npc_phase == "seek_water"

        # 2) Park it on a water glass -> npc_drink, back to foraging (seek_food).
        water_name = next(n for n in sim.water_bodies if "gift" not in n)
        wp = sim.data.xpos[sim.water_bodies[water_name]]
        place_npc(float(wp[0]), float(wp[1]))
        evs = sim.scene_events(1)
        assert any(e["type"] == "npc_drink" for e in evs)
        assert sim._npc_phase == "seek_food"

        # 3) Cooldown elapsed -> leave the forage loop to fetch a gift. Park the
        # parent away from any morsel so it doesn't self-eat on this tick.
        sim._npc_next_deliver = 0.0
        place_npc(8.0, 8.0)
        sim.scene_events(2)
        assert sim._npc_phase == "pickup"
        assert sim._npc_item == "food"  # first delivery is food

        # 4) Reach a live food source -> pick the gift up and carry it.
        live_food = next(
            n for n in sim.food_bodies if "gift" not in n and n not in sim.eaten
        )
        lp = sim.data.xpos[sim.food_bodies[live_food]]
        place_npc(float(lp[0]), float(lp[1]))
        sim.scene_events(3)
        assert sim._npc_carry is True
        assert sim._npc_phase == "deliver"

        # 5) Approach the agent (origin) -> drop the gift far from it and offer.
        place_npc(2.5, 0.0)
        evs = sim.scene_events(4)
        offer = next((e for e in evs if e["type"] == "offer"), None)
        assert offer is not None
        assert offer["item"] == "food"
        assert sim._npc_phase == "seek_food"
        assert sim._npc_item == "water"  # alternates to water next time
        # The gift landed well beyond arm's reach -> the agent must move to it.
        gq = sim._gift_addr["food"][0]
        agent = sim.data.xpos[sim.torso_id]
        gx, gy = float(sim.data.qpos[gq]), float(sim.data.qpos[gq + 1])
        dist = ((gx - float(agent[0])) ** 2 + (gy - float(agent[1])) ** 2) ** 0.5
        assert dist > mod.EAT_RADIUS
        assert "prop_food_gift" not in sim.eaten
    finally:
        sim.close()


def test_parent_delivery_is_need_threshold_gated():
    """The parent provisions on a low reservoir, not on a bare timer."""
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["food", "water", "npc"])
    try:
        sim._npc_next_deliver = 0.0  # refractory satisfied

        # No reservoir info streamed yet -> fall back to the refractory (timer).
        sim._agent_reservoirs = None
        assert sim._parent_delivery_due(5.0) is True

        # Reservoirs known and comfortable -> the parent does NOT provision.
        sim._agent_reservoirs = {"hydration": 0.95, "energy": 0.9, "integrity": 1.0}
        assert sim._parent_delivery_due(5.0) is False

        # A reservoir dips below the threshold -> the parent provisions.
        sim._agent_reservoirs = {"hydration": 0.2, "energy": 0.9, "integrity": 1.0}
        assert sim._parent_delivery_due(5.0) is True

        # Still inside the refractory window -> never, regardless of need.
        sim._npc_next_deliver = 100.0
        assert sim._parent_delivery_due(5.0) is False
    finally:
        sim.close()


def test_parent_explicit_request_selects_visible_gift():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["food", "water", "npc"])
    try:
        assert sim._request_parent("water") is True
        assert sim._npc_item == "water"
        assert sim._npc_requested_item == "water"
        assert sim._npc_phase == "pickup"
        assert sim._caregiver_status() == "requested"

        sim._agent_reservoirs = {"hydration": 0.95, "energy": 0.4, "integrity": 0.3}
        assert sim._request_parent("care") is True
        assert sim._npc_item == "food"
        assert sim._npc_requested_item == "food"
        assert sim._npc_request_kind == "care"
    finally:
        sim.close()


def test_npc_walks_and_stays_upright():
    """The parent walks (alternating legs) toward its target and stays upright."""
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["food", "water", "npc"])
    try:
        r_hip_qadr = sim._npc_anim["r_hip"][0]
        l_hip_qadr = sim._npc_anim["l_hip"][0]
        start = np.array(sim.data.xpos[sim.npc_torso][:2], dtype=float)
        max_antiphase = 0.0
        for i in range(40):
            sim.step(0.05)
            sim.scene_events(i)
            diff = abs(
                float(sim.data.qpos[r_hip_qadr]) - float(sim.data.qpos[l_hip_qadr])
            )
            max_antiphase = max(max_antiphase, diff)
        pos = sim.data.xpos[sim.npc_torso]
        # Stays upright at ~standing height (small walk bob), not collapsed/launched.
        assert 1.2 < float(pos[2]) < 1.6
        # Travelled toward its target rather than gliding in place.
        moved = float(np.linalg.norm(np.array(pos[:2], dtype=float) - start))
        assert moved > 0.2
        # The legs actually alternate (antiphase hip swing) -> a real walk cycle,
        # not a frozen/skating pose.
        assert max_antiphase > 0.3
        # The proprio joint vector is still the agent's own hinges (21 x 2).
        assert len(sim.snapshot().joints) == 42
    finally:
        sim.close()


def test_npc_pause_freezes_in_place():
    """A frozen parent holds its pose and runs no FSM (UI 'pause parent')."""
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, elements=["food", "water", "npc"])
    try:
        mj = sim._mj
        # Park the parent right on a real food morsel: unfrozen it would eat and
        # advance the FSM, so this also proves the FSM is fully gated.
        food_name = next(n for n in sim.food_bodies if "gift" not in n)
        fp = sim.data.xpos[sim.food_bodies[food_name]]
        qa = sim.npc_root_qadr
        sim.data.qpos[qa : qa + 3] = (float(fp[0]), float(fp[1]), mod.STAND_ROOT_HEIGHT)
        mj.mj_forward(sim.model, sim.data)
        sim._npc_x = float(fp[0])
        sim._npc_y = float(fp[1])

        sim._npc_frozen = True
        sim._npc_next_deliver = 0.0  # delivery is due; a frozen parent must ignore it
        start = np.array(sim.data.xpos[sim.npc_torso][:2], dtype=float)
        start_phase = sim._npc_gait_phase

        for i in range(40):
            sim.step(0.05)
            evs = sim.scene_events(i)
            # No parental activity of any kind while frozen.
            assert not any(
                e["type"] in ("npc_eat", "npc_drink", "offer") for e in evs
            )

        pos = np.array(sim.data.xpos[sim.npc_torso][:2], dtype=float)
        assert float(np.linalg.norm(pos - start)) < 0.05  # stayed put
        assert sim._npc_gait_phase == start_phase  # gait did not advance
        assert sim._npc_phase == "seek_food"  # FSM did not transition
        assert sim._npc_carry is False
        assert food_name not in sim.eaten  # never consumed by the frozen parent

        # Releasing the parent lets it forage again (eats the morsel it stands on).
        sim._npc_frozen = False
        evs = sim.scene_events(40)
        assert any(e["type"] == "npc_eat" for e in evs)
    finally:
        sim.close()
