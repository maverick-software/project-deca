"""Joint-brace guidance system: the per-joint ROM curriculum law, the braced
standing statue (no external wrench, feet fully loaded, no glide), the NN
dimension/checkpoint migration, and the body model (ankles/box feet/16-channel
touch/friction) that the braces stand on.

The ROM-curriculum law is exercised as a pure function (no MuJoCo); the braces
and body model are exercised against a real ``HumanoidSim`` when MuJoCo is
available.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"


def _load_adapter(monkeypatch, mode: str = "legacy"):
    """(Re)load the adapter module fresh (brace knobs are read at import time).

    ``mode`` is accepted for call-site compatibility but is inert: the external
    support harness (and its curriculum modes) was replaced by the joint braces.
    """
    name = f"mujoco_decadic_adapter_brace_{mode}"
    spec = importlib.util.spec_from_file_location(name, ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Pure ROM-curriculum law tests (no MuJoCo) -----------------------------


def test_brace_ratchet_loosens_under_sustained_low_pe(monkeypatch):
    mod = _load_adapter(monkeypatch)
    tight, ema, dwell = 1.0, 1.0, 0.0
    prev = tight
    loosened = False
    for _ in range(400):
        tight, ema, dwell = mod.brace_ratchet(
            tightness=tight, pe_ema=ema, pe_now=0.0, dwell=dwell, dt=0.1
        )
        assert tight <= prev + 1e-9  # monotonic: ROM only ever widens
        prev = tight
        if tight < 1.0:
            loosened = True
    assert loosened  # a well-predicted joint earns range of motion
    assert tight < 1.0


def test_brace_ratchet_holds_under_high_pe(monkeypatch):
    mod = _load_adapter(monkeypatch)
    tight, ema, dwell = 1.0, 1.0, 0.0
    for _ in range(400):
        tight, ema, dwell = mod.brace_ratchet(
            tightness=tight, pe_ema=ema, pe_now=5.0, dwell=dwell, dt=0.1
        )
    assert tight == pytest.approx(1.0)  # a surprising joint stays welded
    assert dwell == pytest.approx(0.0)  # dwell never accrues under high PE


def test_brace_ratchet_dwell_resets_on_pe_spike(monkeypatch):
    mod = _load_adapter(monkeypatch)
    # Below threshold accrues dwell; a single spike above threshold resets it.
    _, _, dwell = mod.brace_ratchet(
        tightness=1.0, pe_ema=0.0, pe_now=0.0, dwell=1.0, dt=0.1
    )
    assert dwell > 1.0
    _, _, dwell2 = mod.brace_ratchet(
        tightness=1.0, pe_ema=1.0, pe_now=5.0, dwell=2.0, dt=0.1
    )
    assert dwell2 == pytest.approx(0.0)


def test_brace_ratchet_never_below_zero(monkeypatch):
    mod = _load_adapter(monkeypatch)
    tight, ema, dwell = mod.brace_ratchet(
        tightness=0.0, pe_ema=0.0, pe_now=0.0, dwell=mod.BRACE_DWELL_S, dt=1.0
    )
    assert tight >= 0.0  # fully free: tightness floors at 0


# --- Neural-stack dimension migration (needs torch, not MuJoCo) ------------


def test_forward_pred_dim_follows_actuator_count():
    from decadic.nn.config import neural_config_from_env

    cfg = neural_config_from_env("tiny")
    assert cfg.n_actuators == 21
    assert cfg.forward_pred_dim == 28  # CONTROLLABLE_PROPRIO_BASE (7) + 21


def test_checkpoint_reinitializes_motor_forward_heads_on_actuator_change(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    from decadic.nn.bundle import NeuralBundle

    # Save a 17-actuator brain (the pre-upgrade contract).
    monkeypatch.setenv("DECADIC_N_ACTUATORS", "17")
    old = NeuralBundle.try_build("ckpt-17")
    assert old is not None
    assert old.stack.fwd_l2.out_features == 24  # 7 + 17
    ckpt = tmp_path / "old_brain.pt"
    old.save(ckpt)

    # Rebuild at 21 actuators and load the old checkpoint: must NOT crash; the
    # motor + forward heads keep their fresh (resized) initialization.
    monkeypatch.setenv("DECADIC_N_ACTUATORS", "21")
    new = NeuralBundle.try_build("ckpt-21")
    assert new is not None
    assert new.stack.fwd_l2.out_features == 28
    new.load(ckpt)  # would raise on the size mismatch without the shape filter
    assert new.stack.fwd_l2.out_features == 28
    assert new.stack.cfg.n_actuators == 21


# --- MuJoCo-backed brace behavior ------------------------------------------


def test_body_spawns_with_braces_off_by_default(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        s = sim.snapshot()
        assert s.braces_enabled is False
        assert s.brace_engaged == pytest.approx(0.0)
    finally:
        sim.close()


def test_body_stands_as_braced_statue_with_no_external_wrench(monkeypatch):
    """The core no-glide guarantee: with no brain command the braces hold the body
    upright on FULLY loaded feet, the root never receives an external wrench, and
    it barely drifts (it cannot skate because nothing unloads the feet)."""
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        s0 = sim.snapshot()
        assert s0.brace_engaged == pytest.approx(1.0)  # spawns fully welded
        assert s0.rom_mean == pytest.approx(0.0)

        x0 = np.array(s0.position[:2])
        for _ in range(int(5.0 / 0.05)):
            sim.step(0.05)
            # No external force EVER touches the torso (this is what killed the glide).
            assert float(np.abs(sim.data.xfrc_applied[sim.torso_id]).sum()) == 0.0
        s = sim.snapshot()
        assert abs(s.orientation[0]) < 0.2 and abs(s.orientation[1]) < 0.2  # upright
        assert s.position[2] > 1.1  # standing, not collapsed
        feet = s.part_loads.get("left_foot", 0.0) + s.part_loads.get("right_foot", 0.0)
        assert feet > 0.8  # feet bear ~full body weight -> real friction
        drift = float(np.linalg.norm(np.array(s.position[:2]) - x0))
        assert drift < 0.1  # statue does not skate
    finally:
        sim.close()


def test_joint_pe_from_brain_widens_rom(monkeypatch):
    """Per-joint PE rides in on the motor command and ratchets that joint's ROM
    open: sustained low PE frees the joints (the curriculum's whole point)."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(20.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        s = sim.snapshot()
        assert s.rom_mean > 0.0  # ROM widened under low prediction error
        assert s.brace_engaged < 1.0
        assert len(s.rom_frac) == nH  # per-hinge ROM telemetry present
        assert s.position[2] > 1.1  # still standing while loosening
    finally:
        sim.close()


def test_high_joint_pe_keeps_braces_welded(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(20.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [9.0] * nH}}
            )
            sim.step(0.05)
        s = sim.snapshot()
        assert s.rom_mean == pytest.approx(0.0)  # surprising joints stay welded
        assert s.brace_engaged == pytest.approx(1.0)
    finally:
        sim.close()


def test_recenter_preserves_earned_rom(monkeypatch):
    """Recenter reposes the body upright but must NOT wipe the ROM curriculum --
    re-welding is the separate reset_braces() action, not recentering."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(15.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        earned = sim.snapshot().rom_mean
        assert earned > 0.0  # ROM was earned under sustained low PE

        sim.recenter()
        s = sim.snapshot()
        # Earned ROM carries over unchanged...
        assert s.rom_mean == pytest.approx(earned)
        assert s.brace_engaged == pytest.approx(1.0 - earned)
        # ...and the body is re-posed upright at the stand height with zero drift.
        assert abs(s.position[0]) < 1e-6 and abs(s.position[1]) < 1e-6
        assert s.position[2] == pytest.approx(mod.STAND_ROOT_HEIGHT, abs=1e-6)
        assert abs(s.orientation[0]) < 1e-6 and abs(s.orientation[1]) < 1e-6

        # reset_braces() (the Reset ROM button) still re-welds from scratch.
        sim.reset_braces()
        assert sim.snapshot().rom_mean == pytest.approx(0.0)
    finally:
        sim.close()


def test_reset_braces_rewelds(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(15.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        assert sim.snapshot().rom_mean > 0.0  # loosened
        sim.reset_braces()
        s = sim.snapshot()
        assert s.rom_mean == pytest.approx(0.0)  # re-welded to the stand pose
        assert s.brace_engaged == pytest.approx(1.0)
    finally:
        sim.close()


def test_braces_can_be_switched_off_and_on(monkeypatch):
    """Master toggle: OFF relaxes every hinge to its native spring (free body the
    brain alone holds up) while preserving earned ROM; ON re-engages the braces."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        jid6 = sim.hinge_jid[6]
        # Explicit braces=True -> hinge far stiffer than native.
        assert sim.snapshot().braces_enabled is True
        assert float(sim.model.jnt_stiffness[jid6]) > sim._native_stiff[6] + 1.0

        # Earn some ROM, then switch the braces OFF.
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(15.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        earned = sim.snapshot().rom_mean
        assert earned > 0.0

        sim.set_braces_enabled(False)
        sim.step(0.05)
        s_off = sim.snapshot()
        assert s_off.braces_enabled is False
        # Hinge relaxed to native softness; earned ROM is preserved, not erased.
        assert float(sim.model.jnt_stiffness[jid6]) == pytest.approx(sim._native_stiff[6])
        assert s_off.rom_mean == pytest.approx(earned)

        # Switch back ON: braces re-engage at the preserved tightness.
        sim.set_braces_enabled(True)
        sim.step(0.05)
        assert sim.snapshot().braces_enabled is True
        assert float(sim.model.jnt_stiffness[jid6]) > sim._native_stiff[6] + 1.0
    finally:
        sim.close()


def test_lifeless_relaxes_braces_to_native(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        # Welded by default: a braced hinge is much stiffer than its native spring.
        jid = sim.hinge_jid[6]
        assert float(sim.model.jnt_stiffness[jid]) > sim._native_stiff[6] + 1.0
        sim.set_lifeless(True)
        sim.step(0.05)
        # Lifeless restores native joint springs (true ragdoll, no held statue).
        assert float(sim.model.jnt_stiffness[jid]) == pytest.approx(sim._native_stiff[6])
    finally:
        sim.close()


def test_snapshot_and_observation_expose_brace_telemetry(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        sim.step(0.05)
        snap = sim.snapshot()
        assert hasattr(snap, "rom_mean") and hasattr(snap, "brace_engaged")
        assert len(snap.rom_frac) == len(sim.hinge_qpos_adr)
        body = mod.build_body_observation(snap)["world_state"]["body"]
        assert "rom_mean" in body and "brace_engaged" in body and "rom_frac" in body
        # No leftover external-harness telemetry.
        assert "support_frac" not in body and "upright_assist" not in body
    finally:
        sim.close()


# --- MuJoCo-backed body model (the braces stand on this) -------------------


def test_model_has_ankles_box_feet_sensors_and_bears_weight(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        assert sim.model.nu == 21  # 21-actuator contract (ankles added)

        jnames = {
            mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(sim.model.njnt)
        }
        for nm in ("right_ankle_y", "right_ankle_x", "left_ankle_y", "left_ankle_x"):
            assert nm in jnames

        for foot in ("right_foot", "left_foot"):
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, foot)
            assert sim.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX  # flat sole

        # Full-body touch: exactly 16 channels (Option B, waist merged), with the
        # original feet/hands FIRST so the channel order stays stable.
        names = [n for n, _ in sim.touch_sensors]
        assert len(names) == 16
        assert names[:4] == [
            "touch_right_foot",
            "touch_left_foot",
            "touch_right_hand",
            "touch_left_hand",
        ]
        for nm in (
            "touch_torso",
            "touch_head",
            "touch_waist",
            "touch_butt",
            "touch_right_thigh",
            "touch_left_thigh",
            "touch_right_shin",
            "touch_left_shin",
            "touch_right_uarm",
            "touch_left_uarm",
            "touch_right_larm",
            "touch_left_larm",
        ):
            assert nm in names

        # Braced statue stands and bears its own weight on its feet.
        max_foot_load = 0.0
        for _ in range(120):
            sim.step(0.05)
            s = sim.snapshot()
            max_foot_load = max(max_foot_load, s.foot_load_l + s.foot_load_r)
        snap = sim.snapshot()
        assert snap.position[2] > 1.1
        assert max_foot_load > 0.8  # feet carry ~full weight (braces never lift)
        assert hasattr(snap, "hand_load_l") and hasattr(snap, "hand_load_r")
    finally:
        sim.close()


def test_arms_have_traction_and_stronger_actuators(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        assert sim.model.nu == 21  # friction/gear changes do NOT alter the contract

        # Hands and forearms have frictional (condim=3) contact for traction.
        for geom in ("right_hand", "left_hand", "right_larm", "left_larm"):
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert int(sim.model.geom_condim[gid]) == 3
            assert float(sim.model.geom_friction[gid][0]) > 0.0

        # Arm actuators were strengthened to gear 60.
        for act in (
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ):
            aid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            assert float(sim.model.actuator_gear[aid][0]) == pytest.approx(60.0)

        assert set(sim._hand_touch_adr) == {"l", "r"}
    finally:
        sim.close()


def test_uses_implicitfast_integrator(monkeypatch):
    """The braces are stiff joint springs; the model must use the semi-implicit
    integrator that integrates them stably (explicit RK4 at this dt diverges)."""
    mujoco = pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        assert int(sim.model.opt.integrator) == int(
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        )
    finally:
        sim.close()


def test_part_loads_live_and_flow_into_observation(monkeypatch):
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        max_total = 0.0
        for _ in range(120):
            sim.step(0.05)
            snap = sim.snapshot()
            assert len(snap.part_loads) == 16  # full-body touch map present
            max_total = max(max_total, sum(snap.part_loads.values()))
        assert max_total > 0.0

        obs = mod.build_body_observation(sim.snapshot())
        # Ordered list (tactile target) lines up with the 16 touch channels.
        assert len(obs["proprioception"]["part_loads"]) == 16
        body_loads = obs["world_state"]["body"]["part_loads"]
        assert isinstance(body_loads, dict) and len(body_loads) == 16
        assert "right_foot" in body_loads and "head" in body_loads
    finally:
        sim.close()


def test_default_geom_is_frictional_and_body_on_body_grips(monkeypatch):
    """Friction is explicit and uniform: the default geom class is condim=3 with a
    friction tuple, so the whole body (and spliced props) grip -- including
    body-on-body self-contact."""
    mujoco = pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "torso1")
        assert int(sim.model.geom_condim[gid]) == 3
        assert float(sim.model.geom_friction[gid][0]) > 0.0
        gid_shin = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "right_shin1")
        assert int(sim.model.geom_condim[gid_shin]) == 3
        assert float(sim.model.geom_friction[gid_shin][0]) > 0.0
    finally:
        sim.close()


# --- Stance / motion ROM braces -------------------------------------------

_HINGE_NAMES = [
    "abdomen_z", "abdomen_y", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee", "right_ankle_y", "right_ankle_x",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "left_ankle_y", "left_ankle_x",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]


def test_stance_resolve_preserves_defaults_and_clamps_overrides():
    """resolve(): unspecified hinges keep qpos0 verbatim (so stand == zeros), and
    only explicitly authored joints are deg->rad converted and range-clamped."""
    from decadic.embodiment import stances as st

    n = len(_HINGE_NAMES)
    defaults = [0.0] * n
    ranges = [(-math.inf, math.inf)] * n
    ranges[6] = (math.radians(-160), math.radians(-2))  # right_knee limits

    q_stand = st.resolve(st.get_stance("stand"), _HINGE_NAMES, defaults, ranges)
    assert q_stand == defaults  # stand has no overrides -> exact qpos0 (incl knee 0)

    q_af = st.resolve(st.get_stance("all_fours"), _HINGE_NAMES, defaults, ranges)
    assert q_af[5] == pytest.approx(math.radians(-90.0))  # right_hip_y override
    assert math.radians(-160) <= q_af[6] <= math.radians(-2)  # knee clamped in range


def test_stance_catalog_and_motion_flags():
    from decadic.embodiment import stances as st

    names = {c["name"] for c in st.catalog()}
    assert {
        "stand",
        "all_fours",
        "kneel_left",
        "kneel_right",
        "kneel_upright",
        "crawl",
        "sit_to_stand",
        "kneel_to_stand",
    } <= names
    assert st.get_stance("stand").is_motion is False
    assert st.get_stance("crawl").is_motion is True
    assert st.get_stance("kneel_to_stand").is_motion is True
    assert st.get_stance("does_not_exist").name == "stand"  # safe fallback


def test_motion_ref_interpolates_and_wraps():
    """A looping motion's reference moves with phase and wraps cleanly (phase 1 ==
    phase 0); a one-shot motion holds its final keyframe past phase 1."""
    from decadic.embodiment import stances as st

    n = len(_HINGE_NAMES)
    defaults = [0.0] * n
    crawl = st.get_stance("crawl")
    base = st.motion_ref(crawl, 0.0, _HINGE_NAMES, defaults)
    mid = st.motion_ref(crawl, 0.25, _HINGE_NAMES, defaults)
    assert any(abs(a - b) > 1e-6 for a, b in zip(base, mid))  # the gait moves
    wrap = st.motion_ref(crawl, 1.0, _HINGE_NAMES, defaults)
    assert wrap == pytest.approx(base)  # phase 1 wraps back to phase 0

    rise = st.get_stance("sit_to_stand")
    end = st.motion_ref(rise, 1.0, _HINGE_NAMES, defaults)
    held = st.motion_ref(rise, 1.5, _HINGE_NAMES, defaults)
    assert held == pytest.approx(end)  # one-shot holds the final keyframe

    kneel_rise = st.get_stance("kneel_to_stand")
    assert kneel_rise.root_quat == st.UPRIGHT_QUAT
    start = st.motion_ref(kneel_rise, 0.0, _HINGE_NAMES, defaults)
    end = st.motion_ref(kneel_rise, 1.0, _HINGE_NAMES, defaults)
    held = st.motion_ref(kneel_rise, 1.5, _HINGE_NAMES, defaults)
    assert held == pytest.approx(end)
    right_knee = _HINGE_NAMES.index("right_knee")
    right_hip_y = _HINGE_NAMES.index("right_hip_y")
    assert abs(end[right_knee]) < abs(start[right_knee])
    assert abs(end[right_hip_y]) < abs(start[right_hip_y])


def test_set_stance_reposes_without_rewelding(monkeypatch):
    """Selecting a stance re-poses the body but preserves manual brace state."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        from decadic.embodiment import stances as st

        # Earn ROM in the default stand, then switch to all_fours.
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(15.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        assert sim.snapshot().rom_mean > 0.0

        earned = sim.snapshot().rom_mean
        sim.set_stance("all_fours")
        s = sim.snapshot()
        assert s.stance == "all_fours"
        assert s.rom_mean == pytest.approx(earned)
        assert s.brace_engaged == pytest.approx(1.0 - earned)
        # The body is posed at the stance's spawn height (a low quadruped).
        spawn_z = st.get_stance("all_fours").root_z
        assert s.position[2] == pytest.approx(spawn_z, abs=0.15)
    finally:
        sim.close()


def test_every_stance_finite_with_no_external_wrench(monkeypatch):
    """No stance (static or motion) ever applies an external wrench to the torso,
    and every one stays finite -- the no-glide invariant holds in all postures."""
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        from decadic.embodiment import stances as st

        for name in st.STANCES:
            sim.set_stance(name)
            for _ in range(int(2.0 / 0.05)):
                sim.step(0.05)
                wrench = float(np.abs(sim.data.xfrc_applied[sim.torso_id]).max())
                assert wrench == 0.0  # internal joint braces only -- never a push
            assert bool(np.all(np.isfinite(sim.data.qpos)))
    finally:
        sim.close()


def test_motion_stance_advances_phase_and_retargets(monkeypatch):
    """A motion stance advances its phase over time and retargets the braced
    reference (q_ref) from the trajectory; a static stance leaves q_ref fixed."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        sim.set_stance("stand")
        q_static = list(sim._q_ref)
        for _ in range(int(1.0 / 0.05)):
            sim.step(0.05)
        assert list(sim._q_ref) == pytest.approx(q_static)  # static: fixed reference
        assert sim.snapshot().stance_phase == pytest.approx(0.0)

        sim.set_stance("crawl")
        q_start = list(sim._q_ref)
        for _ in range(int(1.0 / 0.05)):
            sim.step(0.05)
        assert sim.snapshot().stance_phase > 0.0  # phase advanced
        assert any(abs(a - b) > 1e-6 for a, b in zip(q_start, sim._q_ref))  # retargeted
    finally:
        sim.close()


def test_stance_sets_posture_aware_fall_floor(monkeypatch):
    """Low stances carry a low fall floor so the quadruped/kneel postures are not
    perpetually flagged as a fall (which standing's 0.7 m floor would do)."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        sim.set_stance("stand")
        assert sim._stance_fall_z == pytest.approx(0.7)
        sim.set_stance("all_fours")
        assert sim._stance_fall_z < 0.5
    finally:
        sim.close()


def test_movement_hold_keeps_braces_welded(monkeypatch):
    """Hold mode suspends the ROM curriculum: even under sustained low PE (which
    normally widens ROM) every joint stays fully welded -- no range-of-motion
    release -- so the movement is driven rigidly until disabled."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        sim.set_movement_hold(True)
        nu, nH = sim.model.nu, len(sim.hinge_qpos_adr)
        for _ in range(int(20.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        s = sim.snapshot()
        assert s.movement_hold is True
        assert s.rom_mean == pytest.approx(0.0)  # welded -- no ROM released
        assert s.brace_engaged == pytest.approx(1.0)

        # Releasing hold lets the ROM curriculum resume (joints re-earn range).
        sim.set_movement_hold(False)
        for _ in range(int(20.0 / 0.05)):
            sim.apply_action(
                {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "joint_pe": [0.0] * nH}}
            )
            sim.step(0.05)
        assert sim.snapshot().rom_mean > 0.0  # ratchet resumed once hold was off
    finally:
        sim.close()


def test_movement_hold_loops_one_shot_motion(monkeypatch):
    """Hold mode loops every motion, including the one-shot Rise: the phase wraps
    past 1.0 instead of clamping, so the movement runs continuously."""
    pytest.importorskip("mujoco")
    mod = _load_adapter(monkeypatch)
    sim = mod.HumanoidSim(vision=False, view=False, scene="default", braces=True)
    try:
        from decadic.embodiment import stances as st

        period = st.get_stance("sit_to_stand").period_s
        sim.set_stance("sit_to_stand")
        sim.set_movement_hold(True)
        for _ in range(int((period * 1.5) / 0.05)):
            sim.step(0.05)
        s = sim.snapshot()
        assert s.movement_hold is True
        assert s.stance_phase < 0.99  # wrapped (looping), not clamped at 1.0
    finally:
        sim.close()
