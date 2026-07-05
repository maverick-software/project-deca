#!/usr/bin/env python3
"""MuJoCo humanoid → Decadic WebSocket body adapter.

Streams full-body senses (root pose, joint qpos/qvel, palm/sole touch, optional
egocentric vision) as ``ObservationMessage`` JSON and applies Decadic ``move``
actions via root-assist: a PD hold keeps the humanoid standing while pelvis
forces steer it. No RL training required for locomotion.

Examples::

    # Contract check without MuJoCo installed
    python scripts/mujoco_decadic_adapter.py --dry-run --steps 30

    # Persistent embodied run (Ctrl+C to stop); requires `pip install -e ".[body]"`
    python scripts/mujoco_decadic_adapter.py --steps 0 --vision --audio

    # Scenario runs: chasing bear (threat) or consumable food (reward)
    python scripts/mujoco_decadic_adapter.py --steps 0 --vision --view --scene bear
    python scripts/mujoco_decadic_adapter.py --steps 0 --vision --view --scene food

Environment:
    DECADIC_BODY_OBS_INTERVAL_MS   observation throttle (default 80)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decadic.config import (
    randomize_resources_enabled,
    resource_fence_margin,
    resource_min_dist,
    resource_placement_mode,
)
from decadic.state.body_map import BODY_PARTS, normalize_body_map, normalize_effort
from decadic.embodiment import stances as stance_lib
from decadic.embodiment.habitats import (
    active_habitats,
    parent_effective_threshold as _parent_effective_threshold,
)
from decadic.embodiment.npc_controller import CrowdController
from decadic.embodiment.npc_xml import crowd_scene_xml
from decadic.embodiment.resource_placement import scatter_positions

ASSETS_XML = Path(__file__).resolve().parents[1] / "assets" / "humanoid_body.xml"

# Scene prop snippets spliced into the base humanoid model before </worldbody>.
# Scene element snippets spliced into the base humanoid model before
# </worldbody>. Each entry is an independently selectable world element so a
# scenario can be composed from the UI (house, food, water, bear, ball, ...).
_HOUSE_XML = """
    <!-- House: static walls with a doorway facing the spawn, flat roof -->
    <body name="prop_house" pos="-5.0 4.0 0">
      <geom name="prop_house_wall_n" type="box" pos="0 2.0 1.1" size="2.0 0.12 1.1"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_wall_w" type="box" pos="-2.0 0 1.1" size="0.12 2.0 1.1"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_wall_e" type="box" pos="2.0 0 1.1" size="0.12 2.0 1.1"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_wall_s1" type="box" pos="-1.3 -2.0 1.1" size="0.7 0.12 1.1"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_wall_s2" type="box" pos="1.3 -2.0 1.1" size="0.7 0.12 1.1"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_lintel" type="box" pos="0 -2.0 2.0" size="0.6 0.12 0.2"
            rgba="0.75 0.6 0.45 1" condim="3"/>
      <geom name="prop_house_roof" type="box" pos="0 0 2.26" size="2.2 2.2 0.06"
            rgba="0.55 0.2 0.15 1" condim="3"/>
    </body>
"""

_FOOD_XML = """
    <!-- Snacks: morsels dropped within a couple of meters of the spawn -->
    <body name="prop_food_s1" pos="1.8 0.8 0.12">
      <geom name="prop_food_s1_geom" type="sphere" size="0.12" rgba="0.95 0.55 0.1 1" condim="3"/>
    </body>
    <body name="prop_food_s2" pos="-1.5 1.2 0.12">
      <geom name="prop_food_s2_geom" type="sphere" size="0.12" rgba="0.95 0.55 0.1 1" condim="3"/>
    </body>
    <body name="prop_food_s3" pos="0.8 -1.9 0.12">
      <geom name="prop_food_s3_geom" type="sphere" size="0.12" rgba="0.95 0.55 0.1 1" condim="3"/>
    </body>
    <body name="prop_food_s4" pos="-2.2 -1.0 0.12">
      <geom name="prop_food_s4_geom" type="sphere" size="0.12" rgba="0.95 0.55 0.1 1" condim="3"/>
    </body>
"""

_WATER_XML = """
    <!-- Water: drinkable glasses (translucent blue cylinders) within reach of spawn -->
    <body name="prop_water_w1" pos="1.2 -0.7 0.13">
      <geom name="prop_water_w1_geom" type="cylinder" size="0.08 0.13" rgba="0.3 0.55 0.95 0.7" condim="3"/>
    </body>
    <body name="prop_water_w2" pos="-1.1 -0.9 0.13">
      <geom name="prop_water_w2_geom" type="cylinder" size="0.08 0.13" rgba="0.3 0.55 0.95 0.7" condim="3"/>
    </body>
    <body name="prop_water_w3" pos="-0.6 1.7 0.13">
      <geom name="prop_water_w3_geom" type="cylinder" size="0.08 0.13" rgba="0.3 0.55 0.95 0.7" condim="3"/>
    </body>
"""

_MEDICAL_XML = """
    <!-- Medical kits: integrity-restoring supplies the learner can discover and use -->
    <body name="prop_medical_m1" pos="1.6 1.6 0.13">
      <geom name="prop_medical_m1_box" type="box" size="0.18 0.11 0.065" rgba="0.96 0.96 0.92 1" condim="3"/>
      <geom name="prop_medical_m1_handle" type="box" pos="0 0 0.12" size="0.11 0.018 0.025" rgba="0.08 0.10 0.12 1" contype="0" conaffinity="0"/>
      <geom name="prop_medical_m1_cross_a" type="box" pos="0 0 0.07" size="0.035 0.088 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>
      <geom name="prop_medical_m1_cross_b" type="box" pos="0 0 0.084" size="0.09 0.035 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>
    </body>
    <body name="prop_medical_m2" pos="-1.7 1.6 0.13">
      <geom name="prop_medical_m2_box" type="box" size="0.18 0.11 0.065" rgba="0.96 0.96 0.92 1" condim="3"/>
      <geom name="prop_medical_m2_handle" type="box" pos="0 0 0.12" size="0.11 0.018 0.025" rgba="0.08 0.10 0.12 1" contype="0" conaffinity="0"/>
      <geom name="prop_medical_m2_cross_a" type="box" pos="0 0 0.07" size="0.035 0.088 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>
      <geom name="prop_medical_m2_cross_b" type="box" pos="0 0 0.084" size="0.09 0.035 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>
    </body>
"""

_BEAR_XML = """
    <!-- Threat: low-friction free body driven toward the humanoid each step -->
    <body name="prop_bear" pos="6.0 0.0 0.55">
      <joint name="prop_bear_free" type="free" limited="false" armature="0" damping="0.5"/>
      <geom name="prop_bear_body" type="capsule" fromto="-0.45 0 0 0.45 0 0" size="0.32"
            rgba="0.45 0.28 0.12 1" condim="3" friction="0.3 .1 .1" mass="40"/>
      <geom name="prop_bear_head" type="sphere" pos="0.62 0 0.15" size="0.2"
            rgba="0.38 0.22 0.09 1" condim="3" mass="4"/>
    </body>
"""

_BALL_XML = """
    <!-- Ball: a light free-rolling sphere the agent can nudge, chase, and play with -->
    <body name="prop_ball" pos="1.0 1.5 0.22">
      <joint name="prop_ball_free" type="free" limited="false" armature="0" damping="0.02"/>
      <geom name="prop_ball_geom" type="sphere" size="0.18" rgba="0.95 0.85 0.2 1" condim="3" friction="0.6 .05 .05" mass="0.6"/>
    </body>
"""

_OBSTACLES_XML = """
    <!-- Pushable prop: free body the agent can shove around -->
    <body name="prop_box_red" pos="1.5 0.5 0.25">
      <joint name="prop_box_red_free" type="free" limited="false" armature="0" damping="0"/>
      <geom name="prop_box_red_geom" type="box" size="0.25 0.25 0.25" rgba="0.9 0.2 0.2 1" condim="3" friction="1 .1 .1" mass="2"/>
    </body>
    <!-- Static obstacles -->
    <geom name="prop_pillar_blue" type="cylinder" pos="-2.0 1.5 0.6" size="0.3 0.6" rgba="0.2 0.3 0.9 1" condim="3"/>
    <geom name="prop_sphere_green" type="sphere" pos="2.5 -2.0 0.4" size="0.4" rgba="0.2 0.8 0.3 1" condim="3"/>
"""


def _food_ring_xml() -> str:
    """Morsels in rings around the spawn so any wander direction crosses one."""
    spots = [(2.0, i * math.pi / 4) for i in range(8)]
    spots += [(5.0, math.pi / 4 + i * math.pi / 2) for i in range(4)]
    parts = ["\n    <!-- Consumables: the humanoid eats these by walking into them -->"]
    for n, (r, theta) in enumerate(spots, start=1):
        x = round(r * math.cos(theta), 2)
        y = round(r * math.sin(theta), 2)
        parts.append(
            f'''
    <body name="prop_food_{n}" pos="{x} {y} 0.12">
      <geom name="prop_food_{n}_geom" type="sphere" size="0.12" rgba="0.95 0.55 0.1 1" condim="3"/>
    </body>'''
        )
    return "".join(parts) + "\n"


def _npc_humanoid_xml(
    prefix: str = "npc_",
    pos: tuple[float, float, float] = (2.5, 2.5, 1.4),
) -> str:
    """A second humanoid identical to the agent, spliced into the worldbody.

    Derived from the agent asset so the parent looks exactly like the agent, but
    - every name is prefixed (``npc_``) to avoid collisions and to mark it
      non-agent for the discovery loops in :class:`HumanoidSim`,
    - cameras and touch sites are dropped (no egocentric eye, no sensors),
    - it declares no actuators/sensors, so ``model.nu`` is unchanged and the
      brain's 21-actuator motor contract still holds; the body is driven purely
      by applied forces (like the bear),
    - hinge joints are stiffened so the limbs hold a standing pose passively
      while the root is force-driven (no ragdoll; an active gait can be added
      later via qfrc on the npc hinges).

    Three movable "gifts" (free joints) ride along so the parent can carry and
    drop supplies for the agent: ``prop_food_gift`` (edible),
    ``prop_water_gift`` (drinkable), and ``prop_medical_gift`` (healing). They
    are named so the shared resource machinery credits the *agent* (not the
    parent) when the agent consumes them.
    """
    import re
    import xml.etree.ElementTree as ET

    # Strip XML comments first: the asset contains "(--scene)" inside a comment,
    # which MuJoCo tolerates but strict ElementTree rejects (no "--" in comments).
    raw = re.sub(r"<!--.*?-->", "", ASSETS_XML.read_text(encoding="utf-8"), flags=re.DOTALL)
    torso = ET.fromstring(raw).find("./worldbody/body[@name='torso']")
    if torso is None:  # pragma: no cover - asset ships with the repo
        raise RuntimeError("agent asset is missing <body name='torso'>")

    def _rewrite(el: "ET.Element") -> None:
        for child in list(el):
            if child.tag in ("camera", "site"):
                el.remove(child)
            else:
                _rewrite(child)
        if "name" in el.attrib:
            el.attrib["name"] = prefix + el.attrib["name"]
        if el.tag == "joint" and el.attrib.get("type", "hinge") != "free":
            el.attrib["stiffness"] = "60"
            el.attrib["damping"] = "6"

    _rewrite(torso)
    torso.attrib["pos"] = f"{pos[0]} {pos[1]} {pos[2]}"
    humanoid = ET.tostring(torso, encoding="unicode")

    gx, gy = pos[0], pos[1]
    food_gift = (
        f'\n    <body name="prop_food_gift" pos="{gx} {gy} 0.12">'
        f'\n      <joint name="prop_food_gift_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="prop_food_gift_geom" type="sphere" size="0.12"'
        f' rgba="0.95 0.55 0.1 1" condim="3" mass="0.2"/>'
        f"\n    </body>\n"
    )
    # A second movable gift the parent can carry and drop for the agent. Named
    # with "water" so the shared water machinery treats it as drinkable and
    # credits the *agent* (not the parent) when the agent drinks it.
    water_gift = (
        f'\n    <body name="prop_water_gift" pos="{gx + 0.4} {gy} 0.12">'
        f'\n      <joint name="prop_water_gift_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="prop_water_gift_geom" type="sphere" size="0.12"'
        f' rgba="0.2 0.5 0.95 1" condim="3" mass="0.2"/>'
        f"\n    </body>\n"
    )
    medical_gift = (
        f'\n    <body name="prop_medical_gift" pos="{gx + 0.8} {gy} 0.13">'
        f'\n      <joint name="prop_medical_gift_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="prop_medical_gift_box" type="box" size="0.18 0.11 0.065"'
        f' rgba="0.96 0.96 0.92 1" condim="3" mass="0.2"/>'
        f'\n      <geom name="prop_medical_gift_handle" type="box" pos="0 0 0.12"'
        f' size="0.11 0.018 0.025" rgba="0.08 0.10 0.12 1" contype="0" conaffinity="0"/>'
        f'\n      <geom name="prop_medical_gift_cross_a" type="box" pos="0 0 0.07"'
        f' size="0.035 0.088 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>'
        f'\n      <geom name="prop_medical_gift_cross_b" type="box" pos="0 0 0.084"'
        f' size="0.09 0.035 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>'
        f"\n    </body>\n"
    )
    return (
        "\n    <!-- Parent NPC: scripted humanoid + movable food, water & medical gifts -->\n    "
        + humanoid
        + food_gift
        + water_gift
        + medical_gift
    )


# Independently selectable world elements. ``food_ring`` is an internal variant
# (dense rings) used only by the legacy "food" scene; the UI offers the rest.
SCENE_ELEMENTS: dict[str, str] = {
    "house": _HOUSE_XML,
    "food": _FOOD_XML,
    "water": _WATER_XML,
    "medical_kit": _MEDICAL_XML,
    "bear": _BEAR_XML,
    "ball": _BALL_XML,
    "obstacles": _OBSTACLES_XML,
    "npc": _npc_humanoid_xml(),
    # Scripted "village": 8 collisionless kinematic NPCs confined to habitats,
    # each running a per-zone behavior, with co-located respawning resources and
    # one parent that provisions the learner on a need threshold.
    "crowd": crowd_scene_xml(ASSETS_XML),
    "food_ring": _food_ring_xml(),
}

# Elements offered in the UI scenario builder (in display order).
SELECTABLE_ELEMENTS: list[str] = [
    "house",
    "food",
    "water",
    "medical_kit",
    "bear",
    "ball",
    "obstacles",
    "npc",
    "crowd",
]

ALWAYS_ON_ELEMENTS: tuple[str, ...] = ("medical_kit",)


def normalize_scene_elements(elements: list[str]) -> list[str]:
    """De-duplicate scene elements and force always-on survival props."""
    chosen: list[str] = []
    seen: set[str] = set()
    for el in elements:
        key = str(el).strip().lower()
        if key and key in SCENE_ELEMENTS and key not in seen:
            seen.add(key)
            chosen.append(key)
    if chosen:
        for key in ALWAYS_ON_ELEMENTS:
            if key not in seen:
                seen.add(key)
                chosen.append(key)
    return chosen


# Legacy named scenes (CLI --scene / manual runs) expressed as element sets so
# existing commands and tests keep working after the move to composable scenes.
LEGACY_SCENES: dict[str, list[str]] = {
    "default": ["obstacles", "house", "food", "water", "medical_kit"],
    "bear": ["bear", "house", "food", "water", "medical_kit"],
    "food": ["food_ring", "house", "food", "water", "medical_kit"],
    # One-command crowd run: the village plus shelter.
    "village": ["crowd", "house"],
}

CONTACT_EVENT_THRESHOLD = 300.0  # above quiet-standing load (~110 N/foot); real impacts only
CONTACT_EVENT_SCALE = 800.0  # force mapped to intensity 0..1 around this scale
# Impact-based injury: a "collision" is a *sudden* spike in total contact force
# (a real momentum transfer) while the body still carries speed -- not the
# steady weight-bearing load of standing or lying down, which emits nothing.
# Intensity tracks impact energy (~speed^2), and a refractory cooldown keeps one
# tumble from billing damage every frame.
IMPACT_FORCE_SPIKE_N = 400.0  # one-frame rise in summed contact force that flags a hit
IMPACT_SPEED_REF = 3.5  # body speed (m/s) mapped to a full-intensity (1.0) impact
IMPACT_COOLDOWN_STEPS = 18  # ~1.5 s at 80 ms obs cadence between billed impacts
FALL_ROOT_HEIGHT = 0.7  # root z below this counts as fallen
# Spawn root height where the soles rest on the floor at the zeros stand pose
# (empirically ~1.295 settled; see scripts/_gen_stand_pose.py). The old 1.4 left
# the feet ~0.11 m in the air, so a no-lift braced body free-fell and toppled.
STAND_ROOT_HEIGHT = 1.30
FENCE_RADIUS = 18.0  # floor half-size is 20; auto-recenter before walking off the edge
VIEWS_EVERY = 2  # render debug camera views every Nth observation

# Procedural soundscape (16 kHz mono pcm16, one window per observation)
AUDIO_SR = 16000
AUDIO_WINDOW_S = 0.8
FOOTSTEP_MIN_FORCE = 80.0  # sole force above this produces an audible footstep

# Bear scene: chase drive (slides like a puck; catchable but escapable)
BEAR_DRIVE = 300.0  # horizontal chase force (N)
BEAR_DRAG = 430.0  # horizontal drag (N·s/m); terminal speed ≈ DRIVE/DRAG ≈ 0.7 m/s
BEAR_STANDOFF = 0.8  # stop driving when this close (m)
BEAR_CONTACT_FORCE = 80.0  # min summed contact force to count as a hit (N)
BEAR_HIT_COOLDOWN_STEPS = 25  # observations between collision events (~2 s)
THREAT_RADIUS = 3.0  # threat_near emitted inside this distance (m)
THREAT_COOLDOWN_STEPS = 12  # observations between threat_near events (~1 s)

# Parent NPC: scripted humanoid that forages, eats/drinks, then offers to the
# agent. It has no actuators (to preserve the brain's 21-actuator contract), so
# it is animated kinematically: each substep we write its root pose and joint
# angles directly from a walk cycle whose stride phase is locked to distance
# travelled, so the feet look planted (no skating) and the agent sees a
# convincing, stable walking parent to learn from.
NPC_STANDOFF = 0.6  # stop walking when this close to the target (m)
NPC_WALK_SPEED = 0.9  # forward ground speed while walking (m/s)
NPC_STRIDE_LENGTH = 0.8  # metres of travel per full gait cycle (foot-plant tuning)
NPC_TURN_RATE = 2.5  # max yaw turn toward the target (rad/s)
NPC_HIP_SWING = 0.5  # fore-aft thigh swing amplitude (rad)
NPC_KNEE_BASE = 0.1  # baseline knee bend so legs are never hyperextended (rad)
NPC_KNEE_BEND = 0.9  # extra knee flexion on the swing leg (rad)
NPC_ARM_SWING = 0.45  # shoulder counter-swing amplitude (rad)
NPC_BOB_AMP = 0.03  # vertical torso bob amplitude (m)
# Parental provisioning: the parent forages most of the time and only
# occasionally fetches a morsel and drops it for the agent. The drop lands far
# enough away (well beyond EAT/DRINK reach) that the agent must walk to it.
NPC_DELIVER_PERIOD_S = 30.0  # minimum sim-seconds between offers
NPC_DROP_DISTANCE = 3.0  # drop the gift this far from the agent (m)
NPC_PICKUP_RADIUS = 1.0  # reach a source within this distance to pick it up (m)

# Food / water scenes
EAT_RADIUS = 1.0  # root within this distance of a morsel consumes it (arm's reach, m)
DRINK_RADIUS = 1.0  # root within this distance of a glass drinks it (m)
MEDICAL_RADIUS = 1.0  # root within this distance of a kit uses it (m)
# Consumables respawn so the agent can sustain itself indefinitely. Wall-clock
# seconds; tuned for watchable replenishment (not the metabolic survival clock).
FOOD_RESPAWN_S = 30.0
WATER_RESPAWN_S = 25.0
MEDICAL_RESPAWN_S = 40.0
DIRECT_PROVISION_START_M = 2.4
DIRECT_PROVISION_TARGET_M = 0.18
DIRECT_PROVISION_DURATION_S = 2.4
EFFORT_FATIGUE_RISE_S = 2.5
EFFORT_FATIGUE_RECOVERY_S = 8.0
EFFORT_STRAIN_GAIN = 0.45
EFFORT_FATIGUE_PAIN_GAIN = 0.35
EFFORT_STRAIN_PAIN_GAIN = 0.25

# Joint PD hold toward the brace reference (standing) pose. Soft on purpose: the
# brain's equilibrium-point targets ride on top of the stiff joint braces below,
# which do the actual standing work.
PD_KP = 2.5
PD_KD = 0.15

# Total mass of the agent body in kilograms (200 lb ~= 90.7 kg). The agent's
# kinematic subtree is rescaled to this at load time: a heavier body raises the
# friction ceiling (mu*m*g) and lowers the actuator/brace torque-to-weight ratio,
# which together remove the supine "back-glide" without touching the joint
# braces. Override with DECADIC_BODY_MASS_KG to tune.
BODY_MASS_KG = float(os.environ.get("DECADIC_BODY_MASS_KG", "90.7"))

# --- Joint-brace guidance system --------------------------------------------
# Every hinge is braced toward the upright stand pose (q_ref) by a stiff,
# semi-implicit joint spring + damper (MuJoCo native jnt_stiffness / dof_damping;
# the model uses integrator="implicitfast" so this is numerically stable). NO
# external force ever touches the body, so the feet always carry the full weight
# (real friction, no glide) and the body holds itself up from the inside. Each
# joint starts welded (tightness 1.0 -> BRACE_STIFFNESS) and loosens monotonically
# toward its native softness as the brain's per-joint forward-model error falls.
BRACE_STIFFNESS = float(os.environ.get("DECADIC_BRACE_STIFFNESS", "1500.0"))
BRACE_DAMPING = float(os.environ.get("DECADIC_BRACE_DAMPING", "30.0"))
# A joint's ROM widens (tightness drops by BRACE_LOOSEN_STEP) once its smoothed
# forward-model prediction error stays below BRACE_PE_THRESH for BRACE_DWELL_S.
BRACE_PE_THRESH = float(os.environ.get("DECADIC_BRACE_PE_THRESH", "0.02"))
BRACE_PE_TAU = float(os.environ.get("DECADIC_BRACE_PE_TAU", "0.05"))
BRACE_DWELL_S = float(os.environ.get("DECADIC_BRACE_DWELL_S", "4.0"))
BRACE_LOOSEN_STEP = float(os.environ.get("DECADIC_BRACE_LOOSEN_STEP", "0.04"))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def brace_ratchet(
    *,
    tightness: float,
    pe_ema: float,
    pe_now: float,
    dwell: float,
    dt: float,
) -> tuple[float, float, float]:
    """Pure per-joint ROM-curriculum law (one control tick).

    Returns ``(tightness, pe_ema, dwell)``. ``tightness`` in [0, 1] is how
    welded the joint is to its brace reference: 1.0 = fully braced (small ROM),
    0.0 = native softness (full ROM, the brain in control). The brain's
    proprioceptive forward-model error for this joint (``pe_now``) is smoothed
    into ``pe_ema``; once that stays below ``BRACE_PE_THRESH`` for ``BRACE_DWELL_S``
    the joint earns a ROM step and ``tightness`` drops by ``BRACE_LOOSEN_STEP``.
    Monotonic: ROM only ever widens (the joint never re-tightens itself), so the
    body earns freedom of movement by predicting its own body well -- it does not
    lose it on a transient spike (which simply pauses progress by resetting dwell).
    """
    pe_ema = (1.0 - BRACE_PE_TAU) * pe_ema + BRACE_PE_TAU * pe_now
    if pe_ema <= BRACE_PE_THRESH:
        dwell += dt
        if dwell >= BRACE_DWELL_S:
            tightness = max(0.0, tightness - BRACE_LOOSEN_STEP)
            dwell = 0.0
    else:
        dwell = 0.0  # low PE must be *sustained*; a surprise resets the timer
    return tightness, pe_ema, dwell


# Liveness: if no action arrives within this window (~7+ missed cognitive
# cycles AT FULL RATE), assume the brain isn't driving and let the body go
# limp. Env-tunable (2026-07-04 finding): when the brain runs slower than
# ~1 cycle/s, a fixed 1.0 s window marks it dead after nearly every action
# and the body ragdolls permanently -- the watchdog must scale with the
# actual cognitive cadence during diagnosis.
COMMAND_STALE_S = max(
    0.25, float(os.environ.get("DECADIC_BODY_COMMAND_STALE_S", "1.0"))
)


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _post_agent(base_http: str) -> str:
    req = urllib.request.Request(f"{base_http.rstrip('/')}/agent", method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["agent_id"])


def obs_interval_s() -> float:
    raw = os.environ.get("DECADIC_BODY_OBS_INTERVAL_MS", "80")
    try:
        ms = float(raw)
    except ValueError:
        ms = 80.0
    return max(0.016, ms / 1000.0)


# ---------------------------------------------------------------------------
# Pure observation building (unit-testable without MuJoCo)
# ---------------------------------------------------------------------------


def compose_scene(elements: list[str]) -> str:
    """Base humanoid model with the chosen world elements spliced into the worldbody.

    Unknown element names are ignored; duplicates are de-duplicated while
    preserving the first-seen order.
    """
    base = ASSETS_XML.read_text(encoding="utf-8")
    parts: list[str] = []
    for key in normalize_scene_elements(elements):
        parts.append(SCENE_ELEMENTS[key])
    snippet = "".join(parts)
    return base.replace("</worldbody>", f"{snippet}  </worldbody>")


def scene_xml(scene: str) -> str:
    """Legacy named-scene builder; resolves to a composed element set."""
    if scene not in LEGACY_SCENES:
        raise ValueError(f"Unknown scene {scene!r}; choose from {sorted(LEGACY_SCENES)}")
    return compose_scene(LEGACY_SCENES[scene])


def prop_kind(name: str) -> str:
    """Infer an entity kind label from a prop body name."""
    for kind in ("bear", "food", "water", "medical", "house", "ball", "box", "sphere", "npc"):
        if kind in name:
            return "medical_kit" if kind == "medical" else kind
    return "pillar"


def threat_intensity(dist: float, radius: float = THREAT_RADIUS) -> float | None:
    """Inverse-distance threat signal inside ``radius``; None when out of range."""
    if dist > radius:
        return None
    return round(max(0.1, min(1.0, 1.0 - dist / radius)), 4)


def eaten_now(
    root_pos: list[float],
    food_positions: dict[str, list[float]],
    eat_radius: float = EAT_RADIUS,
) -> list[str]:
    """Food ids within eating distance of the root (XY plane)."""
    out: list[str] = []
    for fid, fpos in food_positions.items():
        if math.hypot(fpos[0] - root_pos[0], fpos[1] - root_pos[1]) <= eat_radius:
            out.append(fid)
    return out


def drunk_now(
    root_pos: list[float],
    water_positions: dict[str, list[float]],
    drink_radius: float = DRINK_RADIUS,
) -> list[str]:
    """Water glass ids within drinking distance of the root (XY plane)."""
    return eaten_now(root_pos, water_positions, drink_radius)


def synth_audio_window(
    events: list[dict[str, Any]],
    contacts: dict[str, float],
    *,
    sample_rate: int = AUDIO_SR,
    duration_s: float = AUDIO_WINDOW_S,
    seed: int | None = None,
) -> "np.ndarray":  # noqa: F821 - numpy imported lazily
    """Procedural mono soundscape from physics for this observation window.

    Footsteps from sole forces, thuds/growls/chimes from events, and a faint
    ambient floor so the world is never literally silent.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = max(1, int(sample_rate * duration_s))
    wav = rng.standard_normal(n).astype(np.float32) * 0.004  # ambient air

    def burst(start_s: float, dur_s: float, amp: float, freq: float, tone_mix: float = 0.6) -> None:
        i0 = max(0, min(n - 1, int(start_s * sample_rate)))
        m = min(n - i0, max(1, int(dur_s * sample_rate)))
        tt = np.arange(m, dtype=np.float32) / sample_rate
        env = np.exp(-tt / max(1e-3, dur_s / 4.0)).astype(np.float32)
        tone = np.sin(2.0 * np.pi * freq * tt).astype(np.float32)
        noise = rng.standard_normal(m).astype(np.float32)
        wav[i0 : i0 + m] += amp * env * (tone_mix * tone + (1.0 - tone_mix) * noise)

    # Footsteps: low thuds, louder with sole force
    for name, force in contacts.items():
        if "foot" in name and force >= FOOTSTEP_MIN_FORCE:
            amp = float(min(0.25, force / 2000.0))
            burst(float(rng.uniform(0.0, duration_s * 0.5)), 0.12, amp, 90.0, tone_mix=0.5)

    # Events: each type gets a distinct timbre
    for ev in events:
        etype = str(ev.get("type", ""))
        inten = float(ev.get("intensity", 0.5) or 0.5)
        at = float(rng.uniform(0.0, duration_s * 0.4))
        if etype == "collision":
            burst(at, 0.25, 0.3 + 0.5 * inten, 180.0, tone_mix=0.35)
        elif etype == "fall":
            burst(at, 0.6, 0.7, 60.0, tone_mix=0.55)
        elif etype == "threat_near":
            # Growl: low tone with tremolo, louder as the threat closes in
            i0 = int(at * sample_rate)
            m = min(n - i0, int(0.5 * sample_rate))
            tt = np.arange(m, dtype=np.float32) / sample_rate
            trem = 0.5 * (1.0 + np.sin(2.0 * np.pi * 9.0 * tt))
            growl = np.sin(2.0 * np.pi * 45.0 * tt) * trem
            wav[i0 : i0 + m] += (0.2 + 0.4 * inten) * growl.astype(np.float32)
        elif etype == "food":
            burst(at, 0.3, 0.3, 880.0, tone_mix=0.95)
        elif etype == "water":
            # Gulp: soft low-frequency double blip with a watery noise tail
            burst(at, 0.18, 0.28, 220.0, tone_mix=0.4)
            burst(at + 0.12, 0.16, 0.22, 180.0, tone_mix=0.35)

    return np.clip(wav, -0.95, 0.95)


def audio_payload(wav: "np.ndarray", sample_rate: int = AUDIO_SR) -> dict[str, Any]:  # noqa: F821
    """Encode a float waveform in [-1, 1] as the observation `audio` blob."""
    import numpy as np

    pcm = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    return {
        "encoding": "pcm16_base64",
        "sample_rate": int(sample_rate),
        "duration_s": round(len(pcm) / sample_rate, 4),
        "data": base64.b64encode(data).decode("ascii"),
    }


@dataclass
class BodySnapshot:
    """Sim-state slice consumed by the observation builder."""

    position: list[float]
    orientation: list[float]  # roll, pitch, yaw (radians)
    velocity: list[float]
    joints: list[float]  # interleaved qpos + qvel of hinge joints
    contacts: dict[str, float]  # touch sensor name -> force (N)
    props: list[dict[str, Any]] = field(default_factory=list)  # {id, kind, position}
    moving: bool = False
    motor: list[float] | None = None  # efference copy: last normalized PD targets
    n_actuators: int = 0  # actuator count; contract with brain's motor head
    # Joint-brace guidance telemetry (dashboard-only; not fed to cognition).
    rom_mean: float = 0.0  # mean per-joint range of motion earned (0 welded -> 1 free)
    brace_engaged: float = 1.0  # mean per-joint brace tightness (1 fully welded -> 0 free)
    rom_frac: list[float] = field(default_factory=list)  # per-hinge ROM (1 - tightness)
    braces_enabled: bool = False  # manual scaffold toggle; off -> native springs, free body
    stance: str = "stand"  # active braced posture/motion (stance library name)
    stance_phase: float = 0.0  # motion-stance phase in [0, 1] (0 for static stances)
    movement_hold: bool = False  # hold mode: welded + looping until disabled
    manual_auto_reset: bool = False  # manual scaffold safety: reset a motion stance after falls
    foot_load_l: float = 0.0  # left foot load / body weight
    foot_load_r: float = 0.0  # right foot load / body weight
    hand_load_l: float = 0.0  # left hand load / body weight
    hand_load_r: float = 0.0  # right hand load / body weight
    teacher_support_active: bool = False
    teacher_support_force: float = 0.0
    teacher_support_torque: float = 0.0
    teacher_drop_m: float = 0.0
    teacher_target_drop_m: float = 0.25
    teacher_height_error_m: float = 0.0
    teacher_vertical_velocity: float = 0.0
    teacher_support_mode: str = "off"
    caregiver_parent_present: bool = False
    caregiver_kind: str = "none"
    caregiver_status: str = "missing_parent"
    caregiver_request_kind: str = ""
    caregiver_last_offer_item: str = ""
    caregiver_delivery_count: int = 0
    caregiver_pending_request: bool = False
    # Full-body touch: per-part contact load (sensor short-name -> force/body weight),
    # live in ALL modes. Feeds perception (the tactile sense) and the dashboard map.
    part_loads: dict[str, float] = field(default_factory=dict)
    body_map: dict[str, Any] = field(default_factory=dict)
    effort: dict[str, Any] = field(default_factory=dict)
    # Eval-only ground truth (never fed to cognition): world xpos of the limb
    # extremities, used solely to score discovered "self_part" / agency edges.
    body_parts: dict[str, list[float]] | None = None


@dataclass
class DirectProvision:
    """One visible resource delivery along the egocentric head-camera ray."""

    kind: str
    name: str
    body_id: int
    start: list[float]
    target: list[float]
    elapsed_s: float = 0.0
    duration_s: float = DIRECT_PROVISION_DURATION_S


def body_events(
    contacts: dict[str, float],
    root_height: float,
    *,
    was_fallen: bool,
    prev_contacts: dict[str, float] | None = None,
    velocity: list[float] | None = None,
    step: int = 0,
    cooldown_until: dict[str, int] | None = None,
    spike_threshold: float = IMPACT_FORCE_SPIKE_N,
    speed_ref: float = IMPACT_SPEED_REF,
    cooldown_steps: int = IMPACT_COOLDOWN_STEPS,
    fall_height: float = FALL_ROOT_HEIGHT,
) -> tuple[list[dict[str, Any]], bool]:
    """Impact-driven collisions plus a one-shot fall marker.

    A ``collision`` is emitted only on a genuine impact: a sudden spike in the
    summed contact force (a real momentum transfer) while the body still carries
    speed. Steady weight-bearing -- standing or lying on the ground -- shows no
    spike and produces no event, so resting is never mistaken for a collision.
    Intensity tracks impact energy (~speed^2, clamped 0..1), and a refractory
    cooldown (keyed in ``cooldown_until``) keeps a single tumble from being
    billed every frame. ``prev_contacts``/``velocity`` are required for impact
    detection; without them only the fall marker is produced.
    """
    events: list[dict[str, Any]] = []
    if prev_contacts is not None and contacts:
        total = sum(float(v) for v in contacts.values())
        prev_total = sum(float(v) for v in prev_contacts.values())
        spike = total - prev_total
        ready = cooldown_until is None or step >= int(cooldown_until.get("impact", 0))
        if spike >= spike_threshold and ready:
            speed = 0.0
            if velocity is not None and len(velocity) >= 3:
                speed = math.sqrt(sum(float(v) * float(v) for v in velocity[:3]))
            intensity = min(1.0, (speed / speed_ref) ** 2) if speed_ref > 0 else 0.0
            if intensity > 0.0:
                source = max(contacts.items(), key=lambda kv: float(kv[1]))[0]
                events.append(
                    {
                        "type": "collision",
                        "intensity": round(intensity, 4),
                        "source": source,
                    }
                )
                if cooldown_until is not None:
                    cooldown_until["impact"] = step + cooldown_steps
    fallen = root_height < fall_height
    if fallen and not was_fallen:
        events.append({"type": "fall", "intensity": 0.6, "source": "root"})
    return events, fallen


def build_body_observation(
    snap: BodySnapshot,
    *,
    events: list[dict[str, Any]] | None = None,
    vision_b64: str | None = None,
    vision_resolution: tuple[int, int] = (224, 224),
    debug_views: dict[str, str] | None = None,
    audio: dict[str, Any] | None = None,
    control_mode: str = "root_assist",
) -> dict[str, Any]:
    """Map a body snapshot to Decadic ``ObservationMessage`` JSON."""
    pos = [float(x) for x in snap.position[:3]]
    ori = [float(x) for x in snap.orientation[:3]]
    contact_values = [float(v) for v in snap.contacts.values()]

    entities = []
    for prop in snap.props:
        ppos = [float(x) for x in prop.get("position", [0.0, 0.0, 0.0])[:3]]
        entities.append(
            {
                "id": str(prop.get("id", "prop")),
                "kind": str(prop.get("kind", "prop")),
                "position": ppos,
                "relative": [ppos[i] - pos[i] for i in range(3)],
            }
        )

    standing = pos[2] >= FALL_ROOT_HEIGHT

    # Ordered per-part load list (matches touch-sensor declaration order, so it
    # lines up channel-for-channel with the tactile forward-model target).
    part_load_values = [float(v) for v in snap.part_loads.values()]

    proprio: dict[str, Any] = {
        "position": pos,
        "orientation": ori,
        "velocity": [float(x) for x in snap.velocity[:3]],
        "current_action": f"mujoco_humanoid:{control_mode}",
        "joints": [float(x) for x in snap.joints],
        "contacts": contact_values,
        "part_loads": part_load_values,
        "body_map": normalize_body_map(snap.body_map),
        "effort": normalize_effort(snap.effort),
    }
    # Efference copy: the motor command actually applied to the body this tick.
    if snap.motor is not None:
        proprio["motor"] = [float(x) for x in snap.motor]

    obs: dict[str, Any] = {
        "timestamp": _utc_iso(),
        "proprioception": proprio,
        "events": list(events or []),
        "world_state": {
            "agent": {"id": "self", "position": pos, "orientation": ori},
            "entities": entities,
            "nearby_entities": [{"id": e["id"], "kind": e["kind"]} for e in entities],
            "agent_inventory": [],
            "body": {
                "id": "mujoco_humanoid",
                "control_mode": control_mode,
                "standing": standing,
                "moving": snap.moving,
                "n_actuators": int(snap.n_actuators),
                "rom_mean": round(float(snap.rom_mean), 5),
                "brace_engaged": round(float(snap.brace_engaged), 5),
                "rom_frac": [round(float(v), 5) for v in snap.rom_frac],
                "braces_enabled": bool(snap.braces_enabled),
                "stance": str(snap.stance),
                "stance_phase": round(float(snap.stance_phase), 5),
                "movement_hold": bool(snap.movement_hold),
                "manual_auto_reset": bool(snap.manual_auto_reset),
                "foot_load_l": round(float(snap.foot_load_l), 5),
                "foot_load_r": round(float(snap.foot_load_r), 5),
                "hand_load_l": round(float(snap.hand_load_l), 5),
                "hand_load_r": round(float(snap.hand_load_r), 5),
                "teacher_support_active": bool(snap.teacher_support_active),
                "teacher_support_force": round(float(snap.teacher_support_force), 5),
                "teacher_support_torque": round(float(snap.teacher_support_torque), 5),
                "teacher_drop_m": round(float(snap.teacher_drop_m), 5),
                "teacher_target_drop_m": round(float(snap.teacher_target_drop_m), 5),
                "teacher_height_error_m": round(float(snap.teacher_height_error_m), 5),
                "teacher_vertical_velocity": round(float(snap.teacher_vertical_velocity), 5),
                "teacher_support_mode": str(snap.teacher_support_mode),
                "caregiver_parent_present": bool(snap.caregiver_parent_present),
                "caregiver_kind": str(snap.caregiver_kind),
                "caregiver_status": str(snap.caregiver_status),
                "caregiver_request_kind": str(snap.caregiver_request_kind),
                "caregiver_last_offer_item": str(snap.caregiver_last_offer_item),
                "caregiver_delivery_count": int(snap.caregiver_delivery_count),
                "caregiver_pending_request": bool(snap.caregiver_pending_request),
                "part_loads": {
                    k: round(float(v), 5) for k, v in snap.part_loads.items()
                },
            },
        },
    }
    if vision_b64:
        obs["vision"] = {
            "encoding": "base64_png",
            "data": vision_b64,
            "resolution": list(vision_resolution),
        }
    if debug_views:
        # Spectator camera frames; the server strips these before cognition.
        obs["debug_views"] = dict(debug_views)
    if snap.body_parts:
        # Eval-only ground truth: limb world positions for scoring discovered
        # body-self / agency edges. Inert to cognition (encoders read only
        # vision/audio/proprioception); consumed solely by discovery metrics.
        obs["eval_truth"] = {"body_parts": dict(snap.body_parts)}
    if audio:
        obs["audio"] = dict(audio)
    return obs


def dry_snapshot(step: int) -> BodySnapshot:
    """Synthetic standing-walk snapshot for --dry-run."""
    t = step * 0.08
    contact = {
        "torso": 0.0,
        "head": 0.0,
        "left_upper_arm": 0.0,
        "left_lower_arm": 0.0,
        "left_hand": 0.0,
        "right_upper_arm": 0.0,
        "right_lower_arm": 0.0,
        "right_hand": 0.8 if step and step % 25 == 0 else 0.0,
        "left_thigh": 0.0,
        "left_shin": 0.0,
        "left_foot": 0.03 if step % 2 == 0 else 0.75,
        "right_thigh": 0.0,
        "right_shin": 0.0,
        "right_foot": 0.75 if step % 2 == 0 else 0.03,
    }
    effort = {p: 0.05 + 0.03 * abs(math.sin(t)) for p in BODY_PARTS}
    fatigue = {p: 0.02 * abs(math.sin(t)) for p in BODY_PARTS}
    pain = {p: 0.0 for p in BODY_PARTS}
    body_map = {
        "parts": list(BODY_PARTS),
        "contact_load": contact,
        "effort": effort,
        "work": {p: 0.01 for p in BODY_PARTS},
        "strain": {p: max(contact[p] * 0.1, effort[p] * 0.2) for p in BODY_PARTS},
        "fatigue": fatigue,
        "pain": pain,
    }
    effort_total = sum(effort.values()) / max(1, len(effort))
    return BodySnapshot(
        position=[math.cos(t) * 0.5, math.sin(t) * 0.5, STAND_ROOT_HEIGHT],
        orientation=[0.0, 0.0, t % (2 * math.pi)],
        velocity=[-math.sin(t) * 0.04, math.cos(t) * 0.04, 0.0],
        joints=[math.sin(t + i) * 0.05 for i in range(42)],
        contacts={
            "touch_right_foot": 130.0 if step % 2 == 0 else 5.0,
            "touch_left_foot": 5.0 if step % 2 == 0 else 130.0,
            "touch_right_hand": 520.0 if step and step % 25 == 0 else 0.0,
            "touch_left_hand": 0.0,
        },
        props=[{"id": "prop_box_red", "kind": "box", "position": [1.5, 0.5, 0.25]}],
        moving=True,
        body_map=body_map,
        effort={
            "actuator_effort": [0.05] * 21,
            "actuator_work": [0.01] * 21,
            "joint_strain": [0.02] * 21,
            "joint_fatigue": [0.01] * 21,
            "effort_total": effort_total,
            "work_total": 0.01,
            "strain_total": 0.02,
            "fatigue_total": sum(fatigue.values()) / max(1, len(fatigue)),
            "pain_total": 0.0,
            "support_effort": 0.0,
        },
        body_parts={
            "right_lower_arm": [0.25, -0.2, 1.2 + math.sin(t) * 0.1],
            "left_lower_arm": [0.25, 0.2, 1.2 - math.sin(t) * 0.1],
            "right_foot": [0.0, -0.1, 0.1],
            "left_foot": [0.0, 0.1, 0.1],
        },
    )


# ---------------------------------------------------------------------------
# MuJoCo simulation (imported lazily so --dry-run works without it)
# ---------------------------------------------------------------------------


class HumanoidSim:
    """Humanoid with PD standing hold + pelvis root-assist locomotion."""

    def __init__(
        self,
        *,
        vision: bool,
        vision_size: int = 224,
        view: bool = False,
        scene: str = "default",
        elements: "list[str] | None" = None,
        braces: bool = False,
    ) -> None:
        import mujoco
        import numpy as np

        self._mj = mujoco
        self._np = np
        self.scene = scene
        # An explicit element list (composable scenario) wins over the legacy
        # named scene; otherwise resolve the named scene to its element set.
        if elements is not None:
            self.elements = normalize_scene_elements(elements)
            xml = compose_scene(self.elements)
        else:
            self.elements = normalize_scene_elements(list(LEGACY_SCENES.get(scene, [])))
            xml = scene_xml(scene)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.head_id = head_id if head_id >= 0 else self.torso_id
        # Hinge joints driven by actuators (skip the free root joint, scene props,
        # and the parent NPC's joints -- only the agent's own joints feed the
        # brain's proprioception and the 21-actuator motor contract).
        self.hinge_qpos_adr: list[int] = []
        self.hinge_qvel_adr: list[int] = []
        self.hinge_qpos0: list[float] = []
        self.hinge_jid: list[int] = []  # joint id per hinge (for the joint braces)
        self.hinge_names: list[str] = []  # joint name per hinge (for stance authoring)
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            # Skip scene props and ALL scripted NPCs (legacy "npc_" parent and the
            # crowd's "npc0_".."npcN_" demonstrators); only the agent's own hinges
            # feed proprioception and the 21-actuator motor contract.
            if name.startswith("prop_") or name.startswith("npc"):
                continue
            self.hinge_qpos_adr.append(int(self.model.jnt_qposadr[j]))
            self.hinge_qvel_adr.append(int(self.model.jnt_dofadr[j]))
            self.hinge_qpos0.append(float(self.model.qpos0[self.model.jnt_qposadr[j]]))
            self.hinge_jid.append(int(j))
            self.hinge_names.append(name)

        # actuator -> driven joint hinge index (actuators are joint motors here)
        self.act_joint: list[int] = []
        for a in range(self.model.nu):
            jid = int(self.model.actuator_trnid[a, 0])
            adr = int(self.model.jnt_qposadr[jid])
            self.act_joint.append(self.hinge_qpos_adr.index(adr))

        self.touch_sensors: list[tuple[str, int]] = []
        for s in range(self.model.nsensor):
            if self.model.sensor_type[s] == mujoco.mjtSensor.mjSENS_TOUCH:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"touch_{s}"
                self.touch_sensors.append((name, int(self.model.sensor_adr[s])))

        # Per-foot / per-hand touch addresses (for the guided curriculum's
        # self-support load signal: feet for standing, hands for bracing/push-up).
        self._foot_touch_adr: dict[str, int] = {}
        for side, nm in (("l", "touch_left_foot"), ("r", "touch_right_foot")):
            adr = next((a for n, a in self.touch_sensors if n == nm), None)
            if adr is not None:
                self._foot_touch_adr[side] = adr
        self._hand_touch_adr: dict[str, int] = {}
        for side, nm in (("l", "touch_left_hand"), ("r", "touch_right_hand")):
            adr = next((a for n, a in self.touch_sensors if n == nm), None)
            if adr is not None:
                self._hand_touch_adr[side] = adr

        # Rescale the agent's kinematic subtree (every body rooted at the torso's
        # free joint -- this deliberately excludes the spliced NPC clone and scene
        # props, which root their own subtrees) to a fixed total mass. Inertia
        # scales with mass at fixed geometry, so both are multiplied by the same
        # factor. Done in code so it isolates the agent regardless of scene/props.
        _subtree = [
            b
            for b in range(self.model.nbody)
            if int(self.model.body_rootid[b]) == self.torso_id
        ]
        _cur_mass = sum(float(self.model.body_mass[b]) for b in _subtree)
        if _cur_mass > 0.0:
            _factor = BODY_MASS_KG / _cur_mass
            for b in _subtree:
                self.model.body_mass[b] *= _factor
                self.model.body_inertia[b] *= _factor
            # Keep the derived subtree mass (read just below for body weight and
            # used to normalize touch loads) consistent with the rescaled bodies.
            self.model.body_subtreemass[self.torso_id] = BODY_MASS_KG

        self._body_weight = float(self.model.body_subtreemass[self.torso_id]) * abs(
            float(self.model.opt.gravity[2])
        )
        self._foot_load_l = 0.0
        self._foot_load_r = 0.0
        self._hand_load_l = 0.0
        self._hand_load_r = 0.0

        # --- Stance / motion ROM braces -------------------------------------
        # Per-hinge joint limits (radians) for clamping authored stance angles;
        # unlimited joints get an open range so their qpos0 is preserved verbatim.
        self._hinge_ranges: list[tuple[float, float]] = []
        for h in range(len(self.hinge_qpos_adr)):
            jid = self.hinge_jid[h]
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
                self._hinge_ranges.append((float(lo), float(hi)))
            else:
                self._hinge_ranges.append((-math.inf, math.inf))
        # Active stance: which posture the braces hold the body in. q_ref is the
        # per-joint reference of that stance; a motion stance retargets q_ref each
        # control tick from its phase. Default is the validated upright stand.
        self._stance_name: str = stance_lib.DEFAULT_STANCE
        self._stance_phase: float = 0.0
        _stance = stance_lib.get_stance(self._stance_name)
        self._stance_root_z: float = float(_stance.root_z)
        self._stance_root_quat: tuple[float, float, float, float] = tuple(
            float(c) for c in _stance.root_quat
        )
        self._stance_fall_z: float = float(_stance.fall_z)

        # --- Joint-brace guidance state -------------------------------------
        # q_ref: the stance pose each hinge is braced toward (the active stance's
        # per-joint reference; the default stand is the model's neutral pose,
        # validated as a stable COM-over-feet stand). native_*: the joint's
        # original soft spring/damp, the floor the brace decays toward. tightness:
        # per-joint weld factor (1 = fully braced, 0 = native/free), ratcheted
        # DOWN as the brain's per-joint forward-model PE stays low.
        nH = len(self.hinge_qpos_adr)
        self._q_ref: list[float] = stance_lib.resolve(
            _stance, self.hinge_names, list(self.hinge_qpos0), self._hinge_ranges
        )
        self._native_stiff: list[float] = [
            float(self.model.jnt_stiffness[self.hinge_jid[h]]) for h in range(nH)
        ]
        self._native_damp: list[float] = [
            float(self.model.dof_damping[self.hinge_qvel_adr[h]]) for h in range(nH)
        ]
        self._tightness: list[float] = [1.0] * nH
        self._joint_pe_ema: list[float] = [1.0] * nH  # start high: stay braced until proven
        self._brace_dwell: list[float] = [0.0] * nH
        self._joint_pe_buf: list[float] | None = None  # latest per-joint PE from the brain
        # Master on/off for the manual orthosis. Off (default): hinges relax to
        # native springs and the brain alone holds the body up (it can fall).
        # Earned ROM is preserved when the operator toggles braces manually.
        self._braces_enabled = bool(braces)
        # "Hold movement" mode: when on, the active stance/motion keeps running
        # with every joint brace fully welded (the ROM curriculum is suspended --
        # no range-of-motion release) and motions loop continuously (even the
        # one-shot Rise), until the operator disables it. Off (default): the ROM
        # ratchet runs as usual. Only takes effect while the master braces are on.
        self._movement_hold = False
        self._manual_auto_reset = False
        # Seed the model's spring reference so the body stands as a braced statue
        # from t=0 when braced (before any brain command); apply the matching
        # gains, or relax to native springs when the braces start disabled.
        for h in range(nH):
            self.model.qpos_spring[self.hinge_qpos_adr[h]] = self._q_ref[h]
        if self._braces_enabled:
            self._apply_brace_gains()
        else:
            self._relax_braces()
        # Full-body touch loads (short name -> force/body weight), refreshed every
        # step() in ALL modes by _read_limb_loads(). Ordered list mirrors
        # self.touch_sensors so the tactile target lines up channel-for-channel.
        self._part_loads: dict[str, float] = {
            n[len("touch_") :] if n.startswith("touch_") else n: 0.0
            for n, _ in self.touch_sensors
        }
        self._part_effort: dict[str, float] = {p: 0.0 for p in BODY_PARTS}
        self._part_work: dict[str, float] = {p: 0.0 for p in BODY_PARTS}
        self._part_strain: dict[str, float] = {p: 0.0 for p in BODY_PARTS}
        self._part_fatigue: dict[str, float] = {p: 0.0 for p in BODY_PARTS}
        self._part_pain: dict[str, float] = {p: 0.0 for p in BODY_PARTS}
        self._actuator_effort: list[float] = [0.0] * int(self.model.nu)
        self._actuator_work: list[float] = [0.0] * int(self.model.nu)
        self._joint_strain: list[float] = [0.0] * int(self.model.nu)
        self._joint_fatigue: list[float] = [0.0] * int(self.model.nu)
        self._effort_total = 0.0
        self._work_total = 0.0
        self._strain_total = 0.0
        self._fatigue_total = 0.0
        self._pain_total = 0.0

        self.prop_bodies: list[tuple[str, int]] = []
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith("prop_"):
                self.prop_bodies.append((name, b))

        # Limb extremities (the agent's hands ~ lower arms, and feet). Streamed as
        # eval-only ground truth so the discovery harness can score which slots
        # the agent has correctly identified as "mine" (body parts). The brain
        # never receives these world positions.
        self.body_part_ids: dict[str, int] = {}
        for part in ("right_lower_arm", "left_lower_arm", "right_foot", "left_foot"):
            pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, part)
            if pid >= 0:
                self.body_part_ids[part] = pid

        # Scene actors
        self.bear_body: int | None = None
        self.food_bodies: dict[str, int] = {}
        self.water_bodies: dict[str, int] = {}
        self.medical_bodies: dict[str, int] = {}
        for name, b in self.prop_bodies:
            if "bear" in name:
                self.bear_body = b
            elif "food" in name:
                self.food_bodies[name] = b
            elif "water" in name:
                self.water_bodies[name] = b
            elif "medical" in name or "medkit" in name:
                self.medical_bodies[name] = b
        self.eaten: set[str] = set()
        self._direct_delivery: DirectProvision | None = None
        self._direct_delivery_events: list[dict[str, Any]] = []
        # Seed of the most recent per-life resource scatter (-1 = never scattered);
        # echoed to the brain so the life is reproducible / telemetered.
        self._resource_seed: int = -1
        # Respawn scheduling: name -> monotonic time it should reappear, and the
        # original (alpha, contype, conaffinity) per geom so we can restore it.
        self._respawn_at: dict[str, float] = {}
        self._geom_orig: dict[int, tuple[float, int, int]] = {}
        self._last_threat_step = -(10**9)
        self._last_bear_hit_step = -(10**9)
        self._torso_root = int(self.model.body_rootid[self.torso_id])
        self._bear_root = (
            int(self.model.body_rootid[self.bear_body]) if self.bear_body is not None else -1
        )

        # Parent NPC: an actuator-free humanoid animated kinematically (see the
        # walk-cycle section near _apply_npc). Track its torso, free-root address
        # (we overwrite it each substep), every hinge (held at qpos0 unless the
        # walk cycle animates it), and the addresses of the animated joints. Two
        # movable "gifts" (food & water) let it carry and drop a morsel for the
        # agent some distance away (so the agent must move toward it).
        self.npc_torso: int | None = None
        self.npc_root_qadr = -1
        self.npc_root_dadr = -1
        self.npc_hinges: list[tuple[int, int, float]] = []
        self._npc_anim: dict[str, tuple[int, int, float]] = {}
        # item -> (qpos_adr, dof_adr) of each gift's free joint (-1 when absent).
        self._gift_names = {
            "food": "prop_food_gift",
            "water": "prop_water_gift",
            "medical_kit": "prop_medical_gift",
        }
        self._gift_addr: dict[str, tuple[int, int]] = {
            "food": (-1, -1),
            "water": (-1, -1),
            "medical_kit": (-1, -1),
        }
        self._npc_phase = "seek_food"  # foraging FSM state (distinct from gait phase)
        self._npc_item = "food"  # which gift the parent currently fetches/carries
        self._npc_next_deliver = NPC_DELIVER_PERIOD_S  # earliest sim-time of next offer (refractory)
        self._npc_offers = 0  # delivered offers so far (fades the need threshold)
        self._npc_frozen = False  # UI "pause parent": hold pose, run no FSM
        self._npc_gait_phase = 0.0  # distance-locked walk-cycle phase (rad)
        self._npc_x = 0.0
        self._npc_y = 0.0
        self._npc_yaw = 0.0
        self._npc_requested_item: str | None = None
        self._npc_request_kind = ""
        self._npc_last_offer_item = ""
        self._npc_delivery_count = 0
        # Scripted crowd ("village") controller and the agent's latest reservoir
        # levels (piggybacked on the brain's action message). Known low reservoirs
        # or explicit requests gate provisioning (legacy parent + crowd parent).
        self.crowd: CrowdController | None = None
        self._agent_reservoirs: dict[str, float] | None = None
        if "npc" in self.elements:
            nid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "npc_torso")
            self.npc_torso = nid if nid >= 0 else None
            rj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "npc_root")
            if rj >= 0:
                self.npc_root_qadr = int(self.model.jnt_qposadr[rj])
                self.npc_root_dadr = int(self.model.jnt_dofadr[rj])
            for j in range(self.model.njnt):
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if nm.startswith("npc_") and (
                    self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                ):
                    qa = int(self.model.jnt_qposadr[j])
                    self.npc_hinges.append(
                        (qa, int(self.model.jnt_dofadr[j]), float(self.model.qpos0[qa]))
                    )
            for item, gname in self._gift_names.items():
                gj = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{gname}_free"
                )
                if gj >= 0:
                    self._gift_addr[item] = (
                        int(self.model.jnt_qposadr[gj]),
                        int(self.model.jnt_dofadr[gj]),
                    )

            def _npc_jadr(jname: str) -> tuple[int, int, float] | None:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    return None
                qa = int(self.model.jnt_qposadr[jid])
                return (qa, int(self.model.jnt_dofadr[jid]), float(self.model.qpos0[qa]))

            for key, jname in (
                ("r_hip", "npc_right_hip_y"),
                ("l_hip", "npc_left_hip_y"),
                ("r_knee", "npc_right_knee"),
                ("l_knee", "npc_left_knee"),
                ("r_sh", "npc_right_shoulder1"),
                ("l_sh", "npc_left_shoulder1"),
            ):
                adr = _npc_jadr(jname)
                if adr is not None:
                    self._npc_anim[key] = adr

            # Seed the analytic walk state from the spawn root, facing the origin
            # (where the agent and consumables live).
            if self.npc_root_qadr >= 0:
                self._npc_x = float(self.data.qpos[self.npc_root_qadr])
                self._npc_y = float(self.data.qpos[self.npc_root_qadr + 1])
            self._npc_yaw = math.atan2(-self._npc_y, -self._npc_x)

            hb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "npc_right_lower_arm")
            self.npc_hand: int | None = hb if hb >= 0 else None
            self._npc_carry = False  # forages empty-handed until the first pickup
        else:
            self.npc_hand = None
            self._npc_carry = False

        # Scripted crowd: discover all npcN_ bodies and build their runtime state.
        if "crowd" in self.elements:
            self.crowd = CrowdController(self)
            print(f"[body] crowd: {len(self.crowd.npcs)} scripted NPCs in habitats")

        self.renderer = None
        self.vision_size = vision_size
        if vision:
            self.renderer = mujoco.Renderer(self.model, height=vision_size, width=vision_size)

        # Spectator cameras (everything except the brain's egocentric eye)
        self.view_cameras: list[str] = []
        for c in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) or ""
            if name and name != "egocentric":
                self.view_cameras.append(name)
        self._views_cache: dict[str, str] = {}
        self._views_counter = 0

        self.viewer = None
        if view:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self._move_dir = (0.0, 0.0)
        self._move_speed = 0.0

        # Embodied motor control: per-actuator normalized PD targets from the
        # brain, mapped into each driven joint's range. The brain's targets ride
        # on top of the joint braces, which do the standing work.
        self._motor_targets: "np.ndarray | None" = None  # noqa: F821
        # Skill Dojo live teacher spotter. Separate from manual braces: it adds
        # only temporary vertical support and upright torque while teacher assist
        # is active, never horizontal locomotion.
        self._teacher_assist = 0.0
        self._teacher_live = False
        self._teacher_support_force = 0.0
        self._teacher_support_torque = 0.0
        self._teacher_drop_m = 0.0
        self._teacher_target_drop_m = 0.25
        self._teacher_height_error_m = 0.0
        self._teacher_vertical_velocity = 0.0
        self._teacher_support_mode = "off"
        # When the brain is dead (or sending no commands) the body must go
        # passive: only the brain drives this body, so with no driver it falls
        # limp instead of replaying its last command forever.
        self._lifeless = False
        self.act_target_lo: list[float] = []
        self.act_target_hi: list[float] = []
        for a in range(self.model.nu):
            jid = int(self.model.actuator_trnid[a, 0])
            if bool(self.model.jnt_limited[jid]):
                lo, hi = self.model.jnt_range[jid]
                self.act_target_lo.append(float(lo))
                self.act_target_hi.append(float(hi))
            else:
                self.act_target_lo.append(-1.5)
                self.act_target_hi.append(1.5)

        # Deterministic spawn: drop the agent at the brace standing height (feet
        # on the floor) in the welded stand pose, so it starts as a braced statue
        # rather than free-falling from the asset's 1.4 drop-in height.
        self.recenter()
        # Anti-camping: scatter resources to fresh positions at body spawn so the
        # very first life cannot rely on the deterministic XML layout (no-op when
        # disabled or when the scenario has no static consumables).
        self._randomize_resources()

    def view_open(self) -> bool:
        """True while the native viewer window is open (or when no viewer requested)."""
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def open_viewer(self) -> None:
        """Lazily open the native MuJoCo window on demand (idempotent).

        Launched from the step loop (main thread), never the async receiver,
        so the GL context stays on the right thread. Fails soft on a headless
        host instead of crashing the body.
        """
        if self.viewer is not None:
            return
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[body] live viewer opened", flush=True)
        except Exception as exc:  # noqa: BLE001 - no display / GL unavailable
            self.viewer = None
            print(f"[body] live viewer unavailable: {exc}", flush=True)

    def set_lifeless(self, value: bool) -> None:
        """Toggle passive ragdoll mode (brain dead or no commands streaming)."""
        value = bool(value)
        if value != self._lifeless:
            print(f"[body] lifeless={value} (passive ragdoll)" if value
                  else "[body] reanimated (brain driving again)", flush=True)
        self._lifeless = value

    def set_braces_enabled(self, value: bool) -> None:
        """Master on/off for the joint-brace orthosis.

        On (default): every hinge is braced toward the stand pose and the ROM
        curriculum runs. Off: the braces relax to the model's native joint
        springs at once -- the brain (and gravity) alone govern the body, so it
        can fall. Earned ROM/tightness is preserved, so toggling back on resumes
        the curriculum exactly where it left off.
        """
        value = bool(value)
        if value != self._braces_enabled:
            print("[body] joint braces ON" if value
                  else "[body] joint braces OFF (free body)", flush=True)
        self._braces_enabled = value
        if value:
            self._apply_brace_gains()
        else:
            self._relax_braces()

    def set_movement_hold(self, value: bool) -> None:
        """Toggle "hold movement" -- run the active stance until manually disabled.

        On: the ROM curriculum is suspended (every joint stays fully welded -- no
        range-of-motion release) and motion stances loop continuously, so the
        selected movement runs precisely on repeat until switched off. Off: the
        per-joint ROM ratchet resumes (from welded). Inert while the master joint
        braces are off (then the body is free regardless).
        """
        value = bool(value)
        if value != self._movement_hold:
            print("[body] movement hold ON (welded, looping until disabled)" if value
                  else "[body] movement hold OFF (ROM curriculum resumes)", flush=True)
        self._movement_hold = value

    def set_manual_auto_reset(self, value: bool) -> None:
        """Toggle manual scaffold safety reset for motion stances.

        This is operator scaffolding, not Skill Dojo progression. When enabled,
        a fall while a motion stance is selected re-applies that stance pose after
        a short cooldown without touching the brain, replay memory, or earned
        brace ROM.
        """
        value = bool(value)
        if value != self._manual_auto_reset:
            print(
                "[body] manual motion auto-reset ON"
                if value
                else "[body] manual motion auto-reset OFF",
                flush=True,
            )
        self._manual_auto_reset = value

    def _seed_stance_spring(self) -> None:
        """Push the current ``_q_ref`` into MuJoCo's per-hinge spring reference."""
        for h in range(len(self.hinge_qpos_adr)):
            self.model.qpos_spring[self.hinge_qpos_adr[h]] = self._q_ref[h]

    def set_stance(self, name: str) -> None:
        """Re-pose the body into a stance without changing brace state.

        A stance defines the per-joint reference the braces hold (a static
        posture, or, for a motion stance, the base pose its phase trajectory
        rides on) plus the spawn root height/orientation and a posture-aware
        fall floor. Switching re-poses the body into the stance's start pose and
        updates the spring reference, but does not enable, disable, hold, or
        re-weld braces. ``reset_braces`` remains a separate manual command.
        """
        stance = stance_lib.get_stance(name)
        self._stance_name = stance.name
        self._stance_phase = 0.0
        self._stance_root_z = float(stance.root_z)
        self._stance_root_quat = tuple(float(c) for c in stance.root_quat)
        self._stance_fall_z = float(stance.fall_z)
        self._q_ref = stance_lib.resolve(
            stance, self.hinge_names, list(self.hinge_qpos0), self._hinge_ranges
        )
        self._seed_stance_spring()
        print(f"[body] stance -> {stance.name} ({stance.label})", flush=True)
        self.recenter()

    def apply_action(self, action: Any) -> None:
        """Store the latest Decadic action.

        The brain emits ``type="motor"`` with a per-actuator PD-target vector
        (``ctrl``) and, for the joint-brace curriculum, a per-joint forward-model
        error vector (``joint_pe``) used to ratchet each hinge's ROM open.
        ``teacher_assist`` / ``teacher_live`` drive the Skill Dojo spotter during
        non-autonomous training phases. Legacy ``assist_gain`` and
        ``curriculum_mode`` remain accepted for telemetry/backcompat.
        """
        if not isinstance(action, dict):
            return
        params = action.get("parameters") or {}
        atype = action.get("type")
        if atype not in ("motor", "move"):
            return
        np = self._np
        ctrl = params.get("ctrl")
        if isinstance(ctrl, list) and ctrl:
            vec = np.asarray(ctrl[: self.model.nu], dtype=np.float64)
            if vec.shape[0] < self.model.nu:
                vec = np.pad(vec, (0, self.model.nu - vec.shape[0]))
            self._motor_targets = np.clip(vec, -1.0, 1.0)
        try:
            assist = float(params.get("teacher_assist", 0.0) or 0.0)
        except (TypeError, ValueError):
            assist = 0.0
        self._teacher_assist = max(0.0, min(1.0, assist))
        self._teacher_live = bool(params.get("teacher_live", False)) and self._teacher_assist > 0.0
        # Per-joint proprioceptive forward-model error from the brain. Drives the
        # ROM curriculum: a joint loosens (earns range of motion) only once its own
        # prediction error stays low. Stored finite; the ratchet reads it next tick.
        jpe = params.get("joint_pe")
        if isinstance(jpe, list) and jpe:
            buf: list[float] = []
            for v in jpe:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = 1.0
                buf.append(f if math.isfinite(f) else 1.0)
            self._joint_pe_buf = buf
        # Agent reservoir levels (normalized 0..1), piggybacked by the server on
        # the action message. Gate the parent's need-threshold provisioning; never
        # touch cognition. Absent -> no automatic caregiver provisioning.
        res = params.get("reservoirs")
        if isinstance(res, dict) and res:
            parsed: dict[str, float] = {}
            for k, v in res.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    parsed[str(k)] = max(0.0, min(1.0, f))
            self._agent_reservoirs = parsed or None

    def step(self, sim_seconds: float) -> None:
        n = max(1, int(round(sim_seconds / self.model.opt.timestep)))
        # Refresh full-body touch loads first, unconditionally: limb load is a
        # property of the body, not of a mode, so it must be live whether the
        # body is braced, free or lifeless. Downstream just READS the cache.
        self._read_limb_loads()
        # Joint-brace guidance: update each hinge's ROM (tightness) once per
        # control tick, then hold it across the physics substeps. The braces are
        # MuJoCo native joint springs (no external wrench), so there is nothing to
        # re-apply per substep -- mj_step integrates them. A lifeless body relaxes
        # to its native joint springs (true ragdoll).
        if self._lifeless or not self._braces_enabled:
            # Lifeless OR braces switched off: native joint springs only. When
            # off-but-alive the brain still drives the actuators below; only the
            # bracing assist is withdrawn (a free body that can fall).
            self._relax_braces()
        else:
            # Motion stance: retarget q_ref (and the spring reference) from the
            # phase trajectory before the brace tick, so the stiff braces track a
            # moving pose -- locomotion comes from real contact, the root unforced.
            self._advance_motion(sim_seconds)
            if self._movement_hold:
                # Hold mode: keep every joint welded (no ROM release); the motion
                # keeps looping (advanced above) until the operator disables it.
                self._hold_braces()
            else:
                self._update_braces(sim_seconds)
        for _ in range(n):
            if self._lifeless:
                # No brain, no drive: zero actuator effort so the body collapses
                # under gravity onto its relaxed (native) joints (true ragdoll).
                self.data.ctrl[:] = 0.0
                self.data.xfrc_applied[self.torso_id, :] = 0.0
                self._apply_bear_chase()  # environment keeps moving around the corpse
                self._apply_npc()  # the parent keeps living even if the agent died
                if self.crowd is not None:
                    self.crowd.apply()  # the village keeps living too
            else:
                # No external support harness: the joint braces (native springs)
                # hold the body up from the inside; the feet keep the full weight.
                self.data.xfrc_applied[self.torso_id, :] = 0.0
                self._apply_bear_chase()
                self._apply_npc()
                if self.crowd is not None:
                    self.crowd.apply()
                self._apply_teacher_spotter()
                self._apply_motor()
            self._mj.mj_step(self.model, self.data)
        self._advance_direct_delivery(sim_seconds)
        self._update_effort_telemetry(sim_seconds)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                # User closed the window: release the handle and keep simulating
                # headless instead of treating it as a shutdown signal.
                self.viewer = None
                print("[body] live viewer closed (body keeps running)", flush=True)

    def _apply_bear_chase(self) -> None:
        """Drive the bear horizontally toward the humanoid; drag caps its speed."""
        if self.bear_body is None:
            return
        bpos = self.data.xpos[self.bear_body]
        tpos = self.data.xpos[self.torso_id]
        dx, dy = float(tpos[0] - bpos[0]), float(tpos[1] - bpos[1])
        dist = math.hypot(dx, dy)
        bvel = self.data.cvel[self.bear_body][3:]
        fx = -BEAR_DRAG * float(bvel[0])
        fy = -BEAR_DRAG * float(bvel[1])
        if dist > BEAR_STANDOFF:
            fx += BEAR_DRIVE * dx / dist
            fy += BEAR_DRIVE * dy / dist
        self.data.xfrc_applied[self.bear_body, 0] = fx
        self.data.xfrc_applied[self.bear_body, 1] = fy

    def _nearest_consumable_xy(
        self, bodies: dict[str, int], *, exclude_sub: str | None = None
    ) -> tuple[float, float] | None:
        """XY of the closest live consumable to the NPC (skipping eaten/excluded)."""
        if self.npc_torso is None or not bodies:
            return None
        npos = self.data.xpos[self.npc_torso]
        best: tuple[float, float] | None = None
        best_d = float("inf")
        for name, b in bodies.items():
            if name in self.eaten:
                continue
            if exclude_sub is not None and exclude_sub in name:
                continue
            p = self.data.xpos[b]
            d = math.hypot(float(p[0]) - float(npos[0]), float(p[1]) - float(npos[1]))
            if d < best_d:
                best_d = d
                best = (float(p[0]), float(p[1]))
        return best

    def _agent_xy(self) -> tuple[float, float]:
        t = self.data.xpos[self.torso_id]
        return (float(t[0]), float(t[1]))

    @staticmethod
    def _resource_kind(kind: str) -> str:
        key = str(kind or "").strip().lower()
        if key in {"medical_kit", "medical-kit", "medical", "medkit", "kit", "care"}:
            return "medical_kit"
        return "water" if key == "water" else "food"

    def _resource_bodies(self, kind: str) -> dict[str, int]:
        k = self._resource_kind(kind)
        if k == "water":
            return self.water_bodies
        if k == "medical_kit":
            return self.medical_bodies
        return self.food_bodies

    @staticmethod
    def _is_habitat_resource(name: str) -> bool:
        return "_h" in name

    @staticmethod
    def _is_learner_consumable(name: str) -> bool:
        # Learner-open props (arena resources) and parent/manual gifts can credit
        # the learner. Habitat resources are NPC ecology and must not magically
        # relieve the agent just because it falls near a village pantry.
        return "gift" in name or not HumanoidSim._is_habitat_resource(name)

    def _resource_candidates(self, kind: str) -> list[str]:
        bodies = self._resource_bodies(kind)
        open_props = sorted(
            name for name in bodies if "gift" not in name and not self._is_habitat_resource(name)
        )
        if open_props:
            return open_props
        # Explicit manual/debug provisioning may reuse a gift body when the
        # current scenario has only caregiver supplies (e.g. crowd + house).
        return sorted(name for name in bodies if "gift" in name)

    def _free_joint_addr_for_body(self, name: str) -> tuple[int, int] | None:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if jid < 0:
            return None
        qadr = int(self.model.jnt_qposadr[jid])
        dadr = int(self.model.jnt_dofadr[jid])
        return qadr, dadr

    def _place_resource_body(self, name: str, body_id: int, pos: list[float] | tuple[float, float, float]) -> None:
        joint_addr = self._free_joint_addr_for_body(name)
        if joint_addr is None:
            self.model.body_pos[body_id][:3] = pos
            return
        qadr, dadr = joint_addr
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dadr : dadr + 6] = 0.0

    def give_near(self, kind: str) -> bool:
        """Admin provisioning: relocate a resource prop a step ahead of the agent.

        Reuses an existing static scene prop (never an NPC gift body) by editing
        its world position so it lands ~1.6 m in front of the torso - just outside
        the 1.0 m reach, in the egocentric camera's view - then restores its
        visibility/collidability. The agent must still perceive and walk to it to
        consume it (normal food/water event), so the act->relief loop stays intact.
        Works only when the running scenario includes that element; returns False
        otherwise.
        """
        kind = self._resource_kind(kind)
        bodies = self._resource_bodies(kind)
        candidates = self._resource_candidates(kind)
        if not candidates:
            return False
        name = candidates[0]
        bid = bodies[name]
        ax, ay = self._agent_xy()
        xmat = self.data.xmat[self.torso_id].reshape(3, 3)
        yaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
        dist = 1.6
        base_z = float(self.data.xpos[bid][2])
        self._place_resource_body(
            name,
            bid,
            [ax + math.cos(yaw) * dist, ay + math.sin(yaw) * dist, base_z],
        )
        # Reappear (un-hide + recollide) and cancel any pending auto-respawn.
        self._respawn(name)
        self._respawn_at.pop(name, None)
        self._mj.mj_forward(self.model, self.data)
        return True

    def _head_forward(self) -> tuple[list[float], list[float]]:
        """Return world head position and egocentric forward unit vector."""
        pos = [float(x) for x in self.data.xpos[self.head_id][:3]]
        xmat = self.data.xmat[self.head_id].reshape(3, 3)
        fwd = [float(xmat[0, 0]), float(xmat[1, 0]), float(xmat[2, 0])]
        norm = math.sqrt(sum(v * v for v in fwd)) or 1.0
        return pos, [v / norm for v in fwd]

    def _set_resource_delivery_contact(self, body_id: int, *, enabled: bool) -> None:
        """Disable collision while a direct resource flies into the camera view."""
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) != body_id:
                continue
            if g not in self._geom_orig:
                self._geom_orig[g] = (
                    float(self.model.geom_rgba[g, 3]),
                    int(self.model.geom_contype[g]),
                    int(self.model.geom_conaffinity[g]),
                )
            if enabled:
                alpha, contype, conaffinity = self._geom_orig[g]
                self.model.geom_rgba[g, 3] = alpha
                self.model.geom_contype[g] = contype
                self.model.geom_conaffinity[g] = conaffinity
            else:
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

    def give_direct_visual(self, kind: str) -> bool:
        """Show a resource in the egocentric view and deliver it to the head.

        This replaces the old instant "direct" top-up for learning runs: the
        brain first sees the object, then the normal food/water event fires only
        when the object reaches the head/mouth zone.
        """
        kind = self._resource_kind(kind)
        if self._direct_delivery is not None:
            return False
        bodies = self._resource_bodies(kind)
        candidates = self._resource_candidates(kind)
        if not candidates:
            return False
        name = candidates[0]
        bid = bodies[name]
        head, fwd = self._head_forward()
        start = [head[i] + fwd[i] * DIRECT_PROVISION_START_M for i in range(3)]
        target = [head[i] + fwd[i] * DIRECT_PROVISION_TARGET_M for i in range(3)]
        self._respawn(name)
        self._respawn_at.pop(name, None)
        self._set_resource_delivery_contact(bid, enabled=False)
        self._place_resource_body(name, bid, start)
        self._direct_delivery = DirectProvision(
            kind=kind,
            name=name,
            body_id=bid,
            start=start,
            target=target,
        )
        self._mj.mj_forward(self.model, self.data)
        return True

    def _advance_direct_delivery(self, dt_s: float) -> None:
        delivery = self._direct_delivery
        if delivery is None:
            return
        head, fwd = self._head_forward()
        delivery.target = [
            head[i] + fwd[i] * DIRECT_PROVISION_TARGET_M for i in range(3)
        ]
        delivery.elapsed_s = min(
            delivery.duration_s, delivery.elapsed_s + max(0.0, float(dt_s))
        )
        t = min(1.0, delivery.elapsed_s / max(1e-6, delivery.duration_s))
        # Smoothstep: the object starts and arrives gently while staying visible.
        a = t * t * (3.0 - 2.0 * t)
        pos = [
            delivery.start[i] * (1.0 - a) + delivery.target[i] * a
            for i in range(3)
        ]
        self._place_resource_body(delivery.name, delivery.body_id, pos)
        self._mj.mj_forward(self.model, self.data)
        if t >= 1.0:
            self._consume(delivery.name)
            self._direct_delivery_events.append(
                {"type": delivery.kind, "intensity": 1.0, "source": delivery.name}
            )
            self._direct_delivery = None

    @staticmethod
    def _is_scatterable_prop(name: str) -> bool:
        """True for the agent's own open-arena consumables (the camping targets).

        Excludes (a) gift bodies -- the parent's free-joint provisions, whose pose
        lives in qpos and is driven by the delivery FSM, not ``body_pos`` -- and
        (b) habitat resources (``prop_food_h*`` / ``prop_water_h*``), which belong
        to the NPC ecology: the scripted crowd forages them *in-zone*, so moving
        them out of their habitat would break the foragers (a real bug, not just a
        test artifact). Only the open static props (e.g. the ``prop_food_s*`` ring)
        are scattered for anti-camping.
        """
        return "gift" not in name and "_h" not in name

    def _randomize_resources(self, seed: "int | None" = None) -> int:
        """Scatter the agent's open static food/water props to fresh positions.

        Anti-camping: the location of relief must not be memorizable across lives,
        so each new life re-scatters the consumables and only the SKILL of seeking
        transfers. Habitat resources and gift bodies are deliberately left in place
        (see ``_is_scatterable_prop``). Returns the RNG seed used (so the life is
        reproducible / telemetered), or -1 as a no-op when disabled or the scenario
        has no scatterable consumables.
        """
        if not randomize_resources_enabled():
            return -1
        name_to_bid: dict[str, int] = {}
        for nm, bid in {**self.food_bodies, **self.water_bodies, **self.medical_bodies}.items():
            if self._is_scatterable_prop(nm):
                name_to_bid[nm] = bid
        if not name_to_bid:
            return -1
        used_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        rng = random.Random(used_seed)
        zones = None
        if resource_placement_mode() == "zone":
            zones = [(h.center[0], h.center[1], h.radius) for h in active_habitats()]
        placement = scatter_positions(
            sorted(name_to_bid),
            rng,
            fence_radius=FENCE_RADIUS,
            min_dist=resource_min_dist(),
            margin=resource_fence_margin(),
            mode=resource_placement_mode(),
            zones=zones,
        )
        for nm, (x, y) in placement.items():
            bid = name_to_bid[nm]
            self.model.body_pos[bid][0] = x
            self.model.body_pos[bid][1] = y
            self._respawn_at.pop(nm, None)
            self._respawn(nm)  # un-hide + recollide if it had been consumed
        self.eaten.clear()
        self._resource_seed = used_seed
        self._mj.mj_forward(self.model, self.data)
        return used_seed

    def _npc_source_bodies(self, item: str) -> dict[str, int]:
        return self._resource_bodies(item)

    def _parent_delivery_due(self, now: float) -> bool:
        """Need-threshold trigger for parental provisioning (was a fixed timer).

        The parent offers only once its refractory has elapsed AND the agent is
        actually in need: a reservoir at/below the (fading) threshold. Missing
        reservoir telemetry is treated as unknown, not as permission to deliver.
        """
        if now < self._npc_next_deliver:
            return False
        res = self._agent_reservoirs
        if not res:
            return False
        return min(res.values()) <= _parent_effective_threshold(self._npc_offers)

    def _needed_resource_item(self) -> str:
        res = self._agent_reservoirs or {}
        vals = {
            "hydration": float(res.get("hydration", 100.0)),
            "energy": float(res.get("energy", 100.0)),
            "integrity": float(res.get("integrity", 100.0)),
        }
        need = min(vals, key=vals.get)
        if need == "hydration":
            return "water"
        if need == "integrity":
            return "medical_kit"
        return "food"

    def _request_legacy_parent(self, kind: str) -> bool:
        """Explicit dojo caregiver request through the legacy parent."""
        if self.npc_torso is None or self.npc_root_qadr < 0:
            return False
        request = str(kind or "").strip().lower()
        if request == "water":
            item = "water"
        elif request == "food":
            item = "food"
        elif request == "care":
            item = "medical_kit"
        else:
            return False
        if self._gift_addr.get(item, (-1, -1))[0] < 0:
            return False
        self._npc_frozen = False
        self._npc_item = item
        self._npc_requested_item = item
        self._npc_request_kind = request
        self._npc_next_deliver = 0.0
        self._npc_begin_delivery()
        return True

    def _request_parent(self, kind: str) -> bool:
        """Explicit caregiver request; routes to any visible parent in the scene."""
        if self._request_legacy_parent(kind):
            return True
        if self.crowd is not None:
            return bool(self.crowd.request_parent(kind))
        return False

    def _caregiver_parent_present(self) -> bool:
        return bool(self.npc_torso is not None or (self.crowd is not None and self.crowd.has_parent()))

    def _caregiver_kind(self) -> str:
        if self.npc_torso is not None:
            return "legacy_npc"
        if self.crowd is not None and self.crowd.has_parent():
            return "crowd_parent"
        return "none"

    def _caregiver_status(self) -> str:
        if self.npc_torso is None:
            if self.crowd is not None:
                return self.crowd.caregiver_status()
            return "missing_parent"
        if self._npc_frozen:
            return "paused"
        if self._npc_requested_item is not None:
            return "requested" if self._npc_phase == "pickup" else "delivering"
        if self._npc_phase in {"pickup", "deliver"}:
            return "delivering"
        return "idle"

    def _caregiver_request_kind(self) -> str:
        if self._npc_request_kind:
            return str(self._npc_request_kind)
        if self.crowd is not None:
            return str(self.crowd.request_kind)
        return ""

    def _caregiver_last_offer_item(self) -> str:
        if self._npc_last_offer_item:
            return str(self._npc_last_offer_item)
        if self.crowd is not None:
            return str(self.crowd.last_offer_item)
        return ""

    def _caregiver_delivery_count(self) -> int:
        crowd_count = int(self.crowd.delivery_count) if self.crowd is not None else 0
        return int(self._npc_delivery_count) + crowd_count

    def _caregiver_pending_request(self) -> bool:
        if self._npc_requested_item is not None:
            return True
        return bool(self.crowd is not None and self.crowd.requested_item is not None)

    def _npc_target_xy(self) -> tuple[float, float] | None:
        """Target point for the current parental phase (read-only on phase).

        Forage legs head for the nearest live consumable; the delivery legs walk
        to the chosen source to pick it up, then toward the agent to drop it.
        Always falls back to the agent so the parent has somewhere to go.
        """
        phase = self._npc_phase
        if phase == "pickup":
            xy = self._nearest_consumable_xy(
                self._npc_source_bodies(self._npc_item), exclude_sub="gift"
            )
            return xy if xy is not None else self._agent_xy()
        if phase == "deliver":
            return self._agent_xy()
        if phase == "seek_food":
            xy = self._nearest_consumable_xy(self.food_bodies, exclude_sub="gift")
            if xy is not None:
                return xy
            phase = "seek_water"
        if phase == "seek_water":
            xy = self._nearest_consumable_xy(self.water_bodies, exclude_sub="gift")
            if xy is not None:
                return xy
        # Fallback: accompany the agent.
        return self._agent_xy()

    def _npc_pose(self, phase: float) -> dict[str, float]:
        """Walk-cycle joint angles (radians) for a given stride phase.

        Legs swing fore-aft in antiphase; the swing leg's knee flexes (knees only
        bend one way, hence always negative); the arms counter-swing the legs.
        """
        r, lft = phase, phase + math.pi

        def knee(p: float) -> float:
            return -(NPC_KNEE_BASE + NPC_KNEE_BEND * max(0.0, math.sin(p)))

        return {
            "r_hip": NPC_HIP_SWING * math.sin(r),
            "l_hip": NPC_HIP_SWING * math.sin(lft),
            "r_knee": knee(r + math.pi / 2.0),
            "l_knee": knee(lft + math.pi / 2.0),
            "r_sh": -NPC_ARM_SWING * math.sin(r),
            "l_sh": -NPC_ARM_SWING * math.sin(lft),
        }

    def _apply_npc(self) -> None:
        """Animate the parent kinematically as a walking biped.

        The parent has no actuators, so each substep we write its root pose and
        joint angles directly. The body steers toward the current FSM target and
        advances at a fixed ground speed; the stride phase is locked to distance
        travelled, so the stance foot stays put (no skating) and the gait freezes
        into a neutral stand when the target is reached.
        """
        if self.npc_torso is None or self.npc_root_qadr < 0:
            return
        dt = float(self.model.opt.timestep)

        # Frozen by the UI "pause parent" control: seek no target, so ``moved``
        # stays 0 and the code below holds the current root pose with zero
        # velocity (the gait phase freezes mid-stance and any gift stays in hand).
        target = None if self._npc_frozen else self._npc_target_xy()
        moved = 0.0
        if target is not None:
            dx = target[0] - self._npc_x
            dy = target[1] - self._npc_y
            dist = math.hypot(dx, dy)
            # Rate-limited turn toward the target.
            desired = math.atan2(dy, dx)
            err = math.atan2(
                math.sin(desired - self._npc_yaw), math.cos(desired - self._npc_yaw)
            )
            self._npc_yaw += max(-NPC_TURN_RATE * dt, min(NPC_TURN_RATE * dt, err))
            if dist > NPC_STANDOFF:
                moved = min(NPC_WALK_SPEED * dt, dist - NPC_STANDOFF)
                self._npc_x += math.cos(self._npc_yaw) * moved
                self._npc_y += math.sin(self._npc_yaw) * moved

        # Keep the parent on stage.
        radius = math.hypot(self._npc_x, self._npc_y)
        if radius > FENCE_RADIUS:
            self._npc_x *= FENCE_RADIUS / radius
            self._npc_y *= FENCE_RADIUS / radius

        # Stride phase advances with distance only -> planted feet, and a still
        # parent (moved == 0) holds a neutral standing pose.
        self._npc_gait_phase += 2.0 * math.pi * moved / NPC_STRIDE_LENGTH
        pose = self._npc_pose(self._npc_gait_phase)
        bob = NPC_BOB_AMP * math.sin(2.0 * self._npc_gait_phase)

        # Root: position (with bob), yaw-only orientation, and a matching velocity.
        qa, da = self.npc_root_qadr, self.npc_root_dadr
        self.data.qpos[qa : qa + 3] = (
            self._npc_x,
            self._npc_y,
            STAND_ROOT_HEIGHT + bob,
        )
        half = self._npc_yaw * 0.5
        self.data.qpos[qa + 3 : qa + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        vx = math.cos(self._npc_yaw) * NPC_WALK_SPEED if moved > 0.0 else 0.0
        vy = math.sin(self._npc_yaw) * NPC_WALK_SPEED if moved > 0.0 else 0.0
        self.data.qvel[da : da + 6] = (vx, vy, 0.0, 0.0, 0.0, 0.0)

        # Hold every NPC hinge at its rest angle, then overwrite the animated ones.
        for qadr, vadr, q0 in self.npc_hinges:
            self.data.qpos[qadr] = q0
            self.data.qvel[vadr] = 0.0
        for key, (qadr, vadr, _q0) in self._npc_anim.items():
            self.data.qpos[qadr] = pose[key]
            self.data.qvel[vadr] = 0.0

        # Carry the current gift (food or water) in hand until it is dropped.
        if self._npc_carry and self.npc_hand is not None:
            gq, gd = self._gift_addr.get(self._npc_item, (-1, -1))
            gname = self._gift_names.get(self._npc_item, "")
            if gq >= 0 and gname not in self.eaten:
                hp = self.data.xpos[self.npc_hand]
                self.data.qpos[gq : gq + 3] = (float(hp[0]), float(hp[1]), float(hp[2]))
                self.data.qpos[gq + 3 : gq + 7] = (1.0, 0.0, 0.0, 0.0)
                if gd >= 0:
                    self.data.qvel[gd : gd + 6] = 0.0

    def _npc_recenter(self) -> None:
        """Teleport the parent back near the stage origin, upright and at rest."""
        if self.npc_torso is None or self.npc_root_qadr < 0:
            return
        self._npc_x, self._npc_y = 2.5, 2.5
        self._npc_yaw = math.atan2(-self._npc_y, -self._npc_x)
        self._npc_gait_phase = 0.0
        self._npc_phase = "seek_food"
        self._npc_item = "food"
        self._npc_requested_item = None
        self._npc_request_kind = ""
        self._npc_carry = False
        self._npc_next_deliver = float(self.data.time) + NPC_DELIVER_PERIOD_S
        qa, da = self.npc_root_qadr, self.npc_root_dadr
        half = self._npc_yaw * 0.5
        self.data.qpos[qa : qa + 3] = (self._npc_x, self._npc_y, STAND_ROOT_HEIGHT)
        self.data.qpos[qa + 3 : qa + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        self.data.qvel[da : da + 6] = 0.0
        for qadr, vadr, q0 in self.npc_hinges:
            self.data.qpos[qadr] = q0
            self.data.qvel[vadr] = 0.0
        self.data.xfrc_applied[self.npc_torso, :] = 0.0

    def _npc_begin_delivery(self) -> None:
        """Leave the forage loop to fetch the current gift and carry it to the agent."""
        self._npc_carry = False  # not in hand until the parent reaches a source
        self._npc_phase = "pickup"

    def _npc_pick_up(self) -> None:
        """Take the current gift into hand (restoring it first if it was consumed)."""
        gname = self._gift_names.get(self._npc_item, "")
        if gname in self.eaten:
            self._respawn(gname)
            self._respawn_at.pop(gname, None)
        self._npc_carry = True
        self._npc_phase = "deliver"

    def _npc_drop_gift(self, item: str, x: float, y: float) -> None:
        """Drop the chosen gift on the ground at ``(x, y)`` (a far-away offering)."""
        qa, da = self._gift_addr.get(item, (-1, -1))
        if qa < 0:
            return
        gname = self._gift_names.get(item, "")
        # Restore the gift if it was previously consumed/hidden.
        if gname in self.eaten:
            self._respawn(gname)
            self._respawn_at.pop(gname, None)
        self.data.qpos[qa : qa + 3] = (float(x), float(y), 0.12)
        self.data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
        if da >= 0:
            self.data.qvel[da : da + 6] = 0.0

    def _bear_contact_force(self) -> float:
        """Summed normal force of contacts between the bear and the humanoid (N)."""
        if self.bear_body is None:
            return 0.0
        mujoco = self._mj
        f6 = self._np.zeros(6)
        total = 0.0
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            r1 = int(self.model.body_rootid[self.model.geom_bodyid[con.geom1]])
            r2 = int(self.model.body_rootid[self.model.geom_bodyid[con.geom2]])
            if {r1, r2} == {self._torso_root, self._bear_root}:
                mujoco.mj_contactForce(self.model, self.data, ci, f6)
                total += abs(float(f6[0]))
        return total

    def _consumable_body(self, name: str) -> int | None:
        return (
            self.food_bodies.get(name)
            or self.water_bodies.get(name)
            or self.medical_bodies.get(name)
        )

    def _consume(self, name: str) -> None:
        """Mark a consumable used: hide + disable contact, and schedule respawn."""
        self.eaten.add(name)
        body = self._consumable_body(name)
        if body is None:
            return
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == body:
                if g not in self._geom_orig:
                    self._geom_orig[g] = (
                        float(self.model.geom_rgba[g, 3]),
                        int(self.model.geom_contype[g]),
                        int(self.model.geom_conaffinity[g]),
                    )
                self.model.geom_rgba[g, 3] = 0.0
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
        if name in self.water_bodies:
            delay = WATER_RESPAWN_S
        elif name in self.medical_bodies:
            delay = MEDICAL_RESPAWN_S
        else:
            delay = FOOD_RESPAWN_S
        self._respawn_at[name] = time.monotonic() + delay

    def _respawn(self, name: str) -> None:
        """Restore a consumed item to its original look and collidability."""
        self.eaten.discard(name)
        body = self._consumable_body(name)
        if body is None:
            return
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == body and g in self._geom_orig:
                alpha, contype, conaffinity = self._geom_orig[g]
                self.model.geom_rgba[g, 3] = alpha
                self.model.geom_contype[g] = contype
                self.model.geom_conaffinity[g] = conaffinity

    def _process_respawns(self) -> None:
        now = time.monotonic()
        due = [name for name, t in self._respawn_at.items() if t <= now]
        for name in due:
            self._respawn(name)
            del self._respawn_at[name]

    def scene_events(self, step: int) -> list[dict[str, Any]]:
        """Bear threat/contact, food and water consumption events for this obs."""
        events: list[dict[str, Any]] = []
        self._process_respawns()
        if self._direct_delivery_events:
            events.extend(self._direct_delivery_events)
            self._direct_delivery_events = []
        active_direct = (
            {self._direct_delivery.name} if self._direct_delivery is not None else set()
        )
        tpos = self.data.xpos[self.torso_id]
        root = [float(tpos[0]), float(tpos[1]), float(tpos[2])]

        if self.bear_body is not None:
            bpos = self.data.xpos[self.bear_body]
            dist = math.hypot(float(bpos[0]) - root[0], float(bpos[1]) - root[1])
            force = self._bear_contact_force()
            if (
                force >= BEAR_CONTACT_FORCE
                and step - self._last_bear_hit_step >= BEAR_HIT_COOLDOWN_STEPS
            ):
                self._last_bear_hit_step = step
                events.append(
                    {
                        "type": "collision",
                        "intensity": round(min(1.0, max(0.4, force / CONTACT_EVENT_SCALE)), 4),
                        "source": "prop_bear",
                    }
                )
            ti = threat_intensity(dist)
            if ti is not None and step - self._last_threat_step >= THREAT_COOLDOWN_STEPS:
                self._last_threat_step = step
                events.append(
                    {"type": "threat_near", "intensity": ti, "source": "prop_bear"}
                )

        if self.food_bodies:
            live = {
                name: [float(x) for x in self.data.xpos[b][:3]]
                for name, b in self.food_bodies.items()
                if name not in self.eaten
                and name not in active_direct
                and self._is_learner_consumable(name)
            }
            for name in eaten_now(root, live):
                self._consume(name)
                events.append({"type": "food", "intensity": 1.0, "source": name})

        if self.water_bodies:
            live_w = {
                name: [float(x) for x in self.data.xpos[b][:3]]
                for name, b in self.water_bodies.items()
                if name not in self.eaten
                and name not in active_direct
                and self._is_learner_consumable(name)
            }
            for name in drunk_now(root, live_w):
                self._consume(name)
                events.append({"type": "water", "intensity": 1.0, "source": name})

        if self.medical_bodies:
            live_m = {
                name: [float(x) for x in self.data.xpos[b][:3]]
                for name, b in self.medical_bodies.items()
                if name not in self.eaten
                and name not in active_direct
                and self._is_learner_consumable(name)
            }
            for name in eaten_now(root, live_m, MEDICAL_RADIUS):
                self._consume(name)
                events.append({"type": "medical_kit", "intensity": 1.0, "source": name})

        # Parent NPC behaviour: forage for itself most of the time, and only
        # occasionally fetch a morsel and drop it far from the agent (so the
        # agent must walk to it). Distinct event types (npc_eat/npc_drink/offer)
        # keep the parent's own consumption from ever crediting the agent — only
        # the agent consuming a dropped gift does, via the food/water events above.
        if self.npc_torso is not None and not self._npc_frozen:
            npos = self.data.xpos[self.npc_torso]
            nroot = [float(npos[0]), float(npos[1]), float(npos[2])]
            now = float(self.data.time)
            phase = self._npc_phase
            if phase == "seek_food":
                live_n = {
                    name: [float(x) for x in self.data.xpos[b][:3]]
                    for name, b in self.food_bodies.items()
                    if name not in self.eaten and "gift" not in name
                }
                hit = eaten_now(nroot, live_n)
                if hit:
                    self._consume(hit[0])
                    events.append({"type": "npc_eat", "intensity": 1.0, "source": "npc"})
                    self._npc_phase = "seek_water"
                elif not live_n:
                    self._npc_phase = "seek_water"
                if self._parent_delivery_due(now):
                    self._npc_item = self._needed_resource_item()
                    self._npc_begin_delivery()
            elif phase == "seek_water":
                live_nw = {
                    name: [float(x) for x in self.data.xpos[b][:3]]
                    for name, b in self.water_bodies.items()
                    if name not in self.eaten and "gift" not in name
                }
                hit_w = drunk_now(nroot, live_nw)
                if hit_w:
                    self._consume(hit_w[0])
                    events.append({"type": "npc_drink", "intensity": 1.0, "source": "npc"})
                    self._npc_phase = "seek_food"
                elif not live_nw:
                    self._npc_phase = "seek_food"
                if self._parent_delivery_due(now):
                    self._npc_item = self._needed_resource_item()
                    self._npc_begin_delivery()
            elif phase == "pickup":
                # Walk to the nearest source of the chosen item, then carry it.
                src = self._nearest_consumable_xy(
                    self._npc_source_bodies(self._npc_item), exclude_sub="gift"
                )
                if src is None:
                    self._npc_pick_up()  # nothing to walk to; carry from here
                else:
                    d = math.hypot(float(npos[0]) - src[0], float(npos[1]) - src[1])
                    if d <= NPC_PICKUP_RADIUS:
                        self._npc_pick_up()
            else:  # deliver
                d = math.hypot(float(npos[0]) - root[0], float(npos[1]) - root[1])
                if d <= NPC_DROP_DISTANCE:
                    offered_item = self._npc_item
                    # Drop the gift where the parent stands (~NPC_DROP_DISTANCE
                    # from the agent) so the agent must move to reach it.
                    self._npc_drop_gift(
                        offered_item, float(npos[0]), float(npos[1])
                    )
                    self._npc_carry = False
                    events.append(
                        {
                            "type": "offer",
                            "item": offered_item,
                            "intensity": 1.0,
                            "source": "npc",
                        }
                    )
                    self._npc_last_offer_item = offered_item
                    self._npc_delivery_count += 1
                    self._npc_requested_item = None
                    self._npc_request_kind = ""
                    self._npc_next_deliver = now + NPC_DELIVER_PERIOD_S
                    self._npc_offers += 1  # fades the need threshold over time
                    # Alternate so food and water are both provided over time.
                    self._npc_item = "water" if offered_item == "food" else "food"
                    self._npc_phase = "seek_food"

        # Scripted crowd: per-zone foraging + the crowd parent's threshold-gated
        # provisioning. Reservoirs (if streamed) gate the parent; npc_eat/npc_drink
        # stay credit-isolated, only agent-consumed gifts credit the agent.
        if self.crowd is not None:
            self.crowd.set_reservoirs(self._agent_reservoirs)
            events.extend(self.crowd.events(step, float(self.data.time)))
        return events

    def _apply_brace_gains(self) -> None:
        """Push per-joint tightness into MuJoCo's native spring/damper arrays.

        Each hinge's stiffness/damping interpolate from its native (soft) value
        at tightness 0 up to the fully-braced BRACE_STIFFNESS/BRACE_DAMPING at
        tightness 1. MuJoCo (implicitfast) integrates these joint springs stably,
        so the braced body holds the upright stand pose entirely through internal
        joint torque -- NO external wrench, feet fully loaded, no glide.
        """
        for h, jid in enumerate(self.hinge_jid):
            t = self._tightness[h]
            self.model.jnt_stiffness[jid] = (
                self._native_stiff[h] + (BRACE_STIFFNESS - self._native_stiff[h]) * t
            )
            self.model.dof_damping[self.hinge_qvel_adr[h]] = (
                self._native_damp[h] + (BRACE_DAMPING - self._native_damp[h]) * t
            )

    def _relax_braces(self) -> None:
        """Restore native joint springs/damping (true ragdoll for lifeless mode)."""
        for h, jid in enumerate(self.hinge_jid):
            self.model.jnt_stiffness[jid] = self._native_stiff[h]
            self.model.dof_damping[self.hinge_qvel_adr[h]] = self._native_damp[h]

    def reset_braces(self) -> None:
        """Re-tighten every joint brace to fully welded (ROM curriculum restart).

        Respects the master toggle: if the braces are switched off the joints
        stay relaxed to native springs (the re-welded tightness simply waits to
        take effect the moment they are switched back on).
        """
        n = len(self._tightness)
        self._tightness = [1.0] * n
        self._joint_pe_ema = [1.0] * n
        self._brace_dwell = [0.0] * n
        if self._braces_enabled:
            self._apply_brace_gains()
        else:
            self._relax_braces()

    def _advance_motion(self, dt: float) -> None:
        """Advance a motion stance's phase and retarget the braced reference.

        For a static stance this is a no-op (``_q_ref`` is fixed). For a motion
        stance the phase advances by ``dt / period`` and ``_q_ref`` (plus the
        MuJoCo spring reference) is recomputed from the interpolated keyframe
        trajectory, so the stiff braces drive the limbs through the gait while the
        feet/hands push the floor -- no external wrench, real contact-driven travel.
        """
        stance = stance_lib.get_stance(self._stance_name)
        if not stance.is_motion:
            return
        self._stance_phase += dt / max(1e-6, stance.period_s)
        # Hold mode loops every motion (even the one-shot Rise) so it runs until
        # disabled; otherwise a one-shot motion clamps and holds its final pose.
        if stance.loop or self._movement_hold:
            self._stance_phase %= 1.0
        else:
            self._stance_phase = min(1.0, self._stance_phase)
        self._q_ref = stance_lib.motion_ref(
            stance,
            self._stance_phase,
            self.hinge_names,
            list(self.hinge_qpos0),
            self._hinge_ranges,
        )
        self._seed_stance_spring()

    def _hold_braces(self) -> None:
        """Hold every joint fully welded -- the ROM curriculum is suspended.

        Used by "hold movement" mode: pin each hinge's tightness back to fully
        welded (1.0) and re-apply the brace gains, so the braces drive the stance
        pose / motion trajectory rigidly with no range-of-motion release. Leaves
        the PE EMA / dwell state untouched, so releasing hold resumes the ratchet.
        """
        for h in range(len(self._tightness)):
            self._tightness[h] = 1.0
        self._apply_brace_gains()

    def _update_braces(self, dt: float) -> None:
        """One brace control tick: ratchet each joint's ROM open as its PE falls.

        Reads the brain's latest per-joint proprioceptive forward-model error (set
        by ``apply_action`` from the motor command's ``joint_pe``) and runs the pure
        ``brace_ratchet`` law per hinge: sustained low PE widens that joint's range
        of motion (tightness drops). With no PE yet (no brain, startup) the braces
        stay fully welded, so the body stands as a statue until it has earned slack.
        """
        pe = self._joint_pe_buf
        for h in range(len(self._tightness)):
            pe_now = float(pe[h]) if pe is not None and h < len(pe) else self._joint_pe_ema[h]
            (
                self._tightness[h],
                self._joint_pe_ema[h],
                self._brace_dwell[h],
            ) = brace_ratchet(
                tightness=self._tightness[h],
                pe_ema=self._joint_pe_ema[h],
                pe_now=pe_now,
                dwell=self._brace_dwell[h],
                dt=dt,
            )
        self._apply_brace_gains()

    def _read_limb_loads(self) -> None:
        """Refresh per-part touch loads from every touch sensor (ALL modes).

        Each touch sensor sums the normal contact force on its site; dividing by
        body weight yields a soft, ~0..1+ load fraction. Stored under the
        sensor's short name (``touch_`` prefix stripped) so the snapshot,
        perception and the dashboard all read one source of truth. Load is a
        property of the body, not of a mode, so this runs unconditionally every
        ``step()`` -- braced, free and lifeless alike.
        """
        weight = max(1e-6, self._body_weight)
        sd = self.data.sensordata
        for name, adr in self.touch_sensors:
            key = name[len("touch_") :] if name.startswith("touch_") else name
            self._part_loads[key] = float(sd[adr]) / weight
        # Derived limb loads (back-compat telemetry + curriculum signals).
        self._foot_load_l = self._part_loads.get("left_foot", 0.0)
        self._foot_load_r = self._part_loads.get("right_foot", 0.0)
        self._hand_load_l = self._part_loads.get("left_hand", 0.0)
        self._hand_load_r = self._part_loads.get("right_hand", 0.0)

    @staticmethod
    def _body_part_for_joint(name: str) -> str:
        low = str(name).lower()
        side = "left" if low.startswith("left_") or low.startswith("l_") else "right" if low.startswith("right_") or low.startswith("r_") else ""
        if "abdomen" in low or "waist" in low or "pelvis" in low or "torso" in low:
            return "torso"
        if "neck" in low or "head" in low:
            return "head"
        if "shoulder" in low:
            return f"{side}_upper_arm" if side else "torso"
        if "elbow" in low:
            return f"{side}_lower_arm" if side else "torso"
        if "wrist" in low or "hand" in low:
            return f"{side}_hand" if side else "torso"
        if "hip" in low:
            return f"{side}_thigh" if side else "torso"
        if "knee" in low:
            return f"{side}_shin" if side else "torso"
        if "ankle" in low or "foot" in low:
            return f"{side}_foot" if side else "torso"
        return "torso"

    def _effort_force(self, actuator_idx: int) -> float:
        try:
            af = getattr(self.data, "actuator_force", None)
            if af is not None and actuator_idx < len(af):
                val = float(af[actuator_idx])
                if math.isfinite(val):
                    return val
        except Exception:
            pass
        try:
            val = float(self.data.ctrl[actuator_idx])
        except Exception:
            val = 0.0
        return val if math.isfinite(val) else 0.0

    def _update_effort_telemetry(self, dt: float) -> None:
        n = int(self.model.nu)
        efforts = [0.0] * n
        works = [0.0] * n
        strains = [0.0] * n
        by_part_effort = {p: 0.0 for p in BODY_PARTS}
        by_part_work = {p: 0.0 for p in BODY_PARTS}
        by_part_strain = {p: 0.0 for p in BODY_PARTS}
        by_part_count = {p: 0 for p in BODY_PARTS}
        for a in range(n):
            h = self.act_joint[a] if a < len(self.act_joint) else a
            qd = float(self.data.qvel[self.hinge_qvel_adr[h]]) if h < len(self.hinge_qvel_adr) else 0.0
            force = self._effort_force(a)
            lo_c, hi_c = self.model.actuator_ctrlrange[a]
            ctrl_span = max(1e-6, max(abs(float(lo_c)), abs(float(hi_c))))
            effort = min(1.0, abs(force) / ctrl_span)
            work = min(1.0, abs(force * qd * max(0.0, dt)) / max(1e-6, ctrl_span))
            range_pressure = 0.0
            if h < len(self._hinge_ranges):
                lo, hi = self._hinge_ranges[h]
                if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                    q = float(self.data.qpos[self.hinge_qpos_adr[h]])
                    margin = min(max(0.0, q - lo), max(0.0, hi - q)) / max(1e-6, hi - lo)
                    range_pressure = max(0.0, min(1.0, 0.15 - margin) / 0.15)
            strain = min(1.0, EFFORT_STRAIN_GAIN * effort + 0.25 * min(1.0, abs(qd)) + 0.3 * range_pressure)
            efforts[a] = effort
            works[a] = work
            strains[a] = strain
            part = self._body_part_for_joint(self.hinge_names[h] if h < len(self.hinge_names) else "")
            by_part_effort[part] += effort
            by_part_work[part] += work
            by_part_strain[part] += strain
            by_part_count[part] += 1
        for part in BODY_PARTS:
            c = max(1, by_part_count[part])
            by_part_effort[part] /= c
            by_part_work[part] /= c
            by_part_strain[part] = max(by_part_strain[part] / c, min(1.0, float(self._part_loads.get(part, 0.0)) * 0.15))
            load = max(0.0, float(self._part_loads.get(part, 0.0)))
            stimulus = min(1.0, max(by_part_effort[part], by_part_strain[part], load * 0.2))
            prev = float(self._part_fatigue.get(part, 0.0))
            if stimulus > prev:
                alpha = min(1.0, max(0.0, dt) / max(1e-6, EFFORT_FATIGUE_RISE_S))
            else:
                alpha = min(1.0, max(0.0, dt) / max(1e-6, EFFORT_FATIGUE_RECOVERY_S))
            fatigue = (1.0 - alpha) * prev + alpha * stimulus
            pain = min(1.0, EFFORT_FATIGUE_PAIN_GAIN * fatigue + EFFORT_STRAIN_PAIN_GAIN * by_part_strain[part])
            self._part_effort[part] = by_part_effort[part]
            self._part_work[part] = by_part_work[part]
            self._part_strain[part] = by_part_strain[part]
            self._part_fatigue[part] = fatigue
            self._part_pain[part] = pain
        self._actuator_effort = efforts
        self._actuator_work = works
        self._joint_strain = strains
        self._joint_fatigue = [
            float(self._part_fatigue.get(self._body_part_for_joint(self.hinge_names[self.act_joint[a]]), 0.0))
            if a < len(self.act_joint) and self.act_joint[a] < len(self.hinge_names)
            else 0.0
            for a in range(n)
        ]
        self._effort_total = sum(efforts) / max(1, len(efforts))
        self._work_total = sum(works) / max(1, len(works))
        self._strain_total = max(self._part_strain.values()) if self._part_strain else 0.0
        self._fatigue_total = max(self._part_fatigue.values()) if self._part_fatigue else 0.0
        self._pain_total = max(self._part_pain.values()) if self._part_pain else 0.0

    def _apply_motor(self) -> None:
        """Fast PD tracking of the brain's postural targets (physics-rate inner loop).

        The brain sets equilibrium-point targets at the cognitive rate; the
        body's reflex-like PD loop tracks them every substep. Before any motor
        command arrives, hold the rest pose so the body doesn't collapse.
        """
        targets = self._motor_targets
        for a in range(self.model.nu):
            h = self.act_joint[a]
            q = float(self.data.qpos[self.hinge_qpos_adr[h]])
            qd = float(self.data.qvel[self.hinge_qvel_adr[h]])
            if targets is None:
                target = self._q_ref[h]
            else:
                lo, hi = self.act_target_lo[a], self.act_target_hi[a]
                u = float(targets[a]) if a < targets.shape[0] else 0.0
                target = lo + (u + 1.0) * 0.5 * (hi - lo)
            ctrl = PD_KP * (target - q) - PD_KD * qd
            lo_c, hi_c = self.model.actuator_ctrlrange[a]
            self.data.ctrl[a] = max(float(lo_c), min(float(hi_c), ctrl))

    def _apply_teacher_spotter(self) -> None:
        """Dojo live spotter: vertical support plus roll/pitch stabilization.

        This intentionally applies no horizontal force. It can keep the learner
        from face-planting while the skill is being taught, but it cannot walk the
        body toward resources or satisfy a locomotion objective by itself.
        """
        self._teacher_support_force = 0.0
        self._teacher_support_torque = 0.0
        self._teacher_drop_m = 0.0
        self._teacher_target_drop_m = 0.25
        self._teacher_height_error_m = 0.0
        self._teacher_vertical_velocity = 0.0
        self._teacher_support_mode = "off"
        assist = max(0.0, min(1.0, self._teacher_assist if self._teacher_live else 0.0))
        if assist <= 1e-5:
            return

        pos = self.data.xpos[self.torso_id]
        vel = self.data.cvel[self.torso_id][3:]
        xmat = self.data.xmat[self.torso_id].reshape(3, 3)
        roll = math.atan2(xmat[2, 1], xmat[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))
        ang = self.data.cvel[self.torso_id][:3]
        if not all(
            math.isfinite(float(v))
            for v in (
                pos[2],
                vel[2],
                roll,
                pitch,
                ang[0],
                ang[1],
                self._stance_root_z,
                self._body_weight,
            )
        ):
            return

        # Height-band spotter: let the learner load its feet while staying inside
        # a parent-hold band. It catches before a full fall instead of waiting for
        # the phase failure gate.
        target_drop = 0.25
        emergency_drop = 0.30
        release_drop = 0.15
        drop = max(-0.25, min(0.75, float(self._stance_root_z) - float(pos[2])))
        downward_v = max(0.0, -float(vel[2]))
        height_error = max(0.0, drop - target_drop)
        catch_error = max(0.0, drop - emergency_drop)
        if drop >= emergency_drop or downward_v > 0.8:
            mode = "catch"
        elif height_error > 0.0:
            mode = "recover"
        elif drop >= release_drop:
            mode = "hold_band"
        else:
            mode = "observe"

        if mode == "observe":
            base_unload = 0.08
        elif mode == "hold_band":
            base_unload = 0.22
        elif mode == "recover":
            base_unload = 0.35
        else:
            base_unload = 0.5
        upward = assist * (
            self._body_weight * base_unload
            + 420.0 * height_error
            + 75.0 * downward_v
            + 900.0 * catch_error
        )
        upward = max(0.0, min(1.35 * self._body_weight, upward))
        # MuJoCo xfrc_applied is [force_x, force_y, force_z, torque_x,
        # torque_y, torque_z]. Keep teacher support vertical-only.
        self.data.xfrc_applied[self.torso_id, 2] += upward

        # Upright spotter: resist roll/pitch only. Yaw is left alone.
        tx = assist * (-45.0 * roll - 5.0 * float(ang[0]))
        ty = assist * (-45.0 * pitch - 5.0 * float(ang[1]))
        cap = 35.0
        tx = max(-cap, min(cap, tx))
        ty = max(-cap, min(cap, ty))
        self.data.xfrc_applied[self.torso_id, 3] += tx
        self.data.xfrc_applied[self.torso_id, 4] += ty

        self._teacher_support_force = float(upward / max(1e-6, self._body_weight))
        self._teacher_support_torque = float(math.hypot(tx, ty))
        self._teacher_drop_m = float(max(0.0, drop))
        self._teacher_target_drop_m = target_drop
        self._teacher_height_error_m = float(height_error)
        self._teacher_vertical_velocity = float(downward_v)
        self._teacher_support_mode = mode

    def snapshot(self) -> BodySnapshot:
        pos = self.data.xpos[self.torso_id]
        xmat = self.data.xmat[self.torso_id].reshape(3, 3)
        roll = math.atan2(xmat[2, 1], xmat[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -float(xmat[2, 0]))))
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        lin_vel = self.data.cvel[self.torso_id][3:]

        joints: list[float] = []
        for h in range(len(self.hinge_qpos_adr)):
            joints.append(float(self.data.qpos[self.hinge_qpos_adr[h]]))
            joints.append(float(self.data.qvel[self.hinge_qvel_adr[h]]))

        contacts = {
            name: float(self.data.sensordata[adr]) for name, adr in self.touch_sensors
        }

        props = []
        for name, b in self.prop_bodies:
            if name in self.eaten:
                continue
            p = self.data.xpos[b]
            props.append(
                {
                    "id": name,
                    "kind": prop_kind(name),
                    "position": [float(p[0]), float(p[1]), float(p[2])],
                }
            )

        # The parent NPC is a perceivable entity (kind "npc"): it enters the
        # egocentric graph and working memory like any other entity, and the
        # agent additionally sees it through its camera.
        if self.npc_torso is not None:
            np_ = self.data.xpos[self.npc_torso]
            props.append(
                {
                    "id": "npc",
                    "kind": "npc",
                    "position": [float(np_[0]), float(np_[1]), float(np_[2])],
                }
            )

        # Each scripted crowd NPC is a perceivable entity (kind "npc"), so the
        # learner sees the village demonstrators in its egocentric graph too.
        if self.crowd is not None:
            props.extend(self.crowd.entities())

        speed = math.hypot(float(lin_vel[0]), float(lin_vel[1]))
        motor = (
            [float(x) for x in self._motor_targets.tolist()]
            if self._motor_targets is not None
            else None
        )
        body_parts = {
            name: [float(x) for x in self.data.xpos[bid]]
            for name, bid in self.body_part_ids.items()
        } or None
        return BodySnapshot(
            position=[float(pos[0]), float(pos[1]), float(pos[2])],
            orientation=[roll, pitch, yaw],
            velocity=[float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])],
            joints=joints,
            contacts=contacts,
            props=props,
            moving=speed > 0.08,
            motor=motor,
            n_actuators=int(self.model.nu),
            rom_mean=float(sum(1.0 - t for t in self._tightness) / max(1, len(self._tightness))),
            brace_engaged=(
                float(sum(self._tightness) / max(1, len(self._tightness)))
                if self._braces_enabled and not self._lifeless
                else 0.0
            ),
            rom_frac=[float(1.0 - t) for t in self._tightness],
            braces_enabled=bool(self._braces_enabled),
            stance=str(self._stance_name),
            stance_phase=float(self._stance_phase),
            movement_hold=bool(self._movement_hold),
            manual_auto_reset=bool(self._manual_auto_reset),
            foot_load_l=float(self._foot_load_l),
            foot_load_r=float(self._foot_load_r),
            hand_load_l=float(self._hand_load_l),
            hand_load_r=float(self._hand_load_r),
            teacher_support_active=bool(self._teacher_live and self._teacher_assist > 1e-5),
            teacher_support_force=float(self._teacher_support_force),
            teacher_support_torque=float(self._teacher_support_torque),
            teacher_drop_m=float(self._teacher_drop_m),
            teacher_target_drop_m=float(self._teacher_target_drop_m),
            teacher_height_error_m=float(self._teacher_height_error_m),
            teacher_vertical_velocity=float(self._teacher_vertical_velocity),
            teacher_support_mode=str(self._teacher_support_mode),
            caregiver_parent_present=self._caregiver_parent_present(),
            caregiver_kind=self._caregiver_kind(),
            caregiver_status=self._caregiver_status(),
            caregiver_request_kind=self._caregiver_request_kind(),
            caregiver_last_offer_item=self._caregiver_last_offer_item(),
            caregiver_delivery_count=self._caregiver_delivery_count(),
            caregiver_pending_request=self._caregiver_pending_request(),
            part_loads={k: float(v) for k, v in self._part_loads.items()},
            body_map={
                "parts": list(BODY_PARTS),
                "contact_load": {p: float(self._part_loads.get(p, 0.0)) for p in BODY_PARTS},
                "effort": {p: float(self._part_effort.get(p, 0.0)) for p in BODY_PARTS},
                "work": {p: float(self._part_work.get(p, 0.0)) for p in BODY_PARTS},
                "strain": {p: float(self._part_strain.get(p, 0.0)) for p in BODY_PARTS},
                "fatigue": {p: float(self._part_fatigue.get(p, 0.0)) for p in BODY_PARTS},
                "pain": {p: float(self._part_pain.get(p, 0.0)) for p in BODY_PARTS},
            },
            effort={
                "actuator_effort": [float(x) for x in self._actuator_effort],
                "actuator_work": [float(x) for x in self._actuator_work],
                "joint_strain": [float(x) for x in self._joint_strain],
                "joint_fatigue": [float(x) for x in self._joint_fatigue],
                "effort_total": float(self._effort_total),
                "work_total": float(self._work_total),
                "strain_total": float(self._strain_total),
                "fatigue_total": float(self._fatigue_total),
                "pain_total": float(self._pain_total),
                "support_effort": float(self._teacher_support_force + self._teacher_support_torque),
            },
            body_parts=body_parts,
        )

    def _render_camera_b64(self, camera: str) -> str:
        from PIL import Image

        self.renderer.update_scene(self.data, camera=camera)
        pixels = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_vision_b64(self) -> str | None:
        if self.renderer is None:
            return None
        return self._render_camera_b64("egocentric")

    def render_views_b64(self) -> dict[str, str]:
        """Spectator camera frames, refreshed every VIEWS_EVERY observations."""
        if self.renderer is None or not self.view_cameras:
            return {}
        self._views_counter += 1
        if self._views_cache and (self._views_counter - 1) % VIEWS_EVERY != 0:
            return self._views_cache
        self._views_cache = {
            name: self._render_camera_b64(name) for name in self.view_cameras
        }
        return self._views_cache

    def recenter(self) -> None:
        """Re-pose the body into the active stance at the stage origin.

        The root is placed at the stance's spawn height/orientation and every
        hinge snapped to the stance reference (``_q_ref``) with zero velocity, so
        the body re-poses into whatever stance is active (upright stand, all-fours,
        kneel, ...). Props stay where they are.
        """
        mujoco = self._mj
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        adr = int(self.model.jnt_qposadr[rid])
        dadr = int(self.model.jnt_dofadr[rid])
        self.data.qpos[adr : adr + 3] = (0.0, 0.0, self._stance_root_z)
        self.data.qpos[adr + 3 : adr + 7] = self._stance_root_quat
        self.data.qvel[dadr : dadr + 6] = 0.0
        for h, qadr in enumerate(self.hinge_qpos_adr):
            self.data.qpos[qadr] = self._q_ref[h]
            self.data.qvel[self.hinge_qvel_adr[h]] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self._move_dir = (0.0, 0.0)
        self._move_speed = 0.0
        # Reposition only: snap back into the active stance pose with zero velocity
        # but PRESERVE the earned range of motion (per-joint tightness / PE EMA /
        # dwell carry over). Re-welding the braces is the separate "Reset ROM"
        # action (reset_braces); recentering must not wipe curriculum progress.
        mujoco.mj_forward(self.model, self.data)

    def perturb(self, strength: str = "small") -> None:
        """Apply a brief lateral disturbance for Skill Dojo recovery practice."""
        mag = 0.6 if str(strength).strip().lower() == "small" else 1.0
        rid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, "root")
        dadr = int(self.model.jnt_dofadr[rid])
        # Root free-joint qvel layout starts with angular velocity then linear.
        self.data.qvel[dadr + 3] += mag
        self.data.qvel[dadr + 4] += 0.35 * mag
        self.data.xfrc_applied[self.torso_id, 0] += 35.0 * mag
        self.data.xfrc_applied[self.torso_id, 1] += 20.0 * mag


# ---------------------------------------------------------------------------
# WebSocket loops
# ---------------------------------------------------------------------------


async def _run(
    ws_url: str,
    *,
    steps: int,
    dry_run: bool,
    vision: bool,
    audio: bool = False,
    view: bool = False,
    scene: str = "default",
    elements: "list[str] | None" = None,
    baseline: "str | None" = None,
    braces: bool = False,
) -> None:
    import websockets

    interval = obs_interval_s()
    latest_action: dict[str, Any] = {}
    pending_commands: list[str] = []
    # Shared liveness state between the receiver and the step loop. The brain is
    # the body's only driver: a death signal or a command drought -> go limp.
    liveness: dict[str, Any] = {"lifeless": False, "last_action": time.perf_counter()}

    # The native MuJoCo viewer can momentarily stall the adapter's asyncio loop
    # while GLFW creates/syncs the window. The default websockets keepalive treats
    # that as a dead connection and closes with 1011 ping timeout, which makes
    # the dashboard's Live Window button appear broken. The server receives
    # regular observations as its real liveness signal, so disable client-side
    # keepalive pings here.
    async with websockets.connect(
        ws_url,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    ) as ws:

        async def receiver() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "body_command":
                    cmd = msg.get("command")
                    if isinstance(cmd, str):
                        pending_commands.append(cmd)
                    continue
                if msg.get("type") == "death":
                    # The mind has frozen; stop driving the body immediately.
                    liveness["lifeless"] = True
                    print("[body] brain death received -> going limp", flush=True)
                    continue
                action = msg.get("action")
                if isinstance(action, dict):
                    # A fresh command means the brain is alive and driving again
                    # (revive/reset resumes the action stream).
                    liveness["lifeless"] = False
                    liveness["last_action"] = time.perf_counter()
                    latest_action.clear()
                    latest_action.update(action)

        recv_task = asyncio.create_task(receiver())
        try:
            if dry_run:
                i = 0
                while steps <= 0 or i < steps:
                    snap_dry = dry_snapshot(i)
                    events, _ = body_events(
                        snap_dry.contacts, STAND_ROOT_HEIGHT, was_fallen=False
                    )
                    obs = build_body_observation(
                        snap_dry,
                        events=events,
                        audio=audio_payload(synth_audio_window(events, snap_dry.contacts, seed=i))
                        if audio
                        else None,
                    )
                    await ws.send(json.dumps(obs))
                    await asyncio.sleep(interval)
                    i += 1
                return

            sim = HumanoidSim(
                vision=vision, view=view, scene=scene, elements=elements, braces=braces
            )
            print(
                f"[body] elements={sim.elements} "
                f"props={[n for n, _ in sim.prop_bodies]}"
            )
            # Actuator contract: the brain's motor head emits one PD target per
            # actuator. Mismatch silently corrupts the efferent mapping.
            expected_nu = int(os.environ.get("DECADIC_N_ACTUATORS", "21"))
            if sim.model.nu != expected_nu:
                print(
                    f"[body] WARNING actuator mismatch: model.nu={sim.model.nu} "
                    f"but DECADIC_N_ACTUATORS={expected_nu}. Set "
                    f"DECADIC_N_ACTUATORS={sim.model.nu} on the server so the "
                    f"motor head matches this body."
                )
            else:
                print(f"[body] actuator contract OK: nu={sim.model.nu}")
            # Distinctness baseline: a reactive controller drives the body locally;
            # free the joints so it can actually try to walk, and ignore the brain.
            baseline_ctl: "BaselineController | None" = None
            t_baseline = 0.0
            if baseline:
                sim.set_braces_enabled(False)
                baseline_ctl = BaselineController(sim, baseline)
                print(
                    f"[body] BASELINE mode={baseline}: reactive controller driving "
                    f"the body (brain ignored); braces off"
                )
            was_fallen = False
            env_paused = False
            manual_reset_cooldown_until = 0.0
            prev_contacts: dict[str, float] = {}
            impact_cooldown: dict[str, int] = {}
            i = 0
            last_wall = time.perf_counter()
            # 2026-07-04 body-loop budget ledger: the brain was starving at
            # 0.37 obs/s and the ~2.7 s/iteration had no attribution. Phase
            # times accumulate here and print every 50 steps.
            phase_ms = {"physics": 0.0, "vision": 0.0, "views": 0.0, "audio": 0.0, "send": 0.0}
            while steps <= 0 or i < steps:
                while pending_commands:
                    cmd = pending_commands.pop(0)
                    if cmd == "recenter":
                        sim.recenter()
                        print("[body] recenter command applied")
                    elif cmd == "reset_braces":
                        sim.reset_braces()
                        print("[body] joint braces re-welded (ROM reset)")
                    elif cmd == "braces_on":
                        sim.set_braces_enabled(True)
                    elif cmd == "braces_off":
                        sim.set_braces_enabled(False)
                    elif cmd == "hold_on":
                        sim.set_movement_hold(True)
                    elif cmd == "hold_off":
                        sim.set_movement_hold(False)
                    elif cmd == "manual_auto_reset_on":
                        sim.set_manual_auto_reset(True)
                    elif cmd == "manual_auto_reset_off":
                        sim.set_manual_auto_reset(False)
                    elif cmd.startswith("set_stance:"):
                        sim.set_stance(cmd.split(":", 1)[1])
                    elif cmd.startswith("perturb:"):
                        sim.perturb(cmd.split(":", 1)[1])
                        print(f"[body] perturb command applied ({cmd})", flush=True)
                    elif cmd == "pause":
                        env_paused = True
                        print("[body] environment paused (physics frozen)", flush=True)
                    elif cmd == "resume":
                        env_paused = False
                        # Fresh wall clock + action timer so the first post-resume
                        # step takes a small dt and the staleness timeout doesn't
                        # immediately mark the body lifeless.
                        last_wall = time.perf_counter()
                        liveness["last_action"] = time.perf_counter()
                        print("[body] environment resumed", flush=True)
                    elif cmd == "open_viewer":
                        sim.open_viewer()
                    elif cmd == "close_viewer":
                        sim.close()
                        print("[body] live viewer closed by request", flush=True)
                    elif cmd in ("give_water_near", "give_food_near", "give_medical_kit_near"):
                        if cmd == "give_water_near":
                            kind = "water"
                        elif cmd == "give_medical_kit_near":
                            kind = "medical_kit"
                        else:
                            kind = "food"
                        if sim.give_near(kind):
                            print(f"[body] placed {kind} near the agent", flush=True)
                        else:
                            print(
                                f"[body] no {kind} prop in this scenario to place",
                                flush=True,
                            )
                    elif cmd in (
                        "give_water_direct_visual",
                        "give_food_direct_visual",
                        "give_medical_kit_direct_visual",
                    ):
                        if cmd == "give_water_direct_visual":
                            kind = "water"
                        elif cmd == "give_medical_kit_direct_visual":
                            kind = "medical_kit"
                        else:
                            kind = "food"
                        if sim.give_direct_visual(kind):
                            print(
                                f"[body] delivering {kind} through egocentric view",
                                flush=True,
                            )
                        else:
                            print(
                                f"[body] could not start direct {kind} delivery",
                                flush=True,
                            )
                    elif cmd == "randomize_resources" or cmd.startswith("randomize_resources:"):
                        seed: "int | None" = None
                        if ":" in cmd:
                            try:
                                seed = int(cmd.split(":", 1)[1])
                            except ValueError:
                                seed = None
                        used = sim._randomize_resources(seed)
                        print(f"[body] resources randomized (seed={used})", flush=True)
                    elif cmd in ("npc_pause", "parent_pause"):
                        sim._npc_frozen = True
                        if sim.crowd is not None:
                            sim.crowd.set_frozen(True)
                        print("[body] parent/crowd paused (hold position)", flush=True)
                    elif cmd in ("npc_resume", "parent_resume", "parent_enable"):
                        sim._npc_frozen = False
                        if sim.crowd is not None:
                            sim.crowd.set_frozen(False)
                        print("[body] parent/crowd resumed", flush=True)
                    elif cmd.startswith("parent_request:"):
                        kind = cmd.split(":", 1)[1]
                        if sim._request_parent(kind):
                            print(f"[body] parent caregiver request queued ({kind})", flush=True)
                        else:
                            print(f"[body] parent caregiver request failed ({kind})", flush=True)
                    elif cmd == "parent_status":
                        print(f"[body] parent status: {sim._caregiver_status()}", flush=True)

                if env_paused:
                    # Frozen world: hold pose, integrate no physics, emit no
                    # observations (the brain is paused in lockstep).
                    await asyncio.sleep(interval)
                    continue

                now = time.perf_counter()
                dt = min(0.25, now - last_wall) or interval
                if baseline_ctl is not None:
                    # Reactive baseline drives the actuators; the brain's commands
                    # are ignored and the body always stays "alive" so it keeps
                    # moving for the full comparison.
                    t_baseline += dt
                    ctrl_vec = baseline_ctl.targets(t_baseline)
                    sim.apply_action(
                        {"type": "motor", "parameters": {"ctrl": ctrl_vec}}
                    )
                    sim.set_lifeless(False)
                else:
                    sim.apply_action(dict(latest_action))
                    # Go limp on explicit death or a command drought (covers pause,
                    # disconnect, and server stalls, not just mortality).
                    stale = (now - liveness["last_action"]) > COMMAND_STALE_S
                    sim.set_lifeless(bool(liveness["lifeless"]) or stale)
                sim.step(dt)
                phase_ms["physics"] += (time.perf_counter() - now) * 1000.0
                last_wall = now

                snap = sim.snapshot()
                if math.hypot(snap.position[0], snap.position[1]) > FENCE_RADIUS:
                    print(
                        f"[body] fence: wandered to ({snap.position[0]:.1f},"
                        f"{snap.position[1]:.1f}); recentering"
                    )
                    sim.recenter()
                    snap = sim.snapshot()
                events, was_fallen = body_events(
                    snap.contacts,
                    snap.position[2],
                    was_fallen=was_fallen,
                    prev_contacts=prev_contacts,
                    velocity=snap.velocity,
                    step=i,
                    cooldown_until=impact_cooldown,
                    fall_height=sim._stance_fall_z,
                )
                fell_this_step = any(
                    isinstance(ev, dict) and ev.get("type") == "fall" for ev in events
                )
                if (
                    fell_this_step
                    and sim._manual_auto_reset
                    and stance_lib.get_stance(sim._stance_name).is_motion
                    and time.perf_counter() >= manual_reset_cooldown_until
                ):
                    stance_name = sim._stance_name
                    sim.set_stance(stance_name)
                    snap = sim.snapshot()
                    prev_contacts = {}
                    was_fallen = False
                    manual_reset_cooldown_until = time.perf_counter() + 2.0
                    events.append(
                        {
                            "type": "manual_scaffold_reset",
                            "intensity": 0.4,
                            "source": stance_name,
                        }
                    )
                    print(
                        f"[body] manual auto-reset reapplied motion stance {stance_name}",
                        flush=True,
                    )
                prev_contacts = {k: float(v) for k, v in snap.contacts.items()}
                events.extend(sim.scene_events(i))
                _t0 = time.perf_counter()
                _vision = sim.render_vision_b64()
                _t1 = time.perf_counter()
                _views = sim.render_views_b64()
                _t2 = time.perf_counter()
                _audio = (
                    audio_payload(synth_audio_window(events, snap.contacts))
                    if audio
                    else None
                )
                _t3 = time.perf_counter()
                obs = build_body_observation(
                    snap,
                    events=events,
                    vision_b64=_vision,
                    vision_resolution=(sim.vision_size, sim.vision_size),
                    debug_views=_views,
                    audio=_audio,
                    control_mode="active_inference",
                )
                await ws.send(json.dumps(obs))
                _t4 = time.perf_counter()
                phase_ms["vision"] += (_t1 - _t0) * 1000.0
                phase_ms["views"] += (_t2 - _t1) * 1000.0
                phase_ms["audio"] += (_t3 - _t2) * 1000.0
                phase_ms["send"] += (_t4 - _t3) * 1000.0
                if i % 50 == 0:
                    _n = 50.0 if i else 1.0
                    print(
                        f"[body] step={i} pos=({snap.position[0]:.2f},{snap.position[1]:.2f},"
                        f"{snap.position[2]:.2f}) moving={snap.moving} events={len(obs['events'])}"
                    )
                    print(
                        "[body] budget ms/obs: "
                        + " ".join(f"{k}={v / _n:.1f}" for k, v in phase_ms.items()),
                        flush=True,
                    )
                    phase_ms = {k: 0.0 for k in phase_ms}
                await asyncio.sleep(interval)
                i += 1
            sim.close()
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass


class BaselineController:
    """Reactive, NON-learning controller for the distinctness baseline.

    Drives the body's actuators locally instead of the brain while the body still
    emits identical observations/telemetry - so the same scene and the same
    locomotion metrics (distance, fall-rate, gait regularity) apply for an
    apples-to-apples comparison against the learned policy. Outputs per-actuator
    PD targets in [-1, 1] (the same contract the brain's motor head emits). Modes:

    - ``random``: a slow random walk over the targets (pure motor babble).
    - ``cpg``: a fixed central pattern generator - antiphase left/right legs with
      counter-swinging arms - the classic hand-tuned open-loop gait.
    """

    # (joint substring -> (amplitude, phase offset in radians)). Phase is added to
    # the per-side base phase (right = wt, left = wt + pi) to alternate the legs.
    _CPG_PATTERN: tuple[tuple[str, float, float], ...] = (
        ("hip_y", 0.6, 0.0),
        ("knee", 0.5, 1.3),
        ("ankle_y", 0.25, 0.2),
        ("shoulder1", 0.4, math.pi),  # arms counter-swing the legs
        ("elbow", 0.2, math.pi),
    )
    _CPG_FREQ_HZ = 1.2

    def __init__(self, sim: "HumanoidSim", mode: str) -> None:
        self._np = sim._np
        self.mode = mode
        self.nu = int(sim.model.nu)
        self._rng = self._np.random.default_rng()
        self._targets = self._np.zeros(self.nu, dtype=self._np.float64)
        # Per-actuator (amplitude, total phase) for the CPG, derived from the
        # joint each actuator drives. Unclassified actuators rest at 0.
        self._amp = self._np.zeros(self.nu, dtype=self._np.float64)
        self._phase = self._np.zeros(self.nu, dtype=self._np.float64)
        mj = sim._mj
        classified = 0
        for a in range(self.nu):
            jid = int(sim.model.actuator_trnid[a, 0])
            name = (mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_JOINT, jid) or "").lower()
            side_phase = math.pi if name.startswith("left") else 0.0
            for sub, amp, ph in self._CPG_PATTERN:
                if sub in name:
                    self._amp[a] = amp
                    self._phase[a] = side_phase + ph
                    classified += 1
                    break
        # Fallback so the CPG always produces motion even if joint names differ:
        # alternate actuators in antiphase at a moderate amplitude.
        if classified == 0:
            for a in range(self.nu):
                self._amp[a] = 0.4
                self._phase[a] = math.pi * (a % 2)

    def targets(self, t_seconds: float) -> "list[float]":
        if self.mode == "random":
            self._targets += self._rng.normal(0.0, 0.12, self.nu)
            self._np.clip(self._targets, -1.0, 1.0, out=self._targets)
            return self._targets.tolist()
        # cpg
        theta = 2.0 * math.pi * self._CPG_FREQ_HZ * t_seconds
        vec = self._amp * self._np.sin(theta + self._phase)
        return self._np.clip(vec, -1.0, 1.0).tolist()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo humanoid body for Decadic.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ssl", action="store_true")
    # Default flipped 200 -> 0 (2026-07-04, owner): a body is not a test
    # fixture -- embodiment runs indefinitely unless explicitly bounded. The
    # old smoke-test default silently ended runs at step 200.
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 = run forever (default); N = bounded smoke/test run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Synthetic snapshots; no MuJoCo")
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Egocentric camera → base64 PNG (on by default; use --no-vision to disable)",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Procedural soundscape (footsteps/impacts/growls) → pcm16 base64",
    )
    parser.add_argument("--view", action="store_true", help="Open native MuJoCo viewer window")
    parser.add_argument(
        "--braces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Manual joint-brace orthosis engaged at spawn (off by default; "
            "use --braces to start scaffolded)"
        ),
    )
    parser.add_argument(
        "--scene",
        choices=sorted(LEGACY_SCENES),
        default="default",
        help="Legacy named scene (ignored when --scenario is given)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Comma-separated world elements (overrides --scene): "
            + ",".join(SELECTABLE_ELEMENTS)
        ),
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Bind to an existing agent instead of creating a fresh one",
    )
    parser.add_argument(
        "--baseline",
        choices=["random", "cpg"],
        default=None,
        help=(
            "Distinctness baseline: drive actuators locally with a reactive, "
            "non-learning controller (random babble or a fixed alternating-leg "
            "CPG) instead of the brain. Identical telemetry still streams, so the "
            "same locomotion metrics apply. Disables the joint braces (free body)."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    scheme_http = "https" if args.ssl else "http"
    scheme_ws = "wss" if args.ssl else "ws"
    base_http = f"{scheme_http}://{args.host}:{args.port}"
    aid = args.agent_id or _post_agent(base_http)
    print(f"[body] agent_id={aid}")
    ws_url = f"{scheme_ws}://{args.host}:{args.port}/agent/{aid}/cycle"

    elements: list[str] | None = None
    if args.scenario:
        elements = [e.strip().lower() for e in args.scenario.split(",") if e.strip()]

    try:
        asyncio.run(
            _run(
                ws_url,
                steps=args.steps,
                dry_run=args.dry_run,
                vision=args.vision,
                audio=args.audio,
                view=args.view,
                scene=args.scene,
                elements=elements,
                baseline=args.baseline,
                braces=args.braces,
            )
        )
    except KeyboardInterrupt:
        print("[body] stopped")


if __name__ == "__main__":
    main()
