"""Pure kinematic pose / behavior library for the scripted NPC crowd.

These functions return joint-angle dictionaries (radians) and scalar blends for
the crowd's behaviors. They are intentionally MuJoCo-free so they can be unit
tested directly; :mod:`decadic.embodiment.npc_controller` writes the returned
poses into ``qpos``/``qvel``.

The animated joint keys match the controller's discovered joints:
``r_hip, l_hip, r_knee, l_knee, r_sh, l_sh``.

The walk cycle mirrors the parent gait in the adapter so a crowd walker looks
identical to the legacy parent: legs swing fore-aft in antiphase, the swing
leg's knee flexes (knees only bend one way -> always negative), and the arms
counter-swing the legs.
"""

from __future__ import annotations

import math

# Gait/pose constants (mirror the adapter's NPC_* parent-gait tuning).
HIP_SWING = 0.5  # fore-aft thigh swing amplitude (rad)
KNEE_BASE = 0.1  # baseline knee bend so legs are never hyperextended (rad)
KNEE_BEND = 0.9  # extra knee flexion on the swing leg (rad)
ARM_SWING = 0.45  # shoulder counter-swing amplitude (rad)
BOB_AMP = 0.03  # vertical torso bob amplitude (m)
STRIDE_LENGTH = 0.8  # metres of travel per full gait cycle (foot-plant tuning)
WALK_SPEED = 0.85  # forward ground speed while walking (m/s)
TURN_RATE = 2.5  # max yaw turn toward the target (rad/s)

# Seated pose: deep hip + knee flexion, root dropped toward the ground.
SIT_ROOT_DROP = 0.5  # metres the root lowers when fully seated
SIT_HIP = -1.3  # hip flexion when seated (rad)
SIT_KNEE = -1.5  # knee flexion when seated (rad)
SIT_ARM = 0.2  # slight arm forward rest when seated (rad)

# Conversation gesture: one arm oscillates while the other rests.
COMM_GESTURE_HZ = 0.8  # gesture oscillation frequency
COMM_GESTURE_AMP = 0.5  # gesture shoulder amplitude (rad)

_KEYS = ("r_hip", "l_hip", "r_knee", "l_knee", "r_sh", "l_sh")


def _knee(p: float) -> float:
    """Swing-leg knee flexion (always negative; knees bend one way only)."""
    return -(KNEE_BASE + KNEE_BEND * max(0.0, math.sin(p)))


def walk_pose(phase: float) -> dict[str, float]:
    """Walk-cycle joint angles for a given stride ``phase`` (rad)."""
    r, lft = phase, phase + math.pi
    return {
        "r_hip": HIP_SWING * math.sin(r),
        "l_hip": HIP_SWING * math.sin(lft),
        "r_knee": _knee(r + math.pi / 2.0),
        "l_knee": _knee(lft + math.pi / 2.0),
        "r_sh": -ARM_SWING * math.sin(r),
        "l_sh": -ARM_SWING * math.sin(lft),
    }


def stand_pose() -> dict[str, float]:
    """Neutral standing pose (a touch of knee bend so legs aren't locked)."""
    return {
        "r_hip": 0.0,
        "l_hip": 0.0,
        "r_knee": -KNEE_BASE,
        "l_knee": -KNEE_BASE,
        "r_sh": 0.0,
        "l_sh": 0.0,
    }


def sit_pose() -> dict[str, float]:
    """Seated pose: hips and knees deeply flexed, arms resting forward."""
    return {
        "r_hip": SIT_HIP,
        "l_hip": SIT_HIP,
        "r_knee": SIT_KNEE,
        "l_knee": SIT_KNEE,
        "r_sh": SIT_ARM,
        "l_sh": SIT_ARM,
    }


def communicate_pose(t: float) -> dict[str, float]:
    """Standing pose with one arm gesturing (a conversation)."""
    pose = stand_pose()
    pose["r_sh"] = COMM_GESTURE_AMP * math.sin(2.0 * math.pi * COMM_GESTURE_HZ * t)
    return pose


def lerp_pose(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    """Linearly blend two poses by ``t`` in [0, 1]."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return {k: (1.0 - t) * a.get(k, 0.0) + t * b.get(k, 0.0) for k in _KEYS}


def sit_stand_blend(t: float, period: float = 6.0) -> float:
    """Triangular sit<->stand blend in [0, 1] (0 standing, 1 fully seated)."""
    if period <= 0.0:
        return 0.0
    frac = (t % period) / period
    return 2.0 * frac if frac < 0.5 else 2.0 * (1.0 - frac)
