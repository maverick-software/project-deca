"""Stance and motion library for the joint-brace guidance system.

A *stance* is a body posture the joint braces hold the agent in: a per-joint
reference angle (keyed by joint name, in DEGREES) plus the spawn root height and
orientation, and a fall-floor height appropriate to that posture. A *motion* is
the same, but its per-joint reference is a list of phase keyframes that the stiff
braces track over a cycle (looping for ``crawl``, one-shot for ``sit_to_stand``),
so locomotion emerges from genuine contact with the floor -- the root is never
forced, so the no-glide invariant holds in every stance.

This module is pure data plus tiny resolvers (deg->rad, keyframe interpolation),
free of MuJoCo so it is unit-testable and importable by BOTH the body adapter
(``scripts/mujoco_decadic_adapter.py``) and the API server
(``decadic/api/app.py``) as the single source of truth. Range clamping needs the
model's joint limits, so it is applied by the caller (the adapter) via the
optional ``ranges_rad`` argument; the data here stays model-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Quaternion (w, x, y, z) for an upright torso (no rotation).
UPRIGHT_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _pitch_quat(deg: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a rotation of ``deg`` about the world y-axis.

    A negative angle pitches the torso forward (head/shoulders down, toward +x),
    which is how the quadruped (all-fours / crawl) postures lay the spine over.
    """
    r = math.radians(deg) * 0.5
    return (math.cos(r), 0.0, math.sin(r), 0.0)


@dataclass(frozen=True)
class Stance:
    """A braced posture (static) or a braced trajectory (motion).

    ``joints`` maps a joint *name* to its braced reference angle in DEGREES; any
    hinge absent from the map falls back to the model's neutral pose (qpos0).
    ``root_z`` / ``root_quat`` set the spawn pose of the free root. ``fall_z`` is
    the root height below which this posture counts as "fallen" (low for
    quadruped/kneel stances so they are not perpetually flagged as a fall).

    A motion additionally carries ``keyframes`` -- ``(phase, {joint: deg})`` pairs
    with ``phase`` in [0, 1] -- interpolated over ``period_s`` seconds. ``loop``
    wraps the phase (a repeating gait); otherwise the final keyframe is held once
    reached (a one-shot transition that then settles into a posture).
    """

    name: str
    label: str
    description: str = ""
    joints: dict[str, float] = field(default_factory=dict)
    root_z: float = 1.30
    root_quat: tuple[float, float, float, float] = UPRIGHT_QUAT
    fall_z: float = 0.7
    keyframes: tuple[tuple[float, dict[str, float]], ...] = ()
    period_s: float = 0.0
    loop: bool = False

    @property
    def is_motion(self) -> bool:
        return bool(self.keyframes) and self.period_s > 0.0


# ---------------------------------------------------------------------------
# The registry. Angles are degrees; joints omitted from a pose hold qpos0.
# Static poses (stand / all_fours / kneel_*) are validated for stability under
# the braces by scripts/_gen_stand_pose.py. Motions reuse a static base pose and
# layer a cyclic/transition trajectory on top of it.
# ---------------------------------------------------------------------------

# Quadruped base: torso pitched +90 deg (belly faces the floor, head forward),
# hips flexed ~90 so the thighs point down to the knees, shins folded back, arms
# hanging straight to the hands. With the +90 pitch, body-forward maps to
# world-down, so the (forward-hanging) arms and forward-flexed thighs reach the
# floor -- the body rests on hands + knees.
_ALL_FOURS_JOINTS: dict[str, float] = {
    "right_hip_y": -90.0,
    "left_hip_y": -90.0,
    "right_knee": -90.0,
    "left_knee": -90.0,
    "right_shoulder1": 0.0,
    "right_elbow": -35.0,
    "left_shoulder1": 0.0,
    "left_elbow": -35.0,
}

# Knees-down quadruped kneel: both legs folded deep so the shins/knees (not the
# feet) bear the rear load, with both hands down and the torso twisted toward one
# side -- a stable symmetric four-point base distinguished left/right by the
# twist. (A free-root statue cannot hold a single-knee upright kneel: COM-over-
# support forces a quadruped base, and unequal leg lengths roll it over, so the
# stable distinction is the twist direction rather than which single knee is down.)
_KNEEL_BASE: dict[str, float] = {
    "right_hip_y": -90.0,
    "left_hip_y": -90.0,
    "right_knee": -135.0,
    "left_knee": -135.0,
    "right_shoulder1": 0.0,
    "left_shoulder1": 0.0,
    "right_elbow": -35.0,
    "left_elbow": -35.0,
}
_KNEEL_LEFT_JOINTS: dict[str, float] = {**_KNEEL_BASE, "abdomen_z": 25.0}
_KNEEL_RIGHT_JOINTS: dict[str, float] = {**_KNEEL_BASE, "abdomen_z": -25.0}

# Upright kneel: torso vertical, hips/knees folded under the body. This is the
# correct intermediate for floor-to-stand practice: the agent is sitting up on
# its knees, not pitched forward on hands and knees.
_UPRIGHT_KNEEL_JOINTS: dict[str, float] = {
    "right_hip_y": -35.0,
    "left_hip_y": -35.0,
    "right_knee": -125.0,
    "left_knee": -125.0,
    "right_ankle_y": 30.0,
    "left_ankle_y": 30.0,
    "right_shoulder1": -10.0,
    "left_shoulder1": 10.0,
    "right_elbow": -10.0,
    "left_elbow": -10.0,
}

STANCES: dict[str, Stance] = {
    "stand": Stance(
        name="stand",
        label="Stand & balance",
        description="Upright neutral pose (the validated COM-over-feet stand).",
        joints={},
        root_z=1.30,
        root_quat=UPRIGHT_QUAT,
        fall_z=0.7,
    ),
    "all_fours": Stance(
        name="all_fours",
        label="Kneel on all fours",
        description="Quadruped base: hands and knees on the floor, spine pitched over.",
        joints=_ALL_FOURS_JOINTS,
        root_z=0.47,
        root_quat=_pitch_quat(90.0),
        fall_z=0.25,
    ),
    "kneel_left": Stance(
        name="kneel_left",
        label="Kneel (twist left)",
        description="Knees-down quadruped kneel with the torso twisted left (stable four-point).",
        joints=_KNEEL_LEFT_JOINTS,
        root_z=0.55,
        root_quat=_pitch_quat(90.0),
        fall_z=0.25,
    ),
    "kneel_right": Stance(
        name="kneel_right",
        label="Kneel (twist right)",
        description="Knees-down quadruped kneel with the torso twisted right (stable four-point).",
        joints=_KNEEL_RIGHT_JOINTS,
        root_z=0.55,
        root_quat=_pitch_quat(90.0),
        fall_z=0.25,
    ),
    "kneel_upright": Stance(
        name="kneel_upright",
        label="Sit upright on knees",
        description="Torso upright with both knees folded under the body; floor-to-stand intermediate.",
        joints=_UPRIGHT_KNEEL_JOINTS,
        root_z=0.82,
        root_quat=UPRIGHT_QUAT,
        fall_z=0.35,
    ),
    "crawl": Stance(
        name="crawl",
        label="Crawl (motion)",
        description="Looping diagonal quadruped gait on hands and knees.",
        joints=_ALL_FOURS_JOINTS,
        root_z=0.47,
        root_quat=_pitch_quat(90.0),
        fall_z=0.25,
        period_s=2.4,
        loop=True,
        keyframes=(
            (0.0, {}),
            (
                0.25,
                {
                    "right_shoulder1": -25.0,
                    "right_elbow": -30.0,
                    "left_hip_y": -45.0,
                    "left_knee": -70.0,
                },
            ),
            (0.5, {}),
            (
                0.75,
                {
                    "left_shoulder1": 25.0,
                    "left_elbow": -30.0,
                    "right_hip_y": -45.0,
                    "right_knee": -70.0,
                },
            ),
        ),
    ),
    "sit_to_stand": Stance(
        name="sit_to_stand",
        label="Rise up (motion)",
        description="One-shot push up from all-fours: hips and knees extend, raising the body.",
        joints=_ALL_FOURS_JOINTS,
        root_z=0.47,
        root_quat=_pitch_quat(90.0),
        fall_z=0.25,
        period_s=3.0,
        loop=False,
        keyframes=(
            (0.0, dict(_ALL_FOURS_JOINTS)),
            (
                1.0,
                {
                    "right_hip_y": -15.0,
                    "left_hip_y": -15.0,
                    "right_knee": -15.0,
                    "left_knee": -15.0,
                    "right_elbow": 0.0,
                    "left_elbow": 0.0,
                },
            ),
        ),
    ),
    "kneel_to_stand": Stance(
        name="kneel_to_stand",
        label="Kneel to stand (motion)",
        description="One-shot rise from upright kneeling: tuck feet under, extend hips/knees, and settle into stand.",
        joints=_UPRIGHT_KNEEL_JOINTS,
        root_z=0.82,
        root_quat=UPRIGHT_QUAT,
        fall_z=0.35,
        period_s=4.0,
        loop=False,
        keyframes=(
            (0.0, dict(_UPRIGHT_KNEEL_JOINTS)),
            (
                0.35,
                {
                    "right_hip_y": -70.0,
                    "left_hip_y": -70.0,
                    "right_knee": -115.0,
                    "left_knee": -115.0,
                    "right_ankle_y": 40.0,
                    "left_ankle_y": 40.0,
                    "right_shoulder1": -5.0,
                    "left_shoulder1": 5.0,
                    "right_elbow": -5.0,
                    "left_elbow": -5.0,
                },
            ),
            (
                0.7,
                {
                    "right_hip_y": -35.0,
                    "left_hip_y": -35.0,
                    "right_knee": -55.0,
                    "left_knee": -55.0,
                    "right_ankle_y": 12.0,
                    "left_ankle_y": 12.0,
                    "right_shoulder1": 0.0,
                    "left_shoulder1": 0.0,
                    "right_elbow": 0.0,
                    "left_elbow": 0.0,
                },
            ),
            (
                1.0,
                {
                    "right_hip_y": -5.0,
                    "left_hip_y": -5.0,
                    "right_knee": -5.0,
                    "left_knee": -5.0,
                    "right_ankle_y": 0.0,
                    "left_ankle_y": 0.0,
                    "right_shoulder1": 0.0,
                    "left_shoulder1": 0.0,
                    "right_elbow": 0.0,
                    "left_elbow": 0.0,
                },
            ),
        ),
    ),
}

DEFAULT_STANCE = "stand"


def get_stance(name: str | None) -> Stance:
    """Return the stance for ``name``, falling back to the default stand."""
    if name is None:
        return STANCES[DEFAULT_STANCE]
    return STANCES.get(str(name), STANCES[DEFAULT_STANCE])


def catalog() -> list[dict[str, object]]:
    """Serializable list of stances for the UI (name, label, description, motion)."""
    out: list[dict[str, object]] = []
    for s in STANCES.values():
        out.append(
            {
                "name": s.name,
                "label": s.label,
                "description": s.description,
                "motion": s.is_motion,
            }
        )
    return out


def _apply_joint_overrides(
    base_rad: list[float],
    hinge_names: list[str],
    joints_deg: dict[str, float],
) -> tuple[list[float], set[int]]:
    """Overlay degree-valued joint overrides onto a radian base pose.

    Returns the new pose and the set of hinge indices actually overridden, so the
    caller clamps ONLY commanded joints -- untouched hinges keep the model's
    neutral qpos0 verbatim (so ``stand`` reproduces the validated zero pose).
    """
    q = list(base_rad)
    touched: set[int] = set()
    if not joints_deg:
        return q, touched
    index = {nm: i for i, nm in enumerate(hinge_names)}
    for nm, deg in joints_deg.items():
        i = index.get(nm)
        if i is not None:
            q[i] = math.radians(float(deg))
            touched.add(i)
    return q, touched


def _clamp(
    q: list[float],
    touched: set[int],
    ranges_rad: list[tuple[float, float]] | None,
) -> list[float]:
    if ranges_rad is None or not touched:
        return q
    out = list(q)
    for i in touched:
        if i < len(ranges_rad):
            lo, hi = ranges_rad[i]
            if hi >= lo:
                out[i] = min(hi, max(lo, out[i]))
    return out


def resolve(
    stance: Stance,
    hinge_names: list[str],
    defaults_rad: list[float],
    ranges_rad: list[tuple[float, float]] | None = None,
) -> list[float]:
    """Per-hinge braced reference (radians) for a stance's static pose.

    Hinges absent from the stance hold ``defaults_rad`` (the model's qpos0), so
    ``stand`` (no overrides) reproduces the validated neutral stand exactly.
    ``ranges_rad`` (lo, hi per hinge) clamps authored angles into joint limits.
    """
    q, touched = _apply_joint_overrides(defaults_rad, hinge_names, stance.joints)
    return _clamp(q, touched, ranges_rad)


def motion_ref(
    stance: Stance,
    phase: float,
    hinge_names: list[str],
    defaults_rad: list[float],
    ranges_rad: list[tuple[float, float]] | None = None,
) -> list[float]:
    """Per-hinge braced reference (radians) for a motion stance at ``phase`` (0..1).

    The motion layers an interpolated keyframe trajectory on top of the stance's
    static base pose: each keyframe only specifies the joints it moves, so any
    joint it omits holds the base pose. ``loop`` wraps the phase into a repeating
    gait; otherwise it clamps to [0, 1] so the final keyframe is held one-shot.
    """
    if not stance.is_motion:
        return resolve(stance, hinge_names, defaults_rad, ranges_rad)
    base, base_touched = _apply_joint_overrides(defaults_rad, hinge_names, stance.joints)
    p = (phase % 1.0) if stance.loop else max(0.0, min(1.0, phase))
    kfs = stance.keyframes
    # Find the bracketing keyframes around p.
    lo_p, lo_j = kfs[0]
    hi_p, hi_j = kfs[-1]
    for i in range(len(kfs) - 1):
        if kfs[i][0] <= p <= kfs[i + 1][0]:
            lo_p, lo_j = kfs[i]
            hi_p, hi_j = kfs[i + 1]
            break
    else:
        # p is past the last keyframe: loop wraps back to the first (phase 1 ==
        # phase 0); a one-shot motion simply holds its final keyframe.
        lo_p, lo_j = kfs[-1]
        if stance.loop:
            hi_p, hi_j = 1.0, kfs[0][1]
        else:
            hi_p, hi_j = lo_p, lo_j
    span = hi_p - lo_p
    frac = 0.0 if span <= 1e-9 else (p - lo_p) / span
    lo_q, lo_t = _apply_joint_overrides(base, hinge_names, lo_j)
    hi_q, hi_t = _apply_joint_overrides(base, hinge_names, hi_j)
    q = [lo_q[i] + (hi_q[i] - lo_q[i]) * frac for i in range(len(base))]
    return _clamp(q, base_touched | lo_t | hi_t, ranges_rad)
