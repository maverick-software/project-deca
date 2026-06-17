"""Stance-authoring probe: world-z of key geoms (and COM) at a stance's spawn pose.

Dev companion to ``scripts/_gen_stand_pose.py`` (which *settles* a stance to score
stability). This one shows, BEFORE settling, which body parts are lowest (the
would-be floor contacts) and where the COM sits relative to them -- the two facts
that decide whether a free-root braced pose rests or tips (COM must lie over the
support points). Use it to tune a new stance's joint angles / root_z / root_quat.

Usage:  python scripts/_probe_pose.py <stance_name>   # default: all_fours
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

from decadic.embodiment import stances as stance_lib

ASSET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid_body.xml"))
GEOMS = [
    "head", "butt", "torso1", "right_hand", "left_hand", "right_larm", "left_larm",
    "right_foot", "left_foot", "right_shin1", "left_shin1", "right_thigh1", "left_thigh1",
]


def hinges(model):
    qadr, jids, names = [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("prop_") or nm.startswith("npc"):
            continue
        qadr.append(int(model.jnt_qposadr[j]))
        jids.append(j)
        names.append(nm)
    return qadr, jids, names


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "all_fours"
    stance = stance_lib.get_stance(name)
    m = mujoco.MjModel.from_xml_path(ASSET)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m)
    qadr, jids, names = hinges(m)
    ranges = []
    for j in jids:
        if bool(m.jnt_limited[j]):
            ranges.append((float(m.jnt_range[j][0]), float(m.jnt_range[j][1])))
        else:
            ranges.append((-np.inf, np.inf))
    qpos0 = [float(m.qpos0[a]) for a in qadr]
    q_ref = stance_lib.resolve(stance, names, qpos0, ranges)
    rq = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")])
    mujoco.mj_resetData(m, d)
    for h, a in enumerate(qadr):
        d.qpos[a] = q_ref[h]
    d.qpos[rq : rq + 3] = (0.0, 0.0, stance.root_z)
    d.qpos[rq + 3 : rq + 7] = stance.root_quat
    mujoco.mj_forward(m, d)
    print(f"stance={name} root_z={stance.root_z} quat={stance.root_quat}")
    rows = []
    for g in GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if gid < 0:
            continue
        z = float(d.geom_xpos[gid][2])
        x = float(d.geom_xpos[gid][0])
        rows.append((z, x, g))
    rows.sort()
    com = float(d.subtree_com[0][2])
    comx = float(d.subtree_com[0][0])
    print(f"  COM x={comx:.3f} z={com:.3f}")
    for z, x, g in rows:
        print(f"  {g:14s} z={z:+.3f} x={x:+.3f}")


if __name__ == "__main__":
    main()
