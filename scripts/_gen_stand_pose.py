"""One-time dev validator for the joint-brace standing reference.

Empirically establishes (and re-checks) the constants baked into
scripts/mujoco_decadic_adapter.py for the joint-brace guidance system:

 - STAND_POSE      = zeros (the classic humanoid neutral pose is a stable,
                     COM-over-feet stand; a hand-tuned knee/hip flex was LESS
                     stable, so zeros is the reference q_ref / springref).
 - STAND_ROOT_Z    = 1.30 (soles rest ~0.007 m above the floor, settles ~1.295;
                     spawning at the old 1.40 dropped the feet ~0.11 m in the
                     air, so a no-lift body free-fell and toppled).
 - integrator      = implicitfast (the stable way to integrate stiff joint
                     springs; explicit stiff qfrc PD under RK4 diverges and
                     MuJoCo auto-resets, which masqueraded as "standing").
 - brace           = native per-joint stiffness/damping cranked up toward q_ref
                     (springref). With NO external lift the body stands on fully
                     loaded feet (load ~1.0) and barely drifts.

Run:  python scripts/_gen_stand_pose.py
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

ASSET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid_body.xml")
)

# Baked configuration (kept in sync with the adapter constants).
STAND_ROOT_Z = 1.30
BRACE_STIFFNESS = 1500.0
BRACE_DAMPING = 30.0


def hinges(model):
    qadr, dadr, jids, names = [], [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("prop_") or nm.startswith("npc_"):
            continue
        qadr.append(int(model.jnt_qposadr[j]))
        dadr.append(int(model.jnt_dofadr[j]))
        jids.append(j)
        names.append(nm)
    return qadr, dadr, jids, names


def sole_gap(model, data):
    zs = []
    for gname in ("left_foot", "right_foot"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        zs.append(float(data.geom_xpos[gid][2]) - model.geom_size[gid][2])
    return min(zs)


def stand(model, data, *, seconds, perturb=0.0):
    qadr, dadr, jids, names = hinges(model)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    rq = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")])
    weight = float(np.sum(model.body_mass)) * float(abs(model.opt.gravity[2]))
    fadr = [
        int(model.sensor_adr[s])
        for s in range(model.nsensor)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or "")
        in ("touch_left_foot", "touch_right_foot")
    ]
    for h, j in enumerate(jids):
        model.jnt_stiffness[j] = BRACE_STIFFNESS
        model.dof_damping[dadr[h]] = BRACE_DAMPING
        model.qpos_spring[qadr[h]] = 0.0

    mujoco.mj_resetData(model, data)
    for qa in qadr:
        data.qpos[qa] = 0.0
    data.qpos[rq : rq + 3] = (0.0, 0.0, STAND_ROOT_Z)
    if perturb:
        data.qpos[rq + 3 : rq + 7] = (np.cos(perturb / 2), 0.0, np.sin(perturb / 2), 0.0)
    else:
        data.qpos[rq + 3 : rq + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    g0 = sole_gap(model, data)
    n = int(seconds / model.opt.timestep)
    x0 = data.qpos[rq : rq + 2].copy()
    for _ in range(n):
        data.ctrl[:] = 0.0
        data.xfrc_applied[torso, :] = 0.0
        mujoco.mj_step(model, data)
    up = float(data.xmat[torso].reshape(3, 3)[2, 2])
    load = sum(float(data.sensordata[a]) / max(1e-6, weight) for a in fadr)
    ok = np.all(np.isfinite(data.qpos))
    return {
        "finite": bool(ok),
        "up": round(up, 3),
        "rootz": round(float(data.qpos[rq + 2]), 3),
        "drift": round(float(np.linalg.norm(data.qpos[rq : rq + 2] - x0)), 4),
        "load": round(load, 3),
        "gap0": round(g0, 4),
    }


def settle_stance(model, data, stance, *, seconds):
    """Settle a stance-library pose under full braces and report its stability.

    Mirrors the body adapter: resolve the stance's per-joint reference (deg->rad,
    range-clamped), crank every hinge to the braced spring toward that reference,
    spawn the free root at the stance's height/orientation, then let go (ctrl=0,
    NO external wrench) and report whether the braced body holds the pose.
    """
    from decadic.embodiment import stances as stance_lib

    qadr, dadr, jids, names = hinges(model)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    rq = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")])
    weight = float(np.sum(model.body_mass)) * float(abs(model.opt.gravity[2]))
    ranges = []
    for h, j in enumerate(jids):
        if bool(model.jnt_limited[j]):
            ranges.append((float(model.jnt_range[j][0]), float(model.jnt_range[j][1])))
        else:
            ranges.append((-np.inf, np.inf))
    qpos0 = [float(model.qpos0[a]) for a in qadr]
    q_ref = stance_lib.resolve(stance, names, qpos0, ranges)

    load_adr = {}
    for s in range(model.nsensor):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or ""
        if nm.startswith("touch_"):
            load_adr[nm[len("touch_"):]] = int(model.sensor_adr[s])

    for h, j in enumerate(jids):
        model.jnt_stiffness[j] = BRACE_STIFFNESS
        model.dof_damping[dadr[h]] = BRACE_DAMPING
        model.qpos_spring[qadr[h]] = q_ref[h]

    mujoco.mj_resetData(model, data)
    for h, a in enumerate(qadr):
        data.qpos[a] = q_ref[h]
    data.qpos[rq : rq + 3] = (0.0, 0.0, stance.root_z)
    data.qpos[rq + 3 : rq + 7] = stance.root_quat
    mujoco.mj_forward(model, data)
    n = int(seconds / model.opt.timestep)
    x0 = data.qpos[rq : rq + 2].copy()
    for _ in range(n):
        data.ctrl[:] = 0.0
        data.xfrc_applied[torso, :] = 0.0
        mujoco.mj_step(model, data)
    up = float(data.xmat[torso].reshape(3, 3)[2, 2])
    loads = {k: round(float(data.sensordata[a]) / max(1e-6, weight), 2) for k, a in load_adr.items()}
    return {
        "finite": bool(np.all(np.isfinite(data.qpos))),
        "rootz_target": round(float(stance.root_z), 3),
        "rootz_settled": round(float(data.qpos[rq + 2]), 3),
        "up": round(up, 3),
        "drift": round(float(np.linalg.norm(data.qpos[rq : rq + 2] - x0)), 3),
        "loads": {k: v for k, v in loads.items() if v > 0.02},
    }


def main():
    m = mujoco.MjModel.from_xml_path(ASSET)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m)
    _, _, _, names = hinges(m)
    print(f"nq={m.nq} nu={m.nu} hinges={len(names)} mass={float(np.sum(m.body_mass)):.1f}kg")
    print("hinge order:", names)
    print(f"STAND_POSE = zeros, STAND_ROOT_Z={STAND_ROOT_Z}, "
          f"BRACE_STIFFNESS={BRACE_STIFFNESS}, BRACE_DAMPING={BRACE_DAMPING}")
    print("stand 6s:        ", stand(m, d, seconds=6.0))
    print("stand 6s +tilt8d:", stand(m, d, seconds=6.0, perturb=0.15))

    from decadic.embodiment import stances as stance_lib

    print("\n--- stance library settle (6s, braced, no external wrench) ---")
    for name, stance in stance_lib.STANCES.items():
        print(f"{name:13s}", settle_stance(m, d, stance, seconds=6.0))


if __name__ == "__main__":
    main()
