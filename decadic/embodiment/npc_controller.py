"""MuJoCo-bound controller for the scripted NPC crowd.

Owns a list of :class:`NPCRuntime` structs (one per habitat) and, each physics
substep, writes their root pose + joint angles directly (the bodies have no
actuators). Per-zone behaviors are dispatched from the habitat descriptor; one
NPC is the *parent* that forages and provisions the learner on a need threshold
(not a timer). Distant/behind NPCs are held static (a cheap level-of-detail).

The controller borrows the host :class:`HumanoidSim`'s shared consumable
machinery (``food_bodies``/``water_bodies``/``eaten``/``_consume``/``_respawn``)
so habitat resources are net-additive and respawn exactly like the learner's.
NPC consumption emits ``npc_eat``/``npc_drink`` (ignored by ``classify_events``),
so a demonstrator eating never credits the learner; only the learner eating a
dropped gift does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from decadic.config import DEFAULT_STAND_ROOT_Z as STAND_ROOT_HEIGHT
from decadic.embodiment import npc_behaviors as B
from decadic.embodiment.habitats import (
    BEHAVIOR_COMMUNICATE,
    BEHAVIOR_FORAGE,
    BEHAVIOR_IDLE,
    BEHAVIOR_SIT,
    BEHAVIOR_SIT_STAND,
    BEHAVIOR_WANDER,
    Habitat,
    active_habitats,
    clamp_to_zone,
    crowd_lod_distance,
    parent_effective_threshold,
    parent_refractory_s,
)
from decadic.embodiment.npc_xml import GIFT_NAMES

EAT_RADIUS = 1.0  # mirror of the adapter's consumption reach (m)
DRINK_RADIUS = 1.0
PICKUP_RADIUS = 1.0  # parent reaches a source within this distance to carry it
DROP_REACH = 0.6  # parent has arrived at its drop point within this distance
WAYPOINT_DWELL_S = 3.0  # wander: seconds before choosing a fresh in-zone waypoint
WALK_STANDOFF = 0.3  # stop closing on a target within this distance
LOD_FRUSTUM_COS = math.cos(math.radians(80.0))  # cull NPCs > 80deg off the learner's heading
LOD_NEAR = 6.0  # never cull NPCs closer than this, even if behind


@dataclass
class NPCRuntime:
    """Per-NPC kinematic + behavior state."""

    idx: int
    entity_id: str
    habitat: Habitat
    is_parent: bool
    torso_body: int
    root_qadr: int
    root_dadr: int
    hinges: list[tuple[int, int, float]]
    anim: dict[str, tuple[int, int, float]]
    hand_body: int | None
    x: float
    y: float
    yaw: float
    gait_phase: float = 0.0
    anim_t: float = 0.0
    wp: tuple[float, float] | None = None
    wp_timer: float = 0.0
    phase: str = "seek_food"  # forage/parent FSM state
    item: str = "food"  # gift the parent currently fetches/carries
    carry: bool = False
    next_deliver: float = 0.0
    offers: int = 0
    rng: Any = None


class CrowdController:
    """Animate and run the per-zone behavior FSMs for the scripted crowd."""

    def __init__(self, sim: Any) -> None:
        self._sim = sim
        self._mj = sim._mj
        self._np = sim._np
        self.model = sim.model
        self.data = sim.data
        self.frozen = False
        self.reservoirs: dict[str, float] | None = None
        self.habitats = active_habitats()
        self.npcs: list[NPCRuntime] = []
        self._gift_addr: dict[str, tuple[int, int]] = {"food": (-1, -1), "water": (-1, -1)}
        self._discover()

    # -- discovery ---------------------------------------------------------
    def _jadr(self, jname: str) -> tuple[int, int, float] | None:
        mj = self._mj
        jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            return None
        qa = int(self.model.jnt_qposadr[jid])
        return (qa, int(self.model.jnt_dofadr[jid]), float(self.model.qpos0[qa]))

    def _discover(self) -> None:
        mj = self._mj
        for hi, hab in enumerate(self.habitats):
            prefix = f"npc{hi}_"
            tid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"{prefix}torso")
            if tid < 0:
                continue
            root = self._jadr(f"{prefix}root")
            if root is None:
                continue
            hinges: list[tuple[int, int, float]] = []
            for j in range(self.model.njnt):
                nm = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j) or ""
                if nm.startswith(prefix) and self.model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
                    qa = int(self.model.jnt_qposadr[j])
                    hinges.append((qa, int(self.model.jnt_dofadr[j]), float(self.model.qpos0[qa])))
            anim: dict[str, tuple[int, int, float]] = {}
            for key, jn in (
                ("r_hip", "right_hip_y"), ("l_hip", "left_hip_y"),
                ("r_knee", "right_knee"), ("l_knee", "left_knee"),
                ("r_sh", "right_shoulder1"), ("l_sh", "left_shoulder1"),
            ):
                adr = self._jadr(prefix + jn)
                if adr is not None:
                    anim[key] = adr
            hb = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"{prefix}right_lower_arm")
            cx, cy = hab.center
            yaw = math.atan2(-cy, -cx) if not hab.face else math.atan2(hab.face[1] - cy, hab.face[0] - cx)
            self.npcs.append(
                NPCRuntime(
                    idx=hi, entity_id=f"npc{hi}", habitat=hab, is_parent=hab.is_parent,
                    torso_body=int(tid), root_qadr=root[0], root_dadr=root[1],
                    hinges=hinges, anim=anim, hand_body=hb if hb >= 0 else None,
                    x=float(cx), y=float(cy), yaw=float(yaw),
                    next_deliver=parent_refractory_s(), rng=random.Random(1000 + hi),
                )
            )
        for item, gname in GIFT_NAMES.items():
            gj = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, f"{gname}_free")
            if gj >= 0:
                self._gift_addr[item] = (
                    int(self.model.jnt_qposadr[gj]), int(self.model.jnt_dofadr[gj])
                )

    # -- public controls ---------------------------------------------------
    def set_frozen(self, value: bool) -> None:
        self.frozen = bool(value)

    def set_reservoirs(self, reservoirs: dict[str, float] | None) -> None:
        self.reservoirs = reservoirs

    def entities(self) -> list[dict[str, Any]]:
        out = []
        for npc in self.npcs:
            p = self.data.xpos[npc.torso_body]
            out.append(
                {"id": npc.entity_id, "kind": "npc",
                 "position": [float(p[0]), float(p[1]), float(p[2])]}
            )
        return out

    def recenter(self) -> None:
        for npc in self.npcs:
            cx, cy = npc.habitat.center
            npc.x, npc.y = float(cx), float(cy)
            npc.yaw = math.atan2(-cy, -cx)
            npc.gait_phase = 0.0
            npc.anim_t = 0.0
            npc.phase = "seek_food"
            npc.item = "food"
            npc.carry = False
            self._hold(npc)

    # -- per-substep animation --------------------------------------------
    def apply(self) -> None:
        if not self.npcs:
            return
        dt = float(self.model.opt.timestep)
        t = self._sim.data.xpos[self._sim.torso_id]
        ax, ay = float(t[0]), float(t[1])
        xmat = self.data.xmat[self._sim.torso_id].reshape(3, 3)
        ayaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
        for npc in self.npcs:
            if self.frozen or self._culled(npc, ax, ay, ayaw):
                self._hold(npc)
                continue
            self._animate(npc, dt)

    def _culled(self, npc: NPCRuntime, ax: float, ay: float, ayaw: float) -> bool:
        dx, dy = npc.x - ax, npc.y - ay
        dist = math.hypot(dx, dy)
        if dist > crowd_lod_distance():
            return True
        if dist <= LOD_NEAR or dist == 0.0:
            return False
        cos_off = (math.cos(ayaw) * dx + math.sin(ayaw) * dy) / dist
        return cos_off < LOD_FRUSTUM_COS

    def _animate(self, npc: NPCRuntime, dt: float) -> None:
        npc.anim_t += dt
        beh = npc.habitat.behavior
        if beh == BEHAVIOR_SIT:
            self._write(npc, STAND_ROOT_HEIGHT - B.SIT_ROOT_DROP, B.sit_pose(), 0.0, 0.0)
        elif beh == BEHAVIOR_SIT_STAND:
            blend = B.sit_stand_blend(npc.anim_t)
            pose = B.lerp_pose(B.stand_pose(), B.sit_pose(), blend)
            self._write(npc, STAND_ROOT_HEIGHT - B.SIT_ROOT_DROP * blend, pose, 0.0, 0.0)
        elif beh == BEHAVIOR_IDLE:
            self._write(npc, STAND_ROOT_HEIGHT, B.stand_pose(), 0.0, 0.0)
        elif beh == BEHAVIOR_COMMUNICATE:
            if npc.habitat.face is not None:
                fx, fy = npc.habitat.face
                npc.yaw = math.atan2(fy - npc.y, fx - npc.x)
            self._write(npc, STAND_ROOT_HEIGHT, B.communicate_pose(npc.anim_t), 0.0, 0.0)
        else:  # forage (incl. parent) and wander -> locomotion
            target = self._target_for(npc)
            moved = self._walk_toward(npc, target, dt)
            pose = B.walk_pose(npc.gait_phase) if moved > 0.0 else B.stand_pose()
            bob = B.BOB_AMP * math.sin(2.0 * npc.gait_phase)
            vx = math.cos(npc.yaw) * B.WALK_SPEED if moved > 0.0 else 0.0
            vy = math.sin(npc.yaw) * B.WALK_SPEED if moved > 0.0 else 0.0
            self._write(npc, STAND_ROOT_HEIGHT + bob, pose, vx, vy)
        self._carry_gift(npc)

    def _walk_toward(self, npc: NPCRuntime, target: tuple[float, float] | None, dt: float) -> float:
        moved = 0.0
        if target is not None:
            dx, dy = target[0] - npc.x, target[1] - npc.y
            dist = math.hypot(dx, dy)
            desired = math.atan2(dy, dx)
            err = math.atan2(math.sin(desired - npc.yaw), math.cos(desired - npc.yaw))
            npc.yaw += max(-B.TURN_RATE * dt, min(B.TURN_RATE * dt, err))
            if dist > WALK_STANDOFF:
                moved = min(B.WALK_SPEED * dt, dist - WALK_STANDOFF)
                npc.x += math.cos(npc.yaw) * moved
                npc.y += math.sin(npc.yaw) * moved
        npc.x, npc.y = clamp_to_zone(npc.x, npc.y, npc.habitat.center, npc.habitat.radius)
        npc.gait_phase += 2.0 * math.pi * moved / B.STRIDE_LENGTH
        return moved

    def _write(
        self, npc: NPCRuntime, root_z: float, pose: dict[str, float], vx: float, vy: float
    ) -> None:
        qa, da = npc.root_qadr, npc.root_dadr
        self.data.qpos[qa : qa + 3] = (npc.x, npc.y, root_z)
        half = npc.yaw * 0.5
        self.data.qpos[qa + 3 : qa + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        self.data.qvel[da : da + 6] = (vx, vy, 0.0, 0.0, 0.0, 0.0)
        for qadr, vadr, q0 in npc.hinges:
            self.data.qpos[qadr] = q0
            self.data.qvel[vadr] = 0.0
        for key, (qadr, vadr, _q0) in npc.anim.items():
            self.data.qpos[qadr] = pose.get(key, 0.0)
            self.data.qvel[vadr] = 0.0

    def _hold(self, npc: NPCRuntime) -> None:
        """Cheap static pose hold (frozen or LOD-culled): no behavior compute."""
        self._write(npc, STAND_ROOT_HEIGHT, B.stand_pose(), 0.0, 0.0)
        self._carry_gift(npc)

    def _carry_gift(self, npc: NPCRuntime) -> None:
        if not (npc.is_parent and npc.carry and npc.hand_body is not None):
            return
        gq, gd = self._gift_addr.get(npc.item, (-1, -1))
        gname = GIFT_NAMES.get(npc.item, "")
        if gq < 0 or gname in self._sim.eaten:
            return
        hp = self.data.xpos[npc.hand_body]
        self.data.qpos[gq : gq + 3] = (float(hp[0]), float(hp[1]), float(hp[2]))
        self.data.qpos[gq + 3 : gq + 7] = (1.0, 0.0, 0.0, 0.0)
        if gd >= 0:
            self.data.qvel[gd : gd + 6] = 0.0

    # -- targets -----------------------------------------------------------
    def _target_for(self, npc: NPCRuntime) -> tuple[float, float] | None:
        if npc.habitat.behavior == BEHAVIOR_WANDER:
            if npc.wp is None or npc.anim_t >= npc.wp_timer:
                cx, cy = npc.habitat.center
                ang = npc.rng.uniform(0.0, 2.0 * math.pi)
                rad = npc.rng.uniform(0.0, npc.habitat.radius)
                npc.wp = (cx + rad * math.cos(ang), cy + rad * math.sin(ang))
                npc.wp_timer = npc.anim_t + WAYPOINT_DWELL_S
            return npc.wp
        if npc.is_parent:
            if npc.phase == "pickup":
                xy = self._nearest_in_zone(npc, self._bodies(npc.item))
                return xy if xy is not None else self._drop_point(npc)
            if npc.phase == "deliver":
                return self._drop_point(npc)
        # forage seek: head for the nearest live in-zone consumable.
        seek_water = npc.phase == "seek_water"
        xy = self._nearest_in_zone(npc, self._bodies("water" if seek_water else "food"))
        if xy is not None:
            return xy
        return self._nearest_in_zone(npc, self._bodies("food" if seek_water else "water"))

    def _bodies(self, item: str) -> dict[str, int]:
        return self._sim.water_bodies if item == "water" else self._sim.food_bodies

    def _nearest_in_zone(self, npc: NPCRuntime, bodies: dict[str, int]) -> tuple[float, float] | None:
        cx, cy = npc.habitat.center
        best: tuple[float, float] | None = None
        best_d = float("inf")
        for name, b in bodies.items():
            if name in self._sim.eaten or "gift" in name:
                continue
            p = self.data.xpos[b]
            px, py = float(p[0]), float(p[1])
            if math.hypot(px - cx, py - cy) > npc.habitat.radius + 0.5:
                continue
            d = math.hypot(px - npc.x, py - npc.y)
            if d < best_d:
                best_d, best = d, (px, py)
        return best

    def _drop_point(self, npc: NPCRuntime) -> tuple[float, float]:
        t = self.data.xpos[self._sim.torso_id]
        cx, cy = npc.habitat.center
        dx, dy = float(t[0]) - cx, float(t[1]) - cy
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            return (cx, cy + npc.habitat.radius)
        return (cx + dx / dist * npc.habitat.radius, cy + dy / dist * npc.habitat.radius)

    # -- events (consumption + parental provisioning) ----------------------
    def events(self, step: int, now: float) -> list[dict[str, Any]]:
        if self.frozen or not self.npcs:
            return []
        events: list[dict[str, Any]] = []
        for npc in self.npcs:
            beh = npc.habitat.behavior
            if beh != BEHAVIOR_FORAGE:
                continue
            self._forage_events(npc, now, events)
        return events

    def _forage_events(self, npc: NPCRuntime, now: float, events: list[dict[str, Any]]) -> None:
        npos = self.data.xpos[npc.torso_body]
        nx, ny = float(npos[0]), float(npos[1])
        phase = npc.phase
        if phase in ("seek_food", "seek_water"):
            kind = "water" if phase == "seek_water" else "food"
            hit = self._consume_if_reached(npc, nx, ny, kind)
            if hit:
                events.append(
                    {"type": "npc_drink" if kind == "water" else "npc_eat",
                     "intensity": 1.0, "source": npc.entity_id}
                )
                npc.phase = "seek_food" if kind == "water" else "seek_water"
            if npc.is_parent and self._delivery_due(now, npc):
                npc.carry = False
                npc.phase = "pickup"
        elif phase == "pickup":
            src = self._nearest_in_zone(npc, self._bodies(npc.item))
            if src is None or math.hypot(nx - src[0], ny - src[1]) <= PICKUP_RADIUS:
                self._pick_up(npc)
        elif phase == "deliver":
            dx, dy = self._drop_point(npc)
            if math.hypot(nx - dx, ny - dy) <= DROP_REACH:
                self._drop_gift(npc, dx, dy)
                npc.carry = False
                npc.offers += 1
                npc.next_deliver = now + parent_refractory_s()
                npc.item = "water" if npc.item == "food" else "food"
                npc.phase = "seek_food"
                events.append(
                    {"type": "offer", "item": npc.item, "intensity": 1.0, "source": npc.entity_id}
                )

    def _consume_if_reached(self, npc: NPCRuntime, nx: float, ny: float, kind: str) -> bool:
        radius = DRINK_RADIUS if kind == "water" else EAT_RADIUS
        for name, b in self._bodies(kind).items():
            if name in self._sim.eaten or "gift" in name:
                continue
            cx, cy = npc.habitat.center
            p = self.data.xpos[b]
            px, py = float(p[0]), float(p[1])
            if math.hypot(px - cx, py - cy) > npc.habitat.radius + 0.5:
                continue
            if math.hypot(px - nx, py - ny) <= radius:
                self._sim._consume(name)
                return True
        return False

    def _delivery_due(self, now: float, npc: NPCRuntime) -> bool:
        if now < npc.next_deliver:
            return False
        res = self.reservoirs
        if not res:
            return True
        thr = parent_effective_threshold(npc.offers)
        return min(res.values()) <= thr

    def _pick_up(self, npc: NPCRuntime) -> None:
        gname = GIFT_NAMES.get(npc.item, "")
        if gname in self._sim.eaten:
            self._sim._respawn(gname)
            self._sim._respawn_at.pop(gname, None)
        npc.carry = True
        npc.phase = "deliver"

    def _drop_gift(self, npc: NPCRuntime, x: float, y: float) -> None:
        gq, _gd = self._gift_addr.get(npc.item, (-1, -1))
        if gq < 0:
            return
        gname = GIFT_NAMES.get(npc.item, "")
        if gname in self._sim.eaten:
            self._sim._respawn(gname)
            self._sim._respawn_at.pop(gname, None)
        self.data.qpos[gq : gq + 3] = (float(x), float(y), 0.12)
        self.data.qpos[gq + 3 : gq + 7] = (1.0, 0.0, 0.0, 0.0)
        gd = self._gift_addr.get(npc.item, (-1, -1))[1]
        if gd >= 0:
            self.data.qvel[gd : gd + 6] = 0.0
