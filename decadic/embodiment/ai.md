# ai.md — decadic/embodiment

Quick orientation for future edits in this area.

## Purpose
Scripted **NPC crowd** ("village"): a small set of collisionless, kinematically-
animated demonstrators confined to their own habitats around the learner. Each
NPC runs a per-zone behavior; one is the *parent* that forages and provisions the
learner on a **need threshold** (not a timer). This is pure embodiment/sim work:
the learner's cognition, losses, and 21-actuator motor contract are untouched.
Extracted here so `scripts/mujoco_decadic_adapter.py` does not keep growing and so
the crowd logic is unit-testable without the body process.

## Files (each < 500 lines, house rule)
- `habitats.py` — `Habitat(center, radius, behavior, is_parent, face, food,
  water)` descriptors + `DEFAULT_HABITATS` (8 zones in-view within FENCE_RADIUS),
  env-var config accessors (`crowd_size`, `habitat_radius`,
  `parent_need_threshold`, `parent_fade_per_offer`, `parent_threshold_floor`,
  `parent_refractory_s`, `crowd_lod_distance`), `clamp_to_zone`, and
  `parent_effective_threshold(offers)` (faded + floored). Pure.
- `npc_behaviors.py` — pure kinematic pose library: `walk_pose`, `stand_pose`,
  `sit_pose`, `communicate_pose`, `lerp_pose`, `sit_stand_blend`. Joint keys
  `r_hip,l_hip,r_knee,l_knee,r_sh,l_sh`. Gait constants mirror the adapter parent.
- `npc_xml.py` — multi-NPC humanoid XML: `clone_humanoid_xml` (prefixed,
  actuator-free, **collisionless** clone of the agent torso), `parent_gifts_xml`
  (`prop_food_gift_c`/`prop_water_gift_c` — distinct from the legacy parent's
  `prop_*_gift`), `habitat_resources_xml` (net-additive `prop_food_h{i}_*` /
  `prop_water_h{i}_*`), `zone_markers_xml` (collisionless decor discs), and
  `crowd_scene_xml` (the full worldbody snippet). `GIFT_NAMES` is the shared map.
- `npc_controller.py` — MuJoCo-bound `CrowdController` + `NPCRuntime`. Discovers
  `npc{i}_*` bodies/joints, animates each substep (`apply`), runs per-zone
  behavior + forage/parent FSMs (`events`), publishes `entities()`, and does
  distance/frustum LOD culling (held static when far/behind). Borrows the host
  `HumanoidSim`'s `food_bodies`/`water_bodies`/`eaten`/`_consume`/`_respawn`.
- `stances.py` — **joint-brace stance library** (single source of truth, pure data +
  resolvers, NO MuJoCo). `Stance` dataclass = label + per-joint reference (DEGREES)
  + spawn `root_z`/`root_quat` + posture `fall_z` (+ optional motion keyframes/
  `period_s`/`loop`). `STANCES`: stand, all_fours, kneel_left, kneel_right (static),
  crawl + sit_to_stand (motion). `resolve(stance, hinge_names, defaults_rad,
  ranges_rad)` -> per-hinge radians (clamps ONLY authored joints so unspecified
  hinges keep qpos0 — stand reproduces the validated zero pose exactly);
  `motion_ref(...phase...)` interpolates/wraps keyframes; `catalog()`/`get_stance()`
  for the API/UI. Imported by BOTH `scripts/mujoco_decadic_adapter.py` and
  `decadic/api/app.py`.
- `resource_placement.py` — **pure** anti-camping scatterer (NO MuJoCo). Given prop
  names, an RNG, `fence_radius`, `min_dist`, `margin`, `mode` (`arena`|`zone`), and
  optional `zones`, returns `{name: (x, y)}` inside the usable arena. `arena` samples
  uniformly in the annulus `[min_dist, fence_radius - margin]`; `zone` picks a habitat
  zone then a point within it (still clamped to the arena). Deterministic for a fixed
  seed. Consumed only by `scripts/mujoco_decadic_adapter.py::_randomize_resources`,
  which scatters ONLY the agent's open static props (`_is_scatterable_prop`: excludes
  gifts AND habitat `prop_*_h*` resources, so the NPC ecology forages in-zone intact).

## Invariants (do not break)
- Adding the crowd must NOT change `model.nu` (stays 21) or the agent's 42-value
  proprioceptive joint vector. The agent hinge scan skips any joint whose name
  starts with `npc` (legacy `npc_` parent AND crowd `npc0_..npcN_`).
- NPC humanoid geoms are collisionless (`contype=0 conaffinity=0`): they render
  but never push the learner's physical body.
- Credit isolation: NPC consumption emits `npc_eat`/`npc_drink` (ignored by
  `classify_events`); only the learner consuming a `food`/`water` body credits it.
- Each NPC stays inside its habitat (`clamp_to_zone` on every waypoint/step).
- Parent provisioning is threshold-gated: it fires only once the refractory has
  elapsed AND a reservoir is at/below the (fading) threshold; with no reservoir
  info it falls back to the refractory (legacy timer behavior).

## Stance/brace seams (where stances.py is consumed)
- `scripts/mujoco_decadic_adapter.py`: `HumanoidSim` captures `hinge_names` +
  `_hinge_ranges`, holds the active stance (`_stance_name/_stance_phase/
  _stance_root_z/_stance_root_quat/_stance_fall_z`), resolves `_q_ref` from it, and
  `set_stance()` re-poses with stance-aware `recenter()` without changing brace
  state; `reset_braces()` is a separate manual command. `step()`
  calls `_advance_motion()` (retargets `_q_ref`+`qpos_spring` from the motion phase)
  before the brace tick. `xfrc_applied[torso]` stays zero in every stance.
  "Hold movement" (`_movement_hold` / `set_movement_hold()` / WS `hold_on`/`hold_off`):
  while on, `step()` calls `_hold_braces()` (pins every `_tightness` to 1.0 -- ROM
  curriculum suspended, no release) instead of `_update_braces()`, and
  `_advance_motion()` loops EVERY motion (`stance.loop or _movement_hold`) so the
  one-shot Rise also repeats -- the movement runs until disabled. `movement_hold`
  rides the snapshot/obs JSON. Only effective while the master braces are on.
- `decadic/api/app.py`: `GET /body/stances`, `POST /agent/{id}/body/stance?name=`,
  `POST /agent/{id}/body/movement_hold?enabled=`.
- `decadic/agents/runtime.py`: `stance`/`stance_phase`/`movement_hold` metrics
  (telemetry only).
- `dashboard/src/{api.ts,components/MotorPanel.tsx,components/SkillDojoPanel.tsx}`:
  `STANCES` + selector buttons.
- Dev tools: `scripts/_gen_stand_pose.py::settle_stance` (settle-check stability),
  `scripts/_probe_pose.py` (per-geom world-z + COM at spawn, for pose authoring).
- Stance stability obeys COM-over-support for a free root held only by position
  braces: deep bipedal squats / single-knee kneels tip (feet shoot ahead of COM),
  so kneels are a knees-down quadruped (twist L/R) and sit_to_stand is a push-up
  from all-fours. Full bipedal sit-to-stand needs dynamic balance (future work).

## Where the adapter/server seams live (not here)
- `scripts/mujoco_decadic_adapter.py`: `SCENE_ELEMENTS["crowd"]` (built via
  `crowd_scene_xml`), `SELECTABLE_ELEMENTS`/`LEGACY_SCENES["village"]`,
  `HumanoidSim.crowd` construction + `apply()`/`events()`/`entities()` wiring,
  `_agent_reservoirs` parsing in `apply_action`, and `npc_pause`/`npc_resume`
  freezing the crowd too. The legacy single-parent `npc` element is unchanged
  except its delivery trigger now goes through `_parent_delivery_due`.
- `decadic/agents/runtime.py`: piggybacks normalized reservoirs onto the outbound
  action `parameters.reservoirs` (body-only telemetry; cognition never reads it).
- `decadic/api/environment.py`: `VALID_ELEMENTS` (+`crowd`), `SCENARIO_PRESETS`
  (`Village` -> `["crowd","house"]`), `available_presets` in the status.
- `dashboard/src/components/EnvironmentPanel.tsx`: the one-click preset dropdown.

## Phasing note
The crowd is a *visual* demonstrator: it only teaches the learner once vision is
on (the foraging phase, after basic locomotion). With vision off it is inert
scenery that still respawns resources. Tune via `DECADIC_CROWD_*` env vars.

## Gotchas
- `crowd_scene_xml` is evaluated at adapter import time; the controller discovers
  whatever `npc{i}_torso` bodies exist, so a `DECADIC_CROWD_SIZE` change between
  import and sim construction is tolerated (extra/missing NPCs are skipped).
- The crowd parent drops gifts at its zone edge nearest the agent (keeps the NPC
  confined while still provisioning); the agent must walk to the habitat to eat.
