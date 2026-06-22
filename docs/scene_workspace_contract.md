# Scene Workspace Contract

The Scene Workspace is a pre-cognitive perceptual layer. It maintains a
persistent, egocentric, anonymous scene model from object files before bounded
Working Memory, Global Workspace ignition, episodic storage, and LTM
consolidation.

Runtime flow:

```text
camera/body/touch/audio
-> perception organ
-> anonymous object files
-> SceneWorkspace
-> optional/default-on SceneDynamics prediction
-> WorkingMemory focus cache
-> GlobalWorkspace ignition
-> Decadic cognition
-> episodic + long-term memory
```

## Runtime Fields

Scene entities may contain only anonymous perceptual fields:

- `entity_id`
- `object_id`
- `kind_hint`: `object`, `stuff`, or `body_part_candidate`
- `visible`, `occluded`, `occlusion_age`
- `centroid_uv`
- `relative`
- `depth`
- `motion`
- `confidence`, `persistence`, `salience`
- `agency`
- `looming`, `local_motion`, `retina_contrast`
- `predicted_centroid_uv`, `predicted_relative`
- `prediction_visibility`, `prediction_uncertainty`, `prediction_error`
- numeric `property_evidence`
- lifecycle counters such as `first_cycle`, `last_seen_cycle`, `seen_count`

Scene relations may contain only anonymous relation kinds such as:

- `co_visible`
- `near`
- `far`
- `left_of`
- `right_of`
- `above`
- `below`

## Forbidden Runtime Content

Semantic labels and oracle object classes must not enter SceneWorkspace,
Working Memory, Global Workspace, LTM, replay, or dashboard object payloads.
Forbidden terms include:

- `label`
- `class`
- `kind_name`
- `food`
- `water`
- `hand`
- `wall`
- `building`
- `ball`
- `bear`

Offline scaffolds and evaluation may use simulator truth, segmentation teachers,
or labels, but those signals must be stripped before live cognition.

## Responsibility Split

- Perception organ: detects anonymous perceptual object files.
- SceneWorkspace: maintains persistent live scene state and occlusion.
- SceneDynamics: predicts next anonymous entity state for matching, occlusion,
  focus salience, and health gating. It is optional but default-on with
  `DECADIC_SCENE_DYNAMICS_ENABLED=1`; disabling it falls back to constant-velocity
  prediction diagnostics.
- WorkingMemory: bounded focus cache selected from SceneWorkspace.
- GlobalWorkspace: broadcasts an ignited focus coalition.
- Episodic memory: stores compact event/action/outcome scene context.
- LTM: consolidates durable anonymous entity/property/relation beliefs.
