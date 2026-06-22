# Perception Organ Runtime Contract

## Purpose

The perception organ is a pre-cognitive sensory layer for the Decadic agent. It improves visual binding, motion sensitivity, body coupling, and memory write quality without changing Decadic cognition semantics.

Runtime cognition receives anonymous perceptual object files only. Offline training may use MuJoCo truth, depth, optical flow, or segmentation-teacher data, but those teacher labels must be stripped before live cognition, Working Memory, LTM, replay records, or dashboard object-file payloads.

## Runtime Boundary

Allowed runtime fields:

- `object_id`: stable anonymous tracking id
- `centroid_uv`: image-space location
- `relative`, `bearing`, `depth`: non-semantic spatial estimates when available
- `appearance`: anonymous visual embedding or summary
- `motion`, `flow`, `local_motion`: motion estimates
- `retina_contrast`: local contrast/edge strength
- `looming`: expansion or near-collision signal
- `persistence`: temporal stability
- `agency`: body-coupled motion score
- `kind_hint`: one of `object`, `stuff`, `body_part_candidate`
- `confidence`, `presence`, `spread`, `mask_entropy`

Forbidden runtime fields:

- semantic class labels such as `food`, `water`, `hand`, `wall`, `building`
- task labels, rewards, or success hints
- oracle object names or simulator entity ids that identify meaning
- segmentation-teacher labels

The `kind_hint` field is intentionally coarse and non-semantic. It exists to prevent background/stuff from poisoning object memory and to mark possible body-coupled regions without telling cognition what the object is.

## Health States

The perception layer reports one of:

- `healthy`: object files are separated, confident, and usable for memory writes
- `low_confidence`: proposals exist but are weak or unstable
- `collapsed`: object files are co-located or near-identical
- `no_objects`: no usable foreground object files are present
- `teacher_only`: data is suitable for offline scaffold training only
- `stale_frame`: visual input is not updating

LTM should prefer skipping writes over storing object files from unhealthy perception states.

## Integration Rules

- Stage 1 may consume improved perceptual object files.
- Stage 3 may correlate anonymous object files with memory.
- Stage 10 may write to LTM only when perception health passes.
- The perception organ may emit diagnostics for dashboards and tests.
- The perception organ must not inject semantic labels, task rewards, or teacher actions into the live Decadic cycle.

## Bootstrap Training Rule

Offline scaffold supervision is allowed only for perception weights. Checkpoints produced by perception bootstrap training may be loaded by the perception organ, but the runtime payload remains anonymous object files and health diagnostics.

