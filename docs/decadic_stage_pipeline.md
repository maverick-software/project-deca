# Serial Cognition + Lossless Prefetch

## Contract

The default runtime is `serial_prefetch`: a producer/consumer architecture that
keeps cognition serial while preventing perceptual information loss.

The invariant is:

```text
continuous lossless perception folding
serialized Decadic cognition and reality commitment
```

Every received observation becomes a `DecadicSession`. Producer workers predecode
the observation, prepare anonymous perceptual evidence outside the cognition
lock, and then apply that evidence into the scene model in arrival order. The
Decadic cycle then deep-processes one prepared observation at a time.

The raw prefetch queue is bounded. Default behavior is lossless blocking
backpressure, not silent drop:

```text
DECADIC_PREFETCH_QUEUE_MAX_FRAMES = effective max(32, parallel_sessions * 3), hard max 128
DECADIC_PREFETCH_OVERLOAD_POLICY  = block | drop_oldest
DECADIC_READY_COALESCE_POLICY     = freshest | oldest
```

`block` is the default. `drop_oldest` is explicit opt-in and increments
`information_loss` because unfolded evidence was discarded.

## Runtime Modes

`DECADIC_PROCESSING_MODE` accepts:

- `serial_prefetch`: default. Every frame is prefetched/folded; serial cognition
  consumes prepared observations one at a time.
- `persistent_parallel_perception`: fallback. Perception is pipelined and folded
  into the scene model; cognition samples the latest coherent scene.
- `batching_observations`: legacy fallback. Up to `K` recent observations are
  encoded together and recency-pooled into one cycle.

`stage_pipeline` is accepted as a deprecated compatibility alias for
`serial_prefetch`. `DECADIC_STAGE_PIPELINING_ENABLED=0` changes the unset default
back to `persistent_parallel_perception`.

## Session Lifecycle

1. Observation arrives.
2. Runtime applies immediate body/homeostasis effects and captures snapshots.
3. Producer worker predecodes camera/audio off the cognitive lock.
4. Producer prepares slot/object-file evidence outside the cognition lock.
5. Ordered fold/apply mutates perception/scene state for that frame.
6. Folded session enters the prepared ready queue.
7. Serial Decadic cycle pops one prepared session.
8. `run_neural_cycle` runs once for that observation.
9. The runtime commits action, StateBus, replay, recurrent buffers, episodic
   memory, and LTM gates through the existing serialized path.

Under overload, already-folded sessions may be coalesced out of deep cognition.
This increments `coalesced_sessions`, not `information_loss`.

`freshest` coalescing keeps the newest folded-ready frame when cognition is
behind. `oldest` coalescing preserves FIFO preference by coalescing newer folded
frames first. Both policies preserve folded perceptual evidence.

## Mutability Rules

Producer workers may:

- predecode observations;
- prepare frozen encoder inputs;
- prepare slot/object-file evidence outside the cognition lock;
- apply anonymous object-file/scene evidence under the runtime lock;
- update perception diagnostics.

Producer workers may not:

- emit actions;
- mutate optimizer state;
- mutate recurrent buffers;
- push replay;
- write episodic memory;
- write LTM.

Those operations remain exclusive to the serialized Decadic commit path.

## Diagnostics

Important metrics:

- `frames_received`
- `frames_prefetched`
- `frames_folded`
- `frames_deep_processed`
- `coalesced_sessions`
- `information_loss`
- `producer_overlap_ratio`
- `decode_on_consume_ms`
- `consume_wait_ms`
- `prefetch_queue_depth`
- `prefetch_queue_max`
- `prefetch_backpressure_events`
- `prefetch_backpressure_ms`
- `oldest_unfolded_age_ms`
- `ready_queue_depth`
- `ready_coalesce_policy`
- `fold_lag_ms`
- `stage_pipeline_*` compatibility metrics for older dashboards/tests

Healthy serial-prefetch runs should maintain:

```text
frames_folded == frames_received
information_loss == 0
frames_deep_processed <= cycles_completed
```

## Label Boundary

Session diagnostics strip semantic/oracle keys such as `label`, `class_name`,
`sim_class`, `oracle_label`, `task_label`, and `reward_label`. Runtime cognition
continues to receive anonymous perceptual entities, scene/focus state, affect,
interoception, and prediction errors, not simulator object classes or task labels.

## Compatibility Notes

The module path `decadic/cycle/stage_pipeline.py` and class name
`DecadicStagePipelineSupervisor` remain as compatibility aliases. They now point
to the serial-prefetch supervisor and no longer implement fake ten-stage
candidate work.
