# ai.md — decadic/state

Quick orientation for future edits in this area.

## Purpose
Persistent, maintained **cognitive state** that lives across cycles (as opposed to
the per-cycle neural forward pass in `decadic/nn`). These modules hold the agent's
homeostatic drives, perceptual synthesis, working memory, world graph, curiosity
drive, the A–F State Bus, and — new — the latched **goal lifecycle** that defines
learning-episode boundaries.

## Files (each < 500 lines, house rule)
- `viability.py` — `Homeostasis`: the innate homeostatic reservoirs (hydration /
  energy / integrity), setpoints, and pain/pleasure scaffolding (element B). The
  single innate drive everything else learns from. `interoceptive_drive_pain` is the
  tonic deprivation **pain**; `drive_reduction_reward(prev, cur, gain)` is its phasic
  **relief** complement (reward = per-cycle drive *drop*, bounded to `[0,1]`;
  homeostatic-RL, fully intrinsic). The `*_stub` functions are the non-neural numpy
  pipeline's Phase-1 placeholders only — the neural cycle uses the real signals.
- `state_bus.py` — persistent cognitive state elements A–F (state_of_mind,
  emotion_physio, narrative_emb, metacognition, …) with config-aligned vector dims.
  `prev_drive_pressure` holds last cycle's drive pressure so the relief reward is a
  true per-cycle delta; it is snapshotted + restored so relief is continuous across
  restarts (not spuriously triggered on resume).
- `perceptual_state.py` — maintained perceptual synthesis from streaming obs.
- `working_memory.py` — bounded, decaying slots (object permanence / temporal
  persistence); entities persist at falling salience after leaving view.
- `world_graph.py` — parses `world_state` into a self-indexed egocentric graph
  (spatial / proximity / affective / context edges).
- `curiosity.py` — need-gated *learning-progress* epistemic drive (rewards falling
  forward-model error, not raw surprise → avoids the noisy-TV pathology).
- `goal_lifecycle.py` — **NEW. `GoalState` (latch / hold / close) + `GoalEvent`.**
  Segments the continuous, overlapping homeostatic drive into discrete goal-directed
  episodes so return-based credit assignment has crisp boundaries. `GOAL_LABELS` =
  hydration/energy/integrity; outcomes `OUTCOME_ACHIEVED/ABANDONED/TRUNCATED/DIED`.
  `update(reservoirs_norm, cycle, *, alive=True) -> list[GoalEvent]`:
  - opens (latches) the **dominant** weighted deficit when it crosses
    `onset_deficit`;
  - closes ACHIEVED when the goal's reservoir rises above `satisfy_level`
    (re-opening immediately if another need still presses);
  - closes ABANDONED when a *different* deficit dominates for `abandon_cycles`
    straight (then opens that need);
  - closes TRUNCATED past `max_cycles`; closes DIED when `alive=False`.
  Read-only helpers: `status` (`idle`/`active`), `goal_id`, `episodes`, `dwell(cycle)`.
  Pure (no torch / no MuJoCo) → unit-tested in `tests/test_goal_lifecycle.py`.

## Invariants (do not break)
- `goal_lifecycle.py` stays pure (lists/floats only) so it is trivially testable and
  cheap to run every cycle. No new external reward is invented here — goals are
  derived purely from the existing innate reservoirs.
- A `GoalState.update` call returns AT MOST one `closed` + one `opened` event per
  cycle; callers (runtime) route `closed` events to the episode accumulator.

## Seams (where state is consumed)
- `decadic/agents/runtime.py` owns one `GoalState`; `_advance_goal()` feeds it
  normalized reservoirs each cycle and turns `GoalEvent`s into episode open/close
  on the `EpisodeAccumulator` (see `decadic/consolidation/ai.md`). Goal telemetry
  (`goal`, `goal_status`, `goal_dwell`, `goal_episodes`, `goal_last_outcome`) rides
  `metrics`. `reset()`/`revive()` re-init the goal state for a clean new life.
- `viability.Homeostasis` is the reservoir source; the per-step reward (weighted
  deficit reduction) is computed in `decadic/cycle/neural_pipeline.py`. That cycle
  also assembles the phasic affect: PE pain from the real predictive-coding loss
  (`config.pe_stub_weight()`, default 0) + the `drive_reduction_reward` relief
  (`config.drive_reward_enabled()`), floored into `pleasure_scalar` like curiosity.
