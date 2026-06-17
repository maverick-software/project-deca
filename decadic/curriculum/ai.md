# ai.md — decadic/curriculum

Quick orientation for future edits in this area.

## Purpose
The walking curriculum: a faithful, observation-only "developmental trainer" that
makes locomotion EMERGE from the agent's existing predictive-coding + homeostatic-
drive machinery. It is the "parent that shapes the world and reads gates" — it
NEVER adds a term to the loss. It only (a) retunes live config knobs that REWEIGHT
the existing self-supervised objective, (b) places satisfiers a step ahead, and
(c) reads observational gates to advance phases. The world is a single fixed scene
throughout (house + food + water); no body restarts.

## Files
- `gates.py` — pure, side-effect-free gate logic. `Criterion` (`<=` / `>=` /
  `trend>=`) over a rolling window of metric samples; `evaluate_gate` returns a
  serializable `GateResult` with per-criterion progress. No I/O, no cognition.
- `phases.py` — `PhaseConfig` (live `configure()` kwargs), `SatisfierPolicy`
  (give cadence), `Phase`; the default 4-phase table via `default_phases()` /
  `build_phases()`; optional `affective_phase()` (phase 4, needs a threat); a JSON
  override loader so thresholds/knobs tune without code edits. `CURRICULUM_SCENE`
  is the fixed scene the trainer runs in.
- `supervisor.py` — `CurriculumSupervisor`: single-slot asyncio state machine
  (mirrors `EnvironmentSupervisor`). Polls metrics, places satisfiers, promotes at
  an open gate after a min dwell, demotes + revives on death, checkpoints at phase
  boundaries, and logs to `logs/curriculum_<agent>.log`.

## Faithfulness invariant (do not break)
The supervisor's only agent surface is read / `configure` / `queue_body_command` /
checkpoint (`checkpoint_payload` + `save_brain`) / `revive`. It must never import
`run_neural_cycle`, touch the optimizer, or call `.backward()`. The eval-only
metrics it reads (`distance_traveled`, `net_displacement`, `fall_rate`,
`gait_regularity`, `consume_events`) must never appear in the cognition source
(`decadic/cycle/neural_pipeline.py`, `decadic/nn/neural_stack.py`).
`tests/test_curriculum_faithfulness.py` enforces all of this statically — keep it
green.

## Where the rest lives (not here)
- Telemetry + live-config seams: `decadic/agents/runtime.py`
  (`_capture_gait_and_motion`, `consume_events`, `configure(...)` overrides).
- Override threading: `decadic/cycle/types.py` (CycleContext) +
  `decadic/cycle/neural_pipeline.py` (ctx-or-env reads) +
  `decadic/config.py` (`motor_exploration_sigma(sigma_max=...)`).
- Endpoints + lifespan wiring: `decadic/api/app.py` (`/curriculum*`,
  `app.state.curriculum`).
- UI: `dashboard/src/components/CurriculumPanel.tsx` (Training tab) +
  `dashboard/src/api.ts` (Curriculum* client).
- Distinctness baseline: `scripts/mujoco_decadic_adapter.py --baseline {random,cpg}`.

## Gotchas
- `configure()` and checkpointing acquire `agent.lock`; the poll loop never blocks
  cognition for long (reads are cheap, gives/configs are infrequent).
- Promotion requires BOTH an open gate AND `min_dwell_s` elapsed (a gate must hold,
  not just blip). Demotion (on death) steps back one phase and revives the agent.
- Phase 0 is `immortal` (no death); phases 1+ are `metabolic` and can die.
- Poll interval is `DECADIC_CURRICULUM_POLL_S` (default 2.0s).
- Each file stays < 500 lines (house rule).
