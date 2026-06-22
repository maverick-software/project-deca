# Epistemic Maturity Second Half - WBS

## Summary

Implement the unfinished developmental skepticism system on top of the current
provisional-entry and async-LTM architecture.

The central constraint is performance and Decadic integrity:

```text
Epistemic maturity may guide the cognitive cycle, but maturity evidence aggregation must remain off the critical path.
```

## 1. Epistemic State Core

- Add `decadic/state/epistemic.py`.
- Define `EpistemicState` with:
  - `maturity`
  - `effective_maturity`
  - `evidence_total`
  - `age_cycles`
  - `failure_ema`
  - `promotion_count`
  - `contradiction_override_count`
  - `reopen_events`
  - `last_reopen_cycle`
  - `promotion_precision_threshold`
  - `promotion_min_seen`
  - `belief_update_floor`
  - `plasticity_scale`
- Add methods for:
  - update from cached semantic/LTM stats
  - update from worker reports
  - update failure EMA
  - compute promotion precision threshold
  - compute promotion min-seen threshold
  - compute maturity-scaled belief update gain
  - apply reopening
  - serialize/deserialize snapshots
- Add config helpers for:
  - `DECADIC_EPISTEMIC_MATURITY_ENABLED`
  - `DECADIC_EPISTEMIC_S_MAX`
  - maturity growth rate
  - evidence weights
  - promotion precision scaling
  - promotion min-seen scaling
  - min update rate
  - contradiction override threshold
  - reopen threshold
  - reopen decay
  - reopen cooldown
  - optional plasticity anneal.

Acceptance:

- `EpistemicState` can be constructed without graph access.
- Maturity rises from supplied cached evidence.
- Effective maturity drops when failure EMA rises.
- All update rates retain a nonzero floor.

## 2. Runtime Ownership And Cycle Snapshot

- Add `self.epistemic` to `AgentRuntime`.
- Update epistemic state once per committed cycle from cached LTM stats and recently completed worker reports.
- Add a read-only epistemic snapshot to `CycleContext`.
- Expose epistemic values through runtime metrics and `snapshot_state()`.
- Do not mutate epistemic state inside candidate stages.
- Do not mutate epistemic state inside async LTM worker threads directly.
- Do not block the cycle waiting for LTM worker completion.

Acceptance:

- Runtime metrics include epistemic maturity values.
- `CycleContext` contains stable per-cycle epistemic thresholds.
- Stage pipeline/candidate sessions can serialize diagnostics without tensors or labels.
- No LTM queue flush occurs during maturity update.

## 3. Async LTM Worker Integration

- Extend `WriteBehindLongTermGraph` job reports with cheap summary fields:
  - promoted entity count
  - semantic update counts
  - unstable belief count
  - contradiction override count
  - reopened entity/belief count
  - retention/prune counts
- Add a cheap graph maturity-stats method if needed, backed by cached counters.
- Preserve semantic evidence throttling via `DECADIC_LTM_SEMANTIC_EVIDENCE_INTERVAL`.
- Preserve scene relation caps and retention pruning.
- Ensure delayed worker reports can update maturity later without changing current-cycle decisions.

Acceptance:

- Worker reports provide enough data for maturity updates.
- Runtime can update maturity from cached stats only.
- Graph size growth does not increase cognitive-cycle maturity cost.

## 4. Maturity-Aware Promotion

- Replace fixed Stage 10 promotion threshold with epistemic snapshot values:
  - `effective_precision_threshold = base + S * extra`
  - `effective_min_seen = base_seen + round(S * seen_extra)`
- Keep Working Memory entry permissive.
- Keep provisional semantic evidence recording permissive.
- Keep LTM consolidation async through `enqueue_consolidation_job(...)`.
- Pass maturity thresholds into the job snapshot/report path where the worker needs them.

Acceptance:

- Young state promotes with low thresholds.
- Mature state requires stronger repeated evidence.
- Stage 10 still enqueues work and does not synchronously consolidate.
- Provisional evidence is still recorded when promotion is blocked.

## 5. Maturity-Aware Belief Revision

- Scale property/semantic belief update gain by maturity and prior confidence.
- Clamp update gain to `DECADIC_EPISTEMIC_MIN_UPDATE_RATE`.
- Apply revision scaling primarily in the LTM worker path, where property beliefs and semantic evidence are updated.
- Preserve existing evidence-weighted/Bayesian behavior; maturity modifies gain, not payload meaning.

Acceptance:

- Mature beliefs resist single-frame noise.
- Mature beliefs still update under sustained evidence.
- Minimum update rate never reaches zero.
- Runtime labels or simulator classes are not introduced.

## 6. Contradiction Override

- Make `contradiction_pressure` operational.
- Detect sustained high-confidence contradiction in Working Memory refresh and/or LTM belief updates.
- When override threshold is crossed:
  - reduce entity precision
  - mark entity/belief unstable
  - set provisional/reopened state where appropriate
  - increment override counters
  - report override to runtime maturity
- Override must bypass `S`.

Acceptance:

- A mature high-precision entity is not revised by one noisy frame.
- A mature high-precision entity is revised after sustained contradiction.
- Override count appears in worker reports and runtime metrics.

## 7. Failure EMA And Global Reopening

- Track `failure_ema` from:
  - contradiction override rate
  - unstable belief rate
  - scene prediction instability
  - reidentification failures
  - conclusion/value reversals
- If `failure_ema >= reopen_threshold` and cooldown permits:
  - reduce effective maturity
  - temporarily increase plasticity scale
  - increment reopen count
  - set reopening status in telemetry
- Reopening lowers certainty gates only; it must not delete memory.

Acceptance:

- Synthetic regime change increases failure EMA.
- Failure EMA lowers effective maturity.
- Reopening happens once per cooldown window.
- Maturity can recover after later corroborated evidence.

## 8. Persistence, API, Dashboard

- Persist `EpistemicState` in `checkpoint_payload()` and `apply_checkpoint_payload()`.
- Saved-agent save/load must preserve epistemic state through the existing state JSON.
- Missing saved state loads as a young/default epistemic state.
- Extend API/state/metrics/discovery payloads and TypeScript types with:
  - `epistemic_maturity`
  - `epistemic_effective_maturity`
  - `epistemic_evidence_total`
  - `epistemic_failure_ema`
  - `epistemic_promotion_threshold`
  - `epistemic_min_seen`
  - `epistemic_plasticity_scale`
  - `epistemic_override_count`
  - `epistemic_reopen_events`
- Add compact dashboard readout in LTM/Discovery or Capacity.
- Show status as young, mature, failing, or reopening.

Acceptance:

- Checkpoint/restore preserves maturity.
- Saved-agent load preserves maturity.
- Dashboard build passes.
- Users can tell whether strict promotion is due to maturity or perception failure.

## 9. Test Plan

Unit tests:

- `EpistemicState` maturity rises with cached semantic evidence.
- `S` caps below `1.0`.
- failure EMA lowers effective maturity.
- promotion threshold increases with `S`.
- min update rate never reaches zero.
- contradiction pressure triggers override/reopening.
- global failure spike triggers reopening after cooldown.
- env disable restores fixed-threshold behavior.
- no semantic/oracle labels enter epistemic payloads.

Integration tests:

- Stage 10 uses epistemic thresholds while still enqueuing async LTM work.
- maturity updates from cached/worker stats without flushing the LTM queue.
- cycle timing does not regress when graph size grows.
- mature agent resists one-frame contradiction.
- mature agent revises after sustained contradiction.
- provisional evidence still records when promotion is blocked.
- checkpoint and saved-agent restore preserve epistemic state.

Verification commands:

- targeted epistemic tests
- targeted Stage 10 async tests
- targeted semantic graph/write-behind tests
- API/dashboard tests
- full Python suite
- dashboard build

## Assumptions And Defaults

- First-half provisional entry remains default-on.
- Epistemic maturity is enabled only after tests pass; env disable remains available.
- Default `S_max = 0.85`.
- Default minimum update rate is `0.05`.
- `S` never blocks Working Memory entry.
- `S` affects promotion and revision only.
- Contradiction override bypasses maturity.
- Reopening lowers certainty/plasticity gates but does not erase memory.
- Maturity evidence aggregation stays off the cognitive critical path.
- Runtime remains Decadic-compliant: no semantic labels, rewards, simulator classes, or task hints enter live cognition.

