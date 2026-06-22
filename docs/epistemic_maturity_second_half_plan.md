# Epistemic Maturity Second Half - Updated Plan

## Summary

This document updates the second-half epistemic maturity plan to match the
current codebase. The first half is already implemented: percepts can enter
Working Memory provisionally from moment one, anonymous semantic evidence is
recorded, and LTM promotion is stricter than entry.

The remaining work is the developmental maturity system: a global maturity
scalar `S`, maturity-aware promotion and belief revision, contradiction
override, failure recalibration, and global reopening.

Hard architectural rule:

```text
Epistemic maturity may guide the cognitive cycle, but maturity evidence aggregation must remain off the critical path.
```

The reason is current architecture: Stage 10 no longer performs heavy long-term
memory consolidation directly. It snapshots candidates and uses the
write-behind LTM worker via `enqueue_consolidation_job(...)`. Epistemic maturity
must preserve that design and must not reintroduce graph scans or semantic
aggregation into the synchronous Decadic cycle.

## Current State

Implemented first half:

- Provisional Working Memory entry is default-on.
- Low-confidence percepts can enter WM above the tiny perceptual entry floor.
- Extended entities are first-class structural entities, not discarded.
- WM slots carry `precision`, `provisional`, `evidence_count`, and `contradiction_pressure`.
- Stage 10 records provisional semantic evidence instead of requiring immediate promotion.
- LTM contains Framework-style semantic counters for entities, events, relationships, correlations, conclusions, and values.
- LTM consolidation is off-cycle through `WriteBehindLongTermGraph.enqueue_consolidation_job(...)`.
- LTM matching and consolidation are bounded/cached through match caches, semantic throttling, scene-edge caps, retention pruning, and worker metrics.

Still missing:

- No runtime-owned `EpistemicState`.
- No global maturity scalar `S`.
- `epistemic_maturity_enabled()` and `epistemic_s_max()` exist but are not active behavior controls.
- Stage 10 still uses the fixed `entity_promotion_precision()` threshold.
- Belief revision is not scaled by maturity.
- `contradiction_pressure` is passive telemetry; it does not revise, demote, or reopen beliefs/entities.
- No `failure_ema`.
- No global reopening after persistent model failure.
- No persisted developmental state in checkpoint/saved-agent payloads.
- No API/dashboard maturity readout.

## Target Architecture

Add a runtime-owned `EpistemicState` that represents the agent's developmental
confidence in its learned world model.

Core data flow:

```text
Working Memory / Scene Workspace
  -> Stage 10 promotion candidates
  -> async LTM consolidation worker
  -> cached LTM + semantic stats / worker reports
  -> AgentRuntime EpistemicState update
  -> read-only epistemic snapshot into future cycles
```

Key boundaries:

- `AgentRuntime` owns and mutates `EpistemicState`.
- `CycleContext` receives a read-only epistemic snapshot.
- Stage 10 uses that snapshot only for promotion thresholds.
- Async LTM worker continues to perform consolidation, semantic evidence writes, property belief updates, edge updates, and retention.
- Runtime updates maturity from cached graph stats and worker reports, not direct full graph scans.
- One or more cycles of maturity lag is acceptable.
- `S` affects promotion and revision only.
- `S` never blocks Working Memory entry.
- No semantic labels, rewards, simulator classes, or task hints enter live cognition.

## Developmental Behavior

Young agent:

- low `S`
- permissive promotion thresholds
- high plasticity
- provisional evidence can become stable with relatively little corroboration

Mature agent:

- higher `S`
- stricter promotion thresholds
- slower belief/property revision
- transient noise is less likely to corrupt durable memory

Failing agent:

- elevated `failure_ema`
- reduced effective maturity
- temporary reopening of promotion/revision gates
- no deletion of learned memory

## Guardrails

G1 - hard plasticity/update floor:

- `S` is capped below `1.0`; default `S_max = 0.85`.
- Maturity-scaled update rates are clamped above a nonzero floor.
- A mature belief can become slow to move, but never impossible to update.

G2 - sustained contradiction override:

- Repeated high-confidence contradiction raises operational contradiction pressure.
- Once threshold is crossed, override bypasses `S`.
- Override reduces precision, marks the entity or belief unstable, and reopens revision/promotion status where appropriate.

G3 - global failure EMA and reopening:

- Runtime tracks a failure EMA from contradiction overrides, unstable belief rate, scene prediction instability, reidentification failure, and conclusion/value reversals.
- When failure EMA crosses threshold and cooldown allows, effective maturity drops and plasticity scale temporarily rises.
- Reopening lowers certainty gates but does not erase memory.

## Runtime And Public Surfaces

Expose epistemic maturity as runtime/developmental telemetry:

- `epistemic_maturity`
- `epistemic_effective_maturity`
- `epistemic_evidence_total`
- `epistemic_failure_ema`
- `epistemic_promotion_threshold`
- `epistemic_min_seen`
- `epistemic_plasticity_scale`
- `epistemic_override_count`
- `epistemic_reopen_events`

Dashboard should show the agent as one of:

- young
- mature
- failing
- reopening

The display belongs in the LTM/Discovery or Capacity area because it explains
why memory promotion and belief revision are becoming stricter or reopening.

## Acceptance Criteria

- `epistemic_maturity_enabled()` and `epistemic_s_max()` control active runtime behavior.
- `EpistemicState` exists and is owned by `AgentRuntime`.
- `CycleContext` carries a read-only epistemic snapshot.
- Stage 10 uses maturity-aware promotion thresholds but still enqueues LTM work asynchronously.
- Maturity updates use cached stats or worker reports and never force LTM queue flushes or full graph scans in the cycle.
- Belief update gain is maturity-scaled with a hard nonzero floor.
- `contradiction_pressure` causes real override/revision behavior.
- `failure_ema` can reduce effective maturity and trigger global reopening.
- Checkpoint and saved-agent restore preserve epistemic state.
- Dashboard/API expose maturity state.
- No semantic labels, rewards, simulator classes, or task hints enter cognition, replay, LTM semantic payloads, or dashboard object payloads.

