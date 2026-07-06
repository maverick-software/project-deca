# WBS — Goal-Directed Foraging & Memory-Guided Navigation (WS-FORAGE)

**Companion:** `foraging_goal_navigation_implementation_plan.md` (the "what/where/
risks"), `foraging_goal_navigation_gap_analysis.md` (the "why"). Dependency order
only — no calendar. ⚙ = needs the live rig (GPU/body). Every milestone ends
flag-off byte-identical (suite-enforced) and independently revertible.

**Merge gate (all milestones):** unit tests green · full `pytest -q` green with
all WS-FORAGE flags OFF (parity) *and* ON-at-init (zero-init parity) · the
milestone's probe archived under `reports/`.

---

## M0 — Curriculum: "place within reach" (Gap A)

*Unblocks everything; no neural change; highest leverage.*

- **M0.1** `give_within_reach(kind)` in `scripts/mujoco_decadic_adapter.py` (prop
  relocated ~0.5 m ahead, inside reach but requiring the last-gap close); body
  command `give_{res}_reach`.
- **M0.2** `/agent/{id}/give` route accepts `mode=within_reach`
  (`decadic/api/app.py`); dashboard "Place within reach" button (frontend note).
- **M0.3** Deterministic within-reach placement schedule (scenario option) for
  reproducible A/B when deprived.
- *Accept:* prop lands within reach and is consumable; `resource_provision
  mode=reach` + `nourishment` logged; `test_give_resource.py` / `test_api_dashboard.py`
  green (existing modes untouched).
- *Probe ⚙:* deprived agent + within-reach resource → repeated approach+consume.

## M1 — Credit-horizon safety: (1−γ) return normalization

*Prerequisite for any γ change; pure/small; reversible by flag.*

- **M1.1** Normalize λ-returns / SF targets by `(1−γ)` in
  `decadic/consolidation/returns.py`; consume in `episodes.py:on_close` /
  `consolidator.py`; rescale `SF_VALUE_WEIGHT`. Flag
  `DECADIC_SF_NORMALIZE_RETURNS` (default OFF).
- **M1.2** Unit tests: magnitude ≈ invariant across γ∈{0.97,0.99,0.998} when ON;
  byte-identical when OFF.
- *Accept:* parity OFF; magnitude-invariance ON; SF/consolidation suite green.

## M2 — Credit-horizon extension: γ→0.995, λ→0.8

*Depends on M1. Config + telemetry + soak; no structural code.*

- **M2.1** `SF_GAMMA` 0.97→0.995, `SF_LAMBDA` 0.9→0.8 defaults (env-overridable);
  ship with `DECADIC_SF_NORMALIZE_RETURNS=1`.
- **M2.2** Telemetry: SF-loss + `successor_value` magnitude bounds in metrics;
  `run_body_diag.ps1` summary greps them.
- **M2.3 ⚙** Soak A/B (γ=0.97 vs 0.995): SF loss bounded, value finite &
  discriminative, throughput within noise.
- *Accept:* A/B archived; no instability; no throughput regression. **Gate:** do
  not proceed to minutes (γ→0.998) until an approach that *needs* >50 s is
  observed.

## M3 — Goal→policy bridge I: goal conditioning (Gap C, half 1)

*Architectural spine, step 1. Highest-risk file (`neural_stack.py`) — snapshot,
tiny diffs, full suite after each edit.*

- **M3.1** Freeze `GOAL_VEC_DIM`; `GoalEncoder` (need id + deficit → goal vector,
  zero when no goal) in new `decadic/nn/goal_conditioning.py`; layout test.
- **M3.2** Zero-init `goal_ingress`; add its contribution to `pol_in_t`
  (`neural_stack.py` ~854) before policy/motor heads. Flag
  `DECADIC_GOAL_CONDITIONED_POLICY`.
- **M3.3** Plumb goal vector `GoalState → CycleContext → neural_pipeline → forward`
  (`runtime.py`, `neural_pipeline.py`).
- *Accept:* layout freeze; zero-goal→zero contribution; content-sensitivity after
  one grad step; flag-off & zero-init-on both byte-identical; full suite green.

## M4 — Goal→policy bridge II: egocentric bearing to remembered target (Gap C, half 2)

*Depends on M3. New pure module; additive/low-risk.*

- **M4.1** `resolve_goal_target(goal_id, ltm_graph, beliefs)` in new
  `decadic/state/spatial_recall.py` — need-conditioned query via `predicts_*_relief`
  belief → target entity + last-known `position_json`.
- **M4.2** `egocentric_bearing(self_pos, self_yaw, target_pos)` — pure
  allocentric→egocentric transform → (cosθ, sinθ, norm-dist); masked when no
  memory. Torch-free unit tests on known geometry.
- **M4.3** Extend goal vector with masked bearing channels (versioned layout
  amendment); flag `DECADIC_GOAL_BEARING`.
- **M4.4** Landmark fallback in `resolve_goal_target` (highest-confidence
  co-visible neighbor when target stale) — robustness to a non-stationary world.
- *Accept:* transform unit tests pass (dead-ahead/left/behind); mask-when-empty;
  fallback selection; flag-off byte-identical; "bearing points at remembered
  resource" probe over a scripted recall scenario.

## M5 — Type-2 control flow: need∧¬cue → escalate → memory search

*Depends on M4. The dual-process spine; reuse the gate + imagination.*

- **M5.1** `type2_search` escalation source in `decadic/cycle/attention_gate.py`:
  fires on `deficit≥onset ∧ goal_entity_not_visible`; gate `reason=
  "type2_memory_search"`.
- **M5.2** On Type-2 escalation (`runtime.py`): run `spatial_recall` (M4) and, if
  `imagination_enabled`, a **bounded** online SF rollout
  (`consolidation/imagination.py`, hard step cap `type2_rollout_max_steps`).
- **M5.3** Flags `DECADIC_TYPE2_SEARCH`, `type2_rollout_max_steps`; per-cycle cost
  bounded and logged.
- *Accept:* trigger truth-table (fires only need∧¬cue); bounded cost; flag-off
  byte-identical; behavioral probe — thirsty + resource seen-then-hidden → orients
  to remembered bearing (flag-off control does not).

## M6 — Locomotion & gaze (Gaps D/E) — parallelizable

- **M6.1** Foraging phase in `decadic/training/skills.py` (reuse `stand_teacher` →
  within-reach → nearby, staged assistance reduction).
- **M6.2** *(deferred)* `DECADIC_GAZE_ORIENT_BIAS` — light salience-orient term,
  zero-init/flagged; build only if E proves rate-limiting after A–D.
- *Accept:* dojo phase unit tests; gaze parity if built.

## M7 — Validation campaign

- **M7.1** Foraging verdict script (`scripts/`, gate-probe mold) + flags-off
  ablation.
- **M7.2** Memory-navigation verdict script + flags-on/off ablation.
- **M7.3** Regression sweep: full suite green all-flags-off (byte-identical) and
  all-on; stability soak (all WS-FORAGE flags on, one embodied hour) — 0 stalls,
  bounded per-cycle cost, throughput within noise.
- **M7.4** Checkpoint/bundle compat: new heads ride the versioned bundle;
  route-level save/load round-trip with flags on (WS5-M4.2 precedent).
- *Accept:* both verdicts archived with flags-on/off deltas; soak stamp; save/load
  round-trip green.

---

## Cross-cutting (every milestone)

- **Parity:** flag OFF ⇒ byte-identical; new flags pinned "0" in
  `tests/conftest.py`; enforced by full-suite runs.
- **Backups:** git branch per milestone + `reports/_backups/` snapshot before
  editing `neural_stack.py` / `neural_pipeline.py` / `runtime.py`.
- **Zero-init/ramp:** every new head zero-init; shaping/bearing weights ramp from 0.
- **Anti-hallucination:** SF weights stay detached in the live value path.
- **Cost:** any new per-cycle work measured (`bench_relational.py` pattern) before
  a default flip; Type-2 path stays gated + bounded.
- **Artifacts:** every probe archived under `reports/` with verdict logs.

## Dependency graph

```
M0 (within-reach) ──────────────┐  (supplies training data)
M1 (normalize) ── M2 (γ→0.995) ─┤
                                 ├─> M3 (goal-cond) ─> M4 (bearing) ─> M5 (Type-2)
M6 (locomotion/gaze) ── parallel ┘                                        │
                                                                          v
                                              M7 (validation: probes, soak, ckpt)
M2 → (γ→0.998 "minutes") gated on M4/M5 (a value over minutes needs goal persistence)
```

## Success criteria (workstream headline)

1. A deprived agent **approaches and consumes** a within-reach resource
   (M0+curriculum) — the first closed act→relief loop.
2. With the resource **seen then hidden**, the agent **orients toward its
   remembered location/landmark** (M3–M5) — memory-driven, not cue-driven.
3. The credit horizon spans the **actual approach duration** (M1–M2), extensible
   toward minutes (paired with M4–M5) without SF instability or throughput
   regression.
4. All of the above with **flags off ⇒ today's agent, unchanged** — capability
   grown from experience, never scripted.
