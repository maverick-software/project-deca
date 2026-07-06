# Implementation Plan — Goal-Directed Foraging & Memory-Guided Navigation

**Companion documents:** `foraging_goal_navigation_gap_analysis.md` (why),
`foraging_goal_navigation_wbs.md` (task/milestone breakdown). Working name for
this workstream: **WS-FORAGE**.

**Objective.** Close the loop *need → recall → orient/search → navigate →
relief → reinforcement* so a deprived agent can pursue a resource it does not
currently see, using its own lived memory — and extend the credit horizon from
~8 s to a behaviorally realistic tens-of-seconds-to-minutes — **without**
regressing any existing behavior and **without** hand-coding the policy.

---

## 1. Non-negotiable guardrails (house discipline, enforced every milestone)

These are the rules that keep the work faithful to the project's thesis
("capability is lived, not downloaded") and to the existing engineering house
style. Every milestone is checked against them.

- **G1 — Features ship ON; safety is birth-identity, not an off-switch.** Per the
  owner's standing decision, validated faculties run ON by default. A new
  capability is made safe not by defaulting OFF but by being **birth-identical**:
  zero-initialized so a fresh agent is bit-for-bit unchanged *at cycle 0* and the
  capability only emerges as experience trains it. Env flags remain as escape
  hatches / A-B levers, but the shipped default is ON. The test suite pins the
  relevant params to known values for determinism (the gate / binding-faculty
  pattern: prod-on, tests-pinned) — NOT to keep production off.
- **G2 — Zero-init, ramped.** Every new head/ingress projection is
  zero-initialized so its contribution is exactly 0 at birth; every shaping/value
  weight ramps from 0 (the `sf_value_weight_for_cycle` precedent). This is what
  makes "default ON" safe. **Exception:** a change that is *not* expressible as a
  zero-init addition — notably the M1/M2 value-regime retune (return
  normalization + horizon γ + the compensating value weight) — genuinely alters
  learning dynamics from cycle 0. Such changes still ship ON, but they are
  explicitly flagged as *validation-gated* (the M2 soak) rather than
  birth-identical, and stay fully env-reversible.
- **G3 — Pathway, not policy.** We add *learnable inputs* (goal embedding,
  egocentric bearing, memory-search trigger) to the existing learned policy and
  successor-value machinery. We never write "if thirsty, walk to water." No
  scripted behavior, no pre-loaded world map, no LLM in the loop.
- **G4 — Thin wiring on a large stack.** Reuse existing hooks (`GoalState`, the
  successor-features head, `Entity.position_json`, the attention gate, the
  imagination engine). New substantive code lives in its own small modules; the
  neural stack gains only thin wiring.
- **G5 — Interface freezes.** Any new fixed-dim interface (goal vector, bearing
  vector) is a named constant with a layout test (the WS5-M0.1 pattern), frozen
  before dependents are built.
- **G6 — Anti-hallucination.** Value inputs the policy could exploit stay
  detached where the WS3B/SF precedent requires (the policy cannot inflate its
  own value estimate).

---

## 2. What we build, and exactly where

Grouped by the gap it closes. Each item lists the file(s), the flag, the backup
note, and the test that proves it.

### 2.1 Curriculum: "place within reach" (Gap A) — lowest risk, highest leverage

- **Build:** a shorter-distance placement so a deprived agent can satisfy a need
  by a lean/reach with no locomotion — the first *completable* approach→relief.
- **Where:**
  - `scripts/mujoco_decadic_adapter.py` — new `give_within_reach(kind)` beside
    `give_near` (same prop-relocation seam; distance ~0.5 m in front, inside the
    ~1.0 m reach envelope but still requiring the agent to close the last gap).
    New body command `give_{res}_reach`.
  - `decadic/api/app.py` — extend the `/agent/{id}/give` route `mode` set with
    `within_reach` (maps to `give_{res}_reach`). Preserve existing modes.
  - Dashboard: add a "Place within reach" button (frontend; note only — no
    behavior change server-side).
- **Flag/behavior:** additive; existing modes unchanged. No neural change.
- **Backup:** git branch `ws-forage/m0`; adapter and app.py are large — snapshot
  both before edit (copy to `reports/_backups/` and commit HEAD first).
- **Tests:** unit — new mode routes to the correct command; the prop lands within
  reach. Regression — `tests/test_give_resource.py` full pass (existing modes),
  `test_api_dashboard.py`.

### 2.2 Credit-horizon safety: (1−γ) return normalization (Horizon Step 1)

- **Build:** normalize λ-returns / SF targets by `(1−γ)` (discounted *average*,
  not *sum*) so target magnitude is invariant to γ — the prerequisite for any γ
  increase.
- **Where:** `decadic/consolidation/returns.py` (`lambda_returns`,
  `lambda_returns_vec`) — optional post-scale; `decadic/consolidation/episodes.py`
  (`on_close`) and `consolidator.py` where targets are consumed. Rescale
  `SF_VALUE_WEIGHT` accordingly (config).
- **Flag:** `DECADIC_SF_NORMALIZE_RETURNS` (default OFF → byte-identical; turned
  ON as a bundle with the γ change in 2.3).
- **Backup:** git branch `ws-forage/m1`; `returns.py` is pure/small — low risk.
- **Tests:** unit — with normalization on, target L2 magnitude ≈ invariant across
  γ ∈ {0.97, 0.99, 0.998}; with it off, values byte-identical to today.
  Regression — full consolidation/SF test set.

### 2.3 Credit-horizon extension: γ→0.995, λ→0.8 (Horizon Step 2)

- **Build:** raise the discount to a realistic approach horizon (~50 s). No code
  change — `SF_GAMMA`/`SF_LAMBDA` are already env-tunable — but paired with 2.2
  and with new stability telemetry.
- **Where:** `decadic/config.py` defaults (documented, still env-overridable);
  telemetry in `decadic/cycle/neural_pipeline.py` / metrics — expose SF-loss and
  `successor_value` magnitude bounds; extend `scripts/run_body_diag.ps1` summary
  to grep them.
- **Flag:** the γ/λ values themselves; ships with `DECADIC_SF_NORMALIZE_RETURNS=1`.
- **Backup:** config diff only; trivially revertible.
- **Tests:** unit — horizon math sanity. **Soak A/B** — γ=0.97 vs 0.995 (both with
  normalization): SF loss bounded, `successor_value` finite and *discriminative*
  (approach-action value > standing-still value once trained), no throughput
  regression beyond noise.

### 2.4 Goal→policy bridge, part 1: goal conditioning (Gap C, half 1)

- **Build:** make the motor policy *aware of the active need*. Encode
  `GoalState.goal_id` + latched deficit into a small fixed **goal vector**; feed
  it to the policy via a zero-init ingress.
- **Where:**
  - New module `decadic/nn/goal_conditioning.py` — `GoalEncoder` (need id + deficit
    → `GOAL_VEC_DIM` vector; zero when no goal) and a zero-init `goal_ingress`
    projection (twin of the WS5 slot ingress).
  - `decadic/nn/neural_stack.py` — add `goal_vec` param to `forward`; add its
    zero-init contribution into `pol_in_t` (line ~854) *before* the policy/motor
    heads. Interface dims frozen (G5).
  - `decadic/cycle/neural_pipeline.py` + `decadic/agents/runtime.py` — plumb the
    goal vector from `GoalState` through `CycleContext` into the forward call.
  - `decadic/config.py` — `GOAL_VEC_DIM` constant; `goal_conditioned_policy_enabled()`.
- **Flag:** `DECADIC_GOAL_CONDITIONED_POLICY` (default OFF).
- **Backup:** git branch `ws-forage/m3`; **`neural_stack.py` is the highest-risk
  file** — snapshot before edit, edit via small localized diffs, run the full
  neural suite after each edit.
- **Tests:** unit — goal-vector layout freeze; zero-goal → zero contribution;
  content sensitivity (different need → different policy *after* a grad step).
  Parity — flag-off byte-identical; flag-on at init byte-identical (zero-init).
  Regression — full suite green both states.

### 2.5 Goal→policy bridge, part 2: egocentric bearing to remembered target (Gap C, half 2)

- **Build:** the piece that lets incentive salience pull toward an *out-of-view*
  target. Need-directed memory query → remembered resource (and/or its landmark)
  → egocentric bearing (angle+distance) from current pose → goal-conditioning
  input.
- **Where:**
  - New module `decadic/state/spatial_recall.py`:
    - `resolve_goal_target(goal_id, ltm_graph, beliefs)` — need-conditioned query
      returning the goal-relevant entity (via the `predicts_*_relief` belief) and
      its last-known `position_json`; landmark fallback = highest-confidence
      co-visible neighbor when the target itself is stale/uncertain (Gap C
      robustness to a non-stationary world).
    - `egocentric_bearing(self_pos, self_yaw, target_pos)` — pure allocentric→
      egocentric transform → (cos θ, sin θ, normalized distance); zero/masked when
      no memory. **Pure math, no torch — unit-testable anywhere.**
  - `decadic/nn/goal_conditioning.py` — extend the goal vector with the masked
    bearing channels (still `GOAL_VEC_DIM`-framed, versioned layout amendment).
  - `decadic/agents/runtime.py` — call `spatial_recall` each cycle a goal is
    latched; attach the bearing to the goal vector.
- **Flag:** `DECADIC_GOAL_BEARING` (default OFF; requires 2.4 on).
- **Backup:** git branch `ws-forage/m4`; new modules are additive/low-risk.
- **Tests:** unit — transform correctness on known geometry (target dead ahead →
  θ≈0; to the left → sin θ>0; behind → |θ|≈π); mask when no memory; landmark
  fallback selection. Parity — flag-off byte-identical. Probe — "bearing points
  at the remembered resource" over a scripted recall scenario.

### 2.6 Type-2 control flow: need∧¬cue → escalate → memory search (dual-process)

- **Build:** the trigger and control flow. When a need is active AND the
  goal-resource is *not* in the current percept, raise a Type-2 escalation that
  runs the need-directed memory search (2.5) and sets its result as the active
  goal-conditioning; optionally a bounded online imagination rollout.
- **Where:**
  - `decadic/cycle/attention_gate.py` — add `type2_search` as an escalation source:
    fires on `deficit ≥ onset ∧ goal_entity_not_visible`. Logged as gate
    `reason="type2_memory_search"`. Reuses the existing gate/deliberation economics
    (expensive path, only when triggered).
  - `decadic/agents/runtime.py` — on Type-2 escalation, invoke `spatial_recall`
    (2.5) and, if `imagination_enabled`, a **bounded** online SF rollout
    (`decadic/consolidation/imagination.py`, reused; hard per-cycle step cap).
  - `decadic/config.py` — `type2_search_enabled()`, `type2_rollout_max_steps()`.
- **Flag:** `DECADIC_TYPE2_SEARCH` (default OFF; requires 2.4+2.5 on).
- **Backup:** git branch `ws-forage/m5`.
- **Tests:** unit — trigger fires only on need∧¬cue, never when the cue is visible
  or no need. Cost — per-cycle Type-2 cost bounded (measured). Parity — flag-off
  byte-identical. Behavioral probe — thirsty + resource seen-then-hidden → agent
  orients toward the remembered bearing (vs flag-off control that does not).

### 2.7 Locomotion & gaze (Gaps D/E, parallelizable, mostly existing)

- **Build:** advance the Skill Dojo curriculum toward foraging; add a *light*
  gaze-orient bias only if E proves rate-limiting.
- **Where:** `decadic/training/skills.py` (new foraging phase reusing
  `stand_teacher` → within-reach → nearby); gaze bias (deferred) would be a small
  salience-orienting term in the policy input, zero-init/flagged.
- **Flag:** curriculum config; `DECADIC_GAZE_ORIENT_BIAS` (deferred, default OFF).
- **Tests:** dojo phase unit tests (existing pattern); gaze bias parity if built.

---

## 3. Cross-cutting risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | High-γ **numerical instability** (target magnitude/variance blows up) | Med | High | (1−γ) normalization (2.2) **before** any γ change; lower λ as γ rises to lean on bootstrap; SF-loss-bounded gate in the soak; staged γ (0.995 before 0.998). |
| R2 | **Credit smearing** — long horizon dilutes which action gets credit, *weakening* the approach signal | Med | Med | Match γ to real approach durations (start ~50 s, don't overshoot); keep vector successor-features (structure-preserving); measure approach-value *discrimination* in a probe before pushing further. |
| R3 | **Byte-identical parity broken** (a new path leaks into the default agent) | Med | High | Every new path flag-gated default-OFF + zero-init (G1/G2); parity tests + `conftest.py` autouse pins for every new flag; full-suite run each milestone in both states. |
| R4 | **Philosophy violation** — accidentally hand-coding behavior or downloading world knowledge | Low | High | G3: goal vector + bearing are *inputs to a learned policy*, never scripted actions; SF weights detached (G6); the spatial map is built from lived experience (LTM), never pre-loaded; no LLM. |
| R5 | **Non-stationary world** — remembered exact location is stale (resource moved/consumed) | High | Med | Landmark fallback (2.5): navigate to remembered *context*, re-acquire by cue locally; bearing carries a distance/confidence the policy can learn to distrust. |
| R6 | **Checkpoint/bundle incompatibility** (new heads break saved agents) | Med | Med | New heads ride the versioned bundle; `NeuralBundle.load` already shape-filters (reinit-on-mismatch); add a route-level save/load round-trip test with the new flags on (WS5-M4.2 precedent). |
| R7 | **Throughput regression** from Type-2 search cost | Med | Med | Type-2 is gated (only on need∧¬cue); bounded rollout step cap; per-cycle cost measured in the soak; it is *correct* for a searching agent to cycle slower, but the cost must be bounded and off the fast path. |
| R8 | **`neural_stack.py` edit breaks the stack** (highest-risk file) | Med | High | Snapshot before edit; smallest-possible localized diffs; run the full neural suite after each edit; interface dims frozen with layout tests before wiring. |
| R9 | **Env pollution / non-reproducible A/B** | Med | Med | Clean-env diag script owns all flags; deterministic within-reach placement schedule for A/B; document every flag default. |
| R10 | **Learning never bootstraps** (no resolution episode ever forms) | Med | High | M0 within-reach curriculum guarantees early completable successes; HER + imagination (existing) densify the signal; a "did it approach a within-reach resource" probe is the gating success criterion before building M3–M5. |

---

## 4. Backup, rollback & branching strategy

- **Git first.** Commit a clean HEAD before starting; one feature branch per
  milestone (`ws-forage/m0` … `ws-forage/m7`); squash-merge to main only after
  that milestone's acceptance + full-suite green in both flag states.
- **File snapshots for high-risk edits.** Before editing `neural_stack.py`,
  `neural_pipeline.py`, `runtime.py`, copy the file to
  `reports/_backups/<file>.<utc>.bak` (belt-and-suspenders alongside git).
- **Feature-flag rollback.** Because every path is flag-gated default-OFF, an
  emergency rollback is setting the flag off — no code revert needed. Main stays
  shippable at every commit.
- **Checkpoint safety.** Before any run that could write a bad brain state, the
  existing `backup_to` snapshot is taken; the versioned bundle guarantees older
  saved agents still load (shape-filtered).

## 5. Testing strategy (three tiers, every milestone)

1. **Function (unit) tests** — one per new pure function/interface: the
   egocentric-bearing transform (known geometry), the (1−γ)-normalized returns
   (magnitude invariance), the goal-vector layout freeze, the Type-2 trigger
   truth table, the spatial-recall query + landmark fallback. Torch-free where
   possible so they run anywhere.
2. **Parity / regression tests** — with all new flags OFF, the full suite is
   byte-identical to today (autouse `conftest.py` pins each new flag to "0"). With
   flags ON at init, zero-init guarantees identical outputs until training. Full
   suite (`pytest -q`) green in both states is the merge gate.
3. **Behavioral probes (verdict scripts, gate-probe mold)** — archived under
   `reports/`:
   - *Foraging probe* — deprived agent + within-reach resource → approaches and
     consumes (flags-off ablation must fail to reliably approach).
   - *Memory-navigation probe* — resource seen then hidden while thirsty → agent
     orients toward the remembered bearing / landmark (flags-off control does not).
   - *Horizon A/B* — γ sweep: SF loss bounded, approach-value discrimination.
   - *Stability soak* — the full stack, all WS-FORAGE flags on, one embodied hour:
     no stalls, bounded per-cycle cost, throughput within noise of baseline.

## 6. Sequencing (see WBS for the milestone graph)

M0 (within-reach curriculum) and M1 (return normalization) first — low risk, and
M0 supplies the data everything else needs. M2 (γ) after M1. M3 (goal
conditioning) → M4 (bearing) → M5 (Type-2 search) is the architectural spine and
must be built in order. M6/M7 (locomotion, validation) run in parallel / at the
end. Each milestone is independently verifiable and independently revertible.
