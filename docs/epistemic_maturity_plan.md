# Epistemic Maturity — Implementation Plan

**Principle:** replace the fixed perception-confidence gate with **Bayesian precision-weighting**. A young model accepts almost everything; skepticism grows with corroborated evidence so a mature model protects what it knows — *without* ever fully ossifying.

**Fixes:** the cold-start failure (`skipped_low_confidence` → graph never seeds) *and* implements the developmental skepticism you described.

**Key files touched:** `decadic/perception/object_files.py`, `decadic/state/working_memory.py`, `decadic/perception/scene_workspace.py`, `decadic/cycle/stages/stage_10.py`, `decadic/cycle/neural_pipeline.py`, `decadic/memory/semantic_graph.py`, `decadic/state/state_bus.py`, `decadic/config.py`, **new** `decadic/state/epistemic.py`.

---

## 1. Design summary

Three coupled mechanisms plus three guardrails.

**(a) Provisional entry (the inversion).** Stop gating *entry* on confidence. Every segmented percept above a tiny floor enters working memory as a *provisional* entity. The gate moves downstream to **promotion** (becoming a graph node / consolidating to LTM), which is precision-weighted.

**(b) Per-entity precision `P_e`.** Each entity carries a precision in `[0,1]`, initialized low. Consistent re-sighting raises it (`P_e ← P_e + η·(1−P_e)`); contradiction erodes it and accrues a contradiction counter. This is local confidence earned over time, replacing "presence ≥ 0.2 from one frame."

**(c) Global epistemic maturity `S`.** One scalar in `[0, S_max]`, `S_max < 1`. Grows saturating with *corroborated* evidence (LTM size, age, total consolidated sightings): `S = S_max·(1 − exp(−k·E))`. `S` modulates how much corroboration a new entity needs to be promoted, how stubbornly established beliefs resist revision, and (optionally) global plasticity. Crucially `S` is **not** monotonic with age — it *decays under model failure* (recalibration), so a confidently-wrong model re-opens.

**Guardrails against unbounded skepticism** (see §4): a hard plasticity floor, a per-belief contradiction override, and a global regime-change re-opening.

This is the precision/Kalman-gain story your predictive-coding stack already implies: new-evidence weight = `evidence_precision / (evidence_precision + prior_precision)`; as `S` and `P_e` raise prior precision, the gain shrinks — that *is* skepticism, and it emerges rather than being scheduled.

---

## 2. Where it lives in the codebase

| Concept | Home | Change |
|---|---|---|
| Provisional entry | `object_files.py` (`evaluate_discovery_health`, `LOW_CONFIDENCE_THRESHOLD`), `neural_pipeline.py` discovery block | Lower the *entry* floor toward `slot_presence_threshold`; stop treating `< 0.2` as "skip." Keep "stuff" penalty (anti-floor). |
| Per-entity precision `P_e` | `working_memory.py` slot dataclass + `integrate_discovered`; mirror on `scene_workspace.SceneEntity` | Add `precision`, `contradiction_pressure`; update on match/miss. Reuse existing `seen_count`/`persistence`. |
| Promotion gate | `stage_10.py` (`seen_count ≥ ltm_consolidate_min_seen`), `semantic_graph.consolidate` | Replace constant with `P_e ≥ promote_threshold(S)` (equivalently `effective_min_seen = base + round(S·extra)`). |
| Belief protection | `working_memory.integrate_discovered` (`appearance_ema`), affect EMAs | Scale effective update rate by `(1 − S·P_e)`, floored. |
| Global `S` + recalibration | **new** `decadic/state/epistemic.py` (`EpistemicState`); surfaced via State Bus **E** (metacognition) | Holds `S`, evidence accumulators, failure EMA, re-open logic. **Checkpointed** (maturity is part of the agent's identity). |
| Plasticity annealing (optional) | `neural_pipeline.apply_plasticity_step`, `bundle.optimizer` LR | Scale Hebbian η / LR by `(1 − λ·S)` with a floor. |

---

## 3. Work breakdown (phased; effort in dev-days)

### Phase 0 — Flag, state, baseline (1–2 d)
- **0.1** Add `decadic/state/epistemic.py` with `EpistemicState` (fields: `maturity`, `evidence`, `failure_ema`, counters). Add `DECADIC_EPISTEMIC_MATURITY_ENABLED` (default **off** → byte-identical baseline) and the constants in §6.
- **0.2** Decide rollout: the **provisional-entry** half (Phase 1–2) is also the cold-start fix; consider gating it on its own flag so it can ship first. *Acceptance:* with the flag off, the cycle is byte-identical to today (parity test).

### Phase 1 — Per-entity precision (3–4 d) [0]
- **1.1** Add `precision` + `contradiction_pressure` to the working-memory slot and `SceneEntity`. Initialize provisional (`precision = 0.05`).
- **1.2** In `integrate_discovered`: on a confident re-match, `precision ← precision + η_p·(1−precision)`; on a mismatch/contradiction, erode precision and increment `contradiction_pressure`. *Acceptance:* a repeatedly-seen synthetic object's precision climbs to ≈1; a one-frame blip stays low and is evicted.

### Phase 2 — Provisional entry + precision-gated promotion (4–5 d) [1]
- **2.1** Lower the entry gate: in `evaluate_discovery_health` / the discovery block, accept proposals down to the proposal floor (no hard `0.2`), tagging them provisional. Keep the "stuff" spread penalty so the floor texture still can't flood memory.
- **2.2** Replace `stage_10`'s `seen_count ≥ ltm_consolidate_min_seen` with `precision ≥ promote_threshold(S)` (early `S` → ~1–2 sightings promote; mature `S` → many). *Acceptance:* on a fresh agent the self-indexed graph and LTM start growing within minutes; `skipped_low_confidence` no longer permanently blocks seeding.

### Phase 3 — Global maturity scalar `S` (3–4 d) [1,2]
- **3.1** Compute `evidence E` from `ltm_graph` node count + `log(1+age_cycles)` + total consolidated sightings; `maturity S = S_max·(1−exp(−k·E))`. Update once per cycle; surface into State Bus **E** and metrics.
- **3.2** Persist `EpistemicState` in checkpoints / saved agents (load restores maturity). *Acceptance:* `S` rises fast early, plateaus near `S_max`; reload preserves it.

### Phase 4 — Wire `S` to promotion, protection, plasticity (3–4 d) [2,3]
- **4.1** `promote_threshold(S)` and the protection update-rate `(1 − S·P_e)` consume live `S`.
- **4.2** (Optional) anneal Hebbian η / Adam LR by `(1 − λ·S)` with a floor in `apply_plasticity_step`. *Acceptance:* a mature agent measurably resists overwriting an established entity from a single off-frame, while a young one updates readily.

### Phase 5 — Anti-ossification guardrails (4–5 d) [3,4] — **required**
See §4. *Acceptance:* the three guardrail tests in §5 pass.

### Phase 6 — Telemetry, tests, dashboard (3–4 d) [1–5]
- Metrics: `epistemic_maturity`, `evidence`, `mean_entity_precision`, `promotions`, `overrides`, `reopen_events`, `failure_ema`. Add a small panel / number to the Cognition tab.
- Tests in §5. *Acceptance:* all green; baseline parity preserved with the flag off.

**Indicative total:** ~18–24 dev-days. Phases 1–2 deliver the cold-start fix and the credulous-young behavior; Phases 3–5 deliver developmental skepticism *with* the safety valves.

---

## 4. Protection against unbounded skepticism (the explicit requirement)

Three layers, so the model can always still learn:

**G1 — Hard plasticity floor.** Cap `S ≤ S_max` with `S_max = 0.85` (never 1.0). Floor every `S`-scaled rate: the protection update-rate `(1 − S·P_e)` is clamped `≥ min_update_rate` (e.g. 0.05); if LR/η annealing is used, it floors at a fraction of base. **Nothing is ever fully frozen** — a mature belief still moves, just slowly.

**G2 — Per-belief contradiction override.** Each entity accrues `contradiction_pressure` when *repeated, high-precision* observations disagree with it (not single blips). When it exceeds `override_threshold`, force-revise: slash that entity's precision and re-open it to update (or demote it from the graph). This is "sustained strong evidence beats even a deep belief," and it is **independent of `S`** — maturity cannot veto a persistent, well-supported contradiction.

**G3 — Global regime-change re-opening + recalibration.** Maintain `failure_ema` = the running rate at which *established* (high-`P_e`) beliefs' predictions fail. (a) **Recalibration:** `S` grows on corroborated evidence but is multiplied down by `failure_ema`, so a confidently-wrong model becomes *less* skeptical automatically. (b) **Re-opening:** if `failure_ema` spikes past `reopen_threshold` (the world changed), apply a one-shot `S ← S·reopen_decay` to re-enter a plasticity window — a synthetic critical period. *Acceptance:* after a simulated environment swap, `S` drops, the model re-learns, then `S` re-climbs.

Net effect: skepticism rises with *successful* experience, is capped, can be punched through locally by sustained contradiction, and globally collapses when the model is failing — exactly the human pattern, minus the failure mode of getting more certain and more wrong with age.

---

## 5. Acceptance criteria & tests

| Behavior | Test | Pass condition |
|---|---|---|
| Cold start unblocked | `test_provisional_entry_seeds_graph` | fresh agent grows graph nodes from low-confidence percepts within N cycles; `information_loss` unaffected. |
| Precision accrues | `test_entity_precision_grows_with_resighting` | repeated object → `P_e → ~1`; blip → evicted. |
| Skepticism emerges | `test_maturity_rises_and_protects` | `S` climbs with evidence; mature agent needs more sightings to promote; established belief resists a single off-frame. |
| **Floor (G1)** | `test_plasticity_floor` | with `S = S_max`, update-rate ≥ `min_update_rate`; LR/η ≥ floor. |
| **Override (G2)** | `test_sustained_contradiction_overrides_maturity` | a mature, high-`P_e` belief *is* revised after K consistent contradictions, regardless of `S`. |
| **Re-open (G3)** | `test_regime_change_reopens_learning` | after a feature-distribution swap, `failure_ema` spikes, `S` drops, model re-learns, `S` recovers. |
| Parity | `test_epistemic_flag_off_is_baseline` | flag off → byte-identical to current pipeline. |

---

## 6. Config knobs (`config.py`, all with safe defaults)

- `DECADIC_EPISTEMIC_MATURITY_ENABLED` (default off initially; on after validation)
- `DECADIC_PROVISIONAL_ENTRY_ENABLED` (cold-start fix; may ship first)
- `DECADIC_EPISTEMIC_S_MAX = 0.85` (hard skepticism ceiling — **G1**)
- `DECADIC_EPISTEMIC_K` (evidence→maturity growth rate)
- `DECADIC_ENTITY_PRECISION_ETA` (per-sighting precision accrual)
- `DECADIC_PROMOTE_BASE_SEEN = 2`, `DECADIC_PROMOTE_MATURE_EXTRA` (promotion bar; `effective = base + round(S·extra)`)
- `DECADIC_MIN_UPDATE_RATE = 0.05` (protection floor — **G1**)
- `DECADIC_CONTRADICTION_OVERRIDE_THRESHOLD` (**G2**)
- `DECADIC_REOPEN_THRESHOLD`, `DECADIC_REOPEN_DECAY` (regime-change — **G3**)
- `DECADIC_PLASTICITY_MATURITY_ANNEAL` (optional, default 0 = off)

---

## 7. Risks & tradeoffs

- **Noise flooding.** Provisional entry only works because uncorroborated entities decay fast — keep `scene_workspace`/`working_memory` decay + eviction aggressive; tie decay rate to `(1 − precision)` so junk fades and corroborated persists. Verify `mean_entity_precision` stays meaningful (not dominated by transient junk).
- **Tuning coupling.** `k`, `S_max`, the promotion bar, and the decay rates interact. Land Phases 1–2 first and tune them in isolation before enabling `S` (Phase 3+).
- **Over-eager re-opening.** If `reopen_threshold` is too low, normal surprise re-opens the model constantly (no stability). Calibrate against a steady-state run's baseline `failure_ema`.
- **Checkpoint compatibility.** Persisting `EpistemicState` changes the saved-agent schema; version it and default missing fields to a young state on load.
- **Honest limit.** This makes the *gating* developmentally correct, but the graph still only grows once slot attention actually learns to segment (the training-rate issue). This plan unblocks the cold start and removes the permanent gate; it does not substitute for giving the perception modules enough gradient steps.
