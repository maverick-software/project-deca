# WBS: WS7 — The Dream Trainer

**Companion:** `ws7_dream_trainer_prd.md` · `dream_package_sop.md` · Dependency order only. ⚙ = live rig/GPU.
**Proving skill throughout:** `stand_and_recover` (existing dojo built-in) until MuJoCo locomotion scenarios exist; the Walker reference package is the graduation exercise, not the prerequisite.

---

## M0 — Recorder (G1): experience becomes an artifact

**M0.1 Transition serialization freeze.** Versioned on-disk schema for `Transition` (lance table; arrays as fixed-size lists per WS4 lessons — explicit pyarrow schema, no inference); layout test pins it.
**M0.2 Dream Recorder tap.** `DECADIC_DREAM_RECORD=1` streams every retained transition of a run to `dreams/<run_id>/corpus.lance` (write-behind, log-and-continue — the WS3B decision-log discipline). Off ⇒ byte-identical.
**M0.3 Curation pass.** `scripts/curate_dream.py`: dedupe, salience floor, phase/attempt tagging, corpus stats manifest (counts by origin/phase, salience distribution).
*Accept:* a recorded `stand_and_recover` run ⚙ round-trips disk→memory with bit-identical transitions; recorder overhead <1% cycle time; flag-off parity test.

## M1 — Package format v1 (G4)

**M1.1 Manifest schema + validator.** Compat contract (preset/d_model, actuator count, encoder mode, faculty set, `corpus_kind`), provenance (lineage/agent/run ids), dream schedule, dream/live ratio, versioning rules. `scripts/validate_package.py`.
**M1.2 Bundler.** `scripts/build_package.py`: assembles manifest + skill card + scenario + curated corpus + optional grafts + battery into one versioned zip; sample-report requirement enforced (SOP §4).
*Accept:* a `stand_and_recover` v1 package builds, validates, and a deliberately corrupted manifest/mismatched preset fails validation with a specific diagnostic.

## M2 — The Dreamer (G2+G3): offline replay executor

**M2.1 Corpus loader + compat gate.** Load recorded transitions into consolidation input for a DIFFERENT mind; refuse on contract mismatch (tested: wrong preset, wrong actuator count).
**M2.2 Batch dream executor.** `scripts/dream.py <saved-mind> <package>`: scheduled consolidation epochs at full GPU throughput, no world attached; deterministic (seed, corpus, checkpoint) and resumable at epoch boundaries; `dream_report.json` with per-epoch loss trajectories (PC, imitation).
**M2.3 ⚙ Transfer measurement (THE GATE for the whole WS).** Fresh clone + dreamed package vs undreamed control: awake graduation-gate pass rates. PRD criteria 1–2 (transfer real; compression ratio measured and reported, not promised).
*Accept:* the discriminating result both directions — dreamed ≫ undreamed on the awake gate; report archived.

## M3 — Certifier (G6)

**M3.1 Battery spec + runner.** `battery.json` (skill gates awake at assist 0, safety invariants: threat reflex / calm / viability, named regression probes); `scripts/certify.py` boots the mind in-scenario and emits a signed verdict report.
**M3.2 Negative control.** A deliberately-broken package (corrupted corpus / wrong schedule) must FAIL certification.
*Accept:* PRD criterion 5; verdict artifacts in `reports/` in the probe-culture mold.

## M4 — Grafter (G5): graft, then dream about it

**M4.1 Store merge.** Curated episodic rows / graph entities merged into an existing mind (fresh ids, capped salience, provenance-tagged summaries); zero-corruption store diff test on lance+kuzu.
**M4.2 Integration dream.** One scheduled dream pass over grafted content; pre/post recall-driven behavior probe (PRD criterion 4).
*Accept:* grafted knowledge retrievable pre-dream, better-integrated post-dream, pre-existing memories untouched.

## M5 — Provisioning pipeline + rename (G8)

**M5.1 `scripts/provision.py`:** clone master (Saved Agents) → graft → dream → certify → package the certified mind for shipping; one command, resumable, full audit trail.
**M5.2 Rename.** `DreamTrainerSupervisor` (alias `SkillDojoSupervisor` kept), `/dream/*` canonical with `/dojo/*` delegating (route tests both), dashboard panel retitled, `skill_dojo_methodology.md` header pointing here. Uploaded-skill back-compat tested.
*Accept:* end-to-end provision run ⚙ on `stand_and_recover`; suite green; old dashboard flows unbroken.

## M6 — ⚙ The Walker reference package + the G7 study

**M6.1 Walker curriculum** (MuJoCo era): staged skill card on the existing assist harness (supported stand → assisted stepping → fading support → free gait), lineage lived run recorded (M0 tap).
**M6.2 Walker package** built, dreamed into fresh clones, certified.
**M6.3 Dreamed-vs-lived robustness study** (PRD G7/criterion 6): same skill, both routes, perturbation battery; the measured dream/live ratio lands in the Walker manifest and the SOP.
*Accept:* the Walker package provisions a walking mind; the study is archived and quotable.

## Dependency graph

```
M0.1 -> M0.2 -> M0.3 -> M1.1 -> M1.2 -> M2.1 -> M2.2 -> M2.3 (GATE)
M2.3 -> M3 -> M5.1          M2.3 -> M4 -> M5.1
M5.2 (rename) parallel any time after M1
M6 requires M5.1 + MuJoCo locomotion scenarios
```

Critical path: M0 → M1 → M2.3. Everything commercial hangs off that one measured transfer result.
