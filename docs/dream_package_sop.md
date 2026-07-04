# SOP: Developing a Dream Training Package

**Version:** 1.0 — 2026-07-04 · **Owner process for:** WS7 Dream Trainer
**Extends:** `skill_dojo_methodology.md` §"Skill SOP" (steps 1–9 remain the LIVED-half procedure; this SOP wraps them into a shippable package)

A Training Package teaches one certified capability to any compatible mind: the lineage lives it once, every mind after dreams it. Follow this sequence for every new package. No step is optional; the sample report and negative control exist because a package that cannot fail certification cannot be trusted to pass it.

---

## 1. Define the capability (before any code)

- One sentence: what can a certified mind DO that an uncertified one cannot?
- Write the **awake graduation gate first** (dojo rule, unchanged): measurable, teacher-assist 0, in-world. Everything else scaffolds toward it.
- Declare the **compatibility contract**: body SKU / actuator count, preset, encoder mode, required faculties. This is what the package refuses to run without.
- Classify the content: **sensorimotor** (needs dreaming — weights), **declarative** (needs grafting — stores), or both. Most real packages are both (Walker = gait weights + terrain knowledge).

## 2. Build the curriculum (the lived half — dojo SOP steps 1–7)

- Skill card (phases, adaptive teacher, attempt policy, fail-fast, resets) per `skill_dojo_methodology.md`.
- Scenario files versioned inside the package, never referenced externally.
- Teacher signals remain scaffolds: they may enter replay metadata (`expert_motor`, `demo_weight`), never live cognition. Caregiver support is environmental only.

## 3. Record the lineage run

- Run the curriculum on the lineage agent with `DECADIC_DREAM_RECORD=1` until autonomous graduation.
- Curate (`curate_dream.py`): salience floor, dedupe, phase tags. Target the informative core — near-failures, corrections, transitions — not raw bulk. Record the curation parameters in the corpus manifest (they are part of provenance).
- A corpus whose curation cannot be reproduced from its manifest is not shippable.

## 4. Assemble and validate

- `build_package.py`: manifest + skill card + scenario + corpus + grafts + battery.
- Manifest MUST carry: version, provenance (lineage id, run id, curation params), compat contract, dream schedule, **dream/live ratio** (start from the nearest studied package's measured ratio; Walker's is the reference once M6.3 lands).
- Include the **sample report** from a successful provision (dream_report + certification verdict). A package ships with proof it worked at least once.

## 5. Prove transfer (the package's own M2.3)

- Dream the package into a fresh clone; run the awake gate against an undreamed control clone.
- Required: dreamed ≫ control on gate pass; record the compression ratio (dream wall-time vs lineage lived time).
- Required negative control: a corrupted variant of the package must FAIL certification. Keep the corrupted variant's verdict in the package's development records.

## 6. Certify and register

- Full battery (`certify.py`): skill gates awake at assist 0 → safety invariants (threat reflex, calm, viability management) → regression probes named in `battery.json` (at minimum: every skill the target master image already carries — catastrophic-interference guard).
- Version and register. Bump rules (from the dojo SOP, extended): any change to gates, teacher, scenario, corpus, curation params, dream schedule, or battery = version bump. Corpora are attributable lineage assets; provenance never detaches.

## 7. Field feedback loop

- Deployment incidents and hard cases return as new scenario phases or corpus additions in the NEXT package version — field-to-curriculum, never field-to-weights directly.
- Re-run step 5 on every version; transfer results append to the package's history.

---

## Package checklist (gate for release)

- [ ] Awake graduation gate defined first, teacher-assist 0
- [ ] Compat contract complete and validator-enforced
- [ ] Curriculum passes dojo SOP; lineage graduated autonomously
- [ ] Corpus curated, curation params in manifest, provenance intact
- [ ] Transfer proven vs undreamed control; compression ratio recorded
- [ ] Negative control fails certification
- [ ] Battery includes interference regression probes
- [ ] Sample report bundled; version + registry entry written
