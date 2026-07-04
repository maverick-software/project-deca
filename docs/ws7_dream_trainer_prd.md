# PRD: WS7 — The Dream Trainer (Skill Dojo, Renamed and Completed)

**Version:** 1.0 — 2026-07-04
**Status:** Draft for review
**Companions:** `ws7_dream_trainer_wbs.md` · `dream_package_sop.md` (the packaging protocol)
**Supersedes naming in:** `skill_dojo_methodology.md` (content remains valid; the dojo IS the Dream Trainer's live-practice half)

**Thesis:** one lineage lives a skill once; every mind after that dreams it. The Dream Trainer is the hub that records lived training, packages it with its curriculum and gates, and plays it back through a new mind's own learning rule at GPU speed — sleep-learning, not weight injection. Motor skill still enters through gradient steps; what the package eliminates is the exploration cost (the thousand uninformative falls). Every dreamed skill is certified AWAKE before it ships.

---

## 1. Gap analysis (2026-07-04 code inventory)

### 1.1 What exists — more than expected

| Capability | Where | State |
|---|---|---|
| Structured live practice: phases, adaptive assist-as-needed teacher, attempt lifecycle, fail-fast/timeout/retry, autonomous graduation gates | `decadic/training/supervisor.py` (`SkillDojoSupervisor`), `decadic/training/skills.py` | **Built.** The Dream Trainer's "lived" half is done. |
| Skill-as-artifact: uploadable JSON skill cards, validation, persistence, versioning guidance, `/dojo/*` REST + dashboard panel | `decadic/training/`, `docs/dojo_skills/`, `skill_dojo_methodology.md` (incl. an SOP and WBS template) | **Built.** The package format's curriculum core exists. |
| Dream frames: `Transition` carries the full cycle context (z0/ep/mem/prev_state/prev_motor/proprio+intero+effort+pain targets) AND imitation metadata (`skill_id`, `origin` self/dagger/demo, `expert_motor`, `demo_weight`, `success`) | `decadic/consolidation/replay_buffer.py` | **Built.** The unit of dreaming already exists as a dataclass. |
| Dreaming mechanism: consolidator replays transitions through the SAME learning rule; imitation loss applies iff `expert_motor` + `demo_weight > 0` | `decadic/consolidation/consolidator.py` (`ConsolidationManager`) | **Built**, but tethered to the live loop (see G3). |
| Mind imaging: full save/load of weights + state + episodic + graph, route-proven on lance+kuzu | Saved Agents library, `/checkpoint`/`/restore` (WS4-M4, WS5-M4.2) | **Built.** Clone-from-master provisioning exists. |
| Certification culture: probe/gate/verdict scripts, ablation discipline, archived artifacts | `check_gate_probe.py` mold; eval harness gates; WS2 harness | **Built** as culture; not yet a standardized battery runner. |
| Fleet muscle: cloud deployment, checkpoint shipping | `decadic/api/vast/` | **Built.** Population dreaming has infrastructure. |

### 1.2 The gaps

| # | Gap | Current state | Required |
|---|---|---|---|
| G1 | **Replay is ephemeral** | `ReplayBuffer` is in-memory, capacity-bounded, evicted by salience; a training run's experience dies with the process | **Dream Recorder**: a tap that archives the FULL transition stream of a flagged run to disk (lance table — WS4 machinery; transitions are exactly the row shape lance likes) |
| G2 | **No cross-agent replay** | Buffer is populated only by the owning agent's live cycles | **Dream loader**: ingest a recorded corpus into a DIFFERENT mind's consolidation path, behind a **compatibility contract** (preset/d_model, body SKU/actuator count, encoder mode, faculty set — transitions store post-encoder z0, so they are preset-bound; the manifest must refuse mismatches loudly) |
| G3 | **No offline dream executor** | Consolidation runs beside the live loop at a duty cycle, throttled by real time | **The Dreamer**: batch mode — take (checkpoint, corpus, schedule), run N consolidation epochs at full GPU throughput with NO world attached; deterministic, resumable, reporting loss trajectories per epoch |
| G4 | **No package format** | Skill JSON covers curriculum/teacher/gates only | **Training Package v1**: one versioned bundle = manifest (compat contract, provenance, lineage id) + skill card + scenario + dream corpus + optional graft packs + certification battery + SOP-required sample report |
| G5 | **No memory grafting** | Stores restore whole (replace), never merge | **Grafter**: merge curated episodic rows / graph entities into an existing mind's stores (id-remapped, salience-capped), then a consolidation pass integrates them — "graft, then dream about it" |
| G6 | **Certification is per-skill, ad hoc** | Dojo graduation gates + scattered probe scripts | **Certifier**: standardized battery runner — skill gates (awake, in-world) + safety invariants (threat reflex, calm, viability) + regression probes; signed verdict report per provisioned mind |
| G7 | **Dream fidelity is unmeasured** | Off-policy replay ≠ closed-loop experience; the dreaming mind's actions do not change the next frame | **Dreamed-vs-lived study** as a standing experiment: same skill via both routes, robustness under perturbation compared; result tunes each package's dream/live ratio (a knob in the manifest, not a dogma) |
| G8 | **Naming/UX** | "Skill Dojo" names only the live half | Rename to **Dream Trainer** across API/dashboard/docs with back-compat aliases (`/dojo/*` preserved; `/dream/*` canonical) |

## 2. Implementation plan (the system, end to end)

**The Dream Trainer hub** = five components around the existing supervisor:

1. **Recorder (closes G1).** `DECADIC_DREAM_RECORD=1` on any dojo/live run streams every retained `Transition` to `dreams/<run_id>/corpus.lance` via the write-behind pattern (log-and-continue; the WS3B decision-log discipline). Curation pass afterward: dedupe, salience-floor, phase-tag, keep the informative core. A recorded, curated corpus is a **Dream Recording**.
2. **Package format v1 (closes G4).** Directory bundle, zipped: `manifest.json` (name, version, provenance lineage/agent id, compat contract, dream schedule, dream/live ratio), `skill.json` (existing dojo card), `scenario/` (world files), `corpus.lance/` (the recording), `grafts/` (optional episodic/graph packs), `battery.json` (certification spec), `report_sample.md`. The Walker package is the reference implementation.
3. **The Dreamer (closes G2+G3).** `scripts/dream.py <mind> <package>`: validates compat contract → loads corpus → runs scheduled consolidation epochs (imitation + PC distillation through the existing consolidator, full GPU throughput, no websocket, no world) → emits `dream_report.json` (loss trajectories, epochs, wall time). Deterministic given (seed, corpus, checkpoint); resumable at epoch boundaries.
4. **Grafter (closes G5).** `scripts/graft.py <mind> <package>`: merges graft packs into episodic/graph stores (fresh ids, capped salience, provenance-tagged summaries), then triggers one dream pass over grafted content so it consolidates into weights — knowledge that arrives indexed AND integrated.
5. **Certifier (closes G6).** `scripts/certify.py <mind> <package|battery>`: boots the mind in the package's scenario, runs graduation gates awake (teacher assist 0 — the dojo rule stands), then the safety battery, then designated regression probes. Verdict report archived; **no dream ever substitutes for the awake gate.**

**Provisioning pipeline** (the commercial SOP, one line): clone master → graft → dream → certify → ship. Every arrow is one script above; the Saved Agents library and checkpoint routes carry the artifacts.

**Rename (G8):** `SkillDojoSupervisor` → `DreamTrainerSupervisor` (alias kept), `/dream/*` routes canonical with `/dojo/*` delegating, dashboard panel retitled, methodology doc updated to point here. Uploaded-skill compatibility unbroken.

## 3. Success criteria

1. **Record/replay round trip:** a dojo run recorded, curated, and dreamed into a FRESH clone measurably transfers skill (post-dream awake gate pass rate >> undreamed clone baseline) — the Matrix moment, quantified.
2. **Compression:** dreaming a recorded curriculum reaches gate-pass in wall-clock time an order of magnitude below the lived original (report the measured ratio; no promises before measurement).
3. **Compat contract enforced:** mismatched preset/body/encoder refuses to dream with a diagnostic, tested.
4. **Graft integration:** grafted knowledge is retrievable pre-dream and measurably better-integrated post-dream (recall-driven behavior probe), with zero corruption of pre-existing memories (store diff).
5. **Certification:** every provisioned mind carries a battery verdict; a deliberately-broken package fails certification (negative control).
6. **G7 study:** dreamed-vs-lived robustness comparison archived for the Walker package; dream/live ratio recorded in its manifest from evidence.
7. Full suite green throughout; recorder/dreamer off ⇒ byte-identical live behavior.

## 4. Risks

- **Off-policy drift (the honest physics of G7):** dreams never close the action loop; over-dreaming can overfit to the lineage's trajectory distribution. Mitigations: dream/live ratio per package, awake certification always, perturbation battery.
- **Preset-boundness of z0:** corpora do not transfer across presets. Product consequence, stated plainly: standardized body+preset SKUs are what make packages portable. Recording BELOW the encoder (raw obs streams) is a v2 option that trades storage for portability — deferred, noted in the manifest schema as `corpus_kind: "z0"|"obs"`.
- **Catastrophic interference:** dreaming skill B may erode skill A. The certifier's regression probes (criterion 5) are the guard; interleaved-corpus dreaming (mix old recordings in) is the standard mitigation, schedulable in the manifest.
- **Provenance/licensing:** recordings are a lineage's lived experience — the manifest carries provenance and the registry treats corpora as versioned, attributable assets from day one.

## 5. Dependencies and sequencing

Consumes: dojo (built), consolidator/replay (built), WS4 stores (built), Saved Agents/checkpoints (built), Vast (built). Independent of WS6 speech. Buildable NOW on synthetic scenarios (recorder/dreamer/certifier prove out on `stand_and_recover` in today's rig); the Walker reference package lands with MuJoCo locomotion scenarios. Sequencing intent: WS7 machinery can proceed in parallel with WS6 whenever build capacity allows — it touches training infrastructure, not cognition, so it does not contend with the WS5/WS6 stack work.
