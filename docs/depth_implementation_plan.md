# WS-DEPTH — Implementation Plan

**Companion:** `depth_gap_analysis.md` (the why), `docs/butlin_audit.md` (the
scoreboard). Dependency order only — no calendar. ⚙ = needs the live rig.

**House discipline (unchanged, suite-enforced):** default-ON with zero-init /
identity-init → birth-identical; env flag as escape hatch; conftest pins OFF;
outcome-as-target training only (never a reward a head can game); anything
expensive rides the deliberate path; anything social is solo-scene
zero-overhead; every milestone independently revertible; merge gate = unit
tests green + full-suite parity + probe archived with a machine-readable
verdict. **New guardrail for this workstream: the percept-key invariance
test** — any perception-touching milestone must show episodic keys, gate
novelty, and slot fingerprints statistically unchanged at init and drifting
only under the ramp.

---

## D1 — Metacognitive calibration  *(G2; HOT-2/3 depth; build FIRST)*

*The measurement instrument for everything else, and the cheapest milestone.*

- **D1.1 Targets.** Two supervised heads (or a widened metacog objective) on
  the detached latent: (a) predicted next-cycle `pc_loss` (regression — "how
  surprised am I about to be"), (b) P(drive improves | current action)
  (binary — "is this working"), scored against realized outcomes one cycle
  later via the standard prev-buffer pattern (the FEL/schema seam).
- **D1.2 Calibration telemetry.** Rolling reliability: mean |predicted −
  realized| for (a); binned calibration error for (b) (predicted-probability
  bins vs realized frequency — an ECE-style scalar, `metacog_calibration`).
  Exported through the metrics allowlist (the standing WS5-M5 rule).
- **D1.3 Metacog conditioning.** The calibrated predictions feed the
  metacognition head's input (zero-init fold), so element E carries *earned*
  self-assessment; the confidence scalar joins the D2 self-vector.
- *Accept:* zero-init parity; on synthetic regimes the heads calibrate (a
  noisy-but-stationary loss stream yields honest wide errors, a predictable
  one converges); calibration telemetry finite and improving on recorded runs.
- *Probe ⚙:* `metacog_calibration` trends downward over a 10-min run.

## P1 — Recurrent percept refinement  *(G1a; RPT-1)*

- **P1.1** `nn/percept_refine.py`: a small refinement cell over the patch-
  feature map (2–3 iterations; GRU-style or tied-weights conv), inserted
  between the frozen encoder output and slot attention / pooling. **Zero-init
  residual** (`feat + refine(feat)`, output layer zeroed) → percept statistics
  byte-identical at birth. Trained by next-frame patch prediction error
  (self-supervised, already-available consecutive frames).
- **P1.2** Percept-key invariance test (the new guardrail, first exercised
  here): with the flag ON at init, episodic keys / novelty channel / slot
  fingerprints match the OFF path exactly; under a trained refiner, drift is
  bounded and monotone-beneficial (recall hit-rate must not degrade).
- *Accept:* parity + invariance; refinement demonstrably denoises (synthetic
  occlusion/noise test: refined features closer to clean-frame features).
- *Probe ⚙:* discovered-mode object-binding quality (existing WS5 metrics)
  with refiner on vs off.

## P2 — Generative top-down perception  *(G1b; closes HOT-1)*

- **P2.1** Extend the existing `blend_top_down` pattern one level DOWN:
  a top-down head predicts the pooled patch summary (and optionally the K
  most salient patch features) from the prior cycle's z5/context; blended
  with bottom-up under the same learned precision-gate form already validated
  at z0. Zero-init prediction + gate init to pass-through → parity.
- **P2.2** Precision learning: gate trained by the realized prediction error
  (precise where the world is predictable, bottom-up where it surprises) —
  the same volatility/noise separation logic as E2, at the feature level.
- **P2.3** Telemetry: fraction of percept carried top-down
  (`percept_topdown_frac`) — the "controlled hallucination" dial, watched
  live; hard cap so perception can never fully decouple from the world.
- *Accept:* parity; invariance test; on predictable synthetic streams the
  top-down fraction rises and per-cycle encoder compute can be skipped
  (P2's functional payoff) without behavior change; cap enforced.
- *Probe ⚙:* topdown_frac climbs on repetitive patrol, collapses on novelty.

## D2 — Unified self-model in the workspace  *(G3; HOT-3 + AST-1 depth)*

- **D2.1** `SELF_VEC` frozen layout in `state/self_model.py`: represented-self
  summary ⊕ pooled interoceptive embedding (E8.1) ⊕ attention-schema
  prediction (I1) ⊕ metacognitive confidence (D1). Pure builder + layout test.
- **D2.2** Workspace candidacy: the self-vec projected (zero-init) into a
  well-formed ignition candidate beside the percept slots — the self competes
  for conscious access like any content; when it wins, it broadcasts.
- **D2.3** Telemetry: `self_ignition_rate` — how often the self-model wins the
  workspace (interoceptive crises should spike it; calm foraging should not).
- *Accept:* zero-init parity (candidate never wins at birth — projection zero
  → below ignition threshold); layout freeze; ignition-rate telemetry.
- *Probe ⚙:* deprivation spike → self-ignition rate rises (the agent's
  attention turns inward when its body is the most newsworthy thing).

## D3 — Query-vocabulary deliberation  *(G4; GWT-4 depth; conditional)*

- **D3.1** Query executors, uniform interface: `recall` (episodic/graph read —
  exists), `rollout` (E1.6 planner machinery — exists), `plan` (cmap waypoint —
  exists). Each returns a fixed-width result token.
- **D3.2** Round controller: bootstrap = fixed heuristic order (recall →
  rollout), k ≤ 3, escalated cycles only; a zero-init head over workspace
  state chooses the next query on a ramp (heuristic → learned, the
  assist-fade pattern).
- **D3.3 ⚙ A/B (decides its fate):** detour-then-forage vs plain I2
  draft/commit. Keep the learned chooser only if it wins; keep k-round
  querying at all only if IT beats two-round.
- *Accept:* parity; bounded rounds; per-round telemetry (query chosen, content
  delta); the A/B verdict archived.

## D4 — Ignition-content prediction  *(G5a; AST-1 depth)*

- **D4.1** Per-slot ignition score head (slot feature + gate state → score),
  trained on realized ignition winners (outcome-as-target, the I1 pattern);
  prediction accuracy telemetry vs a salience-ranking baseline.
- **D4.2** The predicted-ignition summary joins the schema vector (layout
  grows; the zero-pad migration pattern covers checkpoints).
- **D4.3** (arena-gated) Other-attention: same head shape over the dominant
  adaptive other's heading/gaze proxy from the E10 registry.
- *Accept:* parity; beats the salience baseline on recorded runs.

## A1 — Two-agent arena  *(G6; the social enabler)*

- **A1.1** Adapter scenario: spawn a second embodied entity,
  `npc_controller`-driven, with `scripted` (patrol/ballistic) and `adaptive`
  (reactive pursuit/avoid) modes; body pose (joints) included in the primary
  agent's observations of it.
- **A1.2** Dashboard: spawn/despawn + mode toggle; diag warns are updated (a
  SECOND BRAIN is still a confound — this entity is a body, not a second
  cognition server).
- **A1.3 ⚙ Probe:** scripted mode → registry tracks, `other_models_active`
  stays 0 (ballistic prior wins); adaptive mode → gate flips within warmup,
  other_vec populates, E10.4 inverse model labels the demonstrator's
  transitions (label error telemetry) — the full E10.3–E10.5 verdict in one
  run. G5b and E12 both unblock here.

## P3 — Encoder liberation  *(G1c; heaviest, gated, last)*

- **P3.1** Offline distillation: a trainable student encoder distilled from
  CLIP on the agent's OWN recorded frames (consolidation-side, off the hot
  path). Go/no-go: student within ε of teacher on percept-key retrieval over
  the agent's episodic store.
- **P3.2** Swap behind `DECADIC_ENCODER_MODE=student`; teacher-anchored
  training (KD term keeps drift bounded) + slow lr + the invariance test as a
  standing canary. Rollback is one env var.
- **P3.3** Progressive experience-weighting: the KD anchor decays as lived
  prediction-error training earns trust — the inherited transducer becomes a
  teacher, then a memory.
- *Accept per stage:* retrieval parity gates; NO stage ships without the
  percept-key invariance canary green.

## A2 — Long-life operations  *(G7; continuous)*

- **A2.1** `scripts/run_long_life.ps1`: 12–24 h run, checkpoint cadence,
  periodic metrics snapshots, and TREND verdicts (schema accuracy ↑,
  metacog_calibration ↓, habit trust vs divergence, code utilization ↑,
  topdown_frac behavior, retention health) — the operator-kit extension so a
  successor operator reads maturation, not just liveness.
- **A2.2** The trained-capacity audit addendum: after the first long life,
  update `butlin_audit.md` evidence columns from "wired" to "matured (n hours
  lived)".

---

## Dependency graph & order

```
D1 ─→ D2 (confidence joins the self-vec)
P1 ─→ P2 ─→ P3 (each gated on the invariance canary)
I1 ─→ D4 ─→ (arena) D4.3
E1.6/cmap ─→ D3 (query executors exist)
A1 ─→ E10.4 application, E12, D4.3
A2 after the first D1/P1 land (something to trend)
```

**Build order: D1 → P1 → P2 → D2 → D3 → D4 → A1 → P3 → A2(continuous).**
Rationale: D1 is the cheapest and instruments the rest; P1/P2 are the only
remaining substrate gap (RPT-1, HOT-1 — the last two structural ◐s); D2 is
integration of parts that all exist after D1; D3/D4 are conditional
deepenings with built-in A/Bs; A1 unlocks the entire social tier; P3 is
deliberately last — it carries the only genuinely irreversible risk class
(perceptual drift against learned memories) and deserves a mature invariance
canary before it runs.

## Standing risks

Percept-statistic drift breaking memory keys (invariance canary, every P
milestone) · top-down decoupling from the world (hard cap on topdown_frac) ·
self-vec double-feeding (zero-init, replace-vs-augment reviewed at D2 A/B) ·
query-vocabulary compute (k ≤ 3, refractory-bounded, A/B-gated) · second-agent
run confounds (A1.2 diag warnings) · encoder swap invalidating a lived past
(P3 teacher anchor + one-var rollback).
