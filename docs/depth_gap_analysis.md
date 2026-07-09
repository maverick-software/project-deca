# WS-DEPTH — Gap Analysis

**Goal:** the best achievable standing against all 14 indicator properties in
Butlin, Long et al. (2023) — which, after WS-EXPAND and WS-IND closed the
checkbox-level gaps (audit: 11✓/3◐/0✗), now means two different kinds of work:
closing the three remaining **partials** (all rooted in the inherited
perception front end) and **deepening the minimal passes** an auditor would
probe hardest. Companion: `depth_implementation_plan.md` (the how);
`docs/butlin_audit.md` (the scoreboard this moves).

The standing framing rule carries over from WS-IND: every item below is
justified by a concrete control/learning payoff first; the indicator it
strengthens is listed second. Nothing ships for the rubric alone.

---

## G1 — Perception is inherited, not lived  *(RPT-1 ◐, HOT-1 ◐, caps HOT-4)*

**Today.** Vision/audio enter through frozen CLIP/Whisper (`nn/frozen_encoders.py`)
— transduction the agent was born with, never reshaped by its life. Recurrence
exists only above the encoders (slot-attention iteration; the WS5 relational
core) and top-down prediction exists only at the very top of perception:
`blend_top_down` predicts z0 and blends it with the bottom-up percept under a
learned precision gate (`z0_eff = z0_hat + gate*(z0_bu − z0_hat)`,
`neural_stack.py` ~739). Below z0 — the patch-feature hierarchy — nothing is
predicted, nothing recurs, nothing learns.

**Missing.** (a) Algorithmic recurrence at the earliest lived stage: iterative
refinement over patch features before slot attention / pooling. (b) A
**generative** top-down path: predicting the patch-level features themselves
from higher latents, precision-weighted — perception as controlled
hallucination, not just a corrected feed. (c) Eventually, an encoder that
experience reshapes (the inherited transducer demoted from crutch to teacher).

**Why it matters functionally.** Occlusion/noise robustness (recurrent
refinement is what cleans a degraded glimpse), cheaper perception under
expectation (a predicted world needs less bottom-up processing), and lived
percept statistics — the entire memory system keys on percept embeddings, and
today those keys are CLIP's opinions, not the agent's.

**Risks (the reason this is a phased workstream, not a patch).** Everything
downstream keys on percept STATISTICS: episodic recall similarity, the gate's
percept-novelty channel, slot appearance fingerprints, graph entity matching.
Any change to what z0 "means" invalidates learned memories and calibrated
channels. Non-negotiable guardrails: zero-init residual refinement (percept
statistics byte-identical at birth), ramped influence, an explicit
**percept-key invariance test**, and encoder liberation (c) held behind
teacher-anchored distillation with go/no-go gates.

## G2 — Metacognition is a readout, not a capacity  *(HOT-2/HOT-3 depth)*

**Today.** `metacog_head` (`Linear(d_model, 24)`) emits element E of the State
Bus with **no objective of its own** — it is whatever the latent happens to
express. WS-IND gave it inputs (schema error tap) and the system per-slot
reliability, but nothing anywhere is *trained to know how good its own
cognition is*.

**Missing.** Calibration: supervised targets the head must earn — predict the
next cycle's own prediction error, predict the probability the current action
improves drive state, and be SCORED on it (rolling calibration telemetry, not
just loss). Trained, calibrated self-confidence is the difference between
having a metacognition head and having metacognition.

**Payoff.** Doubles as the measurement instrument for behavioral
self-awareness probes (calibration curves are exactly what those tests score),
and gives the gate/planner a principled confidence signal later.
**Risk:** target definition only (what counts as "success"); no architectural
risk — outcome-as-target, zero-init, one seam.

## G3 — Three self-models that don't talk  *(HOT-3 + AST-1 depth)*

**Today.** The agent represents itself three ways, in silos: the represented
self (`state/self_model.py`, fed back via `repself_prev`), the felt body-state
(E8.1 interoceptive embedding → affect), and the attention schema (I1 → gate
bias + policy ingress). No single self-object exists; nothing binds "what I
am" + "what I feel" + "where my attention is" into one representation that can
compete for and broadcast through the global workspace.

**Missing.** A unified self-vector (frozen layout: represented-self summary ⊕
pooled interoceptive embedding ⊕ schema prediction ⊕ metacognitive confidence)
entering the workspace as a first-class ignition candidate — the
self-model-in-the-workspace move; consciousness-of-self as *content*, not just
substrate.

**Risk.** Redundant feeding (the parts already enter via their own lanes) —
acceptable initially under zero-init; and workspace candidate plumbing (the
ignition API takes slot candidates; the self-vec must be a well-formed one).

## G4 — Sequential deliberation has no vocabulary  *(GWT-4 depth)*

**Today.** WS-IND I2 gives draft→commit: a fixed two-round program. Genuine
workspace-directed querying means the *content* of round k chooses what to
consult in round k+1.

**Missing.** A bounded query vocabulary — (episodic/graph recall, world-model
rollout, spatial plan) — and a small controller mapping workspace state → next
query, k ≤ 3 rounds, results folded back each round. Bootstrap the controller
as a fixed heuristic order, let a learned zero-init head take over on a ramp.

**Risk.** Compute (bounded: escalated cycles × k, all under the Type-2
refractory); training signal for the query chooser (mitigated by the
heuristic-first ramp); must beat plain draft/commit in the I2.2 A/B or the
vocabulary stays heuristic.

## G5 — The attention schema is gate-only  *(AST-1 depth)*

**Today.** I1's schema predicts the gate's next decision (escalate/reason/
score). It does not predict ignition CONTENT (which slots win the workspace),
and it has no model of anyone else's attention.

**Missing.** (a) A per-slot ignition-score head trained on realized ignition
winners — "what will I attend to," not just "will I deliberate." (b)
Other-attention: the same machinery run over the E10 registry's dominant
adaptive other (gaze/heading as attention proxy) — arena-gated.

## G6 — No second mind in the world  *(enabler for E10.4/E10.5, E12, G5b)*

**Today.** `embodiment/npc_controller.py` exists and the E10 registry +
other-vector lane are live, but no scenario puts a *body-pose-perceivable,
optionally adaptive* second agent in the arena. Everything social —
imitation-from-observation labeling, other-attention, the symbol/grounding
loop — is dark behind this one enabler.

**Missing.** A two-agent arena scenario: spawn a second entity with scripted
AND adaptive movement modes, observations that carry its body pose (E10.4's
labeling needs joints, not just position), dashboard control, and probe
verdicts (adaptivity gate flips; other_vec populates; inverse model labels the
demonstrator).

## G7 — Capacities are trained, lives are short  *(evidence quality, not architecture)*

**Today.** Runs are 10-minute probes and ~1-hour soaks. The newest capacities
— schema accuracy, metacognitive calibration, habit trust, code grounding —
are TRAINED properties whose audit evidence scales with lived time. No runbook
exists for a multi-hour/overnight life with checkpoint cadence, drift
monitoring, and trend verdicts.

**Missing.** Operational, not architectural: the long-life runbook + trend
verdicts (operator-kit extension), so a successor operator can run a 12–24 h
life and read out whether the trained capacities actually matured.

---

## Priority (functional payoff per unit risk)

**D1 (G2, calibration)** — cheapest, deepens two indicators, builds the
measurement instrument everything else gets scored with. → **P1/P2 (G1a/G1b)**
— the only remaining substrate gap; closes RPT-1 and HOT-1. → **D2 (G3,
unified self)** — integration of existing parts. → **D3 (G4, query
vocabulary)** — conditional on beating draft/commit. → **D4 (G5a, ignition
prediction)** → **A1 (G6, arena)** — unlocks the social tier. → **P3 (G1c,
encoder liberation)** — heaviest, gated, last. → **A2 (G7)** — continuous.
