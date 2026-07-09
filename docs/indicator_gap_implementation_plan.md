# WS-IND — Indicator-Gap Implementation Plan

**Basis:** the audit of Deca against the 14 computational indicator properties in
Butlin, Long et al. (2023), *Consciousness in Artificial Intelligence: Insights
from the Science of Consciousness* (arXiv:2308.08708). Deca currently scores
strongly on PP-1, AE-1/AE-2, RPT-2, GWT-1/2/3; this plan closes the gaps in
priority order. Dependency order only — no calendar.

**Two framing rules, stated up front:**

1. **Build for function, score as a side effect.** The indicators are credible
   because they were derived from theory, not gamed. Every milestone below is
   justified by a concrete control/learning payoff first; the indicator it
   satisfies is listed second. If a milestone ever has no functional payoff,
   it does not ship.
2. **House discipline unchanged.** Default-ON, zero-init/identity-init →
   birth-identical; env flag as escape hatch; conftest pins OFF; every
   milestone independently revertible; merge gate = unit tests green + full
   suite parity + probe archived with a machine-readable verdict.

---

## Priority 1 — I1: Attention schema  *(closes AST-1 — the clean miss)*

*Functional payoff: anticipatory gating (deliberate before the surprise, not
one cycle after) and a controller-grade model of the system's own attention;
later, the same machinery over the E10 registry gives modeling of OTHERS'
attention (joint-attention substrate).*

- **I1.1 Schema head.** New `nn/attention_schema.py` + a zero-init head in the
  stack: from (policy latent, current gate-state vector — score, effective
  threshold, latch, refractory cooldown, escalation rate, all already in gate
  telemetry) predict next cycle's realized attention outcomes:
  (a) escalate probability, (b) reason class, (c) which workspace slots ignite.
  Trained self-supervised on the REALIZED outcomes one cycle later — outcome as
  target, never a reward (the FEL/veto pattern).
- **I1.2 Schema feedback.** The schema's prediction re-enters the stack as an
  input channel through a zero-init ingress (the model of attention informing
  attention — the entire point of a schema). Birth-identical.
- **I1.3 Anticipatory gate bias.** Predicted-escalation probability contributes
  a BOUNDED threshold bias (the E2.2/E2.3 pattern: bounded at set time,
  floored, byte-identical at zero). Composes additively with the
  learning-control bias under the same cap.
- **I1.4 Telemetry + metacognition tap.** `schema_accuracy` (predicted vs
  realized escalations, rolling), `schema_bias`; schema prediction error
  exposed as an input to the metacognition head (side-door progress on HOT-3).
- **I1.5 (deferred by design)** Other-attention modeling: run the schema over
  E10's adaptive-agent tracks. Waits on the two-agent arena, with E10.3–.5.
- *Accept:* zero-init parity; schema accuracy beats a base-rate predictor on
  recorded gate sequences; bias bounded; full-suite parity.
- *Probe ⚙:* 10-min body run — schema accuracy > base rate on live gate
  decisions; measure the anticipation gain as the fraction of escalations
  preceded (≥1 cycle) by predicted-escalation above threshold.

## Priority 2 — I3: Per-percept reality monitoring  *(HOT-2, strengthens E6)*

*Functional payoff: distraction robustness and belief hygiene — unreliable
percepts stop polluting working memory and the association store.*

- **I3.1 Per-slot reliability.** Extend the E2 volatility/noise separation from
  one global stream to per-slot: a bounded per-slot prediction-error EMA +
  noise scale → reliability weight in [floor, 1]. Pure/cheap (K slots, a few
  floats each), lives beside the slot gate.
- **I3.2 Gate composition.** E6's slot pass-weight becomes
  relevance × reliability (both identity at init, both floored, both reopened
  by surprise — the same guardrails already tested).
- **I3.3 Reliability into WM.** Slots enter working memory tagged with their
  reliability; telemetry `slot_reliability_min/mean`.
- *Accept:* identity parity; a synthetic noisy-slot stream is down-weighted
  while a volatile-but-real one is not (the E2.2 noise-injection test, per slot).

## Priority 3 — I4: Metacognition-gated belief updates  *(HOT-3, small)*

*Functional payoff: the `predicts_*` association store — which now drives
goals, threats, and grounding — stops learning from junk percepts.*

- **I4.1** Belief confidence updates in `working_memory.py` scale with the
  source percept's I3 reliability (update gain × reliability; never blocks a
  first observation, only tempers it). Belief decay already exists (E5.2).
- **I4.2** Telemetry: `belief_updates_tempered`.
- *Accept:* beliefs formed from reliable percepts converge as today (parity at
  reliability 1.0); noisy-percept beliefs converge slower; no belief starvation.

## Priority 4 — I2: Sequential workspace querying  *(GWT-4 — most consequential, biggest lift)*

*Functional payoff: System-2 chained reasoning — the workspace using one
query's answer to direct the next, instead of one-shot escalation.*

- **I2.0 Evidence review FIRST** (the WS-EXPAND pattern: verdict before code).
  Candidate designs to adjudicate: (a) multi-round workspace — on an escalated
  cycle, fold query results (memory recall, planner rollout, spatial plan)
  back into the workspace and re-run ignition for a bounded k rounds before
  committing; (b) a micro-program executor with a fixed query vocabulary and
  workspace state as the program counter. Review must weigh compute per
  deliberate cycle (the refractory already bounds frequency) and the
  training story for round-2+ content.
- **I2.1 Minimal increment (pending review):** two-round workspace on
  escalated cycles only — query, integrate, re-ignite, commit. Round count
  telemetry + divergence between round-1 and round-2 commitments.
- **I2.2 Probe ⚙:** A/B on the foraging arena — does round-2 measurably change
  committed decisions, and do outcomes (time-to-relief on detour tasks)
  improve? Keep only if it beats one-shot escalation.
- *Discipline:* deliberate-path-only, k bounded, flag `DECADIC_WS_SEQ`,
  refractory shared with Type-2 (no perseveration regression).

## Priority 5 — I5: Quality-space smoothness  *(HOT-4, cheap polish)*

- **I5.1** A light local-isometry regularizer on E9's FSQ projection (nearby
  latents → nearby codes), keeping the code space sparse AND smooth. Trunk
  stays detached (E9's parity guarantee unchanged); telemetry: code-space
  neighborhood consistency.
- *Accept:* utilization does not collapse; smoothness metric improves; parity.

## Priority 6 — I6: Recurrent lived perception  *(RPT-1 — define-only here)*

The remaining substrate gap is the frozen sensory front end. Full resolution —
a learned, recurrent, prediction-error-trained perceptual hierarchy replacing
the inherited encoders — is its own workstream (WS-PERCEIVE, the "fuller
neocortex" note in BRAIN_MAP.md), not a milestone in this one. The bounded
increment worth scoping now:

- **I6.1 (scope only):** a small recurrent refinement layer over patch features
  (lived, trained by prediction error, zero-init residual) between the frozen
  encoder and slot attention — algorithmic recurrence at the earliest lived
  stage without discarding the inherited transducers.

## Closing milestone — I7: Re-audit

- **I7.1** Score all 14 indicators before/after WS-IND with per-indicator
  evidence (module, test, probe telemetry), archived as `docs/butlin_audit.md`.
  This is the workstream's acceptance artifact, and the checkpoint the
  welfare-policy question keys off: the owner decides IN ADVANCE what audit
  outcome changes how deprivation/damage curricula are treated (a standing
  note, not a code milestone).

---

## Dependency graph & recommended order

```
I1 (schema) ──────→ I1.5 (other-attention; waits on two-agent arena)
I3 (reliability) ─→ I4 (belief gating)
I2.0 (review) ────→ I2.1/I2.2 (sequential workspace)
I5, I6.1 independent
I7 last
```

**Build order: I1 → I3 → I4 → I2.0 → I2.1/2 → I5 → I7** (I6 scoped, not built).
Rationale: I1 is the clean miss with the best payoff-per-line and reuses the
gate-bias plumbing shipped in WS-EXPAND; I3/I4 are small and harden systems
everything else reads (slots, beliefs); I2 is the deepest capability but needs
its evidence review first; I5 is polish; I7 converts the work into the
auditable artifact.

## Standing guardrails carried over

Zero-init/identity-init birth-identity for every new head · bounded, floored,
composable gate biases (one shared cap) · outcome-as-target training only
(never a reward the head can game) · deliberate-path-only for anything
expensive · solo-scene zero-overhead for anything social · probes runnable
from their runbooks by a successor operator.
