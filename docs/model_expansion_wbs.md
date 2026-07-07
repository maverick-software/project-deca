# WBS — Capability Expansion (WS-EXPAND)

**Companions:** `model_expansion_integration.md` (where/how each piece hooks in),
`model_expansion_evidence_review.md` (the evidence verdicts this ordering follows).
Dependency order only — no calendar. ⚙ = needs the live rig (GPU/body).

**House discipline (all milestones):** ships **default-ON** with zero-init /
identity-init so a fresh agent is birth-identical at cycle 0; env flag is the
escape hatch and A/B lever, never the safety story; conftest pins params for test
determinism (prod-on, tests-pinned). Every milestone independently revertible.

**Merge gate (all milestones):** unit tests green · full `pytest -q` green with
flag OFF (parity) *and* ON-at-init (zero-init parity) · milestone probe archived
under `reports/` with a machine-readable verdict (operator-kit rule: every probe
runnable from its runbook without this author).

**Cross-cutting rule from the evidence review:** anything marked A/B ships
flag-gated with an explicit accept/reject metric — it is *kept* only if it wins.

---

## E1 — Spatial state estimation + planning  *(BUILD — highest leverage)*

*Completes navigation; direct consumer of the WS-FORAGE spine. Flag
`DECADIC_COGNITIVE_MAP` (+ `DECADIC_PLANNER` for E1.6).*

- **E1.1** `state/cognitive_map.py`: dead-reckoning pose — integrate
  proprioceptive velocity + yaw into a running world pose each cycle (beside
  `_current_goal_vec` in `runtime.py`). Pure-math unit tests on synthetic
  trajectories; documented drift bound.
- **E1.2** Landmark correction: on re-sighting an entity whose graph position is
  known, blend the pose toward the implied estimate (complementary filter).
  Test: injected drift converges after k re-sightings.
- **E1.3** Positional code: extend the goal-vector layout (new frozen slots
  beyond [7]; layout-freeze test updated; checkpoint load stays shape-filtered)
  carrying a small learned pose embedding through the existing zero-init
  `goal_ingress`. Self-supervised next-pose prediction objective.
- **E1.4** Spatial adjacency graph: derive nodes/edges from `scene_near`/scene
  edges + observed geometry. **Edge weights = measured traversal cost** (not raw
  connectivity — evidence: unweighted/spurious edges corrupt plans); confidence
  threshold prunes aliased edges; retention/pruning on the existing graph
  governance rails.
- **E1.5** Waypoint planner: A* over E1.4; `_goal_bearing` (one-line change)
  points at the *next waypoint toward* the target; falls back to straight-line
  bearing when no graph path exists.
- **E1.6** Online rollout selection (deliberate path only): sample K short
  action sequences, roll the forward models, score tails by successor-feature
  value, **bias** `motor_u` (bounded additive term, never override). Short
  horizon + value truncation are load-bearing (compounding-error control).
  Reuses `imagination.py` rollout code. Compute budget: only on escalated
  cycles, already refractory-bounded.
- *Accept:* parity gates; drift bound + correction tests; planner returns
  straight-line when graph empty (byte-identical behavior); rollout bias bounded
  and OFF-cycle-free.
- *Probe ⚙:* **detour task** — obstacle between deprived agent and remembered
  resource; success = reaches around it; compare straight-line-bearing baseline.

## E2 — Multi-channel learning control  *(BUILD WITH GUARDRAILS — do early; E4/E6 consume its signals)*

*One seam: `neural_pipeline.py` modulation calc + `plastic.py::hebbian_update`.
Flag `DECADIC_LEARN_CONTROL_MULTI`.*

- **E2.1** Vectorize the seam: `hebbian_update` takes a 4-vector; at init all
  channels map to the current scalar → **byte-identical parity test** is the
  first test written.
- **E2.2** Expected-uncertainty channel → scales `eta` + biases gate threshold.
  **Volatility/noise separation is mandatory** (evidence: naive surprise→LR
  chases noise): trend of `pc_slope_ema` = volatility (raise LR), variance of
  `pc_ema` under stable slope = noise (lower LR). Unit test: injected pure
  observation noise must *not* raise the channel.
- **E2.3** Surprise channel → transient `eta` boost (decaying) + raises gate
  escalation propensity. Exported: E4 staleness guard and E6 gate-reopen consume
  this signal.
- **E2.4** Horizon channel — **the flagged risk**: viability trend modulates SF
  γ *inside a hard clamp band* (e.g. [0.99, 0.997]), rate-limited per N cycles,
  every move logged to telemetry, env kill-switch pins γ. Rides M1's (1−γ)
  return normalization so value magnitude stays γ-invariant.
- *Accept:* init-parity byte-identical; noise-injection test; γ never observed
  outside clamp in soak telemetry.
- *Probe ⚙:* A/B soak vs scalar baseline — SF loss bounded, `successor_value`
  finite/discriminative, no value blow-up during a lucky feeding streak (the
  documented meta-gradient failure).

## E3 — Motor corrector + phase timing  *(BUILD — best-evidenced item; unblocks locomotion)*

*`neural_stack.py` after the motor head. Flag `DECADIC_MOTOR_CORRECTOR`.*

- **E3.1** Zero-init corrector: `motor_u += tanh(motor_corrector(motor_u,
  proprio))`, trained **supervised** on the proprioceptive forward-model error —
  the error is a target, never a reward (evidence: reward-framing lets the
  corrector exploit model inaccuracy).
- **E3.2** Per-actuator phase generator: learned oscillator per actuator; the
  policy modulates amplitude/frequency setpoints (validated pattern); corrector
  shapes within-phase timing.
- **E3.3** Aperiodic escape: phase machinery bypassed when the deliberate path
  or a threat fast-path is driving (evidence: periodic priors hurt recovery
  motions). Fall-recovery cycles route raw.
- *Accept:* parity gates; corrector-loss decreases on recorded rollouts;
  escape-path test.
- *Probe ⚙:* locomotion A/B — distance traveled per deprivation episode, fall
  count, time-to-reach within-reach vs far resources.

## E4 — Cached vs deliberate dual control  *(BUILD)*

*New fast head beside `policy`; the gate is already the arbitrator. Flag
`DECADIC_CACHED_POLICY`. Depends on E2.3 (staleness guard).*

- **E4.1** Cached head (small stimulus→action net, zero-init → falls back to
  the deliberate policy output until distilled). Gate skip → cached head drives;
  escalate → full goal-conditioned policy.
- **E4.2** Online distillation: buffer (state, deliberate-action) pairs from
  escalated cycles only; continual distillation into the cached head. **Teacher
  outputs only** — never the cached head's own actions (distillation-collapse
  guard, test-enforced).
- **E4.3** Staleness guard: E2.3 surprise forces escalation regardless of gate
  score; arbitration thrash bounded by the existing Type-2 refractory/hysteresis.
  Telemetry: cached-cycle rate, cached-vs-deliberate action divergence.
- *Accept:* parity gates; distillation-source test; divergence telemetry wired.
- *Probe ⚙:* compute per skip-cycle drops with foraging success maintained;
  after an environment change (resource relocated), surprise-driven escalation
  recovers behavior (the stale-habit test).

## E5 — Threat prediction + action veto  *(BUILD — sign-flip reuse of M3–M5)*

*Flags `DECADIC_AVERSIVE_PREDICTION`, `DECADIC_ACTION_VETO`.*

- **E5.1** Avoidance bearing: strong `predicts_pain` / `predicts_integrity_loss`
  belief on a perceived cue → negative-valence bearing (steer-away, mirror of
  M4) into the goal vector; raises fast-path/escalation.
- **E5.2** **Extinction guard (mandatory):** threat-belief confidence decays
  without re-confirmation, and a periodic re-test schedule permits approach when
  deficit pressure is high — otherwise avoidance permanently blocks unlearning
  (documented failure).
- **E5.3** Veto head: zero-init, reads `forward_predict_intero`; on predicted
  viability loss applies **minimal, uncertainty-weighted attenuation** of
  `motor_u` — never a hard zero (evidence: multiplicative zeroing + model error
  → inaction attractor). Telemetry: veto rate; alert if veto rate trends up
  while viability is stable (false-veto signature).
- *Accept:* parity gates; sign-flip geometry tests; decay/re-test unit tests;
  attenuation bounded-below test.
- *Probe ⚙:* hazard placed near a resource → agent detours; hazard removed →
  approach resumes within the decay horizon (extinction probe).

## E6 — Input routing gate + WM salience weighting  *(BUILD)*

*Between `scene_workspace` and stage2; WM admission. Flag
`DECADIC_INPUT_ROUTING`. Depends on E2.3 (reopen signal).*

- **E6.1** Per-slot gate, **identity-init** (all slots pass): weight = bottom-up
  salience ⊕ top-down goal-relevance (dot with goal vector projection).
- **E6.2** Reopen-on-surprise: E2.3 spike relaxes the gate toward identity for
  k cycles — the guard against gating out newly-relevant percepts.
- **E6.3** The same salience map weights working-memory admission (which slots
  reach `wm_slots`).
- *Accept:* identity-init parity; reopen test; no WM starvation under uniform
  salience.
- *Probe ⚙:* distractor scene (extra inert props) — cycle time and goal
  acquisition unchanged vs clean scene; baseline degrades.

## E7 — Scheduled rest consolidation  *(BUILD WITH GUARDRAILS)*

*`stub_loop.py` + `ConsolidationManager` + runtime state machine. Flag
`DECADIC_REST_CYCLE`.*

- **E7.1** Rest state machine: consolidation-load accumulator (active cycles +
  cumulative prediction-error) **bounded by time-since-last-rest** (value-drift
  guard: long wake periods make replayed estimates extrapolate); motor-output
  gate idles the body during rest.
- **E7.2** Two-phase intensive pass: (i) replay of real episodes via
  `consolidator`, then (ii) generative rollouts via `imagination.py` — the
  ordering matters (evidence: the generative second phase is what produces
  positive forward transfer; neither phase is redundant).
- **E7.3 ⚙** **A/B vs always-on (the trigger is our research bet):** retention
  of learned approach behavior + `successor_value` stability across
  rest/wake transitions vs the current always-on regime. Keep whichever wins.
- *Accept:* parity gates; state-machine tests (enter/exit/abort on threat);
  motor gate releases on any fast-path threat during rest.
- *Probe ⚙:* the E7.3 A/B, archived with verdict.

## E8 — A/B-gated heads  *(TEST FIRST — each kept only if it wins)*

- **E8.1** Interoceptive embedding → affect: head over (intero vector + tactile
  + effort) conditioning `emotion_head`. Flag `DECADIC_INTEROCEPTIVE_HEAD`.
  **A/B metric:** affect-prediction error + viability time-in-band vs latent-only
  baseline. (Evidence review: principle solid, architecture unproven — this A/B
  *is* the missing published ablation.)
- **E8.2** Valence-blended replay priority: `α·|valence| + (1−α)·|td_error|`
  **with importance-sampling correction** (never raw |valence| alone). Flag
  `DECADIC_VALENCE_REPLAY`. A/B: sample-efficiency to re-approach after a threat
  episode; value-estimate bias bounded.
- **E8.3** View-orientation term: **soft/continuous** zero-init addition to
  `motor_u` toward the most goal-relevant slot (fixes camera-at-sky), trained
  with an **intrinsic term** — reward views that reduce prediction error on the
  current goal target (evidence: pure task-reward gaze learning is the
  documented unstable regime; discrete hard-attention prohibited). Flag
  `DECADIC_VIEW_ATTENTION`. A/B: goal-target time-in-view, navigation success.
- *Accept:* each behind its metric; reject = flag stays available, default
  reverts OFF for that head only.

## E9 — Discrete abstraction bottleneck  *(BUILD — FSQ, not learned codebook)*

*`nn/symbol.py`; enters via the `mem_tokens` zero-init lane. Flag
`DECADIC_SYMBOLS`.*

- **E9.1** **FSQ quantizer** (fixed scalar grids) over bound slots / `z5` — not
  a growing learned codebook (evidence: collapse + straight-through instability;
  FSQ removes the failure class with full code utilization, no auxiliary losses).
- **E9.2** Next-token prediction head (self-supervised) over the token stream;
  tokens re-enter via `mem_tokens` next cycle (zero-init ingress).
- **E9.3** Grounding telemetry: token↔`predicts_*` belief co-occurrence stats
  (the association store is the semantics); code-utilization metric.
- *Accept:* parity gates; utilization > threshold; next-token loss decreasing.
- *Note:* expression/comprehension (audio) is **E12**, not here.

## E10 — Other-agent modeling  *(built now, activation-gated — the gate is the design)*

*Generalizes `self_model.py::build_represented_self`. Flag
`DECADIC_OTHER_MODELING`. Evidence: architecture validated; benefit exists only
when adaptive others are present; solo it is pure overhead — so the adaptivity
gate below is a first-class deliverable, not a config knob.*

- **E10.1** Adaptivity gate: classify perceived agent-entities as
  *adaptive* (movement not explained by scripted/ballistic priors — prediction
  error under a simple motion model stays high) vs *prop*. Modeling activates
  per-entity only on adaptive classification; scripted props never spawn models.
  Telemetry: models-active count (must be 0 in solo scenes — regression-tested).
- **E10.2** Other-model: per adaptive entity, a small predicted-state model
  reusing the self-model architecture — inferred need/goal/next-action; trained
  self-supervised on the entity's next observed position/action. Written as
  graph beliefs (`predicts_other_goal`, `predicts_other_next`) on the existing
  belief rail, subject to the same decay/retention governance.
- **E10.3** Policy ingress: the dominant other-model summary enters through the
  `mem_tokens` / goal-channel lane, zero-init.
- **E10.4** Observation-imitation *(can land before E10.1–E10.3 — useful even
  with scripted demonstrators)*: inverse-model labeling of an observed agent's
  motion (infer the action that produced the observed transition using our own
  forward/inverse models), feeding the existing teacher-imitation path in
  `training/teachers.py`. Budget note: observation-only imitation is provably
  sample-hungrier than action-labeled — expect slower convergence, don't call it
  a regression.
- **E10.5 ⚙** Two-agent arena support in the MuJoCo adapter + dashboard (spawn a
  second controlled entity with scripted *and* adaptive modes) — the probe rig.
- *Accept:* solo-scene zero-overhead regression test (models-active == 0,
  cycle-time parity); other-model next-position error decreases on recorded
  two-agent runs; belief writes governed.
- *Probe ⚙:* two-agent arena — (i) prediction: other-model beats the ballistic
  prior on the adaptive entity; (ii) behavior: agent yields/follows/avoids
  consistent with the other's predicted path; (iii) imitation: observed
  demonstrator performing reach-consume accelerates the observer's own
  first-success vs no-demonstrator baseline.

## E11 — Goal-stack hierarchy  *(PROVE-IT — burden of proof on the stack)*

*Flag `DECADIC_EXEC_CONTROL`. Depends on E1 (the detour task is the testbed).*

- **E11.1** Goal vector → stack (subgoal + parent slot); `goal_lifecycle`
  arbitrates need → sub-task; continuous conditioning untouched.
- **E11.2 ⚙** **Acceptance experiment (from the evidence):** beat a flat policy
  + matched exploration bonus on detour-then-forage. If it only matches, the
  measured benefit was exploration — drop the stack, keep the bonus.
- *Accept:* parity gates; the E11.2 verdict decides default-ON vs removal.

## E12 — Symbol expression/comprehension  *(gated on E9 + E10 adaptive others)*

*The language loop has no payoff without a listener (evidence-review verdict).*

- **E12.1** Expression: token → articulatory audio synthesis
  (`audio/vocal_tract.py`); comprehension: heard audio tokens keyed back into
  the E9 codebook.
- **E12.2** Grounded-exchange probe ⚙: two agents, one knows a resource
  location; measure whether emitted tokens shift the listener's approach
  behavior. Include a message-length cost (evidence: emergent codes are
  anti-efficient without one).
- *Accept:* only activates when E10.1 reports an adaptive other; solo parity.

## E13 — Phase clock  *(instrumentation only — not built for task benefit)*

- **E13.1** Slow/fast phase variables logged in the cycle scheduler for
  integration-measurement work; **no behavioral coupling** without an ablation
  first showing it beats learned timing (no such evidence exists in the
  literature; agents learn timing implicitly when tasks demand it).

---

## Dependency graph

```
E2 ──┬─→ E4 (staleness guard)
     └─→ E6 (reopen signal)
E1 ──┬─→ E11 (detour testbed)
     └─→ E8.3 (goal-target for view term)
E3      (independent)
E5      (independent; reuses M3–M5)
E7      (independent)
E8.1/8.2 (independent A/Bs)
E9 ──→ E12
E10 ──→ E12 (adaptive-other gate)
```

Recommended start order: **E2 → E1 → E3 → E4 → E5 → E6 → E7 → E8 → E9 → E10 →
E11 → E12**, with E13 passive throughout. E2 first because three later
milestones consume its surprise channel; E1 next because it is the largest
capability unlock and E11's testbed.

## Standing risks carried across the workstream

γ drift (E2.4 clamp + kill-switch) · compounding rollout error (E1.6 short
horizon + value truncation) · graph aliasing (E1.4 weighted edges + pruning) ·
stale habits (E4.3 surprise escalation) · permanent avoidance (E5.2 decay) ·
gate blindness (E6.2 reopen) · rest-phase value drift (E7.1 time bound) ·
solo-scene overhead (E10.1 adaptivity gate, regression-tested).
