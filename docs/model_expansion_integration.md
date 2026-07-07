# Capability Expansion — Integration Design

For each missing capability, **exactly where** it hooks into the existing code
and **exactly how** it fits. The recurring lesson from the codebase: the system
already exposes the right *seams*, so most of this is new heads/modules riding
existing rails under the house discipline (zero-init → birth-identical,
flag-gated default-on, thin wiring, learned-from-experience not imported).

### The five load-bearing seams everything reuses

1. **Keyed-read ingress lanes.** `NeuralCognitiveStack.forward` (`nn/neural_stack.py`)
   already accepts `memory_context`, `wm_slots`, `mem_tokens`, and `goal_vec`,
   each folded in through a **zero-init** projection so a fresh agent is
   byte-identical. Any new input channel (symbols, other-agent models, positional
   codes) rides this exact pattern.
2. **The `predicts_*` association store.** `state/working_memory.py` already
   writes `predicts_energy_relief`, `predicts_hydration_relief`, `predicts_pain`,
   `predicts_integrity_loss` onto entities, persisted as graph beliefs. Any new
   learned cue→outcome association (threat, other-agent goals, symbol grounding)
   is a new property key on the same rail.
3. **The learned world model.** The forward heads (`forward_predict`,
   `forward_predict_intero`, tactile, effort) plus the successor-features head
   are a differentiable predictive model of body state + reservoirs. Planning,
   motor correction, and action inhibition all read it.
4. **The scalar learning-control signal.** One value,
   `pleasure_scalar − pain_scalar`, computed at `cycle/neural_pipeline.py:1741`
   and consumed by `hebbian_update` at `nn/plastic.py:292`. Multi-channel
   learning control replaces the scalar here.
5. **The gate's skip/escalate arbitration.** `cycle/attention_gate.py` already
   decides fast-path (skip → decayed precedent) vs deliberate-path (escalate →
   full stage-4). Cached-vs-deliberate control and view attention key off it.

---

## Tier 1 — load-bearing absences

### 1. Spatial state estimation + model-based planning

**Where.** New `state/cognitive_map.py`; extends `state/spatial_recall.py` and
`consolidation/imagination.py`; consumes the graph's `Entity.position_json` and
the proprioceptive pose already in every observation.

**How, in two halves.**

*Spatial state.* Today `spatial_recall.egocentric_bearing` does a one-shot
world→body-frame transform to a single remembered point. A full spatial index
adds three things, all from signals already present:
- **Self-localization (dead-reckoning)** — integrate proprioceptive velocity +
  orientation (`obs["proprioception"]`) into a running world pose; correct it on
  landmark re-sighting (an observed entity whose graph position is known). Lives
  in `cognitive_map.py`, updated each cycle beside `_current_goal_vec` in `runtime`.
- **A positional code** — a small learned embedding of "where am I," produced
  from the pose, fed to the policy through a **new zero-init channel on the goal
  vector** (extend `GOAL_VEC_DIM`'s reserved space, same pattern as M4's
  bearing), so the policy is conditioned on location, not just a bearing.
- **Obstacle-aware routing** — the graph's `scene_near`/`scene_*` edges plus
  observed geometry give a coarse spatial adjacency graph; a planner (A* over
  that graph, or learned value iteration) outputs the **next waypoint** instead
  of the straight-line target. `_goal_bearing` changes one line: bearing to the
  *next waypoint toward* the target, not bearing to the target.

*Model-based planning.* The pieces already exist — `imagination.py` rolls
`forward_predict_intero` forward under `no_grad` to build value targets. Promote
that from an offline evaluator to an **online, action-selecting** search: when
the deliberate path is engaged, sample K short action sequences, roll the forward
models, score each by successor-feature value, and bias `motor_u` toward the
best. It plugs in at the deliberate branch in `neural_pipeline` where the goal
bearing is applied, and reuses the exact rollout code in `imagination.py`.

**Training.** Positional code: self-supervised next-pose prediction. Planner:
model-based, no new training (uses the learned forward models + successor value).
**Discipline.** `DECADIC_COGNITIVE_MAP`; zero-init positional channel →
birth-identical; planning only on the (already-gated, refractory) deliberate path
so it stays cheap.

### 2. Symbolic abstraction layer

**Where.** New `nn/symbol.py`; enters through the existing `mem_tokens` lane in
`neural_stack.forward`; builds on the relational binding core
(`nn/relational_core.py`) and the audio path (`audio/*`).

**How.** The relational core already *binds* percepts into slots. Add a
**vector-quantization codebook** over bound slots / the `z5` latent: a small,
growing set of discrete, composable tokens (a symbol = a codebook index). Those
symbol tokens ride the `mem_tokens` keyed-read lane back into stage2/stage3 next
cycle — a **recurrent symbolic feedback** loop: an autoregressive head over
symbols whose output re-enters the input next cycle. Grounding is free: a symbol
that reliably co-occurs with a `predicts_*_relief` belief inherits that meaning
(the association store is the semantics). The audio-synthesis module
(`audio/vocal_tract.py`) makes symbols expressible; the inverse (codebook keyed
by heard audio tokens) is comprehension.

**Training.** VQ commitment loss + next-symbol prediction (self-supervised) +
grounding through existing beliefs. **No external language model** — the codebook
is grown from lived binding.
**Discipline.** `DECADIC_SYMBOLS`; zero-init token ingress → birth-identical.

### 3. Multi-channel learning-control signals

**Where.** The single seam at `neural_pipeline.py:1741`
(`pleasure_scalar − pain_scalar`) and `nn/plastic.py:292`
(`hebbian_update(modulation, eta)`).

**How.** Replace the scalar with a **4-vector** computed from state already in
hand, and route each channel to the control it governs:
- **Reward channel** = `pleasure_scalar − pain_scalar` → sign/magnitude of the
  local weight update (current behavior).
- **Expected-uncertainty channel** = a function of `pc_slope_ema` / gate novelty
  → scales the learning rate `eta` and biases attention (fast updates when the
  input stream is reliably informative).
- **Surprise channel** = `pc_loss` spikes / the loss-canary state → transient
  learning-rate boost + raises the gate's escalation propensity.
- **Horizon channel** = viability trend → modulates the successor-feature discount
  `γ` (a thriving agent plans a longer horizon) and the deliberate-path refractory.

`hebbian_update` takes a vector; `apply_plasticity_step` computes the four from
`pc_ema`, `pc_slope_ema`, novelty, and viability (all already threaded there);
the gate reads the uncertainty/surprise channels for its threshold and refractory.

**Training.** None — pure routing of existing signals.
**Discipline.** `DECADIC_LEARN_CONTROL_MULTI`; init maps all four to the current
scalar → byte-identical until tuned.

---

## Tier 2 — present, but a thin slice

### 4. Input routing + gating layer

**Where.** Between `perception/scene_workspace.py` and the `stage2` input in
`neural_stack.forward`; and the broadcast in `nn/workspace.py` /
`cycle/integration_window.py`.

**How.** Insert a **routing gate**: a per-slot weight/mask deciding which percept
slots reach the deep network each cycle, combining bottom-up salience with the
top-down goal vector (goal-relevant slots get through). It sits on the slot
tensor before stage2. Add a **phase variable** driving `integration_window` —
bind percepts within one integration window, sub-gate deliberation on a faster
tick. This is scheduling/gating, not new representation; the existing gate stays,
the routing hub is what's added.
**Discipline.** `DECADIC_INPUT_ROUTING`; identity gate at init.

### 5. Fine-motor error-correction module

**Where.** `neural_stack.forward`, immediately after
`motor_u = tanh(self.motor(pol_in_t))`.

**How.** Add a **zero-init corrector head** `self.motor_corrector(motor_u, proprio)`:
`motor_u = motor_u + tanh(self.motor_corrector(...))`. A fast, high-capacity head
trained purely by proprioceptive prediction error (already computed by
`forward_predict`). Timing rides a learned per-actuator phase/CPG the correction
shapes. This is why locomotion is hard today: the policy sets postural targets
but nothing does fine error-driven motor refinement.
**Training.** Existing proprioceptive forward-model error.
**Discipline.** `DECADIC_MOTOR_CORRECTOR`; zero-init correction → birth-identical.

### 6. Interoceptive representation module

**Where.** `state/viability.py::controllable_intero_vector` (raw interoception,
consumed at `neural_pipeline.py:1064/1844`) → a new interoceptive-state head
feeding `nn/affect_model.py`.

**How.** Today interoception is raw reservoir numbers. Add an **interoceptive
embedding** head over (intero vector + tactile + effort) that produces a felt
body-state representation the affect head reads — affect computed as a readout of
body state, not only the latent. The `emotion_head` conditions on this embedding
so valence estimates become genuinely body-grounded.
**Training.** Predict next intero (already an objective) + affect reconstruction.
**Discipline.** `DECADIC_INTEROCEPTIVE_HEAD`; zero-init → affect unchanged at init.

### 7. Aversive-association (threat-prediction) module

**Where.** Already half-built: `working_memory.py` writes `predicts_pain` /
`predicts_integrity_loss` beliefs; the gate has `fast_path_threat`. The wiring is
what's missing.

**How.** Two connections. (a) **Threat prediction**: when a cue with a strong
`predicts_pain` belief is perceived, feed an *avoidance* signal into the goal
vector — a negative-valence bearing (steer **away**), the mirror of M4's approach
bearing — and raise `fast_path`/escalation. This is the exact M3/M4/M5 machinery
with the sign flipped. (b) **Salience-weighted retention**: the replay buffer
already weights transitions by salience; multiply salience by `|valence|` so
high-valence episodes get preferential consolidation.
**Training.** The `predicts_pain` beliefs already form; add the gate/goal wiring
and the salience weighting.
**Discipline.** `DECADIC_AVERSIVE_PREDICTION`.

### 8. Scheduled offline consolidation phases

**Where.** `consolidation/stub_loop.py` + `ConsolidationManager`; a new
active/rest gate in the `runtime` cycle loop.

**How.** Consolidation is always-on background today. Add a **consolidation-load**
accumulator (rises with active cycles and cumulative prediction-error load) and a
slow **cycle clock**. When load crosses threshold → enter a *rest state*: the
runtime stops emitting motor output (body idle, low activation) and runs
**intensive** consolidation in two phases — replay of real episodes (via
`consolidator`) then generative rollouts (via `imagination.py`). Reuses both
existing engines; adds a state machine + a motor-output gate in the cycle loop.
**Discipline.** `DECADIC_REST_CYCLE`; default off keeps always-on consolidation.

---

## Tier 3 — next-horizon

### 9. Other-agent modeling

**Where.** Generalize `state/self_model.py::build_represented_self` to an
*other*-model; other agents already arrive as `nearby_entities` / controlled
entities (`embodiment/npc_controller.py`, scene workspace) and live in the graph.

**How.** For each perceived agent-entity, maintain a small predicted-state model
(inferred need / goal / next action) using the **same** architecture as the
self-model, stored as graph beliefs (`predicts_other_goal`, `predicts_other_next`).
Feed the dominant other-model to the policy as a `mem_token` / goal channel so the
agent can act on others' predicted behavior. **Imitation**: the teacher path
(`training/teachers.py`) already imitates an expert motor stream — generalize it
to imitate an *observed* controlled-agent's actions.
**Training.** Predict the other agent's next observed action/position
(self-supervised). **Discipline.** `DECADIC_OTHER_MODELING`.

### 10. Hierarchical goals + inhibitory gating

**Where.** `nn/goal_conditioning.py` + `state/goal_lifecycle.py` (hierarchy);
the motor head (inhibition).

**How.** **Hierarchy** — extend the goal vector to a *stack* (subgoal + parent);
`goal_lifecycle` arbitrates high-level needs → sub-tasks. **Inhibition** — a
learned veto head that suppresses `motor_u` when the forward models predict a bad
outcome (`forward_predict_intero` says the action worsens viability → multiply
the motor output down). **Task-switching** — the already-continuous goal
arbitration.
**Discipline.** `DECADIC_EXEC_CONTROL`; zero-init veto (no suppression at init).

### 11. Cached-policy vs deliberate-policy dual control

**Where.** A new fast **cached-action head** beside the deliberate `policy`;
arbitrated by the gate's existing skip/escalate decision.

**How.** When the gate **skips** (fast-path), a fast cached stimulus→action head
drives the body; when it **escalates** (deliberate-path), the full goal-directed
policy runs. Cached actions form by distilling repeated deliberate actions into
the fast head (policy distillation), so over-learned routines run without
deliberation — and free the deliberate path for novelty. The gate already makes
the skip/escalate call; this just routes it to two heads.
**Discipline.** `DECADIC_CACHED_POLICY`; cached head zero-init → falls back to
the policy until distilled.

### 12. Salience-driven view control

**Where.** `perception/scene_workspace.py` (salience) + a view-orientation term
on the motor output (the deferred WS-FORAGE M6.2); head/view actuators in the
MuJoCo adapter.

**How.** A **salience map** over current slots (bottom-up salience + top-down
goal-relevance) that (a) weights which slots enter working memory and (b) emits a
view-orientation command toward the most goal-relevant location. Fixes the
"points the camera at the sky" failure — the agent would orient its view toward
its goal. Plugs a zero-init view-orientation term into `motor_u`.
**Discipline.** `DECADIC_VIEW_ATTENTION`.

### 13. Global timing/phase schedule

**Where.** The cycle scheduler + `integration_window` + `workspace`.

**How.** Global phase variables (a slow one ≈ the integration window, a fast one
≈ the deliberation sub-tick) modulating *when* binding and gating happen. Mostly
of interest for integration-measurement work; lowest priority — a phase clock in
the cycle loop rather than new representation.
**Discipline.** `DECADIC_PHASE_CLOCK`.

---

## The pattern, and a suggested order

Notice how little genuinely new infrastructure any of this needs. Symbols,
other-agent models, positional codes, and attention all enter through the **same
zero-init keyed-read lane** M3/M4 already use. Threat prediction and grounding
are new keys on the **existing association store**. Planning, motor correction,
and inhibition all read the **existing forward-model + successor-feature world
model**. Multi-channel learning control is a **one-seam** change. That is the
dividend of the functional decomposition: the hard architectural work — the
ingress discipline, the association store, the world model, the gate — is done,
and these are heads hung on it.

Highest-leverage order for the current trajectory (goal-directed foraging just
working): **(1) spatial state + planning** completes navigation and consumes the
foraging spine directly; **(3) multi-channel learning control** is a cheap
one-seam win that sharpens all updates; **(6/7) interoceptive head + threat
prediction** are nearly free given `controllable_intero_vector` and the
`predicts_pain` beliefs already exist; **(5) motor corrector** unblocks
locomotion; then **(2) symbolic abstraction** as the deep, long-horizon chapter
once the sensorimotor and spatial substrate is solid.
