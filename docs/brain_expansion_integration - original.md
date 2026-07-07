# Brain Expansion — Integration Design

Companion to `BRAIN_MAP.md`. For each missing brain system, **exactly where** it
hooks into the existing code and **exactly how** it fits. The recurring lesson
from the codebase: Deca already exposes the right *seams*, so most of this is new
heads/modules riding existing rails under the house discipline (zero-init →
birth-identical, flag-gated default-on, thin wiring, grown-not-downloaded).

### The five load-bearing seams everything reuses

1. **Keyed-read ingress lanes.** `NeuralCognitiveStack.forward` (`nn/neural_stack.py`)
   already accepts `memory_context`, `wm_slots`, `mem_tokens`, and `goal_vec`,
   each folded in through a **zero-init** projection so a fresh agent is
   byte-identical. Any new "cortical input" (symbols, other-minds, place codes)
   rides this exact pattern.
2. **The `predicts_*` belief system.** `state/working_memory.py` already writes
   `predicts_energy_relief`, `predicts_hydration_relief`, `predicts_pain`,
   `predicts_integrity_loss` onto entities, persisted as graph beliefs. Any new
   learned association (threat, other-agent goals, symbol grounding) is a new
   property key on the same rail.
3. **The world model.** The forward heads (`forward_predict`,
   `forward_predict_intero`, tactile, effort) plus the successor-features head
   are a differentiable predictive model of body + reservoirs. Planning,
   cerebellar correction, and inhibition all read it.
4. **The neuromodulator.** One scalar, `modulation = pleasure − pain`, computed
   at `cycle/neural_pipeline.py:1741` and consumed by `hebbian_update` at
   `nn/plastic.py:292`. Differentiated neuromodulation replaces the scalar here.
5. **The gate's skip/escalate arbitration.** `cycle/attention_gate.py` already
   decides System-1 (skip → decayed precedent) vs System-2 (escalate → full
   stage-4). Habit-vs-goal-directed control and spatial attention key off it.

---

## Tier 1 — load-bearing absences

### 1. Cognitive map + model-based planning

**Where.** New `state/cognitive_map.py`; extends `state/spatial_recall.py` and
`consolidation/imagination.py`; consumes the LTM graph's `Entity.position_json`
and the proprioceptive pose already in every observation.

**How, in two halves.**

*The map (hippocampal-entorhinal).* Today `spatial_recall.egocentric_bearing`
does a one-shot allocentric→egocentric transform to a single remembered point.
A real map adds three things, all from signals already present:
- **Self-localization / path integration** — integrate proprioceptive velocity +
  orientation (`obs["proprioception"]`) into a running world pose; correct it on
  landmark re-sighting (an observed entity whose graph position is known). Lives
  in `cognitive_map.py`, updated each cycle in `runtime` beside `_current_goal_vec`.
- **A place code** — a small learned embedding of "where am I" (grid/place-cell
  analog) produced from the pose, fed to the policy through a **new zero-init
  channel on the goal vector** (extend `GOAL_VEC_DIM`'s reserved space, same
  pattern as M4's bearing) so navigation is conditioned on location, not just a
  bearing.
- **Obstacle-aware routing** — the graph's `scene_near`/`scene_*` edges plus
  observed geometry give a coarse place-graph; a planner (A* over the place-graph,
  or learned value iteration) outputs the **next waypoint** instead of the
  straight-line target. `_goal_bearing` changes one line: bearing to *next
  waypoint toward* the target, not bearing to the target.

*Model-based planning (PFC).* The pieces already exist — `imagination.py`
rolls `forward_predict_intero` forward under `no_grad` to build value targets.
Promote that from an offline evaluator to an **online, action-selecting** search:
when Type-2 escalates, sample K short action sequences, roll the forward models,
score each by successor-feature value, and bias `motor_u` toward the best. It
plugs in at the deliberate branch in `neural_pipeline` right where the goal
bearing is applied, and reuses the exact rollout code in `imagination.py`.

**Training.** Place code: self-supervised path-integration loss (predict next
pose). Planner: model-based, no new training (uses learned forward models + SF).
**Discipline.** `DECADIC_COGNITIVE_MAP`; zero-init place channel → birth-identical;
planning only on the (already-gated, now-refractory) Type-2 path so it stays cheap.

### 2. Language / symbolic cognition

**Where.** New `nn/symbol.py`; enters cognition through the existing `mem_tokens`
lane in `neural_stack.forward`; builds on the WS5 relational core (`nn/relational_core.py`)
and the audio loop (`audio/*`, WS6).

**How.** The relational core already *binds* percepts into slots. Add a
**vector-quantization codebook** over bound slots / the `z5` latent: a small,
growing vocabulary of discrete, composable tokens (a symbol = a codebook index).
Those symbol tokens ride the `mem_tokens` keyed-read lane back into stage2/stage3
next cycle — this is **inner speech**: an autoregressive head over symbols whose
output re-enters perception. Grounding is free: a symbol that reliably co-occurs
with a `predicts_*_relief` belief inherits that meaning (the belief system is the
semantics). WS6's vocal tract makes symbols *utterable*; comprehension is the
codebook keyed by heard audio tokens.

**Training.** VQ commitment loss + next-symbol prediction (self-supervised) +
grounding through existing beliefs. **No LLM** — the codebook is grown from lived
binding, exactly the "language must be lived" thesis.
**Discipline.** `DECADIC_SYMBOLS`; zero-init token ingress → birth-identical.

### 3. Differentiated neuromodulation

**Where.** The single seam at `neural_pipeline.py:1741`
(`modulation = pleasure − pain`) and `plastic.py:292` (`hebbian_update(modulation, eta)`).

**How.** Replace the scalar with a **4-vector** computed from state already in
hand, and route each channel to the control it governs:
- **DA (reward)** = `pleasure − pain` → Hebbian sign/magnitude (current behavior).
- **ACh (expected uncertainty)** = a function of `pc_slope_ema` / gate novelty →
  scales the learning rate `eta` and biases attention (fast learning when the
  world is reliably informative).
- **NE (unexpected surprise)** = `pc_loss` spikes / the loss-canary state →
  transient plasticity boost + raises the gate's escalation propensity (arousal).
- **5-HT (patience / mood)** = viability trend → modulates the SF discount `γ`
  (a thriving agent plans longer-horizon) and the Type-2 refractory.

`hebbian_update` takes a vector; `apply_plasticity_step` computes the four from
`pc_ema`, `pc_slope_ema`, novelty, and viability (all already threaded there);
the gate reads ACh/NE for its threshold and refractory.

**Training.** None — pure routing of existing signals.
**Discipline.** `DECADIC_NEUROMOD_DIFFERENTIATED`; init maps all four to the
current scalar → byte-identical until tuned.

---

## Tier 2 — present, but a thin slice

### 4. Thalamus (relay/routing + rhythm)

**Where.** Between `perception/scene_workspace.py` and the `stage2` input in
`neural_stack.forward`; and the broadcast in `nn/workspace.py` /
`cycle/integration_window.py`.

**How.** Insert a **thalamic relay**: a per-slot gate that weights/masks which
percept slots reach cortex each cycle, combining bottom-up salience with the
top-down goal vector (goal-relevant slots get through). It sits on the slot
tensor before stage2. The **rhythm** is a phase variable driving
`integration_window` — bind percepts within a "theta" window, sub-gate
deliberation on a "gamma" tick. This is scheduling/gating, not new cognition;
the reticular-nucleus slice (the gate) stays, the relay/hub is what's added.
**Discipline.** `DECADIC_THALAMIC_ROUTING`; identity gate at init.

### 5. Cerebellum (fine coordination / timing)

**Where.** `neural_stack.forward`, immediately after
`motor_u = tanh(self.motor(pol_in_t))`.

**How.** Add a **zero-init corrector head** `self.cerebellum(motor_u, proprio)`:
`motor_u = motor_u + tanh(self.cerebellum(...))`. It's a fast, high-capacity
head trained purely by proprioceptive prediction error (the climbing-fiber
analog — the error is already computed by `forward_predict`). Timing rides a
learned per-actuator phase/CPG the correction shapes. This is why locomotion is
hard today: the policy sets postural targets but nothing does fine error-driven
motor refinement.
**Training.** Existing proprioceptive forward-model error.
**Discipline.** `DECADIC_CEREBELLUM`; zero-init correction → birth-identical.

### 6. Insula (felt body / interoceptive awareness)

**Where.** `state/viability.py::controllable_intero_vector` (raw interoception,
consumed at `neural_pipeline.py:1064/1844`) → a new interoceptive-cortex head
feeding `nn/affect_model.py`.

**How.** Today interoception is raw reservoir numbers. Add an **interoceptive
embedding** head over (intero vector + tactile + effort) that produces the *felt
body state* the emotion head reads — James-Lange: emotion as a read-out of body
state. The `emotion_head` conditions on this embedding rather than only the
latent, so affect becomes genuinely interoceptive.
**Training.** Predict next intero (already an objective) + affect reconstruction.
**Discipline.** `DECADIC_INTEROCEPTIVE_CORTEX`; zero-init → affect unchanged at init.

### 7. Amygdala (fear conditioning + emotional memory tagging)

**Where.** Already half-built: `working_memory.py` writes `predicts_pain` /
`predicts_integrity_loss` beliefs; the gate has `fast_path_threat`. The wiring is
what's missing.

**How.** Two connections. (a) **Fear conditioning**: when a cue with a strong
`predicts_pain` belief is perceived, feed an *avoidance* signal into the goal
vector — a negative-valence bearing (steer **away**), the mirror of M4's
approach bearing — and raise `fast_path`/escalation. This is the exact M3/M4/M5
machinery with the sign flipped. (b) **Emotional memory tagging**: the replay
buffer already weights transitions by salience; multiply salience by `|affect|`
so high-emotion episodes get preferential consolidation (flashbulb memory).
**Training.** The `predicts_pain` beliefs already form; add the gate/goal wiring
and the salience weighting.
**Discipline.** `DECADIC_FEAR_CONDITIONING`.

### 8. Sleep architecture / circadian

**Where.** `consolidation/stub_loop.py` + `ConsolidationManager`; a new
wake/sleep gate in the `runtime` cycle loop.

**How.** Consolidation is always-on background today. Add a **sleep-pressure**
accumulator (rises with wake cycles and cumulative prediction-error load) and a
slow **circadian** phase. When pressure crosses threshold → enter *sleep*: the
runtime stops emitting motor output (body goes limp, low arousal) and runs
**intensive** consolidation in two stages — SWS (replay real episodes via
`consolidator`) then REM (dreaming via `imagination.py`). Reuses both existing
engines; adds a state machine + a motor-output gate in the cycle loop.
**Discipline.** `DECADIC_SLEEP_CYCLE`; default off keeps always-on consolidation.

---

## Tier 3 — next-horizon

### 9. Social cognition / theory of mind

**Where.** Generalize `state/self_model.py::build_represented_self` to an
*other*-model; other agents already arrive as `nearby_entities` / NPC entities
(`embodiment/npc_controller.py`, scene workspace) and live in the LTM graph.

**How.** For each perceived agent-entity, maintain a small predicted-state model
(inferred need / goal / next action) using the **same** architecture as the
self-model, stored as graph beliefs (`predicts_other_goal`, `predicts_other_next`).
Feed the dominant other-model to the policy as a `mem_token` / goal channel so
the agent can act on others' predicted behavior. **Mirror neurons / imitation**:
the Skill Dojo teacher path (`training/teachers.py`) already imitates an expert
motor stream — generalize it to imitate an *observed* NPC's actions.
**Training.** Predict the other agent's next observed action/position
(self-supervised). **Discipline.** `DECADIC_THEORY_OF_MIND`.

### 10. Executive control (inhibition, task-switching, hierarchy)

**Where.** `nn/goal_conditioning.py` + `state/goal_lifecycle.py` (hierarchy);
the motor head (inhibition).

**How.** **Hierarchy** — extend the goal vector to a *stack* (subgoal + parent);
`goal_lifecycle` arbitrates needs → sub-tasks. **Inhibition** — a learned veto
head that suppresses `motor_u` when the forward models predict a bad outcome
(`forward_predict_intero` says the action worsens viability → multiply the motor
output down). **Task-switching** — the already-continuous goal arbitration.
**Discipline.** `DECADIC_EXEC_CONTROL`; zero-init veto (no suppression at init).

### 11. Habit vs. goal-directed dual control

**Where.** A new fast **habit head** beside the goal-directed `policy`; arbitrated
by the gate's existing skip/escalate decision.

**How.** Dorsolateral (habit) vs dorsomedial (goal-directed) striatum: when the
gate **skips** (System-1), a fast cached stimulus→action habit head drives the
body; when it **escalates** (System-2), the full goal-directed policy runs.
Habits form by distilling repeated goal-directed actions into the fast head
(policy distillation), so over-learned routines run without deliberation — and
free the deliberate path for novelty. The gate already makes the skip/escalate
call; this just routes it to two heads.
**Discipline.** `DECADIC_HABIT_SYSTEM`; habit head zero-init → falls back to the
policy until distilled.

### 12. Spatial attention + gaze control

**Where.** `perception/scene_workspace.py` (salience) + a gaze term on the motor
output (the deferred WS-FORAGE M6.2); head/gaze actuators in the MuJoCo adapter.

**How.** A **salience map** over current slots (bottom-up salience + top-down
goal-relevance) that (a) weights which slots enter working memory and (b) emits
a gaze-orient command toward the most goal-relevant location. Fixes the
"stares at the sky" failure — the agent would look where its goal is. Plugs a
zero-init gaze term into `motor_u`.
**Discipline.** `DECADIC_GAZE_ATTENTION`.

### 13. Neural oscillations / large-scale dynamics

**Where.** The cycle scheduler + `integration_window` + `workspace`.

**How.** Global phase variables (theta ≈ the integration window, gamma ≈ the
deliberation sub-tick) modulating *when* binding and gating happen. Mostly of
interest if you later pursue IIT-style integration measurement; lowest priority,
a phase clock in the cycle loop rather than new cognition.
**Discipline.** `DECADIC_OSCILLATIONS`.

---

## The pattern, and a suggested order

Notice how little genuinely new infrastructure any of this needs. Symbols,
theory-of-mind, place codes, and attention all enter through the **same zero-init
keyed-read lane** M3/M4 already use. Fear conditioning and grounding are new
keys on the **existing belief rail**. Planning, cerebellar correction, and
inhibition all read the **existing forward-model + successor-feature world
model**. Differentiated neuromodulation is a **one-seam** change. That is the
dividend of the functional decomposition: the hard architectural work — the
ingress discipline, the belief system, the world model, the gate — is done, and
these are heads hung on it.

Highest-leverage order for the current trajectory (goal-directed foraging just
working): **(1) cognitive map + planning** completes navigation and consumes the
foraging spine directly; **(3) differentiated neuromodulation** is a cheap
one-seam win that sharpens all learning; **(6/7) insula + amygdala** are nearly
free given `controllable_intero_vector` and the `predicts_pain` beliefs already
exist; **(5) cerebellum** unblocks locomotion; then **(2) language** as the
deep, long-horizon chapter once the sensorimotor and spatial substrate is solid.
