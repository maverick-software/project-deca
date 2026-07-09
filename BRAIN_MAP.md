# Project Deca — Brain-Region Map

A reference mapping every brain structure Deca mimics to **how** it is mimicked,
**where** it lives in the code, and **whether its tests pass**.

Deca is a *functional* brain, not an anatomical one: there is no six-layer
cortical column or literal cytoarchitecture. Instead a deep, plastic,
**predictive-coding-trained** network (the neocortex analog) sits atop a set of
subcortical analogs — drives, value, memory, affect, attention. The organizing
loop is the ten-stage "Decadic Cycle," implemented neurally in
`decadic/nn/neural_stack.py` and driven by `decadic/cycle/neural_pipeline.py`.
There is **no LLM in the loop** — cognition is grown from lived experience, not
downloaded.

**WS-EXPAND (2026-07-06)** widened the map: it split the single reinforcement
signal into **differentiated neuromodulation**, gave the cerebellum a true
**error-driven corrector + central pattern generator**, turned the aspirational
hippocampal **cognitive map** into a working pose-estimator-plus-planner, and
added regions Deca did not model before — an **insular interoceptive cortex**, a
**basal-ganglia habit system**, an **amygdalar threat-prediction + prefrontal
veto** pair, a **thalamic-reticular sensory gate**, a **theory-of-mind** module,
a **symbolic-abstraction** bottleneck, and scheduled **sleep**. Every one ships
zero-init / identity-init (birth-identical) and default-on. All were unit-tested
and then **live-verified on the body in one 10-minute probe: 12/12 pathways
active at 4.18 cyc/s with no throughput cost** (`scripts/run_ws_expand_probe.ps1`).

### Test-status legend

| Symbol | Meaning |
|---|---|
| ✅ | Dedicated tests present and passing (full suite last green at ~985 passed / 7 skipped). |
| 🟢 | WS-EXPAND region: unit-tested **and** live-probe-verified on the body (12/12, 2026-07-06). |
| 🟡 | Passing, but a fix from 2026-07-06 is pending your next full re-run. |
| 🧪 | Partial / nascent capability; tests cover what is built. |

The whole suite runs with `.venv\Scripts\python.exe -m pytest -q`. Per-region
status below points at the specific test files.

---

## I. Neocortex — the learned predictive stack

The neocortex analog is everything in `neural_stack.py` above the frozen sensory
front-end: association, evaluation, conclusion, and motor stages, all trained by
prediction-error minimization under a plasticity controller. This is "cortex-like"
in the way that matters computationally — hierarchical predictive processing —
rather than laminar.

| Brain part | How Deca mimics it | Where (modules) | Tests | Status |
|---|---|---|---|---|
| **Sensory transduction** (retina / cochlea / early sensory relay) | Frozen CLIP (vision) + Whisper (audio) encoders + proprioception; predecode and fuse into a percept. An *inherited transducer*, not a lived cortex. | `nn/frozen_encoders.py`, `perception/organ.py`, `perception/bootstrap.py`, `perception/integration.py`, `state/perceptual_state.py` | `test_perception_organ.py`, `test_encoder_precision.py`, `test_predecode.py` | ✅ |
| **Sensory association cortex** (multisensory integration / framing) | Stage-2 Transformer encoder frames the percept; stage-3 fusion MLP correlates it with memory context. | `nn/neural_stack.py` (`stage2`, `stage3`), `cycle/neural_pipeline.py` | `test_neural_cycle.py`, `test_neural_integration.py`, `test_stage_pipeline.py`, `test_neural_presets.py` | ✅ |
| **Visual object binding** ("what + where" / ventral–dorsal streams) | Slot-attention object discovery, object files, scene dynamics, and a relational-binding core (WS5) that keys objects across time (object permanence). | `nn/slots.py`, `nn/relational_core.py`, `nn/scene_dynamics.py`, `perception/{object_files,scene_workspace,discovery_metrics}.py` | `test_object_files.py`, `test_scene_workspace.py`, `test_scene_dynamics.py`, `test_perception_discovery.py`, `test_ws5_binding.py` | ✅ |
| **Cortical learning** (predictive processing + synaptic plasticity) | The core cortical computation: prediction-error (`pc_loss`) minimization, plus **A** Hebbian traces, **B** sparse rewiring, **C** dormant-neuron growth — with a metaplasticity guardian (freeze/thaw) and progress-gated growth governance. | `nn/plastic.py`, `cycle/neural_pipeline.py` (guardian + growth), `nn/optim.py` | `test_plasticity.py`, `test_plasticity_stability.py`, `test_plasticity_parity.py`, `test_sparse_training.py`, `test_growth.py`, `test_growth_governance.py`, `test_nan_firewall.py` | ✅ |
| **Motor cortex** (voluntary action) | Policy head (direction/speed) + motor head emitting 21 per-actuator PD targets; the body tracks them with a fast PD loop. | `nn/neural_stack.py` (`policy`, `motor`), `state/body_map.py`, `scripts/mujoco_decadic_adapter.py` | `test_embodied_motor.py`, `test_body_proprio.py`, `test_body_map_effort.py`, `test_walking_curriculum.py` | ✅ |
| **Cerebellum** (forward models / predictive motor) | Learned forward models predict next proprioception, tactile load, effort, and interoception from an efference copy — enabling active inference (planning through a frozen world model). **WS-EXPAND E3.1:** added the cerebellum's defining computation — a zero-init **feedback-error-learning corrector** (Kawato) that adds a bounded correction to the PD targets, trained on the *realized* per-joint tracking error as a supervised target (never a reward it could game). | `nn/neural_stack.py` (`fwd_*`, `motor_corrector_l1/l2`, `motor_correction`), `cycle/neural_pipeline.py` (FEL term) | `test_embodied_motor.py`, `test_neural_cycle.py`, `test_motor_corrector.py` | 🟢 |
| **Central pattern generators** (spinal / brainstem locomotor rhythm) | **WS-EXPAND E3.2/E3.3:** a per-actuator free-running phase whose motor contribution is `c·sin(phase)`; the amplitude `c` and frequency come from a **zero-init head**, so there is no oscillation at birth — rhythm is *earned* where periodic drive pays. A threat fast-path cycle silences the phase and holds it (aperiodic escape), so recovery motions route raw. | `nn/neural_stack.py` (`cpg_head`, `cpg_phase`), `cycle/neural_pipeline.py` (`cpg_gate`) | `test_motor_corrector.py` | 🟢 |
| **Default-mode / narrative + metacognition** (self-model) | Narrative and metacognition heads; a self-model feedback spine and a "represented self" that folds the prior cycle's self-report back into framing; persistent mental image. | `state/self_model.py`, `cycle/narrative.py`, `nn/neural_stack.py` (`narrative_head`, `metacog_head`, `state_mind_head`) | `test_self_model_feedback.py`, `test_represented_self.py`, `test_self_model_stability.py`, `test_persistent_mental_image.py` | ✅ |
| **Broca / Wernicke** (speech production + comprehension) | WS6 speech loop: audio intake service, a numpy vocal-tract synthesizer, and playback — a "mouth" whose sound re-enters perception. Nascent (specced + partially built). | `audio/{intake,vocal_tract,playback}.py` | `test_ws6_speech.py` | 🧪 |

---

## II. Prefrontal executive & thalamic gating — attention, deliberation, goals

The System-1 / System-2 machinery: what gets deliberated, what the agent is
*trying* to do, and the memory-guided pursuit of goals not currently in view.

| Brain part | How Deca mimics it | Where (modules) | Tests | Status |
|---|---|---|---|---|
| **Thalamic gating / System-1↔2 arbitration** | The WS3 attention gate decides per cycle whether stage-4 deliberation runs (escalate) or a decayed precedent passes through (skip), with hysteresis and a soft escalation budget. | `cycle/attention_gate.py`, `nn/gate_net.py` | `test_attention_gate.py`, `test_ws3b_gate_data.py` | ✅ |
| **Prefrontal working memory** (Baddeley) | Bounded entity slots with decay and capacity eviction; the active "now" the policy reasons over. | `state/working_memory.py`, `nn/workspace.py` | `test_working_memory.py` | ✅ |
| **Global workspace / conscious access (GWT)** | Capacity-limited winner-take-all competition + ignition threshold + broadcast; a temporal integration window binds a span of percepts into one committed "now." | `nn/workspace.py`, `cycle/integration_window.py` | `test_global_workspace.py`, `test_integration_window.py` | ✅ |
| **Goal representation** (dorsolateral PFC) | Continuous goal conditioning (2026-07-06): the policy is conditioned every cycle on the dominant need + graded deficit + an egocentric bearing to the remembered resource. A latch marks credit-assignment episode boundaries only. | `nn/goal_conditioning.py`, `state/goal_lifecycle.py`, `state/spatial_recall.py`, `agents/runtime.py` (`_current_goal_vec`) | `test_goal_conditioning.py`, `test_goal_lifecycle.py`, `test_spatial_recall.py` | ✅ |
| **Deliberate memory-guided pursuit** (PFC–BG System-2 loop) | Type-2 escalation: when a need's relief is *remembered but not here*, the gate escalates into deliberate pursuit — now refractory-gated so one intention forms, executes, and periodically re-checks (no perseveration). | `cycle/attention_gate.py` (`type2_trigger`, refractory), `cycle/neural_pipeline.py` | `test_attention_gate.py` | ✅ |
| **Thalamic reticular nucleus** (top-down sensory gating) | **WS-EXPAND E6:** a per-slot routing gate scores each percept slot for *goal relevance* and admits it to the deep network. **Identity at init** (exact pass-through), floored so nothing is ever fully silenced, and **reopened toward identity by the surprise channel** so gating can never blind the agent to a newly-surprising world. | `nn/neural_stack.py` (`slot_gate`, `slot_relevance`), `cycle/neural_pipeline.py` | `test_ws_expand_tail.py` | 🟢 |
| **Prefrontal prospection** (model-based planning / mental simulation) | **WS-EXPAND E1.6:** on a deliberate cycle, imagine K variations of the chosen action through the interoceptive world model, score each by the deficit-gated successor value, and apply a **bounded bias** (never override) toward the best. Short horizon truncated by the learned value (compounding-error control); gain scales with the value ramp so a naive agent never plans. Fired live (bias 0.125). | `nn/action_planner.py`, `cycle/neural_pipeline.py`, `consolidation/imagination.py` | `test_action_planner.py` | 🟢 |
| **Prefrontal inhibitory control** ("free won't" / action veto) | **WS-EXPAND E5.3:** a zero-init veto head predicts imminent viability loss for the final command and applies a **minimal, capped attenuation** (never a hard zero — the over-conservatism guardrail), trained on realized viability drops. | `nn/neural_stack.py` (`veto_l1/l2`, `motor_veto_raw`), `cycle/neural_pipeline.py` | `test_ws_expand_tail.py` | 🟢 |

---

## III. Subcortical structures

The drive, value, memory, and affect systems the cortex sits on top of — what
makes "does it have a neocortex" a meaningful question.

| Brain part | How Deca mimics it | Where (modules) | Tests | Status |
|---|---|---|---|---|
| **Hypothalamus / brainstem** (homeostasis, drives, survival) | Hydration/energy/integrity reservoirs draining on a real wall-clock; viability = their minimum; mortality when one bottoms out; drive pressure rises with deprivation. | `state/viability.py`, `state/curiosity.py`, homeostasis in `agents/runtime.py` | `test_viability.py`, `test_homeostasis.py`, `test_homeostatic_drive.py`, `test_mortality.py` | ✅ |
| **Basal ganglia / dopaminergic value** (RL, incentive salience) | Successor-features head predicts discounted future reservoir change; composed with deficit-gated innate weights it yields incentive salience; TD(λ) trains it, and it shapes the policy toward relief. | `nn/successor_features.py`, `consolidation/{returns,episodes,imagination,consolidator}.py`, `cycle/neural_pipeline.py` (value shaping) | `test_successor_features.py`, `test_episodic_returns.py`, `test_homeostatic_reward.py`, `test_consolidation.py` | ✅ |
| **Dorsolateral striatum** (habit system / model-free control) | **WS-EXPAND E4:** a fast cached "habit" head, distilled online from the deliberate policy's actions on *escalated cycles only* (teacher outputs only — no self-training). On gate-skip cycles the command blends toward the habit by a **trust weight earned from distillation quality** — zero at birth, and it melts the moment the habit stops matching the teacher. The deliberate path is thus freed for novelty. Trust reached 0.75 live within 10 min (flagged for soak review). | `nn/cached_policy.py`, `nn/neural_stack.py` (`cached_l1/l2`, `cached_action`), `cycle/neural_pipeline.py` | `test_cached_policy.py` | 🟢 |
| **Neuromodulatory systems** (dopamine / acetylcholine / norepinephrine / serotonin) | Was a single `pleasure − pain` scalar gating Hebbian plasticity. **WS-EXPAND E2** split it into **four differentiated channels** computed from lived signals: **reward** (dopamine → weight-update sign/magnitude, the original), **expected-uncertainty** (acetylcholine → learning rate, with a volatility-vs-noise separation so it doesn't chase noise), **surprise** (norepinephrine → transient rate boost + raises gate escalation), and **horizon/patience** (serotonin → modulates the successor discount γ inside a hard clamp band, rate-limited). Live: eta-scale 1.96 early, γ walked 24 moves to 0.9932. | `nn/learning_control.py`, `nn/plastic.py`, `cycle/neural_pipeline.py` | `test_learning_control.py`, `test_plasticity.py` | 🟢 |
| **Hippocampus + entorhinal** (episodic memory + cognitive map) | Episodic store (LanceDB) + a semantic/relational graph (Kuzu) of entities, spatial relations, and property beliefs; spatial recall converts a remembered location into an egocentric bearing. **Write governance** (2026-07-06) keeps the store durable without drowning the flusher. **WS-EXPAND E1** makes the *cognitive map* real: dead-reckoning **pose estimation** corrected by landmark re-sighting (place/grid-like localization), a learned **positional code** fed to the policy (goal vector grew 12→16), an **experiential breadcrumb graph** with measured traversal costs, and **stall-gated A\* waypoint planning** that reroutes through walked space only once the direct route is *evidenced* blocked. Ran 2485 pose updates live. | `memory/{kuzu_graph,semantic_graph,episodic_store,lancedb_store,ltm_write_behind}.py`, `state/cognitive_map.py`, `state/spatial_recall.py`, `state/world_graph.py`, `nn/goal_conditioning.py` | `test_ws4_backends.py`, `test_semantic_graph.py`, `test_cognitive_map.py`, `test_spatial_recall.py`, `test_property_beliefs.py`, `test_world_graph.py` | 🟢 |
| **Insular cortex** (interoception → affect) | **WS-EXPAND E8.1:** a learned **interoceptive embedding** over (reservoirs + tactile load + effort) folded into the affect path through a **zero-init ingress**, so affect becomes a readout of felt body-state, not only the cognitive latent. Ships behind an A/B (birth-identical until it earns its keep). | `nn/neural_stack.py` (`intero_embed_*`, `intero_embedding`), `cycle/neural_pipeline.py` | `test_ws_expand_tail.py` | 🟢 |
| **Sleep / memory consolidation** (hippocampal replay, dreaming) | A dual-network consolidator replays a salience-prioritized buffer, runs imagined rollouts ("dreaming") to train value off-line, with a loss-landscape probe. **WS-EXPAND E7** adds **scheduled rest**: a load accumulator (active cycles + prediction-error), bounded by wake time (value-drift guard), idles the body while learning continues; any threat aborts rest instantly. **E8.2** tags replay salience with a bounded valence multiplier (emotional-salience-weighted consolidation). Rest entered + woke cleanly live. | `consolidation/{consolidator,replay_buffer,imagination,landscape,rest}.py`, `cycle/neural_pipeline.py` (salience blend, rest gate) | `test_consolidation.py`, `test_landscape.py`, `test_ws_expand_tail.py` | 🟢 |
| **Limbic / amygdala** (affect, emotion, threat) | Emotion head + pain/pleasure scalars on the state bus; predictive affect anticipates the next-step affect and colors perception. **WS-EXPAND E5.1:** **aversive prediction** — a remembered threat (`predicts_pain` / `predicts_integrity_loss` beliefs with a known position) produces an egocentric **avoidance bearing** on the goal vector, scaled by belief strength *and* an urgency override (a starving agent re-tests a stale threat rather than dying behind it — extinction-lite). Avoidance is *learned* through the ingress, not hardcoded. | `nn/affect_model.py`, `state/state_bus.py`, `state/spatial_recall.py` (`resolve_threat_target`), `nn/goal_conditioning.py` (threat slots), `cycle/neural_pipeline.py` | `test_affect_bounds.py`, `test_predictive_affect.py`, `test_goal_conditioning.py`, `test_ws_expand_tail.py` | 🟢 |
| **Intrinsic motivation** (curiosity / exploration drive) | A learning-progress ("frontier of the learnable") signal drives exploration when needs are met. | `state/curiosity.py` | `test_curiosity.py`, `test_curiosity_logging.py` | ✅ |

---

## IV. Social & symbolic cortex (WS-EXPAND, next-horizon)

Higher-order associative systems added by WS-EXPAND. Each is built and
live-verified, but each has a deliberately-held second half that waits on a
precondition (a social scene, or a shared symbol channel) — see "Held by
design" below.

| Brain part | How Deca mimics it | Where (modules) | Tests | Status |
|---|---|---|---|---|
| **Theory of mind** (temporoparietal junction / mentalizing) | **WS-EXPAND E10:** a per-entity predicted-state model behind an **adaptivity gate** — every perceived entity gets a ballistic (constant-velocity) motion prior, and only an entity whose movement *repeatedly defeats* that prior is classified "adaptive" and modeled. So a solo scene or scripted prop spawns **zero** models (verified live: 0 tracks, 0 models). The self-model architecture is reused for the other-model; its policy ingress + imitation-from-observation wait on a two-agent arena. | `state/other_agents.py`, `agents/runtime.py` | `test_ws_expand_tail.py` | 🟢 |
| **Symbolic abstraction** (categorical / discrete concepts) | **WS-EXPAND E9:** a **finite-scalar-quantization** bottleneck (no learned codebook, so no collapse) maps a detached `z5` latent onto a small grid of composable discrete codes (4800-code product grid), with a next-code self-supervised dynamics head. Gradients never reach the shared trunk, so behavior is byte-identical — the abstraction layer trains purely on the side. Grounding accrues as codes co-occur with `predicts_*` beliefs. Live utilization 0.21. | `nn/symbol.py`, `nn/neural_stack.py` (`fsq_in`, `fsq_next`) | `test_ws_expand_tail.py` | 🟢 |
| **Neural oscillations** (theta / gamma timing) | **WS-EXPAND E13:** slow and fast phase variables logged as **instrumentation only** — deliberately *not* coupled to any decision path, because the evidence review found no task benefit for phase-scheduled control. Kept for integration-measurement work. | `cycle/neural_pipeline.py` (phase telemetry) | probe telemetry | 🟢 |

---

## V. Body & environment (not brain, but the loop it closes)

| System | How | Where | Tests | Status |
|---|---|---|---|---|
| **Body / embodiment** | 21-actuator MuJoCo humanoid; stances, joint braces, contact-gated consumption. | `scripts/mujoco_decadic_adapter.py`, `embodiment/*.py` | `test_scenes.py`, `test_embodied_motor.py`, `test_crowd.py`, `test_npc.py` | ✅ |
| **Development / curriculum** (caregiver, teaching) | Skill Dojo teacher-guided curricula and a forage curriculum (survival net + within-reach placement). | `training/*.py`, `api/environment.py` | `test_skill_dojo.py`, `test_walking_curriculum.py`, `test_environment_supervisor.py`, `test_training_gates.py` | ✅ |

---

## Notes on status

- **🟢 WS-EXPAND regions** (differentiated neuromodulation, cerebellar corrector
  + CPG, cognitive map + prospection, insula, striatal habit system, amygdalar
  threat + prefrontal veto, thalamic-reticular gate, sleep, theory of mind,
  symbolic abstraction, oscillations): unit-tested across
  `test_learning_control.py`, `test_cognitive_map.py`, `test_action_planner.py`,
  `test_motor_corrector.py`, `test_cached_policy.py`, `test_ws_expand_tail.py`,
  and `test_goal_conditioning.py`, then **live-verified on the body 12/12** in a
  10-minute probe (`scripts/run_ws_expand_probe.ps1`) at 4.18 cyc/s. All ship
  default-on and birth-identical (zero/identity-init).
- **🧪 Speech (Broca/Wernicke):** WS6 is specced with the audio lane, vocal
  tract, and intake partially built; it is not yet a closed learn-to-speak loop.
- Full suite last green at ~985 passed / 7 skipped / 1 xpassed.

## Held by design (built, but a second half awaits a precondition)

Per the WS-EXPAND evidence review, a few pathways are intentionally *not*
finished — the evidence says they add nothing until a precondition exists:

- **Habit staleness / arbitration thrash (E4):** guarded by the existing
  surprise channel (forces re-escalation when the world shifts) + gate
  hysteresis; the live trust-vs-divergence curve is flagged for the next soak.
- **Theory of mind — policy ingress + imitation-from-observation (E10.3–.5):**
  wait on a two-agent arena; useless (net overhead) while solo, which is exactly
  why the adaptivity gate keeps them dark.
- **Symbolic *language* loop (E12):** expression/comprehension via the audio
  lane waits on a listener — emergent codes are non-compositional without
  communicative pressure. The discrete substrate (E9) is live now.
- **Goal-stack hierarchy (E11):** held until it beats a flat policy + exploration
  bonus on a detour task (the literature's null result for hierarchy).

## What a *fuller* neocortex would add

Two things, neither of which is "more stages": a **canonical repeated
microcircuit** (cortical uniformity — one column tiled across modalities) and a
**learned, not frozen, sensory hierarchy**, so perception itself is grown from
lived experience rather than inherited from CLIP/Whisper. Today the encoders are
the retina/cochlea (transduction you may inherit); the true cortex — the part
that must be *lived* — is everything trained above them.
