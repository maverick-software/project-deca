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

### Test-status legend

| Symbol | Meaning |
|---|---|
| ✅ | Dedicated tests present and passing (full suite last green at 875 passed / 7 skipped). |
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
| **Cerebellum** (forward models / predictive motor) | Learned forward models predict next proprioception, tactile load, effort, and interoception from an efference copy — enabling active inference (planning through a frozen world model). | `nn/neural_stack.py` (`fwd_*`), `cycle/neural_pipeline.py` | `test_embodied_motor.py`, `test_neural_cycle.py` | ✅ |
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

---

## III. Subcortical structures

The drive, value, memory, and affect systems the cortex sits on top of — what
makes "does it have a neocortex" a meaningful question.

| Brain part | How Deca mimics it | Where (modules) | Tests | Status |
|---|---|---|---|---|
| **Hypothalamus / brainstem** (homeostasis, drives, survival) | Hydration/energy/integrity reservoirs draining on a real wall-clock; viability = their minimum; mortality when one bottoms out; drive pressure rises with deprivation. | `state/viability.py`, `state/curiosity.py`, homeostasis in `agents/runtime.py` | `test_viability.py`, `test_homeostasis.py`, `test_homeostatic_drive.py`, `test_mortality.py` | ✅ |
| **Basal ganglia / dopaminergic value** (RL, incentive salience) | Successor-features head predicts discounted future reservoir change; composed with deficit-gated innate weights it yields incentive salience; TD(λ) trains it, and it shapes the policy toward relief. | `nn/successor_features.py`, `consolidation/{returns,episodes,imagination,consolidator}.py`, `cycle/neural_pipeline.py` (value shaping) | `test_successor_features.py`, `test_episodic_returns.py`, `test_homeostatic_reward.py`, `test_consolidation.py` | ✅ |
| **Neuromodulatory systems** (dopamine/serotonin) | Three-factor plasticity: Hebbian updates gated by a `pleasure − pain` neuromodulator, so what the agent did when relief arrived is reinforced. | `nn/plastic.py`, `cycle/neural_pipeline.py` (modulation) | `test_plasticity.py`, `test_plasticity_stability.py` | ✅ |
| **Hippocampus + entorhinal** (episodic memory + cognitive map) | Episodic store (LanceDB) + a semantic/relational graph (Kuzu) of entities, spatial relations, and property beliefs; spatial recall converts a remembered location into an egocentric bearing. **Write governance** (2026-07-06) keeps the store durable without drowning the flusher. | `memory/{kuzu_graph,semantic_graph,episodic_store,lancedb_store,ltm_write_behind}.py`, `state/spatial_recall.py`, `state/world_graph.py` | `test_ws4_backends.py`, `test_semantic_graph.py`, `test_episodic_recall_cache.py`, `test_property_beliefs.py`, `test_spatial_recall.py`, `test_world_graph.py` | 🟡 |
| **Sleep / memory consolidation** (hippocampal replay, dreaming) | A dual-network consolidator replays a salience-prioritized buffer, runs imagined rollouts ("dreaming") to train value off-line, with a loss-landscape probe. | `consolidation/{consolidator,replay_buffer,imagination,landscape}.py` | `test_consolidation.py`, `test_landscape.py`, `test_memory_efficient_training.py` | ✅ |
| **Limbic / amygdala** (affect, emotion) | Emotion head + pain/pleasure scalars on the state bus; predictive affect anticipates the next-step affect and colors perception. | `nn/affect_model.py`, `state/state_bus.py`, `cycle/neural_pipeline.py` | `test_affect_bounds.py`, `test_predictive_affect.py` | ✅ |
| **Intrinsic motivation** (curiosity / exploration drive) | A learning-progress ("frontier of the learnable") signal drives exploration when needs are met. | `state/curiosity.py` | `test_curiosity.py`, `test_curiosity_logging.py` | ✅ |

---

## IV. Body & environment (not brain, but the loop it closes)

| System | How | Where | Tests | Status |
|---|---|---|---|---|
| **Body / embodiment** | 21-actuator MuJoCo humanoid; stances, joint braces, contact-gated consumption. | `scripts/mujoco_decadic_adapter.py`, `embodiment/*.py` | `test_scenes.py`, `test_embodied_motor.py`, `test_crowd.py`, `test_npc.py` | ✅ |
| **Development / curriculum** (caregiver, teaching) | Skill Dojo teacher-guided curricula and a forage curriculum (survival net + within-reach placement). | `training/*.py`, `api/environment.py` | `test_skill_dojo.py`, `test_walking_curriculum.py`, `test_environment_supervisor.py`, `test_training_gates.py` | ✅ |

---

## Notes on status

- **🟡 Hippocampus/memory persistence:** the graph-write governance and coalesce
  logic (2026-07-06) fixed the flusher saturation that stalled long runs. A
  promotion-order bug it introduced was caught by `test_ws4_backends.py` and
  fixed; those two tests (`test_ws4b_write_governance_*`,
  `test_kuzu_backup_restore_roundtrip`) are green in the corrected code but
  awaiting your next full re-run to re-confirm alongside the rest of the suite.
- **🧪 Speech (Broca/Wernicke):** WS6 is specced with the audio lane, vocal
  tract, and intake partially built; it is not yet a closed learn-to-speak loop.
- Everything else last ran green (875 passed / 7 skipped / 1 xpassed).

## What a *fuller* neocortex would add

Two things, neither of which is "more stages": a **canonical repeated
microcircuit** (cortical uniformity — one column tiled across modalities) and a
**learned, not frozen, sensory hierarchy**, so perception itself is grown from
lived experience rather than inherited from CLIP/Whisper. Today the encoders are
the retina/cochlea (transduction you may inherit); the true cortex — the part
that must be *lived* — is everything trained above them.
