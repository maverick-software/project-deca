# Logical-Layer Conformance Map

A living map from the six necessary conditions of the **logical layer** in
[The Extra Ingredient: A Structural Solution to the Hard Problem of Consciousness](The%20Extra%20Ingredient_%20A%20Structural%20Solution%20to%20the%20Hard%20Problem%20of%20Consciousness.md)
(conditions enumerated at lines 356–363; Existential Entanglement at lines 391–396)
to the modules that realize each condition in this codebase.

This document is descriptive of the *current implementation*. Where we deliberately
deviate from the paper (notably mortality), the deviation and its rationale are
recorded below.

## The six conditions

| # | Condition (paper) | Where it lives | Notes |
|---|-------------------|----------------|-------|
| 1 | **Representational structure** — a relational graph of nodes *and edges*, not a flat feature list | `decadic/state/world_graph.py` (`egocentric_graph_from_world_state`, `edges_from_nodes`) | Emits `{"nodes": [...], "edges": [...]}` with typed `spatial`, `proximity`, `affective`, and `context` edges. |
| 2 | **Self-indexing** — a privileged self-node that is the origin of the representation | `world_graph.py` (`role: "self"` node) + every edge is indexed to the self-id | Spatial/affective/context edges all originate at `self`; the dashboard renders the graph radially around it. |
| 3 | **Temporal persistence** — state carried forward rather than rebuilt each moment | `decadic/state/working_memory.py` (`WorkingMemory` bounded decaying slots) | The graph is sourced from working-memory slots, so entities persist out of view (object permanence) and fade by salience decay instead of disappearing. |
| 4 | **Integration dynamics** — a workspace that binds inputs into one coherent state | `decadic/cycle/neural_pipeline.py` (attention-weighted blend of `WorkingMemory.attention_vector` into State Bus **A**) + the ten-stage Decadic cycle | Salience-weighted, affect-signed working-memory summary is fused into `state_of_mind` each cycle (Global Workspace, paper lines 1193–1204). |
| 5 | **Affective structure** — edges connecting stimuli to the self's survival concerns | `world_graph.py` (`update_entity_affect`, affective edges) + `decadic/state/perceptual_state.py` (`entity_affect` table) | A decaying per-entity valence table is updated from events (`collision`/`fall`/`threat_near` → negative, `food` → positive) and exposed as self→entity affective edges colored by valence. |
| 6 | **Sufficient complexity** — the self is modeled as a thing among things, carrying history/affect | `working_memory.py` (slots carry salience, affect, `last_seen_cycle`, `seen_count`) + State Bus persistence in `decadic/state/state_bus.py` | The self-node sits in the same graph as other entities; working memory gives entities (and the scene around the self) a tracked history. |

## Parallel sessions (workspace throughput)

Condition 4 also implies a workspace with bounded capacity. `neural_pipeline.encode_observations`
runs up to **K** buffered observations through the frozen encoders in one `no_grad`
pass, decoupled from the single serialized, gradient-bearing learn step. K is the
"parallel-session budget"; observations are buffered in `AgentRuntime._obs_buffer`
(a `deque(maxlen=K)`), drained once per cognitive cycle. K, the working-memory slot
count **S**, and the salience **decay** are configurable:

- env: `DECADIC_PARALLEL_SESSIONS`, `DECADIC_WORKING_MEMORY_SLOTS`, `DECADIC_WM_DECAY`
  (defaults in `decadic/config.py`)
- live: `POST /agent/{id}/config?parallel_sessions=&working_memory_slots=&working_memory_decay=`
  (and the dashboard **Workspace Capacity** panel)

## Perception feedback loops (history shapes perception)

Conditions 3 (temporal persistence) and 4 (integration) imply that perception is
not a fresh, context-free read each cycle: prior state should shape the current
percept, the way it does in biological cognition (predictive coding / Bayesian
perception). Two **learned** loops close this, gated behind a single ablation flag
(`DECADIC_PERCEPTION_FEEDBACK_ENABLED`, default off → byte-identical baseline).
Nothing here is hand-steered: every new pathway is a trainable module optimized by
the **existing self-supervised objective**; the only non-learned choices are safe
initializations (which gradient descent may override) and the enable flag.

### Loop 1 — Precision-gated top-down predictive perception

`decadic/nn/neural_stack.py` (`top_down`, `precision_gate`, `top_down_perceive`) +
`decadic/cycle/neural_pipeline.py`. From **detached** history (the previous cycle's
`z5` via `bundle.prev_state`, the LSTM hidden state, the working-memory
`scene_latent`, and the just-recalled memory `mem_t`) the stack predicts the percept
`z0_hat`, and blends it with the bottom-up encode under a learned precision gate in
predictive-coding form:

```
z0_eff = z0_hat + gate * (z0 - z0_hat)
```

The gate additionally sees interoception (pain / pleasure / viability), so the
network can *learn* to weight senses vs. priors under threat rather than being told
to. A self-supervised term `l_percept = mse(z0_hat, z0.detach())` (weight
`DECADIC_PERCEPTION_PRED_WEIGHT`) teaches top-down to predict perception from
history; the **untouched** `pc_loss` surprise term remains, which is the structural
guard against self-confirming hallucination. History is detached, so the loop shapes
*this* cycle's percept without opening a cross-cycle BPTT path (no gradient runaway).
Init is near-parity (gate ≈ 1 via `DECADIC_PRECISION_GATE_INIT`, `top_down` ≈ 0).
Telemetry: `precision_gate_mean`, `perceptual_pred_error`.

### Loop 2 — Perceptual-similarity episodic retrieval

`decadic/memory/embeddings.py` (`perceptual_key`, appended to both
`query_vector_from_state_bus` and `episode_embedding_from_cycle`; `EMBEDDING_DIM`
raised by `PERCEPT_KEY_DIM`). Episodes are recalled not only by internal-state
similarity but by **sensory likeness**: a parameter-free, L2-normalized compression
of the *learned* percept `z0` is stored with each episode and used in the query. The
similarity therefore reflects CLIP/`z0`'s learned geometry, not any hand feature.
Pre-upgrade rows of the old length auto-skip via the existing length check in
`search_similar`; with the flag off the key is zeros, so cosine ranking is identical
to the pure internal-state embedding (appending equal zeros to both sides is a no-op).

### Loop 3 — Affect generalization (emergent; explicit non-goal)

Cross-stimulus affect transfer — e.g. a never-seen bear inheriting the fear learned
from a cat — is **not** implemented as a symbolic per-kind affect table. Such a rule
would be exactly the kind of programmed steering this experiment forbids, so it is
deliberately **out of scope** as a standalone mechanism. Instead it is expected to
*emerge* from Loops 1+2: a novel bear sits near a cat in the learned `z0`/CLIP space,
so (a) Loop 2 recalls the cat's affect-laden episode by perceptual likeness, and
(b) Loop 1 perceives the bear partly through that recalled prior — while the retained
bottom-up `pc_loss` term lets direct experience flatten any over-generalization. No
code asserts "bear == cat"; the association, if it forms, is a property of learned
weights and learned similarity.

## Perception-derived world graph (emergent structure, learned self)

By default the egocentric graph is *handed* to the agent: `egocentric_nodes_from_world_state`
reads the simulator's `world_state.entities` (oracle ids, kinds, positions). That is a
scaffold, not genuine discovery — the relational structure of conditions 1–3 is given
rather than recovered from the agent's own senses. `DECADIC_PERCEPTION_MODE=discovered`
(default `oracle` ⇒ byte-identical baseline) replaces the scaffold with a graph that
emerges from the egocentric camera, proprioception, and memory. It requires
`DECADIC_ENCODER_MODE=hf` (real CLIP). Per-agent override is live via `configure()` /
`POST /agent/{id}/config?perception_mode=`, and `GET /agent/{id}/discovery` exposes the
evaluation snapshot. As with the other subsystems, nothing here is hand-steered: slot
attention, data association, and the agency head are trainable modules optimized by
self-supervised objectives, and the oracle entity list never enters cognition.

**Condition 1 (representational structure) becomes genuinely emergent.** Frozen CLIP
emits a 7×7 grid of patch features (`frozen_encoders.vision_patch_tokens`); `decadic/nn/slots.py`
(`SlotAttention`, Locatello-style) lets `K` slots compete to explain that grid, so each
slot binds to a coherent region — an *object proposal* the agent parsed for itself. A
spatial-broadcast decoder reconstructs the feature map (DINOSAUR-style feature
reconstruction, `l_slot` in `neural_pipeline.py`): no pixel labels, no oracle. Pooled
slots are injected additively into the ingress latent through a zero-initialized
projection (`slot_ingress`), so the deliberative stack sees object structure while
starting at exact bottom-up parity.

**Condition 2 (self-indexing) is learned, not assigned.** The self node is sensed from
proprioception (`_self_node_from_proprio`), not read from an oracle `agent` blob. Body
parts are *discovered*: `decadic/nn/agency.py` (`AgencyHead`) predicts each tracked
slot's frame-to-frame image motion from the efference copy (`prev_motor`) versus an
efference-blind baseline; the error reduction is a comparator-model sense-of-agency
signal (Blakemore/Frith). Persistent high-agency slots are promoted to `kind="self_part"`
nodes joined to the self by a learned `agency` ("this is mine") edge — exactly how
ownership is acquired developmentally — and coincident touch `contacts` strengthen it.

**Condition 3 (temporal persistence) becomes real object permanence.** Discovered
proposals carry no identity. `WorkingMemory.integrate_discovered` performs greedy data
association by appearance cosine (an EMA'd slot fingerprint) plus constant-velocity
predicted image position; matches reinforce an existing object file, misses coin a fresh
anonymous `obj-NNNN` id. "Remembered = seen-before-but-not-in-view" now has real meaning,
since a slot only exists once the agent has perceived it.

**Oracle demoted to eval-only ground truth.** In discovered mode `world_state.entities`
are stashed in `PerceptualState.oracle_truth` and used *only* by
`decadic/perception/discovery_metrics.py` (`DiscoveryEvaluator`) to score detection
precision/recall (greedy egocentric-direction match), identity stability (id churn), and
body-part agency accuracy (discovered `self_part`s vs. the adapter's eval-only hand/foot
`xpos`, which `runtime` strips before cognition). The score never influences the agent.

## Homeostatic drive reduction (root survival motivation)

The motivational loop is closed so the agent feels deprivation as innate pain and can
learn, on its own, that acting reduces its *predicted* internal drive. This is the
**root drive** (drive #1): always on whenever a body streams reservoirs — there is no
feature flag. The bright line is **phylogeny vs. ontogeny**: we provide only what
phylogeny gives a newborn, and the satisfier (water) is discovered through ontogeny —
the agent's own experience — never taught.

**Innate substrate (what we provide).** Three pieces, none of which references an
external object:

- **Deprivation is aversive.** `interoceptive_drive_pain` in
  `decadic/state/viability.py` maps the largest fractional deficit below a comfort
  setpoint to a bounded pain scalar (drive theory: urgency tracks deprivation *level*,
  not the per-tick delta). It is folded into the same `pain_scalar` / B-affect channel
  as the existing PE/reward pain in `decadic/cycle/neural_pipeline.py`. The fastest
  drainer (thirst) hurts first; this is the bare "this state is bad" prior.
- **A full-reservoir setpoint.** `preferred_intero_vector` = `[1,1,1]` (the innate
  homeostatic prior), with equal `intero_preference_weights` so the most-deprived
  reservoir dominates the gradient on its own (emergent prioritization, not a hardcoded
  ranking).
- **The capacity to predict internal state.** An interoceptive world model
  (`forward_predict_intero` in `decadic/nn/neural_stack.py`, built unconditionally)
  predicts the next normalized reservoir vector from `(state, efference copy, current
  reservoirs)`.

**What is learned (ontogeny).** `l_fwd_intero` trains the interoceptive world model on
*realized* transitions (the agent's own `(state, action) → next reservoirs`), and
`l_pref_intero` pulls the policy toward whatever that model predicts will raise depleted
reservoirs toward the full setpoint — through a **detached** world model, so the policy
cannot hallucinate relief by editing its own predictions. The contingency "being near the
blue percept and moving onto it precedes hydration rising" is thus learned from experience,
not asserted.

**Proprioceptive world model (self-model only, not a goal).** The proprioceptive forward
model (`l_fwd` / `s_hat` in `decadic/cycle/neural_pipeline.py`) is retained: it learns to
predict the body's next controllable state from `(state, action)` transitions. There is
**no** proprioceptive preferred-state term — nothing directly rewards standing upright.
Uprightness can only emerge as *instrumental* to survival (falls damage integrity →
viability → pain). The external locomotion assist harness (fading training wheels in the
adapter) is physical support only; it does not enter cognition or the loss.

**Emergent water-seeking (explicit non-goal as a mechanism).** Directed seeking of water
is the *appetitive* analogue of the documented cat→bear affect transfer: an expected
emergent property of the loops, **not** a coded behavior. There is deliberately **no**
`drink → hydration` rule in the policy, **no** reward shaped toward water, and **no**
"water" label reaching cognition. Purity was verified end to end: vision is unlabeled CLIP
pixels (`FrozenSensoryEncoders.forward` consumes only `vision`/`audio`/`proprioception`);
the `{"type":"water"}` scene event is consumed only by `classify_events()` as a hydration
credit (metabolism), never embedded in a neural channel; reservoirs enter cognition only as
the normalized scalar interoceptive vector; and episodic embeddings carry only internal-state
vectors plus a parameter-free perceptual key. The symbolic `kind` string ("water") lives only
in working-memory/world-graph bookkeeping and UI snapshots — arbitrary entity *ids* may be
hashed as uninterpreted landmarks (like place cells), but the decodable *category* never
feeds the State Bus.

**Caveat.** A one-step interoceptive forward model is myopic, so bridging the delay between
approaching and drinking relies on the recurrent `z5`/LSTM context and the episodic/perceptual
loops. The promise here is to provide the *substrate* for emergent seeking, not to guarantee it
on a fixed schedule; natural follow-ups (a short multi-step rollout or a bootstrapped
interoceptive value) remain fully self-learned and are out of scope.

## Curiosity — an innate epistemic prior (substrate provided, content learned)

The homeostatic drive above answers "act to stay alive"; it gives a *safe, sated* agent no
reason to investigate the world. The same phylogeny-vs-ontogeny logic supplies one further
innate motivational substrate — a **need-gated curiosity** drive
(`decadic/state/curiosity.py`, on by default in production via `DECADIC_CURIOSITY_ENABLED`;
set to `0` for a byte-identical no-curiosity baseline). Like hunger, it is an innate *prior* over what feels rewarding; like
hunger, *what* turns out to be interesting is never coded — only the capacity to find one thing
rewarding is.

**Innate substrate (what we provide).** Curiosity rewards **learning progress** —
`learning_progress()` measures the relative *fall* of forward-model prediction error over a short
rolling window (`DECADIC_CURIOSITY_PROGRESS_WINDOW`), clamped to `[0,1]`. Rewarding error
*reduction* rather than raw surprise is the structural guard against the **noisy-TV trap**: an
irreducibly random stimulus yields no learning progress, so it earns no curiosity pleasure. A
small current-error floor (`epistemic_opportunity()`) keeps a flat-but-wrong state probing rather
than going numb.

**Need-gating (the substrate is conditional).** `survival_urgency()` combines pain and low
viability into a `[0,1]` threat/deprivation signal, and `permission()` makes curiosity fall off
sharply as urgency rises (`DECADIC_CURIOSITY_SAFETY_SHARPNESS`). The drive is therefore expressed
only when the agent is *safe and sated* and suppressed when it is threatened or deprived — the
arbitration Stage 4 is meant to perform, realized as an innate gate rather than a coded rule.

**How it enters cognition (parity-preserving).** When enabled, the gated `(pain=0, pleasure)`
scalar is folded into the **same** pleasure-side B-affect channel as the existing PE/reward
pleasure in `decadic/cycle/neural_pipeline.py` (it floors `pleasure_scalar`, mirroring how
homeostatic drive-pain enters on the aversive axis), is **added** to the `motor_exploration_sigma`
drive term so satisfied curiosity lets exploratory babble relax, and flips the priority label
`explore → investigate` when the epistemic opportunity clears a small threshold and the agent is
not avoiding. Its per-cycle PE history lives on an **ephemeral** `bundle._curiosity` (never
checkpointed); with the flag off the whole block is skipped, so the cycle, weights, and telemetry
are identical to baseline. Telemetry (`curiosity_drive`, `curiosity_pleasure`,
`curiosity_learning_progress`) is `None` when off.

## Consolidation — replay-based long-term learning (dual-network)

Conditions 3 (temporal persistence) and 6 (sufficient complexity) imply that experience should
durably shape the self over time, not just within a cycle. Beyond the episodic store (which
*remembers*), a **dual-network consolidation** subsystem lets the agent *re-learn* from its own
salient past offline, the functional analogue of replay during rest
(`decadic/consolidation/`, on by default in production via `DECADIC_CONSOLIDATION_ENABLED`;
set to `0` to keep the no-op stub heartbeat and byte-identical live weights).

- **Salience-prioritized replay buffer** (`replay_buffer.py`). Each live cycle pushes its realized
  transition — the detached latents needed to recompute the cycle's own self-supervised losses
  (`z0`, episodic/memory context, previous state and efference copy, the proprioceptive and
  interoceptive targets) tagged with a **salience** (prediction-error / affect intensity). The
  buffer is bounded and **evicts the lowest-salience transition when full** — built-in forgetting,
  with a `DECADIC_CONSOLIDATION_PRUNE_MIN_SALIENCE` floor — and `sample()` draws high-salience
  transitions preferentially.
- **Cloned consolidator** (`consolidator.py`, `ConsolidationManager`). A second copy of the
  cognitive stack recomputes the **same** PC / forward-model / interoceptive losses on replayed
  transitions and steps its **own** optimizer, then **Polyak soft-syncs** into the live stack
  (`θ_live ← (1-τ)·θ_live + τ·θ_consolidator`, `DECADIC_CONSOLIDATION_SYNC_TAU`) under the agent
  lock so the live cycle never observes a half-written weight set. Replay bursts run in a thread
  executor (`asyncio.to_thread`) so the cognitive cycle is never blocked. The consolidator is
  ephemeral (re-cloned on resume), so checkpoints are unaffected. Telemetry: `replay_count`,
  `replay_buffer_size`, `consolidator_loss`, `last_sync_cycle`.

Nothing here is hand-steered: replay optimizes the existing self-supervised objective on the
agent's own experience; the only non-learned choices are the buffer's salience priority, the sync
rate, and the enable flag.

## Mortality — Existential Entanglement (deliberate, reversible variant)

The paper's strict artificial-consciousness clause (Existential Entanglement, lines
391–396) requires a system to be **structurally autopoietic** and **fragile**: its
informational integrity coupled to its physical integrity, capable of *permanent*
cessation or corruption of its active self-model.

**We implement a reversible variant.** When viability reaches 0:

- the agent's `status` flips to `dead` and the cognitive cycle freezes
  (`AgentRuntime.die()` in `decadic/agents/runtime.py`);
- neural weights are **retained in memory** (frozen, not erased);
- a one-time **tombstone checkpoint** is written to the backups directory
  (`agent_{id}_brain.pt` + `agent_{id}_tombstone.json` marked `"status": "dead"`)
  for study and reincarnation;
- a `death` event is emitted on the agent's outbound queue.

Administrators may then:

- **Revive** (`revive()` / `POST /agent/{id}/revive`): restore viability and resume
  the *same* mind — identical weights, state bus, working memory, and episodic memory.
- **Reincarnate** (`reset()` / `POST /agent/{id}/reset`): a fresh mind with new weights
  and wiped memory; also valid from the `dead` state.

### Why we deviate

We intentionally do **not** implement irreversible weight-erasure on death:

1. **Administrability** — operators must be able to recover an agent that died from a
   tuning mistake or an environment bug without losing the trained mind.
2. **Study** — a frozen, inspectable corpse (the tombstone) is more scientifically
   useful than an erased one; it lets us compare pre-death weights/state across runs.
3. **Safety** — reversibility keeps a human in the loop for the strongest claim in the
   paper rather than baking irreversible self-destruction into the runtime.

The strict autopoietic+fragile clause is therefore **out of scope** by design; this
section is the record of that choice. The lifecycle state machine is documented in the
implementation plan (`.cursor/plans/logical_layer_upgrade_*.plan.md`).

## Visualization (the layer "must be modeled to be seen")

The paper stresses that the logical layer must be *modeled* to be observed (lines 33,
424). The dashboard is that instrument:

- `dashboard/src/components/GraphPanel.tsx` — self-centered node+edge graph; node
  brightness = salience, affective edges colored by valence, proximity edges dashed,
  learned `agency` ("mine") edges in violet, and `self_part` body-part nodes set apart.
- `dashboard/src/components/CapacityPanel.tsx` — K/S/decay sliders, the
  `perception_mode` (oracle/discovered) selector, and live cycles/s, session, slot,
  encode-time, and GPU-memory readouts.
- `dashboard/src/components/DiscoveryPanel.tsx` — coined object files in image space
  with presence/salience, agency scores, and which slots are flagged "mine"; and
  `dashboard/src/components/EvalPanel.tsx` — discovered-vs-oracle precision/recall,
  id-stability, and body-part accuracy (eval-only, never fed to cognition).
- `dashboard/src/components/AgentControls.tsx` + `App.tsx` — `dead` badge, death
  banner, and a **Revive** button (with **Reincarnate** replacing **Reset**).
- `dashboard/src/components/CognitionPanel.tsx` — the **Cognition / Why** tab (see
  below): per-cycle survival-intent decomposition, input attribution, self-model
  surprise, episodic grounding, interpretability probes, and a narrative.

## Observability — translating the layer into "why" (the Cognition panel)

Modeling the graph shows *what* the self-indexed representation is; the **Cognitive
Trace** translates the opaque State Bus latents into *why the agent acted*. It is
assembled read-only each cycle in `decadic/cycle/cognition_trace.py` (from tensors
the cycle already computes), rendered to prose in `decadic/cycle/narrative.py`,
decoded by `decadic/interpretability/probes.py`, surfaced at `GET /agent/{id}/explain`
and in `snapshot_state()`, and drawn by `CognitionPanel.tsx`. Each of the six
conditions gets a concrete read-out:

| # | Condition | Cognition-panel read-out |
|---|-----------|--------------------------|
| 1 | Representational structure | **Attribution** block: `d|motor_u|/d(input)` split across perception / affect / memory, plus the most-attended working-memory node (the graph node driving the action). |
| 2 | Self-indexing | **Intent** decomposition is expressed entirely in the self's own survival terms (its reservoirs), and the salient node is named relative to the self origin. |
| 3 | Temporal persistence | **Temporal trace** sparklines (pain / risk / surprise over the `cognitive_history` ring buffer) + **episodic grounding** (the most-similar past episode and how it ended). |
| 4 | Integration dynamics | The whole trace is one bound record per cycle; **attribution fractions** show how perception, affect, and memory were integrated into the single emitted command. |
| 5 | Affective structure | **Affect** block (pain / pleasure / risk / priority) and the affect-signed salience of the attended node. |
| 6 | Sufficient complexity | **Self-model surprise** (predicted vs. realized body transition, per dimension) + **interpretability probes** decoding latents into interpretable variables, each labeled with held-out quality (R² / accuracy). |

The **intent** read-out is the load-bearing one: the agent's live objective is the
homeostatic interoceptive drive (keep hydration / energy / integrity near full), so
each cycle the panel asks the agent's *own* world model what the emitted command is
predicted to do to each reservoir versus standing still (`forward_predict_intero`),
and ranks the drivers. The **counterfactuals** view runs the same frozen world model
over alternative motor commands to show the decision landscape the policy is
implicitly optimizing.

### Faithfulness tiers (and an explicit caveat)

The panel deliberately distinguishes three trust tiers, because they are not equally
authoritative:

- **Tier A (high trust)** — intent, self-surprise, attribution, episodic grounding:
  computed directly from the agent's own free-energy objective and world model, on the
  pre-optimizer-step weights that actually produced the action.
- **Tier B (correlational)** — interpretability probes: linear/logistic decoders
  trained offline on **eval-only** ground truth (`oracle_truth` / `eval_truth`), never
  on cognition. They are read-outs labeled with their own quality, not causes.
- **Tier C (gloss)** — the narrative: a templated (or optional frozen-LM) paraphrase of
  the structured record, explicitly non-authoritative.

**Caveat — functional, not phenomenal.** This instrument measures *functional and
structural correlates* of the logical layer: what is globally available for report
(global-workspace access), what the agent's objective is pulling toward, and what its
self-model predicts. Per the source paper, those are necessary conditions and the
honest target of measurement; they are **not** a claim about phenomenal experience
("what it is like"). The Cognition panel reports access/reportability, not qualia, and
nothing in it feeds back into cognition — the agent's behaviour is byte-for-byte
identical whether or not the trace, probes, or narrative are enabled.

## Self-model program — closing the cognitive feedback loop

The conformance map above is descriptive of what exists; this section tracks the
**self-model program**, which closes the architecture's missing edge: the
channels that "sound like inner life" (A state-of-mind, C narrative, E
metacognition, B affect, the workspace) are today *emitted and discarded* — they
are written to the State Bus and never read back. The only cross-cycle feedback
is the GRU/LSTM recurrence plus the affect/viability scalars in the 4-vector
`ep`. The program turns those self-reports into content the stack **acts
through**, phase by phase, each default-OFF and byte-identical when off.

### Falsification + integration measurement (the honesty guard)

Every phase must *prove* it raises integration, not merely relabel outputs.
`decadic/metrics/integration.py` is the shared instrument: a perturbational
complexity proxy (PCI / Φ-style). It drives a `NeuralCognitiveStack` with a
fixed synthetic percept, injects a single bounded pulse into `z0`, and scores how
widely (across stages) and how durably (across cycles) the pulse spreads versus
an unperturbed baseline, via the Lempel-Ziv complexity of the binarized
deviation matrix. A genuinely integrating pathway makes the pulse persist and
differentiate, raising PCI; a cosmetic one does not. `tests/test_signatures.py`
scaffolds the Section-7 P1–P5 signature assays and is **capability-gated** — each
assay auto-activates when its phase lands and skips before then.

| Phase | Mechanism | Flag (default OFF) | Signature assay |
|-------|-----------|--------------------|-----------------|
| 0 | PCI/Φ proxy + signature scaffold | n/a (always-available instrument) | `test_pci_metric_wellformed`, determinism, no-clobber |
| 1 | Self-state feedback spine (A‖C‖E → next cycle) | `DECADIC_SELF_MODEL_FEEDBACK` | severing the loop changes the self-report; proxy persistence rises |
| 2 | Real global workspace (winner-take-all + ignition + broadcast) | `DECADIC_GWT_ENABLED` | moving the ignition threshold changes report + downstream |
| 3 | Explicit temporal-integration window (committed "now") | `DECADIC_INTEGRATION_WINDOW_MS` | window length shifts the committed moment |
| 4 | Constructed predictive affect (predict B forward, route into prior) | `DECADIC_PREDICTIVE_AFFECT` | predicted affect changes perception |
| 5 | Represented self (interoception/affect/capability as self-node content) | (discovered already default) | richer self-node content + edges |
| 6 | Scale + richness (memory-efficient heavy-tier training) | `DECADIC_MEMEFFICIENT_OPTIM` | sweep configs against the Phase-0 measure |

Cross-cutting discipline: each new pathway is a new faculty/flag defaulting OFF
with zero-init or additive injection, so the baseline `state_dict` and numerics
stay byte-identical; any shape change rebuilds on toggle (reset semantics);
feedback is detached first to avoid cross-cycle BPTT; and every new flag is
pinned OFF in `tests/conftest.py`.
