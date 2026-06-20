# Project Brief: Decadic Cycle Cognitive Architecture

## Summary

We are building a cognitive architecture server that implements an original theoretical framework — the **Decadic Cycle of Expression** — as a working computational system. The server runs continuously, processes streaming multimodal input, maintains persistent internal state and memory, and emits actions in real time. It's designed to be environment-agnostic: any 3D world, simulator, or sensory source can connect to it via API and effectively become its "body."

This is a research-first proof of concept. The goal is to demonstrate that a system organized around the Decadic Cycle exhibits cognitively interesting behavior — emergent priorities, learned associations, coherent decision-making over time — that differs measurably from conventional reactive or RL-based agents.

## Background and theoretical foundation

The **Decadic Cycle of Expression** is a ten-stage model of how cognition unfolds from perception to behavior, with continuous internal state (A–E) shaping every step:

**The ten stages:**

1. Sensory Perception
2. Experience Framing & Multisensory Integration
3. Heuristic Assessment & Memory Correlation
4. Risk-Utility Evaluation, Curiosity Trigger & Investigative Examination
5. Pre-Normative Conclusion Development
6. Emotional/Physiological Experience
7. Reprioritization & Update State of Mind
8. Strategy Formation
9. Behavioral Response
10. Normative Memory Mapping

**The persistent state elements (A–E, plus F we've added):**

- A: State of Mind
- B: Emotional/Physiological State (carries pain/pleasure signals)
- C: Internal Narrative
- D: Current Priority
- E: Metacognition
- F: Action history / efference copy (added for self-modeling)

The cycle iterates continuously. The output of stage 10 (memory mapping) feeds the next cycle's starting state. Internal state evolves across cycles, producing developmental trajectories rather than static behavior.

The framework is original to the project owner. Implementation should preserve fidelity to its structure — this is not "implement an RL agent," it's "instantiate this specific cognitive theory."

## What we're building

A **standalone cognitive architecture server**, written in Python, that:

1. Accepts streaming multimodal observations (vision, audio, proprioception, events) via WebSocket
2. Maintains a continuous perceptual integration process that builds a coherent world model from fragmentary input across many cycles
3. Runs the Decadic Cycle as an ongoing cognitive process, asynchronously and independently of the input stream
4. Manages persistent internal state (the State Bus, A–F) shared across processes
5. Implements a viability/pain/pleasure motivational system that drives learning and behavior
6. Emits actions back to the connected environment via WebSocket as they emerge from cognition
7. Stores episodic memory per cycle and runs consolidation in parallel (replay-based learning during operation, no explicit "sleep")
8. Exposes a state-inspection API for dashboards and analysis

Environments connect as clients. The first connected environment is a MuJoCo physics simulation (`assets/humanoid_body.xml` driven by `scripts/mujoco_decadic_adapter.py`). Future clients may include MineDojo, custom test scenarios, and visualization tools.

## Architecture overview

### Process structure

The server runs multiple concurrent processes/loops:

- **WebSocket handler** — manages connections, routes incoming observations to perception, sends outgoing actions
- **Perceptual integration loop** — continuously updates a maintained world model from the streaming observation input
- **Cycle loop** — runs the ten-stage Decadic Cycle, sampling from the perceptual state when needed, updating the State Bus, emitting actions
- **Fast-path handler** — routes high-priority signals (damage, threat) directly to element B without waiting for full cycle processing
- **Consolidation loop** — runs replay-based learning on a parallel copy of the network weights during low-activity periods
- **State inspection endpoint** — read-only API for current state, used by dashboards and analysis tools

These processes communicate through shared state objects (the State Bus, the Perceptual State, the Episodic Memory store), with appropriate concurrency control.

### Data flow

```
[Environment] → WebSocket → [Perception process] → [World model] ↑
                                                                  |
                                              [Cycle loop reads here at stage 1]
                                                                  |
                                                          [Cycle stages 2-10]
                                                                  |
[Environment] ← WebSocket ← [Action emission] ←─────────[Stage 9 output]
```

The perceptual integration runs continuously, asynchronous to the cycle. The cycle queries the current perceptual state when it reaches stage 1, treating perception as a maintained synthesis rather than a per-frame snapshot.

### The cognitive modules

Each of the ten stages is implemented as a small custom neural network (PyTorch). The system does not use pretrained LLMs in the cognitive core. Approximate parameter budget:

- Stage 2 (multimodal fusion): ~5M params (small transformer)
- Stage 4 (risk-utility): ~500K params (MLP)
- Stage 5 (pre-normative conclusion): ~10–20M params (encoder-decoder)
- Stage 6 (emotion update): ~200K params (GRU)
- Stage 7 (state/priority update): ~1M params (LSTM)
- Stage 8 (strategy formation): ~2M params (policy network)
- Plus frozen pretrained encoders (CLIP, Whisper) for raw perception

Total trainable: ~20–30M params, doubled to ~40–60M for the dual-network consolidation setup.

### Learning approach

Networks are randomly initialized (with biologically-motivated minimal priors) and learn online via:

- **Predictive coding** as the primary mechanism: each stage makes predictions about subsequent states; prediction errors drive weight updates
- **Pain/pleasure signals** as motivational scaffolding: viability changes generate aversive/appetitive signals that bias learning and behavior
- **Replay-based consolidation** running in parallel: high-salience cycles (high prediction error, high emotional intensity) are replayed during low-activity periods to update the parallel learning network, which periodically syncs to the active network

No labeled training data is used. The system learns from its own experience stream.

### Motivational architecture (pain/pleasure system)

The system has a **viability** scalar tracking its current standing on a notional life-death axis. Contributors to viability:
- Damage events (collisions, hostile interactions in the connected environment) — fast-path, direct to element B
- Homeostatic deviations (energy, integrity, etc.) — sustained signals proportional to deviation
- Prediction error magnitude — high error reduces viability
- Successful action outcomes — increase viability

Changes in viability generate pain (decrease) and pleasure (increase) signals that populate element B. Element B in turn shapes element D (priority) and influences all downstream stages of the cycle.

Higher-order drives emerge through learned association — the system isn't preloaded with *specific* social goals, object preferences, etc., but develops them through experience as situations become associated with pleasure signals.

**Innate epistemic drive (curiosity).** Alongside the homeostatic drive, the system carries one further innate motivational *substrate*: a need-gated **curiosity** drive (`decadic/state/curiosity.py`, on by default in production via `DECADIC_CURIOSITY_ENABLED`; set to `0` for a byte-identical no-curiosity baseline). As with hunger/thirst, phylogeny fixes the substrate while ontogeny supplies the content: curiosity rewards **learning progress** — the *reduction* of forward-model prediction error over a short window (not raw surprise, which would chase a noisy TV forever) — and enters element B as a pleasure-side affect that can flip the priority to `investigate` and relax exploratory motor babble as it is satisfied. It is **need-gated by survival urgency**: a threatened or deprived agent suppresses it; a safe, sated one expresses it (Stage 4's risk-vs-curiosity arbitration). *What* is interesting is never coded — only the capacity to find error-reduction rewarding.

## Technical stack

- **Language:** Python 3.11+
- **ML framework:** PyTorch 2.x
- **Web framework:** FastAPI (HTTP) + WebSocket support (native or Starlette)
- **Concurrency:** asyncio + threads, with multiprocessing if GPU contention requires
- **Vector store:** LanceDB or Chroma (for episodic memory with embedding queries)
- **Structured store:** SQLite or DuckDB (for cycle metadata and structured episodic records)
- **Pretrained encoders:** CLIP (vision), Whisper-small (audio), via Hugging Face transformers
- **Logging:** Structured JSON logs, with cycle-trace dumps for debugging
- **Testing:** pytest, with a test-client harness that streams synthetic observations

## Hardware target

Single workstation:
- GPU: RTX 4090 or RTX 5090 (24–32GB VRAM)
- RAM: 64–128GB
- Storage: 2TB+ NVMe SSD (episodic store grows during operation)
- CPU: Modern Ryzen 9 or Intel i9

The system is designed to run continuously on a single workstation. Cloud GPU is optional for larger experiments later.

## API surface

### WebSocket: `/agent/{id}/cycle`

Bidirectional streaming. Environment sends observations, server sends actions, both asynchronous.

**Observation message (env → server):**
```json
{
  "timestamp": "ISO 8601",
  "vision": {"encoding": "base64_jpeg", "data": "...", "resolution": [224, 224]},
  "audio": {"encoding": "base64_wav", "data": "...", "duration_ms": 100},
  "proprioception": {
    "position": [x, y, z],
    "orientation": [pitch, yaw, roll],
    "velocity": [vx, vy, vz],
    "current_action": "walking_forward"
  },
  "events": [
    {"type": "collision", "intensity": 0.7, "source": "wall_north"}
  ],
  "world_state": {"nearby_entities": [...], "agent_inventory": [...]}
}
```

**Action message (server → env):**
```json
{
  "timestamp": "ISO 8601",
  "action": {
    "type": "move",
    "parameters": {"direction": [0.7, 0.0, 0.7], "speed": 1.0}
  },
  "predicted_outcome": {"embedding": [...], "expected_position": [x, y, z]}
}
```

### REST endpoints

- `POST /agent` — create new agent instance
- `GET /agent/{id}/state` — current cognitive state (A–F, current cycle, etc.)
- `GET /agent/{id}/memory` — query episodic memory
- `POST /agent/{id}/checkpoint` — save current state
- `POST /agent/{id}/restore` — restore from checkpoint
- `GET /agent/{id}/metrics` — cycle rate, prediction error trends, viability, etc.
- `DELETE /agent/{id}` — terminate agent

## Phased build plan

### Phase 1: Foundation (4–6 weeks)

- Server skeleton with FastAPI + WebSocket
- Async architecture with shared-state objects (State Bus, Perceptual State, Episodic Memory)
- Stub Decadic cycle (modules return placeholder values)
- Pain/pleasure / viability system implemented and integrated with element B
- Fast path for damage signals
- Test client that streams synthetic observations
- Logging and observability infrastructure
- Basic state-inspection API

**Deliverable:** End-to-end loop with a connected test client. Synthetic observations come in, placeholder cycle runs, placeholder actions go out. Pain/pleasure responds to test damage events. Architecture is solid and extensible.

### Phase 2: Real cognitive modules (6–10 weeks)

- Replace stage 4 placeholder with actual Risk-Utility MLP
- Replace stage 6 placeholder with actual Emotion-Update GRU
- Replace stage 7 placeholder with actual State-Update LSTM
- Replace stage 2 placeholder with actual Multisensory Fusion transformer
- Replace stage 5 placeholder with actual Pre-Normative Conclusion encoder-decoder
- Replace stage 8 placeholder with actual Strategy network
- Wire up frozen pretrained encoders (CLIP, Whisper) at the perception entry
- Implement basic predictive coding loss across stages
- Verify learning happens — modules should show prediction error reduction over cycles

**Deliverable:** All ten stages implemented with real networks. System learns from its own experience stream. Behavior visible and analyzable.

### Phase 3: Memory and consolidation (4–6 weeks) — **implemented (dual-network)**

- Full episodic memory store (SQLite + LanceDB)
- Salience tagging on stored cycles
- Consolidation loop running in parallel — `decadic/consolidation/consolidator.py` (`ConsolidationManager`)
- Dual-network setup with periodic weight sync — a cloned cognitive stack trains on replay and Polyak soft-syncs into the live stack under the agent lock (`DECADIC_CONSOLIDATION_SYNC_TAU`)
- Replay sampling and replay-based learning — `decadic/consolidation/replay_buffer.py` (`ReplayBuffer`, salience-prioritized `sample`)
- Forgetting / pruning of low-salience old episodes — bounded buffer evicts the lowest-salience transition when full; `DECADIC_CONSOLIDATION_PRUNE_MIN_SALIENCE` floor

The subsystem is **on by default** in production (`DECADIC_CONSOLIDATION_ENABLED`; set to `0` for the no-op stub heartbeat and a byte-identical baseline) and runs replay in a thread executor so the cognitive cycle is never blocked. Metrics: `replay_count`, `replay_buffer_size`, `consolidator_loss`, `last_sync_cycle`.

**Deliverable:** Memory and consolidation working end-to-end. Long-term learning visible in agent behavior. Comparable performance with and without consolidation can be demonstrated.

### Phase 4: World model perception (6–10 weeks)

- Replace primitive perceptual integrator with proper world model
- Object tracking across observations
- Confidence/uncertainty over scene elements
- Prediction-error generation at perceptual layer
- Integration with cognitive cycle's stage 1

**Deliverable:** System maintains coherent perceptual state across noisy/intermittent input. Object permanence, scene constancy, expected continuity. Perception is no longer per-frame snapshots.

### Phase 5: MuJoCo embodiment

- WebSocket protocol bridged by `scripts/mujoco_decadic_adapter.py`
- Humanoid body (`assets/humanoid_body.xml`) connected to the agent server
- Observation extraction from MuJoCo (proprioception, interoception, tactile/contact, body state, events)
- Action application from server commands to the MuJoCo humanoid's joint actuators
- Live testing of the cognitive system in the physics simulation (Skill Dojo standing/locomotion practice, foraging scenarios)

**Deliverable:** Decadic agent operating in the MuJoCo world. Behavioral observation possible. Initial research observations can begin.

### Later phases (deferred)

- Language module integration (Path B: frozen small LM as element C, gated on cycle coherence)
- Comparative experiments (Decadic vs baseline reactive agent)
- Publication preparation

## Hiring considerations

The ideal developer for this project has:

- **Strong Python and PyTorch experience** — building custom neural network modules from scratch, training loops, debugging gradient flow
- **Async/concurrent systems experience** — comfortable with asyncio, threading, shared state, race condition reasoning
- **Familiarity with cognitive architectures or computational neuroscience** — at least conceptual understanding of predictive coding, world models, embodied agents
- **API and server design** — comfortable building real-time WebSocket services
- **Comfort with research-style work** — willing to iterate on architectural decisions, debug emergent behavior, accept that some experimentation will be needed

Less critical but valuable: experience with physics simulation (MuJoCo), prior work on embodied AI projects, familiarity with vector databases.

This is **not** a project for someone whose primary experience is fine-tuning LLMs or building RAG systems. The work is closer to deep RL, cognitive modeling, or novel ML systems engineering.

## Project ownership and IP

The Decadic Cycle of Expression framework is original intellectual property of the project owner. Any work derived from this framework — implementations, training procedures, derived architectures — is owned by the project owner. Standard work-for-hire or contractor IP assignment applies.

The theoretical framework itself is being defensively published to establish priority. Implementation specifics may be kept as trade secrets or pursued for patent protection on a case-by-case basis depending on technical novelty.

## Success criteria for the proof of concept

The PoC succeeds if it demonstrates:

1. **The architecture runs continuously and stably.** Days of uptime without degradation. Cycles execute reliably. Memory grows in a controlled way.

2. **Learning is visible.** Module prediction errors decrease over cycles. The agent's behavior becomes more competent in its environment over time. Without external rewards, structured behavior emerges.

3. **Internal state evolves coherently.** A–F values show meaningful, traceable patterns. Priorities (D) shift in response to experience. Emotional state (B) responds appropriately to events.

4. **Memory and consolidation work.** Salient experiences are retrieved when relevant. Consolidation produces measurable behavioral improvements over time.

5. **The architecture is observably distinct.** A baseline agent (e.g., simple reactive controller) in the same environment produces qualitatively different behavior. The Decadic structure produces something specific.

If all five are achieved, the project has produced a publishable result demonstrating a working implementation of the framework. If subset are achieved, the project produces partial results that inform what future work would address.

## Open questions and known uncertainties

These remain open and should be discussed during scoping:

- **Action space granularity** — high-level discrete actions vs continuous control. Currently leaning high-level for early phases.
- **World model architecture choice** — slot attention, transformer-based, or something simpler. Needs experimental evaluation.
- **Consolidation scheduling** — *resolved (v1)*: pure periodic background bursts (`DECADIC_CONSOLIDATION_SYNC_INTERVAL_S`) running in a thread executor; activity-gated/hybrid scheduling remains a possible refinement.
- **Language integration timing** — when to introduce element C as a real LM (Path B). Probably after Phase 4 but flexible.
- **Multi-agent setups** — out of scope for v1, but worth keeping the architecture compatible with future multi-agent extensions.

## Contact and next steps

Initial discussion should cover:

1. Developer's experience with the technical stack and the kind of problem
2. Realistic timeline given developer's availability
3. Working arrangement (employee, contractor, equity, etc.)
4. Code repository setup, communication tools, milestone reviews
5. Phase 1 kickoff: starting with the server skeleton and getting the test client running

Project owner can provide additional theoretical documentation on the Decadic framework, the Correlative Framework of Memory (CFM), the Apple knowledge graph structure, and other supporting concepts as needed during scoping and ongoing development.
