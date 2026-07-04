# PRD: WS5 — Relational Binding (Slots Across the Neural Boundary)

**Version:** 1.0 — 2026-07-04
**Status:** Draft for review
**Companion:** `ws5_relational_binding_wbs.md`
**Origin:** 2026-07-04 architecture review — the capability audit that followed the WS3 probe redesign. Settled decisions stated declaratively; open decisions marked.

---

## 1. Why

Variable binding — keeping *what* and *which role* factored so contents can be swapped, compared, and retrieved by role — is the mechanism behind compositional cognition. "The wolf is behind the rock" and "the rock is behind the wolf" contain identical parts; a system that only superimposes features into one vector represents both identically and can therefore never reason about either. This is the live core of the Fodor–Pylyshyn systematicity critique, and it is the capability gap standing between Deca's current cognition (perceive, remember, attend, feel, learn) and the next tier (relational reasoning, long-horizon planning over entities, theory of mind via self-model simulation).

Scale does not buy this. A wider feedforward path superimposes more features; it does not factor them. Binding needs an architectural mechanism: slots plus keyed (attention-based) read/write.

## 2. Gap analysis (from 2026-07-04 code inventory)

### 2.1 What already exists — the symbolic half

| Capability | Where | State |
|---|---|---|
| Entity slots, capacity-bounded, decaying | `decadic/state/working_memory.py` (`WorkingMemory.slots: dict[str, MemorySlot]`, capacity/decay/min-salience) | **Built.** Global Workspace store, populated by perception, snapshot into dashboard + stage-10 reports. |
| Anonymous object files | `decadic/perception/object_files.py` (`ObjectFile`: appearance, motion, agency, persistence, confidence, looming, entity_role; label-free discipline enforced) | **Built.** Designed for visual/embodied discovery (retina contrast, flow, mask entropy) — currently starved by synthetic input. |
| Scene workspace | `decadic/perception/scene_workspace.py` (integrates object files; `scene_slots` source flag reaches stage-10 reports) | **Built.** |
| Persistent entity identity | Kuzu graph: appearance-cosine identity matching, per-entity property beliefs | **Built (WS4).** The world-side half of object permanence. |
| Sub-symbolic scene persistence | `WorkingMemory.scene_latent` (EMA of pooled percept — "the image in the mind") | **Built.** |
| Fast recall over percept keys | `search_similar_percept` + full-mirror L1 (WS4), recency horizon (WS3 redesign) | **Built.** Sub-ms at 100k rows. |

### 2.2 What is missing — the differentiable half

The finding of this audit: **binding dies at the neural boundary, and the cause is pooling.** Three chokepoints, all of the same shape:

1. **Scene → stack:** `top_down_perceive(scene=...)` receives `scene_latent` — an EMA-pooled, chunk-folded single vector (`_fold_scene`). The K entity slots that `WorkingMemory` maintains are never seen by the network. Whatever structure perception discovered is superimposed before cognition can use it.
2. **Memory → stack:** `retrieval_context_vector` mean-pools the top-k recalled episode embeddings into one `mem_t`. Five distinct remembered situations enter as their average.
3. **Stage 3 → 4:** `z4 = risk_mlp(z3)` — risk is computed from one fused vector. No mechanism exists that could compute *a relation between two entities*, because no two entities exist as separate tokens anywhere in the differentiable path.

Secondary gaps that follow from the primary one:

4. **No keyed read from cognition.** WM slots are written by perception and read by reports; the stack cannot query them by content ("retrieve the slot most relevant to current risk").
5. **Graph↔slot bridge is one-way.** Object files carry `object_id`, and the graph knows entities — but slot contents are not keyed by graph-entity embeddings, so a WM slot is not yet a Kahneman/Treisman *object file with identity over time* from the network's point of view.
6. **Environment starvation (external).** On synthetic patrol input the discovery pathway has nothing to discover (`object_files` ≈ 0 all soak; graph beliefs = 0). The perception-side machinery is built but unexercised. This is the standing MuJoCo dependency, not a WS5 work item.

### 2.3 The thesis

WS5 is therefore not "build a binding system." It is: **carry the slots the symbolic half already maintains across the neural boundary as a slot tensor, replace pooling with keyed attention at the three chokepoints, and add one small relational module where relations must be computed.** The symbolic and differentiable halves then meet in the middle, and the graph supplies identity.

## 3. Goals

1. The neural stack receives working memory as a **slot tensor** (K × D_slot), not a pooled vector; the pooled `scene_latent` path remains as the flag-off baseline.
2. **Keyed read** (cross-attention, query from z3/state) over WM slots and over top-k recalled episodes — retrieval by role/content instead of averaging.
3. **Relational core at stage 3→4:** a deliberately small transformer (est. 2 layers / 2 heads) over the token set [WM slots; top-k episode tokens; one interoceptive token], whose pooled output augments the risk head's input. Composes with the WS3 attention gate: relational deliberation is exactly the expensive thing the gate prices.
4. **Graph-entity keying:** WM slot key = the entity's graph appearance embedding when identity-matched (else the anonymous object-file appearance). Object permanence becomes visible to the network.
5. **Falsifiable binding evidence:** a novel-combination generalization probe — train on relations over entity set A, test on never-seen pairings — where success is generalization, not memorization (the criterion set in the 2026-07-04 review).

## 4. Non-goals

- No language, no symbols-in/symbols-out. Binding here is sub-symbolic infrastructure.
- No learned gate (that is its own workstream, sequenced after WS5 + MuJoCo — the relational core changes what "escalation" buys, so the gate learns against the final cost structure).
- No perception rewrite. Slot attention over raw visual input is MuJoCo-era work; WS5 consumes the object files perception already emits.
- No change to episodic/graph storage (WS4 is closed; WS5 is a consumer).

## 5. Design

### 5.1 Slot tensor interface (settled shape)
`WorkingMemory.slot_tensor(k_max, d_slot)` → float32 (K, D_slot) + mask, deterministic slot ordering (salience-ranked, stable ties), fixed-dim projection of each `MemorySlot`'s fields (appearance ⊕ motion ⊕ scalars, zero-padded). Pure read-side adapter: no change to slot lifecycle. Flag `DECADIC_WM_SLOT_TENSOR`; off ⇒ stack sees exactly today's inputs (byte-identical baseline, the parity culture's standard).

### 5.2 Keyed read (settled shape)
Two cross-attention blocks in the stack, both query-from-state:
- **WM read:** query = f(z3_pre); keys/values = slot tensor. Replaces the folded scene vector as the WM contribution when the flag is on.
- **Memory read:** `retrieval_context_tokens(qv, k)` returns top-k episode embeddings as tokens (the store already returns ranked hits; this is an adapter, not a search change). Cross-attention replaces mean-pooling.

### 5.3 Relational core (settled shape, sizing open)
Token set: [K slot tokens; k memory tokens; 1 interoceptive token (pain, pleasure, viability, drive)]. Small pre-norm transformer (open: 2×2 vs 3×4; decide by measured cycle cost on the full preset). Pooled output concatenated into the risk head input: `z4 = risk_mlp(z3 ⊕ relational_summary)`. Flag `DECADIC_RELATIONAL_CORE`; off ⇒ `risk_mlp(z3 ⊕ 0)` weight-compatible fallback or plain z3 path (decide in M3 — whichever keeps checkpoint compatibility cleanest).

### 5.4 Graph keying (settled)
On WM integrate, when the graph identity-matches an object file (existing Kuzu top-1 cosine ≥ 0.6 path), the slot's key vector is set from the graph entity's stored appearance embedding; unmatched slots key on their own appearance. Slot metadata carries `entity_id` (already present as `object_id` plumbing). No graph writes change.

### 5.5 Binding probe (settled criteria, scenario details M5)
Synthetic multi-entity scenario (does not require MuJoCo: the WS client grows scripted `world_state.nearby_entities` with controlled appearance vectors — the discovery pathway is bypassed, object files injected at the WM seam). Train phase exposes relations among entity pairs (e.g., threat-adjacency patterns); test phase presents **novel pairings of familiar entities**. Pass = risk/priority output tracks the relation on unseen combinations; fail = tracks only memorized pairs. This is WS5's analogue of the WS3 probe: the experiment that makes the mechanism falsifiable.

### 5.6 Parity and cost discipline
Every flag off ⇒ full suite byte-identical (existing parity culture). Cycle-cost budget: relational core must fit the measured 70–90 ms cycle envelope on the full preset — measured at M3 with the WS2 overhead harness before any default flips. All defaults stay off until the M5 probe passes.

## 6. Success criteria

1. Full suite green, flags off (byte-identical) and flags on.
2. Slot tensor + keyed reads: unit + parity tests; recall path equivalence when k=1 (single token ≡ today's best-hit semantics).
3. Relational core cycle cost measured and inside the envelope on the full preset (⚙).
4. Binding probe: novel-combination generalization passes; memorization-only explicitly rejected by the held-out design.
5. Graph keying: a re-encountered entity re-binds to the same slot key across an occlusion gap (object-permanence assertion at the WM level).

## 7. Risks

- **Slot starvation on synthetic input** — mitigated by the M5 probe injecting entities at the WM seam; full validation remains MuJoCo-era (explicitly accepted).
- **Checkpoint compatibility** — new stack parameters change `brain.pt` shape; saves made flags-off must load flags-on (zero-init new blocks) and vice versa (drop). Same discipline as growth events; test in M4.
- **Cycle budget** — relational attention is O(K+k)² per cycle; K and k are single-digit, but the full preset on CUDA must be measured, not assumed (the ANN lesson from WS4).
- **Gradient routing** — keyed reads open new cross-cycle paths; detach discipline must match the existing loops (history/memory enter detached; see `top_down_perceive` conventions).
- **Probe leakage** — novel-pairing test must control for feature-level similarity shortcuts (pairings balanced so marginal statistics are uninformative).

## 8. Open decisions (resolve by end of M3)

- Relational core size (2×2 vs 3×4) and whether the interoceptive token is one token or per-scalar tokens.
- `risk_mlp` input strategy for flag-off weight compatibility (zero-concat vs separate head).
- K (slot count exposed to the stack) and D_slot defaults — WM capacity default vs a smaller neural window.
- Whether the WM keyed-read replaces or augments the scene-latent input when on (augment is the conservative default).

---

## Addendum: build decisions and measurements (2026-07-04)

**M1/M2/M3.1 landed** (commits of 2026-07-04): slot tensor (frozen 40-d layout, `docs/ws5_m0_wm_inventory.md`), keyed WM read, memory-token read, relational core. Open decisions resolved during build:

- **risk_mlp input strategy (§8):** neither zero-concat nor a separate head -- the relational summary enters via a zero-init ADDITIVE ingress into the stage-4 input (`z4 = risk_mlp(z3 + rel_ingress(summary))`). Keeps `risk_mlp`'s shape, checkpoint compatibility, and the house zero-init discipline (byte-identical until learned). Flags-off saves load flags-on with zero-init new blocks.
- **WM keyed read replaces vs augments (§8):** augments, as defaulted. The scene-latent path and mean-pooled `zm` are untouched; binding is additive structure beside the legacy signals.
- **K and D_slot (§8):** K = 6 (`DECADIC_WM_SLOT_K`, a cognitive parameter, preset-independent); D_slot = 40 frozen. Interface dims do NOT scale with presets (`test_interface_dims_fixed_across_presets` family).
- **Gate composition (§goal 3):** the relational core computes on DELIBERATIVE cycles only -- `stage4_override` (gate skip) bypasses it entirely, call-count asserted in tests. Relational deliberation is now exactly the compute the WS3 gate prices; the WS3-B shadow tap measures the pre-relational counterfactual (revisit when the gate retrains post-WS5).

**M3.2 measured cost (full preset, RTX 3080, CUDA, forward-only, K=6/k=5, 200 iters):**

| config | off p50/p95 | on p50/p95 | delta p50 |
|---|---|---|---|
| 2 layers x 2 heads | 6.15 / 7.06 ms | 7.67 / 10.10 ms | **+1.52 ms (+24.7% of forward, ~2% of the 70-90 ms cycle envelope)** |
| 3 layers x 4 heads | 5.69 / 6.77 ms | 7.77 / 8.80 ms | +2.07 ms |

"On" includes the full binding path (both keyed reads + relational core). **Sizing decision: default 2x2** (smallest mechanism that can pass the probe); 3x4 is measured, affordable (+0.55 ms), and reserved as the capacity lever if M5.2 flags-on fails for capacity rather than mechanism reasons. `scripts/bench_relational.py` reproduces the table.

**Remaining:** M4.1 graph-keyed slots (object permanence network-visible) · M4.2 checkpoint round-trips through REST · M5 probe (scenario/verdict scaffold exists: `docs/binding_scenarios/`, `gen_binding_scenario.py`, smoke PASS 6/6 slots live).
