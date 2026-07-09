# WS-SYM — Symbol Integration & Relational Memory: Comprehensive Analysis

**Status:** analysis / design draft (2026-07-08). Investigation + literature
review complete; no code changed yet.

## 0. The two goals

1. **Integrate symbols into cognition** — the FSQ discrete code must feed *back
   into* the trunk and shape deliberation/action, not merely be emitted as a
   per-cycle read-out.
2. **Store symbols relationally** — codes must be written into the long-term
   graph bound to the *entities* and *episodic memories* they co-occur with, so
   they can be recalled and become useful.

Both goals are blocked by a single missing wire and then unlock along seams the
architecture already provides. This document establishes current state, grounds
the design in the research literature, proposes an architecture, and gives a
gap analysis + WBS.

## 1. Current state (from code investigation)

**Symbol computation (`neural_pipeline.py:1585-1634`).** Every cycle, when
`symbols_enabled` and `stack.has_symbols`: `_q_proj = fsq_in(z5_t.detach())` →
`fsq_quantize` → `symbol_code_val` (one of 4800 codes; `FSQ_DIMS=5`). The trunk
input is **detached** (byte-identical parity), but `fsq_in` itself carries
gradient so the next-code head (`fsq_next`) and the local-isometry smoothness
loss train it. The previous quantized vector persists as
`bundle._prev_symbol_q` (`:1632`) and the pre-quant projection as
`_prev_symbol_p`.

**The code is a pure read-out.** A full grep confirms `symbol_code` /
`_prev_symbol_q` are consumed by **nothing** except telemetry
(`runtime.py:2734`). No deliberation, action, memory, or trunk-forward path
reads it. This is the crux of Goal 1.

**The single missing wire.** The symbol value is published only to
`runtime.metrics`, never onto `ctx.state_bus` / `ctx.latents`. Because of that,
**Stage 10** (`stage_10.py`) — the consolidation site that has the attended
entities *and* the cycle index in hand — cannot see the code. Routing the code
(and `_prev_symbol_q`) onto the bus unblocks *both* goals at once.

**Trunk feedback seam (Goal 1).** The stack already accepts four zero-init
policy-tier conditioning ingresses — `goal_ingress`, `schema_ingress`,
`draft_ingress`, `other_ingress` (`neural_stack.py:1191-1216`), each a
zero-initialized `nn.Linear(_, pol_in)` added into `pol_in_t`
(`pol_in = lstm_hidden + state_mind_out`). A `symbol_ingress` mirroring
`other_ingress` (`:374-378`) + a `symbol_vec` kwarg (`forward` `:1014-1034`) +
an additive block is the clean, parity-preserving seam. The feedback tensor
already exists (`_prev_symbol_q`); only the consumer is missing.

**Graph schema (Goal 2).** Entities are appearance-fingerprint-keyed nodes;
RELATES edges are `Entity→Entity (kind,weight,count,last_cycle)`;
PropertyBeliefs are `(node_id, property_key)` numeric evidence; SemanticRecords
are `(category,id)` with free-form `payload_json`, `SEMANTIC_CATEGORIES =
(entity,event,relationship,correlation,conclusion,value)` (`semantic_graph.py:60`).
`_upsert_semantic` (`:610`) create-or-accumulates evidence/confidence exactly
like the existing `entity_predicts_event` correlation pattern (`:1062-1101`).
Kuzu mirrors all four tables.

**Label firewall (`semantic_graph.py:47-65`).** `FORBIDDEN_*_TOKENS` strip any
injected label/name; only self-derived `predicts_*_pain` keys are whitelisted
(`_clean_property_key:92-94`). **Self-derived integer codes pass freely**;
oracle *names* for codes would be stripped — so symbols must be stored as codes
and grounded, never as injected labels. This is a hard constraint and it
happens to be exactly right (see §2 grounding).

**Binding site (Goal 2).** Stage 10 has `focus_ids` / `attention_focused`
slots (`stage_10.py:35,81`), `cycle` (`:112`), and calls `consolidate` (`:176`)
and `record_semantic_evidence` (`:201-207`). This is where "entity X co-occurred
with code C at cycle N" is naturally written — once C is on the bus.

**Episodic store.** `EpisodicRecord(cycle_index, summary, salience, embedding)`;
`summary` is free-form JSON → a `summary["symbol_code"]` attaches with **zero
schema change**. Similarity recall via `search_similar` over the 80-d embedding.

**Recall channel.** Stored content re-enters cognition through the
`memory_context` vector (dim 32): `episodic.retrieval_context_vector` → async
`_memory_context_vector` (`runtime.py:1490-1498`) → `mem_proj(memory_context)`
(`neural_stack.py:1058`). There is **no dense graph→trunk channel** today; graph
content reaches cognition only as belief-stat counters and WM-slot re-binding.
A recalled symbol becomes "useful" by riding `memory_context`/`mem_tokens` or by
influencing a WM slot's beliefs.

**Existing grounding substrate.** `predicts_*` beliefs form by
max-accumulate on the *attended* slot's `property_evidence`
(`working_memory.py:638-694`), consolidate through `_upsert_property_beliefs`,
and promote to `conclusion` records at an evidence threshold ≥2.0
(`semantic_graph.py:1073-1083`). Symbols can ground by the identical mechanism.

## 2. Research grounding (verified, cited)

**Integrating discrete codes into a control loop is standard and beneficial —
but the representation choice matters.** DreamerV2 feeds its discrete latent
back into the actor/critic as a flattened one-hot concatenated to the recurrent
state [1]; VQ world models feed the **codebook-embedding-lookup continuous
vector** forward [2]; and for continuous control specifically, discrete-codebook
latents **outperform one-hot and label encodings** as the decision-time state
(DCWM/DC-MPC, which uses FSQ so the codebook is fixed) [3]. The straight-through
path gives biased but low-variance gradients [1]. **Implication:** feed the
*quantized vector* (`_prev_symbol_q`, a continuous 5-vector on FSQ's fixed
geometry), not the raw index or one-hot — it is the representation the
literature finds best for control and it carries no learned-table drift.

**Ungrounded symbols fed back are noise; grounding is co-occurrence + joint
attention + contingency.** Harnad: a symbol has intrinsic meaning only when tied
to sensorimotor feature-detectors, not to other symbols [4]. Cross-situational
statistics *alone* recover word↔referent mappings by aggregating many
individually-ambiguous co-occurrences (Yu & Smith) [5] — but accuracy degrades
with per-frame ambiguity, and **a partner's/attentional focus that isolates the
referent is a stronger predictor than raw exposure** [6]. In referential games,
meaningless codes acquire grounded meaning purely from **contingent partner
success** — meaning emerges from use [7]. Two failure modes to design against:
"positive signaling ≠ positive listening" (a code can correlate with the world
yet not influence behavior) [8], and, without a brevity/efficiency pressure,
codes drift to anti-efficient encodings [9]. **Implication:** bind the code to
the *attended* entity (joint attention), accumulate evidence across many frames
(cross-situational), and validate grounding by *behavioral use*, not
mutual-information alone.

**Memory should be dual-speed and index-based, with concept nodes.**
Hippocampal indexing: an index is a **sparse pointer that binds** the episode's
distributed content, and retrieval is **cue-driven pattern completion** through
that index [10]. Complementary Learning Systems: keep a **fast pattern-separated
episodic tier and a slow distributed semantic tier**; consolidate by
**interleaved replay at a low learning rate** (massed replay causes catastrophic
interference), and replay is *causally required*, not cosmetic [11][12].
Schema-congruent items consolidate faster [11]. Concept cells: a **sparse,
discrete, modality-invariant** code binds to a *specific entity/identity* across
images, text, and sound — the biological analog of an FSQ code bound to an
entity, and the bridge from episodic to semantic memory [13][14].
**Implication:** the entity is the concept node; the FSQ code is its
concept-cell tag; episodes are the fast tier holding per-episode codes; the
graph is the slow tier holding consolidated entity↔code bindings; rest-driven
interleaved replay strengthens them (ties directly to WS-ATTN's rest layer).

**How to bind and retrieve.** An explicit graph gives exact, unbounded,
individually-addressable entity↔code relations and multi-hop recall, at an
extraction cost, and beats flat vector memory on relation-traversal queries
[15][16]; temporal knowledge-graph agent memory (Zep/Graphiti) keeps facts
bi-temporally and supersedes rather than deletes [16]. Vector-Symbolic
Architectures offer the complement: **bind** (similarity-destroying product) an
entity vector with a code and **bundle** (superposition) many such pairs into a
single fixed-width vector, retrievable by unbinding with the entity cue +
a cleanup memory — fixed footprint and holistic similarity, but capacity is
bounded by superposition crosstalk [17][18][19]. **Implication:** store the
exact binding in the graph (fits the existing architecture), *and* use a VSA
bind/bundle of an entity's codes to manufacture the **dense
`memory_context`-width vector that today's graph→trunk channel lacks** — a
hybrid that gives recall-into-cognition without a schema change to the trunk.

**Code stability — what the literature actually supports.** The evidence
separates cleanly into *addressing* stability (strong) and *meaning* stability
(not intrinsic), and it hands us a positive prescription rather than a shrug.

1. *Addressing is stable (established).* FSQ eliminates codebook **collapse** by
   construction — fixed geometry, full utilization, no dead codes, no commitment
   loss / reseeding, matching VQ on generation tasks [20]. Index *k* always names
   the same point in the quantized space. But this is *positional*, not
   *semantic*: it says nothing about whether *k* keeps pointing at the same
   concept.
2. *Learned projections make meaning drift (established, and it applies to us).*
   The mechanistic account [21]: a code's meaning is the region of
   encoder-output space that maps to it; when the encoder moves, that region
   shifts. Here `fsq_in` carries gradient (WS-IND I5) — it is exactly such a
   moving encoder — so FSQ froze the output grid but **not** the input
   projection. The literature therefore does more than fail to promise
   stability; it *predicts* drift for our configuration.
3. *Meaning is external, not intrinsic (established — the key reframe).* Harnad:
   a symbol's meaning comes from grounding to experience, not from the token [4];
   cross-situational learning recovers meaning by **aggregating co-occurrence
   over many exposures** [5], which is inherently robust to per-instance drift.
   **So do not rely on the code's geometry to carry meaning** — carry it in the
   grounded entity↔code *binding*, which then tracks drift automatically. Biology
   does exactly this: neural codes exhibit **representational drift** (units
   remap over time) while behavior/meaning is preserved by population structure
   and continual re-anchoring, not by any unit staying put (well-documented
   neuroscience — Ziv 2013 / Rule & O'Leary; orienting background, outside this
   pass's verified set).

**The three literature-backed stabilizers** (all already in reach): (a) FSQ's
fixed grid + the local-isometry **smoothness loss** (present) preserve the code
space's *relational* structure even when absolute assignments move; (b)
**external grounding by co-occurrence** puts meaning in the binding, where drift
is self-correcting; (c) **slowing/freezing `fsq_in`** once grounding is
established is the standard encoder-drift mitigation [21].

**What is genuinely NOT established** (narrow): that a *specific fixed index*
retains a *constant decoded concept* over a long run — no source measures
index-level semantic constancy over time. We therefore design as if it will
drift, and make drift a **closed-loop controlled quantity** (§5, WBS 5.x): track
entity↔code binding **churn** as the empirical drift proxy and let it *trigger*
the `fsq_in` freeze automatically, rather than treating stability as a leap of
faith.

## 3. Proposed architecture — "the FSQ code as a grounded concept-cell tag"

A single coherent model unifies both goals:

> Each cycle the mind emits a sparse discrete code (concept-cell analog). The
> code is (a) **bound to the attended entity** by accumulating co-occurrence
> evidence in the graph and stamped into the episode (fast tier), (b)
> **consolidated** into a stable entity↔code "concept" during rest by
> interleaved replay (slow tier), (c) **recalled** when the entity recurs —
> the entity cue pattern-completes to its bound code — and (d) **fed back**
> into the next cycle's deliberation through a zero-init ingress, gated on
> grounding maturity so ungrounded codes never inject noise.

Four subsystems, each mapped to a seam:

**S1 — Publish (the missing wire).** Put `symbol_code` (int) and
`_prev_symbol_q` (5-vec) onto `ctx.state_bus`/`ctx.latents` from the pipeline.
Unblocks S2–S4. *(effort S)*

**S2 — Store relationally (Goal 2).**
- *Entity↔code binding:* in Stage 10, for each **attention-focused** slot,
  `_upsert_semantic` a `correlation`/new `symbol` record
  `entity:{id} × code:{C}` with accumulating evidence — the exact
  `entity_predicts_event` pattern (`semantic_graph.py:1062-1101`), joint-
  attention-gated so the code binds to the *right* entity (research [6]), and
  cross-situational so meaning emerges over many frames (research [5]).
  Firewall-safe (integer codes).
- *Episodic stamp:* `summary["symbol_code"] = C` (and/or active-code list) in
  Stage 10 — zero schema change.
- *Promotion:* at evidence ≥ threshold, promote to a stable entity↔code
  `conclusion`/concept record (mirrors existing promotion `:1073-1083`).

**S3 — Recall (make it useful).** When an entity is re-matched (WM slot
re-binding via `entity_appearance()`), surface its consolidated top code(s). Two
delivery options: (i) inject as a WM-slot property belief that already rides the
`wm_slots` path into the trunk; (ii) **VSA-bind** the entity's codes into a
`memory_context`-width vector (research [17]) to fill the missing dense
graph→trunk channel. This is hippocampal pattern completion: entity cue → bound
concept [10].

**S4 — Integrate into cognition (Goal 1).** Add a zero-init `symbol_ingress`
on the policy tier (mirror `other_ingress`), fed the **previous cycle's
quantized vector** `_prev_symbol_q` (fixed-geometry, no drift; research [3]) and
optionally the S3-recalled entity code. **Gate/ramp on grounding maturity** —
feedback weight scales with the code's accumulated grounding evidence, so a
newborn's ungrounded codes contribute nothing and a mature, grounded code shapes
deliberation (research [4][8]; matches the developmental principle already used
across this system).

**S5 — Consolidation & stability (closed-loop, not just telemetry).** Route
symbol consolidation through rest-driven interleaved replay (research [11][12];
reuses WS-ATTN rest + the write-behind warehouse). Make drift a **first-class
controlled quantity**: define **binding churn** = the rate at which an entity's
top-evidence code flips, as the empirical proxy for code-meaning drift (the
thing the literature does *not* guarantee, §2). Emit it as a verdict row, and
**close the loop** — sustained churn above a threshold automatically slows then
freezes `fsq_in`'s learning rate (the standard encoder-drift mitigation [21]),
holding meanings stable once grounding has formed. Meaning ultimately lives in
the binding (research [4][5]), so freezing the projection is safe: it stops the
referent from moving without touching the grounded associations.

## 4. Gap analysis

| # | Capability (target) | Current state | Gap | Effort |
|---|---|---|---|---|
| SG1 | Symbol value visible to memory/cognition | Computed `neural_pipeline.py:1604`; emitted to `runtime.metrics` only | Publish `symbol_code` + `_prev_symbol_q` onto `ctx.state_bus`/`latents` (the one wire that blocks everything) | S |
| SG2 | Entity↔code binding in graph | `predicts_*`/`entity_predicts_event` correlation pattern exists; no symbol category | `_upsert_semantic` an attention-gated entity↔code record in Stage 10; extend `SEMANTIC_CATEGORIES`+stats seeds | M |
| SG3 | Symbol on episodic memory | `summary` free-form; no symbol field | Add `summary["symbol_code"]` in Stage 10 | S |
| SG4 | Grounding accrues by co-occurrence | Evidence-accumulate machinery exists (`_upsert_semantic`, promotion ≥2.0) | Feed symbol co-occurrence through it; joint-attention gate; promotion to stable concept | M |
| SG5 | Recall entity's code into cognition | `memory_context` channel exists; no graph→trunk dense channel; no symbol recall | WM-slot belief path (reuse) and/or VSA-bind entity codes → `memory_context`-width vector | M/L |
| SG6 | Trunk reads symbol back | 4 zero-init policy ingresses exist; **no `symbol_ingress`**; `_prev_symbol_q` persists | Add zero-init `symbol_ingress` + `symbol_vec` kwarg + additive block; feed `_prev_symbol_q` | M |
| SG7 | Feedback gated on grounding | Nothing (code is inert) | Ramp ingress weight by accumulated grounding evidence (no ungrounded noise) | M |
| SG8 | Consolidation + drift under closed-loop control | FSQ no-collapse (addressing stable); smoothness loss present; `fsq_in` drifts (meaning not intrinsic); no binding-stability signal | Rest-driven interleaved replay; **binding-churn as a first-class verdict** that auto-slows/freezes `fsq_in` (closed loop, not passive telemetry) | M/L |

## 5. Risks & tradeoffs

- **Ungrounded feedback = noise (highest risk).** Wiring the code into
  deliberation before it means anything degrades cognition. Mitigation: SG7
  grounding-gated ramp; parity flag defaults off; A/B on the ladder.
- **Code-meaning drift (now a controlled quantity, not an open risk).**
  `fsq_in` carries gradient → a code's referent can shift over a long run; the
  literature does *not* guarantee index-level semantic stability (§2). This is
  handled as **closed-loop control**, not a hope: **binding churn** (rate an
  entity's top code flips) is a first-class verdict; sustained high churn
  auto-slows/freezes `fsq_in` (WBS 5.2–5.3). Layered defenses: smoothness loss
  (present) keeps the code space's relational structure; grounding-by-
  co-occurrence keeps meaning in the *binding* where drift self-corrects
  (research [4][5]); the freeze is the literature-standard encoder-drift
  mitigation [21]. Residual: freezing too early starves grounding, too late
  lets meaning wander — so the freeze is churn-*and*-grounding-gated, not a
  fixed schedule.
- **Positive signaling ≠ listening.** A code may correlate with an entity yet
  not influence behavior. Validate grounding by *behavioral use / prediction*,
  not co-occurrence counts alone (research [8]).
- **Graph vs VSA for recall.** Explicit graph = exact but no dense trunk
  channel; VSA = dense fixed-width but lossy (crosstalk). Recommendation: graph
  for the exact store, VSA-bundle only for the recall-into-`memory_context`
  vector — keep the lossy path off the system-of-record.
- **Checkpoint/parity.** All new ingresses zero-init; new record category +
  episodic key are additive; non-strict load. Firewall: integer codes only.
- **Consolidation interference.** Massed symbol replay could overwrite; use
  interleaved low-rate replay (research [11]).

## 6. WBS

Effort S/M/L. No calendar estimates (work lands in minutes-to-hours). Deps by ID.
All flag-gated, parity-tested (flag off == today), verdict-instrumented,
validated on the 30 min → 2 h → 6 h ladder.

**1.0 Publish the wire (SG1)**
- 1.1 (S) Put `symbol_code` + `_prev_symbol_q` on `ctx.state_bus`/`latents` from the pipeline. Dep: none. **Gates 2.0–5.0.**
- 1.2 (S) Telemetry: expose published code + a per-cycle "active code" on metrics allowlist. Dep: 1.1.

**2.0 Relational storage (SG2, SG3, SG4)**
- 2.1 (M) Extend `SEMANTIC_CATEGORIES` with `symbol`; seed `semantic_stats`/`_cached_belief_stats`. Dep: none.
- 2.2 (M) Stage 10: attention-gated `_upsert_semantic` entity↔code evidence (cross-situational). Dep: 1.1, 2.1.
- 2.3 (S) Stage 10: `summary["symbol_code"]` on the episode. Dep: 1.1.
- 2.4 (M) Promotion to stable entity↔code concept at evidence threshold. Dep: 2.2.
- 2.5 (S) Tests: binding accrues, joint-attention gating, firewall-safe, promotion. Dep: 2.4.

**3.0 Recall into cognition (SG5)**
- 3.1 (M) On entity re-match, surface consolidated top code as a WM-slot belief (reuse `wm_slots` path). Dep: 2.4.
- 3.2 (L) VSA bind/bundle entity codes → `memory_context`-width recall vector (fills the missing dense channel). Dep: 2.4.
- 3.3 (M) Tests: entity cue → correct bound code recall; capacity/crosstalk bound checked. Dep: 3.1/3.2.

**4.0 Integrate into cognition (SG6, SG7)**
- 4.1 (M) Zero-init `symbol_ingress` on policy tier + `symbol_vec` kwarg + additive block. Dep: 1.1.
- 4.2 (M) Feed `_prev_symbol_q` (and S3 recall) through it; grounding-gated ramp weight. Dep: 4.1, 2.4.
- 4.3 (M) Tests: byte-identical at zero weight (parity); grounded code shifts deliberation; ungrounded contributes ~0. Dep: 4.2.

**5.0 Consolidation & stability (SG8) — drift as closed-loop control**
- 5.1 (M) Rest-driven interleaved replay of symbol bindings (reuse WS-ATTN rest + warehouse). Dep: 2.4, WS-ATTN rest.
- 5.2 (S) **Binding-churn verdict**: define churn = rate an entity's top-evidence code flips; emit as a first-class PASS/RED verdict row (drift proxy). Dep: 1.2, 2.2.
- 5.3 (M) **Closed-loop `fsq_in` freeze**: sustained churn above threshold AND grounding-mature → auto-slow then freeze `fsq_in` LR; releases if grounding regresses. Not a fixed schedule. Dep: 5.2, SG4.
- 5.4 (S) Tests: churn metric correct; freeze fires on synthetic sustained-churn; parity when churn low. Dep: 5.3.

**6.0 Validation & operability**
- 6.1 (M) Ladder runs with green verdicts; A/B (feedback off vs on) shows no regression + measurable grounded-use. Dep: 4.3, 3.3.
- 6.2 (S) Operator runbook. Dep: 6.1.

**Critical path:** 1.1 → 2.2 → 2.4 → {3.x, 4.2} → 6.1. Storage (2.0) and
feedback (4.0) both hinge only on 1.1; recall (3.0) and consolidation (5.0)
depend on the binding existing (2.4).

## 7. Open decisions

- **Feedback representation:** quantized 5-vector (recommended — fixed geometry,
  no drift) vs a learnable `nn.Embedding(4800, D)` (richer, re-introduces
  drift). Recommend starting with the vector; add an embedding only if capacity
  is limiting.
- **Recall channel:** WM-slot belief (simplest, reuses existing path) vs VSA
  dense vector (fills the missing channel, lossy). Recommend WM-slot first, VSA
  as the follow-on that unlocks direct recall.
- **When to turn feedback on:** grounding-maturity threshold value; likely tied
  to the same developmental signals the parent/language work uses (an adaptive
  other accelerates grounding — research [7]).
- **Relationship to WS-PARENT:** the language loop supplies the "adaptive other"
  that grounds codes fastest; symbol grounding here and language teaching there
  are the same arc and should share the grounding-maturity signal.

## Sources

[1] DreamerV2 — https://arxiv.org/abs/2010.02193 ; https://eclecticsheep.ai/2023/07/06/dreamer_v2.html
[2] VQ-VAE — https://arxiv.org/abs/1711.00937 ; IRIS — https://arxiv.org/abs/2209.00588
[3] Discrete-codebook world model for control (FSQ) — https://arxiv.org/abs/2503.00653
[4] Harnad, Symbol Grounding Problem — https://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf
[5] Smith & Yu, cross-situational word learning — https://pmc.ncbi.nlm.nih.gov/articles/PMC2271000/
[6] Joint attention & vocabulary — https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.589096/full
[7] Referential games / emergent communication — https://arxiv.org/pdf/1804.03984
[8] Positive signaling ≠ positive listening — https://arxiv.org/pdf/1903.05168
[9] Anti-efficient encoding — https://arxiv.org/pdf/1905.12561
[10] Hippocampal indexing theory — https://pubmed.ncbi.nlm.nih.gov/17696170/
[11] Complementary Learning Systems (updated) — https://stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
[12] Replay causally required — https://pmc.ncbi.nlm.nih.gov/articles/PMC2801761/
[13] Concept cells (Quiroga et al. 2005) — https://www.nature.com/articles/nature03687
[14] Concept cells review — https://pubmed.ncbi.nlm.nih.gov/22760181/
[15] Graph RAG survey — https://arxiv.org/pdf/2408.08921
[16] Zep temporal KG agent memory — https://arxiv.org/abs/2501.13956
[17] VSA/HDC survey (bind/bundle/cleanup) — https://arxiv.org/pdf/2111.06077
[18] Superposition capacity — https://arxiv.org/abs/1707.01429
[19] Neuro-vector-symbolic architecture (NVSA) — https://www.nature.com/articles/s42256-023-00630-8
[20] FSQ — https://arxiv.org/abs/2309.15505
[21] Codebook collapse / encoder drift — https://arxiv.org/abs/2606.11363

**Verification flags:**
- *Established:* FSQ addressing stability / no-collapse [20]; learned-projection
  encoder drift causes code-meaning drift [21]; meaning is external/grounded and
  recoverable by co-occurrence aggregation [4][5]. These are load-bearing and
  cited.
- *NOT established (design around it):* index-level *semantic* constancy of a
  fixed code over a long run — no source measures it. Handled as closed-loop
  control (binding-churn verdict → auto-freeze, §5 / WBS 5.x), not a hope.
- *NOT reliable:* "compositionality ⇒ generalization" in emergent-language
  studies — validate grounded codes by *behavioral use*, never assume transfer.
- *Orienting only (outside this pass's verified set):* biological
  representational drift with preserved function (Ziv 2013 / Rule & O'Leary) —
  cited as motivation for putting meaning in the binding, not as a pinned claim.
