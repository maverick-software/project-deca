# WS5-M0.1: Working-Memory / Slot Inventory Freeze

**Date:** 2026-07-04 · **Status:** frozen (changes to this layout require a PRD amendment)
**Companion:** `ws5_relational_binding_prd.md` §2 · `ws5_relational_binding_wbs.md` M0

---

## 1. MemorySlot field set (decadic/state/working_memory.py)

One slot = one remembered entity (Kahneman/Treisman object file, minus network-visible identity — that arrives in WS5-M4). Fields by family:

- **Identity/kind:** `entity_id` (oracle id `ent-*`/anonymous `obj-NNNN`), `kind`, `kind_hint` (perceptual only; label discipline enforced by `_as_property_evidence`), `entity_role`, `scene_entity_id`.
- **Appearance:** `appearance` (latent fingerprint, EMA-refreshed at `appearance_ema`; the re-identification key and — after M4.1 — the graph-entity key).
- **Space/motion:** `position`, `relative`, `pos_history` (deque 8), `uv`, `uv_history` (deque 8), `bearing`, `motion`, `local_motion`, derived `heading()`, `predicted_uv()` (constant-velocity association).
- **Salience/affect:** `salience` (refreshed to 1.0 on sight, decays by `WorkingMemory.decay` otherwise; provisional slots decay faster), `affective_weight` (event-signed, clamped ±5), `audio_intensity` (faster `AUDIO_DECAY`), `last_event`, `event_links`/`relationship_links` (anonymous class links, capped 16).
- **Epistemics:** `confidence`, `precision` (saturating eta updates), `provisional`/`evidence_count` (promotion at `entity_promotion_precision` + seen_count ≥ 2), `contradiction_pressure` (appearance-flip pressure), `prediction_error`, `prediction_uncertainty`, `occlusion_age`, `property_evidence` (Bayesian evidence dict).
- **Agency:** `agency`, `agency_seen` (efference-explained motion; promotion to `kind="self_part"`).
- **Bookkeeping:** `last_seen_cycle`, `seen_count`.

## 2. Slot lifecycle

- **Integrate (oracle):** `WorkingMemory.integrate()` — decay all → assimilate observed entity nodes by oracle id (new slot at salience 1.0 / refresh) → `_bind_events` (by source id) → evict below `min_salience` → capacity cut (salience-ranked keep).
- **Integrate (discovered):** `integrate_discovered()` — decay (provisional boost) → confidence entry floor → greedy match by appearance-cosine ⊕ predicted-uv (`_match_score`, threshold 0.35) → LTM `reidentify` before coining `obj-NNNN` (reinstatement path) → `_refresh_slot`/`_new_slot` → `_bind_events_discovered` (affect attributed to most salient in-view slot) → evict/capacity as above.
- **Agency update:** `update_agency()` EMA + touch cross-check.
- **Read paths (pre-WS5):** `active_slots()`/`entity_nodes()` (reports, stage 10), `attention_vector()` (State Bus A blend), `workspace_candidates()` (GWT competition), `snapshot()` (dashboard).

## 3. The three pooling chokepoints (the differentiable half's gap)

| # | Boundary | Mechanism that destroys structure | Where |
|---|---|---|---|
| 1 | Scene → stack | `scene_latent` is an EMA of the *pooled* percept (`deposit_scene`), then chunk-mean folded to the model dim (`_fold_scene`) before entering `top_down_perceive(scene=...)`. K slots never reach the network. | `working_memory.py` (`deposit_scene`, `_fold_scene`); `neural_pipeline.py` scene_t assembly (~L660) → `neural_stack.top_down_perceive` |
| 2 | Memory → stack | `retrieval_context_vector()` mean-pools the top-k recalled episode embeddings into one `mem_t`. Five remembered situations enter as their average. | `episodic_store.py` / `lancedb_store.py` (`retrieval_context_vector`); consumed at `neural_pipeline.py` (~L640) |
| 3 | Stage 3 → 4 | `z3 = stage3(cat(z2, ze, zm))`, `z4 = risk_mlp(z3)` — risk from one fused vector; no token set exists over which a relation could be computed. | `neural_stack.py` forward (~L690–701) |

Secondary superposition (recorded, out of WS5 scope): `attention_vector()` hashes each entity into a single channel of the A-blend vector — a fourth pooling, on the State-Bus side; GWT `workspace_candidates()` partially de-pools it already.

## 4. Frozen slot-tensor layout (M0.2, constants in working_memory.py)

`SLOT_TENSOR_DIM = 40` = appearance `[0:16]` ⊕ spatial `[16:27]` ⊕ scalars `[27:40]`.

- **appearance (16):** slot `appearance` truncated/zero-padded (`SLOT_APPEARANCE_SLICE`). Anonymous until M4.1 keys it from the graph entity.
- **spatial (11):** relative-or-position ×3 (`tanh(x/SLOT_POS_SCALE)`, scale 20), bearing ×2 (/π), uv ×2, motion ×2 (tanh), heading (sin, cos).
- **scalars (13):** salience, tanh(affective_weight), audio_intensity, agency, confidence, precision, evidence (cap `SLOT_EVIDENCE_CAP`=8), contradiction_pressure, looming, prediction_error, prediction_uncertainty, staleness (cycles-unseen / `SLOT_STALENESS_HORIZON`=32, capped), in_view.

Ordering: salience-descending, tie-break ascending `entity_id` (deterministic under equal salience — refreshed slots all sit at 1.0, so the tie-break is load-bearing, not cosmetic). Mask row i = True iff a live slot filled it. All values finite and bounded (~[-1, 1]) by construction.

## 5. No-behavior-change assertion

M0.1/M0.2 add constants and a read-side adapter only. No caller of `WorkingMemory` changes; the layout test (`tests/test_ws5_binding.py`) pins the frozen offsets exactly as `test_ws4_backends.py` pins the episodic embedding layout.
