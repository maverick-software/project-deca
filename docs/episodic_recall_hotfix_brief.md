# Dev Brief — Fix the Cognitive-Cycle Hang (Episodic Recall)

**Symptom:** the agent moves in a jerky start/stop way, synced to the Decadic cycle hanging every few cycles. GPU is near-idle during the hang.

**Root cause:** episodic memory recall runs every cycle, synchronously, on the asyncio event loop under the agent lock, and it does a **full table scan** of the entire episodic store with no `LIMIT`, JSON-decoding every embedding and running a Python cosine loop. Cost grows linearly with store size (one row/cycle) — measured ~192 ms at the `Memory Retrieval` stage and rising. It's CPU/SQLite/JSON work, so the GPU stays idle while the loop stalls and the body freezes.

- Hot path: `decadic/cycle/neural_pipeline.py:~458` → `ctx.episodic.retrieval_context_vector(...)` (inside `run_neural_cycle`, under `self.lock`).
- Offending query: `decadic/memory/episodic_store.py` `_iter_scored_rows` → `SELECT ... FROM episodes WHERE embedding_json IS NOT NULL` (no `LIMIT`), then per-row `json.loads` + `_cosine_similarity` in `search_similar`.

Two work items. **Item 1 fixes the cost; Item 3 fixes the blocking. Do both** — neither alone is sufficient.

---

## Item 1 — Bound the recall + keep an in-RAM vector cache

Make per-cycle recall constant-time and JSON-free.

**Do:**
- Maintain an in-memory matrix of candidate embeddings (`float32` `[N, EMBEDDING_DIM]`) plus parallel `salience`/`meta` arrays, kept bounded (e.g. most-recent **N** plus a high-salience long-term subset — not recency alone, so salient old memories survive). Update it on the same path that writes episodes.
- Compute recall as a single vectorized cosine: normalize once, `sims = M @ q`, `argpartition` top‑k. No SQL scan, no `json.loads` on the hot path.
- If you keep the SQL path as a fallback, add `ORDER BY salience DESC LIMIT <cap>` (cap ~512) so it can never scan the whole table.

**Acceptance:**
- `memory_recall_ms` (in `_diagnostics`, visible with `DECADIC_CYCLE_PROFILE=1`) drops to ~1–2 ms **and stays flat** as the store grows to 50k+ episodes.
- Recall results are equivalent to today's top‑k for the candidate set (don't silently drop high-salience memories).

---

## Item 3 — Get recall off the cognitive loop

Even fast, a per-cycle blocking lookup shouldn't sit inside the cycle's critical section.

**Do (either, preferably both):**
- **Decouple cadence:** refresh the memory-context vector every **N** cycles (or on a small worker), cache it on the agent, and have `run_neural_cycle` read the latest cached vector. The context does not need to change every ~50 ms.
- **Off-lock / off-thread:** compute recall *before* taking `self.lock`, or via `asyncio.to_thread`, so the forward pass never waits on it.

**Acceptance:**
- The cognitive cycle never blocks on a memory query; cycle cadence is smooth (no periodic hang) and body motion is continuous.
- `DECADIC_CYCLE_PROFILE=1` shows `memory_recall_ms` no longer on the critical section (or amortized across N cycles).

---

## Constraints / gotchas

- **Write-behind:** reads must not force a synchronous flush of the pending write queue. Feed the RAM cache from the write path so the hot path never touches disk.
- **Cognition parity:** recall feeds stage 3 and the perceptual-similarity loop. Preserve existing semantics (`top_k=5`, `min_salience`) and keep the RAM cache consistent with what is persisted, so behavior doesn't drift.
- **Tests:** the suite pins synchronous episodic writes (`tests/conftest.py`). Keep the baseline path byte-identical or update the affected tests; add a `memory_recall_ms`-stays-flat regression test over a grown store.
- **Don't lose long-term recall:** the bounded cache must retain a high-salience long-term subset, not just the recent window.

## Out of scope (deferred)

- Migrating episodic to **LanceDB** (the original design's vector store) — a separate, deliberate task once this hotfix lands.
- A vector index on the **LTM node embeddings** — only if `ltm_graph.match` becomes a measured bottleneck.
