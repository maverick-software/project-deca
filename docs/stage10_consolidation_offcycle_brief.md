# Dev Brief — Move Stage-10 Consolidation Off the Cycle + Bound the LTM Match

**Symptom:** after the episodic-recall fix (stage 3: 192 ms → 1.89 ms), the cycle still hangs. The bottleneck moved to **stage 10 — Normative Memory Mapping (~548 ms)**.

**It is compute, not I/O.** The SSD is idle (~0%) and write-behind LTM (`DECADIC_LTM_ASYNC`) already moves the SQLite commit off-thread. So the 548 ms is the **synchronous consolidation compute running on the cognitive cycle thread**, and it now fires every cycle because the graph finally grows (the epistemic-maturity fix worked — before, with the graph stuck at 1 node, stage 10 was a near no-op).

**What's slow, in `stage_10.py` (lines ~95–142):**
- `ctx.ltm_graph.consolidate(...)` → re-identification via a **brute-force cosine loop over every node** in the in-memory graph (`semantic_graph.py:521`, `for nid, node in self._nodes.items()`). Same O(N)-growing pattern we just fixed for episodic — now in the LTM, and it also runs in stage 3 re-identification, so it's paid twice.
- `bump_edge(...)` over every scene relation between promoted nodes — O(k²) in co-present entities.
- `record_semantic_evidence(...)` over all slots × events × relationships, every cycle, with property-belief updates.

Note: write-behind only offloads the *commit*, not this *compute* — so "make the write async" does not help. Three work items.

---

## Item A — Defer stage-10 LTM consolidation to a background worker

Consolidation to long-term memory is an offline/rest process; it does not belong in the synchronous perceive→think→act loop. You already run a background consolidation thread (the replay consolidator).

**Do:**
- In stage 10, the cycle only **snapshots the stable slots** (detached copies of appearance/seen_count/precision/scene relations) and enqueues a consolidation job, then returns. The worker performs `consolidate` / `bump_edge` / `record_semantic_evidence` / writes off-thread.
- Take the graph's existing `RLock` in the worker; stage-3 re-identification reads under the same lock. The ~1-cycle visibility lag is already the accepted write-behind contract.
- Snapshot the slot data the worker needs so it can't race the next cycle's working-memory mutation.

**Acceptance:**
- Stage-10 time *on the cycle* drops to ~1–2 ms (just snapshot + enqueue); `DECADIC_CYCLE_PROFILE=1` shows `stage10_ms` off the critical path.
- Cycle cadence is smooth; body motion continuous.

---

## Item B — Bound the LTM `match` (re-identification)

Same medicine as episodic, applied to the graph.

**Do:**
- Replace the per-node Python cosine loop (`semantic_graph.py:521`) with a vectorized matmul over an in-RAM node-embedding matrix (`float32 [N, D]`), kept in sync with the graph.
- Cap/index the candidate set (recency/salience bucket, or an ANN index) so re-identification doesn't scale with total graph size.

**Acceptance:**
- `match` time stays ~constant as the graph grows to 10k+ nodes.
- Speeds stage 3 too (re-identification reads `match` every discovered cycle).

---

## Item C — Bound edge accrual + throttle semantic evidence

**Do:**
- Cap the number of relation pairs `bump_edge` processes per cycle (top-k by confidence), so it can't go O(k²) on a busy scene.
- Throttle `record_semantic_evidence` to every N cycles (or fold it into the Item-A worker). It's a slow statistical update — it does not need per-50 ms cadence.

**Acceptance:**
- Per-cycle consolidation work is bounded regardless of scene complexity or graph size.

---

## Constraints / gotchas

- **Thread-safety:** worker + cycle both touch the graph (use the existing `RLock`) and working memory (pass the worker *snapshots*, not live slot references).
- **Idempotency:** don't re-consolidate the same slot every cycle — key consolidation by node id and the existing `min_seen` / `precision` promotion gates.
- **Cognition parity:** consolidation feeds stage-3 re-identification and the dashboard graph. Preserve semantics — just defer and bound. Stage 3 must tolerate a ~1-cycle-stale graph (already true under write-behind).
- **Not an I/O problem:** the SSD is idle and the commit is already write-behind. Do **not** spend time on disk/write-behind tuning — the win is moving the *compute* off the loop and bounding its growth.
- **Tests:** add a regression that stage-10 cycle-time stays flat as the graph grows to N nodes; preserve the synchronous test path (`tests/conftest.py` pins).

## Out of scope

- Disk / write-behind tuning (SSD isn't the constraint).
- LanceDB episodic migration and a separate graph DB (tracked elsewhere; not needed for this).

---

**Pattern (for the record):** this is the third instance of the same root issue — heavy, *growing*, work running synchronously inside the per-cycle critical section (perceptual fold, episodic recall, now LTM consolidation). The durable rule: the cognitive cycle does only bounded, fast work; memory recall, consolidation, and persistence live *off* the loop, bounded, on a cadence. Faster, and more brain-faithful — perception shouldn't pause to file long-term memories.
