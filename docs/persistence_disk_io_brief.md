# Dev Brief — Persistence Layer: Stop Pegging the SSD

**Symptom:** Disk 1 (D:, the data dir) sustained ~65% active, sawtooth. Two costs: NVMe **wear** (high rate of tiny synchronous writes) and **unbounded growth** (DB files grow forever). Unsustainable.

**This is distinct from the cycle-hang.** Write-behind moved persistence *off the cognitive thread* (fixing latency) but does the *same writes* — so disk load is unchanged. This brief reduces the write load itself; it is complementary to the stage-10 off-cycle brief.

**Root causes (verified):**
1. **No WAL / `synchronous=FULL`.** No journal PRAGMA is set on any connection → SQLite fsyncs on **every** commit (`episodic_store.py`, `semantic_graph.py`).
2. **Commit-per-write.** Episodic does 1 insert+commit/cycle; LTM consolidation commits **several** times/cycle (nodes, edges, beliefs are separate commits: `semantic_graph.py:459/465/471/477`). Many fsyncs/cycle of tiny writes.
3. **JSON-text payloads.** `embedding_json TEXT` / `summary_json TEXT` / `appearance_json TEXT` — bloated bytes and parse cost per write (write amplification).
4. **No table retention.** The in-RAM recall cache is capped (`_prune_recall_items_locked`), but the **SQLite `episodes` table is never pruned** — it grows one row/cycle forever; LTM is unbounded too.

---

## Item A — WAL + relaxed synchronous (biggest win, smallest change)

On every connection open, set:
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```
WAL batches writes and fsyncs only at checkpoints (not per commit); `NORMAL` drops the per-commit fsync while staying crash-safe (worst case: lose the last few transactions on **power loss**, not on app crash — acceptable for a per-cycle diary and a reconstructable graph).

**Acceptance:** Disk 1 active% drops to low single digits at the same cycle rate; no per-commit fsync storm.

## Item B — Batch commits

- Wrap each cycle's LTM consolidation writes in **one transaction → one commit**, not 4+.
- Better: mutate the in-memory graph (it already lives in RAM) and **flush to disk on a cadence** (every N cycles / T seconds / size threshold), checkpoint-style, instead of committing per mutation. Same for episodic — accumulate and commit in batches.

**Acceptance:** ≤ 1 fsync per batch interval, not per write; commit count per cycle ≈ 0 amortized.

## Item C — Store vectors as BLOB, not JSON

- Migrate `embedding_json` / `appearance_json` to `float32` **BLOB** columns (raw bytes). Cuts bytes written and removes `json.dumps`/`loads` on the write path. (Also complements the recall fix.)

**Acceptance:** per-record write size and CPU drop measurably; no JSON on the persistence path.

## Item D — Retention / forgetting on the tables

- Cap the `episodes` table: keep a recent window + a high-salience long-term subset; periodically `DELETE` the rest (evict lowest-salience oldest — the "forgetting" the design already calls for). Run `VACUUM`/checkpoint **off-cycle**, on a cadence.
- Prune stale/low-confidence LTM nodes/edges so the graph file doesn't grow without bound.

**Acceptance:** DB file sizes plateau under continuous run instead of growing linearly forever.

---

## Constraints / gotchas

- **Crash-safety tradeoff:** `synchronous=NORMAL` can lose the last transaction(s) on power loss. Fine for the episodic diary; decide explicitly for the LTM (it's reconstructable from experience, so almost certainly fine). Document the choice.
- **WAL housekeeping:** WAL needs periodic checkpointing; let SQLite auto-checkpoint or trigger it off-cycle. Don't checkpoint inside the cognitive loop.
- **Retention must not drop high-salience old memories** — keep the salient long-term subset, not just recency.
- **Saved-agent / checkpoint bundling** copies the `.db` files; ensure WAL files (`-wal`/`-shm`) are checkpointed/flushed before a bundle/backup so saves are consistent (the write-behind `flush()` path already exists — reuse it).
- **BLOB migration:** add the BLOB column, backfill or dual-read old JSON rows during a transition, then drop JSON.
- **`VACUUM` is heavy** — never on the cycle thread; schedule it.

## Relationship to the other two briefs

- *Episodic recall* brief → fixed read cost (stage 3).
- *Stage-10 off-cycle* brief → fixes consolidation **compute** latency (and its throttle reduces write *frequency*, helping here too).
- *This* brief → fixes the **write/durability load** (fsync rate, write volume, unbounded growth).

Same principle across all three: **RAM is working state; disk is periodic, batched durability.** The cognitive loop should never fsync, scan, or consolidate synchronously — those belong off the loop, bounded, on a cadence.
