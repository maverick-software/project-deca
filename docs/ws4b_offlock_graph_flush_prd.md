# PRD: WS4B — Graph Writes Off the Critical Path (Kuzu Off-Lock Flusher)

**Version:** 1.0 — 2026-07-05
**Status:** Draft for review · **Companion:** `ws4b_offlock_graph_flush_wbs.md`
**Origin:** first embodied (MuJoCo) runs, 2026-07-04/05 — the first time discovered
perception ever populated the graph at per-frame rates.

---

## 1. The measured problem

Controlled A/B, same harness (`run_body_diag.ps1`, 150 s, full preset, CUDA, discovered perception):

| arm | cycles/s | graph flush p-last | flushes | ltm_match_ms | index rebuilds |
|---|---|---|---|---|---|
| graph=sqlite | **4.89** | 13.1 ms | 676 | 0.07 | — |
| graph=kuzu | **2.27** | **292.7 ms** | 316 (~1 per 0.5 s) | 0.14 | 0 |

The cycle-compute ceiling is ~4.9/s (cycle wall ~200 ms); sqlite runs at the
ceiling. Kuzu loses ~230 ms/cycle amortized — almost exactly the flush duration
times its frequency — because the batched flush transaction **executes while
holding `self._lock`**, the same lock perception/integration needs. Reads are
exonerated (match 0.14 ms, in-memory cache in front); rebuild storms are fixed
(60 s floor, 0 rebuilds); the disk fsync storm is fixed (2% active vs 100%).
What remains is lock-hold time: ~4.6 ms of kuzu query planning per statement ×
64-op batches.

History of this write path (context for reviewers):
1. **Original defect** (latent since WS4, exposed by MuJoCo): flush after every
   unbatched mutation with per-op autocommit → fsync storm, 100% disk active,
   0.8 cycles/s, liveness watchdog ragdolling the body.
2. **Hotfix 2026-07-04**: deferred batched flush (64 ops / 2 s, one
   transaction, rollback + per-op replay) + rebuild wall-clock floor. Disk
   fixed; throughput still 2.27/s because the batch runs under the lock.
3. **This WS**: move execution off the lock entirely. SQLite never needed this
   only because its commits cost 13 ms; the architecture principle (memory is
   the source of truth, engines are durability) says NO engine latency should
   ever be observable from the cognitive loop.

## 2. Gap analysis

| # | Gap | Current state | Required |
|---|---|---|---|
| G1 | **Flush executes under `self._lock`** | `_commit_locked` → `_flush_pending_locked` runs the ~293 ms kuzu transaction while holding the graph lock; perception's integrate/match waits | Resolve-under-lock (µs), execute-off-lock on a flusher thread with its own kuzu connection |
| G2 | **Op resolution is coupled to execution** | `_apply_op_locked` decides CREATE-vs-SET from membership sets *and* executes, in one step | Split: `_resolve_ops_locked` (pure: ops → [(cypher, params)] + set updates, under lock) / `_execute_resolved` (kuzu only, off lock) |
| G3 | **No drain barrier** | `backup_to`, `close`, `restore_from`, `clear`, `_ensure_index_locked` call `_flush_pending_locked` synchronously and assume completion | `drain()` barrier: enqueue-and-wait until the flusher confirms all resolved batches executed; these five call sites use it |
| G4 | **Single shared kuzu connection** | One `_kz_conn` used by reads (rare) and writes under one lock | Dedicated write connection owned by the flusher thread; reads keep `_kz_conn`; kuzu supports multiple connections per Database (concurrent transactions serialize engine-side) |
| G5 | **No stall observability** | `sqlite_last_commit_ms` records flush duration but nothing records lock-hold or queue depth | Telemetry: `graph_flush_ms` (off-lock duration), `graph_flush_queue_depth`, `graph_flush_lock_ms` (must be ~0), surfaced in metrics + diag summary |
| G6 | *(deferred, recorded)* Perception-anchored recall & scene semantic prefetch | Async recall exists (`DECADIC_MEMORY_CONTEXT_ASYNC`: query from previous cycle's state, refresh every K cycles); LTM→WM reinstatement carries per-entity beliefs at sighting | Fold-time query anchoring, per-frame refresh, scene-keyed graph prefetch — **quality/scale work, measured as NOT the speed lever** (reads already sub-ms); scheduled with MuJoCo-era memory work, not here |

## 3. Design (settled)

**3.1 Two-phase flush.** Under `self._lock`: swap `self._pending` out, run
`_resolve_ops_locked` — pure Python that consults/updates the membership sets
and emits fully-resolved `[(cypher, params)]` — and append the batch to the
flusher queue. Microseconds; no kuzu calls. Off lock: the flusher thread
executes the batch in one explicit transaction on its own connection.

**3.2 Failure semantics (unchanged culture).** Transaction failure → ROLLBACK →
per-op replay with log-and-continue **on the flusher thread**. Membership-set
consistency: sets are updated at RESOLVE time; a rolled-back CREATE would leave
a set claiming existence, so the replay path re-resolves the failed batch's ops
against an existence check (`MATCH ... RETURN` probe or tolerated
CREATE-then-SET fallback) — decided at M1 by measuring which is cheaper; the
invariant is that memory (`_nodes` etc.) is never touched by failures.

**3.3 Ordering.** One flusher queue, FIFO, single thread: batches execute in
resolve order, preserving today's semantics (the dict-dedupe already collapses
same-key updates within a batch).

**3.4 Drain barrier.** `drain(timeout)` — flushes the current pending set and
blocks until the queue empties. Used by backup/close/restore/clear/index-build.
`close()` drains with a bounded timeout then checkpoints, as today.

**3.5 Threading model.** One daemon flusher thread per graph instance, started
lazily on first resolved batch, joined on `close()`. The write-behind LTM
worker (upstream, existing) keeps feeding mutations; this thread only owns kuzu
execution. Windows caveat: thread must release kuzu handles before
`restore_from`'s directory swap (drain + close handles under the barrier).

**3.6 Flags.** `DECADIC_KUZU_OFFLOCK_FLUSH` (default ON once accepted; OFF
restores the 07-04 inline behavior for A/B). Existing knobs
(`DECADIC_KUZU_FLUSH_OPS/FLUSH_S/REBUILD_MIN_S`) unchanged.

## 4. Success criteria

1. **Throughput parity:** diag harness kuzu arm within 10% of the sqlite arm
   (i.e. ≥ ~4.4 cycles/s at the current ~4.9 ceiling) on the same machine.
2. **Lock cleanliness:** `graph_flush_lock_ms` ≈ 0 under load (resolve-only);
   flush duration may stay ~300 ms but off-lock.
3. **Integrity:** full suite green; WS4 parity tests green; backup → mutate →
   restore round-trip green THROUGH the drain barrier; reopen-after-close row
   counts exact.
4. **Failure drill:** injected op failure (test hook) exercises rollback +
   replay; memory state provably untouched; subsequent flushes healthy.
5. **Soak:** 1-h body-rig soak on pure defaults (kuzu) with disk <10% active
   and no stalls — the embodiment-readiness stamp for the default stack.

## 5. Risks

- **Kuzu connection concurrency:** two connections (read + write) on one
  Database — supported, but engine-side write serialization vs the read path's
  occasional index queries needs the M0 probe test before building on it.
- **Restore/close races:** the directory swap in `restore_from` while the
  flusher holds a connection is the classic Windows file-lock trap (WS4 scar
  tissue); the drain barrier MUST close the write connection too.
- **Ordering with reads-after-write:** `_ensure_index_locked` requires
  committed rows; it drains first (G3). Match paths read memory, unaffected.
- **Shutdown loss window:** unchanged from the 07-04 hotfix (≤2 s / ≤64 ops);
  `close()` drains, so clean shutdowns lose nothing.
