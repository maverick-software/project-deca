# WBS: WS4B — Graph Writes Off the Critical Path

**Companion:** `ws4b_offlock_graph_flush_prd.md` · Dependency order only. ⚙ = needs the live rig.
**Scope discipline:** this WS ends at throughput parity + integrity evidence. The
deferred recall/prefetch items (PRD G6) are recorded here as out-of-scope.

---

## M0 — Ground truth probes (before any refactor)

**M0.1 Dual-connection probe.** Unit test: one kuzu Database, read connection +
write connection, concurrent-ish use (write txn in a thread while the read conn
queries) — proves the G4 assumption on kuzu 0.11.3/Windows before the design
depends on it. Failure ⇒ fall back to a single connection with a flush mutex
(design note updated, throughput target unchanged).
**M0.2 Statement-cost micro-bench.** Measure per-statement ms inside a txn for
MATCH...SET vs CREATE at realistic param sizes (explains the 4.6 ms/statement;
informs whether prepared/cached statements are worth an M3 follow-up).
*Accept:* both findings recorded in the PRD.

## M1 — Two-phase flush (resolve / execute split)

**M1.1 `_resolve_ops_locked`.** Pure transform: pending ops → fully-resolved
`[(cypher, params)]`, membership sets consulted AND updated here; no kuzu
calls. `_apply_op_locked` reduces to execution-only (or is retired).
**M1.2 Failure re-resolution.** The rollback path's per-op replay re-derives
CREATE-vs-SET safely after a failed batch (existence probe or CREATE-fallback —
pick from M0.2 data). Test hook to inject a failing op mid-batch.
*Accept:* unit tests — resolution purity (no connection touched), set updates
correct for create/update/delete interleavings, failed-batch replay leaves
memory untouched and kuzu consistent.

## M2 — Flusher thread + drain barrier

**M2.1 Flusher.** Daemon thread, FIFO queue of resolved batches, own write
connection, one explicit transaction per batch, rollback+replay per M1.2;
lazy start, joined on close. Flag `DECADIC_KUZU_OFFLOCK_FLUSH` (on by default;
off = 07-04 inline behavior, kept for A/B).
**M2.2 `drain()` barrier** wired into `backup_to`, `restore_from`, `close`,
`clear`, `_ensure_index_locked`; restore/close also close the write connection
inside the barrier (Windows file-lock discipline).
**M2.3 Telemetry.** `graph_flush_ms`, `graph_flush_queue_depth`,
`graph_flush_lock_ms` into `persistence_metrics()` → runtime metrics → diag
summary pattern list.
*Accept:* full suite green both flag states; WS4 backend/parity suites green;
route-level checkpoint tests green; drain-order test (mutate → backup → mutate
→ restore → state equals backup) passes with the flusher live.

## M3 — ⚙ Evidence on the rig

**M3.1 Diag A/B rerun.** `run_body_diag.ps1` both arms: kuzu within 10% of
sqlite (PRD criterion 1); `graph_flush_lock_ms` ≈ 0; disk <10% active.
**M3.2 1-h body soak on pure defaults** (kuzu, all faculties on): no stalls,
watchdog restored to 1.0 s, report archived — the embodiment-readiness stamp.
**M3.3 Disposition addendum** in the WS4 bench report (numbers table, before/
after: 0.8 → 2.27 → target ~4.9 cycles/s across the three write-path stages).

## Out of scope (recorded, PRD G6)

Perception-anchored recall query (fold-time anchoring, per-frame refresh) and
scene-keyed semantic prefetch — measured as not the current speed lever (reads
are sub-ms); scheduled with MuJoCo-era memory work / post-WS5. Prepared-
statement caching for kuzu (only if M0.2 shows planning dominates and M3.1
misses the 10% target).

## Dependency graph

```
M0.1 ─┬─> M1.1 -> M1.2 -> M2.1 -> M2.2 -> M2.3 -> M3.1 -> M3.2 -> M3.3
M0.2 ─┘                     (M0.2 also feeds the optional prepared-stmt follow-up)
```
