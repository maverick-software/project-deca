# WS4 Benchmark & Cutover Report — Memory Backends

**Date:** 2026-07-03 · **Box:** i7-12700K / 64GB / RTX 3080 · **Git:** post-M2/M3
**Raw data:** `reports/ws4_bench_{sqlite,lancedb}_{10000,100000}.json`, `ws4_bench_graph-kuzu_10000.json`

## Measured (episodic store, 80-d embeddings, 200 queries)

| metric | sqlite @10k | lance @10k | sqlite @100k | lance @100k |
|---|---|---|---|---|
| add rows/s | 1,266 | 6,337 | 372 | 14,082 |
| search p50 | 0.36 ms | 8.2 ms | 0.55 ms | 29.7 ms |
| search p95 | 0.52 ms | 9.5 ms | 0.78 ms | 35.7 ms |
| percept p50 | 0.13 ms | 7.0 ms | 0.28 ms | 22.5 ms |
| RSS | 53 MB | 133 MB | 50 MB | 138 MB |

Graph (Kuzu, 10k observations → 636 nodes): match p50/p95 = 0.01 ms (match cache dominant), 200/200 hits, add 556 obs/s.

## The critical asymmetry (why the latency columns are not comparable)

The sqlite path's flat ~0.5 ms search is over its **capped in-memory recall cache**, not the corpus — at 100k rows it silently ignores the vast majority of stored episodes. Lance searches the **entire corpus** exhaustively. So the columns trade fidelity against latency:

- sqlite: O(1)-ish latency, recall window shrinks relative to lifetime (misses grow unboundedly);
- lance: full fidelity, latency grows ~linearly (≈0.3 ms per 1k rows), still inside the 70–90 ms cycle budget at 100k (~3 h of agent life at 10 Hz), too slow somewhere past ~300k **on the critical path** (recall can also run off-path via the existing cached-context mechanism).

Ingest is unambiguous: lance is 5× faster at 10k and **38× at 100k** (sqlite's insert path degrades with table size; lance's does not), with RSS ~85 MB higher.

## ANN index finding (M1.3 disposition: implemented, default OFF)

IVF-PQ indexes on both vector columns build correctly (dimension-aware sub-vector counts; `ann_builds` telemetry). Measured effect at 100k: **zero query speedup** — `search_similar`'s where-clause filters dominate without scalar indexes on the filter columns — while synchronous rebuilds cut ingest to 1.5k rows/s. Aggressive rebuild cadence made p50 *worse* (50.9 ms) via fragment/index overhead. Default is therefore `DECADIC_LANCE_INDEX_THRESHOLD=0` (disabled); the code path stays for the follow-up: scalar indexes on `has_embedding`/`salience` (or a post-filter search mode) before re-measuring. Until then, brute force is the measured-best configuration.

## Cutover decision (M5.4) — SUPERSEDED, see below

**Original decision (2026-07-03, morning): defaults stay `sqlite` + `sqlite` for now.** Rationale: current live runs (≤1 h, ≤50k episodes) are fully served; the sqlite cache's fidelity limit is not yet binding; and the untuned lance query path would put 20–30 ms on the hot loop for fidelity today's experiments don't yet exploit.

**Flip criteria (any of):**
1. Sustained runs where the corpus exceeds ~100k episodes AND full-corpus recall fidelity is experimentally required (long-horizon memory studies, the 12-h+ soaks);
2. the filter-aware index follow-up lands and re-measures at single-digit ms;
3. episodic ingest becomes a bottleneck (sqlite's 372 rows/s at 100k is already marginal for >10 Hz + replay writes).

Both backends remain fully validated and one env var away (`DECADIC_MEMORY_BACKEND=lancedb`, `DECADIC_GRAPH_BACKEND=kuzu`): 19/19 parity tests, full 773-test suite green on Kuzu, checkpoint round-trips proven at store level. The WS3 novelty channel (`search_similar_percept`) works on both backends today — the percept-source gate improvement is not gated on the cutover.

### Superseded by owner decision (2026-07-03): defaults flipped to `lancedb` + `kuzu` WITH a full-mirror L1

The fidelity-vs-latency trade above was a false choice: the sqlite path's ~0.5 ms search came from an in-memory cache, and nothing stops the lance store from carrying the same cache **uncapped**. `LanceEpisodicStore` now fronts every search with a **full-mirror L1 recall cache** — a write-through numpy mirror of every live embedding (contiguous float32 `(n, 80)` matrix grown by doubling; parallel id/cycle/salience arrays; the 16-d percept-key matrix is a zero-copy slice view) with Lance as the durability layer:

- **write-through on append** (read-your-writes holds before any flush), **bulk columnar load on open**, **id-mask invalidation on prune**, **rebuild on restore**;
- search is one vectorized normalized-dot-product over the entire corpus with pre-top-k boolean masks for `min_salience`/`exclude_cycle` — cache-speed latency at **full-corpus fidelity** (the asymmetry section above no longer trades anything away). 80-d float32 × 1M rows = 320 MB on a 64 GB box;
- memory guard: `DECADIC_LANCE_CACHE_MAX_ROWS` (default 2,000,000); past the cap the mirror disables itself (logged once) and queries fall back to the measured lance brute-force path. `recall_cache_stats()` reports enabled/size/hits/misses and `bench_memory.py` prints them as `cache=`.

Defaults are flipped in `decadic/memory/factory.py` (`memory_backend()` → `lancedb`, `graph_backend()` → `kuzu`); `DECADIC_MEMORY_BACKEND=sqlite` / `DECADIC_GRAPH_BACKEND=sqlite` remain fully supported as the legacy/parity mode. Mirror-vs-scan exact-equality tests live in `tests/test_ws4_backends.py` (M5 section).

**Measured with the mirror (2026-07-03, 100k rows, 200 queries):** search p50/p95 = **0.84/1.02 ms**, percept p50/p95 = **0.80/0.96 ms**, add = **18,223 rows/s**, RSS 188 MB, `cache=enabled:True,size:100000,hits:400,misses:0`. Full-corpus fidelity at the sqlite cache's latency — flip criterion 2's "single-digit ms" beaten ~10×, criterion 3's ingest bottleneck beaten 49× (18,223 vs 372 rows/s @100k). Two fixes landed during validation, both with regression coverage:

1. **NaN sanitize at the store boundary** (`_row_from_record`): sqlite silently persisted non-finite embeddings for the store's whole history (raw blobs; its cosine never ranked them); lance validates on `add()` and turned one such episode into a failed flush on the caller's thread (`test_api_dashboard` under the flipped defaults). Non-finite values are now zeroed before reaching the mirror or lance — same observable behavior as sqlite (row stored, never wins a search, norm-guard scores it 0.0). Test: `test_lance_nan_embedding_sanitized_at_boundary`.
2. **Gather-free mirror search**: the first bench measured 8.5 ms p50 — the fancy-index row gather (`emb[idxs]`) allocated and copied the full 32 MB matrix per query under permissive filters. Replaced with one contiguous BLAS matvec over the whole mirror + in-place −inf masking of filtered rows (top-k semantics and tie order unchanged); percept queries zero-pad to 80-d so they ride the same contiguous matvec instead of a strided column view. 8.46 → 0.84 ms p50.

## M5.3 — 1-hour soak A/B (2026-07-03 evening): lance+kuzu vs sqlite control

Same config, same box, back-to-back (`soak_20260703_184026` lance+kuzu · `soak_20260703_195521` sqlite control):

| metric | lance+kuzu | sqlite control |
|---|---|---|
| cycles in 1 h | **26,182** | 16,315 |
| cycle rate mean | **7.21 Hz** | 4.46 Hz |
| frames dropped | **693** | 1,775 |
| stalls / NaN recoveries | 0 / 0 | 0 / 0 |
| recall cache hits/misses | 32,727 / **0** | 20,392 / 1 |
| final `neural_pc_loss_last` | 0.1509 | 0.1509 |
| growth events | 4 | 2 |
| checkpoint on shutdown | 2.2 s | 1.2 s |

**Disposition: M5.3 closed, backend exonerated and preferred.** The new stack ran +61% more cycles at the same wall clock (the bench's ingest asymmetry showing up in live runtime — sqlite's insert path degrades with table size), with fewer drops and a 100% mirror hit rate. The pc-loss half-mean gate FAILed on lance and PASSed on sqlite, but this is a throughput artifact, not a learning regression: both runs converged to an **identical endpoint loss (0.1509)**; the faster run simply reached 4 growth events (vs 2), whose transient loss spikes land in the second half-mean — the canary the report already flags as unreliable on synthetic input. The tail-of-run cycle-rate slowdown appeared on **both** backends (growth cost, cognition-side).

Known harness cosmetic: "LTM db: 0.0 MB" measures the legacy sqlite file path; kuzu's store is a directory, so the report reads zero. Fix with the next harness touch.

## Remaining WS4 items

~~M4 checkpoint-route integration test~~ (closed: `tests/test_ws4_checkpoint_routes.py` — /checkpoint→mutate→/restore state equality + full save→load round-trip with directory-shaped snapshots) · ~~M5.3 1-hour soak~~ (closed: A/B above) · 1M-row bench (optional) · filter-aware ANN follow-up (deferred).

## WS4B disposition (2026-07-05): graph writes off the critical path — CLOSED, kuzu now FASTER than sqlite

The embodied-rig write-path arc, four diagnoses each caught by the telemetry the previous fix added (`run_body_diag.ps1`, 150 s arms, full preset, discovered perception):

| stage | cycles/s | evidence that named the next problem |
|---|---|---|
| original (per-op autocommit) | 0.8 | disk 100% active @ 7 MB/s (fsync storm); body watchdog ragdolling |
| + deferred batched flush (07-04) | 2.27 | `sqlite_last_commit_ms=293` — the batch ran UNDER the graph lock |
| + off-lock flusher (WS4B M2) | 3.00 | `lock_ms=0.1` ✓ but `queue_depth=39` — flusher saturated (~170 stmts/s > kuzu's absorption) |
| + window retune 512/10 s | 2.91 | `batch_commit_count == commit_count` in EVERY arm — `write_batch()` exits bypassed the deferral: one forced flush per cycle |
| + write_batch = grouping, not urgency | **5.38** | 23 flushes/150 s, queue 0, lock 0.32 ms, 0 error batches |

**sqlite control arm: 4.89.** Kuzu on the WS4B flusher is ~10% FASTER than sqlite on the same harness — acceptance (within 10%) exceeded. Mechanisms shipped: two-phase resolve-under-lock/execute-off-lock (resolve ~0.3 ms), dedicated write connection (M0.1 dual-connection probe PASSED; readers never queue behind a batch; `graph_dedicated_write_conn` telemetry), queue cap 4 with tail coalescing (bounded backlog, last-wins), set-agnostic per-op replay after rollback (failure drill tested), drain barriers on backup/restore/close/clear/index-build, `DECADIC_KUZU_OFFLOCK_FLUSH=0` inline A/B arm.

**Semantics decision of record:** `write_batch()` marks a consistency GROUPING; durability cadence is the deferral window (`DECADIC_KUZU_FLUSH_OPS=512` / `DECADIC_KUZU_FLUSH_S=10`). Crash window ≤10 s of graph deltas — acceptable because memory is the live source of truth and episodic memory (lance, sub-second flushes) is the experiential ground truth from which beliefs re-derive.

### WS4B M3.2 result (2026-07-05, 1-h pure-defaults body soak `bodydiag_kuzu_20260705_094611`)

**Stability stamp: PASS.** 11,640 cycles / 3,600 s, 0 stalls, 0 flush-error batches, lock 0.58 ms, commit lag 1.5 ms, queue bounded at cap, body upright throughout. **Performance finding:** throughput decayed 5.4 → ~2.1 cycles/s with no plateau. Attribution (metrics dump): (a) PRIMARY — neural growth: **8 growth events, 1,280 awake / 2,048 allocated neurons, pc_loss stuck at 0.91** — the PC-loss growth gate never tapers in an open embodied world, so the brain grew all hour (a POLICY finding, not a bug: growth needs progress-gating/budgets/pruning for lifelong runs — see "growth governance" ledger item); (b) SECONDARY — flusher re-saturation on INSERT-heavy load: append-only streams (new-id semantic records/entities) defeat per-key dedupe; final coalesced batch = 100 s (`graph_flush_ms=100094`), 1,709 flushes (~2 s cadence, ops-threshold driven).

Remaining WS4B: **M3.4** kuzu multi-row inserts (`UNWIND $rows CREATE` — N statements → 1 per table per flush; re-arms the design for insert-heavy load) · **M3.5** re-run the 1-h soak after M3.4 + a growth-frozen control arm (`DECADIC_GROWTH_ENABLED=0`) to isolate the growth share of the decay exactly.

New ledger item (feeds a future WS): **growth governance** — progress-gated growth (grow when improvement STALLS, not when absolute error is high), metabolic capacity budget, pruning/consolidation as the counter-force; first dataset = this soak.
