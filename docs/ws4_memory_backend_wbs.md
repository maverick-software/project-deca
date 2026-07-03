# WBS: WS4 — Memory Backend Migration (LanceDB + Kuzu)

**Version:** 1.0 — 2026-07-03 · **Companion PRD:** `ws4_memory_backend_prd.md`
**Convention:** 1 d = one focused dev-day. ⚙ = needs Charles's machine. Everything else buildable off-box against the seam + fixtures.

---

## M0 — Ground truth and the seam (est. 2 d) — no new engine code

**M0.1 ⚙ Dependency smoke test** (0.25 d)
`uv pip install --python .venv\Scripts\python.exe lancedb kuzu` (pinned into a `[memory]` extra in pyproject). Tiny smoke script: create/insert/vector-search/close for both, on Windows, inside the venv. **Go/no-go gate for Kuzu on Windows** — fallback path (LanceDB-only WS4) decided here, not discovered in M2.
*Acceptance:* both libraries import, round-trip data, and survive process restart on the dev box.

**M0.2 Embedding layout freeze** (0.25 d)
Extract exact sub-range offsets (narrative/emotion/metacog/z5/percept-key) from `decadic/memory/embeddings.py` into named constants; document in the metric catalog. The percept-key slice is load-bearing for 5.5.
*Acceptance:* constants + unit test asserting layout; no magic offsets left in callers.

**M0.3 Backend seam extraction** (1 d)
`EpisodicBackend` / `GraphBackend` protocols mirroring today's public methods; current SQLite code becomes `SqliteEpisodicBackend` / `SqliteGraphBackend` (moved, not rewritten); façades select backend from `DECADIC_MEMORY_BACKEND` / `DECADIC_GRAPH_BACKEND` (default `sqlite`).
*Acceptance:* full pytest suite green with zero env changes (the parity culture's baseline proof); no caller outside `decadic/memory/` touched.

**M0.4 Benchmark harness + SQLite baseline** (0.5 d)
`scripts/bench_memory.py`: synthetic episodes at 10k/100k/1M — add-rate, `search_similar` p50/p95, percept-key search p50/p95, graph node-match p50/p95, RSS. Runs against any backend combination; writes `reports/ws4_bench_<backend>.json`.
*Acceptance:* ⚙ baseline numbers for sqlite-linear recorded (the curves that justify or size the whole workstream). Depends on M0.2, M0.3.

## M1 — LanceDB episodic backend (est. 2 d)

**M1.1 Backend implementation** (1 d)
Table per PRD 5.2 (embedding + dedicated percept_key vector columns), brute-force search below index threshold, ephemeral temp-dir mode for `db_path=None`, writes serialized behind the existing lock, pruning by salience/age via delete-where.
*Acceptance:* backend-level unit tests (add/search/prune/rows/meta) pass. Depends on M0.3.

**M1.2 Parity suite** (0.5 d)
Same operation sequence into sqlite + lancedb backends; assert identical top-k (exact mode), identical pruning outcomes, identical `last_best_similarity`.
*Acceptance:* parity test in CI-able form; full suite green with `DECADIC_MEMORY_BACKEND=lancedb`. Depends on M1.1.

**M1.3 ANN index path** (0.5 d)
Index creation at threshold N (default 100k), recall@5 ≥ 0.95 tolerance test vs brute force, manifest records index state.
*Acceptance:* ⚙ bench at 1M shows recall p95 within PRD target; recall@5 documented. Depends on M1.1, M0.4.

## M2 — Kuzu semantic graph backend (est. 2.5 d; contingent on M0.1 go)

**M2.1 Schema + backend implementation** (1.5 d)
Node/rel/belief tables per PRD 5.3; identity match via vector index top-1 ≥ 0.6 with the existing match cache retained in front; `snapshot()` as windowed query returning today's dict shape; ephemeral in-memory mode for tests.
*Acceptance:* backend-level unit tests incl. belief update semantics (evidence counts, contradiction thresholds) matching SQLite behavior exactly. Depends on M0.3.

**M2.2 Parity suite** (0.5 d)
Identity decisions, belief trajectories, edge accrual, snapshot shape — sqlite vs kuzu on identical event streams.
*Acceptance:* full suite green with `DECADIC_GRAPH_BACKEND=kuzu`. Depends on M2.1.

**M2.3 Dashboard verification** (0.5 d) ⚙ (10 min)
Launch stack with kuzu backend; Graph/LTM panels render identically.
*Acceptance:* visual check + snapshot-shape test. Depends on M2.2.

## M3 — Novelty rewire (est. 1 d) — closes WS3 Phase B fix #1

**M3.1 `search_similar_percept()`** (0.5 d)
Façade method backed by the percept_key column (lancedb) / percept-slice scan (sqlite fallback), plus `last_best_percept_similarity` side channel.
*Acceptance:* unit tests; works on both backends. Depends on M1.1 (fallback path only needs M0.2/M0.3).

**M3.2 Gate integration** (0.5 d)
`DECADIC_GATE_NOVELTY_SOURCE=full|percept` (default `full`); gate extraction reads the percept channel when set; telemetry gains `gate_i_novelty_source`.
*Acceptance:* unit test with synthetic stores showing loop-repeat → low novelty, teleport → high novelty; ⚙ gate probe re-run with `percept`: ambient < 0.5, injected > 0.9, and the calm criterion re-evaluated (the 2026-07-02 blocker). Depends on M3.1.

## M4 — Checkpoint/restore (est. 1 d) — the schedule risk

**M4.1 Per-backend snapshot semantics** (0.75 d)
Quiesce (drain write-behind queues + hold write locks) → LanceDB directory/version copy, Kuzu checkpoint + copy; restore = swap on closed handle; wired into the existing `/agent/{id}/checkpoint` + restore routes.
*Acceptance:* checkpoint → mutate → restore → equality, automated per backend, plus round-trip through the REST routes.

**M4.2 Failure drills** (0.25 d)
Kill -9 during write burst → reopen → store readable, at most the in-flight batch lost (document guarantee per backend).
*Acceptance:* scripted drill passes on both backends. Depends on M4.1.

## M5 — Validation and cutover decision (est. 1 d + machine time)

**M5.1 ⚙ Full matrix run** (0.25 d + ~15 min)
pytest across the four backend combinations.
*Acceptance:* all green (or failures triaged to backend bugs, fixed).

**M5.2 ⚙ Benchmarks + report** (0.25 d + ~1 h)
`bench_memory.py` at 10k/100k/1M on all backends → `reports/ws4_bench_report.md` with the SQLite-vs-new curves.
*Acceptance:* PRD section-6 targets evaluated with real numbers. Depends on M0.4, M1.3, M2.2.

**M5.3 ⚙ 1-hour soak on new backends** (0.25 d + 1 h)
`run_soak.ps1 -Hours 1` with lancedb+kuzu env; WS2 gates must pass; memory-growth metrics compared against the 2026-07-02 sqlite shakedown.
*Acceptance:* gates green; no cycle-rate regression >5%. Depends on M1–M4.

**M5.4 Default-flip decision + docs** (0.25 d)
With bench + soak evidence: flip defaults (or don't — evidence decides), update PRD open-decisions log, recall-cache retire/keep decision, memory notes, and the master plan.
*Acceptance:* decision recorded with rationale; if flipped, full suite green on new defaults.

---

## Totals and sequencing

Dev effort: **~9.5 focused days**. Machine time: M0.1 smoke (5 min), baseline + benches (~1.5 h), gate-probe re-run (~10 min), matrix pytest (~15 min), 1-h soak.

```
M0.1 -> (go/no-go for M2)
M0.2 -> M0.3 -> M0.4 ------------------\
        M0.3 -> M1.1 -> M1.2 -> M1.3 ---+-> M5.1 -> M5.2 -> M5.3 -> M5.4
        M0.3 -> M2.1 -> M2.2 -> M2.3 ---/
M0.2 -> M3.1 -> M3.2 (gate probe re-run)
M1.1, M2.1 -> M4.1 -> M4.2
```

Critical path: M0.3 → M2.1 → M2.2 → M4.1 → M5. M1 and M2 parallelize. M3 can land early via the sqlite fallback path and unblock the WS3 gate-probe re-validation before the engines are even done.

## Explicit dependencies on other workstreams

- **Feeds WS3 Phase B:** M3 is Phase B fix #1; the gate probe's novelty/calm certification is blocked on it.
- **Feeds WS2:** the still-pending 12-hour soak should run post-M5 on whichever backends win, giving criterion-1 evidence on the final architecture (running it twice — once per backend set — is a bonus comparison, not a requirement).
- **Independent of:** the baseline reactive agent (criterion 5), which remains the last unbuilt evidence piece after WS4.
