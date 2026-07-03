# PRD: WS4 — Memory Backend Migration (SQLite → LanceDB + Kuzu)

**Version:** 1.0 — 2026-07-03
**Status:** Draft for review
**Decision of record:** Neo4j dropped (2026-07-02). Targets: **LanceDB** (episodic vectors) + **Kuzu** (semantic graph, native vector index). Both embedded, `uv pip install`-able, no server processes.
**Companion:** `ws4_memory_backend_wbs.md`

Settled decisions stated declaratively; estimates and open decisions marked.

---

## 1. Why (and why now)

Two forces, one architectural:

1. **Scale.** Episodic recall and graph identity-matching are linear scans (numpy cosine over the recall cache; `SemanticGraph` matches new appearances against all nodes at cosine > 0.6, `ltm_match_ms` measured). Fast today (~0 ms at 30k rows), O(N) forever. Days-scale soaks and MuJoCo life put the store at 1M+ episodes, where linear similarity becomes cycle-budget. ANN indexes make recall ~O(log N).
2. **Signal quality (the WS3 finding).** The gate's novelty input is nearly blind (dynamic range ~0.05) because recall queries the full 80-d embedding, where internal-state drift swamps external familiarity. The fix — similarity over the 16-d **percept-key sub-vector** alone — needs a second indexed vector column. That is native in LanceDB and awkward bolted onto the current cache. WS4 is therefore not only a scale play; it unblocks WS3 Phase B fix #1.

## 2. Goals

1. Episodic store on LanceDB behind the existing `EpisodicStore` API; semantic graph on Kuzu behind the existing `SemanticGraph` API. **No cognition-side code changes** beyond the seams.
2. Percept-key sub-vector search exposed (`search_similar_percept(key_16d, top_k)`) and wired into the gate's novelty input.
3. Old backends remain the default until validation completes: `DECADIC_MEMORY_BACKEND=sqlite|lancedb`, `DECADIC_GRAPH_BACKEND=sqlite|kuzu` (defaults `sqlite` — byte-identical baseline preserved, consistent with the project's parity culture).
4. Benchmark evidence at 10k / 100k / 1M episodes: add-rate, recall p50/p95, RSS — SQLite-linear vs LanceDB, and node-match latency vs Kuzu.

## 3. Non-goals

- No schema redesign of what an episode or graph node *is* — same fields, new engine.
- No distributed/cloud storage; single workstation, single process.
- No change to consolidation logic (it reads episodes through the same API).
- The 12-hour soak remains a separate WS2 deliverable (it should run on whichever backend is default at the time; ideally once per backend for comparison).

## 4. Current-state facts the design must honor (from 2026-07-03 code inventory)

- `EpisodicStore`: 9 public methods; 80-d embedding with semantic sub-ranges (narrative/emotion/metacog/z5/percept-key — exact offsets confirmed from `decadic/memory/embeddings.py` in M0); SQLite `episodes` table + in-memory fallback when `db_path=None`; dual-cap recall cache (recent + salient) with dirty-flag renormalization; `threading.RLock`; batch commits + WAL checkpoints + salience/age pruning; `last_best_similarity` side channel (gate novelty). Callers: neural_pipeline (recall), stage_10/write-behind (writes), consolidation (replay reads), api routes (memory query).
- `SemanticGraph`: 11 public methods; nodes/edges/beliefs/semantic tables; identity = appearance cosine > 0.6 via linear scan over recent+salient cache; Bayesian property beliefs; windowed `snapshot()` for the dashboard.
- Write-behind: dedicated background workers with FIFO queues, immutable snapshots, log-and-continue error handling. Backends are called from worker threads → thread-safety contract must hold.
- **Checkpoint/restore uses SQLite's online-backup API** (`backup_to`/`restore_from`, byte-for-byte). LanceDB/Kuzu equivalents must be designed, not assumed.
- All memory tests run in-memory mode; the ~700-test CPU suite must keep passing with zero configuration.

## 5. Design

### 5.1 Backend seam (settled)
Extract `EpisodicBackend` and `GraphBackend` protocol classes capturing today's public surface exactly. `EpisodicStore`/`SemanticGraph` become thin façades choosing a backend from env at construction. SQLite backends = current code moved, not rewritten (lowest-risk refactor; parity trivially holds).

### 5.2 LanceDB episodic backend (settled shape, details M1)
- One table per agent: `episode(cycle, ts, salience, meta_json, embedding: vector(80), percept_key: vector(16))` — percept key stored both inside `embedding` (compatibility) and as its own column (novelty search).
- Search: brute-force under an index threshold (LanceDB scans are already vectorized/fast below ~100k), ANN index (IVF-PQ; estimate) created at threshold N (open: 100k default) on both vector columns.
- The in-memory recall cache is **retained initially** and LanceDB used for spills/large-N (open decision: retire the cache after benchmarks prove parity — leaning retire at M5, one source of truth).
- Ephemeral mode for tests: LanceDB over a temp dir, cleaned on close (in-memory fallback preserved verbatim when `db_path=None`).

### 5.3 Kuzu graph backend (settled shape, details M2)
- Node table `Entity(id, appearance: vector, kind, first_seen, last_seen, ...)`; rel tables `CO_PRESENT`, `SPATIAL`, `TEMPORAL`; `PropertyBelief` as node-attached table (same Bayesian fields).
- Identity match: Kuzu vector-index top-1 cosine ≥ 0.6 replaces the linear scan; the existing match cache stays in front (it's a hit-rate win regardless of engine).
- `snapshot()` implemented as a windowed Cypher query returning the same dict shape the dashboard renders today.
- Ephemeral mode: Kuzu in-memory database for tests.

### 5.4 Checkpoint/restore (risk item, settled approach)
Per-backend snapshot semantics: quiesce writes (write-behind queues drained + lock), then LanceDB = versioned table copy / directory copy; Kuzu = checkpoint + directory copy. Restore = directory swap on a closed handle. Acceptance: checkpoint → mutate → restore → state equals checkpoint, per backend, under the existing API routes.

### 5.5 Novelty rewire (WS3 coupling, settled)
`search_similar_percept()` on the episodic façade; the gate's novelty extraction switches to `1 − best percept-key similarity` behind `DECADIC_GATE_NOVELTY_SOURCE=percept|full` (default `full` until the gate probe re-validates). Expected effect: ambient novelty on a repeated loop drops well below the 0.80–0.87 plateau measured on 2026-07-02, restoring dynamic range for the probe's novelty/calm criteria.

### 5.6 Parity and correctness (settled)
- Parity tests: identical operation sequences into sqlite-backend and new-backend instances; compare top-k result sets (exact-match while brute-force; recall@5 ≥ 0.95 tolerance once ANN is active — ANN is approximate by nature and the tolerance is part of the contract), graph identity decisions, belief updates, snapshots.
- Full suite green in all four backend combinations (matrix run in M5; default combination on every run).

## 6. Success criteria

1. Full pytest suite green with defaults (sqlite) — untouched behavior; green with `lancedb`/`kuzu` opted in.
2. Benchmark report (`reports/ws4_bench_report.md`): at 1M episodes, recall p95 ≤ 10 ms and node-match p95 ≤ 5 ms on the dev box (estimates — calibrate at M0 against measured SQLite-linear curves), with add-rate ≥ current.
3. Checkpoint/restore round-trip proven per backend.
4. Gate probe re-run with `NOVELTY_SOURCE=percept`: ambient novelty baseline < 0.5 on the patrol loop and injected teleports > 0.9 (dynamic-range restoration — the WS3 blocker).
5. 1-hour soak on new backends passes existing WS2 gates.

## 7. Risks

- **Windows wheels/quirks** (Kuzu especially): M0 installs and smoke-tests both libraries on the dev box before any code depends on them. Fallback if Kuzu disappoints on Windows: keep graph on SQLite (the seam makes this cheap) and take only LanceDB; the WS3 novelty fix needs only LanceDB.
- **Write-behind thread contract**: LanceDB/Kuzu handle concurrency differently than SQLite+RLock; backends serialize writes behind the existing locks initially (no throughput regression risk; optimize later).
- **ANN approximation vs determinism culture**: brute-force mode is exact; ANN mode is opt-in with recall@k tolerance documented. Runs record which mode was active (manifest).
- **Checkpoint semantics divergence** — mitigated by 5.4 acceptance test; this is the most likely schedule-slipper.
- **Two new dependencies** in a research codebase — both pinned in pyproject `[memory]` extra; core install unaffected.

## 8. Open decisions (resolve by end of M1/M2)

- ANN index type and threshold (IVF-PQ vs HNSW; 100k default).
- Retire the in-memory recall cache after M5, or keep as L1.
- Kuzu belief storage: node properties vs separate table (perf-dependent).
- Whether the episodic LanceDB table also absorbs the WS2 harness's episode queries (out of scope unless free).
