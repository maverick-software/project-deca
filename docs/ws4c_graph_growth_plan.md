# WS4C — Graph-Growth Governance (fix the 6-hour death spiral)

**Evidence:** `reports/bodydiag_kuzu_20260707_025552` — 4.6 → 1.0 cyc/s over 6 h;
commit_lag 13.74M ms (flusher 3.8 h behind); 10.1M writes deferred (~460/cycle);
deferred depth 10,921; **5,425 edges on 34 nodes**; wall 253→1193 ms; cycles
frozen at 22,139. Reads stayed flat (ltm_match ~0.1 ms) — this is purely the
WRITE side. WS4B's governance was validated at 1 h only; the spiral crosses
over once edge count makes per-cycle write arrival exceed flusher drain rate,
and the flusher thread strangles cognition through the GIL.

**The disease vs the symptom:** the flusher drowning is the symptom. The
disease is relational hygiene — 34 entities cannot meaningfully have 5,425
edges (~160 per node; a near-clique of stale `scene_near`-class relations that
are never pruned, refreshed per cycle as distinct keyed writes).

## M1 — Diagnose the edge factory (no code until this is answered)
- **M1.1** Per-kind write telemetry in `kuzu_graph.py`: writes-staged and
  deferred-depth broken down by op kind (node / edge-by-relation-kind /
  belief). One run answers *what* generates 460 writes/cycle.
- **M1.2** Inspect the 5,425 edges in the saved 107 MB db (offline notebook or
  script): kind histogram, age distribution, duplicate-pair count. Hypothesis
  to confirm: per-cycle scene-relation refresh creates/updates edges that are
  never retired.

## M2 — Relational hygiene (the real fix)
- **M2.1 Edge decay/retention:** scene-class edges get a last-confirmed cycle;
  a retention pass (piggybacking the existing prune机制) retires edges
  unconfirmed for N cycles. Degree cap as a backstop: keep the top-K edges per
  node by weight/recency (K ≈ 16; a 34-node scene then bounds at ~550 edges —
  under the flusher crossover by design).
- **M2.2 Refresh ≠ rewrite:** confirming an existing edge with unchanged
  payload must not stage a write at all (compare-before-stage; the upsert
  carries full state, so equality is checkable at stage time).
- *Accept:* steady-state edge count bounded and stable in a 2 h run;
  writes/cycle drops an order of magnitude.

## M3 — Flusher catch-up math (defense in depth)
- **M3.1 Adaptive batch sizing:** drain batch scales with deferred depth
  (arrival rate must never exceed drain rate; compute both, log the ratio as
  `graph_write_pressure` — >1.0 sustained = red).
- **M3.2 Backpressure escape:** if depth still grows past a ceiling, coalesce
  aggressively (the dedup path fired ZERO times in 6 h — verify why) and shed
  lowest-value writes (scene-edge refreshes first, beliefs last, nodes never).
- **M3.3 GIL relief:** measure flusher-thread CPU share; if still material
  after M2, move serialization work off-thread into precomputed statements.

## M4 — Guardrails + operability
- **M4.1** `graph_write_pressure` + per-kind depths on the metrics allowlist;
  probe verdict row (pressure < 1.0 sustained = PASS).
- **M4.2** Long-life trend poller: fix the nesting bug (regex the raw metrics
  body, as the probe does) and add write-pressure to the trend verdicts.
- **M4.3** Diag warns RED when commit_lag > 60 s (tonight it hit 3.8 h silently).

## M5 — Validation ladder
- 30 min (pressure < 1, edges bounded) → 2 h (rate ≥ 4 cyc/s end-to-end,
  wall_ms flat) → re-run the 6 h life. Only then do the maturation trends and
  the E4-trust / companion-path follow-ups (tracked in memory) get judged on
  solid ground.

**Order: M1 → M2 → M3 → M4 → M5.** M1 first — the fix must target the actual
edge factory, not my best guess at it. All changes flag-gated and
parity-tested per house discipline; tests pin retention/cap params.

---

## STATUS 2026-07-07: M1–M4 implemented; M5 pending

**M1 verdict (`reports/ws4c_m1_verdict.md`):** the dominant write factory was
NOT edges — it was anonymous `event` SemanticRecords (194,074 of 200,000 rows;
fresh id per instance at `semantic_graph.py:1033`, new keys bypass the
throttle). The plan's 460/cyc figure is the *deferred* arrival; staged was
~12/cyc, ~9 of them event CREATEs. Edges: 5,438, zero duplicates, but NO
delete path existed and >60% were stale. `commit_lag_ms` is stage-pipeline
deep-process staleness, not the flusher queue.

**Implemented (all flag-gated, defaults ON; 69 tests green incl.
`tests/test_ws4c_graph_hygiene.py`):**

- M1.1 per-kind write telemetry: `graph_writes_staged_<kind>` /
  `_deferred_<kind>` / `_skipped_<kind>` / `graph_deferred_depth_<kind>`
  (edges by relation kind), in `persistence_metrics`.
- M2.3 keyed events (`DECADIC_LTM_EVENT_KEYED`): one aggregate record per
  event_class — kills the dominant factory at the source.
- M2.1 edge retention (`DECADIC_LTM_EDGE_RETENTION_ENABLED`,
  `_STALE_CYCLES=2000`, `_DEGREE_CAP=16`, `_PRUNE_PREFIXES=scene_`) + new
  `del_edge` op / `_Q_EDGE_DEL` mirror into kuzu.
- M2.2 refresh≠rewrite (`DECADIC_KUZU_SKIP_UNCHANGED`, horizon
  `DECADIC_KUZU_REFRESH_MAX_CYCLES=1000`): unchanged-payload upserts (modulo
  count/last_cycle-class fields) stage nothing. NOTE: drain barriers are now
  exact modulo volatile fields within the horizon.
- M3.1 flusher merges up to `DECADIC_KUZU_FLUSH_MERGE_MAX=4` batches/wake
  (drain scales with backlog; dedup now fires on every multi-batch wake —
  it fired 0 times in 6 h because it only engaged at queue cap).
  `graph_write_pressure` = arrival rows/s ÷ drain capacity rows/s.
- M3.2 shedding past `DECADIC_KUZU_SHED_PRESSURE=1.5` at queue cap: edge
  SETs first, belief SETs past 2×, nodes/creates/deletes never.
- M3.3 measured only: `graph_flusher_cpu_share` (act if still material).
- M4 run_body_diag.ps1: live RED lines (commit_lag > 60 s, pressure ≥ 1.0),
  verdict rows in summary, new keys in the telemetry regex.
  run_long_life.ps1: poller nesting bug fixed (raw-body regex — old property
  access wrote null into EVERY snapshot), pressure/commit-lag trend verdicts.

**M5 next:** 30 min (pressure < 1, edges bounded, event rows plateau) → 2 h
(≥ 4 cyc/s end-to-end, wall flat) → 6 h re-run.
