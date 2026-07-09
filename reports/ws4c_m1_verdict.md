# WS4C M1 Verdict — what actually feeds the flusher

**Evidence base:** offline inspection of the 6 h run's saved kuzu db
(`data/agent_5f294e50-...`, 107 MB, read-only copy) + metrics.json from
`reports/bodydiag_kuzu_20260707_025552` + code reading. M1.1 per-kind
telemetry is now live in `kuzu_graph.py` (flat keys
`graph_writes_staged_<kind>` / `graph_writes_deferred_<kind>` /
`graph_deferred_depth_<kind>`; edges break down by relation kind) and will
give exact splits on the next run.

## Db contents (22,139 cycles)

| table | rows | notes |
|---|---|---|
| Entity | 34 | fine |
| RELATES | 5,438 | 8 kinds, ALL scene_*/co_occurrence; ZERO duplicate (src,dst,kind) — upsert keying is correct |
| PropertyBelief | 816 | fine (~24/node) |
| SemanticRecord | **200,000** | == `DEFAULT_LTM_MAX_SEMANTIC_RECORDS` exactly — the run spent 6 h filling to the cap |

SemanticRecord categories: **event 194,074**, relationship 5,410, entity 387,
correlation/value/conclusion 43 each.

## Findings

1. **The dominant write factory is anonymous event records, not edges.**
   `semantic_graph.py:1033` coins a FRESH id per event
   (`_coin_semantic_id("evt")`), so every cycle's events (~8.8/cycle) create
   brand-new SemanticRecords. New keys always pass the write throttle by
   design, are never coalesced (distinct keys), and are never retired until
   the 200k cap engages retention. Estimated ≈73% of all rows that reached
   kuzu were event CREATEs.

2. **The 460 writes/cycle from the plan are the DEFERRED arrival** (10.1M /
   22,139 cycles = 458/cyc) — scene-edge/node/belief re-touches absorbed by
   the throttle. They cost Python time every cycle but do NOT reach kuzu
   except once per 25-cycle window (`DECADIC_KUZU_WRITE_MIN_CYCLES`).
   Staged arrival ≈12/cyc, ~9 of which are event CREATEs.

3. **Edges are never retired — no code path exists.** `prune_retention`
   mirrors node and sem deletions into kuzu but has no `del_edge` op kind;
   `edge_pruned = 0` is hardwired. Edge age spread confirms accumulation:
   last_cycle min=24, >60% of edges stale by >2,000 cycles. The 5,438 edges
   are a slow leak (bounded only by 34×33×8 kinds ≈ 9k possible keys), not
   the firehose.

4. **`commit_lag_ms` (13.74M ms) is a STAGE-PIPELINE metric** — time since
   the last deep-processed perception session (`stage_pipeline.py:489`), not
   the kuzu flusher queue. The 3.8 h lag means deep processing starved for
   the back half of the run — a victim of the same CPU strangulation, not
   the flusher's queue depth (which was 0 at teardown). M4.3's RED guard
   should watch this metric AND the new `graph_write_pressure`.

5. **Coalesce-dedup fired zero times because the queue never hit its cap
   during normal operation** (`graph_flush_queue_depth` 0 at snapshot;
   coalescing only engages at queue depth ≥4). Not a bug — the backlog
   manifested as flush COST (321 ms/batch at end) and CPU share, not queue
   depth.

## Consequence for M2 scope

The planned edge decay/retention + refresh≠rewrite (M2.1/M2.2) remain
correct and needed — but insufficient alone. The real fix must add
**semantic-record hygiene**: key events by class (aggregate instances like
correlations already do) or give event records aggressive retention. A
34-entity world generating 194k event rows in 6 h is not memory, it is a
log.
