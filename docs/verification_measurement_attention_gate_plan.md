# Plan: Neural-Path Verification, PoC Measurement Harness, Stage 3→4 Attention Gate

**Version:** 1.0 — 2026-07-02
**Status:** Draft for review
**Scope:** Items 2–4 from the July 2026 project review. Excludes the baseline comparison agent (item 5) and the Neo4j / audio-codec decisions (deliberately deferred).

Settled decisions are stated declaratively. Estimates and open decisions are marked as such.

---

## Workstream 1 — Verify neural-path learning post-bf16 fix

### Why

Commit `b83a0d4` fixed a bf16 autocast issue in `decadic/cycle/neural_pipeline.py` (~lines 683–695: bf16 tensors normalized back to fp32 after the forward pass) that blocked NumPy extraction. Phase 2's core claim — prediction error decreases over cycles with `DECADIC_NEURAL_PRESET=full` — has not been re-verified since. The June 23 `health_smoke` eval "pass" is hollow: it attached to a stalled agent and collected a single sample (slope 0.0). Everything downstream (measurement harness, gate training) assumes this claim holds.

### Tasks (implemented 2026-07-02 — run via `.\scripts\run_ws1.ps1`)

1. **Full CPU test pass.** Full pytest suite with JUnit XML output into the run directory.
2. **Neural smoke test.** `health_smoke` eval scenario against a fresh agent driven by the synthetic WebSocket client (agents do not cycle without an observation stream — root cause of the hollow June 23 run). `--agent-id` attach mode added to `run_training_eval.py` and `synthetic_ws_client.py` for this.
3. **Short learning run.** New `docs/eval_scenarios/ws1_learning_run.json`: 3,000 cycles, gates on cycle advancement (delta ≥ 3000), PC-loss decrease (delta < 0), and no sustained divergence.
4. **Trend check.** `scripts/plot_pc_trend.py`: CSV export, half-mean comparison, least-squares slope, optional PNG. The real tooling comes in Workstream 2.

### Exit criteria

- Test suite passes (or failures triaged with filed issues).
- `neural_pc_loss` shows a downward trend over the run — not monotone, but a negative slope over the full window.
- No NaN/Inf, no dtype regressions, memory stable across the run.
- Cycle rate recorded (reference point for the 10+ Hz stages-1–3 benchmark target).

**Estimate:** 1–2 days. Blocked by nothing; blocks both other workstreams.

---

## Workstream 2 — Measurement harness for the PoC success criteria

### Why

The five PoC success criteria (stability, visible learning, coherent state evolution, working memory/consolidation, distinctness from baseline) currently have no evidence pipeline. Telemetry exists but is write-only: `decadic/logging/json_logging.py` writes JSONL to stderr and a rotating file; `/metrics` returns point-in-time snapshots (viability, queue depth, preset); `decadic/metrics/integration.py` serves PCI probes, not live monitoring. There is no time-series aggregation, no PE-trend export, no cycle-rate counters, no rollups. Without this, the system produces behavior but not results.

### Design (settled)

Three components, built on the existing JSONL stream rather than a new telemetry path — rationale: the logging layer already captures per-cycle records; aggregating downstream avoids touching the hot cycle loop.

1. **Metrics aggregator** (`decadic/metrics/harness.py`). Consumes JSONL log files, produces per-minute and per-hour rollups, exports CSV/Parquet. Fields per cycle: `cycle_index`, wall-clock rate, `neural_pc_loss` (total and per-stage where available), viability + the three reservoirs, pain/pleasure scalars, priority label, norms and drift of elements A/C/E, episodic store size, consolidation events, and (once Workstream 3 lands) `gate_escalation_rate`.
2. **Soak runner** (`scripts/soak_run.py`). Launches server + synthetic or MuJoCo client for a configured duration (target: 12 h, then multi-day), monitors for crashes/NaN/memory growth, writes a run manifest (config, git SHA, encoder mode, duration).
3. **Report generator** (`scripts/generate_run_report.py`). Renders a markdown report into `reports/` per run, sectioned by the five PoC criteria: uptime/stability stats, PE curves per stage, A–F trace plots and drift analysis, memory growth + recall-hit stats, consolidation on/off deltas. Criterion 5 (distinctness) gets a placeholder section until the baseline agent exists — the harness must support two-run comparison from day one so the baseline drops in later.

### Estimates and open decisions

- Rollup cadence and retention: per-minute for soak runs, per-hour beyond 24 h (estimate — revisit after first 12 h run).
- Whether consolidation on/off comparison runs sequentially or as parallel agents (open; sequential is simpler and sufficient for v1).
- Plotting: matplotlib static PNGs embedded in the markdown report for v1; dashboard integration deferred.

### Exit criteria

A single command runs a 12-hour soak and produces a report in `reports/` with PE curves, A–F traces, viability history, and memory stats — no manual steps.

**Estimate:** 1–2 weeks. Depends on Workstream 1.

---

## Workstream 3 — Stage 3→4 attention gate

### Why

The gate is the most architecturally novel and publishable component, with a <5% escalation target. **It does not currently exist.** `stage_03.py` and `stage_04.py` are numpy stubs (latent seed + nudge); no gating, escalation, or arbitration mechanism exists anywhere in the codebase. This is new design and build, not instrumentation.

### Design (proposed — review before build)

The gate sits between stage 3 (heuristic assessment & memory correlation) and stage 4 (risk-utility evaluation & investigative examination). It decides, per cycle, whether stage 4 runs its full deliberative path or a cheap default pass-through.

**Gate inputs (from stage 3 outputs and the State Bus):**

- Memory-correlation salience: similarity of current situation to stored episodes (high similarity → precedent exists → skip deliberation).
- Novelty: inverse of best episodic recall similarity.
- Current prediction error magnitude (from the predictive-coding losses).
- Threat/affect signal: pain scalar and drive pressure from element B (fast-path signals always escalate).
- Current priority label from element D.

**Gate output:** binary escalate/skip plus a scalar confidence, both logged per cycle.

**Two-phase build (settled):**

- **Phase A — heuristic gate.** Hand-set thresholds over the inputs above. Rationale: establishes the interface, the telemetry, and a baseline escalation-rate distribution before any learning is involved. Emits `gate_escalation_rate` into the Workstream 2 harness.
- **Phase B — learned gate.** Small MLP (est. <100K params) trained online. Training signal (open decision, the hard problem): candidate is regret-based — escalations that changed the stage 8 strategy output relative to the default path were worthwhile; escalations that didn't are wasted compute. Skipped cycles followed by high prediction error are missed escalations.

### Open decisions

- Whether the skip path bypasses stage 4 entirely or runs a frozen cheap approximation (affects downstream stage 5 input contract).
- Escalation budget enforcement: soft (learned) vs hard cap per time window.
- Whether gate decisions write to element E (metacognition) — theoretically motivated, adds coupling.

### Exit criteria

- Phase A: gate operational in the neural pipeline, escalation rate observable in soak reports, rate tunable toward <5%.
- Phase B: learned gate matches or beats heuristic on the regret metric at equal escalation budget.

**Estimate:** Phase A ~1 week; Phase B 2–3 weeks (higher uncertainty — training-signal design is genuinely open). Phase A depends only on Workstream 1; Phase B benefits from Workstream 2 telemetry.

---

## Sequencing

```
WS1 (verify)  ──►  WS2 (harness)  ──►  first 12h soak + report
      │
      └───────►  WS3-A (heuristic gate)  ──►  WS3-B (learned gate)
```

WS2 and WS3-A can run in parallel after WS1. First integrated milestone: **a 12-hour soak run whose report shows PE trending down, coherent A–F traces, and a measured gate escalation rate** — that is criteria 1–3 evidenced and the flagship component alive.

## Out of scope (recorded for later)

- Baseline reactive agent (PoC criterion 5) — next after WS2, harness already supports two-run comparison.
- Memory-backend migration — DECIDED 2026-07-02: SQLite → LanceDB (episodic ANN) + Kuzu (semantic graph with native vector indexes); Neo4j dropped. Both embedded and pip-installable, no server processes. Gated on the 12-h soak profile showing retrieval >5% of cycle budget (WS4; see `ws3_attention_gate_wbs.md` out-of-scope section). Prosody-preserving audio codec / YAMNet swap — still deferred; no success criterion depends on it.
- `.gitattributes` line-ending normalization — trivial, do alongside WS1.

---

## Appendix (2026-07-04): workstream ledger update + WS5 Relational Binding

**Status of the original workstreams:** WS1 closed (learning verified, stability root-caused). WS2 shipped (harness/soak/report; 12-h soak still pending). WS3 Phase A closed and the redesigned gate probe now PASSES in full (threat 1/1, novelty 2/2 unique first exposures, revisit correctly quiet, calm 0.000 — see `reports/ws3_gate_report.md` addendum). WS4 closed (lancedb+kuzu defaults, full-mirror L1, 0.84 ms full-corpus recall, soak A/B exonerating and preferring the new stack). **WS3-B (learned gate)** — `ws3b_learned_gate_{prd,wbs}.md` — has M0–M2 built (decision log, shadow-deliberation regret labels, GateNet + offline trainer); its live modes (M3+) are deliberately parked, see sequencing below.

### WS5 — Relational Binding (next major workstream)

**Docs:** `ws5_relational_binding_prd.md` · `ws5_relational_binding_wbs.md` (2026-07-04 architecture review).

**The gap in one sentence:** Deca can perceive, remember, attend, and feel, but "the wolf is behind the rock" and "the rock is behind the wolf" are the same state to the network — every structured input is pooled into a single vector before cognition sees it. This is the variable-binding gap; it is an architectural mechanism, not a capacity problem, and no scale fixes it.

**The audit's finding:** the symbolic half already exists (WM entity slots, object files, Kuzu entity identity). The differentiable half dies at three pooling chokepoints: scene→stack (EMA-pooled scene latent), memory→stack (mean-pooled top-k recall), stage 3→4 (risk from one fused vector). WS5 carries the slots across that boundary intact: slot tensor, keyed cross-attention reads, one small relational transformer at stage 4, slot keys from the graph (object permanence made visible to the network).

**Falsifiable by construction:** the binding probe trains relations over familiar entities and tests never-seen pairings — flags-off must fail (it structurally cannot represent the pairing), flags-on must pass. Built-in ablation, same discipline as the WS3 probe.

**Sequencing (dependency order, per the two PRDs):**
1. WS5 begins now that the WS3 gate upgrade is in place (probe PASS, percept-source novelty validated, decision-log channel accruing data in every gate-enabled run).
2. WS3-B M3+ (shadow/live learned gate) waits for WS5: the relational core changes what stage-4 escalation buys, so the gate learns against the final cost structure (`ws5_relational_binding_prd.md` §4).
3. MuJoCo embodiment then carries three re-validations at once: calm certification, novelty/binding probes on embodied percepts, and WS3-B M5 outcome-grounded retraining.

Standing items unchanged: WS2 12-h soak · baseline reactive agent (PoC criterion 5) · WS3 Phase-B input fixes (EMA percept keys, PE normalization) fold into WS5/WS3-B where noted.
