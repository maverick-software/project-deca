# WBS: WS2 — Measurement Harness

**Version:** 1.0 — 2026-07-02 · **Companion PRD:** `ws2_measurement_harness_prd.md`
**Estimating convention:** 1 d = one focused dev-day. Estimates assume the WS1 tooling patterns (process lifecycle, watchdog, REST probing) are reused, not reinvented. Items marked ⚙ run on Charles's machine; everything else is buildable off-box.

---

## Phase A — Sampler and aggregator (est. 2 d)

**A1. Metric catalog freeze** (0.25 d)
Enumerate the exact `/metrics` and `/state` keys against a live server response (one ⚙ probe, or reuse `reports/stallhunt_*/stall_metrics.json`). Output: `docs/ws2_metric_catalog.md` table — name, source endpoint, type, criterion it serves.
*Acceptance:* every metric in PRD §5.1 mapped to a real key or explicitly deferred.

**A2. Sampler** (0.75 d)
`decadic/metrics/harness.py`: async poller (interval configurable, default 2 s) → `harness_samples.jsonl`, one flat JSON object per poll (timestamp, cycle, all catalog keys, `nvidia-smi` VRAM/util, process RSS, run-dir disk usage). Graceful on missing keys and transient HTTP errors (log, skip, never crash).
*Acceptance:* unit test with a stubbed HTTP layer; 5-minute live probe shows well-formed lines. Depends on A1.

**A3. Rollups + export** (0.5 d)
Per-minute (always) and per-hour (runs >2 h) aggregation: mean/min/max/last per numeric key, event-counter deltas, label distributions. Output `rollup_1m.csv`, `rollup_1h.csv`.
*Acceptance:* unit test on synthetic samples fixture; derived cycle-rate column matches hand computation. Depends on A2.

**A4. Overhead measurement hook** (0.5 d)
Sampler on/off cycle-rate comparison mode (`--overhead-check`, two short back-to-back windows).
*Acceptance:* ⚙ shakedown shows <1% delta (PRD §7.2). Depends on A2.

## Phase B — Soak runner (est. 2 d)

**B1. Lifecycle core** (1 d)
`scripts/soak_run.py`: start server with env manifest (git SHA, full `DECADIC_*` snapshot, preset, encoder), readiness wait, create agent, start synthetic client (10 obs/s), start sampler, run `--hours N`, ordered teardown (client → agent DELETE → sampler flush → server), write `soak_summary.json`. PowerShell wrapper `scripts/run_soak.ps1` with visible progress console (WS1 monitor pattern).
*Acceptance:* ⚙ 10-minute run produces complete run dir with manifest, samples, rollups, summary; all processes gone afterward. Depends on A2.

**B2. Stall watchdog + guards** (0.5 d)
Port of `stall_hunt` logic into the runner: 60 s cycle freeze → auto-capture `/debug/tasks` + metrics → policy `abort` (default) or `record-and-continue`. Disk guard (<5 GB free or run dir > cap → clean abort). Hourly checkpoint call if enabled (PRD §9).
*Acceptance:* unit test for policy logic; simulated stall (pause agent via `POST /agent/{id}/pause`) triggers capture and clean abort. Depends on B1.

**B3. Soak gates** (0.5 d)
Gate evaluation over rollups (PRD §5.4 thresholds, config-driven): stalls, cycle-rate floor, NaN, RSS/disk growth, PC-loss slope + half-means, growth_events observation.
*Acceptance:* unit tests with passing and failing fixtures. Depends on A3.

## Phase C — Report generator (est. 1.5 d)

**C1. Single-run report** (1 d)
`scripts/generate_run_report.py`: run dir → `report.md` + PNGs. Five criterion sections per PRD §5.3, verdict block from B3 gate results, matplotlib added to `dev` extra.
*Acceptance:* golden-file test on a fixture run dir; report renders correctly from a real ⚙ shakedown run. Depends on A3, B3.

**C2. Two-run comparison mode** (0.5 d)
`--compare <run_a> <run_b>`: side-by-side deltas, overlaid PC-loss curves, consolidation on/off framing, simple effect sizes. Populates criterion 5 section.
*Acceptance:* comparison report from two shakedown runs reads coherently. Depends on C1.

## Phase D — Validation and the 12-hour soak (est. 1 d dev + machine time)

**D1. ⚙ 1-hour shakedown soak** (0.25 d + 1 h machine)
Full pipeline end-to-end. Calibrate gate thresholds, poll interval, checkpoint cost; run A4 overhead check. Fix fallout.
*Acceptance:* report generated with zero manual steps; open decisions from PRD §9 resolved and recorded in the PRD. Depends on B*, C1.

**D2. ⚙ Second shakedown (consolidation off) + comparison** (0.25 d + 1 h machine)
`DECADIC_CONSOLIDATION_ENABLED=0` run; exercise C2 for real.
*Acceptance:* comparison report committed as the workflow example. Depends on D1, C2.

**D3. ⚙ 12-hour soak** (0.25 d + 12 h machine, overnight)
The real thing, growth enabled, watchdog on `abort`… policy per D1 decision.
*Acceptance:* soak gates evaluated; report committed to `reports/`; PoC criterion 1 evidenced at hours-scale; `growth_events` question (WS1 caveat) answered.
Depends on D1.

**D4. Wrap-up** (0.25 d)
Update `ws1_verification_report.md` cross-references, the WS2 PRD open-decision log, and `docs/verification_measurement_attention_gate_plan.md` status; note WS3-A telemetry hooks (gate escalation rate) as a one-line sampler addition when WS3 lands.
Depends on D3.

---

## Totals and sequencing

Dev effort: **~6.5 focused days** (estimate; WS1 experience suggests the risk is in shakedown fallout, buffered in D1). Machine time: ~2.5 h shakedowns + one overnight 12-hour soak.

```
A1 -> A2 -> A3 -> B3 ----\
        \-> A4            +-> C1 -> C2 -> D2
A2 -> B1 -> B2 ----------/    |
                              D1 -> D3 -> D4
```

Critical path: A1 → A2 → A3 → B3 → C1 → D1 → D3 → D4. Phases A–C need no machine access; ⚙ items batch into three sessions (shakedown ×2, overnight soak).
