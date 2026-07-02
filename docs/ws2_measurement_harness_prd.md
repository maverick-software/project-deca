# PRD: WS2 — Measurement Harness

**Version:** 1.0 — 2026-07-02
**Status:** Draft for review
**Depends on:** WS1 (closed 2026-07-02 — learning verified, fold-drain stall fixed, growth exonerated)
**Companion:** `ws2_measurement_harness_wbs.md`

Settled decisions are stated declaratively; estimates and open decisions are marked.

---

## 1. Problem

The five PoC success criteria (stability, visible learning, coherent state evolution, working memory/consolidation, distinctness from baseline) have no evidence pipeline. Telemetry exists but is write-only and ephemeral: `decadic_server.jsonl` rotates at 8 MB × 4 backups (~32 MB — a 12-hour run overflows it), `/agent/{id}/metrics` is point-in-time, and eval samples exist only for short gated scenarios. WS1 produced results by hand-assembling artifacts across five runs. That does not scale to soaks, ablations, or publication.

## 2. Goal

One command runs an arbitrary-duration agent session and produces, with zero manual steps, a run directory containing raw samples, rollups, and a markdown report that scores the run against the five PoC criteria. A second command compares two runs (the future baseline-agent and ablation workflows drop into this).

## 3. Non-goals

- No baseline reactive agent (that is the next workstream; the harness only needs two-run comparison support).
- No dashboard integration in v1 (static plots in the report; the React dashboard remains live-view only).
- No cloud/multi-machine support; single workstation.
- No new cognition-side instrumentation beyond cheap counters — the harness must not touch the hot cycle loop.

## 4. Users and primary workflow

Charles, running research sessions on the dev box (i7-12700K / 64 GB / RTX 3080 10 GB). Workflow: launch soak (PowerShell wrapper, visible progress console, self-terminating) → machine works unattended → read `reports/<run>/report.md` → commit or discard.

## 5. Architecture (settled)

Three components, sampling via REST polling rather than log scraping — rationale: WS1 showed the REST surface is reliable under load and stall (it stayed responsive during every freeze), while the JSONL file rotates and buffers.

### 5.1 Sampler + aggregator (`decadic/metrics/harness.py`)
- Polls `GET /agent/{id}/metrics` and `GET /agent/{id}/state` at a configurable interval (default 2 s; estimate — revisit after shakedown) and appends one JSON line per poll to `<run_dir>/harness_samples.jsonl`.
- Sampling overhead budget: <1% of cycle-loop throughput (verify in shakedown by comparing cycle rate with sampler on/off).
- Rollups: per-minute always; per-hour additionally for runs >2 h. Output CSV (`rollup_1m.csv`, `rollup_1h.csv`). Parquet deferred.
- Metric catalog v1 (from the live `/metrics` surface, verified during WS1):
  - **Stability:** `cycles_completed`, cycle rate (derived), `stage_timing_ms_total`, `frames_received/prefetched/folded/committed/dropped`, queue depths, `stage_pipeline_active_sessions`, process RSS + VRAM (via `nvidia-smi` poll), disk usage of run dir, `loss_canary_state`, `nan_recovery_events`.
  - **Learning:** `neural_pc_loss_last`, `loss_total`, per-head losses where exposed (`forward_model_error`, `intero_pred_error`, `tactile_pred_error`, `effort_pred_error`, `consolidator_loss`), `loss_dominant_fraction`, `plasticity_*` (alpha, freeze/thaw counts), `rewire_events`, `growth_events`.
  - **State coherence:** A–F snapshot norms and inter-poll drift (from `/state`), `priority_label` distribution, pain/pleasure scalars, viability + hydration/energy/integrity.
  - **Memory/consolidation:** episodic store row count and DB file size, LTM graph node/edge counts, `ltm_consolidation_*`, replay/consolidation event counters, recall-cache hit stats where exposed.
  - **Distinctness:** everything above, keyed by run id — comparison happens in the report layer.

### 5.2 Soak runner (`scripts/soak_run.py` + `scripts/run_soak.ps1`)
- Owns the full lifecycle: start server (env manifest recorded: git SHA, all `DECADIC_*` vars, preset, encoder mode), create agent, start observation client, start sampler, run for `--hours N`, tear everything down. Reuses the WS1 process patterns.
- **Stall watchdog built in** (port of `stall_hunt.ps1` logic): if `cycles_completed` freezes for 60 s, auto-capture `GET /debug/tasks` + metrics into the run dir, then apply the configured policy — `abort` (default) or `record-and-continue` (open decision whether continue is ever meaningful; default abort).
- Disk guard: abort cleanly if free disk < 5 GB or run dir > configurable cap (default 20 GB).
- Observation source v1: synthetic client at 10 obs/s (settled for v1); MuJoCo adapter as source is a v1.1 flag, not a blocker.
- Log rotation: `DECADIC_LOG_DIR` points into the run dir; raise `RotatingFileHandler` caps for soak runs via env (estimate: 64 MB × 8) or accept rotation since the harness samples independently — **open decision, default: accept rotation** (harness_samples.jsonl is the source of truth).

### 5.3 Report generator (`scripts/generate_run_report.py`)
- Input: one run dir (or two, for comparison mode). Output: `report.md` + PNG plots in the run dir.
- Sections map 1:1 to the PoC criteria:
  1. **Stability** — uptime, total cycles, cycle-rate timeline, stall/canary/NaN events, memory & VRAM growth, queue-depth timeline.
  2. **Learning** — PC-loss curve (raw + rolling mean) with lsq slope and half-means (reuse `plot_pc_trend.py` internals), per-head loss curves, plasticity/growth event timeline.
  3. **State coherence** — A–F norm traces, drift stats, priority-label timeline, viability/reservoir traces, pain/pleasure histogram.
  4. **Memory/consolidation** — store growth curves, consolidation event rate, recall stats; consolidation on/off delta when comparing two runs.
  5. **Distinctness** — populated only in comparison mode; side-by-side deltas of the above with simple effect sizes. Placeholder text otherwise.
- Plotting: matplotlib, added to the `dev` extra (it is not currently installed — settled: add it; the report is the point of the workstream).
- Verdict block at top: PASS/FAIL per criterion against soak gates (see 5.4), plus the standing caveat that the dominant-loss canary misfires on synthetic input.

### 5.4 Soak scenario + gates (`docs/eval_scenarios/soak_12h.json` or harness-native gates)
Gates for a 12-hour soak (thresholds are estimates; calibrate in the 1-hour shakedown):
- zero stall-watchdog triggers;
- cycle rate: hourly mean never below 50% of the first-hour mean;
- `nan_recovery_events` = 0 and zero non-finite sampled values;
- process RSS growth < 20%/12 h after warmup; run-dir disk growth bounded;
- PC loss: negative slope over the full window and second-half mean < first-half mean;
- ≥1 `growth_events` observed OR explicit note that the pc-loss threshold was never crossed (closes the WS1 growth caveat).

## 6. Deliverables

`decadic/metrics/harness.py`, `scripts/soak_run.py`, `scripts/run_soak.ps1`, `scripts/generate_run_report.py`, soak gate config, unit tests (sampler rollups, gate evaluation, report generation on a fixture run dir), and a completed 12-hour soak report in `reports/`.

## 7. Success criteria for WS2

1. `.\scripts\run_soak.ps1 -Hours 12` completes unattended and yields `report.md` with all five sections populated and plots rendered — zero manual steps.
2. Sampler overhead <1% measured.
3. The 12-hour soak itself passes the stability gates (this is PoC criterion 1 evidence, hours-scale).
4. Comparison mode produces a coherent two-run report from two shakedown runs.

## 8. Risks and mitigations

- **12 h is a long feedback loop** → 1-hour shakedown soak is a mandatory WBS gate before the 12-hour run; all gate thresholds calibrated there.
- **Episodic store growth over 12 h** (SQLite on D:) → disk guard + store-size metric; pruning behavior is observed, not modified, in WS2.
- **GPU thermals/clock drift on a desktop 3080** → cycle-rate gate uses hourly means, not instantaneous minima; report notes ambient variance.
- **Sampler perturbing the system under test** → fixed overhead budget + on/off A/B in shakedown.
- **Stall recurrence at hours-scale** → watchdog auto-captures `/debug/tasks`; the WS1 diagnostic loop is now cheap to rerun.

## 9. Open decisions (to resolve by end of shakedown)

- Poll interval 2 s vs adaptive (coarser after first hour).
- Stall policy `abort` vs `record-and-continue` for research soaks.
- Log-rotation caps raise vs accept rotation (default: accept).
- Whether the soak also periodically checkpoints agent weights (`POST /agent/{id}/checkpoint`) for post-hoc analysis — leaning yes, hourly, if checkpoint cost is <5 s.
