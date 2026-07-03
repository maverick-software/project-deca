# WBS: WS3 — Stage 3→4 Attention Gate (Phase A: heuristic)

**Version:** 1.0 — 2026-07-02 · **Companion PRD:** `ws3_attention_gate_prd.md`
**Convention:** 1 d = one focused dev-day. ⚙ = runs on Charles's machine. Everything else is buildable off-box. The WS2 12-hour soak (D3) can run overnight in parallel with G1–G4.

---

## Phase G — Gate implementation (est. 3 d)

**G1. Gate module** (1 d)
`decadic/cycle/attention_gate.py`: `GateDecision` dataclass (escalate flag, score, per-input contributions), `AttentionGate` class (normalized weighted sum, hysteresis latch, soft budget pressure, fast-path override), config plumbing (`DECADIC_GATE_*` in `config.py`). Pure Python/numpy, no torch — unit-testable anywhere.
*Acceptance:* unit tests for threshold, hysteresis K, budget pressure raising the threshold, fast-path unconditional escalation, determinism.

**G2. Input extraction** (0.5 d)
Wire the five inputs from existing per-cycle values: recall similarity (stage-3 memory correlation), PC-loss EMA vs baseline, pain/drive from element B, priority label from D, trailing escalation rate. No new computation on the hot path — read-only taps.
*Acceptance:* unit test with a synthetic CycleContext showing each input normalized to [0,1]. Depends on G1.

**G3. Pipeline integration + skip path** (1 d)
Insert the gate into the serial pipeline between stage-3 outputs and stage-4 compute (`neural_pipeline.py`). Implement the precedent pass-through: cache last escalated stage-4 output, decay toward neutral with tau. Element-E write (one slot: last gate score). `DECADIC_GATE_ENABLED=0` default — flag off must be byte-identical (existing test-parity discipline).
*Acceptance:* full pytest green with flag off AND on; stage-5 input contract unchanged (trace test); diagnostics carry gate fields. Depends on G1, G2.

**G4. Telemetry + report panel** (0.5 d)
Gate fields into cycle diagnostics → `/metrics` → WS2 sampler (they flow automatically once in metrics; verify). Add gate panel to `generate_run_report.py` section 2 (escalation-rate timeline, score distribution, contribution breakdown). Metric catalog doc updated.
*Acceptance:* fixture-based report test shows the gate panel. Depends on G3.

## Phase P — Probe scenario and offline tuner (est. 1.5 d)

**P1. `gate_probe` scenario + event-injecting client mode** (1 d)
Extend `synthetic_ws_client.py` with `--events <spec>`: inject collision (threat), novel-observation bursts (position jumps), and quiet periods at scripted cycles. New `docs/eval_scenarios/gate_probe.json` with gates: escalation within N cycles of each injected event; trailing rate <5% during quiet periods; zero suppressed fast-path escalations.
*Acceptance:* scenario file loads; client event injection unit-tested. Depends on G3.

**P2. Offline threshold tuner** (0.5 d)
`scripts/tune_gate.py`: replays recorded soak samples (all gate inputs are in the WS2 sample stream) through the gate decision function across a weight/threshold grid; reports achieved escalation rate and event-response latency per config. Burns zero machine sessions.
*Acceptance:* runs against the 12-h soak samples; recommends a config meeting the 5% target. Depends on G1; benefits from the overnight soak data.

## Phase V — Validation (est. 1 d dev + machine time)

**V1. ⚙ `gate_probe` run** (0.25 d + ~20 min machine)
Server with `DECADIC_GATE_ENABLED=1` and tuned config; probe client; eval gates evaluated.
*Acceptance:* all `gate_probe` gates pass. Depends on P1, P2.

**V2. ⚙ A/B soak: gate-on vs gate-off** (0.25 d + 2 × 1 h machine)
Two 1-hour soaks via `run_soak.ps1` (add `-GateOn` switch), compared with WS2 comparison mode.
*Acceptance:* PC-loss trend within tolerance of gate-off; cycle rate equal or better; escalation rate in band; comparison report committed. Depends on V1.

**V3. Wrap-up** (0.5 d)
WS3 findings report (`reports/ws3_gate_report.md`), PRD open decisions resolved, master plan updated, Phase B (learned gate) PRD stub with the regret-signal data pulled from V2 telemetry.
Depends on V2.

---

## Totals and sequencing

Dev effort: **~5.5 focused days**. Machine time: ~20 min probe + 2 × 1 h A/B (plus the WS2 12-h soak running independently overnight).

```
G1 -> G2 -> G3 -> G4 ----\
        \                 +-> V1 -> V2 -> V3
         P1 (needs G3) ---/
G1 -> P2 (needs soak samples)
```

Critical path: G1 → G2 → G3 → P1 → V1 → V2 → V3. Recommended interleave: launch the WS2 12-hour soak tonight; build G1–G2 while it runs; its samples feed P2 directly.

## Explicitly out of scope (queued behind WS3-A)

- **Phase B learned gate** — separate PRD once V2's regret telemetry exists.
- **WS4 memory-backend migration** (SQLite → **LanceDB** for episodic ANN + **Kuzu** for the semantic graph — DECIDED 2026-07-02, Neo4j dropped): both are embedded, pip-installable (`uv pip install lancedb kuzu`), zero server processes or external config — they open a local directory the way SQLite opens a file. Kuzu carries native vector indexes, covering the "vector graph" requirement in one dependency. Entry criterion unchanged: the 12-h soak profile must show retrieval >5% of cycle budget; benchmark-gated, behind the existing `episodic_store.py`/`semantic_graph.py` interfaces. Current data (`memory_recall_ms`/`ltm_match_ms` ≈ 0 at 30k rows) does not yet justify starting it.
- **Baseline reactive agent** (PoC criterion 5) — drops into WS2 comparison mode after WS3-A.
