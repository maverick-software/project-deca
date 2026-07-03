# PRD: WS3 — Stage 3→4 Attention Gate

**Version:** 1.0 — 2026-07-02
**Status:** Draft for review
**Depends on:** WS1 (closed), WS2 (harness built; 12-h soak pending — runs in parallel with WS3-A)
**Companion:** `ws3_attention_gate_wbs.md`

Settled decisions are stated declaratively; estimates and open decisions are marked.

---

## 1. Problem

The Decadic Cycle currently runs every observation through the full stage-4 path (risk-utility evaluation, curiosity arbitration, investigative examination) on every cycle. Real cognition doesn't: most moments are handled by precedent, and deliberation is reserved for the novel, risky, or surprising. The stage 3→4 attention gate — deciding per cycle whether deep deliberation is warranted — is the architecture's most novel, publishable component, and it does not exist in code (verified 2026-07-02: `stage_03.py`/`stage_04.py` are stubs; no gating mechanism anywhere).

## 2. Goal

A gate between stages 3 and 4 that (a) escalates <5% of cycles to the full deliberative path in steady state, (b) escalates reliably on genuinely novel/threatening/surprising input, and (c) produces telemetry proving both — flowing into the WS2 harness and reports.

## 3. Non-goals

- No learned gate in this workstream's first phase (heuristic first — see phasing).
- No modification to stage 4's internals; the gate wraps, not rewrites.
- No language/narrative involvement (element C stays deferred).

## 4. Theory alignment (settled)

The gate implements the framework's escalation from heuristic assessment (stage 3) to Risk-Utility Evaluation, Curiosity Trigger & Investigative Examination (stage 4). Gate decisions write to element E (metacognition) — the system knows *that* it deliberated — and the escalation rate itself is a metacognitive signal. (Element-E write was an open decision in the master plan; settled yes: it is theoretically motivated and one vector slot.)

## 5. Design

### 5.1 Placement and contract (settled)
The gate runs inside the serial pipeline after stage-3 outputs are available, before stage-4 compute. Two paths:

- **Escalate:** full stage-4 path, exactly as today.
- **Skip:** stage 4 emits a *precedent pass-through* — the cached stage-4 output structure from the last escalated cycle, decayed toward neutral (estimate: exponential decay toward zero risk-delta with configurable tau). Downstream stages 5–10 see a well-formed stage-4 output either way, so their contract is untouched. This resolves the master plan's open decision: skip does NOT bypass stage 4's output slot, it bypasses stage 4's *compute*.

### 5.2 Gate inputs (settled list, weights are Phase-A tunables)
From stage 3 and the State Bus, all already computed per cycle (no new compute on the hot path):

1. **Novelty** — 1 minus best episodic recall similarity (stage-3 memory correlation already retrieves this).
2. **Prediction error** — current `neural_pc_loss` EMA vs its baseline (surprise proxy).
3. **Threat/affect** — pain scalar and drive pressure from element B; fast-path damage events force escalation unconditionally (hard rule, not weighted).
4. **Priority label** — element D in `investigate` biases toward escalation.
5. **Budget pressure** — recent escalation rate vs the 5% target (soft budget; see 5.4).

### 5.3 Decision function
- **Phase A (this WS):** weighted sum of normalized inputs vs threshold, with hysteresis (escalation latches for K cycles, estimate K=3, to avoid flapping). Hand-set weights, all configurable via `DECADIC_GATE_*` env vars. Deterministic given inputs — reproducible runs.
- **Phase B (separate WS, design sketched here):** small MLP (<100K params) trained online with a regret signal — escalations whose stage-4 output materially changed the stage-8 strategy relative to the precedent pass-through were worthwhile; skips followed by prediction-error spikes were missed escalations. Phase B is gated on Phase A telemetry showing the regret signal is computable and non-degenerate.

### 5.4 Budget (settled)
Soft budget: the budget-pressure input raises the threshold as the trailing escalation rate exceeds 5%. No hard cap — a threat storm may legitimately escalate every cycle, and safety-relevant escalations (fast-path) are never suppressed.

### 5.5 Telemetry (settled)
Per cycle into diagnostics (already flows to `/metrics` and the WS2 sampler): `gate_escalated` (0/1), `gate_score`, per-input contributions, `gate_escalation_rate` (trailing 1000-cycle window), `gate_skip_streak`. One-line additions to the WS2 metric catalog; the report generator gets a gate panel in section 2.

### 5.6 Config
`DECADIC_GATE_ENABLED` (default 0 until validated — byte-identical baseline preserved), `DECADIC_GATE_THRESHOLD`, `DECADIC_GATE_TARGET_RATE`, `DECADIC_GATE_HYSTERESIS_K`, `DECADIC_GATE_WEIGHTS` (comma list).

## 6. Validation design (settled — uses WS2 machinery)

1. **Unit tests:** decision function (thresholds, hysteresis, fast-path override, budget pressure), pass-through decay, element-E write.
2. **Scenario `gate_probe`:** synthetic client script with injected novelty/threat events at known cycles; gates assert escalation fires within N cycles of each event and rate stays <5% between events.
3. **A/B soak (1 h × 2, comparison mode):** gate-on vs gate-off. Acceptance: PC-loss trend not degraded (second-half mean within 10% of gate-off run — estimate, calibrate), cycle rate improved or equal (skipping stage 4 should be cheaper), escalation rate in band.

## 7. Success criteria

1. Gate operational behind a flag; full test suite still green with the flag off (byte-identical) and on.
2. `gate_probe` scenario passes: reactive escalation on events, <5% steady-state rate.
3. A/B comparison report shows no learning regression and measurable per-cycle compute savings.
4. Telemetry visible in soak reports; the flagship component is now measurable, hence publishable.

## 8. Risks

- **Skip path starves learning** — stage 4 losses stop flowing on skipped cycles; if PC trend degrades, consider always running stage 4 in no-grad "shadow" mode every Mth skip (fallback design, estimate M=10).
- **Degenerate inputs on synthetic client** — novelty may saturate (repetitive synthetic observations). Mitigate in `gate_probe` by scripted event injection; final calibration belongs to MuJoCo embodiment.
- **Threshold tuning burns machine sessions** — mitigate with a replay-based tuner: gate decisions recomputed offline over recorded soak samples (inputs are all in the sample stream) before any live run.

## 9. Open decisions

- Precedent pass-through decay tau (start 20 cycles; calibrate in gate_probe).
- Whether skipped cycles still write full episodic records or lightweight ones (leaning full — memory criteria depend on it).
- Phase B regret-signal exact form (deferred to Phase B PRD).
