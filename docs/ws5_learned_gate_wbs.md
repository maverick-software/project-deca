# WBS: WS5 — Learned Attention Gate

**Companion:** `ws5_learned_gate_prd.md` · Ordering is dependency order only.
**Gate for the whole workstream:** M0 acceptance (data foundation) before anything trains; M5 does not start until MuJoCo embodiment exists.

---

## M0 — Data foundation: decision log + shadow deliberation

**M0.1 Per-cycle gate decision log.**
`DECADIC_GATE_LOG` (default off) streams one JSONL row per decision via the write-behind pattern: cycle, 8 GateNet features (4 existing `gate_i_*` + drive pressure, escalation rate, precedent age, latch state), decision, reason, threshold_effective, and State-Bus outcome taps (pain, viability, pc_ema).
*Accept:* rows for ≥99% of decisions on a probe run; overhead <1% cycle time in the harness; flag off = byte-identical (parity test).

**M0.2 Shadow deliberation tap.**
`DECADIC_GATE_SHADOW_RATE` (default 0.05): on sampled skips, compute fresh `risk_mlp(z3)` beside the substituted precedent (no autograd, diagnostics only); on sampled escalations, compute the precedent's counterfactual. Emit `shadow_regret_risk`, `shadow_regret_z4`, `shadow_waste` into the decision log.
*Accept:* divergences present at the configured rate; zero effect on live z4/risk outputs (regression test: forward outputs identical with shadow on/off); no cycle-rate regression at 5%.

**M0.3 Data harvest.**
Re-run the redesigned gate probe and one 1-h soak with `GATE_LOG=1, SHADOW_RATE=0.05` (heuristic mode) to bank the first training corpora.
*Accept:* ≥2 runs of logs on disk; log-size sanity (rotation if needed).

## M1 — Dataset builder + labels

**M1.1 `scripts/build_gate_dataset.py`.**
Joins decision rows to forward outcome windows (horizon H, default 50 cycles): pain/viability/PE deltas. Emits train-ready parquet/npz + a stats report (class balance, regret distribution, precedent-age conditioning — the label-circularity check from PRD §5).
*Accept:* deterministic rebuild from the same logs; stats report renders; held-out split is BY RUN.

**M1.2 Label policy.**
`should_escalate = σ(α·(regret − cost))`, outcome-sharpened per PRD §3.2; α, cost, H are CLI hyperparameters recorded in the dataset manifest.
*Accept:* label flips respond to α/cost as documented; sanity task — a synthetic log with planted regret structure yields the planted labels.

## M2 — GateNet + offline training + replay evaluation

**M2.1 `GateNet` module** (`decadic/cycle/gate_net.py`): ~8→16→1 MLP, sigmoid; versioned state_dict key for the brain bundle; loader tolerates absence (old brains = heuristic).
*Accept:* unit tests — shape, determinism, save/load round-trip through the neural bundle.

**M2.2 `scripts/train_gate.py`.**
BCE on M1 labels; run-level split; emits weights + training report (calibration curve, regret-vs-rate frontier against the tuned heuristic).
*Accept:* on the planted-structure sanity dataset, learns to >0.95 AUC; on real logs, report generated (dominance not required here — that is M2.3's call on real signal).

**M2.3 Replay evaluator** (`tune_gate.py --policy gatenet <weights.pt>`).
Scores learned vs heuristic on recorded streams: threat recall, novelty-burst count contract, calm rate, escalation rate, total regret.
*Accept:* evaluator runs both policies on the same logs and prints a comparison verdict; PRD success criterion 2 evaluated and the outcome (dominate / match / insufficient-signal) recorded in the training report. Insufficient-signal on synthetic data is an acceptable, documented outcome (PRD risk G5) — it gates promotion, not the workstream.

## M3 — Runtime integration: shadow mode

**M3.1 `DECADIC_GATE_MODE=shadow`.**
GateNet computes per cycle; heuristic decides; agreement + logit land in the decision log and telemetry (`gate_l_logit`, `gate_l_agree`).
*Accept:* behavior byte-identical to heuristic mode (parity test on recorded cycle stream); agreement telemetry visible in the dashboard/harness.

**M3.2 Rails pinning.**
Tests asserting: fast path escalates regardless of GateNet output; budget raises the learned bar exactly as it raises the heuristic one; hysteresis latches identically; `GATE_ENABLED=0` byte-parity unchanged.
*Accept:* all rail tests green in all three modes.

## M4 — Live learned mode + A/B evidence + persistence

**M4.1 `DECADIC_GATE_MODE=learned`** behind the rails, `gate_reason="net"`.
*Accept:* redesigned gate probe passes in learned mode (PRD criterion 3).

**M4.2 A/B soak** — 1 h learned vs 1 h heuristic, same config.
*Accept:* PRD criterion 4 (no stability regression; escalation in band; PC-loss endpoint within noise).

**M4.3 Persistence + routes.**
GateNet rides `save_brain`/`load_brain`, `/checkpoint`→`/restore`, and the Saved-Agents round-trip; manifest gains `gate_mode` + training-run id.
*Accept:* route-level round-trip test (extend `test_ws4_checkpoint_routes.py` pattern) restores identical GateNet weights and mode.

## M5 — Embodied refinement (MuJoCo-gated; do not start before embodiment)

**M5.1** Re-harvest decision logs in MuJoCo scenes where attention has bodily consequences; rebuild dataset; retrain.
**M5.2** Embodied replay + probe equivalents; PRD criterion 6 (measurable selectivity the heuristic lacks).
**M5.3** Online slow-adaptation experiment (EMA updates, tight rate cap, shadow-verified) — the first time weights move during a live run; separate go/no-go.
**M5.4** Default-mode decision (heuristic vs learned) — owner call on the embodied evidence, recorded here.

## Cross-cutting

- **No timelines anywhere; dependencies only:** M0 → M1 → M2 → M3 → M4 → (MuJoCo) → M5. M0.3 harvest can run concurrently with M1 scaffolding once M0.1/M0.2 are accepted.
- **Docs:** training reports land in `reports/gate_training_<stamp>/`; every promotion decision (shadow→learned, learned-as-default) gets a dated addendum in the PRD.
- **Standing WS3 Phase B siblings** (EMA-stabilized percept keys, PE normalization vs trailing baseline, curiosity damping) remain separate items — they improve GateNet's *inputs* and slot in before M5 retraining if landed.
