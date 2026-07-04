# PRD: WS3B — Learned Attention Gate (WS3 Phase B, core item)

**Version:** 1.0 — 2026-07-04
**Status:** Draft for review
**Predecessors:** WS3 Phase A (heuristic gate: threat reflex 100%, budget converges 44%→6.8%, calm certified on synthetic) · WS4 (full-corpus percept recall, the novelty channel's substrate) · probe redesign 2026-07-04 (memory-honest criteria: unique-target novelty, revisit-must-not-spike, peak telemetry).
**Companion:** `ws3b_learned_gate_wbs.md`
**Sequencing note (updated 2026-07-04, post relational-binding audit):** ordering below is dependency order only. M0–M2 (data channel, labels, GateNet + offline training) are infrastructure and ran immediately — the decision log accrues corpus in every gate-enabled run from here on. **M3+ (shadow/live modes) is sequenced AFTER WS5 relational binding** per `ws5_relational_binding_prd.md` §4: the relational core changes what stage-4 escalation buys, so the gate must learn against the final cost structure, not today's. M5 remains MuJoCo-gated.

---

## 1. What "heuristic, not learned" means today

`AttentionGate.decide()` scores each cycle as a **fixed linear combination** of four normalized inputs — novelty, prediction error, affect, priority — against a hand-set threshold (0.55), with hand-set weights (probe preset 0.5/0.2/0.2/0.1), hand-set hysteresis (k cycles), and a hand-set budget gain. Every sensitivity was chosen by us via offline grid replay (`tune_gate.py`), not by the agent's experience. Psychologically this is an **innate orienting reflex**: fixed salience wiring, incapable of value-driven attentional capture. If mild novelty reliably preceded pain in the agent's world, it could never learn to escalate on it.

The goal: the *policy* of what deserves deliberation becomes a small trainable module shaped by the measured consequences of its own past decisions — while the reflexes around it stay innate.

## 2. Gap analysis (current → required)

| # | Dimension | Current state | Required for a learned gate | Gap size |
|---|---|---|---|---|
| G1 | Decision function | Fixed linear score in `AttentionGate.decide()` | Trainable `GateNet` (tiny MLP → escalation logit) behind the same `GateDecision` interface | New module, small |
| G2 | Training signal | None. No ground truth for "should have escalated" | **Shadow deliberation**: on sampled skip cycles, also compute fresh z4/risk and measure divergence from the decayed precedent actually used (regret); on sampled escalations, measure how little changed (waste) | New mechanism; cheap — stage 4 is one `risk_mlp(z3)` forward, and z3 is in hand |
| G3 | Decision-grained data | Gate telemetry exists (`gate_i_*`) but only at the eval sampler's stride (~6 cycles); spikes and per-decision context fall between samples | Per-cycle **gate decision log** (inputs, decision, reason, shadow divergence, outcome taps) written off the hot path | New JSONL channel; write-behind pattern already exists |
| G4 | Outcome grounding | Pain/viability/PE live on the State Bus per cycle but are never attributed to gate decisions | Outcome windows joined to each decision (pain delta, PE delta, viability delta over horizon H) in the dataset builder | Offline join, no runtime cost |
| G5 | Environment richness | **Proven degenerate** (2026-07-04): percept manifold saturates by cycle ~300; nothing that happens matters differently, so all gating policies score alike | An embodied world where attending correctly changes bodily outcomes (MuJoCo) | External dependency — gates M5, not M0–M4 |
| G6 | Safety rails | Fast-path threat, hysteresis, soft budget, `DECADIC_GATE_ENABLED=0` byte-parity — all present and tested | Same rails wrapped **around** the learned core; fast path and budget must be unlearnable | Preserve + pin with tests |
| G7 | Persistence | `save_brain` serializes the neural bundle only; gate state is ephemeral per `bundle._attention_gate` | GateNet weights ride the brain checkpoint and Saved-Agents round-trip | Small extension |
| G8 | Evaluation | Redesigned probe (count contract, 0.20 calibrated bar, peak telemetry); `tune_gate.py` replays heuristic configs | Replay evaluator extended to score a *learned* policy on recorded runs; A/B (learned vs heuristic) probe + soak protocol | Extension of existing tools |

## 3. Design

### 3.1 GateNet (G1)
A deliberately tiny MLP — salience reflex, not a mind: input ≈ 8 features (the four existing normalized inputs + drive pressure, trailing escalation rate, precedent age, latch state), one hidden layer (~16, tanh), sigmoid output = escalation probability. Decision: `p ≥ p_threshold` (default 0.5), then the existing wrappers apply unchanged and in this order: **fast-path threat overrides everything → hysteresis latch → budget raises the effective bar** (implemented in logit space so the budget keeps its current semantics). `GateDecision` shape, telemetry, and precedent pass-through are untouched — the learned core swaps in exactly where the weighted sum sits today.

Mode switch: `DECADIC_GATE_MODE = heuristic | shadow | learned` (default `heuristic`).
- `heuristic`: today's behavior, byte-identical.
- `shadow`: GateNet computes and logs its logit every cycle; the heuristic still decides. Zero behavior change; produces agreement telemetry and training data.
- `learned`: GateNet decides inside the rails.

### 3.2 Shadow deliberation — the training signal (G2)
The architecture already contains the counterfactual machinery: a skip substitutes a decayed precedent `(z4·decay, risk_logit·decay)` via `stage4_override`. The missing piece is occasionally computing what fresh deliberation *would* have said:

- On a sampled fraction of **skips** (`DECADIC_GATE_SHADOW_RATE`, default 0.05): inside the forward, compute fresh `z4 = risk_mlp(z3)` alongside the override (z3 is already computed; one extra small MLP forward, no autograd). **Regret** = divergence between fresh and substituted: `|risk_fresh − risk_used|` and normalized `‖z4_fresh − z4_used‖`.
- On a sampled fraction of **escalations**: **waste** = divergence between the fresh z4 and what the precedent would have supplied — near-zero divergence means the deliberation bought nothing.
- Shadow results are diagnostics only (never fed back into the live cycle) and land in the decision log.

Label construction (offline, M1): `should_escalate = σ(α·(regret − cost))` where cost reflects the budget's compute price, sharpened by outcome windows (a skip followed by a pain rise within horizon H after high regret is a strong positive; an escalation with near-zero waste and no outcome change is a negative). Thresholds α, cost, and H are dataset hyperparameters, reported with class balance in every training run — not silently baked in.

### 3.3 Decision log (G3, G4)
`DECADIC_GATE_LOG=1` streams one compact JSONL row per gate decision through the existing write-behind pattern (log-and-continue, never on the hot path): cycle, the 8 GateNet features, decision + reason, shadow divergences when sampled, and State-Bus outcome taps (pain, viability, pc_ema). The dataset builder joins forward outcome windows offline. Overhead acceptance: <1% cycle time at 10 Hz, verified in the harness.

### 3.4 Training regime (offline-first)
1. **Offline:** `scripts/train_gate.py` fits GateNet on decision logs from recorded runs (probe runs + soaks, both gate modes). Standard split by run, never by row (rows within a run are autocorrelated).
2. **Replay evaluation:** `tune_gate.py` grows a `--policy gatenet <weights.pt>` mode — same recorded-stream replay used to calibrate the heuristic, now scoring the learned policy on: threat recall (must be 1.0 — though the fast path enforces this structurally), novelty-burst answered per the redesigned count contract, calm rate, escalation rate vs target, and total regret.
3. **Shadow online:** deploy in `shadow` mode during normal runs; measure live agreement/disagreement and harvest more data. Promotion to `learned` requires the replay gates *and* shadow-mode disagreement review.
4. **Online adaptation** (slow EMA-style updates during live runs) is explicitly out of scope until M5 — first the policy must be trustworthy frozen.

### 3.5 What is never learned (G6)
- **Fast-path threat**: hard-coded before the net, exactly as today. Reflexes are not trainable away.
- **Budget and hysteresis**: wrappers outside the learned core.
- **Off-switch parity**: `DECADIC_GATE_ENABLED=0` and `DECADIC_GATE_MODE=heuristic` each remain byte-identical to their current baselines, pinned by tests.

### 3.6 Persistence (G7)
GateNet weights serialize into the existing neural bundle file (alongside the stack, versioned key; absent key = heuristic mode on load, so old brains load unchanged) and therefore ride `/checkpoint`, `/restore`, and the Saved-Agents round-trip for free. The saved-agent manifest records `gate_mode` and the GateNet training-run id.

## 4. Success criteria

1. **Data:** decision log captures ≥ 99% of decisions with <1% cycle-time overhead; shadow deliberation adds no measurable cycle-rate regression at 5% sampling.
2. **Offline:** on held-out recorded runs, GateNet strictly dominates the tuned heuristic on total regret at matched escalation rate (or matches regret at lower rate); threat recall 1.0; calm ≤ 0.10.
3. **Probe:** redesigned gate probe passes in `learned` mode — exactly N first-exposure bursts answered, revisit quiet, calm quiet, threat 100%.
4. **A/B soak:** 1-h soak in `learned` vs `heuristic`, same config: no stability regression (stalls, NaN, cycle rate), escalation rate within target band, PC-loss endpoint within noise of heuristic run.
5. **Parity:** off-switch and heuristic-mode byte-parity tests green; brain checkpoints round-trip GateNet weights.
6. **(M5, MuJoCo)** Retrained on embodied outcome data, the learned gate shows a measurable behavioral difference the heuristic cannot: escalation selectivity for outcome-relevant stimuli (quantified as regret reduction on embodied held-out runs).

## 5. Risks

- **Label circularity:** regret is measured against the precedent mechanism, which the gate's own past behavior shaped. Mitigation: shadow-rate sampling on *both* branches, and dataset reports condition on precedent age.
- **Synthetic-world triviality (G5):** on the current rig the learnable signal may be near-zero — expected, documented, and why M5 exists. M0–M4 still deliver the machinery, the safety story, and the offline evidence that the pipeline learns *when signal exists* (verifiable with a synthetic-label sanity task).
- **Silent behavioral drift in `learned` mode:** mitigated by shadow-mode gating before promotion, the A/B soak, and reason-code telemetry (`gate_reason` gains `"net"` alongside existing codes).
- **Checkpoint schema creep:** versioned key in the bundle; loader tolerates absence (G7).

## 6. Explicit non-goals

- No online weight updates during live runs (until after M5).
- No learned modification of the fast path, budget, or hysteresis.
- No expansion of GateNet into a policy over actions — it decides *whether to think*, not *what to do*.
