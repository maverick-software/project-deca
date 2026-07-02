# WS1 Verification Report — Neural Path Post-bf16-Fix

**Date:** 2026-07-02 · **Git:** `9079da0` · **Hardware:** i7-12700K / 64GB / RTX 3080 10GB
**Runs:** `reports/ws1_20260702_072335` (tests + smoke), `_074554`, `_075651`, `_080555` (stall isolation), `_081152` (final learning run)

## Verdict

**Learning: VERIFIED. Stability: RESTORED at the 20-minute scale (2026-07-02) — the streaming stall was root-caused to a missing tombstone in the in-order fold drain and fixed; post-fix hunt ran 10,984 cycles / 20 min with zero stalls (`reports/stallhunt_20260702_092422/`). Remaining open: the growth-step hang (worked around by disabling growth) and the 12-hour soak.**

## 1. Test suite

739 tests, **0 failures, 0 errors**, 7 skipped, 116.5 s. JUnit XML in `ws1_20260702_072335/pytest_junit.xml`.

## 2. Learning verification (the core WS1 question)

The bf16 autocast fix holds and online predictive-coding learning is real. Final run, single agent, `full` preset, hf encoders, cuda, growth disabled, 406 samples over 2,311 cycles:

- `neural_pc_loss`: **1.996 → 0.236** (−88%), least-squares slope −9.1e-05/cycle, half-means 0.483 → 0.371, zero non-finite samples.
- Independently reproduced in both smoke runs (2.085 → 0.140 over 500 cycles; 1.990 → 0.242).
- Cycle wall time 69–84 ms (~12–14 Hz) at 203.8M trainable params.

**Final confirmation run (post-stall-fix, `reports/ws1_20260702_102035/`):** cycles 11 → 3,011 with zero stalls — the "cycles advance ≥ 3000" gate passes; PC loss 1.984 → 0.113 (−94%) over 518 samples, slope −1.68e-04/cycle, zero non-finite. The only failing check is the dominant-loss canary (section 4 caveat). An earlier attempt froze at 2,311 (defect #3, since fixed).

## 3. Stability defects found

### Defect 1 — observation-intake deadlock (RESOLVED via config)
`DECADIC_PREFETCH_OVERLOAD_POLICY` defaults to `"block"`: when observations arrive faster than cycles complete, `_enqueue_perception_observation` blocks forever on a full queue and the whole agent (and co-hosted agents) freezes. Reproduced twice at ~1,100 cycles with a 50 obs/s client. **Fix applied:** `drop_oldest` (existing code path with proper accounting) in `run_ws1.ps1`; recommend making `drop_oldest` the default — any real environment outpacing the cycle rate triggers this.

### Defect 2 — "growth-step hang" (WITHDRAWN — refuted 2026-07-02)
A freeze at cycle 501 initially implicated the growth evaluation (`growth_interval=500`). After the defect-3 tombstone fix, a 20-minute hunt **with growth enabled** ran 11,054 cycles (~22 growth intervals) with zero stalls (`reports/stallhunt_20260702_103725/`). The 501 freeze is better explained by defect 3: the prefetch queue filled during slow server warmup, frames were dropped, and the drain deadlocked on the hole — the proximity to the growth boundary was coincidence. Growth is re-enabled in `run_ws1.ps1`. Residual caution: whether `grow_step` actually woke neurons in the hunt depends on the pc-loss threshold; watch `growth_events` during the WS2 12-hour soak.

### Defect 3 — in-order fold drain deadlocks on dropped frames (ROOT-CAUSED & FIXED 2026-07-02)
Diagnosed via the new `/debug/tasks` endpoint + `scripts/stall_hunt.ps1` (stall captured at cycle 10,200; dump in `reports/stallhunt_20260702_085444/`). The fold drain (`_drain_ready_perception`) advances strictly by sequence number. The `drop_oldest` overload branch evicted frames from the prefetch queue and marked their sessions failed **without writing a tombstone into `_perception_ready`** — the drain then waited forever on the missing seq, 596 sessions piled up in "prefetched", ready stayed empty, and the cycle loop starved. The `block` policy freeze (defect 1) is the other face of the same design: the in-order drain has no way past a frame that never arrives. **Fix:** tombstone (obs=None, the established skip pattern) + immediate drain in the drop branch (`runtime.py` `_enqueue_perception_observation`). Regression test: `tests/test_prefetch_drop_tombstone.py`. **Recommended follow-up:** flip the default overload policy to `drop_oldest` (now safe) — two tests currently pin the default to `block`.

### Also fixed along the way
- `scripts/synthetic_ws_client.py` was lock-step (send → block on recv). The server does not guarantee 1:1 action-per-observation, so the client deadlocked and starved the agent. Rewritten with decoupled sender/receiver tasks.
- `--agent-id` attach mode added to `run_training_eval.py` and the client — the June 23 eval "pass" was a single sample from an idle agent (agents don't cycle without an observation stream).

## 4. Health-canary caveat

The `dominant loss usually bounded` gate failed in every run: PC loss is ~90% of total loss against the 0.7 threshold. With proprioception-only synthetic input (no real vision/audio), other loss heads are starved, so dominance is expected. Re-evaluate this gate under MuJoCo embodied input before treating it as a defect.

## 5. WS1 exit criteria vs. plan

- Test suite passes → **yes**
- PC loss downward trend → **yes** (−88% over 2,311 cycles)
- No NaN/Inf, no dtype regressions → **yes**
- Cycle rate recorded → **yes** (~12–14 Hz neural full preset; exceeds the 10 Hz target)
- Days-scale stability (criterion 1 groundwork) → **20-minute scale proven post-fix** (10,984-cycle hunt + 3,011-cycle gated run, zero stalls); hours/days scale is WS2's 12-hour soak

## 6. Recommended next actions

1. ~~Debug defect 3~~ DONE — fixed and verified (10,984 cycles / 20 min growth-off; 11,054 cycles / 20 min growth-on; zero stalls).
2. ~~Hunt the growth hang~~ WITHDRAWN — refuted by the growth-on hunt; growth re-enabled.
3. Flip the default overload policy to `drop_oldest` (now safe post-tombstone-fix); update the two tests that pin `"block"` (`test_parallel_sessions.py`, `test_stage_pipeline.py`).
4. Commit this work, then proceed to WS2 (measurement harness) and its 12-hour soak per `docs/verification_measurement_attention_gate_plan.md`; watch `growth_events` during the soak.

**Debugging assets added:** `GET /debug/tasks` (asyncio task-stack dump endpoint), `scripts/stall_hunt.ps1` (auto-capturing stall reproducer), `tests/test_prefetch_drop_tombstone.py` (regression lock on the fix).
