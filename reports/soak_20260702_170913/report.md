# Run Report - soak_20260702_170913

**Result:** completed · **Git:** `3ef83aa` · **Duration target:** 1.0 h · **Samples:** 1266 · **Stalls:** 0

## Verdict

**Overall: PASS**

- [PASS] no stalls - stall_events=0
- [SKIP] cycle rate holds - run shorter than 2 rollup hours - not evaluated
- [PASS] no NaN recoveries - nan_recovery_events=0
- [PASS] pc loss decreases - half-means 0.3375 -> 0.2653
- [PASS] growth events observed (informational) - growth_events=3

> Standing caveat: the dominant-loss canary misfires on synthetic proprioception-only input (PC loss legitimately dominates). Re-evaluate under MuJoCo embodied input.

## 1. Stability

- cycles: 9 -> 30107
- cycle rate (per-minute): mean 8.26 Hz, min 3.03, max 9.53
- frames received/dropped: 32273 / 2080
- nan recoveries: 0
- gpu mem (MiB) first -> last: 3088 -> 4025
- run dir size: 27.5 MB · disk free: 1509.3 GB

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 2. Learning

- `neural_pc_loss_last`: 2.0167 -> 0.1462 (half-means 0.3375 -> 0.2653, down)
- `loss_total`: 2.0203 -> 0.1635 (half-means 0.3626 -> 0.2841, down)
- `forward_model_error`: 0.0000 -> 0.0000 (half-means 0.0012 -> 0.0005, down)
- `intero_pred_error`: 0.0000 -> 0.0026 (half-means 0.0045 -> 0.0021, down)
- `tactile_pred_error`: 0.0000 -> 0.0001 (half-means 0.0013 -> 0.0007, down)
- `effort_pred_error`: 0.0000 -> 0.0000 (half-means 0.0010 -> 0.0003, down)
- `consolidator_loss`: 0.0000 -> 0.0774 (half-means 0.9565 -> 0.1014, down)
- rewire events: 120 · growth events: 3
- plasticity freezes/thaws: 1/1

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 3. State coherence (A-F)

- `a_norm`: insufficient samples
- `b_norm`: insufficient samples
- `c_norm`: insufficient samples
- `e_norm`: insufficient samples
- viability first -> last: 100.00 -> 79.92
- priority distribution: explore: 52.7%, investigate: 34.7%, avoid: 12.6%

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 4. Memory and consolidation

- episodic rows: 3 -> 30105 (40.5 MB)
- LTM db: 0.0 MB · property beliefs: 0
- replay buffer: 2048 · replays: 376
- recall cache hits/misses: 37632/1

*(plot unavailable - matplotlib not installed or too few samples)*

## 5. Distinctness

*Single-run report - populated in comparison mode (`--compare run_a run_b`) once the baseline agent exists.*
