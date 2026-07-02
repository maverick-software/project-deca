# Run Report - soak_20260702_165403

**Result:** aborted_on_stall_at_cycle_833 · **Git:** `e975478` · **Duration target:** 1.0 h · **Samples:** 64 · **Stalls:** 1

## Verdict

**Overall: FAIL**

- [FAIL] no stalls - stall_events=1
- [SKIP] cycle rate holds - run shorter than 2 rollup hours - not evaluated
- [PASS] no NaN recoveries - nan_recovery_events=0
- [FAIL] pc loss decreases - half-means 0.5714 -> 2.2455
- [PASS] growth events observed (informational) - growth_events=0 - pc-loss threshold never crossed or growth idle

> Standing caveat: the dominant-loss canary misfires on synthetic proprioception-only input (PC loss legitimately dominates). Re-evaluate under MuJoCo embodied input.

## 1. Stability

- cycles: 9 -> 833
- cycle rate (per-minute): mean 2.88 Hz, min 0.00, max 4.90
- frames received/dropped: 1007 / 73
- nan recoveries: 0
- gpu mem (MiB) first -> last: 2930 -> 3501
- run dir size: 1.0 MB · disk free: 1509.3 GB

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 2. Learning

- `neural_pc_loss_last`: 2.0199 -> 2.4394 (half-means 0.5714 -> 2.2455, up)
- `loss_total`: 2.0235 -> 2.5151 (half-means 0.6371 -> 2.3165, up)
- `forward_model_error`: 0.0000 -> 0.0048 (half-means 0.0040 -> 0.0047, up)
- `intero_pred_error`: 0.0000 -> 0.0152 (half-means 0.0110 -> 0.0141, up)
- `tactile_pred_error`: 0.0000 -> 0.0024 (half-means 0.0042 -> 0.0024, down)
- `effort_pred_error`: 0.0000 -> 0.0036 (half-means 0.0039 -> 0.0036, down)
- `consolidator_loss`: 0.0000 -> 2.1765 (half-means 1.3855 -> 2.2588, up)
- rewire events: 3 · growth events: 0
- plasticity freezes/thaws: 0/0

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 3. State coherence (A-F)

- `a_norm`: insufficient samples
- `b_norm`: insufficient samples
- `c_norm`: insufficient samples
- `e_norm`: insufficient samples
- viability first -> last: 100.00 -> 78.81
- priority distribution: investigate: 56.2%, explore: 32.8%, avoid: 10.9%

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 4. Memory and consolidation

- episodic rows: 3 -> 831 (0.8 MB)
- LTM db: 0.0 MB · property beliefs: 0
- replay buffer: 821 · replays: 16
- recall cache hits/misses: 1040/1

*(plot unavailable - matplotlib not installed or too few samples)*

## 5. Distinctness

*Single-run report - populated in comparison mode (`--compare run_a run_b`) once the baseline agent exists.*
