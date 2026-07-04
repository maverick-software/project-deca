# Run Report - soak_20260704_104047

**Result:** error: agent creation failed · **Git:** `43f6af4` · **Duration target:** 1.0 h · **Samples:** 0 · **Stalls:** 0

## Verdict

*(no gates.json in run dir)*

> Standing caveat: the dominant-loss canary misfires on synthetic proprioception-only input (PC loss legitimately dominates). Re-evaluate under MuJoCo embodied input.

## 1. Stability

- cycles: None -> None
- frames received/dropped: None / None
- nan recoveries: None
- gpu mem (MiB) first -> last: None -> None
- run dir size: None MB · disk free: None GB

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 2. Learning

- `neural_pc_loss_last`: insufficient samples
- `loss_total`: insufficient samples
- `forward_model_error`: insufficient samples
- `intero_pred_error`: insufficient samples
- `tactile_pred_error`: insufficient samples
- `effort_pred_error`: insufficient samples
- `consolidator_loss`: insufficient samples
- rewire events: None · growth events: None
- plasticity freezes/thaws: None/None

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 3. State coherence (A-F)

- `a_norm`: insufficient samples
- `b_norm`: insufficient samples
- `c_norm`: insufficient samples
- `e_norm`: insufficient samples
- viability first -> last: None -> None

*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*


*(plot unavailable - matplotlib not installed or too few samples)*

## 4. Memory and consolidation

- episodic rows: None -> None (0.0 MB)
- LTM db: 0.0 MB · property beliefs: None
- replay buffer: None · replays: None
- recall cache hits/misses: None/None

*(plot unavailable - matplotlib not installed or too few samples)*

## 5. Distinctness

*Single-run report - populated in comparison mode (`--compare run_a run_b`) once the baseline agent exists.*
