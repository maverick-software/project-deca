# Cycle-12 Plasticity-Freeze — Ablation Results

**Goal:** localize why the plasticity instability guard freezes structural learning at ~cycle 12 (`pc_ema_diverged_or_nonfinite`). Run five short sessions, change one thing each, and see which one stops the freeze.

> These runs must be done on the GPU host (torch + CUDA + the body). Fill the table from each run and paste it back for interpretation.

## Procedure (repeat for each config below)

1. Open a fresh shell, `cd "D:\Users\charl\software\Self-Determination Model"`, activate the venv.
2. Set the run's env flag(s) (config table), plus profiling: `$env:DECADIC_CYCLE_PROFILE="1"`.
3. Launch your **normal** setup that reproduces the freeze (server + body) — **not** `--dry-run`, since the freeze likely depends on the live feedback path. Create a fresh agent.
4. Let it run ~30 cycles (a few seconds at your cycle rate).
5. Capture:
   - **Did it freeze + at what cycle:**
     ```powershell
     Select-String -Path .\logs\decadic_server.jsonl -Pattern "frozen|froze|pc_ema|plasticity" | Select-Object -Last 5
     ```
   - **pc-loss trajectory (first ~15 cycles):**
     ```powershell
     Select-String -Path .\logs\decadic_server.jsonl -Pattern "neural_pc_loss|cycle_profile" | Select-Object -First 15
     ```
   - **Live state:** `GET http://127.0.0.1:8765/agents` → copy the id → `GET /agent/<id>/metrics` → read `plasticity_frozen`, `neural_pc_loss`.
6. Stop the run before starting the next config (so each agent is fresh).

## Configs

| # | Env change (PowerShell) | Hypothesis it tests |
|---|--------------------------|---------------------|
| 0 | *(none)* | baseline — should freeze ~cycle 12 |
| 1 | `$env:DECADIC_SELF_MODEL_FEEDBACK="0"` | the A‖C‖E self-report feedback loop |
| 2 | `$env:DECADIC_PREDICTIVE_AFFECT="0"` | predicted-affect added to `ep` |
| 3 | `$env:DECADIC_REPRESENTED_SELF="0"` | represented-self feedback ingress |
| 4 | `$env:DECADIC_MEMORY_EFFICIENT_TRAINING="0"` | bf16 forward + 8-bit Adam numerics |

*(Reset each flag to its default before the next run, i.e. set the others back / open a clean shell.)*

## Results — fill in

| # | Config | Froze? (Y/N) | Freeze cycle | pc-loss behavior (climbing / sudden-NaN / flat) | Final pc-loss before freeze | Notes |
|---|--------|--------------|--------------|--------------------------------------------------|-----------------------------|-------|
| 0 | baseline                         |   |   |   |   |   |
| 1 | self_model_feedback=0            |   |   |   |   |   |
| 2 | predictive_affect=0              |   |   |   |   |   |
| 3 | represented_self=0               |   |   |   |   |   |
| 4 | memory_efficient_training=0      |   |   |   |   |   |

## Interpretation key

- **A feedback flag (1/2/3) stops the freeze** → that recurrent path's training-time gain is the cause. Fix: damp the fed-back signal (LayerNorm / bounded gate) so the loop is contractive.
- **`memory_efficient_training=0` stops it** → bf16/8-bit-Adam numerics. Fix: add loss scaling / keep the trainable stack fp32 / fp32 for the recurrent heads.
- **pc-loss climbs geometrically** before the freeze → divergence (feedback gain > 1 or LR too high).
- **pc-loss jumps straight to NaN/Inf** in one cycle → numerical overflow (points at bf16).
- **Nothing stops it** → the instability is in the core objective independent of these; lower `DECADIC_LEARNING_RATE` (e.g. 3e-5) and re-test, and check the NaN firewall / input statistics.

> Separately, even after the cause is fixed, make the guard **recoverable** (guarded thaw once `pc_ema` is stable for a window) so a transient early blip can't permanently disable structural learning for the whole run.
