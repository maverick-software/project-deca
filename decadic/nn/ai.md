# ai.md — decadic/nn

Quick orientation for future edits in this area.

## Purpose
The trainable **neural cognitive stack** and its building blocks: frozen sensory
encoders, the forward/interoception/tactile prediction heads, object-centric slots,
agency, plasticity, and — new — the **successor-features (value) head** that lets
the agent predict discounted future features and assign value to seen cues.

## Files (each < 500 lines, house rule)
- `neural_stack.py` — `NeuralCognitiveStack`: the full forward pass + heads
  (`forward_predict`, `forward_predict_intero`, `forward_predict_tactile`, …). Thin
  wiring only for SF: instantiates `self.sf_head` and sets
  `self.has_successor_model=True`; `successor_predict(state, u, *, detach_params)`
  delegates to the head (mirrors `forward_predict`’s `detach_params` anti-cheat).
- `successor_features.py` — **NEW.** `SuccessorFeaturesHead`: a small 2-layer MLP
  `ψ(state, action) -> R^INTERO_PRED_DIM` predicting discounted future feature
  occupancy. Reuses existing dims (`d_model`, `n_actuators`, `motor_hidden`,
  `INTERO_PRED_DIM`) so it does NOT touch `nn/config.py` presets or break
  checkpoints. Output layer is **zero-initialized** → ψ≡0 at birth (exact start
  parity; value contributes nothing until trained). `predict(state, u, *,
  detach_params=False)` detaches *weights* (not inputs) for the policy term, so
  gradient still flows to the action but not to ψ’s parameters.
- `config.py` — architecture presets (`_PRESET_SPECS` table). Unchanged by SF.
- `frozen_encoders.py` — frozen CLIP/Whisper (+ synthetic fallback) and the innate
  intero vectors/weights (`preferred_intero_vector`, `intero_preference_weights`,
  `controllable_intero_vector`) — the feature basis and `w` for value `v = ψ·w`.
- `bundle.py` — owns the stack + encoders + optimizer (save/load).
- `slots.py` / `agency.py` — object-centric slots + sense-of-agency comparator.
- `faculties.py` (default ON) / `plastic.py` (A/B/C switches, default OFF).
- `brain_map.py` — read-only topology export for the dashboard Brain Map.

## Invariants (do not break)
- **Naive-start parity.** ψ must stay zero at init (zero-init `l2`); the value term
  in `decadic/cycle/neural_pipeline.py` is additionally ramped from 0 over
  `sf_value_ramp_cycles`, so behavior is byte-identical to the pre-value baseline at
  the start of every fresh mind.
- SF reuses existing dims and adds a *separate* module → the main forward/motor
  contract (`n_actuators`, `INTERO_PRED_DIM`, proprio dims) is untouched, and old
  checkpoints load (the new `sf_head.*` keys are simply fresh/zero).
- The SF head is trained on the consolidator (TD(λ) + imagined), OFF the live
  critical path; the live cycle only *reads* a detached ψ for value shaping.
- Keep `neural_stack.py` thin re: SF — all real SF logic lives in
  `successor_features.py` (house file-size rule).

## Seams
- Trained in `decadic/consolidation/consolidator.py` (SF TD(λ) loss) +
  `imagination.py`. Value shaping (`v = (w_gated · ψ)`, detached, ramped) added in
  `decadic/cycle/neural_pipeline.py` next to `l_pref_intero`; surfaced as `sf_value`
  / `sf_value_weight` diagnostics → `runtime.py` metrics → dashboard MotorPanel.
- Config: `sf_enabled`, `sf_gamma`, `sf_lambda`, `sf_loss_weight`, `sf_value_weight`,
  `sf_value_ramp_cycles`, `sf_value_weight_for_cycle` in `decadic/config.py`.
