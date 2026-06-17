# ai.md — decadic/consolidation

Quick orientation for future edits in this area.

## Purpose
Off-critical-path **dual-network consolidation**: the live cycle records realized
learning transitions; a separate consolidator network replays them (biological
"sleep" replay) and soft-syncs the improved weights back. New in the distal-credit
work: replay now has a **timeline** (episodes + return-annotated transitions), a
**successor-features TD(λ)** objective, **hindsight relabeling**, and optional
**imagined** rollouts — the substrate that lets a seen resource acquire value and
distal reward land on the actions that preceded it.

## Files (each < 500 lines, house rule)
- `replay_buffer.py` — bounded, **salience-prioritized** `ReplayBuffer` + the
  `Transition` dataclass. `Transition` now also carries timeline/credit fields
  (all optional, default-inert for back-compat): `feat` (per-step feature φ, a
  detached CPU vector), `reward` (scalar drive-reduction), `episode_id`, `step_idx`,
  `goal_id`, `ret` (scalar λ-return), `sf_target` (vector λ-return for the SF head).
- `returns.py` — **NEW, pure.** Forward-view λ-returns: `lambda_returns(rewards, *,
  gamma, lam, values=None, bootstrap=0.0)` (scalar) and `lambda_returns_vec(feats,
  …)` (vector / successor features). `lam=1` → full discounted return; `lam=0` +
  `values` → TD(0). The "respect the journey" credit smear. No torch.
- `episodes.py` — **NEW, pure.** `EpisodeAccumulator(gamma, lam)` collects ordered
  in-episode `Transition`s (`on_open`/`add`/`on_close`), stamps `episode_id`/
  `step_idx`/`goal_id`, and on close computes `ret` + `sf_target` IN PLACE via
  `returns.py`; telemetry: `episodes_closed`, `last_len`, `last_return`,
  `last_outcome`. `achieved_feature(steps)` = net Σφ over the episode;
  `build_hindsight_copies(steps, achieved, *, gamma, lam, k)` = relabeled
  (goal_id `"hindsight"`) copies whose returns bootstrap on the achieved feature.
- `consolidator.py` — `ConsolidationManager`: clones the stack, replays batches,
  soft-syncs. `replay_batch_loss` adds the **SF TD(λ)** term
  `MSE(ψ(s,u), sf_target)` (weight `C.sf_loss_weight()`) when SF is enabled and the
  transition has an `sf_target`. `consolidate_once` adds **imagined** SF loss when
  `imagination_enabled() and sf_enabled()`; tracks `last_imagined_loss`.
- `imagination.py` — **NEW (gated, default off).** `imagined_sf_loss(stack, batch,
  *, gamma, horizon, device)`: rolls `stack.forward_predict_intero` forward for
  `horizon` steps (state+action held fixed) to build discounted imagined-φ targets,
  regresses the SF head on them. Returns `None` when no drive-on transitions exist.
- `landscape.py` — filter-normalized 2D loss-landscape probe (dashboard only).
- `stub_loop.py` — Phase-1 background-clock placeholder.

## Invariants (do not break)
- `returns.py` / `episodes.py` stay pure (lists/floats) → fast + unit-tested
  (`tests/test_episodic_returns.py`). Torch only enters the consolidator/imagination.
- Naive-start parity: the SF head is zero-init (see `decadic/nn/ai.md`), so the SF
  loss starts at 0 and consolidation is byte-identical to the pre-SF baseline until
  the head learns. Existing self-supervised replay losses are unchanged.
- Imagination is bounded-horizon, trust-weighted, and OFF by default (world-model
  bias guard). Keep it off the live critical path — it only runs in the burst.
- New `Transition` fields are append-only and optional; never make them required
  (old checkpoints / non-drive cycles must still round-trip).

## Seams
- `decadic/agents/runtime.py`: builds each `Transition`, pushes to `ReplayBuffer`,
  and routes it through `EpisodeAccumulator` + `GoalState`; on episode close it may
  push hindsight copies. `decadic/cycle/neural_pipeline.py`: fills `feat`/`reward`
  and computes the detached SF value-shaping term. Config knobs (`sf_*`, `her_*`,
  `imagination_*`, `goal_*`) live in `decadic/config.py`.
