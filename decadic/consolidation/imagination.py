"""Model-based imagined replay (Dreamer-lite, gated, default OFF).

"Expanding on it in quiet thought": besides replaying what actually happened, the
consolidator can roll the agent's OWN interoceptive world model forward a few
steps from a real start state and train the successor-features head on the
imagined consequence -- the agent reflecting on what WOULD happen if it kept
acting this way, not only what did.

Honest scope. The stack has interoceptive / proprioceptive / tactile forward
models but NO latent-dynamics model (nothing predicts the next deliberative
latent z5), so a faithful latent rollout is impossible. We therefore roll only
the INTEROCEPTIVE model forward with the start state + action held fixed
("if I keep doing this from here, where do my reservoirs go?"), accumulate the
discounted imagined feature deltas, and regress psi(state, action) toward that
imagined successor feature. This is a bounded approximation: the horizon is short
and the contribution is trust-weighted (``imagination_weight``) so a possibly-wrong
world model cannot fabricate large value. OFF by default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def imagined_sf_loss(
    stack,
    batch: list,
    *,
    gamma: float,
    horizon: int,
    device,
    normalize: bool = False,
) -> "torch.Tensor | None":
    """SF regression loss against imagined interoceptive rollouts. None if no-op.

    For each drive-on transition, hold (state, action) fixed and roll the
    interoceptive forward model ``horizon`` steps, summing the discounted feature
    deltas into an imagined successor-feature target; regress the (trainable) SF
    head toward it. The rollout itself runs under ``no_grad`` (it is a target, not
    a path the policy can edit).

    ``normalize=True`` scales the imagined target by ``(1 - gamma)`` (WS-FORAGE
    M1) so it shares the discounted-AVERAGE scale of the real episode SF targets
    (:func:`decadic.consolidation.returns.lambda_returns_vec`); otherwise the two
    training signals feeding the same head would disagree on magnitude. Default
    OFF -> byte-identical.
    """
    horizon = max(1, int(horizon))
    gamma = float(gamma)
    scale = (1.0 - gamma) if normalize else 1.0
    losses: list[torch.Tensor] = []
    for t in batch:
        if not getattr(t, "drive_on", False) or getattr(t, "prev_intero", None) is None:
            continue
        if getattr(t, "prev_state", None) is None or getattr(t, "prev_motor", None) is None:
            continue
        state = t.prev_state.to(device)
        u = t.prev_motor.to(device)
        with torch.no_grad():
            cur = t.prev_intero.to(device)
            target = torch.zeros_like(cur)
            disc = 1.0
            for _ in range(horizon):
                nxt = stack.forward_predict_intero(state, u, cur)
                if not torch.isfinite(nxt).all():
                    break
                target = target + disc * (nxt - cur)
                cur = nxt
                disc *= gamma
        if normalize:
            target = target * scale
        psi = stack.successor_predict(state, u)
        losses.append(F.mse_loss(psi, target))
    if not losses:
        return None
    return torch.stack(losses).mean()
