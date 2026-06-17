"""Successor-features head: the Layer-2 substrate that lets a SEEN cue acquire value.

``psi(state, u)`` predicts the discounted sum of future per-step features ``phi``
(the reservoir-change basis). Composed with the innate, deficit-gated drive
weights it yields a scalar value ``v = psi . w`` -- so "water in view" inherits
value from the future relief it predicts. That is the computational name for
incentive salience / Pavlovian-instrumental transfer: a cue that reliably
precedes relief acquires motivational pull and invigorates seeking.

Trained only by the consolidator's TD(lambda) loss over the agent's own lived
episodes (off the cognitive critical path); the live policy reads it -- with the
SF weights DETACHED (anti-hallucination: the policy cannot inflate its own value)
-- to shape action toward higher value, with a weight that ramps from 0 so a
fresh agent starts byte-identical to today.

Structurally a twin of the proprio / intero / tactile forward heads in
``neural_stack.py`` (the same two-linear + functional ``detach_params`` trick).
It lives in its own module so the already-large stack gains only thin wiring
(house FORBIDDEN #3). The output projection is zero-initialized, so ``psi == 0``
and ``v == 0`` until experience grows it -- nothing is "pre-trained."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessorFeaturesHead(nn.Module):
    """psi(state, action) -> discounted future feature vector (successor features)."""

    def __init__(self, state_dim: int, action_dim: int, feat_dim: int, hidden: int) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.l1 = nn.Linear(state_dim + action_dim, hidden)
        self.l2 = nn.Linear(hidden, feat_dim)
        # Zero-init the OUTPUT projection -> psi == 0 at birth (exact start parity),
        # while the first layer keeps normal init so gradients flow once it trains.
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.zero_()

    def predict(
        self, state: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict psi from (state latent, motor command).

        ``detach_params=True`` freezes the SF weights for the term (the live policy
        value-shaping path), so the gradient flows into the motor head via the
        inputs without letting the policy edit its own value estimate. The
        consolidator trains with ``detach_params=False``.
        """
        x = torch.cat([state, u], dim=-1)
        w1, b1 = self.l1.weight, self.l1.bias
        w2, b2 = self.l2.weight, self.l2.bias
        if detach_params:
            w1, b1, w2, b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        hidden = F.gelu(F.linear(x, w1, b1))
        return F.linear(hidden, w2, b2)
