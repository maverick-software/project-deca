"""Predictive affect: forward-model the body's next affective context.

Self-model program, Phase 4. The architecture already *reacts* to its body —
viability, pain, pleasure, and priority are read each cycle and projected into
the stack (the 4-D episodic proxy ``ep``). But it never *anticipates* how it will
feel: perception is not shaped by the affect the agent expects in the next
moment. Interoceptive-inference / predictive-affect accounts hold that feeling is
itself a prediction, and that expected affect colours what is perceived.

``AffectPredictor`` is a small head that predicts the next-step affective context
forward from the current one. Its output is a *delta* on the 4-D affect vector,
added to ``ep`` before it is projected into the stack — so the agent perceives in
light of how it expects to feel. The output layer is zero-initialised, so a freshly
built predictor returns a zero delta and the cycle is byte-identical to the
no-prediction baseline until the head learns (its weights sit on the stack's
prediction-error graph, so it is trained for free by the main objective).

Kept deliberately tiny and dependency-free (a 2-layer MLP) so it adds negligible
parameters and never dominates the affect signal.
"""

from __future__ import annotations

import torch
from torch import nn

AFFECT_DIM = 4  # viability, pain, pleasure, priority (the episodic proxy `ep`)


class AffectPredictor(nn.Module):
    """Forward model: current affect context -> predicted next-step affect delta."""

    def __init__(self, affect_dim: int = AFFECT_DIM, hidden: int = 16) -> None:
        super().__init__()
        self.affect_dim = int(affect_dim)
        self.net = nn.Sequential(
            nn.Linear(self.affect_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.affect_dim),
        )
        # Zero-init the output layer => predicted delta is 0 at birth (parity).
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, affect: torch.Tensor) -> torch.Tensor:
        """Predicted delta on the affect context ([..., affect_dim])."""
        return self.net(affect)

    def predict(self, affect: torch.Tensor) -> torch.Tensor:
        """Detached prediction for inference / routing (no graph)."""
        with torch.no_grad():
            return self.forward(affect)
