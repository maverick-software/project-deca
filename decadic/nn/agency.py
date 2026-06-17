"""Sense-of-agency head: does this object move because *I* command it?

A comparator model of agency (Blakemore/Frith): for each discovered slot, one
predictor estimates its next image-space motion from the efference copy (the
motor command the agent just issued), and a second, efference-blind baseline
predicts motion from appearance alone. When the efference-conditioned predictor
beats the baseline, the slot's motion is explained by the agent's own action -
the signature of "this is mine" (a hand), as opposed to an external object that
moves on its own or stays put. The per-slot agency score is that error
reduction, accumulated over time in working memory.

Built only in discovered perception mode; absent (and serialized away) otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AgencyHead(nn.Module):
    """Two motion predictors (efference-conditioned vs efference-blind baseline)."""

    def __init__(self, *, slot_dim: int, n_actuators: int, hidden: int = 64) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.n_actuators = n_actuators
        self.efferent = nn.Sequential(
            nn.Linear(slot_dim + n_actuators, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # predicted (du, dv) image-space motion
        )
        self.baseline = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self, slots: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: [M, slot_dim]; u: [M, n_actuators]. Returns (pred_eff, pred_base) each [M, 2]."""
        pred_eff = self.efferent(torch.cat([slots, u], dim=-1))
        pred_base = self.baseline(slots)
        return pred_eff, pred_base
