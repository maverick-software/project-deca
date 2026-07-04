"""WS3B-M2.1: GateNet -- the learnable core of the stage 3->4 attention gate.

Deliberately tiny (a salience reflex, not a mind): 8 features -> one hidden
tanh layer -> escalation logit. The safety rails (fast-path threat, budget,
hysteresis) live OUTSIDE this module in AttentionGate and are not learnable
(PRD ws3b 3.5).

This module is the single source of truth for featurization: the offline
trainer (scripts/train_gate.py), the policy evaluator, and the runtime
shadow/learned modes (M3/M4) must all call ``featurize``/``normalize`` here
so train-time and decide-time inputs can never drift apart.

Feature order matches the decision log (WS3B-M0.1) and the dataset builder:
    novelty, pe, affect, priority, drive, esc_rate, latch, precedent_age
The first six are already ~[0,1]. latch is capped and scaled; precedent_age
is log-compressed (its useful range spans decades of cycles).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

FEATURES = (
    "novelty",
    "pe",
    "affect",
    "priority",
    "drive",
    "esc_rate",
    "latch",
    "precedent_age",
)
N_FEATURES = len(FEATURES)
STATE_DICT_KEY = "gate_net"  # versioned key inside the neural bundle
STATE_DICT_VERSION = 1
_LATCH_CAP = 8.0
_AGE_LOG_SCALE = 8.0  # log1p(age)/8 ~= 1.0 at age ~3000 cycles


def normalize(x: np.ndarray) -> np.ndarray:
    """Vectorized feature normalization; accepts (n, 8) raw rows."""
    out = np.asarray(x, dtype=np.float32).copy()
    out[:, 0:6] = np.clip(out[:, 0:6], 0.0, 1.0)
    out[:, 6] = np.clip(out[:, 6], 0.0, _LATCH_CAP) / _LATCH_CAP
    out[:, 7] = np.log1p(np.clip(out[:, 7], 0.0, None)) / _AGE_LOG_SCALE
    return out


def featurize(row: dict[str, Any]) -> np.ndarray:
    """One decision-log-shaped dict -> normalized (8,) float32 vector."""
    raw = np.asarray(
        [[float(row.get(f, 0.0) or 0.0) for f in FEATURES]], dtype=np.float32
    )
    return normalize(raw)[0]


class GateNet(nn.Module):
    """8 -> hidden(tanh) -> 1 escalation logit."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n,8) -> (n,)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def decide_prob(self, row: dict[str, Any]) -> float:
        """Runtime path (M3 shadow / M4 learned): one normalized decision."""
        x = torch.as_tensor(featurize(row), dtype=torch.float32).unsqueeze(0)
        return float(torch.sigmoid(self.forward(x))[0])

    # ---------------------------------------------------------- persistence
    def to_payload(self) -> dict[str, Any]:
        return {
            "version": STATE_DICT_VERSION,
            "hidden": self.hidden,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GateNet":
        if int(payload.get("version", -1)) != STATE_DICT_VERSION:
            raise ValueError(
                f"gate_net payload version {payload.get('version')!r} "
                f"!= {STATE_DICT_VERSION}"
            )
        net = cls(hidden=int(payload.get("hidden", 16)))
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return net

    def save(self, path: Path | str) -> None:
        torch.save(self.to_payload(), Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "GateNet":
        return cls.from_payload(torch.load(Path(path), map_location="cpu", weights_only=False))


def logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))
