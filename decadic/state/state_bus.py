"""Persistent cognitive state elements A–F (State Bus).

Vector lengths align with ``decadic.config`` for Phase 2 tensor wiring:

- **A** ``state_of_mind``: STATE_OF_MIND_DIM (default 64)
- **B** ``emotion_physio``: EMOTION_DIM (default 32)
- **C** ``narrative_emb``: NARRATIVE_EMB_DIM (default 48)
- **E** ``metacognition``: METACOG_DIM (default 24)
- **F** ``action_history``: deque maxlen ACTION_HISTORY_MAX (default 32)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decadic.config import (
    ACTION_HISTORY_MAX,
    EMOTION_DIM,
    METACOG_DIM,
    NARRATIVE_EMB_DIM,
    STATE_OF_MIND_DIM,
)


def _zeros(n: int) -> np.ndarray:
    return np.zeros((n,), dtype=np.float32)


def _finite_list(arr: np.ndarray) -> list[float]:
    """Snapshot-only: map NaN/Inf -> 0 so transient neural instability can't emit
    invalid JSON (deserialized as null) that crashes numeric UI widgets. Operates on
    a copy via nan_to_num, so the live state array is never mutated."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()


def _finite(x: float) -> float:
    """Scalar counterpart to ``_finite_list`` for snapshot serialization."""
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else 0.0


@dataclass
class StateBus:
    """Elements A–F shared across the Decadic Cycle."""

    # A — State of Mind
    state_of_mind: np.ndarray = field(default_factory=lambda: _zeros(STATE_OF_MIND_DIM))
    # B — Emotional / physiological (includes pain/pleasure channels in first slots)
    emotion_physio: np.ndarray = field(default_factory=lambda: _zeros(EMOTION_DIM))
    pain_scalar: float = 0.0
    pleasure_scalar: float = 0.0
    # Last cycle's interoceptive drive pressure, so the homeostatic-relief reward
    # (drive-reduction) can be measured as a per-cycle delta. Persisted across
    # restarts so relief is continuous, not spuriously triggered on resume.
    prev_drive_pressure: float = 0.0
    # C — Internal narrative (embedding stub)
    narrative_emb: np.ndarray = field(default_factory=lambda: _zeros(NARRATIVE_EMB_DIM))
    narrative_text_stub: str = ""
    # D — Current priority
    priority_scalar: float = 0.5
    priority_label: str = "explore"
    # E — Metacognition
    metacognition: np.ndarray = field(default_factory=lambda: _zeros(METACOG_DIM))
    # F — Action history / efference copy
    action_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=ACTION_HISTORY_MAX)
    )

    cycle_index: int = 0

    def snapshot_dict(self) -> dict[str, Any]:
        return {
            "A_state_of_mind": _finite_list(self.state_of_mind),
            "B_emotion_physio": _finite_list(self.emotion_physio),
            "B_pain_scalar": _finite(self.pain_scalar),
            "B_pleasure_scalar": _finite(self.pleasure_scalar),
            "prev_drive_pressure": _finite(self.prev_drive_pressure),
            "C_narrative_emb": _finite_list(self.narrative_emb),
            "C_narrative_text_stub": self.narrative_text_stub,
            "D_priority_scalar": _finite(self.priority_scalar),
            "D_priority_label": self.priority_label,
            "E_metacognition": _finite_list(self.metacognition),
            "F_action_history": list(self.action_history),
            "cycle_index": self.cycle_index,
        }
