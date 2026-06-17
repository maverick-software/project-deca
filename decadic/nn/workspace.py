"""Global workspace: capacity-limited competition + ignition + broadcast.

Self-model program, Phase 2. The architecture's "global workspace" today is a
post-hoc EMA blend of the working-memory attention summary into State Bus A
(``a_vec = (1-beta)*a_vec + beta*attn``) -- a soft average that always happens
and is never read back. Global Workspace Theory instead says content competes,
and only a *dominant coalition* wins access to the global broadcast (ignition);
sub-threshold content stays local and unreported.

This module implements that competition over the working-memory slots as a small
numpy selection mechanism (the slots are numpy, not tensors):

1. Score each candidate by salience x attention.
2. Winner-take-all: keep the top-``capacity`` candidates (the coalition).
3. Ignition: the coalition ignites only if it commands at least ``threshold`` of
   the total salience mass. Below threshold there is no ignition (A holds prior).
4. Broadcast: the ignited content is a softmax(score/temperature)-weighted blend
   of the winning candidates' vectors -- the single workspace vector that is then
   blended into A, fed back through the Phase-1 spine, boosts the episodic
   salience, and is described by the narrative. Reportable == broadcast == fed-back.

Off-branch parity: the cycle keeps the existing EMA blend when the flag is off,
so the baseline is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ignition:
    """Outcome of one workspace competition."""

    ignited: bool  # did a coalition break the ignition threshold?
    content: np.ndarray  # the broadcast workspace vector ([dim]); zeros if not ignited
    winners: list[int]  # indices of the winning coalition (empty if not ignited)
    score: float  # the winning coalition's share of the salience mass ([0,1])
    weights: np.ndarray  # broadcast weight per winner (softmax over winner scores)


class GlobalWorkspace:
    """Winner-take-all competition with an ignition threshold over slot content."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        capacity: int = 1,
        temperature: float = 1.0,
    ) -> None:
        self.threshold = float(min(1.0, max(0.0, threshold)))
        self.capacity = max(1, int(capacity))
        self.temperature = max(1e-3, float(temperature))

    def ignite(
        self,
        slots: np.ndarray,
        salience: np.ndarray,
        *,
        attention: np.ndarray | None = None,
    ) -> Ignition:
        """Compete ``slots`` (rows = candidate content vectors) by ``salience``.

        ``attention`` optionally modulates the competition (score = salience x
        attention). Returns the ignition outcome; ``content`` is zeros (no
        broadcast) when no coalition reaches the threshold or there is nothing to
        compete.
        """
        slots = np.asarray(slots, dtype=np.float64)
        if slots.ndim != 2 or slots.shape[0] == 0:
            dim = slots.shape[1] if slots.ndim == 2 else 0
            return Ignition(False, np.zeros(dim, dtype=np.float64), [], 0.0, np.zeros(0))
        n, dim = slots.shape
        sal = np.asarray(salience, dtype=np.float64).reshape(-1)
        sal = np.clip(sal[:n], 0.0, None)
        if attention is not None:
            att = np.clip(np.asarray(attention, dtype=np.float64).reshape(-1)[:n], 0.0, None)
            score = sal * att
        else:
            score = sal
        total = float(score.sum())
        if total <= 0.0:
            return Ignition(False, np.zeros(dim, dtype=np.float64), [], 0.0, np.zeros(0))

        # Winner-take-all: the top-`capacity` candidates form the coalition.
        order = np.argsort(score)[::-1]
        winners = [int(i) for i in order[: self.capacity]]
        coalition = score[winners]
        coalition_share = float(coalition.sum() / total)

        # Ignition: the coalition must command at least `threshold` of the mass.
        if coalition_share < self.threshold:
            return Ignition(False, np.zeros(dim, dtype=np.float64), [], coalition_share, np.zeros(0))

        # Broadcast: softmax(score/T) over the winning coalition.
        z = coalition / self.temperature
        z = z - z.max()
        w = np.exp(z)
        w = w / (w.sum() or 1.0)
        content = (slots[winners] * w[:, None]).sum(axis=0)
        return Ignition(True, content.astype(np.float64), winners, coalition_share, w)
