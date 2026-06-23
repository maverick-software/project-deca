"""Eval-only replay holdout splitter.

This utility keeps a train/eval split for future held-out prediction curves. It
does not alter live cognition; callers must explicitly use the train buffer for
replay and the holdout buffer for scoring.
"""

from __future__ import annotations

import random

from decadic.consolidation.replay_buffer import ReplayBuffer, Transition


class HoldoutReplaySplit:
    def __init__(
        self,
        train_capacity: int,
        *,
        holdout_fraction: float = 0.1,
        min_salience: float = 0.0,
        seed: int | None = None,
    ) -> None:
        frac = min(0.9, max(0.0, float(holdout_fraction)))
        holdout_capacity = max(1, int(round(train_capacity * frac))) if frac > 0 else 1
        self.train = ReplayBuffer(train_capacity, min_salience=min_salience, seed=seed)
        self.holdout = ReplayBuffer(holdout_capacity, min_salience=min_salience, seed=None if seed is None else seed + 1)
        self.holdout_fraction = frac
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> str:
        if self.holdout_fraction > 0.0 and self._rng.random() < self.holdout_fraction:
            self.holdout.push(transition)
            return "holdout"
        self.train.push(transition)
        return "train"

    def sample_train(self, batch_size: int) -> list[Transition]:
        return self.train.sample(batch_size)

    def sample_holdout(self, batch_size: int) -> list[Transition]:
        return self.holdout.sample(batch_size)

