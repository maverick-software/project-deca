"""Bounded, salience-prioritized replay buffer for dual-network consolidation.

Holds the realized learning transitions of the live Decadic cycle so a separate
consolidator network can replay them off the critical path (see consolidator.py).
Two design choices follow biological replay:

- **Salience prioritization.** Each transition carries a salience (how surprising
  / informative it was -- the summed forward-model + predictive-coding error).
  Sampling is weighted by salience, so hard, informative experiences are revisited
  more often, and when the buffer is full the *least* salient memory is evicted
  first. That eviction is the buffer's built-in forgetting: routine, well-predicted
  transitions decay out while surprising ones persist.

- **Thread safety.** The live cycle pushes from the asyncio event-loop thread while
  the consolidator samples from a worker thread (``asyncio.to_thread``), so every
  mutation is guarded by a lock.

The buffer stores detached CPU tensors only; nothing here imports the network or
holds a graph reference, so a captured transition never pins autograd memory.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    """One realized learning transition, captured detached on CPU.

    Holds exactly the latent inputs the live cycle used to compute its
    predictive-coding / forward-model / interoceptive losses, so the consolidator
    can recompute the same objective on its own weights:

    - ``z0`` / ``ep`` / ``mem``: the deliberative stack inputs ([1, *]) -> pc_loss.
    - ``prev_state`` / ``prev_motor`` / ``proprio_target``: the realized
      (state, action) -> next-proprio transition for the forward model.
    - ``prev_intero`` / ``intero_now``: the reservoir transition (only when the
      homeostatic drive was active that cycle; else ``None``).
    - ``salience``: sampling/eviction priority (higher = revisited more, evicted last).

    Episodic / credit-assignment fields (filled by the goal-episode accumulator;
    inert defaults keep a plain one-step transition byte-identical to before):

    - ``feat``: the per-step feature vector phi (reservoir change this cycle), the
      successor-features basis; ``reward`` is its innate-weighted sum (drive
      reduction). Both captured live by the neural pipeline.
    - ``episode_id`` / ``step_idx`` / ``goal_id``: position within a closed goal
      episode (assigned on accumulation).
    - ``ret`` / ``sf_target``: the lambda-return (scalar) and the discounted future
      feature target (vector), filled when the owning episode closes. ``ret`` is
      ``None`` until then, which the consolidator's SF loss uses as a "trainable"
      gate -- only return-annotated transitions feed it.

    Skill Dojo fields are optional and inert unless a dojo run is active. They let
    consolidation replay demonstrations/DAgger hints without touching the live
    cognitive loss.
    """

    z0: Any
    ep: Any
    mem: Any
    prev_state: Any
    prev_motor: Any
    proprio_target: Any
    drive_on: bool = False
    prev_intero: Any | None = None
    intero_now: Any | None = None
    prev_effort: Any | None = None
    effort_now: Any | None = None
    prev_body_pain: Any | None = None
    body_pain_now: Any | None = None
    effort_cost: float = 0.0
    fatigue_pain: float = 0.0
    salience: float = 0.0
    # --- episodic credit assignment (additive; defaults keep old behavior) ---
    feat: Any | None = None
    reward: float = 0.0
    episode_id: int = -1
    step_idx: int = 0
    goal_id: str = ""
    ret: float | None = None
    sf_target: Any | None = None
    # --- Skill Dojo metadata (consolidation-only imitation hints) -----------
    skill_id: str = ""
    origin: str = "self"  # self | demo | dagger
    expert_motor: Any | None = None
    demo_weight: float = 0.0
    success: bool = False
    # --- Anonymous scene dynamics (perception-only replay) -----------------
    scene_prev_features: Any | None = None
    scene_target_features: Any | None = None
    scene_match_mask: Any | None = None


class ReplayBuffer:
    """Fixed-capacity, salience-prioritized store with lowest-salience eviction."""

    def __init__(
        self, capacity: int, *, min_salience: float = 0.0, seed: int | None = None
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.min_salience = float(min_salience)
        self._items: list[Transition] = []
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def push(self, transition: Transition) -> bool:
        """Insert a transition. Returns True if it was retained.

        Transitions below the prune floor are dropped (uninformative). When the
        buffer is full, the new transition replaces the least-salient resident
        only if it is at least as salient -- otherwise the newcomer is dropped, so
        the buffer always keeps the most informative ``capacity`` experiences.
        """
        if transition.salience < self.min_salience:
            return False
        with self._lock:
            if len(self._items) < self.capacity:
                self._items.append(transition)
                return True
            idx_min = min(
                range(len(self._items)), key=lambda i: self._items[i].salience
            )
            if self._items[idx_min].salience <= transition.salience:
                self._items[idx_min] = transition
                return True
            return False

    def sample(self, batch_size: int) -> list[Transition]:
        """Salience-weighted sample (with replacement) of up to ``batch_size`` items."""
        with self._lock:
            if not self._items:
                return []
            k = max(1, int(batch_size))
            weights = [max(1e-9, float(it.salience)) for it in self._items]
            return self._rng.choices(self._items, weights=weights, k=k)

    def salience_stats(self) -> dict[str, float]:
        """Min / mean / max salience of the resident transitions (0 when empty)."""
        with self._lock:
            if not self._items:
                return {"min": 0.0, "mean": 0.0, "max": 0.0, "count": 0.0}
            sal = [float(it.salience) for it in self._items]
            return {
                "min": min(sal),
                "mean": sum(sal) / len(sal),
                "max": max(sal),
                "count": float(len(sal)),
            }
