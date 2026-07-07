"""Goal-episode accumulator: turn the live transition stream into ordered,
return-annotated episodes for distal credit assignment.

The agent runtime feeds this one realized ``Transition`` per cycle plus the
``GoalEvent``s emitted by :class:`decadic.state.goal_lifecycle.GoalState`. While a
goal is open, transitions are collected in order (and stamped with
``episode_id`` / ``step_idx`` / ``goal_id``). When the goal closes, the
lambda-returns are computed over the ordered rewards/features
(``decadic.consolidation.returns``) and written back into each transition's
``ret`` / ``sf_target`` -- IN PLACE, so the copies already living in the salience
replay buffer become return-annotated without a second push and without changing
the existing one-step consolidation data flow.

Only transitions carrying an interoceptive feature (``feat`` is not ``None``) can
contribute a reward/return, so episodes are built from those; idle/feature-less
cycles are skipped for return purposes but never block the existing replay path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from decadic.consolidation.returns import lambda_returns, lambda_returns_vec


def _as_list(vec: Any) -> list[float]:
    """Coerce a torch tensor / list / sequence into a flat python float list."""
    if vec is None:
        return []
    if hasattr(vec, "reshape") and hasattr(vec, "tolist"):
        return [float(x) for x in vec.reshape(-1).tolist()]
    return [float(x) for x in vec]


@dataclass
class EpisodeAccumulator:
    """Collects ordered transitions per goal episode and annotates returns on close."""

    gamma: float
    lam: float
    max_steps: int = 4096  # bound retained refs per open episode (memory safety)
    normalize: bool = False  # WS-FORAGE M1: (1-gamma) target normalization
    # WS-EXPAND E2.4: optional live discount source, resolved at close time so
    # the horizon channel's (clamped, rate-limited) discount reaches episode
    # returns. None -> the constructor-time ``gamma`` exactly (pre-E2 behavior;
    # with the flag off the provider also returns the config value, so both
    # paths agree). Returns are normalized by (1-gamma) when ``normalize`` is
    # on, so magnitude stays comparable across small discount moves.
    gamma_provider: Any = None

    _episode_id: int = 0
    _open: bool = False
    _cur_id: int = -1
    _cur_goal: str = ""
    _steps: list[Any] = field(default_factory=list)
    # Telemetry of the most recent close.
    episodes_closed: int = 0
    last_len: int = 0
    last_return: float = 0.0
    last_outcome: str = ""

    def reset(self) -> None:
        """Drop any open episode (new mind / new life); counters preserved."""
        self._open = False
        self._cur_id = -1
        self._cur_goal = ""
        self._steps = []

    def on_open(self, goal_id: str, onset_cycle: int) -> None:
        self._episode_id += 1
        self._cur_id = self._episode_id
        self._cur_goal = str(goal_id)
        self._open = True
        self._steps = []

    def add(self, transition: Any) -> None:
        """Append the current step's transition to the open episode (if any)."""
        if not self._open or transition is None:
            return
        if getattr(transition, "feat", None) is None:
            return  # no interoceptive feature -> contributes no reward
        transition.episode_id = self._cur_id
        transition.step_idx = len(self._steps)
        transition.goal_id = self._cur_goal
        self._steps.append(transition)
        if len(self._steps) > self.max_steps:
            self._steps.pop(0)

    def on_close(self, outcome: str) -> list[Any]:
        """Compute lambda-returns over the episode and write them back in place.

        Returns the ordered, now-annotated transitions (handy for hindsight
        relabeling); they are already resident in the replay buffer.
        """
        steps = self._steps
        self._open = False
        self._steps = []
        self._cur_id = -1
        self._cur_goal = ""
        if not steps:
            return []
        rewards = [float(getattr(t, "reward", 0.0)) for t in steps]
        feats = [_as_list(getattr(t, "feat", None)) for t in steps]
        gamma = self.gamma
        if self.gamma_provider is not None:
            try:
                gamma = float(self.gamma_provider())
            except Exception:
                gamma = self.gamma  # provider failure -> constructor value
        rets = lambda_returns(
            rewards, gamma=gamma, lam=self.lam, normalize=self.normalize
        )
        sf_targets = lambda_returns_vec(
            feats, gamma=gamma, lam=self.lam, normalize=self.normalize
        )
        for t, g, sft in zip(steps, rets, sf_targets):
            t.ret = float(g)
            t.sf_target = sft
        self.episodes_closed += 1
        self.last_len = len(steps)
        self.last_return = float(rets[0]) if rets else 0.0
        self.last_outcome = str(outcome)
        return steps


def achieved_feature(steps: list[Any]) -> list[float]:
    """Net per-channel feature change realized over an episode (sum of phi)."""
    if not steps:
        return []
    total: list[float] = []
    for t in steps:
        f = _as_list(getattr(t, "feat", None))
        if not f:
            continue
        if not total:
            total = [0.0] * len(f)
        for i in range(min(len(total), len(f))):
            total[i] += f[i]
    return total


def build_hindsight_copies(
    steps: list[Any],
    achieved: list[float],
    *,
    gamma: float,
    lam: float,
    k: int = 1,
    normalize: bool = False,
) -> list[Any]:
    """Hindsight-relabel a non-achieved episode that still found relief.

    "The journey still taught me": even if the latched goal was missed, the
    trajectory reached an end-state that DID accrue relief in some channel
    (``achieved``). We recompute the successor-feature / return targets treating
    that achieved terminal feature as the episode's success bootstrap, then emit
    ``k`` relabeled COPIES (sharing the detached tensor refs; only the credit
    fields + goal tag differ). Pushed into the salience replay buffer, they
    densify the SF signal toward the off-goal relief the agent actually found --
    no fabricated innate reward, just re-exposure of real informative steps.
    """
    if not steps or k <= 0:
        return []
    feats = [_as_list(getattr(t, "feat", None)) for t in steps]
    rewards = [float(getattr(t, "reward", 0.0)) for t in steps]
    scalar_boot = float(sum(achieved)) if achieved else 0.0
    sf_targets = lambda_returns_vec(
        feats, gamma=gamma, lam=lam, bootstrap=achieved or None, normalize=normalize
    )
    rets = lambda_returns(
        rewards, gamma=gamma, lam=lam, bootstrap=scalar_boot, normalize=normalize
    )
    out: list[Any] = []
    for _ in range(int(k)):
        for t, g, sft in zip(steps, rets, sf_targets):
            out.append(
                dataclasses.replace(
                    t, goal_id="hindsight", ret=float(g), sf_target=list(sft)
                )
            )
    return out
