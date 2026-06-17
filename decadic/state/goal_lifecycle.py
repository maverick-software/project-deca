"""Explicit goal lifecycle: turn the always-on homeostatic drive into discrete
goal *episodes* with crisp boundaries for return-based credit assignment.

The interoceptive drive is continuous and overlapping (the agent is always a
little hungry and thirsty), which gives the one-step policy no episode to assign
distal credit over. ``GoalState`` latches the dominant deficit as the *active
goal* once it crosses an onset threshold, holds it while pursued, and closes it
on achievement (the reservoir recovered past the satisfy level), abandonment (a
different need dominates persistently), truncation (the episode ran too long), or
death. The closed ``[onset -> close]`` window is exactly the episode the
successor-features/return learner trains on (see ``decadic/consolidation``).

Pure data + transitions (no torch, no MuJoCo, no I/O) so it is cheap to unit
test; the agent runtime owns one instance and feeds it the normalized reservoir
vector each cycle. Reservoir levels are normalized to 0..1 (1.0 == full), aligned
to ``labels`` (hydration, energy, integrity -- the interoceptive channels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Aligned to frozen_encoders.controllable_intero_vector / cognition_trace labels.
GOAL_LABELS: tuple[str, ...] = ("hydration", "energy", "integrity")

# Episode outcomes (closed events). "achieved" feeds positive return; the others
# feed truncated returns and are the prime candidates for hindsight relabeling.
OUTCOME_ACHIEVED = "achieved"
OUTCOME_ABANDONED = "abandoned"
OUTCOME_TRUNCATED = "truncated"
OUTCOME_DIED = "died"


@dataclass(frozen=True)
class GoalEvent:
    """A lifecycle transition emitted by :meth:`GoalState.update`.

    ``kind`` is ``"opened"`` or ``"closed"``. For a close, ``outcome`` is one of
    the ``OUTCOME_*`` constants and ``close_cycle`` is set. One ``update`` call
    can emit a close followed by an open (e.g. achieve-then-pursue-next-need).
    """

    kind: str
    goal_id: str
    onset_cycle: int
    close_cycle: "int | None" = None
    outcome: "str | None" = None


@dataclass
class GoalState:
    """Latches/holds/closes the dominant homeostatic deficit as the active goal."""

    onset_deficit: float
    satisfy_level: float
    abandon_cycles: int
    max_cycles: int
    labels: tuple[str, ...] = GOAL_LABELS
    # Live state.
    status: str = "idle"  # "idle" | "active"
    goal_id: "str | None" = None
    onset_cycle: int = -1
    last_outcome: "str | None" = None
    episodes: int = 0
    _mismatch_run: int = 0
    _last_cycle: int = -1

    def dwell(self, cycle: "int | None" = None) -> int:
        """Cycles the current goal has been open (0 when idle)."""
        if self.status != "active" or self.onset_cycle < 0:
            return 0
        ref = self._last_cycle if cycle is None else int(cycle)
        return max(0, ref - self.onset_cycle)

    def _dominant(self, reservoirs: Sequence[float]) -> tuple[str, float]:
        """The label of the largest deficit (1 - level) and that deficit."""
        best_i, best_def = 0, -1.0
        for i in range(min(len(self.labels), len(reservoirs))):
            deficit = 1.0 - float(reservoirs[i])
            if deficit > best_def:
                best_i, best_def = i, deficit
        return self.labels[best_i], max(0.0, best_def)

    def _open(self, goal_id: str, cycle: int) -> GoalEvent:
        self.status = "active"
        self.goal_id = goal_id
        self.onset_cycle = int(cycle)
        self._mismatch_run = 0
        return GoalEvent("opened", goal_id, int(cycle))

    def _close(self, cycle: int, outcome: str) -> GoalEvent:
        gid = self.goal_id or "none"
        ev = GoalEvent(
            "closed", gid, self.onset_cycle, close_cycle=int(cycle), outcome=outcome
        )
        self.status = "idle"
        self.goal_id = None
        self.onset_cycle = -1
        self._mismatch_run = 0
        self.last_outcome = outcome
        self.episodes += 1
        return ev

    def update(
        self,
        reservoirs: Sequence[float],
        cycle: int,
        *,
        alive: bool = True,
    ) -> list[GoalEvent]:
        """Advance the lifecycle one cycle; return any opened/closed events.

        ``reservoirs`` are normalized levels (0..1) aligned to ``labels``.
        """
        self._last_cycle = int(cycle)
        events: list[GoalEvent] = []

        # Death closes any open goal (the journey still taught the agent something;
        # see hindsight relabeling). No new goal opens on a dead body.
        if not alive:
            if self.status == "active":
                events.append(self._close(cycle, OUTCOME_DIED))
            return events
        if not reservoirs:
            return events

        dom_label, dom_def = self._dominant(reservoirs)

        if self.status == "idle":
            if dom_def >= self.onset_deficit:
                events.append(self._open(dom_label, cycle))
            return events

        # --- active ---------------------------------------------------------
        gi = self.labels.index(self.goal_id) if self.goal_id in self.labels else 0
        level_g = float(reservoirs[gi]) if gi < len(reservoirs) else 1.0

        # Achievement: the pursued reservoir recovered to the satisfy level.
        if level_g >= self.satisfy_level:
            events.append(self._close(cycle, OUTCOME_ACHIEVED))
            if dom_def >= self.onset_deficit:  # another need already pressing
                events.append(self._open(dom_label, cycle))
            return events

        # Abandonment: a different need dominates for abandon_cycles in a row.
        if dom_label != self.goal_id and dom_def >= self.onset_deficit:
            self._mismatch_run += 1
        else:
            self._mismatch_run = 0
        if self._mismatch_run >= self.abandon_cycles:
            events.append(self._close(cycle, OUTCOME_ABANDONED))
            events.append(self._open(dom_label, cycle))
            return events

        # Truncation: cap an open episode so returns always resolve.
        if (cycle - self.onset_cycle) >= self.max_cycles:
            events.append(self._close(cycle, OUTCOME_TRUNCATED))
            if dom_def >= self.onset_deficit:
                events.append(self._open(dom_label, cycle))
            return events

        return events
