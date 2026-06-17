"""Explicit goal lifecycle: latch on onset, hold, close on achieve/abandon/truncate/die."""

from decadic.state.goal_lifecycle import (
    GoalState,
    OUTCOME_ABANDONED,
    OUTCOME_ACHIEVED,
    OUTCOME_DIED,
    OUTCOME_TRUNCATED,
)


def _gs(**kw):
    base = dict(onset_deficit=0.15, satisfy_level=0.92, abandon_cycles=3, max_cycles=1000)
    base.update(kw)
    return GoalState(**base)


def test_no_goal_when_full():
    gs = _gs()
    assert gs.update([1.0, 1.0, 1.0], 0) == []
    assert gs.status == "idle"


def test_latches_dominant_deficit():
    gs = _gs()
    ev = gs.update([0.80, 0.99, 1.0], 1)
    assert [e.kind for e in ev] == ["opened"]
    assert ev[0].goal_id == "hydration"
    assert gs.status == "active"
    assert gs.dwell(6) == 5


def test_closes_on_achievement_without_reopen():
    gs = _gs()
    gs.update([0.80, 0.99, 1.0], 1)
    ev = gs.update([0.95, 0.99, 1.0], 10)
    assert [e.kind for e in ev] == ["closed"]
    assert ev[0].outcome == OUTCOME_ACHIEVED
    assert ev[0].onset_cycle == 1 and ev[0].close_cycle == 10
    assert gs.status == "idle" and gs.episodes == 1


def test_achievement_reopens_if_another_need_presses():
    gs = _gs()
    gs.update([0.80, 0.99, 1.0], 1)  # hydration goal
    ev = gs.update([0.95, 0.60, 1.0], 5)  # hydration sated, energy now low
    kinds = [(e.kind, e.goal_id) for e in ev]
    assert ("closed", "hydration") in kinds
    assert ("opened", "energy") in kinds
    assert gs.goal_id == "energy"


def test_abandons_after_persistent_mismatch_then_reopens():
    gs = _gs(abandon_cycles=3)
    gs.update([0.99, 0.70, 1.0], 0)  # energy goal
    gs.update([0.50, 0.70, 1.0], 1)  # hydration dominates (mismatch 1)
    gs.update([0.50, 0.70, 1.0], 2)  # mismatch 2
    ev = gs.update([0.50, 0.70, 1.0], 3)  # mismatch 3 -> abandon energy, open hydration
    outcomes = [(e.kind, e.outcome, e.goal_id) for e in ev]
    assert ("closed", OUTCOME_ABANDONED, "energy") in outcomes
    assert ("opened", None, "hydration") in outcomes


def test_truncates_overlong_episode():
    gs = _gs(max_cycles=5)
    gs.update([0.80, 0.99, 1.0], 0)
    ev = gs.update([0.80, 0.99, 1.0], 5)
    assert any(e.outcome == OUTCOME_TRUNCATED for e in ev if e.kind == "closed")


def test_death_closes_open_goal():
    gs = _gs()
    gs.update([0.80, 0.99, 1.0], 1)
    ev = gs.update([0.80, 0.99, 1.0], 9, alive=False)
    assert [e.kind for e in ev] == ["closed"]
    assert ev[0].outcome == OUTCOME_DIED
    assert gs.status == "idle"


def test_dwell_is_zero_when_idle():
    gs = _gs()
    assert gs.dwell(100) == 0
