"""WS-ATTN — salience-priority memory hierarchy (Phases 2-6).

Covers fold-time salience scoring (organ_diag), tier-1 priority deliberation,
the tier-2 priority overflow + tier-3 consolidation cascade, the drain API,
tier pressure, and the pressure-driven rest trigger. Every feature is
flag-gated; the final test pins that flags-off == today's FIFO behavior.

The tier logic is driven by setting session salience directly, so the cascade
is exercised independently of the (separately tested) salience source.
"""

from __future__ import annotations

import asyncio

import pytest

from decadic.consolidation.rest import RestController
from decadic.cycle.stage_pipeline import (
    SerialPrefetchSupervisor,
    _observation_salience,
    _priority,
    _salience_features,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _fold(sup, obs=None, salience=None, urgent=None):
    sess = await sup.enqueue_observation(obs or {})
    await sup.mark_folded(sess.frame_seq)
    if salience is not None:
        sess.salience = float(salience)
    if urgent is not None:
        sess.urgent = bool(urgent)
        if not urgent:
            sup._urgent.pop(sess.frame_seq, None)
    return sess.frame_seq


# ---------------------------------------------------------------------------
# 2.0 salience scoring
# ---------------------------------------------------------------------------
def test_salience_features_events_only_when_rich_off(monkeypatch):
    monkeypatch.setenv("DECADIC_SALIENCE_RICH", "0")
    obs = {"events": [{"intensity": 0.4}]}
    diag = {"looming_count": 3, "local_motion_max": 1.0, "flow_confidence": 1.0}
    assert _salience_features(obs, diag) == _observation_salience(obs)  # diag ignored


def test_salience_features_uses_organ_diag_when_rich_on(monkeypatch):
    monkeypatch.setenv("DECADIC_SALIENCE_RICH", "1")
    # no events -> base 0; strong motion/looming should lift salience
    sal, urgent = _salience_features({}, {"looming_count": 3, "local_motion_max": 1.0, "flow_confidence": 1.0})
    assert sal == pytest.approx(1.0) and urgent is False
    # richer never lowers the events-based floor
    sal2, u2 = _salience_features({"events": [{"intensity": 0.9}]}, {"looming_count": 0})
    assert sal2 == pytest.approx(0.9) and u2 is True


# ---------------------------------------------------------------------------
# 3.0 priority deliberation
# ---------------------------------------------------------------------------
def test_priority_select_picks_highest_salience(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_PRIORITY_SELECT", "1")
    sup = SerialPrefetchSupervisor(capacity=10)

    async def go():
        seqs = []
        for sal in (0.1, 0.9, 0.3, 0.5):
            seqs.append(await _fold(sup, salience=sal, urgent=False))
        sel, _ = await sup.pop_commit_candidate()
        assert sup.last_select_reason == "priority"
        assert sel.salience == pytest.approx(0.9)

    _run(go())


def test_priority_recency_decay_prefers_fresh(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_PRIORITY_SELECT", "1")
    monkeypatch.setenv("DECADIC_PIPELINE_RECENCY_TAU_S", "10")
    sup = SerialPrefetchSupervisor(capacity=10)

    async def go():
        old = await _fold(sup, salience=0.9, urgent=False)
        fresh = await _fold(sup, salience=0.5, urgent=False)
        # age the old one well past several tau
        sup.by_seq[old].folded_s -= 60.0
        sel, _ = await sup.pop_commit_candidate()
        assert sel.frame_seq == fresh  # 0.9*e^-6 < 0.5

    _run(go())


def test_urgent_still_preempts_priority(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_PRIORITY_SELECT", "1")
    sup = SerialPrefetchSupervisor(capacity=10)

    async def go():
        await _fold(sup, salience=0.9, urgent=False)
        u = await _fold(sup, obs={"events": [{"intensity": 0.2}]})  # urgent
        sel, _ = await sup.pop_commit_candidate()
        assert sel.frame_seq == u and sup.last_select_reason == "urgent"

    _run(go())


# ---------------------------------------------------------------------------
# 4.0 tier-2 overflow + promotion
# ---------------------------------------------------------------------------
def test_overflow_keeps_top_cap_evicts_lowest(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW", "1")
    sup = SerialPrefetchSupervisor(capacity=3)

    async def go():
        for i in range(1, 9):
            await _fold(sup, salience=i / 10.0, urgent=False)
        await sup.pop_commit_candidate()  # triggers coalesce+select
        m = sup.metrics()
        # top-3 by salience survive across ready(after 1 pop)+overflow; nothing dropped
        assert m["overflow_depth"] == 5
        assert m["consolidation_dropped"] == 0
        # every folded percept still accounted for (no silent drop)
        assert len(sup.ready) + m["overflow_depth"] == 7  # 8 folded - 1 deep-processed

    _run(go())


def test_overflow_promotes_back_when_capacity_frees(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW", "1")
    sup = SerialPrefetchSupervisor(capacity=2)

    async def go():
        for i in range(1, 7):
            await _fold(sup, salience=i / 10.0, urgent=False)
        # consume repeatedly; overflow should feed the freed T1 slots, best-first
        seen = []
        for _ in range(6):
            sel, _b = await sup.pop_commit_candidate()
            if sel is None:
                break
            seen.append(round(sel.salience, 1))
            await sup.mark_committed(sel.session_id)
        # highest salience processed first overall; all six eventually seen
        assert seen[0] == pytest.approx(0.6)
        assert set(seen) == {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}

    _run(go())


def test_overflow_spills_to_consolidation_ordered(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW", "1")
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW_CAP", "2")
    monkeypatch.setenv("DECADIC_PIPELINE_CONSOLIDATION_CAP", "100")
    sup = SerialPrefetchSupervisor(capacity=1)

    async def go():
        # Build a real backlog (fold all, THEN a pop triggers the cascade);
        # overflow only engages when the consumer falls behind, by design.
        for i in range(1, 8):
            await _fold(sup, salience=i / 10.0, urgent=False)
        await sup.pop_commit_candidate()
        m = sup.metrics()
        assert m["overflow_depth"] == 2
        assert m["overflow_spilled"] >= 1
        drained = sup.drain_consolidation(100)
        sals = [round(s.salience, 2) for s in drained]
        assert len(drained) >= 1
        assert sals == sorted(sals, reverse=True)  # highest-priority first

    _run(go())


def test_consolidation_cap_drops_lowest_only(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW", "1")
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW_CAP", "1")
    monkeypatch.setenv("DECADIC_PIPELINE_CONSOLIDATION_CAP", "2")
    sup = SerialPrefetchSupervisor(capacity=1)

    async def go():
        for i in range(1, 8):
            await _fold(sup, salience=i / 10.0, urgent=False)
        await sup.pop_commit_candidate()  # deliberates the top (0.7), cascades rest
        m = sup.metrics()
        assert m["consolidation_dropped"] >= 1  # forgot the least important
        alive = (
            [s.salience for s in sup.ready.values()]
            + [s.salience for s in sup._overflow.values()]
            + [s.salience for s in sup._consolidation_q]
        )
        # 0.7 was DELIBERATED (best outcome, not "lost"); the top survivor still
        # buffered is 0.6, and only the lowest (0.1..0.3) were forgotten.
        assert max(alive) == pytest.approx(0.6)
        assert min(alive) >= 0.4 - 1e-9

    _run(go())


# ---------------------------------------------------------------------------
# 6.1 pressure
# ---------------------------------------------------------------------------
def test_pressure_rises_with_tier_depth(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_OVERFLOW", "1")
    sup = SerialPrefetchSupervisor(capacity=10)

    async def go():
        p0 = sup.pressure()
        for i in range(5):
            await _fold(sup, salience=0.5, urgent=False)
        assert sup.pressure() > p0

    _run(go())


# ---------------------------------------------------------------------------
# 6.2 pressure-driven rest
# ---------------------------------------------------------------------------
def test_rest_enters_on_pressure():
    rc = RestController(load_threshold=1e9, min_wake_cycles=0, rest_cycles=5, pressure_threshold=2.0)
    # load can never trigger (huge threshold); pressure must
    resting = rc.note_cycle(cycle=1, pc_loss=0.0, threat=False, pressure=2.5)
    assert resting is True and rc.in_rest
    assert rc.telemetry()["rest_pressure"] == pytest.approx(2.5)


def test_rest_pressure_disabled_is_load_only():
    rc = RestController(load_threshold=1e9, min_wake_cycles=0, rest_cycles=5, pressure_threshold=0.0)
    assert rc.note_cycle(cycle=1, pc_loss=0.0, threat=False, pressure=99.0) is False


def test_rest_threat_aborts_even_under_pressure():
    rc = RestController(load_threshold=1.0, min_wake_cycles=0, rest_cycles=5, pressure_threshold=2.0)
    rc.note_cycle(cycle=1, pc_loss=0.0, threat=False, pressure=3.0)  # enters rest
    assert rc.in_rest
    assert rc.note_cycle(cycle=2, pc_loss=0.0, threat=True, pressure=3.0) is False
    assert not rc.in_rest and rc.rests_aborted == 1


# ---------------------------------------------------------------------------
# parity — flags off == today's FIFO behavior
# ---------------------------------------------------------------------------
def test_all_flags_off_is_fifo_parity(monkeypatch):
    for k in (
        "DECADIC_SALIENCE_RICH",
        "DECADIC_PIPELINE_PRIORITY_SELECT",
        "DECADIC_PIPELINE_OVERFLOW",
    ):
        monkeypatch.setenv(k, "0")
    monkeypatch.setenv("DECADIC_PIPELINE_O1_SELECT", "1")
    sup = SerialPrefetchSupervisor(capacity=2)

    async def go():
        seqs = [await _fold(sup, salience=s, urgent=False) for s in (0.9, 0.1, 0.5)]
        # capacity 2 -> one coalesced to the recent ring (dropped), not overflow
        sel, _ = await sup.pop_commit_candidate()
        assert sup.last_select_reason == "fifo"
        assert sel.frame_seq == seqs[1]  # oldest surviving (0.9 was coalesced out)
        assert sup.metrics()["overflow_depth"] == 0  # overflow tier unused

    _run(go())
