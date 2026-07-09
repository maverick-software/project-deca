"""WS-ATTN: O(1) two-lane commit selection + fold-time salience tagging.

Root cause (reports/bodydiag_kuzu_20260707_151250): the commit arbiter scanned
the entire ready queue every cycle to pick one session. With an unbounded
backlog (10,964 deep) that O(n) scan became a feedback loop -- slower cycles
-> consumer falls further behind producer -> queue grows -> slower scan --
which dropped cognition 4.14->3.29 cyc/s and left the agent deep-processing
22-min-stale perception.

Fix: tag salience/urgency ONCE at fold, keep an urgent lane, and pick in O(1)
(urgent head pre-empts, else FIFO head). Because folding is strictly in-order,
lane-head == the min-frame_seq the old arbiter scanned for, so the O(1) path is
a behavioral no-op vs the scan -- pinned by the parity test below.
"""

from __future__ import annotations

import asyncio

import pytest

from decadic.cycle.stage_pipeline import (
    DecadicCommitArbiter,
    SerialPrefetchSupervisor,
    _observation_salience,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _fold(sup: SerialPrefetchSupervisor, obs: dict) -> int:
    sess = await sup.enqueue_observation(obs)
    await sup.mark_folded(sess.frame_seq)
    return sess.frame_seq


# ---------------------------------------------------------------------------
# salience tagging
# ---------------------------------------------------------------------------
def test_observation_salience_helper():
    assert _observation_salience({}) == (0.0, False)
    assert _observation_salience({"events": []}) == (0.0, False)
    sal, urgent = _observation_salience({"events": [{"intensity": 0.42}]})
    assert urgent and abs(sal - 0.42) < 1e-9
    # multiple events -> max intensity; floor at 0.05 for a zero-intensity event
    sal2, u2 = _observation_salience({"events": [{"intensity": 0.0}]})
    assert u2 and sal2 == 0.05


def test_salience_tagged_once_at_fold():
    sup = SerialPrefetchSupervisor(capacity=100)

    async def go():
        seq = await _fold(sup, {"events": [{"intensity": 0.7}]})
        s = sup.by_seq[seq]
        assert s.urgent is True and abs(s.salience - 0.7) < 1e-9
        seq2 = await _fold(sup, {})  # no events
        s2 = sup.by_seq[seq2]
        assert s2.urgent is False and s2.salience == 0.0

    _run(go())


# ---------------------------------------------------------------------------
# O(1) selection semantics
# ---------------------------------------------------------------------------
def test_urgent_lane_preempts_then_fifo(monkeypatch):
    monkeypatch.setenv("DECADIC_PIPELINE_O1_SELECT", "1")
    monkeypatch.setenv("DECADIC_PIPELINE_PRIORITY_SELECT", "0")  # test the FIFO fallback
    sup = SerialPrefetchSupervisor(capacity=100)

    async def go():
        for _ in range(5):
            await _fold(sup, {})  # frames 1..5, non-urgent
        urgent_seq = await _fold(sup, {"events": [{"intensity": 0.9}]})  # frame 6
        sel, _b = await sup.pop_commit_candidate()
        assert sel.frame_seq == urgent_seq  # urgent pre-empts older normals
        assert sup.last_select_reason == "urgent"
        sel2, _b2 = await sup.pop_commit_candidate()
        assert sel2.frame_seq == 1 and sup.last_select_reason == "fifo"
        # metrics expose the lane
        assert sup.metrics()["urgent_queue_depth"] == 0

    _run(go())


def test_o1_matches_arbiter_parity(monkeypatch):
    """The O(1) path must select exactly what the O(n) arbiter would, step for
    step, over a mixed urgent/normal stream."""
    import random

    rng = random.Random(7)
    stream = []
    for i in range(1, 41):
        stream.append({"events": [{"intensity": rng.random()}]} if rng.random() < 0.3 else {})

    def drive(o1: str) -> list[tuple[int, str]]:
        monkeypatch.setenv("DECADIC_PIPELINE_O1_SELECT", o1)
        sup = SerialPrefetchSupervisor(capacity=10)
        picks: list[tuple[int, str]] = []

        async def go():
            it = iter(stream)
            done = False
            while not done:
                # fold two, pop one -- keeps a live backlog under the cap
                for _ in range(2):
                    obs = next(it, None)
                    if obs is None:
                        done = True
                        break
                    await _fold(sup, obs)
                sel, _b = await sup.pop_commit_candidate()
                if sel is not None:
                    picks.append((sel.frame_seq, sup.last_select_reason))
                    await sup.mark_committed(sel.session_id)
            # drain the rest
            while True:
                sel, _b = await sup.pop_commit_candidate()
                if sel is None:
                    break
                picks.append((sel.frame_seq, sup.last_select_reason))
                await sup.mark_committed(sel.session_id)

        _run(go())
        return picks

    assert drive("1") == drive("0")


def test_arbiter_still_importable_for_fallback():
    """The O(n) arbiter remains for DECADIC_PIPELINE_O1_SELECT=0 (A/B parity)."""
    arb = DecadicCommitArbiter()
    assert arb.select([])[1] == "none_ready"
