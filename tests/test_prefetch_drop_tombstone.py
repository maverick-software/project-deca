"""Regression: drop_oldest must leave a tombstone for the in-order fold drain.

The fold drain (`_drain_ready_perception`) advances strictly by sequence
number (`_perception_next_commit`). Before the 2026-07-02 fix, the
drop_oldest overload branch removed a frame from the prefetch queue and
marked its session failed but never wrote a tombstone into
`_perception_ready` — so the drain waited on the dropped seq forever, every
later frame accumulated unprocessed, and the cycle loop starved with
sessions stuck in "prefetched" (stall reproduced at cycle 10,200; dump in
reports/stallhunt_20260702_085444/).
"""

import asyncio

import pytest

from decadic import config as C
from decadic.agents.runtime import AgentRuntime


def _obs(x: float) -> dict:
    return {
        "proprioception": {
            "position": [x, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "idle",
        },
        "events": [],
        "world_state": {},
    }


@pytest.fixture()
def rt(monkeypatch, tmp_path):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_PREFETCH_OVERLOAD_POLICY", "drop_oldest")
    runtime = AgentRuntime("drop-tombstone-test")
    # Keep queued items in place: workers must not drain the queue during
    # this test, so the overload branch is guaranteed to trigger.
    monkeypatch.setattr(runtime, "_ensure_perception_workers", lambda: None)
    return runtime


def test_drop_oldest_leaves_tombstone_and_drain_advances(rt):
    async def scenario() -> None:
        maxsize = rt._perception_queue.maxsize
        assert maxsize > 0, "test requires a bounded prefetch queue"

        # Fill the queue exactly to capacity (frames 1..maxsize).
        for i in range(maxsize):
            await rt._enqueue_perception_observation(_obs(float(i)))
        assert rt._perception_queue.full()
        assert rt._perception_next_commit == 1

        # One more observation forces drop_oldest to evict frame 1 — the
        # exact frame the in-order drain is waiting for.
        await rt._enqueue_perception_observation(_obs(999.0))

        assert rt.stage_pipeline.failed_count >= 1

        # The tombstone must let the drain advance past the dropped frame;
        # without the fix next_commit stays pinned at 1 forever.
        assert rt._perception_next_commit >= 2, (
            "fold drain did not advance past the dropped frame — "
            "drop_oldest left a hole with no tombstone"
        )
        # And the dropped frame must not linger as a pending ready entry.
        assert 1 not in rt._perception_ready

    asyncio.run(scenario())
