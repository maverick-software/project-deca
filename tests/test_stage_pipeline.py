"""Serial cognition + lossless prefetch contracts."""

import asyncio

import pytest

from decadic import config as C
from decadic.cycle.stage_pipeline import (
    DecadicCommitArbiter,
    DecadicSession,
    DecadicStagePipelineSupervisor,
)


def test_prefetch_queue_defaults_and_clamp(monkeypatch):
    monkeypatch.delenv("DECADIC_PREFETCH_QUEUE_MAX_FRAMES", raising=False)
    assert C.prefetch_queue_max_frames(10) == 32
    assert C.prefetch_queue_max_frames(16) == 48
    monkeypatch.setenv("DECADIC_PREFETCH_QUEUE_MAX_FRAMES", "200")
    assert C.prefetch_queue_max_frames(16) == 128
    monkeypatch.setenv("DECADIC_PREFETCH_QUEUE_MAX_FRAMES", "8")
    assert C.prefetch_queue_max_frames(10) == 32


def test_prefetch_policy_defaults(monkeypatch):
    monkeypatch.delenv("DECADIC_PREFETCH_OVERLOAD_POLICY", raising=False)
    monkeypatch.delenv("DECADIC_READY_COALESCE_POLICY", raising=False)
    assert C.prefetch_overload_policy() == "block"
    assert C.ready_coalesce_policy() == "freshest"
    monkeypatch.setenv("DECADIC_PREFETCH_OVERLOAD_POLICY", "drop_oldest")
    monkeypatch.setenv("DECADIC_READY_COALESCE_POLICY", "oldest")
    assert C.prefetch_overload_policy() == "drop_oldest"
    assert C.ready_coalesce_policy() == "oldest"


def test_session_snapshot_is_immutable_and_serializable():
    obs = {"timestamp": "t0", "vision": {"data": "abc"}, "items": [{"x": 1}]}
    sess = DecadicSession.create(
        frame_seq=1,
        observation=obs,
        snapshots={"state_bus_version": 2, "semantic_label": "forbidden"},
    )
    obs["vision"]["data"] = "changed"
    assert sess.observation_snapshot["vision"]["data"] == "abc"
    with pytest.raises(TypeError):
        sess.observation_snapshot["vision"]["data"] = "mutate"
    payload = sess.to_dict()
    assert payload["snapshots"]["state_bus_version"] == 2
    assert "semantic_label" not in payload["snapshots"]


def test_commit_arbiter_is_fifo_with_urgent_preemption():
    arbiter = DecadicCommitArbiter()
    old = DecadicSession.create(frame_seq=1, observation={})
    young = DecadicSession.create(frame_seq=2, observation={})
    selected, reason = arbiter.select([young, old])
    assert selected is old
    assert reason == "fifo"

    urgent = DecadicSession.create(frame_seq=3, observation={"events": [{"kind": "impact"}]})
    selected, reason = arbiter.select([young, old, urgent])
    assert selected is urgent
    assert reason == "urgent"


def test_prefetch_supervisor_folds_before_deep_processing():
    async def run():
        supervisor = DecadicStagePipelineSupervisor(capacity=4)
        first = await supervisor.enqueue_observation({"timestamp": "1"})
        second = await supervisor.enqueue_observation({"timestamp": "2"})
        await supervisor.mark_prefetched(first.frame_seq, elapsed_s=0.01)
        await supervisor.mark_folded(first.frame_seq, elapsed_s=0.02)
        await supervisor.mark_prefetched(second.frame_seq, elapsed_s=0.01)
        await supervisor.mark_folded(second.frame_seq, elapsed_s=0.02)

        metrics = supervisor.metrics()
        assert metrics["frames_received"] == 2
        assert metrics["frames_prefetched"] == 2
        assert metrics["frames_folded"] == 2
        assert metrics["ready_sessions"] == 2
        assert metrics["information_loss"] == 0

        selected, bundle = await supervisor.pop_commit_candidate()
        assert selected is not None and bundle is not None
        assert selected.frame_seq == 1
        await supervisor.mark_committed(selected.session_id, action_type="noop")
        metrics = supervisor.metrics()
        assert metrics["frames_deep_processed"] == 1
        assert metrics["committed_sessions"] == 1
        assert metrics["selected_session"]["frame_seq"] == 1

    asyncio.run(run())


def test_prefetch_supervisor_coalesces_without_information_loss():
    async def run():
        supervisor = DecadicStagePipelineSupervisor(capacity=1)
        for i in range(3):
            sess = await supervisor.enqueue_observation({"timestamp": str(i)})
            await supervisor.mark_prefetched(sess.frame_seq)
            await supervisor.mark_folded(sess.frame_seq)
        selected, _bundle = await supervisor.pop_commit_candidate()
        assert selected is not None
        metrics = supervisor.metrics()
        assert metrics["frames_received"] == 3
        assert metrics["frames_folded"] == 3
        assert metrics["coalesced_sessions"] == 2
        assert metrics["information_loss"] == 0
        assert selected.frame_seq == 3

    asyncio.run(run())


def test_prefetch_supervisor_oldest_policy_coalesces_newer_frames():
    async def run():
        supervisor = DecadicStagePipelineSupervisor(capacity=1, coalesce_policy="oldest")
        for i in range(3):
            sess = await supervisor.enqueue_observation({"timestamp": str(i)})
            await supervisor.mark_prefetched(sess.frame_seq)
            await supervisor.mark_folded(sess.frame_seq)
        selected, _bundle = await supervisor.pop_commit_candidate()
        assert selected is not None
        metrics = supervisor.metrics()
        assert metrics["coalesced_sessions"] == 2
        assert metrics["information_loss"] == 0
        assert selected.frame_seq == 1
        assert metrics["ready_coalesce_policy"] == "oldest"

    asyncio.run(run())


def test_prefetch_supervisor_records_backpressure_without_loss():
    async def run():
        supervisor = DecadicStagePipelineSupervisor(capacity=2)
        await supervisor.record_prefetch_backpressure(elapsed_s=0.025)
        metrics = supervisor.metrics()
        assert metrics["prefetch_backpressure_events"] == 1
        assert metrics["prefetch_backpressure_ms"] == pytest.approx(25.0)
        assert metrics["information_loss"] == 0

    asyncio.run(run())


def test_stage_pipeline_metrics_are_lightweight_and_debug_preserves_full_snapshot():
    async def run():
        supervisor = DecadicStagePipelineSupervisor(capacity=2)
        sess = await supervisor.enqueue_observation(
            {"timestamp": "1"},
            snapshots={"state_bus_version": 12},
        )
        await supervisor.mark_prefetched(sess.frame_seq)
        await supervisor.mark_folded(sess.frame_seq)
        selected, _bundle = await supervisor.pop_commit_candidate()
        assert selected is not None
        await supervisor.mark_committed(selected.session_id, action_type="noop")

        recent = supervisor.metrics()["recent_sessions"]
        assert recent
        assert "snapshots" not in recent[0]
        full = supervisor.debug_sessions()
        assert full
        assert full[0]["snapshots"]["state_bus_version"] == 12

    asyncio.run(run())


def test_runtime_serial_prefetch_folds_and_deep_processes(monkeypatch, tmp_path):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_PROCESSING_MODE", "stage_pipeline")
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.01")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("runtime-serial-prefetch")

    async def run():
        try:
            rt.ensure_cycle_worker()
            await rt.handle_observation_dict(
                {
                    "timestamp": "t0",
                    "proprioception": {"position": [0.0, 0.0, 0.0]},
                    "events": [],
                }
            )
            deadline = asyncio.get_running_loop().time() + 1.0
            while asyncio.get_running_loop().time() < deadline:
                if int(rt.metrics.get("frames_deep_processed", 0)) >= 1:
                    break
                await asyncio.sleep(0.02)
            assert rt.processing_mode == "serial_prefetch"
            assert int(rt.metrics["frames_received"]) >= 1
            assert int(rt.metrics["frames_folded"]) >= 1
            assert int(rt.metrics["frames_deep_processed"]) >= 1
            assert int(rt.metrics["information_loss"]) == 0
            assert int(rt.metrics["cycles_completed"]) >= 1
        finally:
            await rt.stop()

    asyncio.run(run())
