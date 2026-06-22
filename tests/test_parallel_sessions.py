"""Batched no-grad encode parity vs single encode (logical-layer Part C)."""

import asyncio

import torch

from decadic import config as C
from decadic.cycle.neural_pipeline import encode_observations, pool_fused


class _StubEncoders:
    """Deterministic per-observation encoder: maps position[0] to a 2-vector."""

    def __call__(self, obs):
        v = float(obs["proprioception"]["position"][0])
        return torch.tensor([[v, v + 1.0]], dtype=torch.float32)


class _StubBundle:
    def __init__(self):
        self.encoders = _StubEncoders()


def _obs(x):
    return {"proprioception": {"position": [x, 0.0, 0.0]}}


def test_batched_encode_shape_and_parity():
    bundle = _StubBundle()
    obs = [_obs(0.0), _obs(1.0), _obs(2.0)]
    batch = encode_observations(bundle, obs)
    assert batch is not None
    assert batch.shape == (3, 2)
    # each batched row equals the single-encode result for that observation
    for i, o in enumerate(obs):
        single = bundle.encoders(o)[0]
        assert torch.allclose(batch[i], single)


def test_empty_and_none_observations():
    bundle = _StubBundle()
    assert encode_observations(bundle, []) is None
    assert encode_observations(bundle, [None]) is None
    one = encode_observations(bundle, [None, _obs(5.0)])
    assert one is not None and one.shape == (1, 2)


def test_pool_fused_k1_parity():
    latest = torch.tensor([[3.0, 7.0]])
    # no older frames → pooled vector is exactly the latest encode
    assert torch.equal(pool_fused(None, latest, gamma=0.7), latest)
    empty = torch.zeros((0, 2))
    assert torch.equal(pool_fused(empty, latest, gamma=0.7), latest)


def test_pool_fused_recency_weights_favor_latest():
    gamma = 0.5
    older = torch.tensor([[1.0, 0.0], [2.0, 0.0]])  # oldest first
    latest = torch.tensor([[4.0, 0.0]])
    pooled = pool_fused(older, latest, gamma)
    # weights: oldest gamma^2=0.25, middle gamma^1=0.5, latest 1.0 → total 1.75
    expected = (0.25 * 1.0 + 0.5 * 2.0 + 1.0 * 4.0) / 1.75
    assert pooled.shape == (1, 2)
    assert abs(float(pooled[0, 0]) - expected) < 1e-6
    # the latest frame carries the largest weight
    assert 1.0 / 1.75 > 0.5 / 1.75 > 0.25 / 1.75


def test_pool_fused_gradient_flows_through_latest_only():
    older = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # no-grad encodes
    src = torch.tensor([[3.0, 3.0]], requires_grad=True)
    latest = src * 2.0
    pooled = pool_fused(older, latest, gamma=0.7)
    pooled.sum().backward()
    assert src.grad is not None
    # grad through latest = 2.0 (latest scale) * w_latest, w_latest = 1/(1+0.7+0.49)
    w_latest = 1.0 / (1.0 + 0.7 + 0.49)
    assert torch.allclose(src.grad, torch.full((1, 2), 2.0 * w_latest), atol=1e-6)


def test_default_perceptual_pipeline_capacity_is_ten(monkeypatch, tmp_path):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.delenv("DECADIC_PARALLEL_SESSIONS", raising=False)
    monkeypatch.delenv("DECADIC_PERCEPTUAL_PROCESSING_MODE", raising=False)
    monkeypatch.delenv("DECADIC_PERSISTENT_PARALLEL_PERCEPTION", raising=False)
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("pipeline-defaults")
    assert C.DEFAULT_PARALLEL_SESSIONS == 10
    assert rt.parallel_sessions == 10
    assert rt.perceptual_processing_mode == C.PERCEPTUAL_PROCESSING_PERSISTENT
    assert rt.capacity_config()["perceptual_processing_mode"] == "persistent_parallel"


def test_perception_ready_buffer_commits_in_sequence(monkeypatch, tmp_path):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("pipeline-order")
    committed: list[int] = []

    def fake_commit(obs):
        committed.append(int(obs["seq"]))

    rt._commit_perception_observation_locked = fake_commit  # type: ignore[method-assign]

    async def run():
        rt._perception_ready[2] = ({"seq": 2}, 1.0, 1.0)
        await rt._drain_ready_perception()
        assert committed == []
        rt._perception_ready[1] = ({"seq": 1}, 1.0, 1.0)
        await rt._drain_ready_perception()

    asyncio.run(run())
    assert committed == [1, 2]
    assert rt.metrics["frames_committed"] == 2


def test_configure_perceptual_processing_mode(monkeypatch, tmp_path):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("pipeline-config")
    cfg = rt.configure(perceptual_processing_mode="batching_observations")
    assert cfg["perceptual_processing_mode"] == "batching_observations"
    assert rt.metrics["batching_fallback"] is True
    cfg = rt.configure(perceptual_processing_mode="persistent_parallel", parallel_sessions=4)
    assert cfg["perceptual_processing_mode"] == "persistent_parallel"
    assert cfg["parallel_sessions"] == 4
