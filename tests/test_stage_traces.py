"""Per-stage instrumentation: stack stage_metrics and /state last_cycle_trace."""

import math
import time

import pytest
from fastapi.testclient import TestClient


def test_stack_stage_metrics(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.frozen_encoders import CLIP_POOL_DIM, WHISPER_POOL_DIM
    from decadic.nn.neural_stack import NeuralCognitiveStack

    cfg = neural_config_from_env()
    stack = NeuralCognitiveStack(cfg)
    fused = torch.zeros(1, CLIP_POOL_DIM + WHISPER_POOL_DIM + cfg.proprio_emb)
    z0 = stack.ingress(fused)
    out = stack(z0, torch.zeros(1, 4))

    sm = out["stage_metrics"]
    assert [m["stage"] for m in sm] == list(range(2, 10))
    for m in sm:
        assert m["timing_ms"] >= 0.0
        assert math.isfinite(m["activity"])
    # Every measured block should account for some time overall.
    assert sum(m["timing_ms"] for m in sm) > 0.0


def test_state_exposes_last_cycle_trace(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Cycle interval in tests is 0.02s; give the worker a few cycles.
        deadline = time.time() + 3.0
        trace = None
        while time.time() < deadline:
            payload = client.get(f"/agent/{aid}/state").json()["payload"]
            trace = payload.get("last_cycle_trace")
            if trace:
                break
            time.sleep(0.05)

        assert trace, "no cycle trace appeared"
        assert trace["cycle"] >= 1
        stages = trace["stages"]
        assert [s["stage"] for s in stages] == list(range(1, 11))
        for s in stages:
            assert "name" in s
            assert float(s["payload"]["timing_ms"]) >= 0.0

        # Reset wipes the trace.
        client.post(f"/agent/{aid}/reset")
        payload = client.get(f"/agent/{aid}/state").json()["payload"]
        trace_after = payload.get("last_cycle_trace")
        assert trace_after is None or trace_after["cycle"] <= 1
