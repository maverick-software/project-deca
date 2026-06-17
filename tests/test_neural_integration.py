"""Integration tests: neural WebSocket metrics, checkpoint + brain files, memory similarity."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")


@pytest.fixture
def neural_api_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("DECADIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.02")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.api.app import create_app

    return create_app()


def _minimal_obs():
    return {
        "timestamp": "2026-05-07T12:00:00Z",
        "proprioception": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "idle",
        },
        "events": [],
    }


def _drain_ws(ws, cap: int = 64) -> None:
    for _ in range(cap):
        try:
            ws.receive_json(timeout=0.05)
        except Exception:
            break


def test_neural_ws_updates_metrics(neural_api_app):
    with TestClient(neural_api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            msg = ws.receive_json()
            assert msg["action"]["type"] == "motor"
            assert isinstance(msg["action"]["parameters"]["ctrl"], list)
            assert any(
                t.get("payload", {}).get("neural") is True for t in msg.get("trace", [])
            )


def test_neural_checkpoint_brain_file_and_restore_cycle(neural_api_app, tmp_path):
    backups = Path(tmp_path / "backups")
    with TestClient(neural_api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()
            time.sleep(0.15)

        ck = client.post(f"/agent/{aid}/checkpoint").json()
        assert ck.get("neural_brain") == f"agent_{aid}_brain.pt"
        brain_path = backups / ck["neural_brain"]
        assert brain_path.is_file()

        snap = client.get(f"/agent/{aid}/state").json()
        c_keep = snap["payload"]["state_bus"]["cycle_index"]

        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            for _ in range(4):
                ws.send_json(_minimal_obs())
                try:
                    ws.receive_json()
                except Exception:
                    pass
            time.sleep(0.2)

        snap2 = client.get(f"/agent/{aid}/state").json()
        assert snap2["payload"]["state_bus"]["cycle_index"] >= c_keep

        client.post(f"/agent/{aid}/restore")
        snap3 = client.get(f"/agent/{aid}/state").json()
        assert snap3["payload"]["state_bus"]["cycle_index"] == c_keep
        assert snap3["payload"]["neural_enabled"] is True


def test_memory_similar_after_two_cycles(neural_api_app):
    with TestClient(neural_api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()
            time.sleep(0.12)
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()
            time.sleep(0.12)

        r = client.get(f"/agent/{aid}/memory/similar", params={"top_k": 5}).json()
        assert r["agent_id"] == aid
        assert len(r["matches"]) >= 1
        assert "similarity" in r["matches"][0]
        assert "cycle_index" in r["matches"][0]


def test_recent_memory_includes_embedding(neural_api_app):
    with TestClient(neural_api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()
            time.sleep(0.12)

        mem = client.get(f"/agent/{aid}/memory", params={"limit": 5}).json()
        eps = mem["episodes"]
        assert eps
        assert "embedding" in eps[0]
        from decadic.memory.embeddings import EMBEDDING_DIM

        assert len(eps[0]["embedding"]) == EMBEDDING_DIM
