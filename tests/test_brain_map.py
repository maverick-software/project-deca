"""Brain Map topology export, stage activations, and preset switching."""

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_app_neural(tmp_path, monkeypatch):
    """Like api_app but with the real (tiny) neural stack on CPU."""
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


def _tiny_stack():
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(neural_config_from_env("tiny"))


def test_brain_topology_shape():
    from decadic.nn.brain_map import MAX_FIBERS_TOTAL, brain_topology

    stack = _tiny_stack()
    topo = brain_topology(stack, preset="tiny")

    layers = {layer["id"]: layer for layer in topo["layers"]}
    assert len(layers) >= 15
    assert topo["totals"]["params"] == sum(p.numel() for p in stack.parameters())
    assert topo["totals"]["connections"] == sum(
        p.numel() for p in stack.parameters() if p.dim() >= 2
    )
    assert topo["totals"]["neurons"] >= sum(layer["units"] for layer in topo["layers"])
    assert topo["totals"]["preset"] == "tiny"
    assert topo["totals"]["d_model"] == 96

    total_fibers = 0
    for edge in topo["edges"]:
        assert edge["src"] in layers and edge["dst"] in layers
        assert edge["weight_count"] > 0
        assert edge["w_rms"] >= 0
        for f in edge["fibers"]:
            assert 0 <= f["di"] < layers[edge["dst"]]["units"]
            assert 0 <= f["si"] < layers[edge["src"]]["units"]
        total_fibers += len(edge["fibers"])
    assert 0 < total_fibers <= MAX_FIBERS_TOTAL + len(topo["edges"]) * 8


def test_stage_activations_in_forward():
    import torch

    stack = _tiny_stack()
    out = stack(torch.zeros(1, stack.cfg.d_model), torch.zeros(1, 4))
    for m in out["stage_metrics"]:
        assert len(m["activations"]) == 32
        assert all(v >= 0 for v in m["activations"])


def test_topology_endpoint_stub_404(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        assert client.get(f"/agent/{aid}/brain/topology").status_code == 404
        assert client.get("/agent/nope/brain/topology").status_code == 404


def test_topology_endpoint_and_preset_switch(api_app_neural):
    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]

        r = client.get(f"/agent/{aid}/brain/topology")
        assert r.status_code == 200
        topo = r.json()
        assert topo["totals"]["preset"] == "tiny"
        assert topo["totals"]["d_model"] == 96

        agents = client.get("/agents").json()["agents"]
        assert agents[0]["preset"] == "tiny"

        # Invalid preset rejected
        assert client.post(f"/agent/{aid}/preset?preset=galactic").status_code == 422

        # Switch to medium: rebuilt brain, larger topology, metrics updated
        r = client.post(f"/agent/{aid}/preset?preset=medium")
        assert r.status_code == 200
        assert r.json()["preset"] == "medium"

        topo2 = client.get(f"/agent/{aid}/brain/topology").json()
        assert topo2["totals"]["d_model"] == 256
        assert topo2["totals"]["params"] > topo["totals"]["params"]

        metrics = client.get(f"/agent/{aid}/metrics").json()["metrics"]
        assert metrics["preset"] == "medium"
        assert int(metrics["cycles_completed"]) >= 0  # reset happened, still alive


def test_activations_reach_cycle_trace(api_app_neural):
    obs = {
        "timestamp": "2026-06-12T00:00:00Z",
        "proprioception": {"position": [0, 0, 1.4]},
        "events": [],
        "world_state": {"nearby_entities": [], "agent_inventory": []},
    }
    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            ws.send_json(obs)
            ws.receive_json()

        trace = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            payload = client.get(f"/agent/{aid}/state").json()["payload"]
            trace = payload.get("last_cycle_trace")
            if trace:
                break
            time.sleep(0.05)
        assert trace, "no cycle trace produced"
        staged = {s["stage"]: s for s in trace["stages"]}
        with_acts = [s for s in staged.values() if "activations" in s.get("payload", {})]
        assert len(with_acts) >= 6
        assert all(len(s["payload"]["activations"]) == 32 for s in with_acts)
