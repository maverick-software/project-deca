from fastapi.testclient import TestClient


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


def test_rest_create_state_delete(api_app):
    with TestClient(api_app) as client:
        r = client.post("/agent")
        assert r.status_code == 200
        aid = r.json()["agent_id"]
        st = client.get(f"/agent/{aid}/state")
        assert st.status_code == 200
        assert st.json()["payload"]["viability"]["value"] == 100.0
        d = client.delete(f"/agent/{aid}")
        assert d.status_code == 200


def test_websocket_streams_actions(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            msg = ws.receive_json()
            assert msg["action"]["type"] == "move"
            assert "predicted_outcome" in msg


def test_fast_path_collision_lowers_viability(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(
                {
                    **_minimal_obs(),
                    "events": [{"type": "collision", "intensity": 0.9}],
                }
            )
            _ = ws.receive_json()
        metrics = client.get(f"/agent/{aid}/metrics").json()
        assert metrics["metrics"]["fast_path_hits"] >= 1
        assert metrics["metrics"]["viability"] < 100.0


def test_checkpoint_roundtrip(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            _drain_ws(ws)
            ws.send_json(_minimal_obs())
            _ = ws.receive_json()
        client.post(f"/agent/{aid}/checkpoint")
        snap_before = client.get(f"/agent/{aid}/state").json()
        client.post(f"/agent/{aid}/restore")
        snap_after = client.get(f"/agent/{aid}/state").json()
        assert snap_before["payload"]["state_bus"]["cycle_index"] == snap_after["payload"][
            "state_bus"
        ]["cycle_index"]
