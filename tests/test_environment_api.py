"""/environment REST routes: status, guards, and single-slot rejection."""

from fastapi.testclient import TestClient


def test_environment_status_when_stopped(api_app):
    with TestClient(api_app) as client:
        r = client.get("/environment")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "stopped"
        assert body["running"] is False
        assert body["agent_id"] is None
        # Available elements are advertised for the scenario builder.
        for el in ("house", "food", "water", "bear", "ball", "obstacles"):
            assert el in body["available_elements"]


def test_pause_without_running_is_conflict(api_app):
    with TestClient(api_app) as client:
        assert client.post("/environment/pause").status_code == 409
        assert client.post("/environment/resume").status_code == 409


def test_start_with_empty_elements_is_conflict(api_app):
    with TestClient(api_app) as client:
        r = client.post("/environment", json={"elements": []})
        assert r.status_code == 409
        # Only invalid elements is likewise rejected (nothing spawned).
        r2 = client.post("/environment", json={"elements": ["volcano"]})
        assert r2.status_code == 409


def test_start_rejects_unknown_neural_preset(api_app):
    with TestClient(api_app) as client:
        r = client.post("/environment", json={"elements": ["house"], "preset": "not-real"})
        assert r.status_code == 422


def test_stop_when_idle_is_ok(api_app):
    with TestClient(api_app) as client:
        r = client.post("/environment/stop")
        assert r.status_code == 200
        assert r.json()["state"] == "stopped"
