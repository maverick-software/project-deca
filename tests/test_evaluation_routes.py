import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_eval_scenarios_available_without_agent(api_app):
    with TestClient(api_app) as client:
        resp = client.get("/eval/scenarios")
        assert resp.status_code == 200
        body = resp.json()
        ids = {s["scenario"] for s in body["scenarios"]}
        assert "health_smoke" in ids
        assert client.get("/eval/scenarios/health_smoke").json()["scenario"] == "health_smoke"


def test_eval_reports_empty_and_fetch(api_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    report = {
        "scenario": "health_smoke",
        "status": "pass",
        "agent_id": "a1",
        "failures": [],
        "samples_path": "reports/x.jsonl",
    }
    (reports / "training_eval_health_smoke_20260623_120000.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with TestClient(api_app) as client:
        listed = client.get("/eval/reports").json()["reports"]
        assert listed[0]["report_id"] == "training_eval_health_smoke_20260623_120000"
        fetched = client.get("/eval/reports/training_eval_health_smoke_20260623_120000").json()
        assert fetched["scenario"] == "health_smoke"


def test_eval_start_refuses_second_job_and_stop(api_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(api_app) as client:
        first = client.post(
            "/eval/start",
            json={
                "scenario": "health_smoke",
                "cycles": 999999,
                "poll_interval_s": 1.0,
                "timeout_s": 60.0,
            },
        )
        assert first.status_code == 200
        assert first.json()["state"] in {"starting", "running"}
        second = client.post("/eval/start", json={"scenario": "health_smoke", "cycles": 1})
        assert second.status_code == 409
        stopped = client.post("/eval/stop")
        assert stopped.status_code == 200
        assert stopped.json()["state"] in {"cancelled", "stopping", "idle"}


def test_eval_status_reports_no_body_warning(api_app):
    with TestClient(api_app) as client:
        status = client.get("/eval/status").json()
        assert status["state"] == "idle"
        assert status["body_connected"] is False
        assert "body" in status["body_warning"].lower()

