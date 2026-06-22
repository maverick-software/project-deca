"""Provisioning: AgentRuntime.give_resource + the /agent/{id}/give route.

The explicit "admin" credit mirrors a real drink/meal (reservoir + pleasure)
minus the body. The normal "direct" route now queues a body-side visual delivery
so the agent sees the object before the usual food/water event provides relief.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient


class _FakeEnvironment:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def is_running(self) -> bool:
        return True

    def status(self) -> dict:
        return {"agent_id": self.agent_id}


def _runtime(tmp_path, monkeypatch, agent_id="give"):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.delenv("DECADIC_VIABILITY_MODE", raising=False)
    from decadic.agents.runtime import AgentRuntime

    return AgentRuntime(agent_id)


# --- Admin credit (runtime) -------------------------------------------------


def test_give_water_credits_hydration_and_pleasure(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        assert rt.viability_mode == "metabolic"
        rt.homeostasis.hydration = 40.0
        rt.state_bus.pleasure_scalar = 0.0

        out = await rt.give_resource("water")
        assert rt.homeostasis.hydration > 40.0
        assert rt.state_bus.pleasure_scalar > 0.0
        assert out["resource"] == "water"
        assert out["amount"] > 0.0
        assert rt.viability.value == rt.homeostasis.viability
        await rt.stop()

    asyncio.run(go())


def test_give_food_credits_energy(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.homeostasis.energy = 40.0
        await rt.give_resource("food")
        assert rt.homeostasis.energy > 40.0
        await rt.stop()

    asyncio.run(go())


def test_give_resource_amount_override(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.homeostasis.hydration = 50.0
        await rt.give_resource("water", amount=5.0)
        assert rt.homeostasis.hydration == pytest.approx(55.0, abs=1e-6)
        await rt.stop()

    asyncio.run(go())


def test_give_resource_immortal_registers_pleasure_without_credit(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.configure(viability_mode="immortal")
        rt.state_bus.pleasure_scalar = 0.0
        await rt.give_resource("water")
        # Reservoirs are pinned full in immortal mode, but the affect still fires.
        assert rt.homeostasis.hydration == 100.0
        assert rt.state_bus.pleasure_scalar > 0.0
        await rt.stop()

    asyncio.run(go())


def test_give_resource_rejects_unknown_resource(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            await rt.give_resource("soda")
        await rt.stop()

    asyncio.run(go())


# --- Endpoint validation ----------------------------------------------------


def test_give_endpoint_admin_ok(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=admin")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "water_admin"
        assert "hydration" in body


def test_give_endpoint_direct_queues_visual_delivery(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        client.post(f"/agent/{aid}/pause")  # keep cycle actions out of the queue
        api_app.state.environment = _FakeEnvironment(aid)

        r = client.post(f"/agent/{aid}/give?resource=food&mode=direct")
        assert r.status_code == 200
        assert r.json()["status"] == "food_direct_visual_queued"

        with client.websocket_connect(f"/agent/{aid}/cycle") as ws:
            msg = ws.receive_json()
        assert msg["type"] == "body_command"
        assert msg["command"] == "give_food_direct_visual"


def test_give_endpoint_unknown_agent_404(api_app):
    with TestClient(api_app) as client:
        r = client.post("/agent/nope/give?resource=water&mode=direct")
        assert r.status_code == 404


def test_give_endpoint_bad_resource_400(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=lava&mode=direct")
        assert r.status_code == 400


def test_give_endpoint_bad_mode_400(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=teleport")
        assert r.status_code == 400


def test_give_endpoint_direct_without_body_409(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=direct")
        assert r.status_code == 409


def test_give_endpoint_near_without_body_409(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=near")
        assert r.status_code == 409
