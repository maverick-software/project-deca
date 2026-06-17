"""Admin provisioning: AgentRuntime.give_resource + the /agent/{id}/give route.

The "direct" credit mirrors a real drink/meal (reservoir + pleasure) minus the
body; the "near" mode just queues a body command. These cover the credit math,
mode guards, and endpoint validation.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient


def _runtime(tmp_path, monkeypatch, agent_id="give"):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.delenv("DECADIC_VIABILITY_MODE", raising=False)
    from decadic.agents.runtime import AgentRuntime

    return AgentRuntime(agent_id)


# --- Direct credit (runtime) ------------------------------------------------


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


def test_give_endpoint_direct_ok(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=direct")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "water_direct"
        assert "hydration" in body


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


def test_give_endpoint_near_without_body_409(api_app):
    with TestClient(api_app) as client:
        aid = client.post("/agent").json()["agent_id"]
        r = client.post(f"/agent/{aid}/give?resource=water&mode=near")
        assert r.status_code == 409
