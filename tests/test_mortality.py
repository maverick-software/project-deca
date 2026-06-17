"""Mortality lifecycle: death freeze + tombstone, revive, reincarnate (Part D)."""

import asyncio
import json

from decadic.agents.runtime import AgentRuntime


def _runtime(tmp_path, monkeypatch, agent_id="mortal"):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    return AgentRuntime(agent_id)


def test_death_on_zero_writes_tombstone(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    assert rt.status == "alive"
    rt.state_bus.cycle_index = 7
    # Any reservoir bottoming out ends viability (viability = min of reservoirs).
    rt.homeostasis.integrity = 0.0
    rt._check_death()

    assert rt.status == "dead"
    assert rt.died_at_cycle == 7
    tomb = tmp_path / "agent_mortal_tombstone.json"
    assert tomb.is_file()
    payload = json.loads(tomb.read_text(encoding="utf-8"))
    assert payload["status"] == "dead"
    assert payload["tombstone"] is True
    # a death event is queued for any connected environment
    msg = rt.out_queue.get_nowait()
    assert msg["type"] == "death"


def test_revive_restores_same_mind(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.homeostasis.energy = 0.0
        rt._check_death()
        assert rt.status == "dead"

        rt.revive(42.0)
        assert rt.status == "alive"
        assert rt.died_at_cycle is None
        # Revive restores every reservoir, so the derived viability matches.
        assert rt.viability.value == 42.0
        assert rt.homeostasis.hydration == 42.0
        assert rt.homeostasis.energy == 42.0
        assert rt.homeostasis.integrity == 42.0
        await rt.stop()

    asyncio.run(go())


def test_reincarnate_via_reset_from_dead(tmp_path, monkeypatch):
    async def go():
        rt = _runtime(tmp_path, monkeypatch)
        rt.homeostasis.hydration = 0.0
        rt._check_death()
        assert rt.status == "dead"

        await rt.reset()
        assert rt.status == "alive"
        assert rt.died_at_cycle is None
        assert rt.viability.value == 100.0
        assert rt.homeostasis.viability == 100.0
        await rt.stop()

    asyncio.run(go())


def test_no_double_death(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    rt.homeostasis.integrity = 0.0
    rt._check_death()
    first_cycle = rt.died_at_cycle
    rt.state_bus.cycle_index = 999
    rt._check_death()  # already dead → no-op
    assert rt.died_at_cycle == first_cycle
