"""CurriculumSupervisor behaviour with a fake agent/registry (no torch, no body).

Verifies the supervisor's only effects are read / configure / world / checkpoint:
phase config application, satisfier placement on a cadence, promotion at an open
gate, demotion + revive on death, and a checkpoint written at a phase boundary.
"""

import asyncio
import json

import pytest

from decadic.curriculum.supervisor import CurriculumError, CurriculumSupervisor


class _Viability:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value


class FakeAgent:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.metrics: dict[str, float] = {
            "forward_model_error": 0.5,
            "tactile_pred_error": 0.5,
            "rom_mean": 0.0,
            "fall_rate": 0.0,
            "gait_regularity": 0.0,
            "distance_traveled": 0.0,
            "consume_events": 0.0,
        }
        self.viability = _Viability(100.0)
        self.status = "alive"
        self._has_body = True
        self.configure_calls: list[dict] = []
        self.body_commands: list[str] = []
        self.revived = 0
        self.saved = 0

    def has_body(self) -> bool:
        return self._has_body

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        return {"ok": True}

    def queue_body_command(self, command: str) -> bool:
        self.body_commands.append(command)
        return True

    def checkpoint_payload(self) -> dict:
        return {"agent": "fake", "cycle": 1}

    def save_brain(self, backups_dir) -> str:
        self.saved += 1
        return "fake_brain.pt"

    def revive(self) -> None:
        self.revived += 1
        self.status = "alive"


class FakeRegistry:
    def __init__(self, agent: FakeAgent) -> None:
        self._agent = agent

    def get(self, agent_id: str):
        return self._agent if agent_id == "A" else None


def _sup(tmp_path, agent, monkeypatch, poll="0.01"):
    monkeypatch.setenv("DECADIC_CURRICULUM_POLL_S", poll)
    return CurriculumSupervisor(
        FakeRegistry(agent), backups_dir=tmp_path / "backups", log_dir=tmp_path / "logs"
    )


def test_start_applies_phase0_and_binds(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        st = await sup.start("A")
        assert st["state"] == "running"
        assert st["phase_index"] == 0
        assert st["agent_id"] == "A"
        # Phase 0 pins immortal + low babble.
        assert agent.configure_calls[0]["viability_mode"] == "immortal"
        assert "motor_babble_sigma" in agent.configure_calls[0]
        # A second start is rejected while one runs.
        with pytest.raises(CurriculumError):
            await sup.start("A")
        await sup.stop()

    asyncio.run(go())


def test_start_unknown_agent_raises(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        with pytest.raises(CurriculumError):
            await sup.start("MISSING")

    asyncio.run(go())


def test_loop_promotes_when_gate_open(tmp_path, monkeypatch):
    agent = FakeAgent()
    # Make phase 0's gate trivially satisfiable (low PE + some ROM), fast dwell.
    agent.metrics.update(
        forward_model_error=0.01, tactile_pred_error=0.01, rom_mean=0.1
    )
    overrides = {
        "Self-modeling": {"min_dwell_s": 0, "min_samples": 2},
        "Postural control": {"min_dwell_s": 0, "min_samples": 2},
    }
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        await sup.start("A", overrides=overrides)
        # Phase 0 gate opens; phase 1 needs rom_mean>=0.25 (we have 0.1) -> stall at 1.
        for _ in range(80):
            await asyncio.sleep(0.01)
            if sup.status()["phase_index"] == 1:
                break
        st = sup.status()
        assert st["phase_index"] == 1
        # Promotion applied phase 1's metabolic config and checkpointed at boundary.
        assert any(c.get("viability_mode") == "metabolic" for c in agent.configure_calls)
        assert agent.saved >= 1
        await sup.stop()

    asyncio.run(go())


def test_demote_and_revive_on_death(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        await sup.start("A")
        # Force into phase 2, then kill: demote_on_death steps back to phase 1.
        await sup.set_phase(2)
        assert sup.status()["phase_index"] == 2
        agent.status = "dead"
        for _ in range(80):
            await asyncio.sleep(0.01)
            if sup.status()["phase_index"] < 2:
                break
        assert sup.status()["phase_index"] == 1
        assert agent.revived >= 1
        assert agent.status == "alive"
        await sup.stop()

    asyncio.run(go())


def test_satisfier_places_on_cadence(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        await sup.start("A")
        # Jump to the locomotion phase (satisfier enabled, food + water).
        await sup.set_phase(2)
        for _ in range(60):
            await asyncio.sleep(0.01)
            if agent.body_commands:
                break
        assert "give_food_near" in agent.body_commands
        assert "give_water_near" in agent.body_commands
        await sup.stop()

    asyncio.run(go())


def test_checkpoint_written_at_boundary(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        await sup.start("A")
        await sup._checkpoint(agent, "test")
        await sup.stop()

    asyncio.run(go())
    ckpt = tmp_path / "backups" / "agent_A_checkpoint.json"
    assert ckpt.is_file()
    assert json.loads(ckpt.read_text())["agent"] == "fake"


def test_pause_resume_stop_lifecycle(tmp_path, monkeypatch):
    agent = FakeAgent()
    sup = _sup(tmp_path, agent, monkeypatch)

    async def go():
        await sup.start("A")
        assert sup.pause()["paused"] is True
        assert sup.resume()["paused"] is False
        stopped = await sup.stop()
        assert stopped["state"] == "stopped"
        # Pause without a running curriculum errors.
        with pytest.raises(CurriculumError):
            sup.pause()

    asyncio.run(go())
