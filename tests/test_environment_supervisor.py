"""EnvironmentSupervisor lifecycle with a stubbed subprocess and fake registry."""

import asyncio

import pytest

from decadic.api.environment import (
    VALID_ELEMENTS,
    EnvironmentControlError,
    EnvironmentSupervisor,
    _self_port,
)


class FakeAgent:
    def __init__(self) -> None:
        self.paused = False
        self.body_commands: list[str] = []

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def queue_body_command(self, command: str) -> bool:
        self.body_commands.append(command)
        return True


class FakeRegistry:
    def __init__(self) -> None:
        self.agents: dict[str, FakeAgent] = {}
        self.created: list[str] = []
        self.created_presets: dict[str, str | None] = {}
        self.deleted: list[str] = []

    def create_agent(self, agent_id: str, preset: str | None = None) -> FakeAgent:
        agent = FakeAgent()
        self.agents[agent_id] = agent
        self.created.append(agent_id)
        self.created_presets[agent_id] = preset
        return agent

    def get(self, agent_id: str) -> FakeAgent | None:
        return self.agents.get(agent_id)

    async def delete_agent(self, agent_id: str) -> None:
        self.deleted.append(agent_id)
        self.agents.pop(agent_id, None)


class FakeProc:
    def __init__(self) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def _patch_spawn(monkeypatch, proc: FakeProc) -> None:
    async def fake_exec(*args, **kwargs):
        fh = kwargs.get("stdout")
        if hasattr(fh, "write"):
            fh.write("[body] fake start\n")
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def test_start_pause_resume_stop(tmp_path, monkeypatch):
    reg = FakeRegistry()
    proc = FakeProc()
    _patch_spawn(monkeypatch, proc)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        st = await sup.start(["house", "food", "bear", "nonsense"], vision=True, audio=False)
        # unknown element filtered, the rest preserved in order
        assert st["elements"] == ["house", "food", "bear"]
        assert st["state"] == "running"
        assert st["running"] is True
        assert st["pid"] == 4321
        assert st["agent_id"] in reg.created
        agent = reg.get(st["agent_id"])

        # A second start is rejected while one is live.
        with pytest.raises(EnvironmentControlError):
            await sup.start(["house"])

        paused = sup.pause()
        assert paused["state"] == "paused"
        assert agent.paused is True
        assert "pause" in agent.body_commands

        resumed = sup.resume()
        assert resumed["paused"] is False
        assert agent.paused is False
        assert "resume" in agent.body_commands

        stopped = await sup.stop()
        assert stopped["state"] == "stopped"
        assert stopped["running"] is False
        assert proc.terminated is True

    asyncio.run(go())


def test_delete_removes_bound_agent(tmp_path, monkeypatch):
    reg = FakeRegistry()
    proc = FakeProc()
    _patch_spawn(monkeypatch, proc)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        st = await sup.start(["house"])
        agent_id = st["agent_id"]
        result = await sup.delete()
        assert agent_id in reg.deleted
        assert result["agent_id"] is None
        assert result["state"] == "stopped"

    asyncio.run(go())


def test_start_with_no_valid_elements_raises(tmp_path, monkeypatch):
    reg = FakeRegistry()
    _patch_spawn(monkeypatch, FakeProc())
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        with pytest.raises(EnvironmentControlError):
            await sup.start(["volcano", "atlantis"])
        # No agent should have been created on a rejected start.
        assert reg.created == []

    asyncio.run(go())


def test_start_forwards_neural_preset_to_created_agent(tmp_path, monkeypatch):
    reg = FakeRegistry()
    _patch_spawn(monkeypatch, FakeProc())
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        st = await sup.start(["house"], preset="medium")
        assert reg.created_presets[st["agent_id"]] == "medium"

    asyncio.run(go())


def test_pause_without_running_raises(tmp_path):
    reg = FakeRegistry()
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)
    with pytest.raises(EnvironmentControlError):
        sup.pause()
    with pytest.raises(EnvironmentControlError):
        sup.resume()


def test_replace_supersedes_running_body_and_keeps_old_agent(tmp_path, monkeypatch):
    reg = FakeRegistry()
    # A fresh process for each spawn so the replacement looks live (the first
    # proc is "terminated" -> returncode 0 -> no longer running).
    procs = [FakeProc(), FakeProc()]
    spawned: list[FakeProc] = []

    async def fake_exec(*args, **kwargs):
        proc = procs[len(spawned)]
        spawned.append(proc)
        fh = kwargs.get("stdout")
        if hasattr(fh, "write"):
            fh.write("[body] fake start\n")
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        first = await sup.start(["house", "food"])
        first_agent = first["agent_id"]

        # Without replace a second start is still rejected.
        with pytest.raises(EnvironmentControlError):
            await sup.start(["water"])

        # With replace the running body is superseded by a new agent+body.
        second = await sup.start(["water"], replace=True)
        assert second["state"] == "running"
        assert second["running"] is True
        assert second["agent_id"] != first_agent
        assert procs[0].terminated is True  # old body stopped
        assert second["pid"] == procs[1].pid
        # The previous mind is kept (not deleted) - just bodiless now.
        assert first_agent not in reg.deleted
        assert first_agent in reg.agents

    asyncio.run(go())


def test_self_port_defaults_to_project_port(monkeypatch):
    monkeypatch.delenv("DECADIC_SELF_PORT", raising=False)
    monkeypatch.delenv("DECADIC_PORT", raising=False)
    assert _self_port() == 8765
    monkeypatch.setenv("DECADIC_SELF_PORT", "9001")
    assert _self_port() == 9001


def test_crowd_element_in_sync_with_adapter():
    """The supervisor's element set mirrors the adapter's SELECTABLE_ELEMENTS."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("adapter_sync", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adapter_sync"] = mod
    spec.loader.exec_module(mod)
    assert "crowd" in VALID_ELEMENTS
    assert set(mod.SELECTABLE_ELEMENTS) == set(VALID_ELEMENTS)


def test_status_exposes_braces_option_no_legacy_presets(tmp_path, monkeypatch):
    reg = FakeRegistry()
    proc = FakeProc()
    _patch_spawn(monkeypatch, proc)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        st = await sup.start(["crowd", "house"])
        assert st["elements"] == ["crowd", "house"]
        assert "crowd" in st["available_elements"]
        # Braces default off is surfaced in the options; the retired in-tab
        # preset map is gone (presets now live in the dedicated preset store).
        assert st["options"]["braces"] is False
        assert "available_presets" not in st

    asyncio.run(go())


def test_start_braces_on_passes_braces_flag(tmp_path, monkeypatch):
    reg = FakeRegistry()
    captured: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        captured.append([str(a) for a in args])
        fh = kwargs.get("stdout")
        if hasattr(fh, "write"):
            fh.write("[body] fake start\n")
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        st = await sup.start(["house"])
        assert "--braces" not in captured[-1]
        assert st["options"]["braces"] is False

        await sup.stop()
        await sup.start(["house"], braces=True)
        assert "--braces" in captured[-1]

    asyncio.run(go())


def test_crashed_state_when_process_exits_nonzero(tmp_path, monkeypatch):
    reg = FakeRegistry()
    proc = FakeProc()
    _patch_spawn(monkeypatch, proc)
    sup = EnvironmentSupervisor(reg, log_dir=tmp_path)

    async def go():
        await sup.start(["house"])
        # Simulate the body process dying on its own.
        proc.returncode = 1
        status = sup.status()
        assert status["state"] == "crashed"
        assert status["running"] is False

    asyncio.run(go())
