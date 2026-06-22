"""Server-managed MuJoCo environment lifecycle (single-slot supervisor).

The dashboard speaks only to this FastAPI server, while the body/world is a
separate process ([scripts/mujoco_decadic_adapter.py]). This supervisor lets the
UI compose a scenario from selectable elements, then start, pause (brain and
world together), stop, and delete the environment without using the terminal.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from decadic.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

# World elements that can be composed into a scenario. Mirrors
# ``SELECTABLE_ELEMENTS`` in scripts/mujoco_decadic_adapter.py (kept in sync by
# tests/test_environment_supervisor.py).
VALID_ELEMENTS: tuple[str, ...] = (
    "house",
    "food",
    "water",
    "bear",
    "ball",
    "obstacles",
    "npc",
    "crowd",
)

class EnvironmentControlError(RuntimeError):
    """Invalid environment action (e.g. starting while one already runs)."""


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _adapter_script() -> Path:
    return _workspace_root() / "scripts" / "mujoco_decadic_adapter.py"


def _self_host() -> str:
    host = str(os.environ.get("DECADIC_SELF_HOST", "127.0.0.1")).strip()
    return host or "127.0.0.1"


def _self_port() -> int:
    """Port the adapter should connect back to.

    The uvicorn bind port is not visible in-process, so it is supplied via
    ``DECADIC_SELF_PORT`` (falling back to ``DECADIC_PORT`` then 8765 - the
    project's standard server port, used by the launcher, README and dashboard).
    Set this to match the port the server is actually serving on.
    """
    raw = os.environ.get("DECADIC_SELF_PORT") or os.environ.get("DECADIC_PORT", "8765")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 8765


class EnvironmentSupervisor:
    """Owns at most one MuJoCo body subprocess bound to one agent."""

    def __init__(self, registry: AgentRegistry, *, log_dir: Path | None = None) -> None:
        self._registry = registry
        self._log_dir = log_dir or (_workspace_root() / "logs")
        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._log_fh: Any = None
        self._agent_id: str | None = None
        self._elements: list[str] = []
        self._options: dict[str, Any] = {}
        self._paused: bool = False
        self._started_at: float = 0.0
        self._log_path: Path | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(
        self,
        elements: list[str],
        *,
        vision: bool = True,
        audio: bool = False,
        braces: bool = False,
        replace: bool = False,
        preset: str | None = None,
    ) -> dict[str, Any]:
        """Create a fresh agent and spawn the adapter bound to it (single-slot).

        With ``replace=True`` a running body is superseded: it is terminated and
        a new agent+body is started in its place. The previously bound agent
        (brain) is kept in the registry as a now-bodiless mind; only the body
        process is stopped. This lets the dashboard's "New agent" button start a
        body by default even while another body is live.
        """
        async with self._lock:
            if self.is_running() and not replace:
                raise EnvironmentControlError(
                    "An environment is already running; stop it first."
                )
            # Terminate any running body (replace) and reclaim bookkeeping from a
            # previously finished process. The old agent is intentionally kept.
            await self._terminate_locked()

            chosen = [
                e
                for e in (str(x).strip().lower() for x in elements)
                if e in VALID_ELEMENTS
            ]
            if not chosen:
                raise EnvironmentControlError(
                    f"No valid elements; choose from {list(VALID_ELEMENTS)}"
                )

            agent_id = str(uuid.uuid4())
            self._registry.create_agent(agent_id, preset=preset)

            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._log_dir / f"environment_{agent_id}.log"
            log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

            argv = [
                sys.executable,
                str(_adapter_script()),
                "--host",
                _self_host(),
                "--port",
                str(_self_port()),
                "--agent-id",
                agent_id,
                "--scenario",
                ",".join(chosen),
                "--steps",
                "0",
            ]
            if vision:
                argv.append("--vision")
            if audio:
                argv.append("--audio")
            # Braces are a manual scaffold. Bodies start free by default; only
            # the explicit on case needs a flag.
            if braces:
                argv.append("--braces")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdout=log_fh, stderr=asyncio.subprocess.STDOUT
                )
            except Exception:
                log_fh.close()
                await self._registry.delete_agent(agent_id)
                raise

            self._proc = proc
            self._log_fh = log_fh
            self._log_path = log_path
            self._agent_id = agent_id
            self._elements = chosen
            self._options = {
                "vision": bool(vision),
                "audio": bool(audio),
                "braces": bool(braces),
            }
            self._paused = False
            self._started_at = time.time()
            logger.info(
                "environment_started agent_id=%s pid=%s elements=%s",
                agent_id,
                proc.pid,
                chosen,
            )
            return self._status_dict()

    async def stop(self) -> dict[str, Any]:
        """Terminate the body process; the bound agent (brain) is retained."""
        async with self._lock:
            await self._terminate_locked()
            return self._status_dict()

    async def delete(self) -> dict[str, Any]:
        """Stop the body process and delete the bound agent (the scenario)."""
        async with self._lock:
            await self._terminate_locked()
            if self._agent_id:
                try:
                    await self._registry.delete_agent(self._agent_id)
                except Exception:
                    logger.exception(
                        "environment_agent_delete_failed agent_id=%s", self._agent_id
                    )
            self._agent_id = None
            self._elements = []
            self._options = {}
            self._log_path = None
            self._started_at = 0.0
            return self._status_dict()

    def pause(self) -> dict[str, Any]:
        """Freeze brain and world together."""
        if not self.is_running():
            raise EnvironmentControlError("No running environment to pause.")
        agent = self._registry.get(self._agent_id) if self._agent_id else None
        if agent is not None:
            agent.pause()
            agent.queue_body_command("pause")
        self._paused = True
        return self._status_dict()

    def resume(self) -> dict[str, Any]:
        """Reanimate brain and world together."""
        if not self.is_running():
            raise EnvironmentControlError("No running environment to resume.")
        agent = self._registry.get(self._agent_id) if self._agent_id else None
        if agent is not None:
            agent.resume()
            agent.queue_body_command("resume")
        self._paused = False
        return self._status_dict()

    def status(self) -> dict[str, Any]:
        return self._status_dict()

    async def _terminate_locked(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self._proc = None
        self._paused = False

    def _state(self) -> str:
        proc = self._proc
        if proc is None:
            return "stopped"
        if proc.returncode is None:
            return "paused" if self._paused else "running"
        return "crashed" if proc.returncode not in (0, None) else "stopped"

    def _status_dict(self) -> dict[str, Any]:
        proc = self._proc
        return {
            "state": self._state(),
            "running": self.is_running(),
            "paused": self._paused and self.is_running(),
            "agent_id": self._agent_id,
            "elements": list(self._elements),
            "options": dict(self._options),
            "pid": proc.pid if proc is not None else None,
            "returncode": proc.returncode if proc is not None else None,
            "started_at": self._started_at or None,
            "log_path": str(self._log_path) if self._log_path else None,
            "available_elements": list(VALID_ELEMENTS),
        }
