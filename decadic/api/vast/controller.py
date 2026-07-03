"""Deployment state machine + background provisioning for a Vast.ai GPU box.

One ``VastController`` lives on ``app.state``. It owns at most one deployment:
search -> create -> wait-running -> upload code -> install deps -> run brain
(+ scene) -> open ssh tunnel -> create the remote agent. Progress is exposed as
a phase + a rolling log so the dashboard can render a live stepper. While a
deployment is ``ready`` the proxy (see ``proxy.py``) forwards agent traffic to
the tunnelled remote, so the existing panels show the remote agent learning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import tarfile
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decadic.api.presets.store import BUILTIN_PRESETS
from decadic.api.vast.cli import VastCli, VastCliError, open_tunnel, parse_ssh_url, ssh_exec
from decadic.api.vast.settings_store import VastSettingsStore

logger = logging.getLogger(__name__)

# Element lists for the 6 built-in scenario presets, sourced from the SAME
# BUILTIN_PRESETS the dashboard's "+ New agent" dropdown and /agent-presets
# both use - so a Vast deploy resolves a scene identically to a local agent
# instead of a second hand-maintained bear/food mapping. "mind" maps to []
# (no body) since its own elements list is already empty.
_BUILTIN_SCENE_ELEMENTS: dict[str, list[str]] = {
    str(p["id"]): list(p["elements"]) for p in BUILTIN_PRESETS
}
# Back-compat aliases for the old ad-hoc scene shorthand this endpoint used to
# accept directly (pre-preset-unification): "bear" meant the predator
# scenario, "food" meant the foraging scenario. Kept so already-saved
# ~/.decadic/vast.json defaults and any manual API callers keep working.
_LEGACY_SCENE_ALIASES: dict[str, str] = {"bear": "predator", "food": "forage"}

REMOTE_ROOT = "/workspace/app"
REMOTE_PORT = 8765
# Project subtree shipped to the box (the dashboard stays local).
PAYLOAD_INCLUDE = ("decadic", "scripts", "assets", "deploy", "pyproject.toml", "requirements.txt")
# Ordered phases for the UI stepper.
PHASE_ORDER = (
    "creating",
    "waiting",
    "uploading",
    "installing",
    "serving",
    "tunneling",
    "starting_agent",
    "ready",
)
TERMINAL_BAD_STATUS = {"exited", "offline", "unknown"}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclass
class DeployRequest:
    offer_id: int
    image: str
    disk: int
    preset: str
    encoder: str
    whisper_model: str
    scene: str
    restore_agent: str | None = None


@dataclass
class DeploymentState:
    phase: str = "idle"  # idle | <PHASE_ORDER> | error | stopped | destroying
    instance_id: int | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    local_port: int | None = None
    dph: float | None = None  # $/hr
    agent_id: str | None = None
    scene: str | None = None
    preset: str | None = None
    error: str | None = None
    created_at: float | None = None
    ready_at: float | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=400))


class VastController:
    """Owns a single Vast deployment + the background provisioning task."""

    def __init__(self, store: VastSettingsStore, *, log_dir: Path | None = None) -> None:
        self._store = store
        self._cli = VastCli(store.get_api_key)
        self._log_dir = log_dir or (_workspace_root() / "logs")
        self._lock = asyncio.Lock()
        self._state = DeploymentState()
        self._task: asyncio.Task | None = None
        self._tunnel: asyncio.subprocess.Process | None = None
        self._abort = False

    # --- public surface ----------------------------------------------------
    @property
    def cli(self) -> VastCli:
        return self._cli

    @property
    def active(self) -> bool:
        return self._state.phase == "ready" and self._state.local_port is not None

    def proxy_base(self) -> str | None:
        if self.active and self._state.local_port:
            return f"http://127.0.0.1:{self._state.local_port}"
        return None

    def is_busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def _log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self._state.log.append(line)
        logger.info("vast_deploy %s", msg)

    def snapshot(self) -> dict[str, Any]:
        st = self._state
        now = time.time()
        elapsed = (now - st.created_at) if st.created_at else 0.0
        est_cost = (st.dph or 0.0) * (elapsed / 3600.0) if st.dph else None
        return {
            "phase": st.phase,
            "phase_order": list(PHASE_ORDER),
            "busy": self.is_busy(),
            "active": self.active,
            "instance_id": st.instance_id,
            "ssh_host": st.ssh_host,
            "ssh_port": st.ssh_port,
            "dph": st.dph,
            "elapsed_s": round(elapsed, 1),
            "est_cost_usd": round(est_cost, 4) if est_cost is not None else None,
            "agent_id": st.agent_id,
            "scene": st.scene,
            "preset": st.preset,
            "error": st.error,
            "ready": self.active,
            "log": list(st.log)[-60:],
        }

    async def deploy(self, req: DeployRequest) -> dict[str, Any]:
        async with self._lock:
            if self.is_busy() or self._state.phase not in ("idle", "error", "stopped"):
                raise RuntimeError("A deployment is already in progress or active; destroy it first.")
            if not self._cli.available():
                raise RuntimeError("vastai CLI not found on the server. Install it with: pip install vastai")
            if not self._store.has_api_key():
                raise RuntimeError("No Vast.ai API key set. Save one first.")
            self._abort = False
            self._state = DeploymentState(
                phase="creating", scene=req.scene, preset=req.preset, created_at=time.time()
            )
            self._task = asyncio.create_task(self._provision(req), name="vast-provision")
        return self.snapshot()

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            iid = self._state.instance_id
            if iid is None:
                raise RuntimeError("No instance to stop.")
            self._log(f"stopping instance {iid} (disk billing continues)")
            await self._close_tunnel()
            await self._cli.stop_instance(iid)
            self._state.phase = "stopped"
            return self.snapshot()

    async def destroy(self) -> dict[str, Any]:
        async with self._lock:
            self._abort = True
            await self._cancel_task()
            iid = self._state.instance_id
            self._state.phase = "destroying"
            # Best-effort: checkpoint the remote agent, then close the tunnel.
            if self.proxy_base() and self._state.agent_id:
                await self._best_effort_checkpoint(self.proxy_base(), self._state.agent_id)
            await self._close_tunnel()
            if iid is not None:
                self._log(f"destroying instance {iid}")
                try:
                    await self._cli.destroy_instance(iid)
                except VastCliError as exc:
                    self._log(f"destroy error: {exc}")
            self._state = DeploymentState(phase="idle")
            return self.snapshot()

    async def shutdown(self) -> None:
        """Server-shutdown hook: cancel provisioning + close the tunnel only.

        The rented instance is intentionally NOT destroyed here (a server
        restart should not kill a paid box); use destroy() for that.
        """
        self._abort = True
        await self._cancel_task()
        await self._close_tunnel()

    # --- provisioning steps ------------------------------------------------
    async def _provision(self, req: DeployRequest) -> None:
        try:
            await self._step_create(req)
            await self._step_wait_running()
            await self._step_upload(req)
            await self._step_install(req)
            await self._step_serve(req)
            await self._step_tunnel()
            await self._step_start_agent(req)
            self._state.phase = "ready"
            self._state.ready_at = time.time()
            self._log("deployment ready -- watching the remote agent")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self._state.error = str(exc)
            self._state.phase = "error"
            self._log(f"ERROR: {exc}")
            logger.exception("vast_provision_failed")
            # Auto-destroy a created-but-unusable instance to stop billing.
            iid = self._state.instance_id
            if iid is not None and not self._abort:
                self._log(f"auto-destroying instance {iid} after failure")
                try:
                    await self._close_tunnel()
                    await self._cli.destroy_instance(iid)
                    self._state.instance_id = None
                except VastCliError as de:
                    self._log(f"auto-destroy error: {de}")

    async def _step_create(self, req: DeployRequest) -> None:
        self._state.phase = "creating"
        self._log(f"renting offer {req.offer_id} ({req.image}, {req.disk}GB)")
        iid = await self._cli.create_instance(
            req.offer_id, image=req.image, disk=req.disk, ssh=True, direct=True
        )
        self._state.instance_id = iid
        self._log(f"instance created: {iid}")

    async def _step_wait_running(self, timeout: float = 600.0) -> None:
        self._state.phase = "waiting"
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            self._raise_if_aborted()
            inst = await self._cli.show_instance(self._state.instance_id)
            status = inst.get("actual_status")
            if inst.get("dph_total") is not None:
                self._state.dph = float(inst["dph_total"])
            if status != last:
                self._log(f"instance status: {status or 'provisioning'}")
                last = status
            if status == "running":
                host = inst.get("ssh_host")
                port = inst.get("ssh_port")
                if not host or not port:
                    url = await self._cli.ssh_url(self._state.instance_id)
                    host, port = parse_ssh_url(url)
                self._state.ssh_host = str(host)
                self._state.ssh_port = int(port)
                self._log(f"running; ssh root@{host}:{port}")
                return
            if status in TERMINAL_BAD_STATUS:
                raise VastCliError(f"instance entered bad status '{status}' before running")
            await asyncio.sleep(10.0)
        raise VastCliError("timed out waiting for instance to reach 'running'")

    async def _step_upload(self, req: DeployRequest) -> None:
        self._state.phase = "uploading"
        self._log("packaging code payload")
        payload = self._build_payload(include_backups=bool(req.restore_agent))
        try:
            await self._ssh(f"mkdir -p {REMOTE_ROOT}")
            self._log("uploading payload to the box")
            await self._cli.copy(f"local:{payload}", f"{self._state.instance_id}:{REMOTE_ROOT}/payload.tgz")
            await self._ssh(f"cd {REMOTE_ROOT} && tar xzf payload.tgz && rm -f payload.tgz")
            self._log("payload extracted")
        finally:
            try:
                os.unlink(payload)
            except OSError:
                pass

    async def _step_install(self, req: DeployRequest) -> None:
        self._state.phase = "installing"
        self._log("installing deps + prewarming encoders (this can take a few minutes)")
        env = f"ENCODER={req.encoder} WHISPER_MODEL={req.whisper_model}"
        code, out, err = await self._ssh(
            f"{env} bash {REMOTE_ROOT}/deploy/vast/setup_remote.sh", timeout=1500.0
        )
        if code != 0:
            raise VastCliError(f"remote setup failed (exit {code}): {(err or out)[-500:]}")
        self._log("dependencies installed")

    async def _step_serve(self, req: DeployRequest) -> None:
        self._state.phase = "serving"
        self._log(f"starting brain server (preset={req.preset}, encoder={req.encoder})")
        env = (
            f"PRESET={req.preset} ENCODER={req.encoder} "
            f"WHISPER_MODEL={req.whisper_model} APP_ROOT={REMOTE_ROOT}"
        )
        code, out, err = await self._ssh(
            f"{env} bash {REMOTE_ROOT}/deploy/vast/run_remote.sh", timeout=120.0
        )
        if code != 0:
            raise VastCliError(f"remote run failed (exit {code}): {(err or out)[-500:]}")

    async def _step_tunnel(self) -> None:
        self._state.phase = "tunneling"
        local_port = _find_free_port()
        self._tunnel = await open_tunnel(
            self._state.ssh_host,
            self._state.ssh_port,
            local_port,
            REMOTE_PORT,
            key_path=self._store.get_ssh_key_path() or None,
        )
        self._state.local_port = local_port
        self._log(f"tunnel up: localhost:{local_port} -> remote:{REMOTE_PORT}; waiting for server")
        if not await self._wait_remote_ready(f"http://127.0.0.1:{local_port}", timeout=180.0):
            raise VastCliError("remote server did not become reachable through the tunnel")
        self._log("remote server is up")

    async def _step_start_agent(self, req: DeployRequest) -> None:
        self._state.phase = "starting_agent"
        base = f"http://127.0.0.1:{self._state.local_port}"
        if req.restore_agent:
            agent_id = await self._restore_remote_agent(base, req)
            if agent_id:
                self._state.agent_id = agent_id
                self._log(f"restored agent {agent_id}")
                return
            self._log("restore failed; falling back to a fresh agent")
        elements = self._scene_elements(req.scene)
        if not elements:
            agent_id = await self._remote_create_agent(base)
            self._state.agent_id = agent_id
            self._log(f"fresh mind-only agent {agent_id}")
        else:
            status = await self._remote_start_environment(base, elements)
            self._state.agent_id = status.get("agent_id")
            self._log(f"scene '{req.scene}' started; agent {self._state.agent_id}")

    # --- helpers -----------------------------------------------------------
    def _scene_elements(self, scene: str | None) -> list[str]:
        """Resolve a deploy-request scene value to world elements.

        Accepts a built-in preset id (calm/forage/parent/village/predator/
        mind - the same ids /agent-presets returns), a legacy bear/food
        alias, or a raw comma-separated element list for manual/API use.
        """
        s = (scene or "none").strip().lower()
        if s in ("none", "", "mind", "mind_only"):
            return []
        s = _LEGACY_SCENE_ALIASES.get(s, s)
        if s in _BUILTIN_SCENE_ELEMENTS:
            return list(_BUILTIN_SCENE_ELEMENTS[s])
        return [e for e in (p.strip() for p in s.split(",")) if e]

    def _build_payload(self, *, include_backups: bool) -> str:
        root = _workspace_root()
        fd, tmp = tempfile.mkstemp(suffix=".tgz", prefix="vast_payload_")
        os.close(fd)

        def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
            name = ti.name
            if "__pycache__" in name or name.endswith((".pyc", ".sqlite", ".sqlite-journal")):
                return None
            if "/node_modules/" in name or "/.venv/" in name:
                return None
            return ti

        with tarfile.open(tmp, "w:gz") as tar:
            for rel in PAYLOAD_INCLUDE:
                p = root / rel
                if p.exists():
                    tar.add(p, arcname=rel, filter=_filter)
            if include_backups:
                bdir = root / "backups"
                if bdir.is_dir():
                    for ck in bdir.glob("agent_*_checkpoint.json"):
                        tar.add(ck, arcname=f"backups/{ck.name}")
                    for pt in bdir.glob("agent_*_brain.pt"):
                        tar.add(pt, arcname=f"backups/{pt.name}")
        return tmp

    async def _ssh(self, command: str, *, timeout: float = 900.0) -> tuple[int, str, str]:
        self._raise_if_aborted()
        return await ssh_exec(
            self._state.ssh_host,
            self._state.ssh_port,
            command,
            key_path=self._store.get_ssh_key_path() or None,
            timeout=timeout,
        )

    async def _wait_remote_ready(self, base: str, *, timeout: float) -> bool:
        import httpx

        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                self._raise_if_aborted()
                try:
                    r = await client.get(f"{base}/agents")
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(3.0)
        return False

    async def _remote_create_agent(self, base: str) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{base}/agent")
            r.raise_for_status()
            return str(r.json()["agent_id"])

    async def _remote_start_environment(self, base: str, elements: list[str]) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{base}/environment",
                json={"elements": elements, "vision": True, "audio": True, "replace": True},
            )
            r.raise_for_status()
            return r.json()

    async def _restore_remote_agent(self, base: str, req: DeployRequest) -> str | None:
        """Best-effort: ship a local checkpoint under a new remote id + restore."""
        import httpx

        rid = req.restore_agent
        root = _workspace_root()
        ck = root / "backups" / f"agent_{rid}_checkpoint.json"
        if not ck.is_file():
            self._log(f"no local checkpoint for agent {rid}")
            return None
        try:
            ck_preset = json.loads(ck.read_text(encoding="utf-8")).get("preset")
        except Exception:
            ck_preset = None
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                new_id = (await client.post(f"{base}/agent")).json()["agent_id"]
                if ck_preset:
                    await client.post(f"{base}/agent/{new_id}/preset", params={"preset": ck_preset})
                # Ship checkpoint files renamed to the new agent id.
                remote_backups = f"{REMOTE_ROOT}/backups"
                await self._ssh(f"mkdir -p {remote_backups}")
                await self._cli.copy(
                    f"local:{ck}", f"{self._state.instance_id}:{remote_backups}/agent_{new_id}_checkpoint.json"
                )
                pt = root / "backups" / f"agent_{rid}_brain.pt"
                if pt.is_file():
                    await self._cli.copy(
                        f"local:{pt}", f"{self._state.instance_id}:{remote_backups}/agent_{new_id}_brain.pt"
                    )
                await client.post(f"{base}/agent/{new_id}/restore")
                return str(new_id)
        except Exception as exc:  # noqa: BLE001
            self._log(f"restore error: {exc}")
            return None

    async def _best_effort_checkpoint(self, base: str, agent_id: str) -> None:
        import httpx

        try:
            self._log(f"checkpointing remote agent {agent_id} before teardown")
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(f"{base}/agent/{agent_id}/checkpoint")
        except Exception as exc:  # noqa: BLE001
            self._log(f"checkpoint error (continuing teardown): {exc}")

    async def _close_tunnel(self) -> None:
        proc = self._tunnel
        self._tunnel = None
        self._state.local_port = None
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

    async def _cancel_task(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None

    def _raise_if_aborted(self) -> None:
        if self._abort:
            raise VastCliError("deployment aborted")
