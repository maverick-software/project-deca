"""FastAPI routes for the Vast.ai control plane (mounted by decadic.api.app).

Endpoints (all local-only; the server must stay bound to 127.0.0.1):
- GET/POST /vast/settings   - masked key + deploy defaults
- GET      /vast/account    - balance/email (best-effort vastai show user)
- GET      /vast/offers     - GPU offer search
- GET      /vast/local-checkpoints - agents available to ship/restore
- GET      /vast/browse-fs  - server-side directory listing (SSH key file picker)
- POST     /vast/deploy     - start provisioning
- GET      /vast/deployment - live phase + log
- POST     /vast/deployment/stop|destroy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from decadic.api.vast.cli import VastCliError
from decadic.api.vast.controller import DeployRequest, VastController
from decadic.api.vast.settings_store import VastSettingsStore

logger = logging.getLogger(__name__)


class VastSettingsUpdate(BaseModel):
    api_key: str | None = None
    clear_api_key: bool = False
    ssh_key_path: str | None = None
    clear_ssh_key_path: bool = False
    defaults: dict[str, Any] | None = None


class DeployBody(BaseModel):
    offer_id: int
    preset: str | None = None
    encoder: str | None = None
    whisper_model: str | None = None
    scene: str | None = None
    disk: int | None = None
    image: str | None = None
    restore_agent: str | None = None


def _build_offer_filter(
    gpu_name: str | None,
    num_gpus: int,
    max_dph: float | None,
    min_gpu_ram: float | None,
    verified: bool,
) -> str:
    parts = ["rentable=true", "direct_port_count>=1", f"num_gpus={max(1, num_gpus)}"]
    if gpu_name:
        # Vast query tokens are whitespace-separated, so a model name must use
        # underscores (RTX_4090, not "RTX 4090").
        parts.append(f"gpu_name={gpu_name.strip().replace(' ', '_')}")
    if verified:
        parts.append("verified=true")
    if max_dph and max_dph > 0:
        parts.append(f"dph_total<={max_dph}")
    if min_gpu_ram and min_gpu_ram > 0:
        parts.append(f"gpu_ram>={min_gpu_ram}")
    return " ".join(parts)


def _mb_to_gb(mb: Any) -> float | None:
    """Convert a Vast.ai megabyte RAM value to GB.

    Quirk: `search offers --raw` reports `gpu_ram`/`cpu_ram` in MB (e.g. 24564 for
    a 24 GB card), even though the query parser interprets `gpu_ram>=N` in GB. We
    surface GB to the UI so the value matches the "RAM (GB)" column and the
    min-RAM filter unit.
    """
    try:
        v = float(mb)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return round(v / 1024.0, 1)


def _normalize_offer(o: dict[str, Any]) -> dict[str, Any]:
    dph = o.get("dph_total")
    dlperf = o.get("dlperf")
    per_usd = o.get("dlperf_per_dphtotal")
    if per_usd is None and dlperf and dph:
        try:
            per_usd = float(dlperf) / float(dph) if float(dph) > 0 else None
        except (TypeError, ValueError):
            per_usd = None
    verification = str(o.get("verification", "")).lower()
    return {
        "id": o.get("id"),
        "gpu_name": o.get("gpu_name"),
        "num_gpus": o.get("num_gpus"),
        "gpu_ram_gb": _mb_to_gb(o.get("gpu_ram")),
        "cpu_ram_gb": _mb_to_gb(o.get("cpu_ram")),
        "dph_total": dph,
        "dlperf": dlperf,
        "dlperf_per_usd": per_usd,
        "cuda_max_good": o.get("cuda_max_good"),
        "geolocation": o.get("geolocation"),
        "reliability": o.get("reliability2", o.get("reliability")),
        "verified": verification == "verified",
    }


def register_vast_routes(application: FastAPI) -> None:
    """Define the /vast/* routes on the given app."""

    def _store(application: FastAPI) -> VastSettingsStore:
        return application.state.vast_settings

    def _controller(application: FastAPI) -> VastController:
        return application.state.vast_controller

    @application.get("/vast/settings")
    async def get_vast_settings() -> JSONResponse:
        store = _store(application)
        ctrl = _controller(application)
        view = store.public_view()
        view["cli_available"] = ctrl.cli.available()
        return JSONResponse(view)

    @application.post("/vast/settings")
    async def set_vast_settings(body: VastSettingsUpdate) -> JSONResponse:
        store = _store(application)
        if body.clear_api_key:
            store.clear_api_key()
        elif body.api_key:
            store.set_api_key(body.api_key)
        if body.clear_ssh_key_path:
            store.clear_ssh_key_path()
        elif body.ssh_key_path is not None:
            store.set_ssh_key_path(body.ssh_key_path)
        if body.defaults:
            store.set_defaults(body.defaults)
        view = store.public_view()
        view["cli_available"] = _controller(application).cli.available()
        return JSONResponse(view)

    @application.get("/vast/account")
    async def get_vast_account() -> JSONResponse:
        ctrl = _controller(application)
        store = _store(application)
        if not store.has_api_key():
            raise HTTPException(status_code=400, detail="No Vast.ai API key set.")
        if not ctrl.cli.available():
            raise HTTPException(status_code=503, detail="vastai CLI not installed (pip install vastai).")
        try:
            user = await ctrl.cli.show_user()
        except VastCliError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(
            {
                "balance": user.get("balance"),
                "credit": user.get("credit"),
                "email": user.get("email"),
                "id": user.get("id"),
            }
        )

    @application.get("/vast/offers")
    async def get_vast_offers(
        gpu_name: str | None = None,
        num_gpus: int = 1,
        max_dph: float | None = None,
        min_gpu_ram: float | None = None,
        verified: bool = True,
        limit: int = 50,
    ) -> JSONResponse:
        ctrl = _controller(application)
        store = _store(application)
        if not store.has_api_key():
            raise HTTPException(status_code=400, detail="No Vast.ai API key set.")
        if not ctrl.cli.available():
            raise HTTPException(status_code=503, detail="vastai CLI not installed (pip install vastai).")
        query = _build_offer_filter(gpu_name, num_gpus, max_dph, min_gpu_ram, verified)
        try:
            offers = await ctrl.cli.search_offers(query)
        except VastCliError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        normed = [_normalize_offer(o) for o in offers[: max(1, min(limit, 200))]]
        return JSONResponse({"query": query, "offers": normed})

    @application.get("/vast/gpu-names")
    async def get_vast_gpu_names(limit: int = 2000) -> JSONResponse:
        """Distinct rentable GPU model names (+ live availability count) for the
        UI dropdown. Derived by deduping `gpu_name` across a broad offer search
        (the marketplace `metrics gpu` endpoint needs a host-only permission)."""
        ctrl = _controller(application)
        store = _store(application)
        if not store.has_api_key():
            raise HTTPException(status_code=400, detail="No Vast.ai API key set.")
        if not ctrl.cli.available():
            raise HTTPException(status_code=503, detail="vastai CLI not installed (pip install vastai).")
        try:
            offers = await ctrl.cli.search_offers(
                "rentable=true", order="num_gpus-", no_default=True, limit=max(1, min(limit, 5000))
            )
        except VastCliError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        counts: dict[str, int] = {}
        for o in offers:
            name = o.get("gpu_name")
            if name:
                counts[str(name)] = counts.get(str(name), 0) + 1
        names = [
            {"name": n, "value": n.replace(" ", "_"), "count": c} for n, c in counts.items()
        ]
        names.sort(key=lambda x: (-x["count"], x["name"]))
        return JSONResponse({"gpu_names": names})

    @application.get("/vast/browse-fs")
    async def browse_fs(path: str | None = None) -> JSONResponse:
        """List directories/files under ``path`` (default: the server's home
        directory) so the dashboard can offer a folder-browse picker for the
        SSH key file. The server only binds to 127.0.0.1, so this is exposing
        the operator's own filesystem to their own browser tab - the same
        trust boundary as every other /vast/* route."""
        base = Path(path).expanduser() if path else Path.home()
        try:
            base = base.resolve()
        except OSError:
            pass
        if not base.exists():
            raise HTTPException(status_code=404, detail=f"No such path: {base}")
        if not base.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {base}")
        entries: list[dict[str, Any]] = []
        try:
            children = sorted(
                base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        for child in children:
            if child.name.startswith(".") and child.name not in (".ssh",):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entries.append({"name": child.name, "path": str(child), "is_dir": is_dir})
        parent = str(base.parent) if base.parent != base else None
        return JSONResponse({"path": str(base), "parent": parent, "entries": entries})

    @application.get("/vast/local-checkpoints")
    async def get_local_checkpoints() -> JSONResponse:
        backups_dir: Path = application.state.backups_dir
        ids: list[dict[str, Any]] = []
        if backups_dir.is_dir():
            for ck in sorted(backups_dir.glob("agent_*_checkpoint.json")):
                aid = ck.name[len("agent_") : -len("_checkpoint.json")]
                has_brain = (backups_dir / f"agent_{aid}_brain.pt").is_file()
                ids.append({"agent_id": aid, "has_brain": has_brain})
        return JSONResponse({"checkpoints": ids})

    @application.post("/vast/deploy")
    async def post_vast_deploy(body: DeployBody) -> JSONResponse:
        store = _store(application)
        ctrl = _controller(application)
        d = store.get_defaults()
        # Persist the operator's chosen (non-secret) deploy options.
        store.set_defaults(
            {
                k: v
                for k, v in {
                    "preset": body.preset,
                    "encoder": body.encoder,
                    "whisper_model": body.whisper_model,
                    "scene": body.scene,
                    "disk": body.disk,
                    "image": body.image,
                }.items()
                if v is not None
            }
        )
        req = DeployRequest(
            offer_id=body.offer_id,
            image=body.image or d["image"],
            disk=int(body.disk or d["disk"]),
            preset=body.preset or d["preset"],
            encoder=body.encoder or d["encoder"],
            whisper_model=body.whisper_model or d["whisper_model"],
            scene=body.scene or d["scene"],
            restore_agent=body.restore_agent,
        )
        try:
            snap = await ctrl.deploy(req)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(snap)

    @application.get("/vast/deployment")
    async def get_vast_deployment() -> JSONResponse:
        return JSONResponse(_controller(application).snapshot())

    @application.post("/vast/deployment/stop")
    async def post_vast_stop() -> JSONResponse:
        ctrl = _controller(application)
        try:
            return JSONResponse(await ctrl.stop())
        except (RuntimeError, VastCliError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/vast/deployment/destroy")
    async def post_vast_destroy() -> JSONResponse:
        ctrl = _controller(application)
        try:
            return JSONResponse(await ctrl.destroy())
        except (RuntimeError, VastCliError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
