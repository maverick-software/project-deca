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
import time
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


# Host-quality floors (2026-07-05, per operator request). Cheap community boxes
# repeatedly failed the Docker image pull (registry-1.docker.io TLS handshake
# timeouts) and stranded deploys in "loading"; these steer selection toward
# reliable, well-connected, datacenter-grade hosts. All are overridable per
# request. Units are Vast's *query* units (confirmed against docs.vast.ai):
#   reliability : 0..1 fraction        inet_up/inet_down : Mb/s (e.g. 1000 = 1 Gbps)
#   cpu_ram     : GB                    cpu_ghz           : GHz
#   cpu_cores   : count                disk_bw           : MB/s (1000 = ~1 GB/s)
#   gpu_ram     : GB
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_MIN_INET_MBPS = 1000.0
DEFAULT_MIN_CPU_RAM_GB = 32.0
DEFAULT_MIN_CPU_GHZ = 3.0
DEFAULT_MIN_CPU_CORES = 4
DEFAULT_MIN_DISK_BW_MBS = 1000.0
DEFAULT_MIN_GPU_RAM_GB = 16.0


def _build_offer_filter(
    gpu_name: str | None,
    num_gpus: int,
    max_dph: float | None,
    min_gpu_ram: float | None,
    verified: bool,
    *,
    min_reliability: float | None = DEFAULT_MIN_RELIABILITY,
    min_inet_up: float | None = DEFAULT_MIN_INET_MBPS,
    min_inet_down: float | None = DEFAULT_MIN_INET_MBPS,
    min_cpu_ram: float | None = DEFAULT_MIN_CPU_RAM_GB,
    min_cpu_ghz: float | None = DEFAULT_MIN_CPU_GHZ,
    min_cpu_cores: int | None = DEFAULT_MIN_CPU_CORES,
    min_disk_bw: float | None = DEFAULT_MIN_DISK_BW_MBS,
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
    # Host-quality floors (each skipped when None/<=0 so callers can relax any of
    # them individually -- e.g. drop cpu_ghz, which datacenter EPYC/Xeon parts
    # often report below 4.0).
    if min_reliability and min_reliability > 0:
        parts.append(f"reliability>={min_reliability}")
    if min_inet_up and min_inet_up > 0:
        parts.append(f"inet_up>={min_inet_up}")
    if min_inet_down and min_inet_down > 0:
        parts.append(f"inet_down>={min_inet_down}")
    if min_cpu_ram and min_cpu_ram > 0:
        parts.append(f"cpu_ram>={min_cpu_ram}")
    if min_cpu_ghz and min_cpu_ghz > 0:
        parts.append(f"cpu_ghz>={min_cpu_ghz}")
    if min_cpu_cores and min_cpu_cores > 0:
        parts.append(f"cpu_cores>={min_cpu_cores}")
    if min_disk_bw and min_disk_bw > 0:
        parts.append(f"disk_bw>={min_disk_bw}")
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


def _mbps(mb_per_s: Any) -> float | None:
    """Convert Vast's documented `inet_up`/`inet_down` unit (MB/s) to Mbps.

    ASSUMPTION pending live confirmation (docs/gpu_offer_search_wbs.md Phase D0):
    Vast's own site displays bandwidth numbers that look ~8x smaller than a raw
    MB/s value would (e.g. "3047 Mbps" for what would be an implausible 3047
    MB/s == ~24 Gbps residential-grade uplink); the most likely explanation is
    the site shows Mbps (the conventional unit for network speed) computed as
    MB/s * 8. If a live sample shows otherwise, fix here only - one place.
    """
    try:
        v = float(mb_per_s)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return round(v * 8.0, 1)


def _is_datacenter(o: dict[str, Any]) -> bool | None:
    """Best-effort datacenter/community host flag.

    Vast's API has a `datacenter` *filter* ("show only datacenter offers") but
    the offer *response* schema (per docs.vast.ai) only documents a numeric
    `hosting_type` with no published legend. Treat a present, nonzero
    `hosting_type` as datacenter; if the field is entirely absent, return None
    (unknown) rather than guessing - never claim community/datacenter when we
    don't actually know. Confirm the real encoding via Phase D0 and simplify
    this once known.
    """
    if "hosting_type" in o and o.get("hosting_type") is not None:
        try:
            return int(o["hosting_type"]) != 0
        except (TypeError, ValueError):
            return None
    if "datacenter" in o and o.get("datacenter") is not None:
        return bool(o["datacenter"])
    return None


def _days_remaining(end_date: Any) -> float | None:
    """Days until `end_date` (unix seconds), Vast's likely source for the "Max
    Duration" badge shown on their own site (not itself a documented offer
    field - docs.vast.ai only documents `end_date`/`duration` as inputs, not a
    duration-in-days output). None if `end_date` is missing or already past."""
    try:
        end = float(end_date)
    except (TypeError, ValueError):
        return None
    remaining = (end - time.time()) / 86400.0
    return round(remaining, 1) if remaining > 0 else None


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
        "total_flops": o.get("total_flops"),
        "gpu_mem_bw_gbps": o.get("gpu_mem_bw"),
        "inet_up_mbps": _mbps(o.get("inet_up")),
        "inet_down_mbps": _mbps(o.get("inet_down")),
        "cpu_name": o.get("cpu_name"),
        "cpu_cores": o.get("cpu_cores"),
        "cpu_cores_effective": o.get("cpu_cores_effective"),
        "cpu_ghz": o.get("cpu_ghz"),
        "cpu_ram_gb_raw": o.get("cpu_ram"),
        "disk_bw_mbs": o.get("disk_bw"),
        "direct_port_count": o.get("direct_port_count"),
        "host_id": o.get("host_id"),
        "machine_id": o.get("machine_id"),
        "is_datacenter": _is_datacenter(o),
        "days_remaining": _days_remaining(o.get("end_date")),
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
        min_gpu_ram: float | None = DEFAULT_MIN_GPU_RAM_GB,
        verified: bool = True,
        limit: int = 50,
        min_reliability: float | None = DEFAULT_MIN_RELIABILITY,
        min_inet_up: float | None = DEFAULT_MIN_INET_MBPS,
        min_inet_down: float | None = DEFAULT_MIN_INET_MBPS,
        min_cpu_ram: float | None = DEFAULT_MIN_CPU_RAM_GB,
        min_cpu_ghz: float | None = DEFAULT_MIN_CPU_GHZ,
        min_cpu_cores: int | None = DEFAULT_MIN_CPU_CORES,
        min_disk_bw: float | None = DEFAULT_MIN_DISK_BW_MBS,
    ) -> JSONResponse:
        ctrl = _controller(application)
        store = _store(application)
        if not store.has_api_key():
            raise HTTPException(status_code=400, detail="No Vast.ai API key set.")
        if not ctrl.cli.available():
            raise HTTPException(status_code=503, detail="vastai CLI not installed (pip install vastai).")
        query = _build_offer_filter(
            gpu_name,
            num_gpus,
            max_dph,
            min_gpu_ram,
            verified,
            min_reliability=min_reliability,
            min_inet_up=min_inet_up,
            min_inet_down=min_inet_down,
            min_cpu_ram=min_cpu_ram,
            min_cpu_ghz=min_cpu_ghz,
            min_cpu_cores=min_cpu_cores,
            min_disk_bw=min_disk_bw,
        )
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
