"""FastAPI application exposing REST + WebSocket agent endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from decadic.agents.registry import AgentRegistry
from decadic.api.environment import EnvironmentControlError, EnvironmentSupervisor
from decadic.embodiment import stances as stance_lib
from decadic.api.schemas import (
    AgentCreateResponse,
    AgentStateResponse,
    CheckpointResponse,
    MemoryQueryResponse,
    MemorySimilarResponse,
    MetricsResponse,
    ObservationMessage,
)
from decadic.api.saved_agents.routes import register_saved_agents_routes
from decadic.api.presets.routes import register_preset_routes
from decadic.api.presets.store import PresetStore
from decadic.api.vast.controller import VastController
from decadic.api.vast.proxy import install_vast_proxy
from decadic.api.vast.routes import register_vast_routes
from decadic.api.vast.settings_store import VastSettingsStore
from decadic.training.routes import register_skill_dojo_routes
from decadic.training.store import UploadedSkillStore
from decadic.training.supervisor import SkillDojoSupervisor
from decadic.logging import setup_logging, stop_logging
from decadic.memory.embeddings import query_vector_from_state_bus
from decadic.nn.faculties import CognitionFaculties
from decadic.nn.config import VALID_PRESETS
from decadic.nn.plastic import PlasticityFlags

logger = logging.getLogger(__name__)


def _flags_to_dict(flags: PlasticityFlags) -> dict[str, object]:
    """Serialize plasticity flags for the /settings/agent-defaults endpoints."""
    return {
        "plasticity_enabled": bool(flags.plastic),
        "sparse_enabled": bool(flags.sparse),
        "growth_enabled": bool(flags.growth),
        "plasticity_alpha": float(flags.alpha),
        "sparse_density": float(flags.density),
        "max_neurons": int(flags.max_neurons),
        "growable_hidden_ceiling": int(flags.hidden_ceiling),
    }


def _faculties_to_dict(fac: CognitionFaculties) -> dict[str, object]:
    """Serialize cognitive faculties for the /settings/agent-defaults endpoints."""
    return {
        "perception_feedback": bool(fac.perception_feedback),
        "self_model_feedback": bool(fac.self_model_feedback),
        "predictive_affect": bool(fac.predictive_affect),
        "represented_self": bool(fac.represented_self),
        "perception_mode": str(fac.perception_mode),
        "encoder_mode": str(fac.encoder_mode),
    }


def _resolve_default_flags(registry: AgentRegistry) -> PlasticityFlags:
    """Current new-agent plasticity defaults (registry override or process env)."""
    return registry.new_agent_flags or PlasticityFlags.from_env()


def _resolve_default_faculties(registry: AgentRegistry) -> CognitionFaculties:
    """Current new-agent faculty defaults (registry override or process env)."""
    return registry.new_agent_faculties or CognitionFaculties.from_env()


def _agent_defaults_dict(registry: AgentRegistry) -> dict[str, object]:
    """Merged new-agent defaults (plasticity flags + cognitive faculties)."""
    return {
        **_flags_to_dict(_resolve_default_flags(registry)),
        **_faculties_to_dict(_resolve_default_faculties(registry)),
    }


def _validate_neural_preset(preset: str | None) -> str | None:
    """Validate an explicit neural architecture preset for new-agent creation."""
    if preset is None:
        return None
    name = preset.strip().lower()
    if not name:
        return None
    if name not in VALID_PRESETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown preset {preset!r}; choose from {list(VALID_PRESETS)}",
        )
    return name


def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "DECADIC_CORS_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


class EnvironmentStartRequest(BaseModel):
    """Scenario composition + sensory options for a managed environment."""

    elements: list[str] = Field(default_factory=list)
    vision: bool = True
    audio: bool = False
    # Whether the manual joint-brace orthosis starts engaged. Default off keeps
    # new bodies free unless the operator explicitly enables the scaffold.
    braces: bool = False
    # When true, supersede a running body instead of erroring; the old agent is
    # kept (now bodiless) and a new agent+body is started in its place.
    replace: bool = False
    # Optional neural architecture preset for the fresh mind.
    preset: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_dir = Path(os.environ.get("DECADIC_LOG_DIR", _workspace_root() / "logs"))
    setup_logging(log_dir)
    data_dir = Path(os.environ.get("DECADIC_DATA_DIR", _workspace_root() / "data"))
    app.state.registry = AgentRegistry(data_dir)
    backups_dir = Path(os.environ.get("DECADIC_BACKUPS_DIR", _workspace_root() / "backups"))
    backups_dir.mkdir(parents=True, exist_ok=True)
    app.state.backups_dir = backups_dir
    # Durable Saved Agents library, intentionally separate from backups/ so the
    # tombstone pruner never touches it.
    saved_dir = Path(os.environ.get("DECADIC_SAVED_DIR", _workspace_root() / "saved_agents"))
    saved_dir.mkdir(parents=True, exist_ok=True)
    app.state.saved_dir = saved_dir
    # Agent presets (named scenario/body configs for the dashboard dropdown),
    # seeded with built-ins on first run. Kept separate from saved_agents/.
    presets_dir = Path(os.environ.get("DECADIC_PRESETS_DIR", _workspace_root() / "presets"))
    presets_dir.mkdir(parents=True, exist_ok=True)
    app.state.presets_dir = presets_dir
    app.state.preset_store = PresetStore(presets_dir)
    skills_dir = Path(os.environ.get("DECADIC_SKILLS_DIR", data_dir / "skills"))
    skills_dir.mkdir(parents=True, exist_ok=True)
    app.state.skills_dir = skills_dir
    app.state.skill_store = UploadedSkillStore(skills_dir)
    app.state.environment = EnvironmentSupervisor(app.state.registry, log_dir=log_dir)
    # Skill Dojo: generalized skill-training curricula (teacher hints enter only
    # replay/consolidation metadata; the live cognitive loop stays self-supervised).
    app.state.skill_dojo = SkillDojoSupervisor(
        app.state.registry,
        backups_dir=backups_dir,
        log_dir=log_dir,
        skill_loader=app.state.skill_store.get_any,
    )
    # Vast.ai GPU deployment control plane (UI-driven). The settings store
    # persists the API key + deploy defaults; the controller owns at most one
    # rented instance and the ssh tunnel feeding the reverse proxy.
    app.state.vast_settings = VastSettingsStore()
    app.state.vast_controller = VastController(app.state.vast_settings, log_dir=log_dir)
    yield
    try:
        await app.state.skill_dojo.stop()
    except Exception:
        logger.exception("skill_dojo_shutdown_failed")
    # Shut down any managed body process so it does not outlive the server.
    try:
        await app.state.environment.stop()
    except Exception:
        logger.exception("environment_shutdown_failed")
    # Tear down the ssh tunnel + provisioning task (the rented box is left
    # running on purpose; use the dashboard's Destroy to terminate billing).
    try:
        await app.state.vast_controller.shutdown()
    except Exception:
        logger.exception("vast_shutdown_failed")
    # Flush + stop the background logging listener last so the shutdown logs above
    # are drained to stdout/file before the worker thread retires.
    stop_logging()


def create_app() -> FastAPI:
    """Construct FastAPI app (supports isolated instances for tests)."""
    application = FastAPI(title="Decadic Cycle Cognitive Architecture", lifespan=lifespan)
    # Reverse proxy added first so it is INNER; CORS added last stays OUTERMOST
    # and owns CORS headers for both local and proxied (remote-agent) responses.
    install_vast_proxy(application)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.post("/agent", response_model=AgentCreateResponse)
    async def create_agent(preset: str | None = None) -> AgentCreateResponse:
        registry: AgentRegistry = application.state.registry
        agent_id = str(uuid.uuid4())
        registry.create_agent(agent_id, preset=_validate_neural_preset(preset))
        return AgentCreateResponse(agent_id=agent_id)

    @application.get("/settings/agent-defaults")
    async def get_agent_defaults() -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        return JSONResponse(_agent_defaults_dict(registry))

    @application.post("/settings/agent-defaults")
    async def set_agent_defaults(
        plasticity_enabled: bool | None = None,
        sparse_enabled: bool | None = None,
        growth_enabled: bool | None = None,
        plasticity_alpha: float | None = None,
        sparse_density: float | None = None,
        max_neurons: int | None = None,
        growable_hidden_ceiling: int | None = None,
        perception_feedback: bool | None = None,
        self_model_feedback: bool | None = None,
        predictive_affect: bool | None = None,
        represented_self: bool | None = None,
        perception_mode: str | None = None,
        encoder_mode: str | None = None,
    ) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        # Start from the currently-resolved defaults so unspecified fields stick.
        current = _resolve_default_flags(registry)
        registry.new_agent_flags = PlasticityFlags(
            plastic=current.plastic if plasticity_enabled is None else bool(plasticity_enabled),
            alpha=current.alpha if plasticity_alpha is None else max(0.0, float(plasticity_alpha)),
            sparse=current.sparse if sparse_enabled is None else bool(sparse_enabled),
            density=current.density
            if sparse_density is None
            else min(1.0, max(0.01, float(sparse_density))),
            growth=current.growth if growth_enabled is None else bool(growth_enabled),
            hidden_ceiling=current.hidden_ceiling
            if growable_hidden_ceiling is None
            else max(1, int(growable_hidden_ceiling)),
            max_neurons=current.max_neurons if max_neurons is None else max(1, int(max_neurons)),
        )
        cur_fac = _resolve_default_faculties(registry)
        registry.new_agent_faculties = CognitionFaculties(
            perception_feedback=cur_fac.perception_feedback
            if perception_feedback is None
            else bool(perception_feedback),
            self_model_feedback=cur_fac.self_model_feedback
            if self_model_feedback is None
            else bool(self_model_feedback),
            predictive_affect=cur_fac.predictive_affect
            if predictive_affect is None
            else bool(predictive_affect),
            represented_self=cur_fac.represented_self
            if represented_self is None
            else bool(represented_self),
            perception_mode=cur_fac.perception_mode if perception_mode is None else perception_mode,
            encoder_mode=cur_fac.encoder_mode if encoder_mode is None else encoder_mode,
        )
        return JSONResponse(_agent_defaults_dict(registry))

    @application.get("/agents")
    async def list_agents() -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agents = []
        for aid in registry.ids():
            agent = registry.get(aid)
            if agent is None:
                continue
            agents.append(
                {
                    "agent_id": aid,
                    "neural_enabled": agent.neural is not None,
                    "cycles_completed": int(agent.metrics.get("cycles_completed", 0)),
                    "paused": agent.paused,
                    "status": agent.status,
                    "died_at_cycle": agent.died_at_cycle,
                    "encoder_mode": agent.encoder_mode(),
                    "has_body": agent.has_body(),
                    "preset": agent.preset,
                }
            )
        return JSONResponse({"agents": agents})

    @application.post("/agent/{agent_id}/pause")
    async def pause_agent(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        agent.pause()
        return JSONResponse({"agent_id": agent_id, "status": "paused"})

    @application.post("/agent/{agent_id}/resume")
    async def resume_agent(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        agent.resume()
        return JSONResponse({"agent_id": agent_id, "status": "running"})

    @application.post("/agent/{agent_id}/reset")
    async def reset_agent(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        await agent.reset()
        return JSONResponse({"agent_id": agent_id, "status": "reset"})

    @application.post("/agent/{agent_id}/preset")
    async def set_preset(agent_id: str, preset: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        name = preset.strip().lower()
        if name not in VALID_PRESETS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown preset {preset!r}; choose from {list(VALID_PRESETS)}",
            )
        # Architecture change requires a fresh mind; reset rebuilds at the new size.
        await agent.reset(preset=name)
        return JSONResponse({"agent_id": agent_id, "status": "reset", "preset": agent.preset})

    @application.post("/agent/{agent_id}/revive")
    async def revive_agent(agent_id: str, restore_to: float | None = None) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        agent.revive(restore_to)
        return JSONResponse(
            {"agent_id": agent_id, "status": agent.status, "viability": agent.viability.value}
        )

    @application.post("/agent/{agent_id}/config")
    async def configure_agent(
        agent_id: str,
        parallel_sessions: int | None = None,
        working_memory_slots: int | None = None,
        working_memory_decay: float | None = None,
        assist_override: float | None = None,
        curriculum_mode: str | None = None,
        viability_mode: str | None = None,
        metabolic_compression: float | None = None,
        ai_intero_pref_weight: float | None = None,
        drive_priority_gain: float | None = None,
        motor_babble_sigma: float | None = None,
        plasticity_alpha: float | None = None,
        sparse_density: float | None = None,
        max_neurons: int | None = None,
        perception_mode: str | None = None,
        perception_feedback: bool | None = None,
        self_model_feedback: bool | None = None,
        predictive_affect: bool | None = None,
        represented_self: bool | None = None,
        encoder_mode: str | None = None,
        cognition_trace: bool | None = None,
        probe_capture: bool | None = None,
        gwt_enabled: bool | None = None,
        integration_window_ms: float | None = None,
        episodic_async: bool | None = None,
        ltm_async: bool | None = None,
    ) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            # assist_override: negative (e.g. -1) clears to Auto; >= 0 pins the level.
            # curriculum_mode: "guided" (assist-as-needed harness) | "legacy" (assist).
            # viability_mode: "metabolic" | "immortal" (pins reservoirs, no death).
            # plasticity_alpha/sparse_density/max_neurons: live A/B/C knobs.
            # perception_mode/perception_feedback/encoder_mode: core faculties that
            #   rebuild the brain (fresh weights) when changed.
            # cognition_trace/probe_capture: live read-only observation toggles.
            # gwt_enabled: live global-workspace competition toggle (no rebuild).
            # integration_window_ms: live temporal-integration window (0=off).
            # episodic_async/ltm_async: live write-behind persistence toggles.
            config = agent.configure(
                parallel_sessions=parallel_sessions,
                working_memory_slots=working_memory_slots,
                working_memory_decay=working_memory_decay,
                assist_override=assist_override,
                curriculum_mode=curriculum_mode,
                viability_mode=viability_mode,
                metabolic_compression=metabolic_compression,
                ai_intero_pref_weight=ai_intero_pref_weight,
                drive_priority_gain=drive_priority_gain,
                motor_babble_sigma=motor_babble_sigma,
                plasticity_alpha=plasticity_alpha,
                sparse_density=sparse_density,
                max_neurons=max_neurons,
                perception_mode=perception_mode,
                perception_feedback=perception_feedback,
                self_model_feedback=self_model_feedback,
                predictive_affect=predictive_affect,
                represented_self=represented_self,
                encoder_mode=encoder_mode,
                cognition_trace=cognition_trace,
                probe_capture=probe_capture,
                gwt_enabled=gwt_enabled,
                integration_window_ms=integration_window_ms,
                episodic_async=episodic_async,
                ltm_async=ltm_async,
            )
        return JSONResponse({"agent_id": agent_id, **config})

    @application.get("/agent/{agent_id}/discovery")
    async def get_discovery(agent_id: str) -> JSONResponse:
        """Discovered-mode perception report: graph + slots + eval vs oracle truth."""
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            perc = agent.perceptual
            payload = {
                "agent_id": agent_id,
                "perception_mode": perc.perception_mode,
                "egocentric_graph": {
                    "nodes": list(perc.egocentric_nodes),
                    "edges": list(perc.egocentric_edges),
                },
                "working_memory": perc.working_memory.snapshot(),
                "object_files": list(getattr(perc, "object_files", [])),
                "discovery_health": (
                    dict(perc.discovery_health)
                    if getattr(perc, "discovery_health", None)
                    else None
                ),
                "perception_organ": (
                    dict(perc.perception_organ)
                    if getattr(perc, "perception_organ", None)
                    else None
                ),
                "retinotopic_map": (
                    dict(perc.retinotopic_map)
                    if getattr(perc, "retinotopic_map", None)
                    else None
                ),
                "ltm_consolidation": dict(getattr(perc, "ltm_consolidation", {})),
                "discovery": (
                    perc.discovery_eval.snapshot()
                    if perc.perception_mode == "discovered"
                    else None
                ),
                "oracle_truth_count": len(perc.oracle_truth),
            }
        return JSONResponse(payload)

    @application.get("/agent/{agent_id}/explain")
    async def get_explain(
        agent_id: str,
        history: int = 0,
        attribution: int = 0,
        counterfactuals: int = 0,
    ) -> JSONResponse:
        """Cognitive-trace ("why") report: the latest structured explanation, an
        optional compact temporal history, and (on demand) input attribution /
        counterfactual rollouts. Read-only; never mutates cognition."""
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            payload = agent.explain(
                history=max(0, int(history)),
                attribution=bool(attribution),
                counterfactuals=bool(counterfactuals),
            )
        return JSONResponse(payload)

    @application.get("/agent/{agent_id}/vision")
    async def get_vision(agent_id: str, camera: str | None = None) -> Response:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            png = agent.last_vision_png(camera)
        if png is None:
            raise HTTPException(status_code=404, detail="No vision frame observed")
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @application.get("/agent/{agent_id}/audio")
    async def get_audio(agent_id: str) -> Response:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            wav = agent.last_audio_wav()
        if wav is None:
            raise HTTPException(status_code=404, detail="No audio observed")
        return Response(content=wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @application.get("/agent/{agent_id}/brain/topology")
    async def get_brain_topology(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            topo = agent.brain_topology()
        if topo is None:
            raise HTTPException(status_code=404, detail="Agent runs stub cognition (no neural stack)")
        return JSONResponse(topo)

    @application.get("/agent/{agent_id}/brain/landscape")
    async def get_brain_landscape(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        surface = agent.brain_landscape()
        if surface is None:
            # Feature off, or enabled but the first surface is not computed yet.
            return JSONResponse(
                {"ready": False, "detail": "loss landscape warming up (or DECADIC_LANDSCAPE_ENABLED=0)"},
                status_code=202,
            )
        return JSONResponse({"ready": True, **surface})

    @application.post("/agent/{agent_id}/body/recenter")
    async def recenter_body(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        if not agent.queue_body_command("recenter"):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": "recenter_queued"})

    @application.post("/agent/{agent_id}/body/reset_braces")
    async def reset_braces(agent_id: str) -> JSONResponse:
        """Re-weld every joint brace (restart the ROM curriculum from fully welded)."""
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        if not agent.queue_body_command("reset_braces"):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": "reset_braces_queued"})

    @application.post("/agent/{agent_id}/body/braces")
    async def set_braces(agent_id: str, enabled: bool = True) -> JSONResponse:
        """Master on/off for the joint-brace orthosis.

        ``enabled=false`` relaxes every hinge to its native joint spring -- the
        brain alone holds the body up (so it can fall); earned ROM is preserved
        and resumes when switched back on.
        """
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        cmd = "braces_on" if enabled else "braces_off"
        if not agent.queue_body_command(cmd):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"{cmd}_queued"})

    @application.post("/agent/{agent_id}/body/movement_hold")
    async def set_movement_hold(agent_id: str, enabled: bool = True) -> JSONResponse:
        """Hold the active stance/motion running until manually disabled.

        ``enabled=true`` welds every joint brace (suspends the ROM curriculum -- no
        range-of-motion release) and loops motion stances continuously, so the
        selected movement runs on repeat; ``enabled=false`` resumes the ROM ratchet.
        """
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        cmd = "hold_on" if enabled else "hold_off"
        if not agent.queue_body_command(cmd):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"{cmd}_queued"})

    @application.get("/body/stances")
    async def list_stances() -> JSONResponse:
        """Catalog of selectable joint-brace stances (name, label, motion flag).

        Static list from the stance library -- the single source of truth shared
        with the body adapter -- so the dashboard can render the stance selector.
        """
        return JSONResponse({"stances": stance_lib.catalog()})

    @application.post("/agent/{agent_id}/body/stance")
    async def set_stance(agent_id: str, name: str) -> JSONResponse:
        """Re-pose the body into a stance and restart that stance's ROM curriculum.

        The body re-poses into the stance start pose and re-welds every joint
        brace (a new posture is a new skill, learned from fully braced). Unknown
        names fall back to the default stand.
        """
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        stance = stance_lib.get_stance(name)
        if not agent.queue_body_command(f"set_stance:{stance.name}"):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"stance_{stance.name}_queued"})

    @application.post("/agent/{agent_id}/body/npc")
    async def npc_freeze(agent_id: str, paused: bool = True) -> JSONResponse:
        """Pause/resume the parent NPC in place (it stops walking, foraging, and
        offering) without touching the agent's brain or the rest of the world."""
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        cmd = "npc_pause" if paused else "npc_resume"
        if not agent.queue_body_command(cmd):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"{cmd}_queued"})

    @application.post("/agent/{agent_id}/body/viewer")
    async def body_viewer(agent_id: str, open: bool = True) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        sup: EnvironmentSupervisor = application.state.environment
        if not sup.is_running() or sup.status().get("agent_id") != agent_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No running body for this agent; start or restart an environment "
                    "before opening the MuJoCo live window."
                ),
            )
        cmd = "open_viewer" if open else "close_viewer"
        if not agent.queue_body_command(cmd):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"{cmd}_queued"})

    @application.post("/agent/{agent_id}/give")
    async def give_resource(
        agent_id: str,
        resource: str,
        mode: str = "near",
        amount: float | None = None,
    ) -> JSONResponse:
        """Provision the agent with water or food, two ways.

        ``mode=direct`` credits the reservoir immediately (admin top-up; works
        without a body). ``mode=near`` asks the connected body to place the
        (unlabeled) prop a step away so the agent must perceive and walk to it,
        preserving the self-learned act->relief loop.
        """
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        res = str(resource).strip().lower()
        if res not in ("water", "food"):
            raise HTTPException(status_code=400, detail="resource must be 'water' or 'food'")
        md = str(mode).strip().lower()
        if md not in ("near", "direct"):
            raise HTTPException(status_code=400, detail="mode must be 'near' or 'direct'")
        if md == "direct":
            try:
                result = await agent.give_resource(res, amount)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return JSONResponse({"agent_id": agent_id, "status": f"{res}_direct", **result})
        sup: EnvironmentSupervisor = application.state.environment
        if not sup.is_running() or sup.status().get("agent_id") != agent_id:
            raise HTTPException(
                status_code=409,
                detail="No running body for this agent; start a scenario with water/food first.",
            )
        if not agent.queue_body_command(f"give_{res}_near"):
            raise HTTPException(status_code=503, detail="Command queue full")
        return JSONResponse({"agent_id": agent_id, "status": f"{res}_near_queued"})

    @application.delete("/agent/{agent_id}")
    async def delete_agent(agent_id: str) -> JSONResponse:
        registry: AgentRegistry = application.state.registry
        if registry.get(agent_id) is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        await registry.delete_agent(agent_id)
        return JSONResponse({"status": "terminated", "agent_id": agent_id})

    @application.get("/environment")
    async def get_environment() -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        return JSONResponse(sup.status())

    @application.post("/environment")
    async def start_environment(req: EnvironmentStartRequest) -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        try:
            status = await sup.start(
                req.elements,
                vision=req.vision,
                audio=req.audio,
                braces=req.braces,
                replace=req.replace,
                preset=_validate_neural_preset(req.preset),
            )
        except EnvironmentControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(status)

    @application.post("/environment/pause")
    async def pause_environment() -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        try:
            return JSONResponse(sup.pause())
        except EnvironmentControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/environment/resume")
    async def resume_environment() -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        try:
            return JSONResponse(sup.resume())
        except EnvironmentControlError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/environment/stop")
    async def stop_environment() -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        return JSONResponse(await sup.stop())

    @application.delete("/environment")
    async def delete_environment() -> JSONResponse:
        sup: EnvironmentSupervisor = application.state.environment
        return JSONResponse(await sup.delete())

    @application.get("/agent/{agent_id}/state", response_model=AgentStateResponse)
    async def get_state(agent_id: str) -> AgentStateResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            payload = agent.snapshot_state()
        return AgentStateResponse(agent_id=agent_id, payload=payload)

    @application.get("/agent/{agent_id}/metrics", response_model=MetricsResponse)
    async def get_metrics(agent_id: str) -> MetricsResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            metrics = dict(agent.metrics)
            metrics["queue_depth"] = agent.out_queue.qsize()
            metrics["viability"] = agent.viability.value
            metrics["priority_label"] = agent.state_bus.priority_label
            metrics["paused"] = agent.paused
            metrics["status"] = agent.status
            metrics["died_at_cycle"] = agent.died_at_cycle
            metrics["encoder_mode"] = agent.encoder_mode()
            metrics["preset"] = agent.preset
            metrics.update(agent.capacity_config())
        return MetricsResponse(agent_id=agent_id, metrics=metrics)

    @application.get("/agent/{agent_id}/memory", response_model=MemoryQueryResponse)
    async def get_memory(agent_id: str, limit: int = 50) -> MemoryQueryResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        episodes = agent.episodic.recent(limit=max(1, min(limit, 500)))
        return MemoryQueryResponse(agent_id=agent_id, episodes=episodes)

    @application.get("/agent/{agent_id}/memory/similar", response_model=MemorySimilarResponse)
    async def get_memory_similar(
        agent_id: str,
        top_k: int = 5,
        min_salience: float = 0.0,
    ) -> MemorySimilarResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        async with agent.lock:
            q = query_vector_from_state_bus(agent.state_bus)
            raw = agent.episodic.search_similar(
                q,
                top_k=max(1, min(top_k, 50)),
                min_salience=min_salience,
            )
        slim = [{k: v for k, v in row.items() if k != "embedding"} for row in raw]
        return MemorySimilarResponse(agent_id=agent_id, matches=slim)

    @application.post("/agent/{agent_id}/checkpoint", response_model=CheckpointResponse)
    async def checkpoint_agent(agent_id: str) -> CheckpointResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        backups_dir: Path = application.state.backups_dir
        backups_dir.mkdir(parents=True, exist_ok=True)
        path = backups_dir / f"agent_{agent_id}_checkpoint.json"
        brain_name: str | None = None
        async with agent.lock:
            payload = agent.checkpoint_payload()
            brain_name = agent.save_brain(backups_dir)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return CheckpointResponse(
            agent_id=agent_id, path=str(path), neural_brain=brain_name
        )

    @application.post("/agent/{agent_id}/restore", response_model=AgentStateResponse)
    async def restore_agent(agent_id: str) -> AgentStateResponse:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent")
        backups_dir: Path = application.state.backups_dir
        path = backups_dir / f"agent_{agent_id}_checkpoint.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Checkpoint file not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        async with agent.lock:
            agent.apply_checkpoint_payload(payload)
            agent.load_brain(backups_dir)
            snapshot = agent.snapshot_state()
        return AgentStateResponse(agent_id=agent_id, payload=snapshot)

    @application.websocket("/agent/{agent_id}/cycle")
    async def agent_cycle_socket(websocket: WebSocket, agent_id: str) -> None:
        registry: AgentRegistry = application.state.registry
        agent = registry.get(agent_id)
        if agent is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        agent.ensure_cycle_worker()

        async def sender() -> None:
            while True:
                msg = await agent.out_queue.get()
                await websocket.send_json(msg)

        sender_task = asyncio.create_task(sender(), name=f"ws-sender-{agent_id}")
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("bad_json agent_id=%s", agent_id)
                    continue
                try:
                    obs = ObservationMessage.model_validate(data)
                except Exception:
                    logger.warning("validation_error agent_id=%s", agent_id)
                    continue
                await agent.handle_observation_dict(obs.model_dump())
        except WebSocketDisconnect:
            logger.info("ws_disconnect agent_id=%s", agent_id)
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    register_vast_routes(application)
    register_saved_agents_routes(application)
    register_preset_routes(application)
    register_skill_dojo_routes(application)

    return application


app = create_app()
