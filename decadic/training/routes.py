"""FastAPI routes for Skill Dojo training runs."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from decadic.training.store import SkillValidationError, UploadedSkillStore
from decadic.training.supervisor import SkillDojoError, SkillDojoSupervisor


class SkillDojoStartRequest(BaseModel):
    agent_id: str
    skill_id: str
    auto_retry: bool | None = None
    max_attempts: int | None = None
    timeout_multiplier: float | None = None


class SkillDojoPhaseRequest(BaseModel):
    index: int


def register_skill_dojo_routes(application: FastAPI) -> None:
    """Attach Skill Dojo endpoints to ``application``."""

    def _dojo() -> SkillDojoSupervisor:
        sup = getattr(application.state, "skill_dojo", None)
        if sup is None:
            raise HTTPException(status_code=500, detail="Skill Dojo is not initialized")
        return sup

    def _store() -> UploadedSkillStore:
        store = getattr(application.state, "skill_store", None)
        if store is None:
            raise HTTPException(status_code=500, detail="Skill store is not initialized")
        return store

    @application.get("/dojo/skills")
    async def dojo_skills() -> dict[str, Any]:
        return {"skills": _store().list_all()}

    @application.get("/dojo/skills/{skill_id}")
    async def dojo_skill(skill_id: str) -> dict[str, Any]:
        skill = _store().get_any_dict(skill_id)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"unknown skill {skill_id!r}")
        return skill

    @application.post("/dojo/skills/upload")
    async def dojo_upload_skill(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return _store().save(body)
        except SkillValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.delete("/dojo/skills/{skill_id}")
    async def dojo_delete_skill(skill_id: str) -> dict[str, Any]:
        try:
            deleted = _store().delete(skill_id)
        except SkillValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail=f"unknown uploaded skill {skill_id!r}")
        return {"deleted": skill_id}

    @application.post("/dojo/start")
    async def dojo_start(req: SkillDojoStartRequest) -> dict[str, Any]:
        try:
            return await _dojo().start(
                req.agent_id,
                req.skill_id,
                auto_retry=req.auto_retry,
                max_attempts=req.max_attempts,
                timeout_multiplier=req.timeout_multiplier,
            )
        except SkillDojoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.get("/dojo/status")
    async def dojo_status() -> dict[str, Any]:
        return _dojo().status()

    @application.post("/dojo/pause")
    async def dojo_pause() -> dict[str, Any]:
        try:
            return _dojo().pause()
        except SkillDojoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/dojo/resume")
    async def dojo_resume() -> dict[str, Any]:
        try:
            return await _dojo().resume()
        except SkillDojoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/dojo/stop")
    async def dojo_stop() -> dict[str, Any]:
        return await _dojo().stop()

    @application.post("/dojo/phase")
    async def dojo_phase(req: SkillDojoPhaseRequest) -> dict[str, Any]:
        try:
            return await _dojo().set_phase(req.index)
        except SkillDojoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
