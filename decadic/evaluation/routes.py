"""FastAPI routes for training-evaluation scenarios, jobs, and reports."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from decadic.agents.registry import AgentRegistry
from decadic.api.environment import EnvironmentSupervisor
from decadic.evaluation.runner import (
    SCENARIO_DIR,
    build_report,
    load_eval_spec,
    report_path_for,
    write_samples_jsonl,
)
from decadic.evaluation.sampling import normalize_eval_metrics, target_end_cycle
from decadic.evaluation.types import EvalReport, EvalSample, EvalSpec
from decadic.nn.config import VALID_PRESETS
from decadic.training.supervisor import SkillDojoError, SkillDojoSupervisor


REPORT_DIR = Path("reports")


class EvalStartRequest(BaseModel):
    scenario: str
    cycles: int | None = None
    seeds: list[int] | None = None
    preset: str | None = None
    dojo_skill_id: str | None = None
    poll_interval_s: float | None = None
    timeout_s: float | None = None
    agent_id: str | None = None


def _validate_preset(preset: str | None) -> str | None:
    if preset is None:
        return None
    p = str(preset).strip().lower()
    if p not in VALID_PRESETS:
        raise HTTPException(status_code=422, detail=f"Unknown preset {preset!r}")
    return p


def _safe_report_id(report_id: str) -> str:
    stem = Path(report_id).stem
    if not stem or stem != report_id:
        raise HTTPException(status_code=422, detail="Invalid report id")
    return stem


def _scenario_summary(spec: EvalSpec) -> dict[str, Any]:
    return {
        **spec.to_dict(),
        "body_required": bool(spec.dojo_skill_id or "resource" in spec.scenario or "stand" in spec.scenario),
        "estimated_runtime_s": float(spec.timeout_s),
    }


def _report_summary(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "report_id": path.stem,
        "path": str(path),
        "scenario": raw.get("scenario", ""),
        "status": raw.get("status", "unknown"),
        "agent_id": raw.get("agent_id"),
        "failures_count": len(raw.get("failures") or []),
        "failures": list(raw.get("failures") or [])[:3],
        "samples_path": raw.get("samples_path", ""),
        "mtime": path.stat().st_mtime,
    }


class EvalJobManager:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        environment: EnvironmentSupervisor,
        dojo: SkillDojoSupervisor | None,
        report_dir: Path = REPORT_DIR,
    ) -> None:
        self.registry = registry
        self.environment = environment
        self.dojo = dojo
        self.report_dir = report_dir
        self._task: asyncio.Task[None] | None = None
        self._cancel = asyncio.Event()
        self._status: dict[str, Any] = {
            "state": "idle",
            "job_id": None,
            "scenario": None,
            "agent_id": None,
            "started_at": None,
            "elapsed_s": 0.0,
            "samples": 0,
            "cycles": 0,
            "target_cycles": 0,
            "report_path": "",
            "samples_path": "",
            "error": "",
            "body_connected": False,
            "body_warning": "",
        }

    def status(self) -> dict[str, Any]:
        out = dict(self._status)
        if out.get("started_at"):
            out["elapsed_s"] = round(time.time() - float(out["started_at"]), 3)
        env = self.environment.status()
        out["body_connected"] = bool(env.get("running", False))
        out["body_warning"] = "" if out["body_connected"] else "No running body/environment detected."
        return out

    async def start(self, req: EvalStartRequest) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            raise HTTPException(status_code=409, detail="An eval job is already running")
        spec = load_eval_spec(req.scenario)
        if req.cycles is not None:
            spec.cycles = max(1, int(req.cycles))
        if req.seeds:
            spec.seeds = [int(x) for x in req.seeds]
        if req.preset is not None:
            spec.agent_preset = _validate_preset(req.preset)
        if req.dojo_skill_id is not None:
            spec.dojo_skill_id = req.dojo_skill_id
        if req.poll_interval_s is not None:
            spec.poll_interval_s = max(0.25, float(req.poll_interval_s))
        if req.timeout_s is not None:
            spec.timeout_s = max(1.0, float(req.timeout_s))

        agent_id = req.agent_id
        if agent_id:
            if self.registry.get(agent_id) is None:
                raise HTTPException(status_code=404, detail="Unknown agent")
        else:
            agent_id = str(uuid.uuid4())
            self.registry.create_agent(agent_id, preset=_validate_preset(spec.agent_preset))

        self._cancel = asyncio.Event()
        job_id = f"{spec.scenario}_{int(time.time())}"
        self._status = {
            "state": "starting",
            "job_id": job_id,
            "scenario": spec.scenario,
            "agent_id": agent_id,
            "started_at": time.time(),
            "elapsed_s": 0.0,
            "samples": 0,
            "cycles": 0,
            "target_cycles": spec.cycles,
            "report_path": "",
            "samples_path": "",
            "error": "",
            "body_connected": bool(self.environment.status().get("running", False)),
            "body_warning": "",
        }
        self._task = asyncio.create_task(self._run(spec, agent_id), name=f"training-eval-{job_id}")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        if self._task is None or self._task.done():
            self._status["state"] = "idle"
            return self.status()
        self._status["state"] = "stopping"
        self._cancel.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
        return self.status()

    async def _run(self, spec: EvalSpec, agent_id: str) -> None:
        samples: list[EvalSample] = []
        t0 = time.perf_counter()
        try:
            if spec.dojo_skill_id and self.dojo is not None:
                try:
                    await self.dojo.start(agent_id, spec.dojo_skill_id)
                except SkillDojoError as exc:
                    self._status["error"] = str(exc)
                    self._status["state"] = "failed"
                    return
            self._status["state"] = "running"
            deadline = time.perf_counter() + spec.timeout_s
            start_cycle: int | None = None
            end_cycle: int | None = None
            while time.perf_counter() < deadline and not self._cancel.is_set():
                agent = self.registry.get(agent_id)
                if agent is None:
                    raise RuntimeError("Agent disappeared during eval")
                metrics = dict(agent.metrics)
                metrics["viability"] = agent.viability.value
                metrics["paused"] = agent.paused
                metrics["status"] = agent.status
                metrics["preset"] = agent.preset
                metrics["cycles_completed"] = int(agent.state_bus.cycle_index)
                discovery = {
                    "discovery_health": dict(getattr(agent.perceptual, "discovery_health", {}) or {}),
                    "discovery": agent.perceptual.discovery_eval.snapshot(),
                    "object_files": list(getattr(agent.perceptual, "object_files", []) or []),
                    "ltm_consolidation": dict(getattr(agent.perceptual, "ltm_consolidation", {}) or {}),
                }
                dojo_status = self.dojo.status() if self.dojo is not None else None
                metrics = normalize_eval_metrics(metrics, discovery, dojo_status)
                sample = EvalSample(
                    cycle=int(metrics.get("cycles_completed", 0) or 0),
                    t_s=round(time.perf_counter() - t0, 6),
                    metrics=metrics,
                    discovery=discovery,
                    dojo=dojo_status,
                )
                samples.append(sample)
                if start_cycle is None:
                    start_cycle = sample.cycle
                    end_cycle = target_end_cycle(start_cycle, spec.cycles)
                    self._status["start_cycle"] = start_cycle
                    self._status["target_end_cycle"] = end_cycle
                self._status["samples"] = len(samples)
                self._status["cycles"] = sample.cycle
                self._status["observed_cycles"] = max(0, sample.cycle - int(start_cycle or sample.cycle))
                if end_cycle is not None and sample.cycle >= end_cycle:
                    break
                await asyncio.sleep(spec.poll_interval_s)

            out_path = report_path_for(spec.scenario, self.report_dir)
            samples_path = out_path.with_suffix(".jsonl")
            write_samples_jsonl(samples, samples_path)
            report = build_report(
                spec=spec,
                samples=samples,
                agent_id=agent_id,
                samples_path=str(samples_path),
            )
            if self._cancel.is_set():
                report.status = "fail"
                report.failures.append("eval cancelled")
                self._status["state"] = "cancelled"
            elif samples and end_cycle is not None and samples[-1].cycle < end_cycle:
                report.status = "fail"
                observed = max(0, samples[-1].cycle - int(start_cycle or samples[-1].cycle))
                report.failures.append(f"target cycles not reached: observed {observed}/{spec.cycles}")
                self._status["state"] = "failed"
            else:
                self._status["state"] = "completed" if report.status == "pass" else "failed"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            self._status["report_path"] = str(out_path)
            self._status["samples_path"] = str(samples_path)
            self._status["error"] = "; ".join(report.failures[:3])
        except Exception as exc:
            self._status["state"] = "failed"
            self._status["error"] = str(exc)


def register_evaluation_routes(application: FastAPI) -> None:
    def _manager() -> EvalJobManager:
        mgr = getattr(application.state, "eval_jobs", None)
        if mgr is None:
            mgr = EvalJobManager(
                registry=application.state.registry,
                environment=application.state.environment,
                dojo=getattr(application.state, "skill_dojo", None),
            )
            application.state.eval_jobs = mgr
        return mgr

    @application.get("/eval/scenarios")
    async def eval_scenarios() -> dict[str, Any]:
        scenarios = []
        for path in sorted(SCENARIO_DIR.glob("*.json")):
            scenarios.append(_scenario_summary(load_eval_spec(path)))
        return {"scenarios": scenarios}

    @application.get("/eval/scenarios/{scenario_id}")
    async def eval_scenario(scenario_id: str) -> dict[str, Any]:
        try:
            return _scenario_summary(load_eval_spec(scenario_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown eval scenario") from exc

    @application.get("/eval/reports")
    async def eval_reports() -> dict[str, Any]:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        reports = [
            s
            for p in sorted(REPORT_DIR.glob("training_eval_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if (s := _report_summary(p)) is not None
        ]
        return {"reports": reports}

    @application.get("/eval/reports/{report_id}")
    async def eval_report(report_id: str) -> dict[str, Any]:
        stem = _safe_report_id(report_id)
        path = REPORT_DIR / f"{stem}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Unknown eval report")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Report could not be read") from exc

    @application.post("/eval/start")
    async def eval_start(req: EvalStartRequest) -> dict[str, Any]:
        return await _manager().start(req)

    @application.get("/eval/status")
    async def eval_status() -> dict[str, Any]:
        return _manager().status()

    @application.post("/eval/stop")
    async def eval_stop() -> dict[str, Any]:
        return await _manager().stop()
