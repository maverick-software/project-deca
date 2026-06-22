import asyncio
import json
from pathlib import Path

import pytest

from decadic.training.skills import get_skill, list_skills
from decadic.training.store import SkillValidationError, UploadedSkillStore, parse_skill
from decadic.training.supervisor import SkillDojoError, SkillDojoSupervisor
from decadic.training.teachers import StandTeacher
from decadic.training.gates import Criterion
from decadic.training.types import SkillGate, SkillPhase, SkillSpec, TeacherAdaptation
from decadic.agents.runtime import AgentRuntime


torch = pytest.importorskip("torch")


class _Viability:
    value = 100.0


class FakeAgent:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.metrics = {
            "forward_model_error": 0.01,
            "tactile_pred_error": 0.01,
            "fall_rate": 0.0,
            "rom_mean": 0.1,
            "brace_engaged": 0.0,
            "root_height": 1.3,
            "torso_tilt": 0.0,
            "stance_phase": 0.0,
            "movement_hold": False,
            "braces_enabled": False,
            "foot_load_l": 0.0,
            "foot_load_r": 0.0,
            "hand_load_l": 0.0,
            "hand_load_r": 0.0,
            "teacher_motor_agreement": 1.0,
            "teacher_support_active": False,
            "teacher_support_force": 0.0,
            "teacher_support_torque": 0.0,
            "teacher_drop_m": 0.0,
            "teacher_target_drop_m": 0.25,
            "teacher_height_error_m": 0.0,
            "teacher_vertical_velocity": 0.0,
            "teacher_support_mode": "off",
        }
        self.viability = _Viability()
        self.configure_calls = []
        self.body_commands = []
        self.dojo_training = None
        self.saved = 0
        self.neural = None
        self.status = "alive"
        self.revived = 0

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        return {"ok": True}

    def queue_body_command(self, command: str) -> bool:
        self.body_commands.append(command)
        return True

    def checkpoint_payload(self):
        return {"agent": "fake"}

    def save_brain(self, backups_dir):
        self.saved += 1
        return "brain.pt"

    def revive(self):
        self.status = "alive"
        self.revived += 1


class FakeRegistry:
    def __init__(self, agent):
        self.agent = agent

    def get(self, agent_id):
        return self.agent if agent_id == "A" else None


def sample_uploaded_skill(skill_id="mini_recover"):
    return {
        "skill_id": skill_id,
        "version": "1.0",
        "name": "Mini Recover",
        "description": "A short uploaded stand-and-recover variant for tests.",
        "target_behavior": "Return to upright after a small perturbation.",
        "teacher": "stand_teacher",
        "required_sensors": ["proprioception", "contacts"],
        "checkpoint_on_graduate": False,
        "phases": [
            {
                "index": 0,
                "name": "Assisted Upright",
                "description": "Collect stable upright samples without manual braces.",
                "teacher_weight": 0.75,
                "config": {"viability_mode": "metabolic", "motor_babble_sigma": 0.0},
                "body_commands": ["set_stance:stand", "recenter"],
                "reset_commands": ["set_stance:stand", "recenter"],
                "timeout_s": 30.0,
                "max_attempts": 4,
                "auto_retry": True,
                "failure_criteria": [
                    {
                        "key": "root_height",
                        "comparator": "<=",
                        "threshold": 0.5,
                        "label": "root collapsed",
                        "unit": "m",
                    }
                ],
                "min_dwell_s": 0.0,
                "gate": {
                    "min_samples": 2,
                    "criteria": [
                        {
                            "key": "root_height",
                            "comparator": ">=",
                            "threshold": 1.0,
                            "label": "root high",
                            "unit": "m",
                        }
                    ],
                },
            },
            {
                "index": 1,
                "name": "Autonomous Check",
                "description": "Run without teacher hints.",
                "teacher_weight": 0.0,
                "body_commands": ["perturb:small"],
                "periodic_body_commands": [{"command": "perturb:small", "period_s": 1.0}],
                "demote_on_death": True,
                "min_dwell_s": 0.0,
                "is_terminal": True,
                "gate": {
                    "min_samples": 2,
                    "criteria": [
                        {
                            "key": "fall_rate",
                            "comparator": "<=",
                            "threshold": 0.05,
                            "label": "low falls",
                            "unit": "",
                        }
                    ],
                },
            },
        ],
    }


def test_skill_catalog_contains_stand_and_recover():
    skill = get_skill("stand_and_recover")
    assert skill is not None
    assert skill.phases[-1].teacher_weight == 0.0
    assert any(s["skill_id"] == "stand_and_recover" for s in list_skills())
    assert get_skill("developmental_locomotion") is not None
    assert get_skill("affective_locomotion") is not None


def test_stand_teacher_is_deterministic_neutral_target():
    assert StandTeacher().motor_target(n_actuators=4, metrics={}) == [0.0, 0.0, 0.0, 0.0]


def test_uploaded_skill_store_roundtrip_and_validation(tmp_path):
    store = UploadedSkillStore(tmp_path)
    saved = store.save(sample_uploaded_skill())
    assert saved["skill_id"] == "mini_recover"
    assert saved["source"] == "uploaded"
    assert saved["builtin"] is False
    assert store.get("mini_recover").phases[0].teacher_weight == 0.75
    assert store.get("mini_recover").phases[0].reset_commands[-1] == "recenter"
    assert store.get("mini_recover").phases[0].timeout_s == 30.0
    assert store.get("mini_recover").phases[0].max_attempts == 4
    assert store.get("mini_recover").phases[0].auto_retry is True
    assert store.get("mini_recover").phases[0].teacher_adaptation.max_weight == 0.75
    assert store.get("mini_recover").phases[0].failure_gate.criteria[0].key == "root_height"
    assert store.get("mini_recover").phases[1].periodic_body_commands[0].command == "perturb:small"
    assert store.get("mini_recover").phases[1].demote_on_death is True
    assert any(s["skill_id"] == "mini_recover" for s in store.list_all())

    with pytest.raises(SkillValidationError):
        bad = sample_uploaded_skill("bad_command")
        bad["phases"][0]["body_commands"] = ["shell:rm"]
        store.save(bad)

    with pytest.raises(SkillValidationError):
        bad = sample_uploaded_skill("bad_scaffold")
        bad["phases"][0]["body_commands"] = ["braces_on"]
        store.save(bad)

    with pytest.raises(SkillValidationError):
        bad = sample_uploaded_skill("bad_periodic")
        bad["phases"][1]["periodic_body_commands"] = [{"command": "shell:rm", "period_s": 1}]
        store.save(bad)

    assert store.delete("mini_recover") is True
    assert store.get("mini_recover") is None


def test_uploaded_skill_store_migrates_legacy_scaffold_commands(tmp_path):
    store = UploadedSkillStore(tmp_path)
    raw = sample_uploaded_skill("legacy_recover")
    raw["phases"][0]["body_commands"] = ["set_stance:stand", "braces_on", "hold_off"]
    raw["phases"][0]["reset_commands"] = ["set_stance:stand", "reset_braces", "recenter"]
    (tmp_path / "legacy_recover.json").write_text(json.dumps(raw), encoding="utf-8")

    spec = store.get("legacy_recover")
    assert spec is not None
    assert spec.phases[0].body_commands == ("set_stance:stand",)
    assert spec.phases[0].reset_commands == ("set_stance:stand", "recenter")
    assert spec.warnings


def test_packaged_uploadable_skills_validate():
    skill_dir = Path("docs/dojo_skills")
    for path in skill_dir.glob("*.json"):
        spec = parse_skill(json.loads(path.read_text(encoding="utf-8")))
        assert spec.skill_id
        assert spec.phases[-1].teacher_weight == 0.0
        assert spec.phases[-1].timeout_s > 0.0


def retry_skill(*, timeout_s=0.0, max_attempts=3) -> SkillSpec:
    return SkillSpec(
        skill_id="retry_skill",
        version="1.0",
        name="Retry Skill",
        description="Test retry lifecycle.",
        target_behavior="Stand.",
        teacher="stand_teacher",
        checkpoint_on_graduate=False,
        phases=(
            SkillPhase(
                index=0,
                name="Retry Phase",
                description="Retry until root is high.",
                teacher_weight=0.0,
                body_commands=("set_stance:stand",),
                reset_commands=("set_stance:stand", "recenter"),
                gate=SkillGate(
                    (Criterion("root_height", ">=", 1.0, "root high", "m"),),
                    min_samples=2,
                ),
                failure_gate=SkillGate(
                    (Criterion("root_height", "<=", 0.5, "root collapsed", "m"),),
                    min_samples=1,
                ),
                timeout_s=timeout_s,
                max_attempts=max_attempts,
                auto_retry=True,
                min_dwell_s=0.0,
            ),
            SkillPhase(
                index=1,
                name="Done",
                description="Terminal check.",
                teacher_weight=0.0,
                gate=SkillGate((Criterion("root_height", ">=", 1.0, "root high", "m"),), min_samples=1),
                timeout_s=timeout_s,
                max_attempts=max_attempts,
                auto_retry=True,
                min_dwell_s=0.0,
                is_terminal=True,
            ),
        ),
    )


def test_failure_criteria_retries_attempt_and_resets_body(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.4
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: retry_skill(max_attempts=10),
    )

    async def go():
        await sup.start("A", "retry_skill")
        await asyncio.sleep(0.04)
        st = sup.status()
        assert st["state"] == "running"
        assert st["attempt_failures"] >= 1
        assert st["attempt_index"] >= 2
        assert st["failure_reason"] == "root collapsed"
        assert "recenter" in agent.body_commands
        await sup.stop()

    asyncio.run(go())


def test_exhausted_retries_marks_run_failed_and_clears_teacher(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.4
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: retry_skill(max_attempts=1),
    )

    async def go():
        await sup.start("A", "retry_skill")
        await asyncio.sleep(0.03)
        st = sup.status()
        assert st["state"] == "failed"
        assert st["running"] is False
        assert st["failure_reason"] == "root collapsed"
        assert agent.dojo_training is None

    asyncio.run(go())


def test_timeout_retries_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.8
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: retry_skill(timeout_s=0.02, max_attempts=3),
    )

    async def go():
        await sup.start("A", "retry_skill")
        await asyncio.sleep(0.05)
        st = sup.status()
        assert st["state"] == "running"
        assert st["last_attempt_outcome"] == "timeout"
        assert st["attempt_failures"] >= 1
        await sup.stop()

    asyncio.run(go())


def test_successful_retry_promotes_to_next_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.4
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: retry_skill(max_attempts=3),
    )

    async def go():
        await sup.start("A", "retry_skill")
        await asyncio.sleep(0.02)
        agent.metrics["root_height"] = 1.2
        await asyncio.sleep(0.04)
        st = sup.status()
        assert st["phase_index"] == 1
        assert st["last_attempt_outcome"] != "failed"
        await sup.stop()

    asyncio.run(go())


def test_manual_scaffold_blocks_graduation_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["braces_enabled"] = True
    spec = SkillSpec(
        skill_id="pure_balance",
        version="1.0",
        name="Pure Balance",
        description="Graduate only without manual scaffold.",
        target_behavior="Stand.",
        teacher="stand_teacher",
        checkpoint_on_graduate=False,
        phases=(
            SkillPhase(
                index=0,
                name="Gate",
                description="Root high.",
                teacher_weight=0.0,
                gate=SkillGate((Criterion("root_height", ">=", 1.0, "root high", "m"),), min_samples=1),
                min_dwell_s=0.0,
                is_terminal=True,
            ),
        ),
    )
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: spec,
    )

    async def go():
        await sup.start("A", "pure_balance")
        await asyncio.sleep(0.03)
        st = sup.status()
        assert st["state"] == "running"
        assert st["manual_scaffold_active"] is True
        assert st["attempt_failures"] == 0
        agent.metrics["braces_enabled"] = False
        await asyncio.sleep(0.03)
        assert sup.status()["state"] == "graduated"

    asyncio.run(go())


def adaptive_skill(*, teacher_weight=0.0, is_terminal=False) -> SkillSpec:
    return SkillSpec(
        skill_id="adaptive_skill",
        version="1.0",
        name="Adaptive Skill",
        description="Exercise adaptive teacher control.",
        target_behavior="Stand.",
        teacher="stand_teacher",
        checkpoint_on_graduate=False,
        phases=(
            SkillPhase(
                index=0,
                name="Adaptive Phase",
                description="Adapt assist.",
                teacher_weight=teacher_weight,
                teacher_adaptation=TeacherAdaptation(
                    enabled=not is_terminal,
                    min_weight=0.0,
                    max_weight=1.0 if not is_terminal else 0.0,
                    rise_rate=25.0,
                    fade_rate=25.0,
                    stable_dwell_s=0.0,
                    unstable_dwell_s=0.0,
                    danger_thresholds={
                        "root_height_min": 1.0,
                        "torso_tilt_max": 0.5,
                        "fall_rate_max": 0.2,
                    },
                    stability_thresholds={
                        "root_height_min": 1.1,
                        "torso_tilt_max": 0.3,
                        "fall_rate_max": 0.05,
                    },
                ),
                gate=SkillGate((Criterion("root_height", ">=", 1.0, "root high", "m"),), min_samples=1),
                min_dwell_s=0.1 if not is_terminal else 0.0,
                is_terminal=is_terminal,
            ),
        ),
    )


def test_adaptive_teacher_increases_when_posture_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.6
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: adaptive_skill(teacher_weight=0.0),
    )

    async def go():
        await sup.start("A", "adaptive_skill")
        await asyncio.sleep(0.04)
        st = sup.status()
        assert st["teacher_assist"] > 0.0
        assert st["assist_reason"] == "root height dropping"
        assert agent.dojo_training["demo_weight"] == st["teacher_assist"]
        assert agent.dojo_training["origin"] in {"dagger", "demo"}
        await sup.stop()

    asyncio.run(go())


def test_adaptive_teacher_increases_on_partial_root_drop_before_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 1.12
    agent.metrics["teacher_drop_m"] = 0.27
    agent.metrics["teacher_target_drop_m"] = 0.25
    agent.metrics["teacher_support_mode"] = "recover"
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: adaptive_skill(teacher_weight=0.0),
    )

    async def go():
        await sup.start("A", "adaptive_skill")
        await asyncio.sleep(0.04)
        st = sup.status()
        assert st["teacher_assist"] > 0.0
        assert st["assist_reason"] == "recovering height"
        assert st["teacher_drop_m"] == pytest.approx(0.27)
        await sup.stop()

    asyncio.run(go())


def test_adaptive_teacher_fades_after_stable_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: adaptive_skill(teacher_weight=1.0),
    )

    async def go():
        await sup.start("A", "adaptive_skill")
        start = sup.status()["teacher_assist"]
        await asyncio.sleep(0.04)
        st = sup.status()
        assert start == 1.0
        assert st["teacher_assist"] < start
        assert st["assist_reason"] == "stable, fading"
        await sup.stop()

    asyncio.run(go())


def test_adaptive_teacher_does_not_fade_outside_height_band(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["teacher_drop_m"] = 0.28
    agent.metrics["teacher_target_drop_m"] = 0.25
    agent.metrics["teacher_height_error_m"] = 0.03
    agent.metrics["teacher_support_mode"] = "recover"
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: adaptive_skill(teacher_weight=1.0),
    )

    async def go():
        await sup.start("A", "adaptive_skill")
        await asyncio.sleep(0.04)
        st = sup.status()
        assert st["teacher_assist"] == pytest.approx(1.0)
        assert st["assist_reason"] == "recovering height"
        await sup.stop()

    asyncio.run(go())


def test_terminal_phase_forces_teacher_assist_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["root_height"] = 0.4
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: adaptive_skill(teacher_weight=1.0, is_terminal=True),
    )

    async def go():
        await sup.start("A", "adaptive_skill")
        await asyncio.sleep(0.02)
        st = sup.status()
        assert st["teacher_assist"] == 0.0
        assert agent.dojo_training["demo_weight"] == 0.0
        assert agent.metrics["teacher_override_fraction"] == 0.0
        await sup.stop()

    asyncio.run(go())


def test_live_teacher_preserves_student_motor_action_for_body():
    agent = object.__new__(AgentRuntime)
    agent.metrics = {}
    agent.dojo_training = {
        "expert_motor": [1.0, -1.0],
        "demo_weight": 0.5,
        "origin": "dagger",
        "assist_reason": "test assist",
        "objective_confidence": 0.25,
        "confidence_reason": "building",
        "teacher_live": True,
    }
    outbound = {"action": {"type": "motor", "parameters": {"ctrl": [-1.0, 1.0]}}}

    AgentRuntime._apply_live_teacher(agent, outbound)

    params = outbound["action"]["parameters"]
    assert params["student_ctrl"] == [-1.0, 1.0]
    assert params["teacher_ctrl"] == [1.0, -1.0]
    assert params["ctrl"] == [-1.0, 1.0]
    assert params["teacher_assist"] == 0.5
    assert params["teacher_origin"] == "dagger"
    assert params["teacher_live"] is True
    assert params["assist_reason"] == "test assist"
    assert params["objective_confidence"] == 0.25
    assert agent.metrics["teacher_motor_agreement"] == 0.0
    assert agent.metrics["teacher_live_assist"] == 0.5


def test_live_teacher_full_assist_never_zeros_student_motor_action():
    agent = object.__new__(AgentRuntime)
    agent.metrics = {}
    agent.dojo_training = {
        "expert_motor": [0.0, 0.0],
        "demo_weight": 1.0,
        "origin": "demo",
        "teacher_live": True,
    }
    outbound = {"action": {"type": "motor", "parameters": {"ctrl": [0.4, -0.7]}}}

    AgentRuntime._apply_live_teacher(agent, outbound)

    params = outbound["action"]["parameters"]
    assert params["student_ctrl"] == [0.4, -0.7]
    assert params["teacher_ctrl"] == [0.0, 0.0]
    assert params["ctrl"] == [0.4, -0.7]
    assert params["teacher_assist"] == 1.0
    assert params["teacher_live"] is True


def test_live_teacher_assist_zero_keeps_student_action():
    agent = object.__new__(AgentRuntime)
    agent.metrics = {}
    agent.dojo_training = {
        "expert_motor": [1.0, -1.0],
        "demo_weight": 0.0,
        "origin": "self",
        "teacher_live": False,
    }
    outbound = {"action": {"type": "motor", "parameters": {"ctrl": [-0.25, 0.75]}}}

    AgentRuntime._apply_live_teacher(agent, outbound)

    params = outbound["action"]["parameters"]
    assert params["student_ctrl"] == [-0.25, 0.75]
    assert params["teacher_ctrl"] == [1.0, -1.0]
    assert params["ctrl"] == [-0.25, 0.75]
    assert params["teacher_assist"] == 0.0
    assert params["teacher_live"] is False
    assert agent.metrics["teacher_motor_agreement"] == pytest.approx(0.25)
    assert agent.metrics["teacher_live_assist"] == 0.0


def test_stance_phase_is_sampled_for_uploaded_skill_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics["stance_phase"] = 1.0
    agent.metrics["root_height"] = 1.2
    agent.metrics["fall_rate"] = 0.0
    spec = parse_skill(json.loads(Path("docs/dojo_skills/stand_up_from_floor_balance.json").read_text(encoding="utf-8")))
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: spec,
    )

    async def go():
        await sup.start("A", "stand_up_from_floor_balance")
        await sup.set_phase(1)
        await asyncio.sleep(0.04)
        assert sup.status()["gate"]["criteria"][1]["key"] == "stance_phase"
        assert sup.status()["gate"]["criteria"][1]["satisfied"] is True
        await sup.stop()

    asyncio.run(go())


def test_supervisor_start_applies_phase_and_teacher_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    sup = SkillDojoSupervisor(
        FakeRegistry(agent), backups_dir=tmp_path / "backups", log_dir=tmp_path / "logs"
    )

    async def go():
        st = await sup.start("A", "stand_and_recover")
        assert st["state"] == "running"
        assert st["phase_index"] == 0
        assert agent.configure_calls[0]["viability_mode"] == "metabolic"
        assert st["caregiver_enabled"] is True
        assert "set_stance:stand" in agent.body_commands
        assert agent.dojo_training["skill_id"] == "stand_and_recover"
        assert agent.dojo_training["demo_weight"] == 1.0
        await sup.set_phase(3)
        assert agent.dojo_training["demo_weight"] == 0.0
        assert agent.metrics["teacher_override_fraction"] == 0.0
        stopped = await sup.stop()
        assert stopped["state"] == "stopped"
        assert agent.dojo_training is None

    asyncio.run(go())


def caregiver_skill() -> SkillSpec:
    return SkillSpec(
        skill_id="caregiver_skill",
        version="1.0",
        name="Caregiver Skill",
        description="Exercises caregiver monitor.",
        target_behavior="Stay alive while training.",
        teacher="stand_teacher",
        caregiver_enabled=True,
        caregiver_threshold=80.0,
        checkpoint_on_graduate=False,
        phases=(
            SkillPhase(
                index=0,
                name="Care",
                description="Needs visible support.",
                teacher_weight=0.0,
                gate=SkillGate((Criterion("root_height", ">=", 1.0, "root high", "m"),), min_samples=1),
                min_dwell_s=0.0,
                is_terminal=True,
            ),
        ),
    )


def test_caregiver_monitor_requests_lowest_reservoir(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics.update(
        {
            "hydration": 79.0,
            "energy": 90.0,
            "integrity": 95.0,
            "caregiver_parent_present": True,
        }
    )
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: caregiver_skill(),
    )

    async def go():
        await sup.start("A", "caregiver_skill")
        await asyncio.sleep(0.03)
        st = sup.status()
        assert "parent_enable" in agent.body_commands
        assert "parent_request:water" in agent.body_commands
        assert not any(cmd.startswith("give_") for cmd in agent.body_commands)
        assert st["caregiver_status"] in {"requested", "refractory"}
        assert st["caregiver_need"] == "hydration"
        assert st["state"] == "running"
        await sup.stop()

    asyncio.run(go())


@pytest.mark.parametrize(
    ("reservoir", "command"),
    [
        ("hydration", "parent_request:water"),
        ("energy", "parent_request:food"),
        ("integrity", "parent_request:care"),
    ],
)
def test_caregiver_monitor_maps_reservoir_to_parent_request(tmp_path, reservoir, command):
    agent = FakeAgent()
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: caregiver_skill(),
    )
    sup._skill = caregiver_skill()
    sup._caregiver_enabled = True
    sup._caregiver_threshold = 80.0
    sample = {
        "hydration": 95.0,
        "energy": 95.0,
        "integrity": 95.0,
        "caregiver_parent_present": 1.0,
        "caregiver_delivery_count": 0.0,
        "caregiver_status": "",
    }
    sample[reservoir] = 79.0

    sup._update_caregiver(agent, sample)

    assert command in agent.body_commands
    assert "parent_enable" in agent.body_commands
    assert not any(cmd.startswith("give_") for cmd in agent.body_commands)


def test_caregiver_monitor_reports_missing_parent_and_blocks_graduation(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    agent.metrics.update(
        {
            "hydration": 79.0,
            "energy": 90.0,
            "integrity": 95.0,
            "caregiver_parent_present": False,
            "caregiver_missing_parent": True,
        }
    )
    sup = SkillDojoSupervisor(
        FakeRegistry(agent),
        backups_dir=tmp_path / "backups",
        skill_loader=lambda sid: caregiver_skill(),
    )

    async def go():
        await sup.start("A", "caregiver_skill")
        await asyncio.sleep(0.03)
        st = sup.status()
        assert st["caregiver_missing_parent"] is True
        assert st["caregiver_status"] == "caregiver_missing_parent"
        assert st["state"] == "running"
        assert "parent_request:water" not in agent.body_commands
        await sup.stop()

    asyncio.run(go())


def test_supervisor_periodic_commands_and_death_demote(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DOJO_POLL_S", "0.01")
    agent = FakeAgent()
    sup = SkillDojoSupervisor(
        FakeRegistry(agent), backups_dir=tmp_path / "backups", log_dir=tmp_path / "logs"
    )

    async def go():
        await sup.start("A", "developmental_locomotion")
        await sup.set_phase(2)
        await asyncio.sleep(0.03)
        assert "give_food_near" in agent.body_commands
        assert "give_water_near" in agent.body_commands
        agent.status = "dead"
        await asyncio.sleep(0.03)
        st = sup.status()
        assert agent.revived >= 1
        assert st["phase_index"] == 1
        await sup.stop()

    asyncio.run(go())


def test_supervisor_rejects_unknown_skill(tmp_path):
    agent = FakeAgent()
    sup = SkillDojoSupervisor(FakeRegistry(agent), backups_dir=tmp_path / "backups")

    async def go():
        with pytest.raises(SkillDojoError):
            await sup.start("A", "nope")

    asyncio.run(go())


def test_dojo_routes_smoke(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app) as client:
        skills = client.get("/dojo/skills")
        assert skills.status_code == 200
        assert any(s["skill_id"] == "stand_and_recover" for s in skills.json()["skills"])
        missing = client.post("/dojo/start", json={"agent_id": "missing", "skill_id": "stand_and_recover"})
        assert missing.status_code == 409


def test_dojo_skill_upload_routes(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app) as client:
        upload = client.post("/dojo/skills/upload", json=sample_uploaded_skill("uploaded_balance"))
        assert upload.status_code == 200
        assert upload.json()["source"] == "uploaded"

        listed = client.get("/dojo/skills")
        assert listed.status_code == 200
        assert any(s["skill_id"] == "uploaded_balance" for s in listed.json()["skills"])

        fetched = client.get("/dojo/skills/uploaded_balance")
        assert fetched.status_code == 200
        assert fetched.json()["builtin"] is False

        protected = client.delete("/dojo/skills/stand_and_recover")
        assert protected.status_code == 409

        deleted = client.delete("/dojo/skills/uploaded_balance")
        assert deleted.status_code == 200
        assert client.get("/dojo/skills/uploaded_balance").status_code == 404


def test_replay_transition_metadata_and_imitation_loss(monkeypatch):
    from decadic.consolidation.consolidator import replay_batch_loss
    from decadic.consolidation.replay_buffer import Transition
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    cfg = neural_config_from_env("tiny")
    stack = NeuralCognitiveStack(cfg)
    z0 = torch.zeros(1, cfg.d_model)
    ep = torch.zeros(1, 4)
    mem = torch.zeros(1, cfg.memory_context_dim)
    prev_state = torch.zeros(1, cfg.d_model)
    prev_motor = torch.zeros(1, cfg.n_actuators)
    proprio = torch.zeros(1, cfg.forward_pred_dim)
    base = Transition(z0, ep, mem, prev_state, prev_motor, proprio, salience=1.0)
    hinted = Transition(
        z0,
        ep,
        mem,
        prev_state,
        prev_motor,
        proprio,
        salience=1.0,
        skill_id="stand_and_recover",
        origin="demo",
        expert_motor=[1.0] * cfg.n_actuators,
        demo_weight=1.0,
    )
    no_hint = float(replay_batch_loss(stack, [base], torch.device("cpu")).detach())
    with_hint = float(replay_batch_loss(stack, [hinted], torch.device("cpu")).detach())
    assert hinted.skill_id == "stand_and_recover"
    assert with_hint > no_hint
