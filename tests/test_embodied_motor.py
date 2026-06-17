"""Embodied motor learning via active inference.

Covers the corrected sensorimotor loop: a motor PD-target head with real
gradient, the proprioceptive forward model, the active-inference losses, the
fading-assist / babble curriculum schedules, and the MuJoCo adapter's PD
tracking + assist fade.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod


def _bundle(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    from decadic.nn.bundle import NeuralBundle

    bundle = NeuralBundle.try_build("unit-motor")
    assert bundle is not None
    return bundle


def _body_obs(i: int) -> dict:
    import math

    return {
        "timestamp": f"t{i}",
        "proprioception": {
            "position": [0.0, 0.0, 1.2 + 0.01 * i],
            "orientation": [0.1, 0.05, 0.2],
            "velocity": [0.01, 0.0, 0.0],
            "current_action": "mujoco_humanoid:active_inference",
            "joints": [0.05 * math.sin(i + j) for j in range(34)],
            "contacts": [120.0, 110.0, 0.0, 0.0],
            # Full-body soft per-part loads (16 channels) -> tactile world model target.
            "part_loads": [0.5 + 0.4 * math.sin(i + k) for k in range(16)],
        },
        "events": [],
    }


def _run(monkeypatch, n: int, *, assist_override: float | None = None):
    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    bundle = _bundle(monkeypatch)
    bus, percept, ep = StateBus(), PerceptualState(), EpisodicStore(None)
    # Mild deprivation so the always-on homeostatic drive pulls the motor head.
    homeo = Homeostasis(hydration=60.0, energy=100.0, integrity=100.0)
    via = ViabilityState(value=homeo.viability)
    out = None
    for i in range(n):
        ctx = CycleContext(
            state_bus=bus,
            perceptual=percept,
            viability=via,
            episodic=ep,
            homeostasis=homeo,
            last_observation=_body_obs(i),
            pending_observations=[_body_obs(i)],
            assist_override=assist_override,
        )
        out = run_neural_cycle(ctx, bundle)
    return bundle, out


# --- Brain: motor head + forward model -------------------------------------


def test_motor_head_and_forward_model_shapes(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.frozen_encoders import CLIP_POOL_DIM, WHISPER_POOL_DIM
    from decadic.nn.neural_stack import NeuralCognitiveStack

    cfg = neural_config_from_env("tiny")
    stack = NeuralCognitiveStack(cfg)
    fused = CLIP_POOL_DIM + WHISPER_POOL_DIM + cfg.proprio_emb
    z0 = stack.ingress(torch.zeros(1, fused))
    out = stack(z0, torch.zeros(1, 4), torch.zeros(1, cfg.memory_context_dim))
    assert out["motor_u"].shape == (1, cfg.n_actuators)
    assert out["s_hat"].shape == (1, cfg.forward_pred_dim)
    # detached-params prediction has the same shape and does not error.
    pred = stack.forward_predict(out["z5"], out["motor_u"], detach_params=True)
    assert pred.shape == (1, cfg.forward_pred_dim)
    # Tactile forward-model head: a separate prediction of the next per-part load.
    # It does NOT change the proprio forward dim (28) or the motor head.
    import decadic.config as C

    assert stack.has_tactile_model is True
    assert stack.fwd_tactile_l2.out_features == C.TACTILE_PRED_DIM
    assert out["t_hat"].shape == (1, C.TACTILE_PRED_DIM)
    t_pred = stack.forward_predict_tactile(out["z5"], out["motor_u"], detach_params=True)
    assert t_pred.shape == (1, C.TACTILE_PRED_DIM)
    assert cfg.forward_pred_dim == 28  # proprio forward head is untouched


def test_motor_head_receives_gradient(monkeypatch):
    """The efferent head learns via the homeostatic drive-reduction objective."""
    bundle, out = _run(monkeypatch, n=3)
    grad = bundle.stack.motor[-1].weight.grad
    assert grad is not None
    assert float(grad.norm()) > 0.0
    # The world model also learns from realized transitions.
    fwd_grad = bundle.stack.fwd_l2.weight.grad
    assert fwd_grad is not None
    assert float(fwd_grad.norm()) > 0.0
    # The tactile world model also learns which actions load which body part.
    tac_grad = bundle.stack.fwd_tactile_l2.weight.grad
    assert tac_grad is not None
    assert float(tac_grad.norm()) > 0.0


def test_cycle_emits_motor_action_and_telemetry(monkeypatch):
    import decadic.config as C

    bundle, out = _run(monkeypatch, n=2)
    assert out["action"]["type"] == "motor"
    params = out["action"]["parameters"]
    assert len(params["ctrl"]) == bundle.cfg.n_actuators
    assert 0.0 <= params["assist_gain"] <= 1.0
    # Per-joint forward-model error rides on the motor command: it is exactly the
    # joint-tail of the proprio forward head, so one channel per predicted hinge.
    n_joint_pe = bundle.cfg.forward_pred_dim - int(C.CONTROLLABLE_PROPRIO_BASE)
    assert "joint_pe" in params
    assert len(params["joint_pe"]) == n_joint_pe
    assert all(float(x) >= 0.0 for x in params["joint_pe"])  # squared errors
    diag = out["_diagnostics"]
    for key in (
        "forward_model_error",
        "tactile_pred_error",
        "joint_pred_error",
        "assist_gain",
        "motor_babble_sigma",
        "motor_activity_rms",
        "motor_command",
    ):
        assert key in diag
    assert len(diag["motor_command"]) == bundle.cfg.n_actuators
    assert diag["joint_pred_error"] is not None and len(diag["joint_pred_error"]) == n_joint_pe
    # Tactile PE is a real (finite, non-negative) MSE once a body is streaming.
    assert diag["tactile_pred_error"] is not None
    assert diag["tactile_pred_error"] >= 0.0


def test_no_body_skips_active_inference(monkeypatch):
    """Without a streaming body, the AI losses are inert (no garbage training)."""
    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    bundle = _bundle(monkeypatch)
    ctx = CycleContext(
        state_bus=StateBus(),
        perceptual=PerceptualState(),
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        last_observation=None,
    )
    out = run_neural_cycle(ctx, bundle)
    assert out["_diagnostics"]["forward_model_error"] == 0.0
    assert "preferred_state_error" not in out["_diagnostics"]
    assert bundle.prev_state is None  # nothing buffered without a body


# --- Curriculum schedules ---------------------------------------------------


def test_assist_gain_schedule(monkeypatch):
    monkeypatch.setenv("DECADIC_ASSIST_DECAY_CYCLES", "100")
    from decadic.config import assist_gain_for_cycle

    assert assist_gain_for_cycle(0) == pytest.approx(1.0)
    assert assist_gain_for_cycle(50) == pytest.approx(0.5)
    assert assist_gain_for_cycle(100) == pytest.approx(0.0)
    assert assist_gain_for_cycle(9999) == pytest.approx(0.0)
    # monotonically non-increasing
    prev = 1.0
    for c in range(0, 120, 10):
        g = assist_gain_for_cycle(c)
        assert g <= prev + 1e-9
        prev = g


def test_motor_exploration_is_need_and_error_gated(monkeypatch):
    monkeypatch.setenv("DECADIC_MOTOR_BABBLE_SIGMA", "0.4")
    monkeypatch.setenv("DECADIC_MOTOR_BABBLE_FLOOR", "0.05")
    monkeypatch.setenv("DECADIC_BABBLE_ERROR_HALFSAT", "0.5")
    from decadic.config import motor_exploration_sigma

    # Sated + a perfect world model -> no exploration (the agent may rest); this is
    # NOT the dark room because nothing is deprived.
    assert motor_exploration_sigma(drive=0.0, fwd_error=0.0) == pytest.approx(0.0)
    # Forward-model surprise drives exploration even when sated (learn the world).
    assert motor_exploration_sigma(drive=0.0, fwd_error=0.5) == pytest.approx(0.2)
    # An unmet need floors exploration above zero even with a perfect world model:
    # keep acting until the action->relief contingency is discovered.
    assert motor_exploration_sigma(drive=0.1, fwd_error=0.0) >= 0.05
    # Severe deprivation + high surprise saturates toward the max scale.
    assert motor_exploration_sigma(drive=1.0, fwd_error=10.0) == pytest.approx(0.4)
    # sigma == 0 disables exploration entirely.
    monkeypatch.setenv("DECADIC_MOTOR_BABBLE_SIGMA", "0")
    assert motor_exploration_sigma(drive=1.0, fwd_error=10.0) == 0.0


def test_assist_gain_decays_over_cycles(monkeypatch):
    monkeypatch.setenv("DECADIC_ASSIST_DECAY_CYCLES", "8")
    bundle, _ = _run(monkeypatch, n=1)
    # Re-run a fresh agent advancing the cycle clock; assist should reach 0.
    _, late = _run(monkeypatch, n=10)
    assert late["_diagnostics"]["assist_gain"] == pytest.approx(0.0)


# --- Controllable / preferred proprio vectors -------------------------------


def test_controllable_proprio_vector_layout():
    from decadic.config import CONTROLLABLE_PROPRIO_BASE
    from decadic.nn.frozen_encoders import controllable_proprio_vector

    obs = {
        "proprioception": {
            "orientation": [0.3, -0.2, 0.7],
            "position": [9.0, 9.0, 1.35],
            "velocity": [0.5, -0.1, 0.0],
            "joints": [0.11, 99.0, 0.22, 99.0, 0.33, 99.0],  # (qpos, qvel) interleaved
        }
    }
    dim = CONTROLLABLE_PROPRIO_BASE + 3
    v = controllable_proprio_vector(obs, dim)
    assert len(v) == dim
    assert v[0:4] == pytest.approx([0.3, -0.2, 0.7, 1.35])  # roll, pitch, yaw, height
    assert v[4:7] == pytest.approx([0.5, -0.1, 0.0])  # velocity
    assert v[7:10] == pytest.approx([0.11, 0.22, 0.33])  # qpos only, qvel dropped


def test_forward_model_error_trends_down():
    """The world model can learn a fixed (state, action) -> outcome mapping."""
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    torch.manual_seed(0)
    cfg = neural_config_from_env("tiny")
    stack = NeuralCognitiveStack(cfg)
    z = torch.randn(4, cfg.d_model)
    u = torch.randn(4, cfg.n_actuators).clamp(-1, 1)
    target = torch.randn(4, cfg.forward_pred_dim)
    opt = torch.optim.Adam([stack.fwd_l1.weight, stack.fwd_l1.bias,
                            stack.fwd_l2.weight, stack.fwd_l2.bias], lr=1e-2)
    first = last = None
    for step in range(80):
        pred = stack.forward_predict(z, u)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first * 0.5


# --- MuJoCo adapter: PD tracking + fading assist ----------------------------


def test_adapter_applies_motor_with_no_external_wrench():
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        nu = sim.model.nu
        assert nu == 21  # actuator contract with the brain's motor head (incl. ankles)

        # Distinct nonzero PD targets. (Legacy assist_gain is accepted but inert:
        # the external support harness was replaced by internal joint braces.)
        sim.apply_action(
            {
                "type": "motor",
                "parameters": {"ctrl": [0.6] * nu, "assist_gain": 1.0},
            }
        )
        assert sim._motor_targets is not None
        sim.step(0.05)
        # The brain's command actually drives the actuators.
        assert float(np.abs(sim.data.ctrl).sum()) > 0.0
        # NO external force ever touches the torso -- the braces hold the body up
        # from the inside, so the feet keep full weight (no glide, by construction).
        assert float(np.abs(sim.data.xfrc_applied[sim.torso_id]).sum()) == 0.0

        # Efference copy + contract are surfaced in the observation.
        snap = sim.snapshot()
        assert snap.n_actuators == nu
        assert snap.motor is not None and len(snap.motor) == nu
        obs = mod.build_body_observation(snap, control_mode="active_inference")
        assert obs["world_state"]["body"]["n_actuators"] == nu
        assert "motor" in obs["proprioception"]

        # Still no external wrench after a quiet command (it was never there).
        sim.apply_action(
            {"type": "motor", "parameters": {"ctrl": [0.0] * nu, "assist_gain": 0.0}}
        )
        sim.step(0.01)
        assert float(np.abs(sim.data.xfrc_applied[sim.torso_id]).sum()) == pytest.approx(0.0)
    finally:
        sim.close()


def test_adapter_pd_tracking_moves_toward_target():
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        nu = sim.model.nu
        # Hold the torso fully so we isolate joint tracking from balance dynamics.
        sim.apply_action(
            {"type": "motor", "parameters": {"ctrl": [0.9] * nu, "assist_gain": 1.0}}
        )
        for _ in range(40):
            sim.step(0.02)
        # ctrl reflects PD effort toward the (out-of-rest) targets on multiple joints.
        assert int(np.count_nonzero(np.abs(sim.data.ctrl) > 1e-4)) > nu // 2
    finally:
        sim.close()


# --- Manual assist override (pipeline + runtime/API) ------------------------


def test_pipeline_assist_override_pins_gain(monkeypatch):
    """A manual override replaces the curriculum schedule for the emitted gain."""
    monkeypatch.setenv("DECADIC_ASSIST_DECAY_CYCLES", "8")
    # With no override, after the decay window the curriculum drives assist -> 0.
    _, auto = _run(monkeypatch, n=10)
    assert auto["_diagnostics"]["assist_gain"] == pytest.approx(0.0)
    # Pinned to 2.0, the same late cycles keep the harness above baseline.
    _, pinned = _run(monkeypatch, n=10, assist_override=2.0)
    assert pinned["_diagnostics"]["assist_gain"] == pytest.approx(2.0)
    assert pinned["action"]["parameters"]["assist_gain"] == pytest.approx(2.0)
    # Pinned to 0.0, the harness is off even at cycle 0 where the curriculum is 1.0.
    _, off = _run(monkeypatch, n=1, assist_override=0.0)
    assert off["_diagnostics"]["assist_gain"] == pytest.approx(0.0)


def test_runtime_configure_assist_override(tmp_path, monkeypatch):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("assist-cfg")
    # Default is 0 (no training-wheel assist unless the operator opts in).
    assert rt.assist_override == pytest.approx(0.0)
    assert rt.capacity_config()["assist_override"] == pytest.approx(0.0)
    # Pin to a manual level.
    cfg = rt.configure(assist_override=2)
    assert rt.assist_override == pytest.approx(2.0)
    assert cfg["assist_override"] == pytest.approx(2.0)
    # Negative sentinel clears back to Auto.
    cfg = rt.configure(assist_override=-1)
    assert rt.assist_override is None
    assert cfg["assist_override"] is None
    # Omitting the field leaves the current value untouched.
    rt.configure(assist_override=3)
    rt.configure(parallel_sessions=2)
    assert rt.assist_override == pytest.approx(3.0)


def test_runtime_configure_curriculum_mode(tmp_path, monkeypatch):
    from decadic.agents.runtime import AgentRuntime

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_CURRICULUM_MODE", "guided")
    rt = AgentRuntime("curriculum-cfg")
    # Default is seeded from the launch env so the UI shows the right state.
    assert rt.curriculum_mode == "guided"
    assert rt.capacity_config()["curriculum_mode"] == "guided"
    # Switch to legacy and back; "standard" is a friendly alias for legacy.
    assert rt.configure(curriculum_mode="legacy")["curriculum_mode"] == "legacy"
    assert rt.configure(curriculum_mode="guided")["curriculum_mode"] == "guided"
    assert rt.configure(curriculum_mode="standard")["curriculum_mode"] == "legacy"
    # An unknown value is ignored (mode unchanged).
    rt.configure(curriculum_mode="bogus")
    assert rt.curriculum_mode == "legacy"


def test_damage_grace_tracks_brace_engagement(tmp_path, monkeypatch):
    from decadic.agents.runtime import AgentRuntime
    from decadic.config import damage_grace_floor

    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path))
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    rt = AgentRuntime("grace-cfg")
    # Braces fully loosened (body moving on its own): falls hurt for real.
    rt.metrics["brace_engaged"] = 0.0
    free = rt._damage_grace()
    assert free == pytest.approx(1.0)
    # Braces holding 70%: a body still on its training orthosis gets toddler grace.
    rt.metrics["brace_engaged"] = 0.7
    braced = rt._damage_grace()
    assert braced < free
    assert braced == pytest.approx(max(damage_grace_floor(), 1.0 - 0.7))
    # Fully welded (spawn): the heaviest discount, floored.
    rt.metrics["brace_engaged"] = 1.0
    assert rt._damage_grace() == pytest.approx(max(damage_grace_floor(), 0.0))


# --- MuJoCo adapter: lifeless body on death / staleness ---------------------


def test_adapter_goes_limp_and_reanimates():
    import numpy as np

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        nu = sim.model.nu
        # Drive the body with a real command first (braced, no external wrench).
        sim.apply_action(
            {"type": "motor", "parameters": {"ctrl": [0.7] * nu, "assist_gain": 1.0}}
        )
        sim.step(0.05)
        assert float(np.abs(sim.data.ctrl).sum()) > 0.0
        assert float(np.abs(sim.data.xfrc_applied[sim.torso_id]).sum()) == 0.0

        # The brain dies / stops sending: body goes limp.
        h0 = float(sim.data.xpos[sim.torso_id][2])
        sim.set_lifeless(True)
        for _ in range(60):
            sim.step(0.02)
        # No actuation and no harness wrench while lifeless => true ragdoll.
        assert float(np.abs(sim.data.ctrl).sum()) == pytest.approx(0.0)
        assert float(np.abs(sim.data.xfrc_applied[sim.torso_id]).sum()) == pytest.approx(0.0)
        # Unsupported body falls under gravity.
        assert float(sim.data.xpos[sim.torso_id][2]) < h0

        # A fresh action revives it: actuation resumes.
        sim.set_lifeless(False)
        sim.apply_action(
            {"type": "motor", "parameters": {"ctrl": [0.7] * nu, "assist_gain": 1.0}}
        )
        sim.step(0.05)
        assert float(np.abs(sim.data.ctrl).sum()) > 0.0
    finally:
        sim.close()


# --- Scene: water consumables + respawn -------------------------------------


def test_drunk_now_detects_glass_in_reach():
    mod = _load_adapter_module()
    root = [0.0, 0.0, 1.0]
    glasses = {"prop_water_w1": [0.4, 0.2, 0.13], "prop_water_far": [5.0, 5.0, 0.13]}
    drunk = mod.drunk_now(root, glasses, drink_radius=1.0)
    assert "prop_water_w1" in drunk
    assert "prop_water_far" not in drunk


def test_default_scene_has_water_and_no_bear():
    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        assert sim.bear_body is None  # bear removed from the default scene
        assert len(sim.water_bodies) >= 1  # drinkable glasses present
        assert len(sim.food_bodies) >= 1  # food still near spawn
    finally:
        sim.close()


def test_consumable_respawns_after_delay():
    import time as _time

    pytest.importorskip("mujoco")
    mod = _load_adapter_module()
    sim = mod.HumanoidSim(vision=False, view=False, scene="default")
    try:
        name = next(iter(sim.water_bodies))
        body = sim.water_bodies[name]
        geoms = [g for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) == body]
        assert geoms

        sim._consume(name)
        assert name in sim.eaten
        assert name in sim._respawn_at
        # Hidden + non-collidable while consumed.
        assert float(sim.model.geom_rgba[geoms[0], 3]) == 0.0
        assert int(sim.model.geom_contype[geoms[0]]) == 0

        # Force the respawn deadline into the past, then process it.
        sim._respawn_at[name] = _time.monotonic() - 1.0
        sim._process_respawns()
        assert name not in sim.eaten
        assert name not in sim._respawn_at
        assert float(sim.model.geom_rgba[geoms[0], 3]) > 0.0  # visible again
        assert int(sim.model.geom_contype[geoms[0]]) != 0  # collidable again
    finally:
        sim.close()
