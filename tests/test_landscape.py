"""Live loss-landscape probe: filter-normalized directions, grid eval, parity.

Covers the plan's requirements:
- filter-normalized directions match the per-filter weight norms and skip 1-D params;
- the random basis is deterministic for a fixed seed;
- the grid is grid x grid and entirely finite;
- the surface center equals the consolidator's held-out loss at theta* (same objective);
- the probe never mutates the live weights;
- empty / tiny buffers degrade gracefully;
- API: 404 unknown agent, warming-up 202, 200 with a crafted surface;
- parity-when-off: no landscape task starts and the accessor stays None.
"""

import pytest

from decadic.consolidation.replay_buffer import ReplayBuffer, Transition

torch = pytest.importorskip("torch")

_FWD_DIM_OBS_JOINTS = 34


def _body_obs(i: int) -> dict:
    return {
        "timestamp": f"t{i}",
        "proprioception": {
            "position": [0.0, 0.0, 1.2],
            "orientation": [0.01 * i, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "mujoco_humanoid:active_inference",
            "joints": [0.0 for _ in range(_FWD_DIM_OBS_JOINTS)],
            "contacts": [120.0, 110.0, 0.0, 0.0],
        },
        "events": [],
    }


def _collect(monkeypatch, n_cycles: int):
    """Build a tiny neural bundle, run it, and collect emitted replay transitions."""
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_ENABLED", "1")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    torch.manual_seed(0)
    bundle = NeuralBundle.try_build("unit-landscape")
    assert bundle is not None
    bus, percept, ep = StateBus(), PerceptualState(), EpisodicStore(None)
    homeo = Homeostasis(hydration=100.0, energy=100.0, integrity=100.0)
    via = ViabilityState(value=homeo.viability)
    transitions: list[Transition] = []
    for i in range(n_cycles):
        ctx = CycleContext(
            state_bus=bus,
            perceptual=percept,
            viability=via,
            episodic=ep,
            homeostasis=homeo,
            last_observation=_body_obs(i),
            pending_observations=[_body_obs(i)],
        )
        out = run_neural_cycle(ctx, bundle)
        payload = out.get("_transition")
        if payload is not None:
            transitions.append(Transition(**payload))
    return bundle, transitions


def _param_distance(stack_a, stack_b) -> float:
    a = dict(stack_a.named_parameters())
    total = 0.0
    for name, pb in stack_b.named_parameters():
        pa = a.get(name)
        if pa is None:
            continue
        total += float((pa.detach() - pb.detach()).pow(2).sum().item())
    return total


# --- directions --------------------------------------------------------------


def test_filter_normalized_dirs_match_filter_norms(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, _ = _collect(monkeypatch, 2)
    probe = LossLandscapeProbe(bundle, seed=0)
    params = probe._filter_params()
    assert params, "tiny stack must expose weight matrices (dim>=2)"
    # No 1-D params (biases/norm gains) are perturbed.
    assert all(p.dim() >= 2 for _, p in params)
    theta = {n: p.detach().clone() for n, p in params}
    probe._ensure_raw_dirs(params)
    d1, _d2 = probe._normalized_dirs(theta)
    for name, w in theta.items():
        wf = w.reshape(w.shape[0], -1)
        df = d1[name].reshape(d1[name].shape[0], -1)
        w_norm = wf.norm(dim=1)
        d_norm = df.norm(dim=1)
        # Each filter direction is rescaled to its weight filter's norm (Li et al.).
        nonzero = w_norm > 1e-6
        assert torch.allclose(d_norm[nonzero], w_norm[nonzero], rtol=1e-4, atol=1e-4)


def test_directions_deterministic_with_seed(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, _ = _collect(monkeypatch, 2)
    p1 = LossLandscapeProbe(bundle, seed=7)
    p2 = LossLandscapeProbe(bundle, seed=7)
    params1 = p1._filter_params()
    params2 = p2._filter_params()
    p1._ensure_raw_dirs(params1)
    p2._ensure_raw_dirs(params2)
    for name in {n for n, _ in params1}:
        assert torch.equal(p1._raw[0][name], p2._raw[0][name])
        assert torch.equal(p1._raw[1][name], p2._raw[1][name])


# --- grid evaluation ---------------------------------------------------------


def test_grid_shape_and_finiteness(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, transitions = _collect(monkeypatch, 12)
    assert len(transitions) >= 4
    probe = LossLandscapeProbe(bundle, seed=0)
    surface = probe.compute(transitions[:6], grid=9, span=0.5)
    assert surface is not None
    assert surface["grid"] == 9
    assert len(surface["alphas"]) == 9
    assert len(surface["betas"]) == 9
    assert len(surface["z"]) == 9 and all(len(row) == 9 for row in surface["z"])
    flat = [v for row in surface["z"] for v in row]
    assert all(v == v for v in flat)  # no NaNs
    assert surface["z_min"] <= surface["center_loss"] <= surface["z_max"]


def test_center_loss_matches_held_out(monkeypatch):
    from decadic.consolidation.consolidator import ConsolidationManager
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, transitions = _collect(monkeypatch, 12)
    batch = transitions[:6]
    probe = LossLandscapeProbe(bundle, seed=0)
    surface = probe.compute(batch, grid=5, span=0.4)
    assert surface is not None

    mgr = ConsolidationManager(bundle)
    mgr.cons_stack.reset_recurrent_state()
    held = mgr.held_out_loss(batch)
    # Both clone theta* and zero the recurrent state, so the center IS the held-out loss.
    assert surface["center_loss"] == pytest.approx(held, abs=1e-5)


def test_probe_does_not_mutate_live_weights(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, transitions = _collect(monkeypatch, 12)
    before = {n: p.detach().clone() for n, p in bundle.stack.named_parameters()}
    probe = LossLandscapeProbe(bundle, seed=0)
    probe.compute(transitions[:6], grid=7, span=1.0)
    after = dict(bundle.stack.named_parameters())
    for name, pb in before.items():
        assert torch.equal(pb, after[name].detach())


def test_empty_batch_returns_none(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, _ = _collect(monkeypatch, 2)
    probe = LossLandscapeProbe(bundle, seed=0)
    assert probe.compute([], grid=5, span=1.0) is None


def test_grid_is_clamped(monkeypatch):
    from decadic.consolidation.landscape import LossLandscapeProbe

    bundle, transitions = _collect(monkeypatch, 8)
    probe = LossLandscapeProbe(bundle, seed=0)
    surface = probe.compute(transitions[:4], grid=999, span=1.0)
    assert surface is not None
    assert surface["grid"] <= 41  # MAX_GRID safety cap


# --- runtime parity-when-off + accessor --------------------------------------


def test_runtime_no_landscape_task_when_off(monkeypatch):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_LANDSCAPE_ENABLED", "0")
    from decadic.agents.runtime import AgentRuntime

    agent = AgentRuntime("unit-ls-off")
    assert agent.brain_landscape() is None
    assert agent._landscape_task is None


# --- API endpoint ------------------------------------------------------------


@pytest.fixture
def api_app_neural(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("DECADIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.02")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.api.app import create_app

    return create_app()


def test_landscape_endpoint_unknown_agent_404(api_app_neural):
    from fastapi.testclient import TestClient

    with TestClient(api_app_neural) as client:
        assert client.get("/agent/nope/brain/landscape").status_code == 404


def test_landscape_endpoint_warming_up_202(api_app_neural):
    from fastapi.testclient import TestClient

    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        # Flag pinned off in conftest -> no surface cached yet -> warming up.
        r = client.get(f"/agent/{aid}/brain/landscape")
        assert r.status_code == 202
        assert r.json()["ready"] is False


def test_landscape_endpoint_returns_cached_surface_200(api_app_neural):
    from fastapi.testclient import TestClient

    with TestClient(api_app_neural) as client:
        aid = client.post("/agent").json()["agent_id"]
        crafted = {
            "alphas": [-1.0, 0.0, 1.0],
            "betas": [-1.0, 0.0, 1.0],
            "z": [[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]],
            "center_loss": 1.0,
            "z_min": 1.0,
            "z_max": 3.0,
            "grid": 3,
            "span": 1.0,
            "batch": 4,
            "cycle": 11,
            "preset": "tiny",
        }
        agent = api_app_neural.state.registry.get(aid)
        agent._last_landscape = crafted
        r = client.get(f"/agent/{aid}/brain/landscape")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["grid"] == 3
        assert body["center_loss"] == 1.0
        assert len(body["z"]) == 3
