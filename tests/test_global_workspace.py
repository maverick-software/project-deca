"""Global workspace: winner-take-all + ignition + broadcast (Phase 2).

Covers the competition module (no-ignition below threshold, broadcast above it,
capacity/coalition behaviour), the working-memory candidate decomposition, the
live configure toggle, and that the on-branch actually diverges from the
byte-identical EMA off-branch in a real cycle.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_no_candidates_no_ignition():
    from decadic.nn.workspace import GlobalWorkspace

    gw = GlobalWorkspace(threshold=0.0)
    ign = gw.ignite(np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert ign.ignited is False
    assert ign.content.shape == (8,)
    assert float(np.abs(ign.content).sum()) == 0.0


def test_zero_salience_no_ignition():
    from decadic.nn.workspace import GlobalWorkspace

    slots = np.eye(4, dtype=np.float32)
    ign = GlobalWorkspace(threshold=0.0).ignite(slots, np.zeros(4, dtype=np.float32))
    assert ign.ignited is False


def test_below_threshold_holds_prior():
    """A diffuse field (no dominant coalition) must not ignite at a high threshold."""
    from decadic.nn.workspace import GlobalWorkspace

    slots = np.eye(5, dtype=np.float32)
    salience = np.ones(5, dtype=np.float32)  # perfectly uniform -> top-1 share = 0.2
    ign = GlobalWorkspace(threshold=0.5, capacity=1).ignite(slots, salience)
    assert ign.ignited is False
    assert ign.score == pytest.approx(0.2, abs=1e-6)
    assert float(np.abs(ign.content).sum()) == 0.0


def test_above_threshold_ignites_and_broadcasts():
    from decadic.nn.workspace import GlobalWorkspace

    slots = np.eye(4, dtype=np.float32)
    salience = np.array([0.1, 0.1, 0.1, 5.0], dtype=np.float32)  # one dominant
    ign = GlobalWorkspace(threshold=0.5, capacity=1).ignite(slots, salience)
    assert ign.ignited is True
    assert ign.winners == [3]
    # capacity=1 => content is exactly the winning slot.
    assert np.allclose(ign.content, slots[3])
    assert ign.score > 0.5


def test_capacity_expands_coalition():
    from decadic.nn.workspace import GlobalWorkspace

    slots = np.eye(4, dtype=np.float32)
    salience = np.array([0.05, 0.05, 4.0, 4.0], dtype=np.float32)
    ign = GlobalWorkspace(threshold=0.5, capacity=2).ignite(slots, salience)
    assert ign.ignited is True
    assert set(ign.winners) == {2, 3}
    # Two near-equal winners -> content blends both channels.
    assert ign.content[2] > 0.0 and ign.content[3] > 0.0


def test_workspace_candidates_from_working_memory():
    from decadic.state.working_memory import WorkingMemory

    wm = WorkingMemory()
    wm.deposit_scene([1.0, -1.0, 0.5, 0.25, 0.1, 0.0, -0.5, 2.0])
    vecs, sal = wm.workspace_candidates(8)
    # At least the scene candidate is present and dimensioned correctly.
    assert len(vecs) == len(sal) >= 1
    assert all(len(v) == 8 for v in vecs)
    assert all(s >= 0.0 for s in sal)


# --- Live toggle + real-cycle behaviour --------------------------------------


def _build_ctx(*, gwt_enabled):
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import ViabilityState

    perceptual = PerceptualState()
    # Give the workspace something to compete over (an ambient scene latent).
    perceptual.working_memory.deposit_scene([float(i) for i in range(16)])
    return CycleContext(
        state_bus=StateBus(),
        perceptual=perceptual,
        viability=ViabilityState(),
        episodic=EpisodicStore(None),
        homeostasis=None,
        gwt_enabled=gwt_enabled,
        last_observation={
            "proprioception": {
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "current_action": "idle",
            }
        },
    )


def _tiny_bundle(monkeypatch, name):
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    from decadic.nn.bundle import NeuralBundle

    b = NeuralBundle.try_build(name)
    assert b is not None
    return b


def test_pipeline_gwt_on_diverges_from_ema(monkeypatch):
    pytest.importorskip("torch")
    from decadic.cycle.neural_pipeline import run_neural_cycle

    # OFF branch (legacy EMA): the working-memory summary is blended into A.
    b_off = _tiny_bundle(monkeypatch, "gwt-off")
    ctx_off = _build_ctx(gwt_enabled=False)
    run_neural_cycle(ctx_off, b_off)
    a_off = np.array(ctx_off.state_bus.state_of_mind, dtype=np.float64).copy()

    # ON branch: a single ambient coalition has 100% share -> ignites and the
    # broadcast content (the scene candidate) is blended in instead of the EMA mix.
    monkeypatch.setenv("DECADIC_GWT_ENABLED", "1")
    b_on = _tiny_bundle(monkeypatch, "gwt-off")  # same name/seed as off -> same weights
    ctx_on = _build_ctx(gwt_enabled=True)
    out = run_neural_cycle(ctx_on, b_on)
    a_on = np.array(ctx_on.state_bus.state_of_mind, dtype=np.float64)

    assert not np.allclose(a_off, a_on)
    ws = ctx_on.latents.get("workspace")
    assert isinstance(ws, dict) and ws.get("enabled") is True


def test_configure_gwt_toggle_is_live(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    from decadic.agents.runtime import AgentRuntime
    from decadic.nn.faculties import CognitionFaculties

    agent = AgentRuntime(
        "gwt-live",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    before = agent.neural
    assert agent.capacity_config()["gwt_enabled"] is False  # conftest pins off

    cfg_on = agent.configure(gwt_enabled=True)
    assert agent.neural is before  # live toggle: no rebuild
    assert cfg_on["gwt_enabled"] is True
    assert agent.gwt_enabled is True

    cfg_off = agent.configure(gwt_enabled=False)
    assert cfg_off["gwt_enabled"] is False
