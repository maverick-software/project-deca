"""Dual-network memory consolidation: replay buffer + consolidator + wiring parity.

Covers the plan's requirements:
- prioritized sampling favors high-salience transitions;
- a clone matches the live stack, and replay reduces a held-out loss on the
  consolidator;
- the Polyak soft-sync moves the live weights toward the consolidator with no NaNs;
- eviction/forgetting keeps the most-salient transitions and bounds the buffer;
- parity when off (the live cycle emits no transition with the flag disabled).
"""

import pytest

from decadic.consolidation.replay_buffer import ReplayBuffer, Transition

torch = pytest.importorskip("torch")


# --- Part 1: ReplayBuffer (salience prioritization + eviction) ---------------


def _dummy(salience: float) -> Transition:
    # The buffer only reads .salience; the latent tensors are irrelevant here.
    return Transition(
        z0=None,
        ep=None,
        mem=None,
        prev_state=None,
        prev_motor=None,
        proprio_target=None,
        salience=salience,
    )


def test_eviction_keeps_most_salient():
    buf = ReplayBuffer(3)
    for s in (1.0, 2.0, 3.0):
        buf.push(_dummy(s))
    assert len(buf) == 3
    buf.push(_dummy(4.0))  # full -> evicts the least-salient (1.0)
    assert len(buf) == 3
    stats = buf.salience_stats()
    assert stats["min"] == 2.0
    assert stats["max"] == 4.0
    # A newcomer below the resident minimum is dropped (existing memories persist).
    assert buf.push(_dummy(0.5)) is False
    assert buf.salience_stats()["min"] == 2.0


def test_prune_floor_rejects_low_salience():
    buf = ReplayBuffer(10, min_salience=1.0)
    assert buf.push(_dummy(0.5)) is False
    assert len(buf) == 0
    assert buf.push(_dummy(2.0)) is True
    assert len(buf) == 1


def test_prioritized_sampling_favors_high_salience():
    buf = ReplayBuffer(10, seed=0)
    for _ in range(9):
        buf.push(_dummy(0.01))
    hi = _dummy(100.0)
    buf.push(hi)
    hits = sum(1 for _ in range(60) if buf.sample(1)[0] is hi)
    assert hits > 30  # the high-salience transition dominates single draws


def test_empty_buffer_sample_is_empty():
    assert ReplayBuffer(4).sample(8) == []


def test_clear_empties_buffer():
    buf = ReplayBuffer(4)
    buf.push(_dummy(1.0))
    buf.clear()
    assert len(buf) == 0


# --- Part 2: ConsolidationManager (clone, replay, soft-sync) -----------------

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


def _collect(monkeypatch, n_cycles: int, *, enabled: bool):
    """Build a tiny bundle and run it, collecting the emitted replay transitions."""
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_ENABLED", "1" if enabled else "0")

    from decadic.cycle.neural_pipeline import run_neural_cycle
    from decadic.cycle.types import CycleContext
    from decadic.memory.episodic_store import EpisodicStore
    from decadic.nn.bundle import NeuralBundle
    from decadic.state.perceptual_state import PerceptualState
    from decadic.state.state_bus import StateBus
    from decadic.state.viability import Homeostasis, ViabilityState

    torch.manual_seed(0)
    bundle = NeuralBundle.try_build("unit-consolidation")
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


def test_clone_matches_active_stack(monkeypatch):
    from decadic.consolidation.consolidator import ConsolidationManager

    bundle, _ = _collect(monkeypatch, 4, enabled=True)
    mgr = ConsolidationManager(bundle)
    assert _param_distance(bundle.stack, mgr.cons_stack) == pytest.approx(0.0, abs=1e-9)


def test_replay_reduces_held_out_loss(monkeypatch):
    from decadic.consolidation.consolidator import ConsolidationManager

    bundle, transitions = _collect(monkeypatch, 18, enabled=True)
    assert len(transitions) >= 8
    train, held = transitions[:-4], transitions[-4:]
    buf = ReplayBuffer(200)
    for t in train:
        # Equal salience so the held-out comparison is about learning, not sampling.
        t.salience = 1.0
        buf.push(t)

    mgr = ConsolidationManager(bundle)
    mgr.cons_stack.reset_recurrent_state()
    before = mgr.held_out_loss(held)
    for _ in range(80):
        mgr.consolidate_once(buf, 4)
    mgr.cons_stack.reset_recurrent_state()
    after = mgr.held_out_loss(held)
    assert after < before


def test_soft_sync_moves_active_toward_consolidator(monkeypatch):
    from decadic.consolidation.consolidator import ConsolidationManager

    bundle, transitions = _collect(monkeypatch, 16, enabled=True)
    buf = ReplayBuffer(200)
    for t in transitions:
        buf.push(t)

    mgr = ConsolidationManager(bundle)
    # Diverge the consolidator from the live stack via replay training.
    for _ in range(40):
        mgr.consolidate_once(buf, 4)
    d_before = _param_distance(bundle.stack, mgr.cons_stack)
    assert d_before > 0.0

    mgr.soft_sync(0.5)
    d_after = _param_distance(bundle.stack, mgr.cons_stack)
    assert d_after < d_before
    # No NaNs leaked into the live weights.
    for _, p in bundle.stack.named_parameters():
        assert torch.isfinite(p).all()


def test_soft_sync_zero_tau_is_noop(monkeypatch):
    from decadic.consolidation.consolidator import ConsolidationManager

    bundle, transitions = _collect(monkeypatch, 16, enabled=True)
    buf = ReplayBuffer(200)
    for t in transitions:
        buf.push(t)
    mgr = ConsolidationManager(bundle)
    for _ in range(20):
        mgr.consolidate_once(buf, 4)
    d_before = _param_distance(bundle.stack, mgr.cons_stack)
    mgr.soft_sync(0.0)
    assert _param_distance(bundle.stack, mgr.cons_stack) == pytest.approx(d_before)


# --- Part 3: parity when off + transition emission ---------------------------


def test_no_transition_emitted_when_off(monkeypatch):
    _, transitions = _collect(monkeypatch, 6, enabled=False)
    assert transitions == []


def test_transition_emitted_when_on(monkeypatch):
    _, transitions = _collect(monkeypatch, 6, enabled=True)
    assert len(transitions) >= 1
    t = transitions[0]
    assert t.salience >= 0.0
    assert t.drive_on is True  # full reservoirs -> the homeostatic drive is active
    assert t.prev_intero is not None


def _neural_env(monkeypatch, *, enabled: bool):
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_ENABLED", "1" if enabled else "0")


def test_runtime_builds_buffer_when_enabled(monkeypatch):
    _neural_env(monkeypatch, enabled=True)
    from decadic.agents.runtime import AgentRuntime

    agent = AgentRuntime("unit-rt-on")
    assert agent.replay_buffer is not None


def test_runtime_no_buffer_when_disabled(monkeypatch):
    _neural_env(monkeypatch, enabled=False)
    from decadic.agents.runtime import AgentRuntime

    agent = AgentRuntime("unit-rt-off")
    assert agent.replay_buffer is None
