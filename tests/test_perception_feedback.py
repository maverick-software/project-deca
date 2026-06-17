"""Perception feedback loops: top-down predictive perception + perceptual retrieval.

Covers parity (flag off / gate=1 ⇒ z0_eff == ingress(fused)), the precision-gated
blend math, the parameter-free perceptual key, EMBEDDING_DIM consistency, and a
short synthetic learning check (top-down learns to predict the percept; the gate
receives gradient and is free to self-tune).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _cfg():
    from decadic.nn.config import neural_config_from_env

    return neural_config_from_env("tiny")


def _stack(monkeypatch, *, enabled: bool):
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    # Feedback now defaults ON, so the off-case is expressed explicitly (not by
    # unsetting the var). The baseline autouse fixture also pins "0", but a test
    # could delenv, so set it here unambiguously.
    monkeypatch.setenv("DECADIC_PERCEPTION_FEEDBACK_ENABLED", "1" if enabled else "0")
    from decadic.nn.neural_stack import NeuralCognitiveStack

    return NeuralCognitiveStack(_cfg())


# --- Loop 1: top-down predictive perception ---------------------------------


def test_flag_off_is_pure_bottom_up_parity(monkeypatch):
    stack = _stack(monkeypatch, enabled=False)
    assert stack.has_perception_feedback is False
    # Default-off must not add modules (so the state_dict matches the baseline).
    assert not hasattr(stack, "top_down")
    assert not hasattr(stack, "precision_gate")
    z0 = torch.randn(1, _cfg().d_model)
    z0_eff, z0_hat, gate = stack.top_down_perceive(z0)
    assert z0_eff is z0
    assert z0_hat is None and gate is None


def test_flag_on_builds_modules_and_returns_blend(monkeypatch):
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    assert stack.has_perception_feedback is True
    assert hasattr(stack, "top_down") and hasattr(stack, "precision_gate")
    z0 = torch.randn(1, cfg.d_model)
    z0_eff, z0_hat, gate = stack.top_down_perceive(z0)
    assert z0_eff.shape == (1, cfg.d_model)
    assert z0_hat.shape == (1, cfg.d_model)
    assert gate.shape == (1, cfg.d_model)
    # Gate is a sigmoid → strictly inside (0, 1).
    g = gate.detach()
    assert float(g.min()) > 0.0 and float(g.max()) < 1.0


def test_gate_one_is_exact_identity(monkeypatch):
    """gate ≈ 1 ⇒ z0_eff == z0_bu regardless of the top-down prediction."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    with torch.no_grad():
        # Random (non-zero) top-down so the identity is meaningful.
        stack.top_down[-1].weight.normal_()
        stack.top_down[-1].bias.normal_()
        stack.precision_gate.bias.fill_(50.0)  # sigmoid(50) == 1.0 in float32
    z0 = torch.randn(1, cfg.d_model)
    z0_eff, z0_hat, gate = stack.top_down_perceive(z0)
    assert torch.allclose(gate, torch.ones_like(gate))
    assert torch.allclose(z0_eff, z0, atol=1e-6)


def test_gate_zero_collapses_to_prediction(monkeypatch):
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    with torch.no_grad():
        stack.top_down[-1].weight.normal_()
        stack.top_down[-1].bias.normal_()
        stack.precision_gate.bias.fill_(-50.0)  # sigmoid(-50) == 0.0
    z0 = torch.randn(1, cfg.d_model)
    z0_eff, z0_hat, gate = stack.top_down_perceive(z0)
    assert torch.allclose(gate, torch.zeros_like(gate))
    assert torch.allclose(z0_eff, z0_hat, atol=1e-6)


def test_blend_is_kalman_form(monkeypatch):
    """z0_eff = z0_hat + gate * (z0_bu - z0_hat); with gate=0.5 this is the midpoint."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    with torch.no_grad():
        stack.top_down[-1].weight.normal_()
        stack.top_down[-1].bias.normal_()
        stack.precision_gate.weight.zero_()
        stack.precision_gate.bias.zero_()  # sigmoid(0) == 0.5
    z0 = torch.randn(1, cfg.d_model)
    z0_eff, z0_hat, gate = stack.top_down_perceive(z0)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))
    assert torch.allclose(z0_eff, 0.5 * (z0 + z0_hat), atol=1e-6)


def test_context_handles_missing_history(monkeypatch):
    """All history sources None ⇒ zero-filled context, no crash, correct shape."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    z0 = torch.randn(1, cfg.d_model)
    # Provide odd-sized scene/mem to exercise pad/truncate + avg-pool compression.
    z0_eff, z0_hat, gate = stack.top_down_perceive(
        z0,
        prev_z5=None,
        lstm_h=None,
        mem=torch.randn(1, cfg.memory_context_dim + 5),
        scene=torch.randn(1, 1337),
        intero=torch.tensor([[0.2, 0.1, 0.9]]),
    )
    assert z0_eff.shape == (1, cfg.d_model)
    assert torch.isfinite(z0_eff).all()


def test_top_down_learns_to_predict_percept(monkeypatch):
    """The self-supervised l_percept term drives the prediction error down."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    torch.manual_seed(0)
    target = torch.randn(1, cfg.d_model)
    prev_z5 = torch.randn(1, cfg.d_model)
    lstm_h = torch.randn(1, cfg.lstm_hidden)
    mem = torch.randn(1, cfg.memory_context_dim)
    intero = torch.tensor([[0.3, 0.2, 0.8]])
    opt = torch.optim.Adam(stack.top_down.parameters(), lr=1e-2)

    def pred_err() -> float:
        _, z0_hat, _ = stack.top_down_perceive(
            target, prev_z5=prev_z5, lstm_h=lstm_h, mem=mem, intero=intero
        )
        return float(torch.nn.functional.mse_loss(z0_hat, target.detach()).item())

    first = pred_err()
    for _ in range(150):
        opt.zero_grad()
        _, z0_hat, _ = stack.top_down_perceive(
            target, prev_z5=prev_z5, lstm_h=lstm_h, mem=mem, intero=intero
        )
        loss = torch.nn.functional.mse_loss(z0_hat, target.detach())
        loss.backward()
        opt.step()
    assert pred_err() < first * 0.5


def test_gate_receives_gradient(monkeypatch):
    """The precision gate is trainable (self-tuning), not a fixed schedule."""
    cfg = _cfg()
    stack = _stack(monkeypatch, enabled=True)
    with torch.no_grad():
        stack.top_down[-1].weight.normal_()  # ensure z0_hat != z0 so the gate matters
        stack.top_down[-1].bias.normal_()
    z0 = torch.randn(1, cfg.d_model)
    z0_eff, _, _ = stack.top_down_perceive(
        z0, prev_z5=torch.randn(1, cfg.d_model), intero=torch.tensor([[0.1, 0.1, 0.5]])
    )
    z0_eff.pow(2).sum().backward()
    assert stack.precision_gate.weight.grad is not None
    assert float(stack.precision_gate.weight.grad.abs().sum()) > 0.0


# --- Loop 2: perceptual-similarity retrieval --------------------------------


def test_embedding_dim_includes_perceptual_key():
    from decadic.memory import embeddings as E

    assert E.EMBEDDING_DIM == 64 + E.PERCEPT_KEY_DIM


def test_perceptual_key_is_deterministic_and_normalized():
    from decadic.memory.embeddings import PERCEPT_KEY_DIM, perceptual_key

    v = np.linspace(-3.0, 5.0, 137).astype(np.float32)
    k1 = perceptual_key(v)
    k2 = perceptual_key(v)
    assert k1.shape == (PERCEPT_KEY_DIM,)
    assert np.array_equal(k1, k2)  # deterministic
    assert abs(float(np.linalg.norm(k1)) - 1.0) < 1e-5  # unit norm
    # Empty / None percept ⇒ zeros (no contribution to cosine).
    assert np.array_equal(perceptual_key(None), np.zeros(PERCEPT_KEY_DIM, dtype=np.float32))


def test_query_and_stored_share_the_perceptual_layout():
    from decadic.memory.embeddings import (
        EMBEDDING_DIM,
        PERCEPT_KEY_DIM,
        episode_embedding_from_cycle,
        perceptual_key,
        query_vector_from_state_bus,
    )
    from decadic.state.state_bus import StateBus

    sb = StateBus()
    percept = np.random.randn(96).astype(np.float32)
    q = query_vector_from_state_bus(sb, percept)
    e = episode_embedding_from_cycle(sb, np.random.randn(96).astype(np.float32), percept)
    assert q.size == EMBEDDING_DIM and e.size == EMBEDDING_DIM
    # The perceptual key occupies the same trailing slot on both sides.
    key = perceptual_key(percept)
    assert np.allclose(q[-PERCEPT_KEY_DIM:], key)
    assert np.allclose(e[-PERCEPT_KEY_DIM:], key)


def test_zero_key_preserves_cosine_ranking():
    """Appending equal zero tails to both sides is a cosine no-op (off ⇒ parity)."""
    from decadic.memory.embeddings import PERCEPT_KEY_DIM

    rng = np.random.default_rng(0)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)

    def cos(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    z = np.zeros(PERCEPT_KEY_DIM, dtype=np.float32)
    base = cos(a, b)
    padded = cos(np.concatenate([a, z]), np.concatenate([b, z]))
    assert abs(base - padded) < 1e-6


def test_old_length_rows_are_skipped():
    """A pre-upgrade (length-64) episode is ignored by similarity search."""
    from decadic.memory.embeddings import EMBEDDING_DIM
    from decadic.memory.episodic_store import EpisodicRecord, EpisodicStore

    store = EpisodicStore(None)
    store.append(
        EpisodicRecord(
            cycle_index=1,
            summary={},
            salience=1.0,
            embedding=[0.1] * 64,  # legacy width
        )
    )
    store.append(
        EpisodicRecord(
            cycle_index=2,
            summary={},
            salience=1.0,
            embedding=[0.1] * EMBEDDING_DIM,  # current width
        )
    )
    hits = store.search_similar(np.ones(EMBEDDING_DIM, dtype=np.float32), top_k=5)
    cycles = {h["cycle_index"] for h in hits}
    assert 2 in cycles  # current-width row retrievable
    assert 1 not in cycles  # legacy-width row skipped
