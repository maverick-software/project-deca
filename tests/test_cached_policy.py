"""WS-EXPAND E4: cached (habit) vs deliberate dual control.

Pure tests pin the ring buffer's teacher-window semantics and the earned-trust
curve (0 at birth, linear opening below threshold, collapse on NaN/stale).
Torch tests pin the cached head's zero-init, boundedness, and that online
distillation actually compresses a teacher mapping well enough to earn trust.
"""

import pytest

from decadic.nn.cached_policy import DistillBuffer, trust_weight


# ------------------------------------------------------------------ trust math


def test_trust_is_zero_at_birth_and_on_bad_input():
    assert trust_weight(None, threshold=0.01, max_w=1.0) == 0.0
    assert trust_weight(float("nan"), threshold=0.01, max_w=1.0) == 0.0
    assert trust_weight(-1.0, threshold=0.01, max_w=1.0) == 0.0
    assert trust_weight("junk", threshold=0.01, max_w=1.0) == 0.0


def test_trust_opens_linearly_below_threshold_and_closes_above():
    assert trust_weight(0.02, threshold=0.01, max_w=1.0) == 0.0  # not earned
    assert trust_weight(0.01, threshold=0.01, max_w=1.0) == 0.0  # boundary
    assert trust_weight(0.005, threshold=0.01, max_w=1.0) == pytest.approx(0.5)
    assert trust_weight(0.0, threshold=0.01, max_w=1.0) == pytest.approx(1.0)
    assert trust_weight(0.0, threshold=0.01, max_w=0.4) == pytest.approx(0.4)
    # Stale-habit guard is bidirectional: a rising EMA melts trust back down.
    assert trust_weight(0.009, threshold=0.01, max_w=1.0) < trust_weight(
        0.001, threshold=0.01, max_w=1.0
    )


# ------------------------------------------------------------------ ring buffer


def test_buffer_ring_semantics_and_recent_ordering():
    buf = DistillBuffer(capacity=4)
    assert len(buf) == 0 and buf.recent(8) == []
    for i in range(3):
        buf.push(f"z{i}", f"u{i}")
    assert [z for z, _ in buf.recent(2)] == ["z1", "z2"]  # newest last
    for i in range(3, 10):  # wrap several times
        buf.push(f"z{i}", f"u{i}")
    assert len(buf) == 4
    assert buf.total_pushed == 10
    assert [z for z, _ in buf.recent(4)] == ["z6", "z7", "z8", "z9"]
    assert [z for z, _ in buf.recent(2)] == ["z8", "z9"]
    # Rotating window: deterministic, wraps, sweeps the whole ring over
    # successive offsets (the trust-overfit guard).
    seen = set()
    for off in range(4):
        for z, _ in buf.window(off, 2):
            seen.add(z)
    assert seen == {"z6", "z7", "z8", "z9"}  # coverage of everything resident
    assert buf.window(3, 2) == buf.window(7, 2)  # modular determinism
    assert len(buf.window(0, 99)) == 4  # n >= size -> the whole buffer
    buf.clear()
    assert len(buf) == 0
    assert buf.window(0, 2) == []


# ------------------------------------------------------------- torch: the head


def _tiny_stack(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.nn.plastic import PlasticityFlags

    # Deterministic INIT: the distillation-convergence test depends on the
    # cached head's random starting weights; unseeded construction made it
    # flake at the threshold boundary (ema 0.0114 vs 0.01 on one run, passing
    # on the previous). Seeded, it either always passes or genuinely fails.
    torch.manual_seed(1234)
    return NeuralCognitiveStack(neural_config_from_env("tiny"), PlasticityFlags())


def test_cached_head_zero_init_and_bounded(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    assert stack.has_cached_policy is True
    assert torch.count_nonzero(stack.cached_l2.weight) == 0
    assert torch.count_nonzero(stack.cached_l2.bias) == 0
    z = torch.randn(4, stack.cfg.d_model)
    with torch.no_grad():
        out = stack.cached_action(z)
    assert torch.count_nonzero(out) == 0  # zero-init -> null action
    with torch.no_grad():
        stack.cached_l2.weight.normal_(0.0, 5.0)  # hostile weights
        out = stack.cached_action(z * 100)
    assert float(out.abs().max()) <= 1.0  # tanh bound holds regardless


def test_distillation_compresses_a_teacher_and_earns_trust(monkeypatch):
    torch = pytest.importorskip("torch")
    stack = _tiny_stack(monkeypatch)
    d, n_act = stack.cfg.d_model, stack.cfg.n_actuators
    # A deterministic "deliberate teacher": a fixed linear map of the input.
    gen = torch.Generator().manual_seed(11)
    w_teacher = torch.randn(d, n_act, generator=gen) * 0.2

    def teacher(z):
        return torch.tanh(z @ w_teacher)

    buf = DistillBuffer(capacity=256)
    params = [p for n, p in stack.named_parameters() if n.startswith("cached_l")]
    opt = torch.optim.Adam(params, lr=5e-3)
    ema = None
    # Live dynamics: teacher pairs arrive interleaved with distillation steps,
    # and the training window ROTATES over the whole buffer (the overfit guard
    # — training only on the newest 32 memorizes them, collapses the loss EMA,
    # and opens trust on a habit that never compressed the teacher; this test
    # caught exactly that failure in the first implementation). Early-exit
    # once trust is earned; the ceiling bounds runtime (a live agent has hours
    # of cycles — the test only proves the mechanism converges).
    for cycle in range(8000):
        z = torch.randn(1, d, generator=gen)
        buf.push(z, teacher(z))
        if len(buf) < 64:
            continue
        pairs = buf.window(cycle, 32)
        zb = torch.cat([p[0] for p in pairs], dim=0)
        ub = torch.cat([p[1] for p in pairs], dim=0)
        opt.zero_grad()
        l = torch.nn.functional.mse_loss(stack.cached_action(zb), ub)
        l.backward()
        opt.step()
        cur = float(l.item())
        ema = cur if ema is None else 0.95 * ema + 0.05 * cur
        if ema < 0.005 and cycle > 200:
            break
    # The habit compressed the deliberation well enough to earn the body.
    assert ema < 0.01
    assert trust_weight(ema, threshold=0.01, max_w=1.0) > 0.0
    # And on fresh inputs it tracks the teacher (it generalized, not
    # memorized). Calibration: the memorization failure this test caught
    # scored 0.62 here — indistinguishable from the null model (always-zero
    # habit), whose error is the teacher's own mean magnitude (~0.45). A real
    # compression must beat that baseline decisively; 256 banked pairs can't
    # (and needn't) pin the map to arbitrary precision.
    z_new = torch.randn(16, d, generator=gen)
    with torch.no_grad():
        err = float((stack.cached_action(z_new) - teacher(z_new)).abs().mean())
        null_err = float(teacher(z_new).abs().mean())
    assert err < 0.5 * null_err  # decisively better than doing nothing
    assert err < 0.2  # and small in absolute terms (memorization scored 0.62)
