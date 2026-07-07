"""WS-EXPAND E2: multi-channel learning control.

Covers the four channels and every guardrail the evidence review made
mandatory: neutrality at init (birth-identity), volatility/noise separation
(a noisy-but-stationary error stream must NOT raise the learning rate),
bounded transient surprise, and the discount channel's hard clamp band +
rate limit (the meta-gradient instability guard). Plus the gate coupling's
byte-parity at zero bias and the close-time discount provider.
"""

import math
import random
import types

import pytest

from decadic import config as C
from decadic.consolidation.episodes import EpisodeAccumulator
from decadic.consolidation.returns import lambda_returns
from decadic.cycle.attention_gate import AttentionGate, GateInputs
from decadic.nn.learning_control import (
    LearningController,
    ModulationChannels,
    effective_sf_gamma,
    publish_gamma,
    reset_published_gamma,
)


@pytest.fixture(autouse=True)
def _lc_on(monkeypatch):
    """These tests exercise the controller itself -> flag ON, short warmup."""
    monkeypatch.setenv("DECADIC_LEARN_CONTROL_MULTI", "1")
    monkeypatch.setenv("DECADIC_LC_WARMUP_CYCLES", "8")
    monkeypatch.setenv("DECADIC_SF_GAMMA", "0.995")  # inside the default band
    reset_published_gamma()
    yield
    reset_published_gamma()


# ------------------------------------------------------------- E2.1 neutrality


def test_neutral_during_warmup():
    lc = LearningController()
    for i in range(7):  # warmup is 8 -> all of these are pre-warmup
        ch = lc.update(pc_loss=1.0 + i, reward=0.5, viability=90.0)
        assert ch.eta_scale == 1.0
        assert ch.surprise == 0.0
        assert ch.gamma == C.sf_gamma()
        assert ch.reward == 0.5  # reward channel is a pure passthrough
    assert lc.gate_threshold_bias() == 0.0


def test_reward_channel_is_the_old_scalar():
    lc = LearningController()
    for _ in range(50):
        ch = lc.update(pc_loss=1.0, reward=-0.25, viability=90.0)
    assert ch.reward == -0.25  # sign/magnitude untouched by the other channels


def test_quiet_converged_stream_reads_neutral():
    # trend ~ 0 AND noise ~ 0 -> ratio = floor/floor = 1.0 (not min-clamped):
    # a calm, well-predicting agent learns at exactly the configured rate.
    lc = LearningController()
    for _ in range(200):
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    assert ch.eta_scale == pytest.approx(1.0, abs=0.05)


# --------------------------------------------- E2.2 volatility vs noise (the guardrail)


def test_pure_noise_never_raises_the_rate():
    # Stationary level + white noise: the naive "error is high -> learn
    # faster" rule chases this; the channel must NOT (evidence: volatility
    # raises LR, irreducible stochasticity lowers it).
    rng = random.Random(7)
    lc = LearningController()
    scales = []
    for _ in range(400):
        pc = 1.0 + rng.gauss(0.0, 0.2)
        ch = lc.update(pc_loss=pc, reward=0.0, viability=90.0)
        scales.append(ch.eta_scale)
    post = scales[100:]
    assert max(post) <= 1.0 + 1e-9  # never above neutral on noise alone
    assert min(post) < 0.9  # and it actively slows down


def test_sustained_regime_shift_raises_the_rate():
    # A real level change (volatility): fast EMA moves, slow EMA lags ->
    # trend >> noise once the fast EMA settles at the new level.
    lc = LearningController()
    for _ in range(50):
        lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    scales = []
    for _ in range(120):
        ch = lc.update(pc_loss=3.0, reward=0.0, viability=90.0)
        scales.append(ch.eta_scale)
    assert max(scales) > 1.5  # clearly above neutral during the shift
    assert max(scales) <= C.lc_eta_max_scale() + 1e-9  # and capped


# ----------------------------------------------------------- E2.3 surprise


def test_surprise_spikes_then_decays():
    lc = LearningController()
    for _ in range(100):
        lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    ch = lc.update(pc_loss=10.0, reward=0.0, viability=90.0)  # the spike
    assert ch.surprise == 1.0
    assert lc.gate_threshold_bias() > 0.0  # escalation propensity raised
    decays = []
    for _ in range(6):
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
        decays.append(ch.surprise)
    assert all(a > b for a, b in zip(decays, decays[1:]))  # strictly fading
    for _ in range(200):
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    assert ch.surprise == 0.0  # fully extinguished, snapped to zero


def test_eta_scale_never_exceeds_cap_even_with_surprise():
    lc = LearningController()
    rng = random.Random(3)
    worst = 0.0
    for i in range(500):
        pc = 1.0 if i % 37 else 50.0  # periodic violent spikes on a shifting base
        pc += rng.gauss(0.0, 0.05) + (i / 250.0)
        ch = lc.update(pc_loss=pc, reward=0.0, viability=90.0)
        worst = max(worst, ch.eta_scale)
    assert worst <= C.lc_eta_max_scale() + 1e-9


# ------------------------------------------------------- E2.4 discount channel


def test_gamma_stays_in_band_and_rate_limited(monkeypatch):
    monkeypatch.setenv("DECADIC_LC_GAMMA_RATE_CYCLES", "5")
    monkeypatch.setenv("DECADIC_LC_GAMMA_STEP", "0.0005")
    lc = LearningController()
    lo, hi = C.lc_gamma_min(), C.lc_gamma_max()
    mid = (lo + hi) / 2.0
    prev = None
    gammas = []
    # Thriving: viability climbing EVERY cycle (unsaturated) -> target at band
    # max; gamma walks up under the rate limit and never leaves the band.
    v = 10.0
    for _ in range(600):
        v += 0.5
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=v)
        if prev is not None:
            assert abs(ch.gamma - prev) <= 0.0005 + 1e-12  # rate limit per move
        assert lo - 1e-12 <= ch.gamma <= hi + 1e-12  # hard band, always
        prev = ch.gamma
        gammas.append(ch.gamma)
    assert gammas[-1] == pytest.approx(hi)  # sustained thriving -> horizon max
    # Saturated (viability flat): trend decays to 0 -> gamma settles back
    # toward band-mid (a full-but-static agent is not "thriving harder").
    for _ in range(1200):
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=v)
        assert lo - 1e-12 <= ch.gamma <= hi + 1e-12
    assert ch.gamma == pytest.approx(mid, abs=0.0005)
    # Crashing: viability falling -> gamma walks down toward band min.
    for _ in range(1200):
        v -= 0.8
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=v)
        assert lo - 1e-12 <= ch.gamma <= hi + 1e-12
    assert ch.gamma == pytest.approx(lo)


def test_gamma_moves_only_on_the_rate_window(monkeypatch):
    monkeypatch.setenv("DECADIC_LC_GAMMA_RATE_CYCLES", "50")
    lc = LearningController()
    v = 50.0
    seen = set()
    for _ in range(300):
        v = min(100.0, v + 0.5)
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=v)
        seen.add(round(ch.gamma, 9))
    # 300 cycles / window 50 -> at most 6 moves -> at most 7 distinct values.
    assert len(seen) <= 7


def test_band_expands_to_include_pinned_config_gamma(monkeypatch):
    # Suite-style pin far below the band: warmup exit must be CONTINUOUS
    # (gamma starts at the config value, no jump to the band edge).
    monkeypatch.setenv("DECADIC_SF_GAMMA", "0.97")
    lc = LearningController()
    for _ in range(20):
        ch = lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    assert abs(ch.gamma - 0.97) < 0.01  # near the pinned base, no 0.99 snap


def test_effective_gamma_kill_switch(monkeypatch):
    publish_gamma(0.9931)
    assert effective_sf_gamma() == pytest.approx(0.9931)
    monkeypatch.setenv("DECADIC_LEARN_CONTROL_MULTI", "0")
    assert effective_sf_gamma() == C.sf_gamma()  # kill switch: config, exactly
    monkeypatch.setenv("DECADIC_LEARN_CONTROL_MULTI", "1")
    reset_published_gamma()
    assert effective_sf_gamma() == C.sf_gamma()  # nothing published -> config


# ------------------------------------------------------------- gate coupling


def _gate(**kw):
    defaults = dict(
        threshold=0.55,
        weights=(0.35, 0.30, 0.25, 0.10),
        target_rate=0.05,
        hysteresis_k=3,
        rate_window=100,
        budget_gain=0.5,
    )
    defaults.update(kw)
    return AttentionGate(**defaults)


def test_zero_bias_is_byte_identical():
    a, b = _gate(), _gate()
    b.set_modulation_bias(0.0)
    seq = [GateInputs(novelty=i / 10.0, prediction_error=(10 - i) / 10.0) for i in range(11)]
    ra = [(a.decide(s).escalate, round(a.decide(s).score, 9), a.decide(s).threshold_effective) for s in seq]
    rb = [(b.decide(s).escalate, round(b.decide(s).score, 9), b.decide(s).threshold_effective) for s in seq]
    assert ra == rb


def test_bias_lowers_threshold_bounded():
    g = _gate()
    base = g.decide(GateInputs()).threshold_effective
    g.set_modulation_bias(0.10)
    biased = g.decide(GateInputs()).threshold_effective
    assert biased == pytest.approx(base - 0.10)
    # A hostile caller can't wedge the gate open or push it negative.
    g.set_modulation_bias(99.0)
    capped = g.decide(GateInputs()).threshold_effective
    assert capped >= 0.05
    assert base - capped <= C.lc_gate_max_bias() + 1e-9
    g.set_modulation_bias(float("nan"))
    assert g.decide(GateInputs()).threshold_effective == pytest.approx(base)


def test_bias_can_tip_a_borderline_score_into_deliberation():
    g = _gate(threshold=0.50)
    borderline = GateInputs(novelty=0.55, prediction_error=0.45, affect=0.4, priority_investigate=0.0)
    d0 = g.decide(borderline)
    assert d0.escalate is False  # just under the bar
    g2 = _gate(threshold=0.50)
    g2.set_modulation_bias(0.10)
    d1 = g2.decide(borderline)
    assert d1.escalate is True and d1.reason == "score"


def test_controller_bias_composition():
    lc = LearningController()
    for _ in range(100):
        lc.update(pc_loss=1.0, reward=0.0, viability=90.0)
    lc.update(pc_loss=10.0, reward=0.0, viability=90.0)  # spike
    bias = lc.gate_threshold_bias()
    assert 0.0 < bias <= C.lc_gate_max_bias()


# ------------------------------------------- close-time discount provider


def _tr(reward: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(feat=[0.1, 0.0], reward=reward, ret=None, sf_target=None)


def test_episode_accumulator_uses_provider_gamma():
    rewards = [0.0, 0.0, 1.0]
    acc = EpisodeAccumulator(gamma=0.5, lam=0.9, gamma_provider=lambda: 0.9)
    acc.on_open("hydration", onset_cycle=0)
    steps = [_tr(r) for r in rewards]
    for t in steps:
        acc.add(t)
    out = acc.on_close("satisfied")
    expected = lambda_returns(rewards, gamma=0.9, lam=0.9, normalize=False)
    assert [t.ret for t in out] == pytest.approx(expected)
    # No provider -> constructor gamma exactly (pre-E2 behavior).
    acc2 = EpisodeAccumulator(gamma=0.5, lam=0.9)
    acc2.on_open("hydration", onset_cycle=0)
    steps2 = [_tr(r) for r in rewards]
    for t in steps2:
        acc2.add(t)
    out2 = acc2.on_close("satisfied")
    expected2 = lambda_returns(rewards, gamma=0.5, lam=0.9, normalize=False)
    assert [t.ret for t in out2] == pytest.approx(expected2)


def test_provider_failure_falls_back_to_constructor_gamma():
    def boom() -> float:
        raise RuntimeError("provider died")

    acc = EpisodeAccumulator(gamma=0.5, lam=0.9, gamma_provider=boom)
    acc.on_open("energy", onset_cycle=0)
    steps = [_tr(1.0)]
    for t in steps:
        acc.add(t)
    out = acc.on_close("satisfied")
    assert out[0].ret == pytest.approx(lambda_returns([1.0], gamma=0.5, lam=0.9, normalize=False)[0])


# ---------------------------------------------------------------- robustness


def test_nan_inputs_never_poison_the_controller():
    lc = LearningController()
    for _ in range(50):
        ch = lc.update(pc_loss=float("nan"), reward=float("inf"), viability=float("-inf"))
    assert math.isfinite(ch.eta_scale) and math.isfinite(ch.gamma) and math.isfinite(ch.reward)
    t = lc.telemetry()
    assert all(math.isfinite(float(v)) for v in t.values())
