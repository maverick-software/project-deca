"""WS-EXPAND E2 — multi-channel learning control.

Replaces the single scalar that modulates local plasticity with four routed
channels, all computed from state the cycle already produces:

- ``reward``      pleasure - pain -> sign/magnitude of the trace update
                  (bit-identical to the pre-E2 behavior; this channel IS the
                  old scalar).
- ``eta_scale``   expected-uncertainty -> multiplies the plasticity rate.
                  Volatility (a sustained shift in prediction-error level)
                  raises it; plain variance (noise) LOWERS it. The separation
                  is mandatory: a naive "error is high -> learn faster" rule
                  chases irreducible noise (evidence review, E2.2).
- ``surprise``    prediction-error spikes -> a transient, decaying boost to
                  the rate and to the attention gate's escalation propensity.
                  Consumed later by E4 (staleness guard) and E6 (gate reopen).
- ``gamma``       viability trend -> modulates the successor-features discount
                  INSIDE a hard clamp band, rate-limited, telemetry-logged.
                  The documented meta-gradient failure mode is a lucky streak
                  inflating the horizon inflating the value targets; the band
                  and the rate limit exist to make that loop impossible
                  (evidence review, E2.4).

Neutrality contract (birth-identity): until ``warmup`` cycles have passed the
controller returns reward passthrough, eta_scale 1.0, surprise 0.0 and the
config discount unchanged — a fresh agent is byte-identical at cycle 0 with
the flag ON. ``DECADIC_LEARN_CONTROL_MULTI=0`` is the kill switch (everything
reads the config default exactly); tests pin it OFF via conftest.

Pure Python (no torch): unit-testable anywhere, a handful of float ops per
cycle. The published discount is shared through a module-level slot guarded by
a lock so the background consolidation thread reads the same value the cycle
loop writes.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from decadic import config as C

_EPS = 1e-9


@dataclass(frozen=True)
class ModulationChannels:
    """One cycle's routed learning-control outputs."""

    reward: float  # sign/magnitude channel (== the pre-E2 scalar)
    eta_scale: float  # combined uncertainty x surprise rate multiplier, >= 0
    surprise: float  # transient spike level in [0, 1]
    gamma: float  # effective successor-features discount this cycle

    @classmethod
    def neutral(cls, reward: float) -> "ModulationChannels":
        return cls(
            reward=float(reward),
            eta_scale=1.0,
            surprise=0.0,
            gamma=C.sf_gamma(),
        )


class LearningController:
    """Per-bundle state for the E2 channels. Lives beside the attention gate
    on the bundle (lazily created); reset with the bundle."""

    def __init__(self) -> None:
        self._n = 0
        self._pc_fast: float | None = None  # fast EMA of prediction error
        self._pc_slow: float | None = None  # slow EMA (baseline level)
        self._noise: float | None = None  # EMA of |pc - fast| = noise proxy
        self._surprise = 0.0
        self._gamma: float | None = None  # rate-limited published discount
        self._cycles_since_gamma = 0
        self._gamma_moves = 0
        self._viab_prev: float | None = None
        self._viab_trend = 0.0  # EMA of per-cycle viability delta
        # Last computed channels (telemetry + next-cycle gate bias).
        self.last: ModulationChannels = ModulationChannels.neutral(0.0)

    # -- core ------------------------------------------------------------------
    def update(self, *, pc_loss: float, reward: float, viability: float) -> ModulationChannels:
        pc = _finite(pc_loss)
        reward = _finite(reward)
        viability = _finite(viability)

        # EMAs always update (the controller warms even while neutral).
        fast_prev = self._pc_fast
        noise_prev = self._noise if self._noise is not None else 0.0
        if self._pc_fast is None:
            self._pc_fast = pc
            self._pc_slow = pc
            self._noise = 0.0
        else:
            dev = abs(pc - self._pc_fast)
            self._noise = _ema(self._noise, dev, C.lc_noise_alpha())
            self._pc_fast = _ema(self._pc_fast, pc, C.lc_fast_alpha())
            self._pc_slow = _ema(self._pc_slow, pc, C.lc_slow_alpha())

        # Surprise: a spike is a deviation beyond k noise-scales from the
        # recent level. Level-set then exponentially decayed, so a single
        # spike produces a bounded, fading boost rather than a step.
        floor = C.lc_noise_floor()
        if fast_prev is not None and pc > fast_prev + C.lc_spike_k() * max(noise_prev, floor):
            self._surprise = 1.0
        else:
            self._surprise *= math.exp(-1.0 / max(1.0, C.lc_surprise_tau()))
        if self._surprise < 1e-4:
            self._surprise = 0.0

        # Viability trend (per-cycle delta EMA, in reservoir points/cycle).
        if self._viab_prev is not None:
            self._viab_trend = _ema(self._viab_trend, viability - self._viab_prev, C.lc_trend_alpha())
        self._viab_prev = viability

        self._n += 1
        if self._n < C.lc_warmup_cycles():
            self.last = ModulationChannels.neutral(reward)
            return self.last

        # Expected-uncertainty: sustained level shift (volatility) vs noise.
        # The trend must clear a MARGIN over the noise scale before the rate
        # rises: under pure stationary noise |fast-slow| fluctuates around a
        # fraction of the noise EMA and can transiently exceed it, so a bare
        # trend/noise ratio would briefly boost the rate on noise alone (the
        # exact failure the volatility/noise separation exists to prevent;
        # caught by test_pure_noise_never_raises_the_rate). With the margin:
        # noise-dominated -> ratio << 1 (learn slower); a real regime change
        # -> trend >> margin*noise -> ratio > 1; quiet+converged (both ~0)
        # -> floor/floor = 1 (neutral).
        trend = abs((self._pc_fast or 0.0) - (self._pc_slow or 0.0))
        margin = C.lc_trend_margin() * (self._noise or 0.0)
        ratio = (trend + floor) / (margin + floor)
        unc_scale = _clamp(ratio, C.lc_eta_min_scale(), C.lc_eta_max_scale())

        # Surprise boost multiplies on top; the combined multiplier is clamped
        # to the same band so the two channels cannot compound past the cap.
        eta_scale = _clamp(
            unc_scale * (1.0 + C.lc_surprise_gain() * self._surprise),
            C.lc_eta_min_scale(),
            C.lc_eta_max_scale(),
        )

        # Horizon: viability trend -> discount target inside the clamp band,
        # approached under a rate limit. gamma NEVER leaves the band and NEVER
        # moves faster than lc_gamma_step per lc_gamma_rate_cycles. The band
        # expands to include a config base pinned outside it, so warmup exit is
        # continuous (no discount jump) under any env override.
        base = C.sf_gamma()
        lo = min(C.lc_gamma_min(), base)
        hi = max(C.lc_gamma_max(), base)
        if self._gamma is None:
            self._gamma = base
        norm_trend = _clamp(self._viab_trend / max(_EPS, C.lc_gamma_trend_scale()), -1.0, 1.0)
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        target = _clamp(mid + norm_trend * half, lo, hi)
        self._cycles_since_gamma += 1
        if self._cycles_since_gamma >= C.lc_gamma_rate_cycles():
            self._cycles_since_gamma = 0
            step = C.lc_gamma_step()
            delta = _clamp(target - self._gamma, -step, step)
            if abs(delta) > 0.0:
                self._gamma = _clamp(self._gamma + delta, lo, hi)
                self._gamma_moves += 1

        self.last = ModulationChannels(
            reward=reward,
            eta_scale=eta_scale,
            surprise=self._surprise,
            gamma=self._gamma,
        )
        return self.last

    # -- gate coupling (consumed next cycle; neutral -> exactly 0 bias) ---------
    def gate_threshold_bias(self) -> float:
        """How much this controller lowers the gate's effective threshold.

        Surprise raises escalation propensity (E2.3); sustained volatility
        (eta_scale above neutral) adds a smaller push (E2.2). Bounded so the
        gate keeps a floor; neutral channels return exactly 0.0.
        """
        ch = self.last
        bias = C.lc_gate_surprise_gain() * ch.surprise + C.lc_gate_uncertainty_gain() * max(
            0.0, ch.eta_scale - 1.0
        )
        return _clamp(bias, 0.0, C.lc_gate_max_bias())

    def telemetry(self) -> dict[str, float | int]:
        ch = self.last
        return {
            "lc_eta_scale": round(ch.eta_scale, 6),
            "lc_surprise": round(ch.surprise, 6),
            "lc_gamma": round(ch.gamma, 6),
            "lc_gamma_moves": self._gamma_moves,
            "lc_noise": round(float(self._noise or 0.0), 6),
            "lc_trend": round(abs((self._pc_fast or 0.0) - (self._pc_slow or 0.0)), 6),
            "lc_viab_trend": round(self._viab_trend, 6),
        }


# -- shared discount (cycle loop writes, consolidation thread reads) ------------

_gamma_lock = threading.Lock()
_published_gamma: float | None = None


def publish_gamma(gamma: float) -> None:
    global _published_gamma
    with _gamma_lock:
        _published_gamma = float(gamma)


def reset_published_gamma() -> None:
    global _published_gamma
    with _gamma_lock:
        _published_gamma = None


def effective_sf_gamma() -> float:
    """The successor-features discount every consumer should read.

    Flag off (or nothing published yet, e.g. pre-warmup): the config value,
    exactly — the kill switch restores pre-E2 behavior everywhere at once.
    """
    if not C.learn_control_multi_enabled():
        return C.sf_gamma()
    with _gamma_lock:
        g = _published_gamma
    return C.sf_gamma() if g is None else g


# -- small pure helpers ----------------------------------------------------------

def _ema(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return float(value)
    return float((1.0 - alpha) * prev + alpha * value)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _finite(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0
