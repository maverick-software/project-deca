"""Autonomous, need-gated curiosity drive (epistemic motivation -> element B).

Curiosity here is a *learning-progress* epistemic drive: the agent feels a
pleasure-side signal when its forward-model prediction error is *falling*
(it is successfully making some part of the world more predictable), NOT when
raw surprise is high. Rewarding raw surprise drives an agent straight at
irreducible noise -- the "noisy-TV" pathology, where a screen of static is
maximally interesting forever. Rewarding the *reduction* of error instead pulls
the agent toward the frontier of what it can still learn, and lets interest
fade once a contingency is mastered.

The drive is need-gated by survival urgency: a threatened or deprived agent
suppresses curiosity (you do not explore while a predator looms or while you
starve), and a safe, sated agent expresses it. This mirrors the innate
homeostatic-drive pattern already in ``viability.py`` -- phylogeny fixes the
*substrate* (that learnable structure is rewarding; that danger silences it),
while *what* is interesting is learned. It also matches Stage 4's risk-vs-
curiosity arbitration: the pull to investigate only wins when discomfort is low.

All functions here are pure and side-effect free except :class:`CuriosityState`,
which holds the short cross-cycle prediction-error history. Nothing in this
module imports torch or touches the network; the neural pipeline calls
:func:`compute_curiosity` once per cycle behind ``config.curiosity_enabled()``,
so the baseline is byte-identical when the flag is off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Internal shaping constants (not env-tunable; the four env knobs live in
# decadic/config.py as curiosity_enabled / curiosity_gain /
# curiosity_progress_window / curiosity_safety_sharpness).
ERROR_FLOOR_WEIGHT = 0.25  # how much current (flat) forward-model error still invites a probe
ERROR_FLOOR_HALFSAT = 0.5  # forward-model error giving half of the exploration floor (saturating)
INVESTIGATE_THRESHOLD = 0.05  # epistemic value (permission x opportunity) above which priority -> investigate


def _finite(x: float, default: float = 0.0) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return default
    return xf if math.isfinite(xf) else default


def learning_progress(history: list[float], *, eps: float = 1e-6) -> float:
    """Bounded [0, 1] signal: how much forward-model error has *fallen* recently.

    The prediction-error history (oldest first) is split into an older and a
    newer half; learning progress is the relative decrease
    ``(older_mean - newer_mean) / older_mean``. It is positive only while error
    is dropping (the agent is making the world more predictable), and 0 when
    error is flat, rising, or there is too little history (< 4 samples). Clamped
    to [0, 1] so a single-step collapse cannot spike the drive.
    """
    vals = [_finite(v) for v in history]
    n = len(vals)
    if n < 4:
        return 0.0
    half = n // 2
    older = vals[:half]
    newer = vals[half:]
    older_mean = sum(older) / len(older)
    newer_mean = sum(newer) / len(newer)
    if older_mean <= eps:
        return 0.0  # already near-perfect prediction: nothing left to learn here
    progress = (older_mean - newer_mean) / (older_mean + eps)
    return float(min(1.0, max(0.0, progress)))


def survival_urgency(*, pain: float, viability: float, viability_max: float = 100.0) -> float:
    """[0, 1] threat/deprivation gate: high when in pain OR when viability is low.

    ``pain`` is the felt-pain affect scalar (already EMA-bounded to [0, 1]);
    the viability deficit is ``1 - viability/viability_max``. Urgency is the max
    of the two, so either an acute threat (pain) or chronic depletion (low
    viability) is enough to silence curiosity.
    """
    p = min(1.0, max(0.0, _finite(pain)))
    vmax = max(1e-6, float(viability_max))
    deficit = min(1.0, max(0.0, 1.0 - _finite(viability) / vmax))
    return float(max(p, deficit))


def permission(urgency: float, *, sharpness: float) -> float:
    """Epistemic permission in [0, 1]: ``(1 - urgency) ** sharpness``.

    ``sharpness >= 1`` makes permission fall off fast as urgency rises, so even
    moderate threat/deprivation sharply suppresses exploration while a fully safe
    agent keeps permission ~1.
    """
    u = min(1.0, max(0.0, _finite(urgency)))
    s = max(1.0, _finite(sharpness, 1.0))
    return float((1.0 - u) ** s)


def epistemic_opportunity(
    progress: float,
    fwd_error: float,
    *,
    floor_weight: float = ERROR_FLOOR_WEIGHT,
    halfsat: float = ERROR_FLOOR_HALFSAT,
) -> float:
    """[0, 1] opportunity to learn: learning progress + a small error floor.

    Learning progress dominates (reward error reduction, not surprise). A small
    saturating term in the *current* error keeps a stuck-but-wrong agent probing
    a little even before any progress shows, so it can escape a flat-but-high-
    error plateau. ``floor_weight = 0`` recovers a pure learning-progress drive.
    """
    prog = min(1.0, max(0.0, _finite(progress)))
    err = max(0.0, _finite(fwd_error))
    hs = max(1e-6, float(halfsat))
    floor = max(0.0, float(floor_weight)) * (err / (err + hs))
    return float(min(1.0, prog + floor))


def curiosity_signal(
    *,
    learning_progress: float,
    fwd_error: float,
    pain: float,
    viability: float,
    gain: float = 1.0,
    safety_sharpness: float = 2.0,
    viability_max: float = 100.0,
) -> tuple[float, float]:
    """Need-gated epistemic drive -> ``(pain, pleasure)`` non-negative scalars.

    Curiosity is a pure pleasure-side drive (it never produces pain). The
    pleasure is ``gain x opportunity x permission``: epistemic *opportunity*
    (learning progress, plus a small error floor) times *permission* to indulge
    it (high only when safe and sated). The pleasure is returned on the same
    scale the affect path expects -- the caller hands it straight to
    ``apply_pain_pleasure_to_B`` / ``ema_affect`` (which clamp to [0, 1]), so the
    default ``gain = 1`` makes a fully-permitted, fully-learning state worth
    about one unit of felt pleasure.
    """
    urg = survival_urgency(pain=pain, viability=viability, viability_max=viability_max)
    perm = permission(urg, sharpness=safety_sharpness)
    opp = epistemic_opportunity(learning_progress, fwd_error)
    pleasure = max(0.0, float(gain)) * opp * perm
    return 0.0, float(pleasure)


@dataclass
class CuriosityState:
    """Per-agent cross-cycle epistemic state: a short forward-model PE history.

    Ephemeral like ``NeuralBundle.prev_state`` -- never checkpointed, rebuilt on
    load. Lives only while ``config.curiosity_enabled()`` is true.
    """

    window: int = 8
    history: list[float] = field(default_factory=list)

    def observe(self, fwd_error: float, *, window: int | None = None) -> float:
        """Append this cycle's forward-model error; return current learning progress."""
        if window is not None and window >= 2:
            self.window = int(window)
        self.history.append(_finite(fwd_error))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]
        return learning_progress(self.history)


@dataclass
class CuriosityOutput:
    """Everything the neural pipeline needs from one curiosity computation."""

    drive: float  # [0, 1] gated epistemic drive (added to the motor-babble gate)
    pleasure: float  # pleasure-side affect scalar (-> element B / pleasure EMA)
    pain: float  # always 0.0 (curiosity never hurts)
    learning_progress: float
    permission: float
    investigate: bool  # priority should become "investigate" (when not already avoiding)


def compute_curiosity(
    state: CuriosityState,
    *,
    fwd_error: float,
    pain: float,
    viability: float,
    gain: float = 1.0,
    window: int = 8,
    safety_sharpness: float = 2.0,
    viability_max: float = 100.0,
) -> CuriosityOutput:
    """Single per-cycle hook: update the PE history and return the gated drive.

    Combines :func:`learning_progress` (from the rolling history), the survival
    gate, and the epistemic opportunity into one :class:`CuriosityOutput`. The
    epistemic *value* used for the investigate decision is the pure
    permission x learning-progress (no error floor), so a flat-but-wrong probe
    nudges babble without spuriously hijacking the priority label.
    """
    progress = state.observe(fwd_error, window=window)
    urg = survival_urgency(pain=pain, viability=viability, viability_max=viability_max)
    perm = permission(urg, sharpness=safety_sharpness)
    opp = epistemic_opportunity(progress, fwd_error)
    drive = float(min(1.0, max(0.0, opp * perm)))
    pleasure = max(0.0, float(gain)) * drive
    epistemic_value = perm * min(1.0, max(0.0, progress))
    return CuriosityOutput(
        drive=drive,
        pleasure=pleasure,
        pain=0.0,
        learning_progress=progress,
        permission=perm,
        investigate=bool(epistemic_value >= INVESTIGATE_THRESHOLD),
    )
