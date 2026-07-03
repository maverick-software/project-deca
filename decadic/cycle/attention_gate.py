"""Stage 3->4 attention gate (WS3 Phase A: heuristic).

Decides per cycle whether stage 4 (risk-utility evaluation, curiosity
arbitration, investigative examination) runs its full deliberative compute
or is skipped in favor of a decayed precedent pass-through. See
docs/ws3_attention_gate_prd.md.

Design constraints:
- Pure Python, no torch/numpy: unit-testable anywhere, zero hot-path cost
  beyond a handful of float ops.
- Deterministic given inputs and config (reproducible runs).
- Fast-path threat escalation is unconditional and can never be suppressed
  by budget pressure.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field


# -- config (env-overridable, additive to decadic.config conventions) --------

DEFAULT_GATE_ENABLED = False
DEFAULT_GATE_THRESHOLD = 0.55
DEFAULT_GATE_TARGET_RATE = 0.05
DEFAULT_GATE_HYSTERESIS_K = 3
DEFAULT_GATE_RATE_WINDOW = 1000
DEFAULT_GATE_PASS_THROUGH_TAU = 20.0
# Input order: novelty, prediction_error, affect, priority, (budget applies
# to the threshold, not the score).
DEFAULT_GATE_WEIGHTS = (0.35, 0.30, 0.25, 0.10)
DEFAULT_GATE_BUDGET_GAIN = 0.5  # threshold increase per unit of rate overshoot


def gate_enabled() -> bool:
    return os.environ.get(
        "DECADIC_GATE_ENABLED", "1" if DEFAULT_GATE_ENABLED else "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def gate_threshold() -> float:
    return float(os.environ.get("DECADIC_GATE_THRESHOLD", str(DEFAULT_GATE_THRESHOLD)))


def gate_target_rate() -> float:
    return float(os.environ.get("DECADIC_GATE_TARGET_RATE", str(DEFAULT_GATE_TARGET_RATE)))


def gate_hysteresis_k() -> int:
    return max(1, int(os.environ.get("DECADIC_GATE_HYSTERESIS_K", str(DEFAULT_GATE_HYSTERESIS_K))))


def gate_rate_window() -> int:
    return max(10, int(os.environ.get("DECADIC_GATE_RATE_WINDOW", str(DEFAULT_GATE_RATE_WINDOW))))


def gate_pass_through_tau() -> float:
    return max(
        1.0, float(os.environ.get("DECADIC_GATE_PASS_THROUGH_TAU", str(DEFAULT_GATE_PASS_THROUGH_TAU)))
    )


def gate_budget_gain() -> float:
    return max(0.0, float(os.environ.get("DECADIC_GATE_BUDGET_GAIN", str(DEFAULT_GATE_BUDGET_GAIN))))


def gate_novelty_source() -> str:
    """Where the gate's novelty input comes from (WS4-M3.2).

    "full" (default): 1 - best similarity over the full 80-d episode
    embedding - the WS3-measured signal with ~0.05 dynamic range (internal
    state swamps external familiarity). "percept": 1 - best similarity over
    the 16-d percept-key subvector only - external familiarity, the Phase B
    fix. Default stays "full" until the gate probe re-validates.
    """
    value = os.environ.get("DECADIC_GATE_NOVELTY_SOURCE", "full").strip().lower()
    return value if value in ("full", "percept") else "full"


def gate_weights() -> tuple[float, float, float, float]:
    raw = os.environ.get("DECADIC_GATE_WEIGHTS", "").strip()
    if raw:
        try:
            parts = [float(x) for x in raw.split(",")]
            if len(parts) == 4 and all(p >= 0 for p in parts):
                return (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            pass
    return DEFAULT_GATE_WEIGHTS


# -- inputs / decision --------------------------------------------------------

def _clamp01(x: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass
class GateInputs:
    """Normalized [0,1] evidence for deliberation. Extraction happens in the
    pipeline (G2); this module only combines."""

    novelty: float = 0.0  # 1 - best episodic recall similarity
    prediction_error: float = 0.0  # pc-loss EMA vs baseline, normalized
    affect: float = 0.0  # pain scalar magnitude + drive pressure
    priority_investigate: float = 0.0  # 1.0 when element D says investigate
    fast_path_threat: bool = False  # damage event this cycle -> unconditional


@dataclass
class GateDecision:
    escalate: bool
    score: float
    threshold_effective: float
    reason: str  # "fast_path" | "score" | "hysteresis" | "skip"
    contributions: dict[str, float] = field(default_factory=dict)


class AttentionGate:
    """Heuristic stage 3->4 gate with hysteresis and a soft escalation budget."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        weights: tuple[float, float, float, float] | None = None,
        target_rate: float | None = None,
        hysteresis_k: int | None = None,
        rate_window: int | None = None,
        budget_gain: float | None = None,
    ) -> None:
        self.threshold = gate_threshold() if threshold is None else float(threshold)
        self.weights = gate_weights() if weights is None else weights
        self.target_rate = gate_target_rate() if target_rate is None else float(target_rate)
        self.hysteresis_k = gate_hysteresis_k() if hysteresis_k is None else max(1, int(hysteresis_k))
        self.rate_window = gate_rate_window() if rate_window is None else max(10, int(rate_window))
        self.budget_gain = gate_budget_gain() if budget_gain is None else max(0.0, float(budget_gain))
        self._history: deque[int] = deque(maxlen=self.rate_window)
        self._latch_remaining = 0
        self.decisions = 0
        self.escalations = 0

    # -- observability -------------------------------------------------------
    @property
    def escalation_rate(self) -> float:
        """Trailing escalation rate over the window (0 when no history)."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def skip_streak(self) -> int:
        streak = 0
        for v in reversed(self._history):
            if v:
                break
            streak += 1
        return streak

    # -- core ------------------------------------------------------------------
    def decide(self, inputs: GateInputs) -> GateDecision:
        self.decisions += 1
        w_nov, w_pe, w_aff, w_pri = self.weights
        w_total = max(1e-9, w_nov + w_pe + w_aff + w_pri)
        contributions = {
            "novelty": w_nov * _clamp01(inputs.novelty) / w_total,
            "prediction_error": w_pe * _clamp01(inputs.prediction_error) / w_total,
            "affect": w_aff * _clamp01(inputs.affect) / w_total,
            "priority": w_pri * _clamp01(inputs.priority_investigate) / w_total,
        }
        score = sum(contributions.values())

        # Soft budget: overshooting the target rate raises the bar; it can
        # never block the fast path.
        overshoot = max(0.0, self.escalation_rate - self.target_rate)
        threshold_eff = self.threshold + self.budget_gain * overshoot

        if inputs.fast_path_threat:
            decision = GateDecision(True, score, threshold_eff, "fast_path", contributions)
            self._latch_remaining = self.hysteresis_k
        elif self._latch_remaining > 0:
            decision = GateDecision(True, score, threshold_eff, "hysteresis", contributions)
            self._latch_remaining -= 1
        elif score >= threshold_eff:
            decision = GateDecision(True, score, threshold_eff, "score", contributions)
            self._latch_remaining = self.hysteresis_k
        else:
            decision = GateDecision(False, score, threshold_eff, "skip", contributions)

        self._history.append(1 if decision.escalate else 0)
        if decision.escalate:
            self.escalations += 1
        return decision

    def telemetry(self) -> dict[str, float | int]:
        return {
            "gate_decisions": self.decisions,
            "gate_escalations": self.escalations,
            "gate_escalation_rate": round(self.escalation_rate, 6),
            "gate_skip_streak": self.skip_streak,
            "gate_latch_remaining": self._latch_remaining,
        }


THREAT_EVENT_TYPES = ("collision", "damage", "attack", "bite", "fall_impact")


def extract_gate_inputs(
    *,
    best_recall_similarity: float | None,
    pc_ema: float | None,
    pain_scalar: float,
    drive_pressure: float,
    priority_label: str,
    observation_events: list | None,
) -> GateInputs:
    """Normalize raw per-cycle signals into GateInputs (all already computed
    on the hot path; this is read-only assembly, WS3-G2).

    - novelty: 1 - best episodic similarity; a missing similarity (empty
      store, recall off critical path and not yet warmed) reads as fully
      novel -> early cycles escalate, which is correct behavior.
    - prediction_error: pc_ema mapped monotonically to [0,1) via x/(x+1).
    - affect: positive pain plus drive pressure, clamped.
    - fast_path: any threat-typed event in this cycle's observation.
    """
    novelty = 1.0 if best_recall_similarity is None else 1.0 - _clamp01(best_recall_similarity)
    pe = 0.0
    if pc_ema is not None and pc_ema == pc_ema and pc_ema > 0:
        pe = pc_ema / (pc_ema + 1.0)
    affect = _clamp01(max(0.0, float(pain_scalar)) + max(0.0, float(drive_pressure)))
    fast = False
    for ev in observation_events or []:
        if isinstance(ev, dict) and str(ev.get("type", "")).lower() in THREAT_EVENT_TYPES:
            fast = True
            break
    return GateInputs(
        novelty=_clamp01(novelty),
        prediction_error=_clamp01(pe),
        affect=affect,
        priority_investigate=1.0 if priority_label == "investigate" else 0.0,
        fast_path_threat=fast,
    )


class PrecedentPassThrough:
    """Holds the last escalated stage-4 output and decays it toward neutral.

    Values are plain floats or lists of floats; decay multiplies by
    exp(-cycles_since/tau) so a stale precedent fades to zero influence.
    Downstream stages always receive a well-formed stage-4 output structure.
    """

    def __init__(self, tau: float | None = None) -> None:
        self.tau = gate_pass_through_tau() if tau is None else max(1.0, float(tau))
        self._cached: dict[str, object] | None = None
        self._age = 0

    def store(self, stage4_output: dict[str, object]) -> None:
        self._cached = dict(stage4_output)
        self._age = 0

    @property
    def has_precedent(self) -> bool:
        return self._cached is not None

    def _decay_factor(self) -> float:
        import math

        return math.exp(-float(self._age) / self.tau)

    def emit(self) -> dict[str, object] | None:
        """Decayed copy of the cached output; None when no precedent exists
        yet (caller must escalate on the first cycles)."""
        if self._cached is None:
            return None
        self._age += 1
        k = self._decay_factor()
        out: dict[str, object] = {}
        for key, val in self._cached.items():
            if isinstance(val, bool):
                out[key] = val
            elif isinstance(val, (int, float)):
                out[key] = float(val) * k
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                out[key] = [float(x) * k for x in val]
            else:
                out[key] = val
        out["gate_pass_through"] = True
        out["gate_precedent_age"] = self._age
        return out
