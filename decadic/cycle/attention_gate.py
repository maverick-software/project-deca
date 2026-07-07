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

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# -- config (env-overridable, additive to decadic.config conventions) --------

# Defaults flipped ON at the validated operating point (owner decision
# 2026-07-04, after the first full probe PASS: threat 1/1, novelty 2/2,
# revisit quiet, calm 0.000). Tests pin the gate OFF via conftest for
# deterministic baselines, same as every other default-on faculty.
DEFAULT_GATE_ENABLED = True
DEFAULT_GATE_THRESHOLD = 0.30
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

    "percept" (default since 2026-07-04, probe-validated): 1 - best
    similarity over the 16-d percept-key subvector only - external
    familiarity with the recency horizon. "full": the legacy whole-embedding
    signal (WS3-measured dynamic range ~0.05; internal drift swamps external
    familiarity) - kept for A/B work.
    """
    value = os.environ.get("DECADIC_GATE_NOVELTY_SOURCE", "percept").strip().lower()
    return value if value in ("full", "percept") else "percept"


def gate_novelty_recency_horizon() -> int:
    """Cycles of recent memory EXCLUDED from the percept-novelty search.

    Probe 2026-07-04 finding: with write-through read-your-writes, the best
    percept match is always the frame stored one cycle ago (consecutive-key
    cosine p50 = 1.0000), so "1 - best similarity" degenerates into a
    single-cycle delta detector -- novelty measured identically ~0 even during
    injected events, while the stored keys were demonstrably discriminative
    (event keys at cosine ~0.25 to their pre-event baseline). Excluding the
    last N cycles restores the intended semantics: familiar means seen BEFORE
    the recent past, so an event percept stays novel for ~N cycles instead of
    one frame. The default (64) sits above the eval sampler's stride (event
    novelty persists long enough to be observed) and below the 200-cycle
    patrol lap (a repeated loop still reads familiar via the previous lap).
    0 disables the horizon.
    """
    return max(0, int(os.environ.get("DECADIC_GATE_NOVELTY_RECENCY", "64")))


def gate_novelty_peak_window() -> int:
    """Cycles over which the novelty telemetry's rolling max is held.

    A first-exposure novelty spike lasts ~1-3 cycles (top-down perception
    assimilates the new percept almost immediately), while the eval sampler
    reads metrics every ~6 cycles -- the raw per-cycle value is invisible to
    offline verdicts. ``gate_i_novelty_peak`` holds the max over this many
    cycles so any spike survives at least a few sampler reads. Decision
    logic is untouched; this is telemetry only.
    """
    return max(1, int(os.environ.get("DECADIC_GATE_NOVELTY_PEAK_WINDOW", "32")))


# -- WS3B-M0: decision log + shadow deliberation config ------------------------


def gate_log_enabled() -> bool:
    """WS3B-M0.1: per-cycle gate decision logging (training-data channel).
    Default OFF -- zero new IO on existing runs."""
    return os.environ.get("DECADIC_GATE_LOG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gate_log_dir() -> Path:
    """Where decision logs land: DECADIC_GATE_LOG_DIR, else the run's
    DECADIC_LOG_DIR (the probe/soak wrappers already set it), else ./logs."""
    for var in ("DECADIC_GATE_LOG_DIR", "DECADIC_LOG_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            return Path(v)
    return Path("logs")


def gate_shadow_rate() -> float:
    """WS3B-M0.2: fraction of gate decisions that also run shadow deliberation
    (fresh stage-4 beside the substituted precedent, diagnostics only)."""
    try:
        v = float(os.environ.get("DECADIC_GATE_SHADOW_RATE", "0.05"))
    except ValueError:
        v = 0.05
    return min(1.0, max(0.0, v))


def shadow_sampled(cycle: int, rate: float) -> bool:
    """Deterministic per-cycle sampling (reproducible runs, no RNG state):
    Knuth multiplicative hash of the cycle index against the rate."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    h = ((int(cycle) * 2654435761) >> 16) & 0xFFFF
    return h < int(rate * 65536.0)


class GateDecisionLog:
    """Off-thread JSONL sink for per-cycle gate decisions (WS3B-M0.1).

    Log-and-continue AND off the cognitive thread (2026-07-05): the cycle
    thread only serializes a small flat dict and drops the line on a bounded
    queue (``put_nowait`` -> never blocks); a daemon writer thread owns the
    open file handle and does ALL disk I/O (append + periodic fsync). The
    prior design buffered but still opened/wrote/closed the file inline every
    32 rows, so cognition periodically paid for telemetry -- exactly the kind
    of on-thread tax that contaminates cycle-time measurement. Now the hot
    path cost is one ``json.dumps`` + one queue put. On overflow (writer can't
    keep up) newest rows are dropped and counted, never blocking the loop.
    Mirror of the WS4B off-lock flusher discipline.
    """

    FLUSH_SECONDS = 2.0
    QUEUE_MAX = 8192  # ~ many seconds of headroom; overflow drops, never blocks

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._failed = False
        self._dropped = 0
        self._warned_drop = False
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=self.QUEUE_MAX)
        self._thread: threading.Thread | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("gate decision log disabled (%s): %s", self.path, exc)
            self._failed = True
            return
        self._thread = threading.Thread(
            target=self._writer_loop, name="gate-log-writer", daemon=True
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        try:
            f = self.path.open("a", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - IO failure path
            logger.warning("gate decision log disabled (%s): %s", self.path, exc)
            self._failed = True
            return
        last = time.monotonic()
        try:
            while True:
                try:
                    item = self._q.get(timeout=self.FLUSH_SECONDS)
                except queue.Empty:
                    f.flush()
                    last = time.monotonic()
                    continue
                if item is None:  # close sentinel: drain remaining, then stop
                    while True:
                        try:
                            more = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if more is not None:
                            f.write(more)
                            f.write("\n")
                    f.flush()
                    return
                f.write(item)
                f.write("\n")
                now = time.monotonic()
                if now - last >= self.FLUSH_SECONDS:
                    f.flush()
                    last = now
        except OSError as exc:  # pragma: no cover - IO failure path
            logger.warning("gate decision log disabled (%s): %s", self.path, exc)
            self._failed = True
        finally:
            try:
                f.close()
            except OSError:
                pass

    def log(self, row: dict) -> None:
        if self._failed:
            return
        try:
            line = json.dumps(row, separators=(",", ":"), default=float)
        except (TypeError, ValueError):
            return  # one malformed row is not worth a crash
        try:
            self._q.put_nowait(line)
        except queue.Full:
            # Telemetry must never stall cognition: drop and count. Warn once.
            self._dropped += 1
            if not self._warned_drop:
                self._warned_drop = True
                logger.warning(
                    "gate decision log writer behind; dropping rows (%s)", self.path
                )

    def flush(self) -> None:
        # No-op for API compatibility: the writer thread fsyncs on its own
        # cadence (<= FLUSH_SECONDS). Kept so callers/tests can still call it.
        return

    def close(self) -> None:
        if self._failed or self._thread is None:
            return
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:  # pragma: no cover - shutdown best effort
            pass
        self._thread.join(timeout=5.0)

    @property
    def dropped(self) -> int:
        return self._dropped


def open_gate_log() -> GateDecisionLog:
    """New per-process log file (one live agent per probe/soak server)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return GateDecisionLog(
        gate_log_dir() / f"gate_decisions_{stamp}_{os.getpid()}.jsonl"
    )


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
    type2_search: bool = False  # WS-FORAGE M5: active need whose relief is
    # remembered but not here -> unconditional escalate into the deliberate,
    # memory-guided (System-2) path even when nothing novel is in view.


@dataclass
class GateDecision:
    escalate: bool
    score: float
    threshold_effective: float
    reason: str  # "fast_path" | "type2_memory_search" | "hysteresis" | "score" | "skip"
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
        type2_refractory: int | None = None,
    ) -> None:
        self.threshold = gate_threshold() if threshold is None else float(threshold)
        self.weights = gate_weights() if weights is None else weights
        self.target_rate = gate_target_rate() if target_rate is None else float(target_rate)
        self.hysteresis_k = gate_hysteresis_k() if hysteresis_k is None else max(1, int(hysteresis_k))
        self.rate_window = gate_rate_window() if rate_window is None else max(10, int(rate_window))
        self.budget_gain = gate_budget_gain() if budget_gain is None else max(0.0, float(budget_gain))
        if type2_refractory is None:
            from decadic.config import type2_refractory_cycles

            type2_refractory = type2_refractory_cycles()
        self.type2_refractory = max(0, int(type2_refractory))
        self._type2_cooldown = 0
        # WS-EXPAND E2.2/E2.3: learning-control threshold bias. Set each cycle
        # from the LearningController's channels (surprise + excess volatility
        # -> more willing to deliberate). Bounded by the setter; 0.0 (the
        # default and the neutral-channel value) is byte-identical to pre-E2.
        self._modulation_bias = 0.0
        self._history: deque[int] = deque(maxlen=self.rate_window)
        self._latch_remaining = 0
        self.decisions = 0
        self.escalations = 0
        # Telemetry-only rolling max of the novelty input (see
        # gate_novelty_peak_window): (cycle, value) pairs inside the window.
        self._novelty_peak_window = gate_novelty_peak_window()
        self._novelty_hist: deque[tuple[int, float]] = deque()
        self.novelty_peak = 0.0

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
        if self._type2_cooldown > 0:
            self._type2_cooldown -= 1
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
        # never block the fast path. The learning-control bias (E2) lowers the
        # bar when the world just got surprising/volatile; it is bounded at
        # set time, and a floor keeps the gate from becoming free.
        overshoot = max(0.0, self.escalation_rate - self.target_rate)
        threshold_eff = self.threshold + self.budget_gain * overshoot
        if self._modulation_bias > 0.0:
            # Floor applies only on the bias path: bias == 0 (flag off /
            # neutral channels) leaves the arithmetic byte-identical.
            threshold_eff = max(0.05, threshold_eff - self._modulation_bias)

        if inputs.fast_path_threat:
            decision = GateDecision(True, score, threshold_eff, "fast_path", contributions)
            self._latch_remaining = self.hysteresis_k
        elif inputs.type2_search and self._type2_cooldown <= 0:
            # WS-FORAGE M5: need + remembered-but-not-here -> engage the
            # deliberate path to pursue the resource from memory. Additive
            # (never suppresses another escalation). REFRACTORY (2026-07-06):
            # the condition is level-true for long stretches of a hungry life,
            # and unbounded re-firing re-deliberated the same intention every
            # cycle (observed 55% of ALL cycles). The deliberation's output
            # persists between fires -- the goal vector + bearing condition the
            # policy continuously and the stage-4 precedent decays -- so one
            # fire forms the intention, the cooldown covers execution, and the
            # next fire is a periodic re-check. Type-2 deliberation is thus
            # bounded near (1 + hysteresis_k) / refractory of cycles.
            decision = GateDecision(
                True, score, threshold_eff, "type2_memory_search", contributions
            )
            self._latch_remaining = self.hysteresis_k
            self._type2_cooldown = self.type2_refractory
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

    def set_modulation_bias(self, bias: float) -> None:
        """WS-EXPAND E2: per-cycle learning-control threshold bias.

        Callers pass ``LearningController.gate_threshold_bias()`` (already
        bounded); re-clamped here so no caller can push the gate negative or
        wedge it open. NaN reads as 0 (neutral).
        """
        b = float(bias)
        if b != b:  # NaN guard
            b = 0.0
        from decadic.config import lc_gate_max_bias

        self._modulation_bias = min(max(0.0, b), lc_gate_max_bias())

    def note_novelty(self, novelty: float, cycle: int) -> float:
        """Record this cycle's novelty input; returns the rolling-window max.

        Telemetry only (never feeds ``decide``): keeps short spikes visible to
        the sparsely-sampling eval harness via ``gate_i_novelty_peak``.
        """
        c = int(cycle)
        self._novelty_hist.append((c, float(novelty)))
        floor = c - self._novelty_peak_window
        while self._novelty_hist and self._novelty_hist[0][0] < floor:
            self._novelty_hist.popleft()
        self.novelty_peak = max(v for _, v in self._novelty_hist)
        return self.novelty_peak

    def telemetry(self) -> dict[str, float | int]:
        return {
            "gate_decisions": self.decisions,
            "gate_escalations": self.escalations,
            "gate_escalation_rate": round(self.escalation_rate, 6),
            "gate_skip_streak": self.skip_streak,
            "gate_latch_remaining": self._latch_remaining,
            "gate_i_novelty_peak": round(float(self.novelty_peak), 6),
            "gate_modulation_bias": round(float(self._modulation_bias), 6),
        }


THREAT_EVENT_TYPES = ("collision", "damage", "attack", "bite", "fall_impact")


def type2_trigger(
    goal_vec: "list | None",
    *,
    far_distance: float,
    min_deficit: float,
) -> bool:
    """WS-FORAGE M5: should this cycle escalate into deliberate memory-guided
    pursuit? True when the goal vector says a resource that relieves the active
    need is REMEMBERED (target mask set) but NOT here (normalized distance
    beyond ``far_distance``) and the need is worth deliberating over (graded
    deficit at least ``min_deficit`` -- System-2 economics now that goal
    conditioning is continuous). Pure python; layout indices follow
    ``decadic.nn.goal_conditioning`` ([3]=deficit, [6]=distance, [7]=mask)."""
    if goal_vec is None or len(goal_vec) < 8:
        return False
    try:
        return (
            float(goal_vec[7]) >= 0.5
            and float(goal_vec[6]) >= float(far_distance)
            and float(goal_vec[3]) >= float(min_deficit)
        )
    except (TypeError, ValueError, IndexError):
        return False


def extract_gate_inputs(
    *,
    best_recall_similarity: float | None,
    pc_ema: float | None,
    pain_scalar: float,
    drive_pressure: float,
    priority_label: str,
    observation_events: list | None,
    type2_search: bool = False,
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
        type2_search=bool(type2_search),
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
