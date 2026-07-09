"""WS-DEPTH D1 — metacognitive calibration tracking (pure).

The calibration heads live in the stack; this module scores them: rolling
absolute error for the next-prediction-error head, and a binned reliability
(ECE-style) scalar for the P(drive improves) head. Calibration — predicted
probabilities matching realized frequencies — is the difference between
having a metacognition head and having metacognition, and it is exactly what
behavioral self-awareness probes measure.
"""

from __future__ import annotations


class CalibrationTracker:
    """Rolling calibration telemetry for the two D1 heads."""

    def __init__(self, window: int = 512, bins: int = 5) -> None:
        self.window = max(16, int(window))
        self.bins = max(2, int(bins))
        self._err_abs: list[float] = []  # |predicted next pc - realized|
        self._probs: list[float] = []  # predicted P(drive improves)
        self._hits: list[int] = []  # realized improvement (0/1)
        self.total = 0

    def note(self, pred_err: float, realized_err: float, p_improve: float, improved: bool) -> None:
        try:
            self._err_abs.append(abs(float(pred_err) - float(realized_err)))
            self._probs.append(min(1.0, max(0.0, float(p_improve))))
            self._hits.append(1 if improved else 0)
        except (TypeError, ValueError):
            return
        if len(self._err_abs) > self.window:
            self._err_abs.pop(0)
            self._probs.pop(0)
            self._hits.pop(0)
        self.total += 1

    def _ece(self) -> float:
        """Expected calibration error over probability bins."""
        n = len(self._probs)
        if n == 0:
            return 0.0
        ece = 0.0
        for b in range(self.bins):
            lo, hi = b / self.bins, (b + 1) / self.bins
            idx = [
                i
                for i, p in enumerate(self._probs)
                if (lo <= p < hi) or (b == self.bins - 1 and p == 1.0)
            ]
            if not idx:
                continue
            conf = sum(self._probs[i] for i in idx) / len(idx)
            acc = sum(self._hits[i] for i in idx) / len(idx)
            ece += (len(idx) / n) * abs(conf - acc)
        return ece

    def telemetry(self) -> dict[str, float | int]:
        n = max(1, len(self._err_abs))
        return {
            "metacog_samples": self.total,
            "metacog_err_mae": round(sum(self._err_abs) / n, 6),
            "metacog_calibration": round(self._ece(), 6),  # lower = better calibrated
        }
