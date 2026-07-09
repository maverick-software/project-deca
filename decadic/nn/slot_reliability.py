"""WS-IND I3 — per-percept reality monitoring (per-slot reliability).

WS-EXPAND E2 already built the reality-monitoring computation — separating
volatility (a real change worth learning from) from noise (jitter worth
discounting) — but at ONE global granularity. This module runs the same idea
per workspace slot: each slot position keeps a fast feature EMA and a noise
EMA (mean deviation from that EMA); a slot whose stream is noisier than its
peers earns a reliability below 1, floored, and the E6 routing gate composes
relevance x reliability. Uniform noise (all slots equally jittery) reads as
reliability 1.0 everywhere — reliability is RELATIVE evidence quality, not a
global damper (the pipeline's learning-rate channel already handles global
noise).

Honest approximation, documented: state is keyed by slot INDEX, and slot
order is only approximately stable across consecutive cycles (persistent
entities keep their slots; eviction reshuffles occasionally). The floor, the
surprise-reopen on the E6 gate, and the relative (not absolute) formulation
bound the harm of a reshuffle to a few cycles of mild mis-weighting.

Pure python + list math on small K; torch tensors are accepted and read
elementwise via ``tolist``.
"""

from __future__ import annotations

from typing import Any


class SlotReliability:
    """Per-slot-index noise tracking -> relative reliability weights."""

    def __init__(
        self,
        *,
        max_slots: int,
        fast_alpha: float,
        noise_alpha: float,
        floor: float,
        warmup: int,
    ) -> None:
        self.max_slots = max(1, int(max_slots))
        self.fast_alpha = float(fast_alpha)
        self.noise_alpha = float(noise_alpha)
        self.floor = min(1.0, max(0.0, float(floor)))
        self.warmup = max(1, int(warmup))
        self._fast: dict[int, list[float]] = {}
        self._noise: dict[int, float] = {}
        self._seen: dict[int, int] = {}

    def update(self, slots: Any) -> "list[float]":
        """Advance per-slot state and return reliability in [floor, 1] per slot.

        ``slots``: [K, D] or [1, K, D] tensor/nested list of slot features.
        Never raises; malformed input -> all-ones (identity = parity).
        """
        try:
            rows = self._rows(slots)
            if not rows:
                return []
            k = len(rows)
            rel = [1.0] * k
            noises: list[float] = []
            for i in range(min(k, self.max_slots)):
                row = rows[i]
                prev = self._fast.get(i)
                if prev is None or len(prev) != len(row):
                    self._fast[i] = list(row)
                    self._noise[i] = 0.0
                    self._seen[i] = 1
                    noises.append(0.0)
                    continue
                dev = sum(abs(a - b) for a, b in zip(row, prev)) / max(1, len(row))
                self._noise[i] = (1.0 - self.noise_alpha) * self._noise[i] + self.noise_alpha * dev
                self._fast[i] = [
                    (1.0 - self.fast_alpha) * b + self.fast_alpha * a
                    for a, b in zip(row, prev)
                ]
                self._seen[i] = self._seen.get(i, 0) + 1
                noises.append(self._noise[i])
            warmed_idx = [
                i
                for i in range(min(k, self.max_slots))
                if self._seen.get(i, 0) >= self.warmup
            ]
            if len(warmed_idx) < 2:
                return rel  # relative reliability needs at least two peers
            eps = 1e-3
            margin = 1.5  # the E2 pattern: must exceed the field by a margin
            for i in warmed_idx:
                # EXCLUDE-SELF reference: "noisier than the REST of the field."
                # Including the slot's own noise both dilutes a true outlier
                # (it drags the mean toward itself) and punishes ordinary
                # fluctuation among equally-noisy peers (caught by the
                # uniform-noise test). The margin keeps statistical jitter
                # among peers reading as exactly 1.0.
                others = [noises[j] for j in warmed_idx if j != i]
                ref = sum(others) / len(others)
                r = margin * (ref + eps) / (noises[i] + eps)
                rel[i] = max(self.floor, min(1.0, r))
            return rel
        except Exception:
            return []

    @staticmethod
    def _rows(slots: Any) -> "list[list[float]]":
        if slots is None:
            return []
        if hasattr(slots, "detach"):
            t = slots.detach()
            if t.dim() == 3:
                t = t[0]
            if t.dim() != 2:
                return []
            return [[float(x) for x in row] for row in t.cpu().tolist()]
        return [[float(x) for x in row] for row in slots]

    def telemetry(self) -> dict[str, float | int]:
        noises = list(self._noise.values())
        return {
            "slot_reliability_tracked": len(self._fast),
            "slot_noise_max": round(max(noises), 6) if noises else 0.0,
            "slot_noise_mean": round(sum(noises) / len(noises), 6) if noises else 0.0,
        }
