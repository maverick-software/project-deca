"""WS-EXPAND E7 — scheduled rest: the state machine (pure python).

Consolidation load accrues with active cycles and prediction-error volume;
when it crosses threshold (and enough wake time has passed — the value-drift
guard: long wake periods make replayed estimates extrapolate, so rest is
bounded by time-since-last-rest, not load alone), the agent enters a REST
state: the runtime gates motor output to idle while the (already always-on)
consolidation engines work against a quiet body. Any threat fast-path ABORTS
rest immediately — a sleeping agent must still startle awake.

Honest scope: this milestone ships the scheduling + motor gate + telemetry.
The two-phase intensive pass (real replay then generative rollouts) already
exists as machinery (consolidator + imagination); driving it HARDER during
rest is the E7.3 A/B — to be tuned against the always-on baseline on the rig,
per the evidence review (the load trigger is our research bet).
"""

from __future__ import annotations

from decadic import config as C


class RestController:
    """Load-triggered rest scheduling with a hard wake-time bound."""

    def __init__(
        self,
        *,
        load_threshold: float | None = None,
        min_wake_cycles: int | None = None,
        rest_cycles: int | None = None,
        pc_load_scale: float | None = None,
        pressure_threshold: float | None = None,
    ) -> None:
        self.load_threshold = float(
            load_threshold if load_threshold is not None else C.rest_load_threshold()
        )
        self.min_wake_cycles = int(
            min_wake_cycles if min_wake_cycles is not None else C.rest_min_wake_cycles()
        )
        self.rest_cycles = int(rest_cycles if rest_cycles is not None else C.rest_cycles())
        self.pc_load_scale = float(
            pc_load_scale if pc_load_scale is not None else C.rest_pc_load_scale()
        )
        # WS-ATTN 6.2: backpressure trigger. When the salience-priority tiers
        # fill (deliberation can't keep up with perception), pressure rises;
        # crossing this bound enters rest to drain/consolidate the backlog.
        # 0 disables the pressure trigger (load-only, legacy behavior).
        self.pressure_threshold = float(
            pressure_threshold if pressure_threshold is not None else C.rest_pressure_threshold()
        )
        self._last_pressure = 0.0
        self._load = 0.0
        self._rest_remaining = 0
        self._last_rest_end: int | None = None
        self.rests_entered = 0
        self.rests_aborted = 0

    @property
    def in_rest(self) -> bool:
        return self._rest_remaining > 0

    def note_cycle(
        self, *, cycle: int, pc_loss: float, threat: bool, pressure: float = 0.0
    ) -> bool:
        """Advance one cycle; returns whether the agent is resting NOW.

        Threat aborts rest instantly (and keeps it aborted this cycle).
        ``pressure`` (WS-ATTN tier backpressure) is an additional rest trigger
        alongside accrued load.
        """
        self._last_pressure = float(pressure) if pressure == pressure else 0.0
        if threat:
            if self._rest_remaining > 0:
                self._rest_remaining = 0
                self._last_rest_end = int(cycle)
                self.rests_aborted += 1
            # A threatened agent accrues load but never enters rest this cycle.
            self._load += 1.0
            return False
        if self._rest_remaining > 0:
            self._rest_remaining -= 1
            if self._rest_remaining == 0:
                self._last_rest_end = int(cycle)
            return True
        pc = float(pc_loss) if pc_loss == pc_loss else 0.0  # NaN guard
        self._load += 1.0 + self.pc_load_scale * max(0.0, pc)
        wake_ok = (
            self._last_rest_end is None
            or (int(cycle) - self._last_rest_end) >= self.min_wake_cycles
        )
        pressure_hit = (
            self.pressure_threshold > 0.0 and self._last_pressure >= self.pressure_threshold
        )
        if (self._load >= self.load_threshold or pressure_hit) and wake_ok:
            self._rest_remaining = self.rest_cycles
            self._load = 0.0
            self.rests_entered += 1
            return True
        return False

    def telemetry(self) -> dict[str, float | int]:
        return {
            "rest_active": 1 if self.in_rest else 0,
            "rest_load": round(self._load, 3),
            "rest_remaining": self._rest_remaining,
            "rests_entered": self.rests_entered,
            "rests_aborted": self.rests_aborted,
            "rest_pressure": round(self._last_pressure, 3),
        }
