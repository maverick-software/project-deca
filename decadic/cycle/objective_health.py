"""Objective-health canary for the live neural update.

This module supervises optimizer stability only. It does not add reward, labels,
or cognitive inputs; it decides whether this cycle's gradient step is safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from decadic import config as C

LossCanaryState = Literal["healthy", "warming", "warning", "diverging"]
OptimizerAction = Literal["normal", "scaled", "skipped"]


@dataclass
class ObjectiveHealthReport:
    state: LossCanaryState = "warming"
    reason: str = "initializing"
    pressure: float = 0.0
    optimizer_action: OptimizerAction = "normal"
    step_scale: float = 1.0
    total_loss: float | None = None
    pc_loss: float | None = None
    loss_ema: float | None = None
    pc_ema: float | None = None
    loss_slope_ema: float = 0.0
    pc_slope_ema: float = 0.0
    loss_jump_ratio: float = 1.0
    cycle_count: int = 0
    warmup_cycles: int = 0


@dataclass
class ObjectiveHealthCanary:
    """Stateful finite-divergence detector for the live loss stream."""

    beta: float = 0.98
    cycle_count: int = 0
    prev_total_loss: float | None = None
    loss_ema: float | None = None
    pc_ema: float | None = None
    loss_slope_ema: float = 0.0
    pc_slope_ema: float = 0.0
    last_report: ObjectiveHealthReport = field(default_factory=ObjectiveHealthReport)

    def update(
        self,
        *,
        total_loss: float,
        pc_loss: float,
        forward_finite: bool,
    ) -> ObjectiveHealthReport:
        self.cycle_count += 1
        warmup = C.loss_canary_warmup_cycles()
        enabled = C.loss_canary_enabled()
        state: LossCanaryState = "healthy" if self.cycle_count > warmup else "warming"
        reason = "warming" if state == "warming" else ""
        step_scale = 1.0
        jump_ratio = 1.0

        total_finite = bool(forward_finite and math.isfinite(total_loss) and math.isfinite(pc_loss))
        if not enabled:
            report = ObjectiveHealthReport(
                state="healthy",
                reason="disabled",
                pressure=0.0,
                optimizer_action="normal",
                step_scale=1.0,
                total_loss=total_loss if math.isfinite(total_loss) else None,
                pc_loss=pc_loss if math.isfinite(pc_loss) else None,
                loss_ema=self.loss_ema,
                pc_ema=self.pc_ema,
                loss_slope_ema=self.loss_slope_ema,
                pc_slope_ema=self.pc_slope_ema,
                loss_jump_ratio=jump_ratio,
                cycle_count=self.cycle_count,
                warmup_cycles=warmup,
            )
            self.last_report = report
            return report

        if not total_finite:
            report = ObjectiveHealthReport(
                state="diverging",
                reason="nonfinite",
                pressure=1.0,
                optimizer_action="skipped",
                step_scale=0.0,
                total_loss=None,
                pc_loss=None,
                loss_ema=self.loss_ema,
                pc_ema=self.pc_ema,
                loss_slope_ema=self.loss_slope_ema,
                pc_slope_ema=self.pc_slope_ema,
                loss_jump_ratio=jump_ratio,
                cycle_count=self.cycle_count,
                warmup_cycles=warmup,
            )
            self.last_report = report
            return report

        total = max(0.0, float(total_loss))
        pc = max(0.0, float(pc_loss))

        if self.prev_total_loss is not None and self.prev_total_loss > 1e-12:
            jump_ratio = total / max(self.prev_total_loss, 1e-12)

        hard_jump = jump_ratio >= C.loss_canary_hard_jump_ratio()
        warn_jump = jump_ratio >= C.loss_canary_warn_jump_ratio()

        prev_loss_ema = self.loss_ema
        prev_pc_ema = self.pc_ema
        self.loss_ema = total if self.loss_ema is None else self.beta * self.loss_ema + (1.0 - self.beta) * total
        self.pc_ema = pc if self.pc_ema is None else self.beta * self.pc_ema + (1.0 - self.beta) * pc
        if prev_loss_ema is not None:
            self.loss_slope_ema = 0.9 * self.loss_slope_ema + 0.1 * float(self.loss_ema - prev_loss_ema)
        if prev_pc_ema is not None:
            self.pc_slope_ema = 0.9 * self.pc_slope_ema + 0.1 * float(self.pc_ema - prev_pc_ema)
        self.prev_total_loss = total

        warmup_done = self.cycle_count >= warmup
        hard_ema = warmup_done and self.pc_ema is not None and self.pc_ema >= C.loss_canary_hard_pcema()
        warn_ema = warmup_done and self.pc_ema is not None and self.pc_ema >= C.loss_canary_warn_pcema()

        if hard_jump or hard_ema:
            state = "diverging"
            reason = "loss_jump_ratio" if hard_jump else "pc_ema_hard"
            step_scale = 0.0
        elif warn_jump or warn_ema:
            state = "warning"
            reason = "loss_jump_ratio" if warn_jump else "pc_ema_warning"
            step_scale = C.loss_canary_warning_step_scale()

        pressure = 1.0 if state == "diverging" else (0.5 if state == "warning" else 0.0)
        action: OptimizerAction = "skipped" if step_scale <= 0.0 else ("scaled" if step_scale < 1.0 else "normal")
        report = ObjectiveHealthReport(
            state=state,
            reason=reason,
            pressure=pressure,
            optimizer_action=action,
            step_scale=step_scale,
            total_loss=total,
            pc_loss=pc,
            loss_ema=self.loss_ema,
            pc_ema=self.pc_ema,
            loss_slope_ema=self.loss_slope_ema,
            pc_slope_ema=self.pc_slope_ema,
            loss_jump_ratio=jump_ratio,
            cycle_count=self.cycle_count,
            warmup_cycles=warmup,
        )
        self.last_report = report
        return report

