"""Dual-network memory consolidation (Option B): replay learner + Polyak soft-sync.

A second cognitive stack -- the *consolidator* -- is cloned from the live agent's
stack and trains only on salience-prioritized replay of past transitions, off the
cognitive critical path. Periodically it is blended back into the live stack with
a slow Polyak update (``theta_active <- (1-tau) theta_active + tau theta_cons``),
so consolidation reinforces hard-won structure without overwriting fresh online
learning and without ever stalling the cycle loop.

This mirrors the complementary-learning-systems story: the live stack is the fast
hippocampal learner reacting to the moment; the consolidator is the slow neocortical
learner that replays and integrates experience, with the soft-sync as the gradual
transfer between them. Replay steps run in a worker thread (``asyncio.to_thread``)
so the event loop is never blocked, and the only shared-state write (the sync) is
performed under the agent lock.

The consolidator is ephemeral: it is re-cloned from the live stack whenever the
loop (re)starts, so it never needs its own checkpoint. OFF by default
(``config.consolidation_enabled()``) -> the runtime keeps the no-op stub heartbeat
and the live weights are byte-identical to baseline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

from decadic import config as C
from decadic.config import (
    ai_fwd_weight,
    ai_intero_fwd_weight,
    ai_intero_pref_weight,
)
from decadic.nn.frozen_encoders import intero_preference_weights, preferred_intero_vector
from decadic.nn.neural_stack import NeuralCognitiveStack

logger = logging.getLogger(__name__)


def replay_batch_loss(stack: NeuralCognitiveStack, batch: list, device) -> torch.Tensor:
    """Recompute the live cycle's self-supervised objective for a batch on ``stack``.

    Shared by the consolidator's replay step and the loss-landscape probe so both
    score the *same* objective (predictive coding + proprioceptive forward model +
    interoceptive forward/preference). The recurrent cognitive stack only forwards
    batch size 1 (its GRU/LSTM hidden buffers are ``[1, H]``), so each transition is
    forwarded individually and the per-sample losses are averaged; the recurrent
    buffers are detached on every call, so replaying transitions in arbitrary order
    opens no cross-sample BPTT path. The summed term weights mirror neural_pipeline's
    live learn step (the live per-cycle pain reweight and severity reweight are
    omitted -- those are momentary affective gains, not consolidation targets).
    """
    dev = device
    losses: list[torch.Tensor] = []
    for t in batch:
        out = stack(t.z0.to(dev), t.ep.to(dev), t.mem.to(dev))
        loss = out["pc_loss"]
        prev_state = t.prev_state.to(dev)
        prev_motor = t.prev_motor.to(dev)
        pred = stack.forward_predict(prev_state, prev_motor)
        loss = loss + ai_fwd_weight() * F.mse_loss(pred, t.proprio_target.to(dev))
        if t.drive_on and t.prev_intero is not None and t.intero_now is not None:
            pin = t.prev_intero.to(dev)
            inow = t.intero_now.to(dev)
            pred_i = stack.forward_predict_intero(prev_state, prev_motor, pin)
            l_fwd_i = F.mse_loss(pred_i, inow)
            dim = int(inow.shape[-1])
            s_pref = torch.as_tensor(
                [preferred_intero_vector(dim)], device=dev, dtype=inow.dtype
            )
            w_pref = torch.as_tensor(
                [intero_preference_weights(dim)], device=dev, dtype=inow.dtype
            )
            pred_pref = stack.forward_predict_intero(
                out["z5"], out["motor_u"], inow, detach_params=True
            )
            l_pref = (w_pref * (pred_pref - s_pref).pow(2)).mean()
            loss = loss + ai_intero_fwd_weight() * l_fwd_i + ai_intero_pref_weight() * l_pref
        # Successor-features TD(lambda) regression: the SF head learns the discounted
        # future feature target (the episode's lambda-return of phi) for the realized
        # (state, action). This is the reward-free predictive structure that lets a
        # seen cue inherit value. Only return-annotated transitions (sf_target filled
        # on episode close) feed it; the rest train the one-step objective as before.
        if (
            C.sf_enabled()
            and getattr(stack, "has_successor_model", False)
            and t.sf_target is not None
        ):
            sf_target = torch.as_tensor(
                [t.sf_target], device=dev, dtype=prev_state.dtype
            )
            psi = stack.successor_predict(prev_state, prev_motor)
            loss = loss + C.sf_loss_weight() * F.mse_loss(psi, sf_target)
        # Skill Dojo imitation term: train the consolidator toward a teacher
        # motor target only when replay metadata explicitly carries one. This path
        # never runs in live cognition and the phase-controlled weight decays to
        # zero before autonomous evaluation.
        if t.expert_motor is not None and float(getattr(t, "demo_weight", 0.0) or 0.0) > 0:
            target = torch.as_tensor(t.expert_motor, device=dev, dtype=out["motor_u"].dtype).reshape(1, -1)
            n = int(out["motor_u"].shape[-1])
            if target.shape[-1] < n:
                target = F.pad(target, (0, n - target.shape[-1]))
            target = target[:, :n]
            loss = loss + float(t.demo_weight) * F.mse_loss(out["motor_u"], target)
        losses.append(loss)
    return torch.stack(losses).mean()


class ConsolidationManager:
    """Owns the cloned consolidator stack + its optimizer; runs replay and soft-sync."""

    def __init__(self, bundle, *, lock: asyncio.Lock | None = None, lr: float | None = None) -> None:
        self.bundle = bundle
        self.device = bundle.device
        self.lock = lock
        self.cons_stack = self._clone_stack()
        if lr is None:
            lr = float(bundle.optimizer.param_groups[0]["lr"])
        self.cons_opt = torch.optim.Adam(self.cons_stack.parameters(), lr=lr)
        self.replay_steps = 0
        self.last_loss = 0.0
        self.last_imagined_loss = 0.0
        self.last_sync_cycle = 0
        self.syncs = 0

    # --- cloning ------------------------------------------------------------

    def _clone_stack(self) -> NeuralCognitiveStack:
        """Build a fresh stack of the live architecture and copy its weights."""
        stack = NeuralCognitiveStack(
            self.bundle.cfg, self.bundle.flags, self.bundle.faculties
        ).to(self.device)
        stack.load_state_dict(self.bundle.stack.state_dict())
        return stack

    def refresh_from_active(self) -> None:
        """Re-clone the consolidator weights from the current live stack."""
        self.cons_stack.load_state_dict(self.bundle.stack.state_dict())

    # --- replay loss --------------------------------------------------------

    def replay_loss(self, batch: list) -> torch.Tensor:
        """Recompute the live cycle's objective on the consolidator for a batch.

        Thin wrapper over the shared ``replay_batch_loss`` so the consolidator and
        the loss-landscape probe score the identical objective.
        """
        return replay_batch_loss(self.cons_stack, batch, self.device)

    def held_out_loss(self, batch: list) -> float:
        """Evaluate the consolidator's replay loss on a batch without training."""
        if not batch:
            return 0.0
        self.cons_stack.eval()
        with torch.no_grad():
            loss = self.replay_loss(batch)
        self.cons_stack.train()
        return float(loss.detach().cpu().item())

    def consolidate_once(self, buffer, batch_size: int) -> float | None:
        """One prioritized-replay gradient step on the consolidator. None if no-op."""
        batch = buffer.sample(batch_size)
        if not batch:
            return None
        self.cons_stack.train()
        self.cons_opt.zero_grad(set_to_none=True)
        loss = self.replay_loss(batch)
        # Model-based imagined replay (gated, default OFF): add a trust-weighted SF
        # loss against short imagined interoceptive rollouts from the batch's real
        # start states. Kept OUT of the shared replay_loss so the loss-landscape
        # probe still scores the pure lived objective.
        self.last_imagined_loss = 0.0
        if C.sf_enabled() and C.imagination_enabled():
            from decadic.consolidation.imagination import imagined_sf_loss

            il = imagined_sf_loss(
                self.cons_stack,
                batch,
                gamma=C.sf_gamma(),
                horizon=C.imagination_horizon(),
                device=self.device,
            )
            if il is not None and torch.isfinite(il):
                loss = loss + C.imagination_weight() * il
                self.last_imagined_loss = float(il.detach().cpu().item())
        if not torch.isfinite(loss):
            return None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cons_stack.parameters(), 5.0)
        self.cons_opt.step()
        self.replay_steps += 1
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss

    # --- soft sync ----------------------------------------------------------

    def soft_sync(self, tau: float) -> None:
        """Polyak update: nudge the LIVE stack toward the consolidator by ``tau``.

        Parameters only (recurrent/mask buffers are left untouched, so the live
        agent keeps its own working state). MUST be called with the agent lock
        held -- it mutates the live weights in place.
        """
        tau = float(min(1.0, max(0.0, tau)))
        if tau <= 0.0:
            return
        with torch.no_grad():
            cons_params = dict(self.cons_stack.named_parameters())
            for name, p_act in self.bundle.stack.named_parameters():
                p_cons = cons_params.get(name)
                if p_cons is None or p_cons.shape != p_act.shape:
                    continue
                if not torch.isfinite(p_cons).all():
                    continue
                p_act.mul_(1.0 - tau).add_(p_cons, alpha=tau)
        self.syncs += 1

    def _consolidate_burst(self, buffer, batch_size: int, steps: int) -> float | None:
        last: float | None = None
        for _ in range(max(1, int(steps))):
            result = self.consolidate_once(buffer, batch_size)
            if result is not None:
                last = result
        return last

    # --- background loop ----------------------------------------------------

    async def run_loop(
        self,
        buffer,
        *,
        should_continue: Callable[[], bool],
        current_cycle: Callable[[], int],
        on_sync: Callable[[int, float, int], None] | None = None,
    ) -> None:
        """Periodic replay-burst + soft-sync loop. Replaces the stub heartbeat.

        Sleeps ``consolidation_sync_interval_s`` between bursts. Each burst replays
        in a worker thread (off the event loop); the soft-sync then runs under the
        agent lock so it serializes with the live optimizer step.
        """
        interval = C.consolidation_sync_interval_s()
        if interval <= 0:
            return
        batch = C.consolidation_replay_batch()
        tau = C.consolidation_sync_tau()
        steps = C.consolidation_steps_per_burst()
        logger.info(
            "consolidation_start agent_id=%s interval_s=%.2f batch=%s tau=%.3f steps=%s",
            self.bundle.agent_id,
            interval,
            batch,
            tau,
            steps,
        )
        while should_continue():
            await asyncio.sleep(interval)
            if not should_continue():
                break
            if len(buffer) < batch:
                continue
            loss = await asyncio.to_thread(self._consolidate_burst, buffer, batch, steps)
            if loss is None:
                continue
            if self.lock is not None:
                async with self.lock:
                    await asyncio.to_thread(self.soft_sync, tau)
            else:
                self.soft_sync(tau)
            self.last_sync_cycle = int(current_cycle())
            logger.info(
                "consolidation_sync agent_id=%s cycle=%s replay_steps=%s loss=%.5f",
                self.bundle.agent_id,
                self.last_sync_cycle,
                self.replay_steps,
                self.last_loss,
            )
            if on_sync is not None:
                on_sync(self.replay_steps, self.last_loss, self.last_sync_cycle)
