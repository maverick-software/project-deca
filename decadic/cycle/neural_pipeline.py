"""Neural Decadic cycle — Phase 2 forward, predictive coding, stage 10."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from decadic import config as C
from decadic.config import (
    DEFAULT_SESSION_RECENCY,
    ai_fwd_weight,
    assist_gain_for_cycle,
    motor_exploration_sigma,
)
from decadic.memory.embeddings import perceptual_key, query_vector_from_state_bus
from decadic.metrics.integration import self_state_vector
from decadic.nn.workspace import GlobalWorkspace
from decadic.perception.object_files import (
    evaluate_discovery_health,
    object_files_from_proposals,
)
from decadic.perception.organ import PerceptionOrgan
from decadic.cycle.integration_window import IntegrationWindow
from decadic.cycle import cognition_trace, narrative as cognition_narrative
from decadic.cycle.stages import stage_10
from decadic.interpretability import probes as interp_probes
from decadic.cycle.types import CycleContext, StageTrace
from decadic.nn.bundle import NeuralBundle
from decadic.nn.config import viability_pe_scale
from decadic.nn.frozen_encoders import (
    controllable_effort_vector,
    controllable_intero_vector,
    controllable_proprio_vector,
    controllable_tactile_vector,
    intero_preference_weights,
    preferred_intero_vector,
)
from decadic.state.curiosity import CuriosityState, compute_curiosity
from decadic.state.viability import (
    apply_pain_pleasure_to_B,
    drive_reduction_reward,
    ema_affect,
    interoceptive_drive_pain,
    reward_success_stub,
    stub_prediction_error_penalty,
    viability_delta_to_signals,
)


def _utc_ts() -> str:
    return datetime.now(UTC).isoformat()


def _np_assign(dest: np.ndarray, src: np.ndarray) -> None:
    flat = np.asarray(src, dtype=np.float32).reshape(-1)
    n = min(dest.shape[0], flat.shape[0])
    dest[:] = 0
    dest[:n] = flat[:n]


def _slot_object_losses(attn: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Label-free regularizers that discourage slot collapse."""
    eps = 1e-8
    b, k, n = attn.shape
    flat = attn.reshape(b, k, n)
    norm = flat / (flat.norm(dim=-1, keepdim=True) + eps)
    pair = torch.bmm(norm, norm.transpose(1, 2))
    eye = torch.eye(k, device=attn.device, dtype=attn.dtype).unsqueeze(0)
    diversity = (pair * (1.0 - eye)).sum() / max(1, b * k * (k - 1))

    per_patch = attn / (attn.sum(dim=1, keepdim=True) + eps)
    entropy = -(per_patch * (per_patch + eps).log()).sum(dim=1).mean()

    uv = centroids[..., :2]
    d = torch.cdist(uv, uv, p=2)
    close = torch.exp(-(d * d) / 0.01) * (1.0 - eye)
    spatial = close.sum() / max(1, b * k * (k - 1))
    return diversity, entropy, spatial


def _slot_mask_entropies(attn: torch.Tensor) -> list[float]:
    eps = 1e-8
    w = attn[0] / (attn[0].sum(dim=-1, keepdim=True) + eps)
    ent = -(w * (w + eps).log()).sum(dim=-1)
    if w.shape[-1] > 1:
        ent = ent / math.log(float(w.shape[-1]))
    return [float(x) for x in ent.detach().float().cpu().tolist()]


def _stable_object_file_snapshots(wm: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in getattr(wm, "active_slots", lambda: [])():
        out.append(
            {
                "object_id": str(getattr(s, "entity_id", "")),
                "idx": -1,
                "centroid_uv": list(getattr(s, "uv", None) or []) or None,
                "relative": list(getattr(s, "relative", None) or []) or None,
                "bearing": list(getattr(s, "bearing", None) or []) or None,
                "appearance": list(getattr(s, "appearance", None) or []) or None,
                "motion": list(getattr(s, "motion", None) or []) or None,
                "depth": None,
                "persistence": float(getattr(s, "salience", 0.0) or 0.0),
                "agency": float(getattr(s, "agency", 0.0) or 0.0),
                "kind_hint": str(getattr(s, "kind_hint", "object")),
                "confidence": float(getattr(s, "confidence", 0.0) or 0.0),
                "presence": float(getattr(s, "salience", 0.0) or 0.0),
                "spread": None,
                "mask_entropy": None,
                "flow": list(getattr(s, "motion", None) or []) or None,
                "local_motion": float(getattr(s, "local_motion", 0.0) or 0.0),
                "retina_contrast": float(getattr(s, "retina_contrast", 0.0) or 0.0),
                "looming": float(getattr(s, "looming", 0.0) or 0.0),
                "property_evidence": dict(getattr(s, "property_evidence", {}) or {}),
            }
        )
    return out


def encode_observations(bundle: NeuralBundle, observations: list[dict]) -> torch.Tensor | None:
    """No-grad encode of buffered observations into one [K, D] batch.

    This is the "parallel sessions" pass: several percepts are run through the
    frozen encoders within a single cycle, decoupled from the serialized,
    gradient-bearing learn step. Encoding per-item then stacking is numerically
    identical to a single encode, so callers get batched throughput with parity.
    """
    fused: list[torch.Tensor] = []
    with torch.no_grad():
        for obs in observations:
            if obs is None:
                continue
            fused.append(bundle.encoders(obs))
    if not fused:
        return None
    return torch.cat(fused, dim=0)


def pool_fused(
    encoded_old: torch.Tensor | None, fused_latest: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Recency-weighted pool of older no-grad encodes with the latest grad encode.

    Weights are gamma^age (latest frame has age 0 and weight 1, oldest the
    smallest), normalized to sum to 1. Gradient flows only through
    `fused_latest`. With no older frames this is exactly `fused_latest`.
    """
    n_old = int(encoded_old.shape[0]) if encoded_old is not None else 0
    if n_old == 0:
        return fused_latest
    assert encoded_old is not None
    weights = [gamma ** (n_old - j) for j in range(n_old)]  # oldest first
    w_total = sum(weights) + 1.0
    pooled_old = sum(w * encoded_old[j : j + 1] for j, w in enumerate(weights))
    return (pooled_old + fused_latest) / w_total


def _plasticity_instability(stack, pc_ema: float | None, threshold: float) -> bool:
    """True if any plastic tensor is non-finite or the pc-loss EMA has diverged."""
    if pc_ema is not None and threshold > 0 and pc_ema > threshold:
        return True
    for blk in stack.plastic_blocks():
        for t in (blk.l1_weight, blk.l2_weight, blk.hebb1, blk.hebb2, blk.alpha):
            if not torch.isfinite(t).all():
                return True
    return False


def apply_plasticity_step(
    ctx: CycleContext, bundle: NeuralBundle, *, pc_loss: float, modulation: float
) -> dict:
    """Post-optimizer A/B/C updates + instability guard; returns telemetry.

    No-op (empty dict) unless the stack was built with at least one plastic
    block. Each subsystem is independently gated by its config flag so the three
    can be enabled in any combination.
    """
    stack = bundle.stack
    if not getattr(stack, "has_plastic", False):
        return {}
    st = bundle.plasticity_state
    if st is None:
        return {}

    st.pc_ema = pc_loss if st.pc_ema is None else (0.98 * st.pc_ema + 0.02 * pc_loss)

    # Instability guard: freeze further plastic updates and sanitize the whole
    # stack to finite values (NaN/inf -> 0), then drop the optimizer moments so
    # learning restarts cleanly. Plasticity stays frozen until the next reset().
    froze = False
    if not st.frozen and _plasticity_instability(stack, st.pc_ema, C.plasticity_instability_pcloss()):
        st.frozen = True
        froze = True
        st.pc_ema = None
        with torch.no_grad():
            for blk in stack.plastic_blocks():
                blk.reset_plastic_trace()
            for p in stack.parameters():
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
            for buf in stack.buffers():
                if buf.dtype.is_floating_point:
                    buf.copy_(torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0))
            for p in bundle.encoders.parameters():
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0))
        bundle.optimizer.state.clear()
        # Drop the cross-cycle transition buffers so a poisoned state can't feed
        # the next forward model step.
        bundle.prev_state = None
        bundle.prev_motor = None
        bundle.prev_intero = None

    structural = False
    # Per-cycle edge counts (non-zero only on the cycle the event fires).
    connections_rewired = 0
    neurons_woken = 0
    if not st.frozen:
        # A — neuromodulated Hebbian trace update (gated by pleasure - pain).
        if C.plasticity_enabled():
            stack.hebbian_update_all(modulation, C.plasticity_eta())

        # B — keep pruned/dormant weights pinned at 0; periodic prune+grow rewire.
        if C.sparse_enabled():
            stack.enforce_masks_all()
            st.cycles_since_rewire += 1
            if st.cycles_since_rewire >= C.sparse_rewire_interval():
                st.cycles_since_rewire = 0
                connections_rewired = stack.rewire_all(C.sparse_rewire_fraction())
                if connections_rewired:
                    bundle.reset_optimizer_state(stack.rewire_changed_params())
                    st.rewire_events += 1
                    structural = True

        # C — wake dormant neurons up to the live cap while pc-loss stays high.
        if C.growth_enabled():
            st.cycles_since_growth += 1
            if st.cycles_since_growth >= C.growth_interval():
                st.cycles_since_growth = 0
                cap = st.max_neurons or C.max_neurons()
                if (st.pc_ema or 0.0) > C.growth_pcloss_threshold() and stack.growth_room(cap):
                    awake_before = stack.awake_neurons()
                    changed = stack.grow_step(C.growth_step(), cap)
                    if changed:
                        neurons_woken = stack.awake_neurons() - awake_before
                        bundle.reset_optimizer_state(changed)
                        st.growth_events += 1
                        structural = True
                        cost = C.growth_viability_cost()
                        if cost > 0:
                            ctx.viability.value = max(0.0, ctx.viability.value - cost)

    if structural:
        st.structural_version += 1

    return {
        "plasticity_alpha": round(stack.plastic_alpha_mean(), 6),
        "sparse_density": round(stack.connection_density(), 6),
        "awake_neurons": stack.awake_neurons(),
        "allocated_neurons": stack.allocated_neurons(),
        "active_connections": stack.active_connections(),
        "rewire_events": st.rewire_events,
        "growth_events": st.growth_events,
        "plasticity_frozen": st.frozen,
        "structural_change": structural,
        "structural_version": st.structural_version,
        # Per-cycle edge events (True only on the cycle they fire) + counts, for
        # structured event logging in the runtime.
        "rewired": bool(connections_rewired),
        "connections_rewired": int(connections_rewired),
        "grew": bool(neurons_woken),
        "neurons_woken": int(neurons_woken),
        "froze": bool(froze),
    }


def run_neural_cycle(ctx: CycleContext, bundle: NeuralBundle) -> dict:
    ctx.state_bus.cycle_index += 1
    traces: list[StageTrace] = []
    ctx.latents.clear()

    # Parallel-session encode: drain up to K buffered percepts. Older frames go
    # through the frozen encoders under no_grad; only the latest frame carries
    # gradient, keeping the single serialized learn step decoupled from the
    # batched perception pass.
    pending = ctx.pending_observations or (
        [ctx.last_observation] if ctx.last_observation is not None else []
    )
    older = pending[:-1]
    latest = pending[-1] if pending else ctx.last_observation

    te0 = time.perf_counter()
    encoded_old = encode_observations(bundle, older)
    encode_phase_ms = (time.perf_counter() - te0) * 1000.0

    t0 = time.perf_counter()
    fused_latest = bundle.encoders(latest)
    # Recency-weighted pool over [oldest..latest]: w ∝ gamma^age (latest age 0).
    # With K=1 this reduces exactly to the single-frame encode (parity).
    gamma = float(os.environ.get("DECADIC_SESSION_RECENCY", str(DEFAULT_SESSION_RECENCY)))
    fused = pool_fused(encoded_old, fused_latest, gamma)
    n_old = int(encoded_old.shape[0]) if encoded_old is not None else 0
    parallel_sessions = n_old + (1 if latest is not None else 0)
    # Bottom-up percept. z0_bu is the pure sensory encode; z0 (set after the
    # top-down blend below) is what actually feeds the deliberative stack.
    z0_bu = bundle.stack.ingress(fused)

    # --- Discovered perception: object-centric slot attention -----------------
    # In discovered mode the egocentric graph emerges from the camera. Slot
    # attention parses the patch-feature map into object proposals; the pooled
    # slots are injected additively into z0 (zero-init projection -> starts at
    # exact bottom-up parity) so the deliberative stack sees object structure.
    discovered = ctx.perception_mode == "discovered" and getattr(
        bundle.stack, "has_slots", False
    )
    slot_out = None
    patch_tokens = None
    if discovered:
        patch_tokens = bundle.encoders.vision_patch_tokens(latest)
        if patch_tokens is not None:
            patch_tokens = patch_tokens.to(device=bundle.device, dtype=z0_bu.dtype)
            slot_out = bundle.stack.slot_encode(patch_tokens)
            pooled = bundle.stack.slot_pool(slot_out["slots"], slot_out["presence"])
            z0_bu = z0_bu + bundle.stack.slot_ingress(pooled)

    # --- Temporal-integration window: commit a bound "now" (Phase 3) -----------
    # Accumulate the bottom-up percept over a wall-clock window; act on the LAST
    # committed moment until the window closes and a new "now" is bound, so
    # changing the window length shifts when "now" updates. OFF (window_ms <= 0)
    # => the freshest percept is always "now" (byte-identical to before).
    win_ms = (
        ctx.integration_window_ms
        if ctx.integration_window_ms is not None
        else C.integration_window_ms()
    )
    window_block: dict | None = None
    if win_ms > 0:
        iw = bundle._integration_window
        if iw is None or float(getattr(iw, "window_ms", -1.0)) != float(win_ms):
            iw = IntegrationWindow(
                window_ms=win_ms, max_frames=C.integration_window_max_frames()
            )
            bundle._integration_window = iw
            bundle._committed_now = None
        res = iw.push(z0_bu.detach().float().cpu().numpy().reshape(-1), now_s=time.time())
        if res.committed is not None:
            bundle._committed_now = torch.as_tensor(
                res.committed, device=z0_bu.device, dtype=z0_bu.dtype
            ).reshape(1, -1)
        if bundle._committed_now is not None:
            z0_bu = bundle._committed_now
        window_block = {
            "enabled": True,
            "window_ms": float(win_ms),
            "committed": bool(res.committed is not None),
            "buffered": int(res.buffered),
        }
        ctx.latents["integration_window"] = window_block
    else:
        bundle._integration_window = None
        bundle._committed_now = None

    enc_ms = (time.perf_counter() - t0) * 1000.0  # stage 1: sensory perception
    ep = torch.tensor(
        [
            [
                ctx.viability.value / 100.0,
                ctx.state_bus.pain_scalar,
                ctx.state_bus.pleasure_scalar,
                ctx.state_bus.priority_scalar,
            ]
        ],
        device=bundle.device,
        dtype=z0_bu.dtype,
    )
    # --- Predictive affect: colour the proxy with the expected next affect ------
    # The previous cycle's actual affect context (detached) is run through the
    # forward model; its predicted delta is added to ep so the agent perceives in
    # light of how it expects to feel. The predictor's weights stay on the graph
    # (trained for free by the main objective). No-op until prev_affect exists;
    # zero-init => byte-identical until learned. Off => prev_affect stays None.
    affect_predicted = False
    if getattr(bundle.stack, "has_predictive_affect", False):
        ep_actual = ep.detach().clone()
        if bundle.prev_affect is not None:
            pa = bundle.prev_affect.to(device=ep.device, dtype=ep.dtype).reshape(1, -1)
            if pa.shape[-1] == ep.shape[-1]:
                ep = ep + C.predictive_affect_gain() * bundle.stack.affect_predictor(pa)
                affect_predicted = True
        bundle.prev_affect = ep_actual
    # Loop 2 — perceptual-similarity retrieval. The query carries a parameter-free
    # key compressed from the *learned* percept, so recall is driven by sensory
    # likeness in z0's learned geometry (off ⇒ key is None ⇒ zero tail ⇒ parity).
    feedback_on = bundle.stack.has_perception_feedback
    percept_key_np = None
    if feedback_on:
        with torch.no_grad():
            percept_key_np = perceptual_key(
                z0_bu.detach().float().cpu().reshape(-1).numpy()
            )
        ctx.latents["percept_key"] = percept_key_np.astype(float).tolist()
    tm0 = time.perf_counter()
    qv = query_vector_from_state_bus(ctx.state_bus, percept_key_np)
    mem_np = ctx.episodic.retrieval_context_vector(
        qv,
        bundle.cfg.memory_context_dim,
        top_k=int(os.environ.get("DECADIC_MEMORY_TOP_K", "5")),
        min_salience=float(os.environ.get("DECADIC_MEMORY_MIN_SALIENCE", "0")),
    )
    mem_ms = (time.perf_counter() - tm0) * 1000.0  # episodic recall → stage 3
    mem_t = torch.as_tensor(mem_np, device=bundle.device, dtype=z0_bu.dtype).unsqueeze(0)

    # Loop 1 — precision-gated top-down predictive perception. History (last z5,
    # the LSTM state, the WM scene latent, and the just-recalled memory) and
    # interoception are all *detached*, so the loop shapes this cycle's percept
    # without opening a cross-cycle BPTT path. Off (or gate→1) ⇒ z0 == z0_bu.
    wm_obj = getattr(ctx.perceptual, "working_memory", None)
    scene_t = None
    if feedback_on:
        scene_list = getattr(wm_obj, "scene_latent", None) if wm_obj is not None else None
        if scene_list:
            scene_t = torch.as_tensor(
                scene_list, device=bundle.device, dtype=z0_bu.dtype
            ).unsqueeze(0)
    intero_t = torch.tensor(
        [
            [
                float(ctx.state_bus.pain_scalar),
                float(ctx.state_bus.pleasure_scalar),
                float(ctx.viability.value) / 100.0,
            ]
        ],
        device=bundle.device,
        dtype=z0_bu.dtype,
    )
    z0_eff, z0_hat, gate = bundle.stack.top_down_perceive(
        z0_bu,
        prev_z5=bundle.prev_state,
        lstm_h=bundle.stack.lstm_h,
        mem=mem_t,
        scene=scene_t,
        intero=intero_t,
    )
    z0 = z0_eff
    # Self-state feedback spine: condition this cycle on the previous cycle's
    # detached self-report. None on the first cycle / when the faculty is off
    # (then the stack ignores it) -> full parity with the no-spine baseline.
    self_prev_fed = bundle.prev_self if bundle.stack.has_self_model_feedback else None
    # Represented self (Phase 5): feed the previous cycle's modelled self embedding.
    repself_fed = (
        bundle.prev_repself if getattr(bundle.stack, "has_represented_self", False) else None
    )
    # bf16 autocast on the forward only when the memory-efficient path is on (CUDA);
    # a nullcontext otherwise, so the fp32 / CPU / test path is byte-identical.
    with bundle.train_autocast():
        out = bundle.stack(z0, ep, mem_t, self_prev=self_prev_fed, repself_prev=repself_fed)
    fwd_ms = (time.perf_counter() - t0) * 1000.0

    # Autocast can leave the forward's float outputs in bf16 on CUDA. The rest of
    # the pipeline (NumPy extraction -> NumPy has no bf16, the State Bus, the
    # active-inference losses) expects fp32, so normalize the top-level floating
    # tensors back to fp32 here, where the autocast region has ended. The bf16
    # forward activations are already realized (the memory saving stands) and the
    # cast preserves the grad graph so backward still works. No-op on the
    # fp32/CPU/test path (train_autocast is a nullcontext there).
    out = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in out.items()
    }

    # NaN firewall (always on, independent of plasticity): if the forward pass or
    # the persistent recurrent buffers went non-finite, this cycle's update is
    # skipped and the recurrent state is reset below, so a single NaN can't lock
    # the body forever by re-poisoning every subsequent forward pass.
    forward_finite = bool(
        torch.isfinite(out["motor_u"]).all()
        and torch.isfinite(out["z5"]).all()
        and torch.isfinite(out["state_mind"]).all()
        and torch.isfinite(bundle.stack.gru_h).all()
        and torch.isfinite(bundle.stack.lstm_h).all()
        and torch.isfinite(bundle.stack.lstm_c).all()
    )

    pain_w = 1.0 + float(os.environ.get("DECADIC_PAIN_PC_WEIGHT", "0.35")) * float(
        ctx.state_bus.pain_scalar
    )
    loss = out["pc_loss"] * pain_w

    # Loop 1 self-supervised term: top-down learns to predict the bottom-up
    # percept from history (target detached so prediction chases perception, not
    # the reverse). The untouched pc_loss above keeps surprise sensitivity, so
    # this cannot collapse into a self-confirming hallucination.
    l_percept = torch.zeros((), device=z0_bu.device, dtype=z0_bu.dtype)
    if z0_hat is not None:
        l_percept = torch.nn.functional.mse_loss(z0_hat, z0_bu.detach())
        loss = loss + C.perception_pred_weight() * l_percept

    # --- Active inference: forward-model PE + homeostatic drive reduction -------
    # Only learn the sensorimotor loop when a body is actually streaming state.
    has_body = latest is not None
    # Homeostatic drive is the root motivation: always on when a body streams
    # reservoirs and the interoceptive head is present (built unconditionally).
    drive_on = ctx.homeostasis is not None and getattr(
        bundle.stack, "has_intero_model", False
    )
    fwd_dim = bundle.cfg.forward_pred_dim
    motor_u = out["motor_u"]
    z5_t = out["z5"]
    l_fwd = torch.zeros((), device=z0.device, dtype=z0.dtype)
    l_fwd_tactile = torch.zeros((), device=z0.device, dtype=z0.dtype)
    l_fwd_effort = torch.zeros((), device=z0.device, dtype=z0.dtype)
    l_fwd_intero = torch.zeros((), device=z0.device, dtype=z0.dtype)
    l_pref_intero = torch.zeros((), device=z0.device, dtype=z0.dtype)
    # Successor-features value shaping (Layer-2). sf_value_last: scalar value of the
    # chosen action; sf_value_w: the (ramped) shaping weight actually applied. 0 until
    # the SF head has learned and the ramp has opened, so behavior starts identical.
    sf_value_last = 0.0
    sf_value_w = 0.0
    # Per-joint proprioceptive forward-model error: the squared error of each
    # predicted joint qpos channel. Feeds the body's joint-brace ROM curriculum
    # (a joint earns range of motion only once the brain predicts it well). The
    # controllable proprio vector is [orientation/height/velocity (BASE dims)] +
    # joint qpos, so dims [BASE:] map channel-for-channel onto the body's hinges.
    joint_pred_error: list[float] = []
    if has_body:
        s_target = torch.as_tensor(
            [controllable_proprio_vector(latest, fwd_dim)], device=z0.device, dtype=z0.dtype
        )
        # World model learns the realized transition (prev_state, prev_motor) -> s_target.
        if bundle.prev_state is not None and bundle.prev_motor is not None:
            pred_prev = bundle.stack.forward_predict(
                bundle.prev_state, bundle.prev_motor, detach_params=False
            )
            l_fwd = torch.nn.functional.mse_loss(pred_prev, s_target.detach())
            with torch.no_grad():
                per_dim_se = (pred_prev.detach() - s_target.detach()) ** 2
                base = int(C.CONTROLLABLE_PROPRIO_BASE)
                joint_pred_error = [
                    (float(x) if math.isfinite(float(x)) else 1.0)
                    for x in per_dim_se.reshape(-1)[base:].tolist()
                ]
        loss = loss + ai_fwd_weight() * l_fwd

        # Tactile active inference: the world model learns which actions load which
        # body part from the realized transition (prev_state, prev_motor) -> realized
        # per-part loads. PE-only: touch has no innate setpoint, so there is no
        # preference term -- this is the per-limb credit-assignment signal that the
        # brain needs to discover how to push off with each part.
        if getattr(bundle.stack, "has_tactile_model", False):
            tactile_dim = int(C.TACTILE_PRED_DIM)
            t_target = torch.as_tensor(
                [controllable_tactile_vector(latest, tactile_dim)],
                device=z0.device,
                dtype=z0.dtype,
            )
            if bundle.prev_state is not None and bundle.prev_motor is not None:
                pred_prev_t = bundle.stack.forward_predict_tactile(
                    bundle.prev_state, bundle.prev_motor, detach_params=False
                )
                l_fwd_tactile = torch.nn.functional.mse_loss(
                    pred_prev_t, t_target.detach()
                )
            loss = loss + C.ai_tactile_fwd_weight() * l_fwd_tactile

        # Effort/body-map active inference: predict localized effort, strain,
        # fatigue and pain caused by the previous action. A small detached cost
        # term gives the policy pressure toward efficient movement without making
        # stillness the dominant objective.
        if getattr(bundle.stack, "has_effort_model", False):
            effort_dim = int(C.EFFORT_PRED_DIM)
            e_target = torch.as_tensor(
                [controllable_effort_vector(latest, effort_dim)],
                device=z0.device,
                dtype=z0.dtype,
            )
            if bundle.prev_state is not None and bundle.prev_motor is not None:
                pred_prev_e = bundle.stack.forward_predict_effort(
                    bundle.prev_state, bundle.prev_motor, detach_params=False
                )
                l_fwd_effort = torch.nn.functional.mse_loss(
                    pred_prev_e, e_target.detach()
                )
            loss = loss + C.ai_effort_fwd_weight() * l_fwd_effort
            pred_effort_pref = bundle.stack.forward_predict_effort(
                z5_t, motor_u, detach_params=True
            )
            loss = loss + C.ai_effort_cost_weight() * pred_effort_pref.pow(2).mean()

        # Interoceptive active inference: the root survival drive. The world model
        # learns reservoir dynamics from realized transitions; the policy is pulled
        # toward whatever it predicts will raise depleted reservoirs toward the full
        # setpoint (drive reduction). The satisfier is never labeled - seeking is
        # discovered from the agent's own (state, action) -> reservoir transitions.
        if drive_on:
            intero_dim = int(C.INTERO_PRED_DIM)
            intero_now = torch.as_tensor(
                [controllable_intero_vector(ctx.homeostasis, intero_dim)],
                device=z0.device,
                dtype=z0.dtype,
            )
            if (
                bundle.prev_state is not None
                and bundle.prev_motor is not None
                and bundle.prev_intero is not None
            ):
                pred_prev_i = bundle.stack.forward_predict_intero(
                    bundle.prev_state, bundle.prev_motor, bundle.prev_intero, detach_params=False
                )
                l_fwd_intero = torch.nn.functional.mse_loss(pred_prev_i, intero_now.detach())
            s_pref_i = torch.as_tensor(
                [preferred_intero_vector(intero_dim)], device=z0.device, dtype=z0.dtype
            )
            w_pref_i = torch.as_tensor(
                [intero_preference_weights(intero_dim)], device=z0.device, dtype=z0.dtype
            )
            pred_pref_i = bundle.stack.forward_predict_intero(
                z5_t, motor_u, intero_now, detach_params=True
            )
            l_pref_intero = (w_pref_i * (pred_pref_i - s_pref_i).pow(2)).mean()
            # Severity-weighted priority: the more depleted the reservoirs, the more
            # the policy prioritizes drive reduction over its other objectives
            # ("your priorities change when starving"). The multiplier is convex and
            # compounding, mirroring the deprivation-pain curve, and detached so it
            # reweights the loss without itself receiving gradient.
            deficit_i = (s_pref_i - intero_now).clamp(min=0.0)
            severity = float(deficit_i.pow(2).sum().detach().item())
            # Live curriculum overrides reweight (never extend) this objective;
            # None -> the process-env default (exact parity).
            pref_w_base = (
                ctx.ai_intero_pref_weight
                if ctx.ai_intero_pref_weight is not None
                else C.ai_intero_pref_weight()
            )
            drive_gain = (
                ctx.drive_priority_gain
                if ctx.drive_priority_gain is not None
                else C.drive_priority_gain()
            )
            pref_w = pref_w_base * (1.0 + drive_gain * severity)
            loss = (
                loss
                + C.ai_intero_fwd_weight() * l_fwd_intero
                + pref_w * l_pref_intero
            )

            # --- Successor-features value shaping (Layer-2 incentive salience) ---
            # psi(z5, motor_u) predicts the discounted FUTURE reservoir change this
            # action leads to; composed with the innate, deficit-gated drive weights
            # it is a scalar value v. Pulling the policy toward higher v makes
            # APPROACHING a seen resource rewarding BEFORE consumption -- the cue
            # inherits value from the relief it predicts. SF weights are detached
            # (the policy can't inflate its own value), and the weight ramps from 0,
            # so a naive agent is byte-identical to today until experience grows it.
            if C.sf_enabled() and getattr(bundle.stack, "has_successor_model", False):
                sf_value_w = C.sf_value_weight_for_cycle(int(ctx.state_bus.cycle_index))
                if sf_value_w > 0.0:
                    psi = bundle.stack.successor_predict(z5_t, motor_u, detach_params=True)
                    w_gated = (w_pref_i * deficit_i).detach()
                    value = (w_gated * psi).sum()
                    if torch.isfinite(value):
                        loss = loss - sf_value_w * value  # maximize value
                        sf_value_last = float(value.detach().item())

    # --- Cognitive trace: capture the raw "why" arrays from the action-producing
    # (pre-optimizer-step) weights. Read-only and gated, so the oracle/no-trace
    # path is byte-identical. The only added compute is two tiny frozen forward
    # passes under no_grad (action vs. standing still).
    cog_raw: dict | None = None
    trace_on = ctx.cognition_trace if ctx.cognition_trace is not None else C.cognition_trace_enabled()
    if has_body and trace_on:
        try:
            cog_raw = cognition_trace.collect_cognition_inputs(
                bundle=bundle,
                z5=z5_t,
                motor_u=motor_u,
                ctx=ctx,
                fwd_dim=fwd_dim,
                drive_on=drive_on,
                s_target=s_target,
            )
        except Exception:  # interpretability must never break the cycle
            cog_raw = None
        # Gated input attribution: sample d|motor_u|/d(inputs) every N cycles. The
        # extra forward snapshots/restores the recurrent state, so it is read-only.
        interval = C.cognition_attribution_interval()
        if cog_raw is not None and interval and (int(ctx.state_bus.cycle_index) % interval == 0):
            try:
                cog_raw["attribution"] = cognition_trace.attribution_pass(
                    bundle=bundle,
                    z0=z0,
                    ep=ep,
                    mem_t=mem_t,
                    wm=getattr(ctx.perceptual, "working_memory", None),
                )
            except Exception:
                pass

    # --- Discovered perception: self-supervised slot recon + data association +
    # agency (self-vs-other) learning. All gated by discovered mode + a real
    # camera frame, so the oracle path is byte-identical.
    l_slot = torch.zeros((), device=z0.device, dtype=z0.dtype)
    l_agency = torch.zeros((), device=z0.device, dtype=z0.dtype)
    discovery_diag: dict = {}
    agency_scores: dict[str, float] = {}
    if slot_out is not None and patch_tokens is not None:
        from decadic.cycle.discovery import extract_proposals

        # DINOSAUR-style self-supervision: slots must reconstruct the (frozen)
        # patch-feature map. No pixel labels, no oracle.
        l_slot = torch.nn.functional.mse_loss(slot_out["recon"], patch_tokens.detach())
        loss = loss + C.slot_recon_weight() * l_slot

        with torch.no_grad():
            slots_np = slot_out["slots"][0].detach().float().cpu().numpy()
            presence_np = slot_out["presence"][0].detach().float().cpu().numpy()
            # Use slot-attention routing for object localization. Decoder alpha
            # masks can start uniform, which collapses every centroid to frame
            # center and poisons object memory.
            centroids = bundle.stack.slots_module.centroids(slot_out["attn"])
            centroids_np = centroids[0].detach().float().cpu().numpy()
            mask_entropies = _slot_mask_entropies(slot_out["attn"])
        div_loss, ent_loss, sep_loss = _slot_object_losses(slot_out["attn"], centroids)
        loss = loss + C.slot_diversity_weight() * div_loss
        loss = loss + C.slot_entropy_weight() * ent_loss
        loss = loss + C.slot_spatial_separation_weight() * sep_loss
        proposals = extract_proposals(
            slots_np, presence_np, centroids_np, threshold=C.slot_presence_threshold()
        )
        for p in proposals:
            idx = int(p.get("idx", -1))
            if 0 <= idx < len(mask_entropies):
                p["mask_entropy"] = mask_entropies[idx]
        organ = getattr(bundle, "_perception_organ", None)
        if organ is None:
            organ = PerceptionOrgan()
            bundle._perception_organ = organ
        proposals, organ_diag, ret_map = organ.process(
            latest,
            proposals,
            prev_motor=bundle.prev_motor,
        )
        object_files = object_files_from_proposals(proposals)
        wm_disc = getattr(ctx.perceptual, "working_memory", None)
        events = (latest or {}).get("events") if isinstance(latest, dict) else None
        matched = []
        if wm_disc is not None:
            matched = wm_disc.integrate_discovered(
                [f.to_working_memory_proposal() for f in object_files],
                events=events if isinstance(events, list) else [],
                appearance_weight=C.assoc_appearance_weight(),
                match_threshold=C.assoc_match_threshold(),
                appearance_ema=C.appearance_ema(),
                reidentify=(
                    ctx.ltm_graph.match if getattr(ctx, "ltm_graph", None) is not None else None
                ),
            )
        stable_count = sum(
            1
            for s in getattr(wm_disc, "slots", {}).values()
            if int(getattr(s, "seen_count", 0)) >= C.ltm_consolidate_min_seen()
        ) if wm_disc is not None else 0
        health = evaluate_discovery_health(
            object_files,
            tracked_count=len(getattr(wm_disc, "slots", {}) or {}) if wm_disc is not None else 0,
            stable_tracked_objects=stable_count,
        )
        if hasattr(ctx.perceptual, "object_files"):
            ctx.perceptual.object_files = (
                _stable_object_file_snapshots(wm_disc)
                if wm_disc is not None
                else [f.to_dict() for f in object_files]
            )
        if hasattr(ctx.perceptual, "discovery_health"):
            ctx.perceptual.discovery_health = health.to_dict()
        if hasattr(ctx.perceptual, "ltm_consolidation"):
            ctx.perceptual.ltm_consolidation = {
                "status": "not_evaluated",
                "reason": health.reason,
            }
        if hasattr(ctx.perceptual, "perception_organ"):
            ctx.perceptual.perception_organ = organ_diag
        if hasattr(ctx.perceptual, "retinotopic_map"):
            ctx.perceptual.retinotopic_map = ret_map
        # Agency (comparator model): predict each tracked slot's realized image
        # motion from the efference copy vs an efference-blind baseline; the error
        # reduction is the per-slot "this is mine" signal.
        usable = [m for m in matched if int(m.get("idx", -1)) >= 0]
        if usable and bundle.prev_motor is not None:
            idxs = [int(m["idx"]) for m in usable]
            idx_t = torch.as_tensor(idxs, device=z0.device, dtype=torch.long)
            slots_m = slot_out["slots"][0].index_select(0, idx_t)
            n_act = bundle.cfg.n_actuators
            u_prev = bundle.prev_motor.detach().reshape(1, -1)
            if u_prev.shape[1] != n_act:
                u_prev = torch.nn.functional.pad(u_prev, (0, max(0, n_act - u_prev.shape[1])))[
                    :, :n_act
                ]
            u_m = u_prev.expand(len(idxs), -1)
            motion = [
                [m["cur_uv"][0] - m["prev_uv"][0], m["cur_uv"][1] - m["prev_uv"][1]]
                for m in usable
            ]
            target_m = torch.as_tensor(motion, device=z0.device, dtype=z0.dtype)
            pred_eff, pred_base = bundle.stack.agency(slots_m, u_m)
            l_agency = torch.nn.functional.mse_loss(
                pred_eff, target_m
            ) + torch.nn.functional.mse_loss(pred_base, target_m)
            loss = loss + C.agency_weight() * l_agency
            with torch.no_grad():
                eff_err = (pred_eff - target_m).pow(2).mean(dim=1)
                base_err = (pred_base - target_m).pow(2).mean(dim=1)
                reduction = (base_err - eff_err).detach().cpu().numpy()
            for m, r in zip(usable, reduction):
                agency_scores[str(m["entity_id"])] = float(r)
        discovery_diag = {
            "slots_present": len(proposals),
            "slot_recon_error": round(float(l_slot.detach().cpu().item()), 6),
            "slot_diversity_loss": round(float(div_loss.detach().cpu().item()), 6),
            "slot_entropy_loss": round(float(ent_loss.detach().cpu().item()), 6),
            "slot_spatial_loss": round(float(sep_loss.detach().cpu().item()), 6),
            "discovered_objects": len(getattr(wm_disc, "slots", {}) or {}),
            "object_files": health.object_files,
            "stable_tracked_objects": health.stable_tracked_objects,
            "perception_collapsed": 1.0 if health.collapsed else 0.0,
            "perception_health": health.status,
            "perception_health_reason": health.reason,
            "centroid_spread": health.centroid_spread,
            "appearance_cosine_mean": health.appearance_cosine_mean,
            "flow_confidence": health.flow_confidence,
            "looming_count": health.looming_count,
            "stuff_count": health.stuff_count,
            "body_candidate_count": health.body_candidate_count,
        }
    elif discovered:
        health = evaluate_discovery_health([], tracked_count=0, stable_tracked_objects=0)
        if hasattr(ctx.perceptual, "object_files"):
            ctx.perceptual.object_files = []
        if hasattr(ctx.perceptual, "discovery_health"):
            ctx.perceptual.discovery_health = health.to_dict()
        if hasattr(ctx.perceptual, "ltm_consolidation"):
            ctx.perceptual.ltm_consolidation = {
                "status": "not_evaluated",
                "reason": health.reason,
            }
        if hasattr(ctx.perceptual, "perception_organ"):
            ctx.perceptual.perception_organ = {
                "frame_seen": False,
                "stale_frame": False,
                "grid_size": 0,
                "flow_confidence": 0.0,
                "global_motion": 0.0,
                "local_motion_max": 0.0,
                "local_motion_mean": 0.0,
                "looming_count": 0,
                "stuff_count": 0,
                "body_candidate_count": 0,
                "foreground_count": 0,
                "checkpoint_status": "no_frame",
            }
        discovery_diag = {
            "slots_present": 0,
            "discovered_objects": 0,
            "object_files": 0,
            "stable_tracked_objects": 0,
            "perception_collapsed": 0.0,
            "perception_health": health.status,
            "perception_health_reason": health.reason,
            "centroid_spread": 0.0,
            "flow_confidence": 0.0,
            "looming_count": 0,
            "stuff_count": 0,
            "body_candidate_count": 0,
        }

    tb0 = time.perf_counter()
    bundle.optimizer.zero_grad(set_to_none=True)
    # Backstop: never apply a non-finite update. A non-finite forward pass, a
    # non-finite loss, or a non-finite grad norm all skip the step (clip_grad_norm_
    # does NOT stop NaN: a single NaN grad makes the total norm NaN, so the clip
    # coefficient is NaN and every parameter is scaled to NaN). Skipping the step
    # leaves weights finite; grads stay None, which the plasticity step already
    # tolerates (rewire falls back to random scoring, Hebbian uses cached
    # activations).
    if forward_finite and torch.isfinite(loss):
        loss.backward()
        train_params = list(bundle.stack.parameters()) + list(bundle.encoders.parameters())
        total_norm = torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        if torch.isfinite(total_norm):
            bundle.optimizer.step()
            bundle.after_optimization_step()
    if not forward_finite:
        # Recover: zero the transient recurrent buffers (the poison carrier) and
        # drop the cross-cycle transition buffers so the next forward starts clean.
        # Learned weights are untouched, so the brain survives the spike.
        bundle.stack.reset_recurrent_state()
        bundle.prev_state = None
        bundle.prev_motor = None
        bundle.prev_intero = None
        bundle.prev_affect = None
        bundle.prev_repself = None
    bwd_ms = (time.perf_counter() - tb0) * 1000.0

    def _finite(x: float, default: float = 0.0) -> float:
        return x if math.isfinite(x) else default

    pc_val = _finite(float(out["pc_loss"].detach().cpu().item()))
    l_fwd_val = _finite(float(l_fwd.detach().cpu().item()))
    tactile_pe_val = _finite(float(l_fwd_tactile.detach().cpu().item()))
    effort_pe_val = _finite(float(l_fwd_effort.detach().cpu().item()))
    intero_pe_val = _finite(float(l_fwd_intero.detach().cpu().item()))

    # Dual-network consolidation: capture the realized transition the live cycle
    # just learned from, for off-path replay by the consolidator. OFF by default
    # (zero cost). Detached CPU tensors only -> no autograd graph is pinned. Only a
    # genuine learned transition (a body streaming, with a buffered prev state+motor)
    # is stored; salience is the total prediction surprise (pc + forward + intero).
    transition_payload = None
    if (
        C.consolidation_enabled()
        and has_body
        and bundle.prev_state is not None
        and bundle.prev_motor is not None
    ):
        transition_payload = {
            "z0": z0.detach().to("cpu"),
            "ep": ep.detach().to("cpu"),
            "mem": mem_t.detach().to("cpu"),
            "prev_state": bundle.prev_state.detach().to("cpu"),
            "prev_motor": bundle.prev_motor.detach().to("cpu"),
            "proprio_target": s_target.detach().to("cpu"),
            "drive_on": bool(drive_on),
            "salience": float(pc_val + l_fwd_val + tactile_pe_val + effort_pe_val + intero_pe_val),
        }
        if has_body:
            transition_payload["effort_now"] = torch.as_tensor(
                [controllable_effort_vector(latest, int(C.EFFORT_PRED_DIM))],
                dtype=torch.float32,
            )
        if drive_on and bundle.prev_intero is not None:
            prev_i_cpu = bundle.prev_intero.detach().to("cpu")
            now_i_cpu = intero_now.detach().to("cpu")
            transition_payload["prev_intero"] = prev_i_cpu
            transition_payload["intero_now"] = now_i_cpu
            # Per-step feature phi = reservoir change this cycle (successor-features
            # basis); its innate-weighted sum is the per-step drive-reduction reward
            # r_t (positive on reservoir gain, ~0 on slow drain, spikes on consume).
            # The goal-episode accumulator spreads r_t backward as a lambda-return so
            # distal relief credits the approach that earned it. No new innate signal
            # -- this is the SAME interoceptive setpoint, just propagated over time.
            phi = (now_i_cpu - prev_i_cpu).reshape(-1)
            w_i = torch.as_tensor(
                intero_preference_weights(int(C.INTERO_PRED_DIM)), dtype=phi.dtype
            )
            transition_payload["feat"] = phi
            transition_payload["reward"] = float((w_i * phi).sum().item())

    # Perception-feedback telemetry (None when the loop is off).
    gate_mean = (
        _finite(float(gate.detach().mean().cpu().item())) if gate is not None else None
    )
    percept_pe = (
        _finite(float(l_percept.detach().cpu().item())) if z0_hat is not None else None
    )

    # Neuroplasticity post-step (A/B/C). Runs while .grad is still populated so
    # sparse rewiring can score inactive edges by gradient magnitude. No-op when
    # the stack has no plastic blocks (full parity with the dense baseline).
    plast_diag = apply_plasticity_step(
        ctx,
        bundle,
        pc_loss=pc_val,
        modulation=float(ctx.state_bus.pleasure_scalar - ctx.state_bus.pain_scalar),
    )

    # Sanitize every value pulled out of the forward pass: if the network just
    # diverged (NaN/inf), the instability guard above already froze plasticity
    # and reset the weights — here we make sure no NaN leaks into the State Bus,
    # affect, or motor command, so the next cycle starts from a finite state.
    def _san(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        z0_rms = _finite(float(z0_bu.detach().float().pow(2).mean().sqrt().cpu().item()))
        emo = _san(out["emotion"].detach().cpu().numpy())
        sm = _san(out["state_mind"].detach().cpu().numpy())
        nar = _san(out["narrative"].detach().cpu().numpy())
        meta = _san(out["metacognition"].detach().cpu().numpy())
        risk_p = _finite(torch.sigmoid(out["risk_logit"]).detach().cpu().item(), 0.5)
        direction = _san(out["direction"].detach().cpu().numpy().reshape(-1))
        speed = _finite(float(out["speed"].detach().cpu().item()))
        z5 = _san(out["z5"].detach().cpu().numpy().reshape(-1))
        motor_np = _san(motor_u.detach().cpu().numpy().reshape(-1))

    ctx.latents["z5_snapshot"] = z5.astype(float).tolist()

    # Motor babble: need-and-error-gated exploration (NOT a clock decay). It scales
    # with unmet homeostatic drive AND forward-model surprise, and is floored above
    # zero while any reservoir is below full, so a deprived agent keeps trying
    # actions until it discovers the action->relief contingency. This is what pulls
    # the policy off the dark-room fixed point; a sated, well-predicting agent
    # explores ~0 and may rest. ``l_fwd_val`` is this cycle's forward-model error.
    cycle_idx = int(ctx.state_bus.cycle_index)
    # Autonomous, need-gated curiosity (epistemic drive). OFF by default -> this
    # block is skipped and the baseline cycle is byte-identical. When enabled it
    # tracks the recent *fall* of forward-model error (learning progress) and
    # turns that, gated by survival urgency, into a pleasure that flows into B,
    # biases the priority toward "investigate", and adds exploratory drive to the
    # babble gate (so satisfied curiosity lets babble relax). The PE history is
    # ephemeral, kept on the bundle like the prev_state transition buffers (never
    # checkpointed; rebuilt on load).
    cur_out = None
    if C.curiosity_enabled():
        cur_state = getattr(bundle, "_curiosity", None)
        if cur_state is None:
            cur_state = CuriosityState(window=C.curiosity_progress_window())
            bundle._curiosity = cur_state
        cur_out = compute_curiosity(
            cur_state,
            fwd_error=(l_fwd_val if has_body else pc_val),
            pain=float(ctx.state_bus.pain_scalar),
            viability=float(ctx.viability.value),
            gain=C.curiosity_gain(),
            window=C.curiosity_progress_window(),
            safety_sharpness=C.curiosity_safety_sharpness(),
            viability_max=float(ctx.viability.max_value),
        )
    curiosity_drive = cur_out.drive if cur_out is not None else 0.0
    if has_body:
        drive_now = (
            interoceptive_drive_pain(
                ctx.homeostasis,
                comfort=C.drive_comfort_setpoint(),
                gain=C.drive_pain_gain(),
                exponent=C.drive_pain_exponent(),
            )
            if (drive_on and ctx.homeostasis is not None)
            else 0.0
        )
        babble_sigma = motor_exploration_sigma(
            drive=drive_now + curiosity_drive,
            fwd_error=l_fwd_val,
            sigma_max=ctx.motor_babble_sigma,
        )
    else:
        babble_sigma = 0.0
    if babble_sigma > 0.0:
        noise = np.random.normal(0.0, babble_sigma, size=motor_np.shape).astype(np.float32)
        u_emit = np.clip(motor_np + noise, -1.0, 1.0)
    else:
        u_emit = np.clip(motor_np, -1.0, 1.0)
    # Manual override pins the harness level; otherwise follow the fading curriculum.
    assist_gain = (
        float(ctx.assist_override)
        if ctx.assist_override is not None
        else assist_gain_for_cycle(cycle_idx)
    )
    motor_rms = float(np.sqrt(np.mean(np.square(u_emit)))) if u_emit.size else 0.0

    # Buffer the realized transition (state, executed command) for next cycle's
    # forward-model prediction-error term. Never store a non-finite state, and
    # never store anything on a firewall-recovery cycle (z5 is computed before
    # the LSTM, so it can be finite even when the cycle as a whole went NaN), so
    # a poisoned cycle can't re-poison the next one.
    if has_body:
        z5_detached = z5_t.detach()
        if forward_finite and torch.isfinite(z5_detached).all():
            bundle.prev_state = z5_detached.clone()
            bundle.prev_motor = torch.as_tensor(
                u_emit, device=z0.device, dtype=z0.dtype
            ).unsqueeze(0)
            if drive_on:
                bundle.prev_intero = torch.as_tensor(
                    [controllable_intero_vector(ctx.homeostasis, int(C.INTERO_PRED_DIM))],
                    device=z0.device,
                    dtype=z0.dtype,
                )
            else:
                bundle.prev_intero = None
        else:
            bundle.prev_state = None
            bundle.prev_motor = None
            bundle.prev_intero = None

    _np_assign(ctx.state_bus.emotion_physio, emo)
    _np_assign(ctx.state_bus.state_of_mind, sm)
    _np_assign(ctx.state_bus.narrative_emb, nar)
    _np_assign(ctx.state_bus.metacognition, meta)

    # --- Global workspace: competition + ignition (Phase 2) OR the legacy EMA ----
    # Off-branch (default): the salience-weighted working-memory summary is
    # EMA-blended into A -- byte-identical to before. On-branch (gwt_enabled):
    # working-memory coalitions compete; only a dominant coalition (>= ignition
    # threshold of the salience mass) ignites and is globally broadcast (blended
    # into A here, fed back via the spine below, boosts the episodic salience in
    # stage 10, and is described by the narrative). Below threshold => no ignition
    # => A holds its prior (nothing reaches global broadcast).
    wm = getattr(ctx.perceptual, "working_memory", None)
    wm_slots = 0
    gwt_on = ctx.gwt_enabled if ctx.gwt_enabled is not None else C.gwt_enabled()
    workspace_block: dict | None = None
    if wm is not None:
        # Persist the pooled percept as the scene latent (EMA across cycles).
        if hasattr(wm, "deposit_scene"):
            wm.deposit_scene(fused.detach().cpu().reshape(-1).tolist())
        wm_slots = len(getattr(wm, "slots", {}) or {})
        dim = ctx.state_bus.state_of_mind.shape[0]
        beta = float(os.environ.get("DECADIC_WM_ATTENTION_WEIGHT", "0.25"))
        if gwt_on:
            vecs, sal = wm.workspace_candidates(dim)
            slots_arr = (
                np.asarray(vecs, dtype=np.float32) if vecs else np.zeros((0, dim), np.float32)
            )
            ign = GlobalWorkspace(
                threshold=C.gwt_ignition_threshold(),
                capacity=C.gwt_capacity(),
                temperature=C.gwt_temperature(),
            ).ignite(slots_arr, np.asarray(sal, dtype=np.float32))
            if ign.ignited:
                a_vec = ctx.state_bus.state_of_mind
                a_vec[:] = (1.0 - beta) * a_vec + beta * ign.content.astype(np.float32)
            workspace_block = {
                "enabled": True,
                "ignited": bool(ign.ignited),
                "share": round(float(ign.score), 4),
                "threshold": round(float(C.gwt_ignition_threshold()), 4),
                "n_candidates": int(len(vecs)),
                "winners": list(ign.winners),
            }
            ctx.latents["workspace"] = workspace_block
        elif wm_slots or getattr(wm, "scene_latent", None):
            attn = np.asarray(wm.attention_vector(dim), dtype=np.float32)
            a_vec = ctx.state_bus.state_of_mind
            a_vec[:] = (1.0 - beta) * a_vec + beta * attn

    # Self-state feedback spine: carry this cycle's self-report (A||C||E) forward
    # so the next cycle's stack is conditioned on its own prior state. Detached
    # (no cross-cycle BPTT); zeroed on a non-finite cycle so a poisoned report
    # can't re-poison the loop. No-op (prev_self stays None) when the faculty is
    # off. The continuity readout (cosine vs the vector this cycle was fed) makes
    # the narrative report *of* the fed-back content rather than a dead end.
    self_model_block: dict | None = None
    if bundle.stack.has_self_model_feedback:
        # When the workspace ignited, the reported A reflects the broadcast
        # content, so the spine feeds THAT back (reportable == broadcast == fed
        # back). Otherwise the raw state-of-mind head is fed (Phase-1 behavior,
        # byte-identical off-branch).
        workspace_fed = bool(
            gwt_on and workspace_block is not None and workspace_block.get("ignited")
        )
        if workspace_fed:
            a_src = torch.as_tensor(
                ctx.state_bus.state_of_mind,
                device=out["state_mind"].device,
                dtype=out["state_mind"].dtype,
            ).reshape(1, -1)
            sv = torch.cat([a_src, out["narrative"], out["metacognition"]], dim=-1).detach()
        else:
            sv = self_state_vector(out)
        continuity = None
        if self_prev_fed is not None:
            with torch.no_grad():
                continuity = _finite(
                    float(
                        torch.nn.functional.cosine_similarity(
                            self_prev_fed.reshape(1, -1), sv.reshape(1, -1)
                        ).item()
                    )
                )
        if forward_finite and bool(torch.isfinite(sv).all()):
            bundle.prev_self = sv.clone()
        else:
            bundle.prev_self = None
        self_model_block = {
            "active": True,
            "continuity": continuity,
            "workspace_fed": workspace_fed,
        }

    # Finalize discovered perception: fold per-slot agency into the object files
    # (promoting persistent high-agency slots to body parts) and rebuild the
    # egocentric graph from the freshly-associated object files.
    if discovered and wm is not None:
        if agency_scores:
            # Touch cross-check: a strong contact this frame corroborates agency
            # for slots already commanded by efference (feel + command == mine).
            contacts = (latest or {}).get("proprioception", {}).get("contacts") if isinstance(latest, dict) else None
            touch_active = bool(
                isinstance(contacts, list)
                and contacts
                and max((abs(float(c)) for c in contacts), default=0.0)
                > float(os.environ.get("DECADIC_AGENCY_TOUCH_N", "50"))
            )
            wm.update_agency(
                agency_scores,
                ema=C.agency_ema(),
                threshold=C.agency_threshold(),
                min_seen=C.agency_min_seen(),
                touch_active=touch_active,
            )
        if hasattr(ctx.perceptual, "rebuild_discovered_graph"):
            ctx.perceptual.rebuild_discovered_graph(latest)
        live = list((getattr(wm, "slots", {}) or {}).values())
        agency_vals = [float(s.agency) for s in live if getattr(s, "agency_seen", 0) > 0]
        discovery_diag["self_parts"] = sum(
            1 for s in live if getattr(s, "kind", None) == "self_part"
        )
        discovery_diag["agency_mean"] = (
            round(float(np.mean(agency_vals)), 6) if agency_vals else 0.0
        )
        discovery_diag["agency_loss"] = round(float(l_agency.detach().cpu().item()), 6)

    ctx.state_bus.priority_scalar = float(risk_p)
    ctx.state_bus.priority_label = "avoid" if risk_p < 0.42 else "explore"
    # Curiosity bends the (non-avoiding) priority toward active investigation when
    # there is safe, unexhausted learning to be had. It never overrides "avoid":
    # survival gating already collapses the epistemic value to ~0 under threat.
    if cur_out is not None and cur_out.investigate and ctx.state_bus.priority_label == "explore":
        ctx.state_bus.priority_label = "investigate"

    # --- Represented self (Phase 5): self as a modelled object -----------------
    # Assemble the agent's interoception/affect/capability into a compact self and
    # (a) write it as content onto the egocentric self-node + bind "controls" edges
    # to its learned body parts (observable; off by default), and (b) feed the
    # self-node embedding back through the zero-init spine ingress so the modelled
    # self conditions the next cycle. No-op + parity when the faculty is off.
    if getattr(bundle.stack, "has_represented_self", False):
        from decadic.state.self_model import build_represented_self

        rs = build_represented_self(
            viability=ctx.viability.value,
            homeostasis=ctx.homeostasis,
            pain=ctx.state_bus.pain_scalar,
            pleasure=ctx.state_bus.pleasure_scalar,
            priority=ctx.state_bus.priority_scalar,
            working_memory=wm,
        )
        rv = torch.as_tensor(rs.embedding(), device=z0.device, dtype=z0.dtype).reshape(1, -1)
        bundle.prev_repself = rv if (forward_finite and bool(torch.isfinite(rv).all())) else None
        nodes = getattr(ctx.perceptual, "egocentric_nodes", None)
        edges = getattr(ctx.perceptual, "egocentric_edges", None)
        if isinstance(nodes, list):
            self_node = next((n for n in nodes if n.get("role") == "self"), None)
            if self_node is not None:
                self_node["self_model"] = rs.node_content()
                if isinstance(edges, list):
                    ent_nodes = [n for n in nodes if n.get("role") == "entity"]
                    edges.extend(rs.semantic_edges(str(self_node.get("id", "self")), ent_nodes))
        if self_model_block is None:
            self_model_block = {"active": True}
        self_model_block["represented"] = rs.node_content()

    prop = ctx.perceptual.proprio_position or [0.0, 0.0, 0.0]
    # Motor action: per-actuator PD targets are the real efferent output. The
    # legacy direction/speed ride along to drive the fading assist harness.
    ctx.latents["action"] = {
        "type": "motor",
        "parameters": {
            "ctrl": [round(float(x), 5) for x in u_emit.tolist()],
            # Per-joint forward-model error -> the body's joint-brace ROM curriculum.
            "joint_pe": [round(float(x), 6) for x in joint_pred_error],
            "assist_gain": round(float(assist_gain), 5),
            "direction": direction.tolist(),
            "speed": speed,
            "risk": risk_p,
            "babble_sigma": round(float(babble_sigma), 5),
        },
    }
    # Only forward the support-system selection when the operator has set one;
    # otherwise the body keeps its env default (no per-cycle override).
    if ctx.curriculum_mode is not None:
        ctx.latents["action"]["parameters"]["curriculum_mode"] = ctx.curriculum_mode
    z5_head = z5[:16] if z5.shape[0] >= 16 else np.pad(z5, (0, 16 - z5.shape[0]))
    ctx.latents["predicted_outcome"] = {
        "embedding": z5_head.astype(float).tolist(),
        "expected_position": list(prop),
    }

    pc_parts = out["pc_parts"]
    names = {
        1: "sensory_perception",
        2: "experience_framing",
        3: "heuristic_memory",
        4: "risk_utility",
        5: "pre_normative",
        6: "emotion_physio",
        7: "reprioritize",
        8: "strategy",
        9: "behavioral_response",
    }
    # pc_i_j measures how badly stage i's latent predicted stage j's; attribute
    # the surprise to the *predicted* stage (j shifted onto Decadic numbering).
    pc_by_stage = {
        3: pc_parts.get("pc_1_2"),
        4: pc_parts.get("pc_2_3"),
        5: pc_parts.get("pc_3_4"),
        6: pc_parts.get("pc_4_5"),
    }
    stage_metrics = {int(m["stage"]): m for m in out.get("stage_metrics", [])}
    for i in range(1, 10):
        payload: dict = {"neural": True}
        if i == 1:
            payload["timing_ms"] = round(enc_ms, 4)
            payload["activity"] = round(z0_rms, 6)
        else:
            m = stage_metrics.get(i, {})
            ms = float(m.get("timing_ms", 0.0))
            if i == 3:
                ms += mem_ms
            payload["timing_ms"] = round(ms, 4)
            if "activity" in m:
                payload["activity"] = m["activity"]
            if "activations" in m:
                payload["activations"] = m["activations"]
        if pc_by_stage.get(i) is not None:
            payload["pc_part"] = round(float(pc_by_stage[i]), 6)
        traces.append(StageTrace(stage=i, name=names[i], payload=payload))
    ctx.latents["stage_traces"] = traces

    pe_stub = stub_prediction_error_penalty(
        ctx.perceptual.integration_ticks, ctx.state_bus.cycle_index
    )
    # The genuine surprise term is the predictive-coding loss; the cycle-counter
    # oscillation (pe_stub) is a Phase-1 placeholder blended in only at
    # pe_stub_weight() (default 0.0 -> removed in production; 0.25 in tests).
    pe_delta = -viability_pe_scale() * pc_val + C.pe_stub_weight() * pe_stub

    # Prediction error no longer drains viability (the homeostatic reservoirs own
    # survival). It still produces affect, and feeds the metabolic stress term.
    p_pe, pl_pe = viability_delta_to_signals(pe_delta)
    ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
        ctx.state_bus.emotion_physio, p_pe, pl_pe
    )

    # Homeostatic relief reward: phasic pleasure proportional to the per-cycle
    # *reduction* in interoceptive drive pressure (drive-reduction reward; the
    # positive complement to the tonic deprivation pain applied below). Grounded
    # solely in the agent's own reservoirs vs. innate setpoints -- no external
    # satisfier, no clock. When disabled (or no body streams reservoirs) the
    # legacy periodic placeholder is retained so the test baseline is unchanged.
    reward_drive_on = drive_on and C.drive_reward_enabled()
    if reward_drive_on:
        cur_drive_pressure = interoceptive_drive_pain(
            ctx.homeostasis,
            comfort=C.drive_comfort_setpoint(),
            gain=C.drive_pain_gain(),
            exponent=C.drive_pain_exponent(),
        )
        reward_delta = drive_reduction_reward(
            ctx.state_bus.prev_drive_pressure,
            cur_drive_pressure,
            gain=C.drive_reward_gain(),
        )
        ctx.state_bus.prev_drive_pressure = cur_drive_pressure
    else:
        reward_delta = reward_success_stub(ctx.state_bus.cycle_index)
    p_rw, pl_rw = viability_delta_to_signals(reward_delta)
    ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
        ctx.state_bus.emotion_physio, p_rw, pl_rw
    )

    ctx.state_bus.pain_scalar = ema_affect(ctx.state_bus.pain_scalar, p_pe + p_rw)
    ctx.state_bus.pleasure_scalar = ema_affect(ctx.state_bus.pleasure_scalar, pl_pe + pl_rw)
    # Floor the felt pleasure at a genuine relief so it registers rather than
    # washing out in the slow pleasure EMA (mirrors the curiosity floor below).
    # Phasic and gated: applied only on cycles where the drive actually dropped
    # and only when enabled, so the disabled/legacy path stays byte-identical.
    if reward_drive_on and pl_rw > 0.0:
        ctx.state_bus.pleasure_scalar = max(
            ctx.state_bus.pleasure_scalar, min(1.0, pl_rw)
        )

    # Curiosity: a survival-gated, pleasure-side epistemic affect. Mirror the innate
    # drive-pain pattern on the pleasure axis -- inject into B and floor the felt
    # pleasure at the current (clamped) curiosity level, so sustained safe learning
    # stays felt rather than washing out in the slow pleasure EMA above.
    if cur_out is not None and cur_out.pleasure > 0.0:
        ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
            ctx.state_bus.emotion_physio, 0.0, cur_out.pleasure
        )
        ctx.state_bus.pleasure_scalar = max(
            ctx.state_bus.pleasure_scalar, min(1.0, cur_out.pleasure)
        )

    # Innate thirst/hunger valence: tonic pain in proportion to how far a reservoir
    # sits below its comfort setpoint (drive theory). It names no external object -
    # it is the bare "this state is bad" prior that phylogeny fixes. We inject it
    # into the B affect channel and floor the felt pain so persistent deprivation
    # stays felt (bounded in [0,1]) without integrating to saturation like the
    # phasic PE/reward pains above.
    drive_pain = 0.0
    if drive_on:
        drive_pain = interoceptive_drive_pain(
            ctx.homeostasis,
            comfort=C.drive_comfort_setpoint(),
            gain=C.drive_pain_gain(),
            exponent=C.drive_pain_exponent(),
        )
        if drive_pain > 0.0:
            ctx.state_bus.emotion_physio = apply_pain_pleasure_to_B(
                ctx.state_bus.emotion_physio, drive_pain, 0.0
            )
            ctx.state_bus.pain_scalar = max(ctx.state_bus.pain_scalar, drive_pain)

    t10 = time.perf_counter()
    tr10 = stage_10.run(ctx)
    ms10 = (time.perf_counter() - t10) * 1000.0
    tr10.payload["timing_ms"] = round(ms10, 4)
    traces.append(tr10)

    action = ctx.latents["action"]
    predicted = ctx.latents["predicted_outcome"]
    ctx.state_bus.action_history.append({"cycle": ctx.state_bus.cycle_index, "action": action})

    gpu_mem = 0
    if bundle.device.type == "cuda":
        gpu_mem = int(torch.cuda.max_memory_allocated(bundle.device))

    diagnostics = {
        "prediction_error_delta": pe_delta,
        "drive_reward_delta": reward_delta,
        # Back-compat aliases (legacy metric names; identical values).
        "stub_prediction_error_delta": pe_delta,
        "stub_reward_delta": reward_delta,
        "stage_timing_ms_total": round(fwd_ms + bwd_ms + ms10, 4),
        "viability_value": ctx.viability.value,
        "salience_hint": float(tr10.payload.get("salience", 0.0)),
        "neural_pc_loss": pc_val,
        "neural_pc_parts": pc_parts,
        "neural_forward_ms": round(fwd_ms, 4),
        "neural_backward_ms": round(bwd_ms, 4),
        "learning_rate": float(bundle.optimizer.param_groups[0]["lr"]),
        "gpu_memory_max_allocated": gpu_mem,
        "parallel_sessions": parallel_sessions,
        "encode_phase_ms": round(encode_phase_ms, 4),
        # Per-section cycle split surfaced for DECADIC_CYCLE_PROFILE (measure-first):
        # encoders = parallel-session encode + latest sensory encode/slot attention;
        # stage10 includes the per-cycle episodic SQLite write.
        "encoders_ms": round(enc_ms + encode_phase_ms, 4),
        "memory_recall_ms": round(mem_ms, 4),
        "stage10_ms": round(ms10, 4),
        "working_memory_slots": wm_slots,
        "forward_model_error": round(l_fwd_val, 6),
        "joint_pred_error": ([round(float(x), 6) for x in joint_pred_error] if has_body else None),
        "tactile_pred_error": (round(tactile_pe_val, 6) if has_body else None),
        "effort_pred_error": (round(effort_pe_val, 6) if has_body else None),
        "assist_gain": round(float(assist_gain), 5),
        "motor_babble_sigma": round(float(babble_sigma), 5),
        "motor_activity_rms": round(motor_rms, 5),
        "motor_command": [round(float(x), 4) for x in u_emit.tolist()],
        "perception_feedback": bool(feedback_on),
        "precision_gate_mean": (round(gate_mean, 6) if gate_mean is not None else None),
        "perceptual_pred_error": (round(percept_pe, 6) if percept_pe is not None else None),
        "homeostatic_drive": bool(drive_on),
        "intero_drive": (round(drive_pain, 6) if drive_on else None),
        "intero_pred_error": (round(intero_pe_val, 6) if drive_on else None),
        # Successor-features value of the chosen action and the active (ramped)
        # value-shaping weight (None until the homeostatic drive is on).
        "sf_value": (round(sf_value_last, 6) if drive_on else None),
        "sf_value_weight": (round(float(sf_value_w), 6) if drive_on else None),
        # Need-gated curiosity (None when the drive is disabled -> parity).
        "curiosity_drive": (round(float(cur_out.drive), 6) if cur_out is not None else None),
        "curiosity_pleasure": (round(float(cur_out.pleasure), 6) if cur_out is not None else None),
        "curiosity_learning_progress": (
            round(float(cur_out.learning_progress), 6) if cur_out is not None else None
        ),
        "perception_mode": ctx.perception_mode,
        "nan_recovery": (not forward_finite),
    }
    diagnostics.update(plast_diag)
    if discovered:
        diagnostics["discovered_perception"] = True
        diagnostics.update(discovery_diag)

    # --- Cognitive trace: assemble the human-readable "why" record (read-only;
    # narrative is filled by the narrative layer). Defensive: never break the
    # cycle if interpretability assembly fails.
    cognitive = None
    if trace_on:
        # Eval-only probe capture + read-only probe read-out (both no-ops unless
        # their flags/paths are set; supervision never enters cognition).
        try:
            interp_probes.maybe_capture(ctx, latest)
        except Exception:
            pass
        try:
            probes_block = interp_probes.readout_for_cycle(ctx)
        except Exception:
            probes_block = None
        try:
            cognitive = cognition_trace.build(
                cycle=int(ctx.state_bus.cycle_index),
                raw=cog_raw,
                fwd_dim=fwd_dim,
                affect={
                    "pain": round(float(ctx.state_bus.pain_scalar), 4),
                    "pleasure": round(float(ctx.state_bus.pleasure_scalar), 4),
                    "risk": round(float(risk_p), 4),
                    "priority": ctx.state_bus.priority_label,
                },
                episodic=ctx.episodic,
                qv=qv,
                probes=probes_block,
                self_model=self_model_block,
                workspace=workspace_block,
            ).to_dict()
        except Exception:
            cognitive = None
        # Tier C: render prose from the structured record (read-out only).
        if cognitive is not None:
            try:
                text = cognition_narrative.render(cognitive, C.narrative_mode())
                cognitive["narrative"] = text
                ctx.state_bus.narrative_text_stub = text
            except Exception:
                pass

    return {
        "timestamp": _utc_ts(),
        "action": action,
        "predicted_outcome": predicted,
        "trace": [asdict(t) for t in traces],
        "_diagnostics": diagnostics,
        "_cognitive": cognitive,
        "_transition": transition_payload,
    }
