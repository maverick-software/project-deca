"""Trainable Decadic stages (Phase 2) — fusion, risk, narrative GRU/LSTM, policy."""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from decadic import config as _cfg
from decadic.nn.config import NeuralArchitectureConfig
from decadic.nn.frozen_encoders import (
    CLIP_N_PATCHES,
    CLIP_PATCH_DIM,
    CLIP_POOL_DIM,
    WHISPER_POOL_DIM,
)
from decadic.nn.faculties import CognitionFaculties
from decadic.nn.plastic import PlasticityFlags, PlasticSparseGrowableMLP


def _stage_timer(device: torch.device):
    """perf_counter clock; DECADIC_STAGE_TIMING=sync adds per-block CUDA syncs.

    Without sync, CUDA timings are kernel *dispatch* times (kernels run async),
    which is adequate for visualization at negligible cost.
    """
    sync = (
        device.type == "cuda"
        and os.environ.get("DECADIC_STAGE_TIMING", "").strip().lower() == "sync"
    )

    def now() -> float:
        if sync:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    return now


class NeuralCognitiveStack(nn.Module):
    """Differentiable cognitive pipeline blocks + predictive-coding heads."""

    # Marker: this build supports the self-state feedback spine (self-model
    # program, Phase 1). The spine itself is gated by the faculty flag below; this
    # attribute just lets the falsification harness detect that the capability
    # exists at all (vs an older build that predates the feature).
    _supports_self_model_feedback = True

    def __init__(
        self,
        cfg: NeuralArchitectureConfig,
        flags: PlasticityFlags | None = None,
        faculties: CognitionFaculties | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # Neuroplasticity flags (A/B/C). When none are enabled the stack is built
        # with plain nn.Sequential MLPs, byte-identical to the dense baseline.
        self.plasticity = flags or PlasticityFlags()
        # Core cognitive faculties (perception-feedback loop, perception mode).
        # Threaded per-agent rather than read from the process env so the dashboard
        # can build different agents with different faculties.
        self.faculties = faculties or CognitionFaculties.from_env()
        self._plastic_names: list[str] = []
        fused_in = CLIP_POOL_DIM + WHISPER_POOL_DIM + cfg.proprio_emb
        self.ingress = nn.Linear(fused_in, cfg.d_model)
        # Top-down predictive-perception loop. When on, a learned prediction of z0
        # (from detached history) is blended with the bottom-up encode under a
        # learned precision gate. When off the gate is a no-op so the state_dict is
        # identical to the dense baseline (parity).
        self.has_perception_feedback = self.faculties.perception_feedback
        if self.has_perception_feedback:
            # context = prev z5 (d) + lstm_h + recalled memory + compressed scene (d)
            self._pf_ctx_dim = cfg.d_model + cfg.lstm_hidden + cfg.memory_context_dim + cfg.d_model
            self.top_down = nn.Sequential(
                nn.Linear(self._pf_ctx_dim, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            # Gate also sees interoception (pain, pleasure, viability) so it can
            # *learn* to trust the senses under threat (not a hardcoded schedule).
            self.precision_gate = nn.Linear(self._pf_ctx_dim + 3, cfg.d_model)
            with torch.no_grad():
                # Safe starting prior (learning is free to move both): top_down
                # predicts ~nothing, and the gate opens to ~1 so z0_eff == z0.
                self.top_down[-1].weight.zero_()
                self.top_down[-1].bias.zero_()
                self.precision_gate.weight.zero_()
                self.precision_gate.bias.fill_(float(_cfg.precision_gate_init()))
        if self.plasticity.any_enabled:
            self.stage1 = self._plastic_block(
                "stage1", cfg.d_model, cfg.d_model, cfg.d_model, pre_layernorm=True
            )
        else:
            self.stage1 = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.stage2 = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer_layers)
        self.epi_proj = nn.Linear(4, cfg.d_model)
        self.mem_proj = nn.Linear(cfg.memory_context_dim, cfg.d_model)
        if self.plasticity.any_enabled:
            self.stage3 = self._plastic_block(
                "stage3", cfg.d_model * 3, cfg.d_model, cfg.d_model
            )
            self.risk_mlp = self._plastic_block(
                "risk_mlp", cfg.d_model, cfg.d_model, cfg.risk_hidden
            )
        else:
            self.stage3 = nn.Sequential(
                nn.Linear(cfg.d_model * 3, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.risk_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.risk_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.risk_hidden, cfg.d_model),
            )
        self.risk_scalar = nn.Linear(cfg.d_model, 1)
        narr_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.stage5_enc = nn.TransformerEncoder(narr_layer, num_layers=cfg.encoder_decoder_layers)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        dec_layers = max(1, cfg.encoder_decoder_layers // 2)
        self.stage5_dec = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.gru_cell = nn.GRUCell(cfg.d_model * 2, cfg.gru_hidden)
        self.emotion_head = nn.Linear(cfg.gru_hidden, cfg.emotion_out)
        lstm_in = cfg.d_model + cfg.emotion_out
        self.lstm_cell = nn.LSTMCell(lstm_in, cfg.lstm_hidden)
        self.state_mind_head = nn.Linear(cfg.lstm_hidden, cfg.state_mind_out)
        self.narrative_head = nn.Linear(cfg.d_model, cfg.narrative_out)
        self.metacog_head = nn.Linear(cfg.d_model, cfg.metacog_out)
        pol_in = cfg.lstm_hidden + cfg.state_mind_out
        self.policy = nn.Sequential(
            nn.Linear(pol_in, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 4),
        )
        # Motor head: one normalized PD target per actuator (equilibrium-point
        # control). The body tracks these targets with a fast PD loop at physics
        # rate, so the brain only sets postural targets at the cognitive rate.
        if self.plasticity.any_enabled:
            self.motor = self._plastic_block(
                "motor", pol_in, cfg.n_actuators, cfg.motor_hidden
            )
        else:
            self.motor = nn.Sequential(
                nn.Linear(pol_in, cfg.motor_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.motor_hidden, cfg.n_actuators),
            )
        # Proprioceptive forward model: (state latent, efference copy) -> next
        # controllable-proprio vector. Explicit layers so the policy can predict
        # through it with detached parameters (active inference) while the model
        # is trained only on realized transitions.
        self.fwd_l1 = nn.Linear(cfg.d_model + cfg.n_actuators, cfg.motor_hidden)
        self.fwd_l2 = nn.Linear(cfg.motor_hidden, cfg.forward_pred_dim)
        # Interoceptive forward model (homeostatic drive reduction): predicts the
        # next reservoir vector from (state, efference copy, current reservoirs).
        # Always built: survival via drive reduction is the root motivation.
        # Same detach-params trick as the proprio model lets the policy plan
        # through a frozen world model.
        self.has_intero_model = True
        self._intero_dim = int(_cfg.INTERO_PRED_DIM)
        self.fwd_intero_l1 = nn.Linear(
            cfg.d_model + cfg.n_actuators + self._intero_dim, cfg.motor_hidden
        )
        self.fwd_intero_l2 = nn.Linear(cfg.motor_hidden, self._intero_dim)
        # Tactile forward model (full-body touch): (state latent, efference copy)
        # -> next soft per-part contact load. Mirrors the proprio head (no
        # current-state conditioning needed); always built so the brain forms
        # tactile expectations and learns from prediction error which actions
        # load which body part -- the per-limb credit-assignment signal. Touch
        # has no innate setpoint, so this head is trained PE-only.
        self.has_tactile_model = True
        self._tactile_dim = int(_cfg.TACTILE_PRED_DIM)
        self.fwd_tactile_l1 = nn.Linear(cfg.d_model + cfg.n_actuators, cfg.motor_hidden)
        self.fwd_tactile_l2 = nn.Linear(cfg.motor_hidden, self._tactile_dim)
        # Effort/body-state forward model: predicts localized effort, work,
        # strain, fatigue, pain, and aggregate effort totals from (state, action).
        self.has_effort_model = True
        self._effort_dim = int(_cfg.EFFORT_PRED_DIM)
        self.fwd_effort_l1 = nn.Linear(cfg.d_model + cfg.n_actuators, cfg.motor_hidden)
        self.fwd_effort_l2 = nn.Linear(cfg.motor_hidden, self._effort_dim)
        # Successor-features head (long-horizon value / incentive salience):
        # psi(state, command) -> discounted future reservoir-change features. Built
        # unconditionally like the other world-model heads, but its OUTPUT layer is
        # zero-initialized and it is NEVER called from forward(), so the stack is
        # byte-identical until the consolidator's TD(lambda) loss grows it. The
        # substantive code lives in decadic/nn/successor_features.py (FORBIDDEN #3).
        from decadic.nn.successor_features import SuccessorFeaturesHead

        self.has_successor_model = True
        self._sf_dim = int(_cfg.INTERO_PRED_DIM)
        self.sf_head = SuccessorFeaturesHead(
            state_dim=cfg.d_model,
            action_dim=cfg.n_actuators,
            feat_dim=self._sf_dim,
            hidden=cfg.motor_hidden,
        )
        # Object-centric perception (discovered mode): slot attention over the
        # egocentric patch-feature map + a learned agency head. Built only in
        # discovered mode, so the oracle-mode state_dict is byte-identical to the
        # baseline. Object structure is injected *additively* into z0 via a
        # zero-initialized projection, so the stack starts at exact bottom-up
        # parity and learning is free to grow the object pathway.
        self.has_slots = self.faculties.discovered
        if self.has_slots:
            from decadic.nn.agency import AgencyHead
            from decadic.nn.scene_dynamics import (
                SCENE_DYNAMICS_FEATURE_DIM,
                SceneDynamicsHead,
            )
            from decadic.nn.slots import SlotAttention

            self._slots_k = _cfg.slots_k()
            self._slot_dim = _cfg.slot_dim()
            self.slots_module = SlotAttention(
                in_dim=CLIP_PATCH_DIM,
                n_patches=CLIP_N_PATCHES,
                k=self._slots_k,
                slot_dim=self._slot_dim,
                iters=_cfg.slot_iters(),
            )
            self.slot_ingress = nn.Linear(self._slots_k * self._slot_dim, cfg.d_model)
            with torch.no_grad():
                self.slot_ingress.weight.zero_()
                self.slot_ingress.bias.zero_()
            self.agency = AgencyHead(slot_dim=self._slot_dim, n_actuators=cfg.n_actuators)
            self.has_scene_dynamics = _cfg.scene_dynamics_enabled()
            if self.has_scene_dynamics:
                self.scene_dynamics = SceneDynamicsHead(
                    feature_dim=SCENE_DYNAMICS_FEATURE_DIM,
                    motor_dim=cfg.n_actuators,
                    hidden=cfg.motor_hidden,
                )
        else:
            self.has_scene_dynamics = False
        # Self-state feedback spine (self-model program, Phase 1). When the
        # faculty is on, this projection injects the previous cycle's self-report
        # (A state-of-mind || C narrative || E metacognition) additively into the
        # stage-3 fuse, so the channels that "sound like inner life" actually shape
        # the next cycle. Zero-initialized => the first cycle (and every cycle
        # until learning moves it) is byte-identical to the no-spine baseline; the
        # fed-back vector is detached upstream so no gradient crosses cycles.
        self._self_dim = cfg.state_mind_out + cfg.narrative_out + cfg.metacog_out
        self.has_self_model_feedback = self.faculties.self_model_feedback
        if self.has_self_model_feedback:
            self.self_ingress = nn.Linear(self._self_dim, cfg.d_model)
            with torch.no_grad():
                self.self_ingress.weight.zero_()
                self.self_ingress.bias.zero_()
        # Predictive affect (self-model program, Phase 4). When the faculty is on,
        # this zero-init forward model predicts the next-step affective context; the
        # cycle adds its delta to the 4-D episodic proxy before projection, so the
        # agent perceives in light of how it expects to feel. Zero-init => parity
        # until learned; it rides the main prediction-error graph (trained for free).
        self.has_predictive_affect = self.faculties.predictive_affect
        if self.has_predictive_affect:
            from decadic.nn.affect_model import AffectPredictor

            self.affect_predictor = AffectPredictor(affect_dim=4)
        # Represented self (self-model program, Phase 5). A dedicated zero-init
        # ingress that injects the compact self-node embedding (interoception ‖
        # affect ‖ capability) into the stage-3 fuse, parallel to the A‖C‖E spine,
        # so the modelled self conditions the next cycle. Zero-init => parity.
        from decadic.state.self_model import REPSELF_DIM as _REPSELF_DIM

        self._repself_dim = int(_REPSELF_DIM)
        self.has_represented_self = self.faculties.represented_self
        if self.has_represented_self:
            self.repself_ingress = nn.Linear(self._repself_dim, cfg.d_model)
            with torch.no_grad():
                self.repself_ingress.weight.zero_()
                self.repself_ingress.bias.zero_()
        # WS5-M1 (relational binding): keyed read over the WM slot tensor.
        # Working memory's K entity tokens (frozen 40-d layout, see
        # docs/ws5_m0_wm_inventory.md) are attended with a query from the
        # pre-stage-3 latent; the readout enters additively through a
        # zero-init ingress -- byte-identical until learning moves it, exactly
        # the self_ingress/repself_ingress discipline. The slot tensor itself
        # is a per-cycle constant (built from WM state, no grad history), so
        # no cross-cycle BPTT path opens.
        from decadic.state.working_memory import SLOT_TENSOR_DIM as _SLOT_DIM

        self._slot_dim = int(_SLOT_DIM)
        self.has_wm_slot_tensor = getattr(self.faculties, "wm_slot_tensor", False)
        if self.has_wm_slot_tensor:
            slot_att = max(16, cfg.d_model // 4)
            self.slot_query = nn.Linear(cfg.d_model, slot_att)
            self.slot_key = nn.Linear(self._slot_dim, slot_att)
            self.slot_value = nn.Linear(self._slot_dim, cfg.d_model)
            self.slot_ingress = nn.Linear(cfg.d_model, cfg.d_model)
            with torch.no_grad():
                self.slot_ingress.weight.zero_()
                self.slot_ingress.bias.zero_()
        self.pc_heads = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model) for _ in range(4)])
        self.register_buffer("gru_h", torch.zeros(1, cfg.gru_hidden))
        self.register_buffer("lstm_h", torch.zeros(1, cfg.lstm_hidden))
        self.register_buffer("lstm_c", torch.zeros(1, cfg.lstm_hidden))

    def _plastic_block(
        self,
        name: str,
        in_features: int,
        out_features: int,
        hidden: int,
        *,
        pre_layernorm: bool = False,
    ) -> PlasticSparseGrowableMLP:
        f = self.plasticity
        block = PlasticSparseGrowableMLP(
            in_features,
            out_features,
            hidden,
            hidden_ceiling=f.ceiling_for(hidden),
            dropout=self.cfg.dropout,
            pre_layernorm=pre_layernorm,
            plastic=f.plastic,
            plastic_alpha=f.alpha,
            sparse=f.sparse,
            density=f.density,
            growth=f.growth,
        )
        self._plastic_names.append(name)
        return block

    # --- neuroplasticity controllers (no-ops when no block is plastic) ---------
    @property
    def has_plastic(self) -> bool:
        return bool(self._plastic_names)

    def plastic_blocks(self) -> list[PlasticSparseGrowableMLP]:
        return [getattr(self, n) for n in self._plastic_names]

    def hebbian_update_all(self, modulation: float, eta: float) -> None:
        for blk in self.plastic_blocks():
            blk.hebbian_update(modulation, eta)

    def enforce_masks_all(self) -> None:
        for blk in self.plastic_blocks():
            blk.enforce_masks()

    def rewire_all(self, fraction: float) -> int:
        return sum(blk.rewire(fraction) for blk in self.plastic_blocks())

    def grow_step(self, step: int, cap: int) -> list[torch.nn.Parameter]:
        """Wake up to ``step`` dormant neurons per growable block toward the per-block ``cap``.

        Returns the parameters whose optimizer state must be re-initialized.
        """
        changed: list[torch.nn.Parameter] = []
        for blk in self.plastic_blocks():
            if not blk.growth:
                continue
            target = min(int(cap), blk.hidden_ceiling)
            room = target - blk.awake_count()
            if room <= 0:
                continue
            if blk.grow(min(int(step), room)):
                changed.extend(blk.structural_params())
        return changed

    def growth_room(self, cap: int) -> bool:
        """True if any growable block can still wake neurons toward ``cap``."""
        for blk in self.plastic_blocks():
            if blk.growth and blk.awake_count() < min(int(cap), blk.hidden_ceiling):
                return True
        return False

    def set_awake_ceiling_all(self, n: int) -> list[torch.nn.Parameter]:
        changed: list[torch.nn.Parameter] = []
        for blk in self.plastic_blocks():
            if blk.growth and blk.set_awake_ceiling(n):
                changed.extend(blk.structural_params())
        return changed

    def set_density_all(self, density: float) -> list[torch.nn.Parameter]:
        changed: list[torch.nn.Parameter] = []
        for blk in self.plastic_blocks():
            if blk.sparse and blk.reseed_density(density):
                changed.extend([blk.l1_weight, blk.l2_weight])
        return changed

    def set_alpha_all(self, alpha: float) -> None:
        for blk in self.plastic_blocks():
            if blk.plastic:
                with torch.no_grad():
                    blk.alpha.fill_(float(alpha))

    def set_effective_alpha_all(self, alpha: float) -> None:
        for blk in self.plastic_blocks():
            if blk.plastic:
                blk.set_effective_alpha(float(alpha))

    def connection_density(self) -> float:
        total = 0
        active = 0
        for blk in self.plastic_blocks():
            aw = blk.awake_count()
            total += aw * blk.in_features + blk.out_features * aw
            active += blk.active_connections()
        return float(active / total) if total else 1.0

    def rewire_changed_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for blk in self.plastic_blocks():
            if blk.sparse:
                params.extend([blk.l1_weight, blk.l2_weight])
        return params

    def awake_neurons(self) -> int:
        return sum(blk.awake_count() for blk in self.plastic_blocks())

    def allocated_neurons(self) -> int:
        return sum(blk.hidden_ceiling for blk in self.plastic_blocks())

    def active_connections(self) -> int:
        return sum(blk.active_connections() for blk in self.plastic_blocks())

    def plastic_alpha_mean(self) -> float:
        blocks = [blk for blk in self.plastic_blocks() if blk.plastic]
        if not blocks:
            return 0.0
        return float(sum(float(blk.alpha.detach().abs().item()) for blk in blocks) / len(blocks))

    def plastic_effective_alpha_mean(self) -> float:
        blocks = [blk for blk in self.plastic_blocks() if blk.plastic]
        if not blocks:
            return 0.0
        return float(sum(blk.effective_alpha_value() for blk in blocks) / len(blocks))

    def plastic_overlay_ratio_stats(self) -> tuple[float, float]:
        vals = [blk.overlay_ratio_stats() for blk in self.plastic_blocks() if blk.plastic]
        if not vals:
            return 0.0, 0.0
        means = [v[0] for v in vals]
        maxes = [v[1] for v in vals]
        return float(sum(means) / len(means)), float(max(maxes))

    def plastic_arch_meta(self) -> dict[str, Any]:
        return {n: getattr(self, n).arch_meta() for n in self._plastic_names}

    def reset_plastic_traces(self) -> None:
        for blk in self.plastic_blocks():
            blk.reset_plastic_trace()

    def reset_recurrent_state(self) -> None:
        """Zero the transient recurrent buffers (short-term context, not weights).

        Used by the cycle's NaN firewall: clearing these on a non-finite cycle
        stops a poisoned hidden state from re-poisoning every subsequent forward
        pass, while leaving all learned parameters intact.
        """
        with torch.no_grad():
            self.gru_h.zero_()
            self.lstm_h.zero_()
            self.lstm_c.zero_()

    def _assemble_perception_context(
        self,
        prev_z5: torch.Tensor | None,
        lstm_h: torch.Tensor | None,
        mem: torch.Tensor | None,
        scene: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fixed-width [1, ctx_dim] history vector (all sources detached, zeros if absent)."""
        dev = self.ingress.weight.device
        dt = self.ingress.weight.dtype

        def fix(t: torch.Tensor | None, size: int) -> torch.Tensor:
            if t is None:
                return torch.zeros(1, size, device=dev, dtype=dt)
            t = t.detach().to(device=dev, dtype=dt).reshape(1, -1)
            if t.shape[1] == size:
                return t
            if t.shape[1] > size:
                return t[:, :size]
            return F.pad(t, (0, size - t.shape[1]))

        d = self.cfg.d_model
        pz = fix(prev_z5, d)
        lh = fix(lstm_h, self.cfg.lstm_hidden)
        mm = fix(mem, self.cfg.memory_context_dim)
        if scene is None:
            sc = torch.zeros(1, d, device=dev, dtype=dt)
        else:
            s = scene.detach().to(device=dev, dtype=dt).reshape(1, 1, -1)
            sc = F.adaptive_avg_pool1d(s, d).reshape(1, d)  # parameter-free compression
        return torch.cat([pz, lh, mm, sc], dim=-1)

    def top_down_perceive(
        self,
        z0_bu: torch.Tensor,
        *,
        prev_z5: torch.Tensor | None = None,
        lstm_h: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        scene: torch.Tensor | None = None,
        intero: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Blend bottom-up z0 with a learned top-down prediction under a precision gate.

        Returns ``(z0_eff, z0_hat, gate)``. With the loop disabled (or the gate
        learned/initialized to 1) this is exactly the bottom-up encode. The
        prediction is computed from *detached* history, so no gradient flows
        across cycles (the within-cycle params still learn). Predictive-coding
        form: ``z0_eff = z0_hat + gate * (z0_bu - z0_hat)``.
        """
        if not self.has_perception_feedback:
            return z0_bu, None, None
        ctx = self._assemble_perception_context(prev_z5, lstm_h, mem, scene)
        z0_hat = self.top_down(ctx)
        if intero is None:
            intero_t = torch.zeros(1, 3, device=z0_bu.device, dtype=z0_bu.dtype)
        else:
            intero_t = intero.detach().to(device=z0_bu.device, dtype=z0_bu.dtype).reshape(1, -1)
        gate = torch.sigmoid(self.precision_gate(torch.cat([ctx, intero_t], dim=-1)))
        z0_eff = z0_hat + gate * (z0_bu - z0_hat)
        return z0_eff, z0_hat, gate

    def slot_encode(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run slot attention over [1, N, CLIP_PATCH_DIM] patch features."""
        return self.slots_module(patch_tokens)

    def slot_pool(self, slots: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Presence-gated flatten of [B, K, slot_dim] -> [B, K*slot_dim] for slot_ingress."""
        gated = slots * presence.unsqueeze(-1)
        return gated.reshape(slots.shape[0], -1)

    def forward_predict(
        self, state: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict the next controllable-proprio vector from (state, motor command).

        ``detach_params=True`` freezes the forward-model weights for the term,
        so the active-inference policy update flows into the motor head / stack
        without letting the policy cheat by editing its own world model.
        """
        x = torch.cat([state, u], dim=-1)
        w1, b1 = self.fwd_l1.weight, self.fwd_l1.bias
        w2, b2 = self.fwd_l2.weight, self.fwd_l2.bias
        if detach_params:
            w1, b1, w2, b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        hidden = F.gelu(F.linear(x, w1, b1))
        return F.linear(hidden, w2, b2)

    def forward_predict_intero(
        self,
        state: torch.Tensor,
        u: torch.Tensor,
        intero_cur: torch.Tensor,
        *,
        detach_params: bool = False,
    ) -> torch.Tensor:
        """Predict the next interoceptive (reservoir) state from (state, command, reservoirs).

        Mirrors :meth:`forward_predict` for the homeostatic reservoirs. The current
        reservoir vector is part of the input so the model can predict its own
        evolution. ``detach_params=True`` freezes the world-model weights so the
        active-inference policy update flows into the motor head without letting the
        policy edit its own predictions (the anti-hallucination guarantee).
        """
        x = torch.cat([state, u, intero_cur], dim=-1)
        w1, b1 = self.fwd_intero_l1.weight, self.fwd_intero_l1.bias
        w2, b2 = self.fwd_intero_l2.weight, self.fwd_intero_l2.bias
        if detach_params:
            w1, b1, w2, b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        hidden = F.gelu(F.linear(x, w1, b1))
        return F.linear(hidden, w2, b2)

    def forward_predict_tactile(
        self, state: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict the next soft per-part contact load from (state, motor command).

        Mirrors :meth:`forward_predict` for full-body touch (no current-state
        conditioning needed). ``detach_params=True`` freezes the world-model
        weights so an active-inference term flows into the motor head without
        letting the policy edit its own tactile predictions.
        """
        x = torch.cat([state, u], dim=-1)
        w1, b1 = self.fwd_tactile_l1.weight, self.fwd_tactile_l1.bias
        w2, b2 = self.fwd_tactile_l2.weight, self.fwd_tactile_l2.bias
        if detach_params:
            w1, b1, w2, b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        hidden = F.gelu(F.linear(x, w1, b1))
        return F.linear(hidden, w2, b2)

    def forward_predict_effort(
        self, state: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict next localized effort/body-state from (state, motor command)."""
        x = torch.cat([state, u], dim=-1)
        w1, b1 = self.fwd_effort_l1.weight, self.fwd_effort_l1.bias
        w2, b2 = self.fwd_effort_l2.weight, self.fwd_effort_l2.bias
        if detach_params:
            w1, b1, w2, b2 = w1.detach(), b1.detach(), w2.detach(), b2.detach()
        hidden = F.gelu(F.linear(x, w1, b1))
        return F.linear(hidden, w2, b2)

    def successor_predict(
        self, state: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict discounted future reservoir-change features psi(state, command).

        Mirrors :meth:`forward_predict`; delegates to the SF head (its own module).
        Used by the consolidator's TD(lambda) loss (params trainable) and by the
        live policy value-shaping term (``detach_params=True``, anti-hallucination).
        """
        return self.sf_head.predict(state, u, detach_params=detach_params)

    def scene_dynamics_predict(
        self, features: torch.Tensor, u: torch.Tensor, *, detach_params: bool = False
    ) -> torch.Tensor:
        """Predict next anonymous scene entity state from prior entity state + motor."""
        if not getattr(self, "has_scene_dynamics", False):
            raise RuntimeError("scene dynamics head is not built")
        if not detach_params:
            return self.scene_dynamics(features, u)
        params = list(self.scene_dynamics.parameters())
        old = [p.requires_grad for p in params]
        try:
            for p in params:
                p.requires_grad_(False)
            return self.scene_dynamics(features, u)
        finally:
            for p, flag in zip(params, old):
                p.requires_grad_(flag)

    def forward(
        self,
        z0: torch.Tensor,
        episodic_proxy: torch.Tensor,
        memory_context: torch.Tensor | None = None,
        self_prev: torch.Tensor | None = None,
        repself_prev: torch.Tensor | None = None,
        stage4_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        stage4_shadow: bool = False,
        wm_slots: torch.Tensor | None = None,
        wm_slots_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # Per-stage instrumentation: wall time of each block (in execution
        # order) is attributed to its conceptual Decadic stage number.
        now = _stage_timer(z0.device)
        stage_ms: dict[int, float] = {}
        t_last = now()

        def mark(stage: int) -> None:
            nonlocal t_last
            t = now()
            stage_ms[stage] = stage_ms.get(stage, 0.0) + (t - t_last) * 1000.0
            t_last = t

        z1 = self.stage1(z0)
        z2 = self.stage2(z1.unsqueeze(1)).squeeze(1)
        mark(2)  # experience framing
        ze = self.epi_proj(episodic_proxy)
        if memory_context is None:
            memory_context = torch.zeros(
                z0.shape[0],
                self.cfg.memory_context_dim,
                device=z0.device,
                dtype=z0.dtype,
            )
        zm = self.mem_proj(memory_context)
        # Self-state feedback spine: additively fold the previous cycle's
        # self-report into the experience-framing latent before the stage-3 fuse.
        # No-op when the faculty is off or no prior self-state exists; zero-init
        # projection keeps it byte-identical at first.
        if self.has_self_model_feedback and self_prev is not None:
            sp = self_prev.to(device=z2.device, dtype=z2.dtype).reshape(z2.shape[0], -1)
            if sp.shape[-1] == self._self_dim:
                z2 = z2 + self.self_ingress(sp)
        # Represented self (Phase 5): inject the modelled self embedding, parallel
        # to the A‖C‖E spine above. Zero-init ingress => no-op until learned.
        if self.has_represented_self and repself_prev is not None:
            rp = repself_prev.to(device=z2.device, dtype=z2.dtype).reshape(z2.shape[0], -1)
            if rp.shape[-1] == self._repself_dim:
                z2 = z2 + self.repself_ingress(rp)
        # WS5-M1.2: keyed read over WM slot tokens. Query from the pre-stage-3
        # latent ("what am I in the middle of processing?"), keys/values from
        # the K entity tokens; masked softmax excludes empty slots. Zero-init
        # ingress => byte-identical until learned; no-op when the faculty is
        # off, no slots arrive, or every slot is masked out.
        if (
            self.has_wm_slot_tensor
            and wm_slots is not None
            and wm_slots_mask is not None
        ):
            m = wm_slots_mask.to(device=z2.device).reshape(-1).bool()
            if bool(m.any()) and wm_slots.shape[-1] == self._slot_dim:
                ks = wm_slots.to(device=z2.device, dtype=z2.dtype).reshape(
                    -1, self._slot_dim
                )
                q = self.slot_query(z2).reshape(-1)  # (A,)
                k = self.slot_key(ks)  # (K, A)
                logits = (k @ q) * (float(k.shape[-1]) ** -0.5)  # (K,)
                logits = logits.masked_fill(~m, float("-inf"))
                attn = torch.softmax(logits, dim=0)  # (K,)
                readout = attn.unsqueeze(0) @ self.slot_value(ks)  # (1, d_model)
                z2 = z2 + self.slot_ingress(readout)
        z3 = self.stage3(torch.cat([z2, ze, zm], dim=-1))
        mark(3)  # memory retrieval / heuristic fusion
        shadow_z4: torch.Tensor | None = None
        shadow_risk_logit: torch.Tensor | None = None
        if stage4_override is not None:
            # Stage 3->4 attention gate skip path (WS3): a decayed precedent
            # (cached z4 / risk_logit from the last escalated cycle) replaces
            # deliberative compute. Detached constants: no gradient flows
            # through risk_mlp on skipped cycles by design.
            z4 = stage4_override[0].to(device=z3.device, dtype=z3.dtype)
            risk_logit = stage4_override[1].to(device=z3.device, dtype=z3.dtype)
            if stage4_shadow:
                # WS3B-M0.2 shadow deliberation: what WOULD fresh stage 4 have
                # said? Diagnostics only -- no_grad on a detached z3, never
                # substituted into the live path, so the forward's outputs are
                # bit-identical with the tap on or off (regression-tested).
                with torch.no_grad():
                    shadow_z4 = self.risk_mlp(z3.detach())
                    shadow_risk_logit = self.risk_scalar(shadow_z4)
        else:
            z4 = self.risk_mlp(z3)
            risk_logit = self.risk_scalar(z4)
        mark(4)  # risk-utility evaluation
        z5a = self.stage5_enc(z4.unsqueeze(1)).squeeze(1)
        mark(5)  # pre-normative conclusion
        z5 = self.stage5_dec(z5a.unsqueeze(1)).squeeze(1)
        mark(8)  # strategy formation (decoder half)
        gru_in = torch.cat([z5, z4], dim=-1)
        # Clone the detached recurrent state so the in-place buffer copy_ below
        # cannot bump the version of a tensor that the motor/active-inference
        # backward path now depends on (the recurrent path is in-graph now).
        gh = self.gru_cell(gru_in, self.gru_h.detach().clone())
        self.gru_h.copy_(gh.detach())
        emotion = self.emotion_head(gh)
        mark(6)  # emotional/physiological experience
        lstm_in_t = torch.cat([z5, emotion], dim=-1)
        h, c = self.lstm_cell(
            lstm_in_t, (self.lstm_h.detach().clone(), self.lstm_c.detach().clone())
        )
        self.lstm_h.copy_(h.detach())
        self.lstm_c.copy_(c.detach())
        state_mind = self.state_mind_head(h)
        mark(7)  # reprioritization / state-of-mind update
        narrative = self.narrative_head(z5)
        metacognition = self.metacog_head(z5a)
        mark(8)  # strategy formation (narrative + metacog heads)
        pol_in_t = torch.cat([h, state_mind], dim=-1)
        pol = self.policy(pol_in_t)
        direction = torch.tanh(pol[:, :3])
        speed = torch.sigmoid(pol[:, 3:4]) * 2.0
        # Motor command: normalized PD targets in [-1, 1] (one per actuator).
        motor_u = torch.tanh(self.motor(pol_in_t))
        # Predicted next controllable-proprio state given this efference copy.
        s_hat = self.forward_predict(z5, motor_u)
        # Predicted next soft per-part contact load given this efference copy.
        t_hat = self.forward_predict_tactile(z5, motor_u)
        e_hat = self.forward_predict_effort(z5, motor_u)
        mark(9)  # behavioral response

        zs = [z1, z2, z3, z4, z5]
        pc_loss = torch.zeros((), device=z0.device, dtype=z0.dtype)
        pc_parts: dict[str, float] = {}
        for i in range(4):
            pred = self.pc_heads[i](zs[i])
            target = zs[i + 1].detach()
            li = F.mse_loss(pred, target)
            pc_parts[f"pc_{i + 1}_{i + 2}"] = float(li.item())
            pc_loss = pc_loss + li

        # Per-stage activity (RMS of each stage's representative latent),
        # read out in a single batched transfer to avoid per-stage GPU syncs.
        act_src: dict[int, torch.Tensor] = {
            2: z2,
            3: z3,
            4: z4,
            5: z5a,
            6: emotion,
            7: state_mind,
            8: z5,
            9: pol,
        }
        with torch.no_grad():
            act_t = torch.stack(
                [t.detach().float().pow(2).mean().sqrt() for t in act_src.values()]
            )
            # 32-bin |activation| profile per stage for the Brain Map (padded
            # for outputs narrower than 32 units); one batched CPU transfer.
            pooled_t = torch.stack(
                [
                    F.adaptive_avg_pool1d(
                        F.pad(flat, (0, max(0, 32 - flat.shape[-1]))).reshape(1, 1, -1),
                        32,
                    ).flatten()
                    for flat in (t.detach().float().abs().reshape(-1) for t in act_src.values())
                ]
            )
        acts = act_t.cpu().tolist()
        pooled = pooled_t.cpu().tolist()
        stage_metrics = [
            {
                "stage": s,
                "timing_ms": round(stage_ms.get(s, 0.0), 4),
                "activity": round(float(a), 6),
                "activations": [round(float(x), 5) for x in p],
            }
            for s, a, p in zip(act_src.keys(), acts, pooled)
        ]

        out_shadow: dict[str, Any] = {}
        if shadow_z4 is not None:
            out_shadow["shadow_z4"] = shadow_z4
            out_shadow["shadow_risk_logit"] = shadow_risk_logit
        return {
            **out_shadow,
            "z5": z5,
            "z4": z4,
            "risk_logit": risk_logit,
            "emotion": emotion,
            "state_mind": state_mind,
            "narrative": narrative,
            "metacognition": metacognition,
            "direction": direction,
            "speed": speed,
            "motor_u": motor_u,
            "s_hat": s_hat,
            "t_hat": t_hat,
            "e_hat": e_hat,
            "pc_loss": pc_loss,
            "pc_parts": pc_parts,
            "stage_metrics": stage_metrics,
        }
