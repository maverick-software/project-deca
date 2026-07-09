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
        # WS-FORAGE M3: goal-conditioning ingress. A fixed-width goal vector
        # (active need + deficit; M4 adds a remembered-target bearing) is folded
        # additively into the policy input so the motor heads can pursue what the
        # agent NEEDS-and-remembers, not only what it currently sees. Zero-init
        # (weight AND bias) -> contributes exactly 0 until trained, so the stack
        # is byte-identical at birth (house rule G2). Built unconditionally like
        # the world-model heads; forward() only folds it when a goal vec is
        # supplied, so it is a true no-op when goal conditioning is off.
        from decadic.nn.goal_conditioning import GOAL_VEC_DIM

        self.has_goal_conditioning = True
        self.goal_ingress = nn.Linear(GOAL_VEC_DIM, pol_in)
        with torch.no_grad():
            self.goal_ingress.weight.zero_()
            self.goal_ingress.bias.zero_()
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
        # WS-EXPAND E3.1: fine-motor error corrector. (motor command, current
        # controllable-proprio vector) -> a bounded additive correction on the
        # PD targets. FINAL layer zero-init (weight AND bias) -> the correction
        # is exactly 0 at birth (parity). Trained by feedback-error learning in
        # the pipeline: the realized per-joint tracking error is a SUPERVISED
        # target for what the correction should have been — never a reward the
        # head could game (evidence-review guardrail). Built unconditionally
        # like the world-model heads; forward() only applies it when a proprio
        # vector is supplied.
        self.has_motor_corrector = True
        self.motor_corrector_l1 = nn.Linear(
            cfg.n_actuators + cfg.forward_pred_dim, cfg.motor_hidden
        )
        self.motor_corrector_l2 = nn.Linear(cfg.motor_hidden, cfg.n_actuators)
        with torch.no_grad():
            self.motor_corrector_l2.weight.zero_()
            self.motor_corrector_l2.bias.zero_()
        # WS-EXPAND E3.2: per-actuator phase generator. A free-running phase
        # buffer per actuator whose motor contribution is c_j * sin(phase_j);
        # c and the frequency modulation come from a ZERO-INIT head, so there
        # is no oscillation at birth and rhythm is EARNED (the policy learns
        # to open per-actuator amplitude where periodic drive pays). forward()
        # touches it only when the pipeline passes a gate value; gate 0.0 is
        # the E3.3 aperiodic escape (threat/recovery cycles route raw).
        self.has_cpg = True
        self.cpg_head = nn.Linear(pol_in, 2 * cfg.n_actuators)
        with torch.no_grad():
            self.cpg_head.weight.zero_()
            self.cpg_head.bias.zero_()
        self.register_buffer("cpg_phase", torch.zeros(cfg.n_actuators))
        # WS-EXPAND E4.1: cached (habit) action head — a fast stimulus->action
        # map from the fused input z0, distilled online from the deliberate
        # policy's actions on ESCALATED cycles (teacher outputs only). It never
        # runs inside forward(); the pipeline calls it on gate-skip cycles and
        # blends by an earned trust weight (0 until distillation quality proves
        # the habit) — so the stack itself stays byte-identical.
        self.has_cached_policy = True
        self.cached_l1 = nn.Linear(cfg.d_model, cfg.motor_hidden)
        self.cached_l2 = nn.Linear(cfg.motor_hidden, cfg.n_actuators)
        with torch.no_grad():
            self.cached_l2.weight.zero_()
            self.cached_l2.bias.zero_()
        # WS-EXPAND E5.3: action veto — predicts imminent viability loss for
        # (state, command); the pipeline turns positive predictions into a
        # MINIMAL multiplicative attenuation (never a hard zero — the
        # over-conservatism guardrail). Output layer zero-init -> tanh(0)=0 ->
        # no attenuation at birth. Trained supervised on REALIZED viability
        # drops (error as target, never a reward).
        self.has_action_veto = True
        self.veto_l1 = nn.Linear(cfg.d_model + cfg.n_actuators, cfg.motor_hidden)
        self.veto_l2 = nn.Linear(cfg.motor_hidden, 1)
        with torch.no_grad():
            self.veto_l2.weight.zero_()
            self.veto_l2.bias.zero_()
        # WS-EXPAND E8.1 (TEST-FIRST): interoceptive embedding — a felt
        # body-state representation over (reservoirs + touch + effort) folded
        # into the affect path (the GRU input) through a ZERO-INIT ingress, so
        # affect is unchanged at birth and the A/B decides whether it stays.
        self.has_intero_embed = True
        _ie_in = int(_cfg.INTERO_PRED_DIM) + int(_cfg.TACTILE_PRED_DIM) + int(
            _cfg.EFFORT_PRED_DIM
        )
        self.intero_embed_l1 = nn.Linear(_ie_in, cfg.motor_hidden)
        self.intero_embed_l2 = nn.Linear(cfg.motor_hidden, cfg.motor_hidden)
        self.intero_embed_ingress = nn.Linear(cfg.motor_hidden, cfg.d_model * 2)
        with torch.no_grad():
            self.intero_embed_ingress.weight.zero_()
            self.intero_embed_ingress.bias.zero_()
        # WS-EXPAND E6: per-slot routing gate — bottom-up salience is already
        # in the slots; this scores top-down goal relevance per slot. Learned
        # SUPPRESSION grows downward from an exact-identity init
        # (1 - relu(tanh(0)) = 1.0), floored in the pipeline; reopened on the
        # E2.3 surprise channel so gating can't blind the agent to novelty.
        self.has_slot_gate = True
        from decadic.nn.goal_conditioning import GOAL_VEC_DIM as _GV_DIM

        self.slot_gate = nn.Linear(int(_cfg.slot_dim()) + _GV_DIM, 1)
        with torch.no_grad():
            self.slot_gate.weight.zero_()
            self.slot_gate.bias.zero_()
        # WS-EXPAND E9: discrete abstraction bottleneck (FSQ — no learned
        # codebook, no collapse mode). Projects a DETACHED z5 into a small
        # quantized code and self-supervises next-code prediction; gradients
        # never reach the shared trunk, so behavior is byte-identical and the
        # abstraction layer trains purely on the side.
        from decadic.nn.symbol import FSQ_DIMS

        self.has_symbols = True
        self.fsq_in = nn.Linear(cfg.d_model, FSQ_DIMS)
        self.fsq_next = nn.Linear(FSQ_DIMS, FSQ_DIMS)
        # WS-SYM 4.0: symbol FEEDBACK into cognition. The previous cycle's
        # quantized code (bundle._prev_symbol_q, a FSQ_DIMS vector) conditions
        # THIS cycle's policy through a zero-init lane, so the discrete symbol
        # the mind emits actually shapes its next deliberation instead of being
        # a detached read-out. Zero-init -> birth-identical; the trunk LEARNS
        # to use its own codes. On by default; meaning lives in the grounded
        # binding, so this is drift-robust (ws_symbol_integration_analysis.md).
        self.has_symbol_ingress = True
        self.symbol_ingress = nn.Linear(FSQ_DIMS, pol_in)
        with torch.no_grad():
            self.symbol_ingress.weight.zero_()
            self.symbol_ingress.bias.zero_()
        # WS-IND I1: attention schema — a predictive model of the system's own
        # attention. From (latent, current gate state) predict next cycle's
        # realized gate outcome: p(escalate), reason class, next score.
        # Output layer zero-init -> a fresh agent predicts nothing (p=0.5 ->
        # zero anticipatory bias) and the zero-init ingress feeds nothing back:
        # birth-identical. Trained on realized outcomes (outcome as target).
        from decadic.nn.attention_schema import GATE_STATE_DIM, SCHEMA_VEC_DIM

        self.has_attention_schema = True
        self.schema_l1 = nn.Linear(cfg.d_model + GATE_STATE_DIM, cfg.motor_hidden)
        self.schema_l2 = nn.Linear(cfg.motor_hidden, SCHEMA_VEC_DIM)
        with torch.no_grad():
            self.schema_l2.weight.zero_()
            self.schema_l2.bias.zero_()
        self.schema_ingress = nn.Linear(SCHEMA_VEC_DIM, pol_in)
        with torch.no_grad():
            self.schema_ingress.weight.zero_()
            self.schema_ingress.bias.zero_()
        # WS-IND I2: sequential deliberation — on an escalated cycle a DRAFT
        # forward runs first (no-grad, recurrent state restored afterward) and
        # its conclusion (z5, chosen action) re-enters the final forward
        # through this zero-init ingress: "think again with your draft in
        # view". Zero-init -> round 2 == round 1 at birth (parity).
        self.has_draft_ingress = True
        self.draft_ingress = nn.Linear(cfg.d_model + cfg.n_actuators, pol_in)
        with torch.no_grad():
            self.draft_ingress.weight.zero_()
            self.draft_ingress.bias.zero_()
        # WS-EXPAND E10.3: other-agent ingress — the dominant adaptive other's
        # egocentric state (presence, bearing, predicted-next bearing,
        # adaptivity strength) conditions the policy through a zero-init lane.
        # The adaptivity gate keeps the vector ALL-ZERO in solo scenes, so the
        # channel is inert exactly when there is no one to model.
        from decadic.state.other_agents import OTHER_VEC_DIM

        self.has_other_ingress = True
        self.other_ingress = nn.Linear(OTHER_VEC_DIM, pol_in)
        with torch.no_grad():
            self.other_ingress.weight.zero_()
            self.other_ingress.bias.zero_()
        # WS-EXPAND E10.4 (prerequisite): inverse dynamics — infer the action
        # that produced an observed proprio transition. Trained SUPERVISED on
        # the agent's own lived (proprio, proprio', executed action) triples
        # (the same buffers FEL uses); never called in forward() (parity).
        # This is the labeling model imitation-from-observation needs; the
        # demonstrator-labeling step itself waits on percepts that carry the
        # OTHER agent's body pose.
        self.has_inverse_model = True
        self.inv_l1 = nn.Linear(2 * cfg.forward_pred_dim, cfg.motor_hidden)
        self.inv_l2 = nn.Linear(cfg.motor_hidden, cfg.n_actuators)
        # WS-DEPTH D1: metacognitive calibration — predict own next
        # prediction-error and P(drive improves | current action), scored
        # against realized outcomes. Zero-init -> predicts nothing at birth.
        self.has_metacog_cal = True
        self.metacog_cal = nn.Linear(cfg.d_model, 2)
        with torch.no_grad():
            self.metacog_cal.weight.zero_()
            self.metacog_cal.bias.zero_()
        # WS-DEPTH P1 (stage A): recurrent percept refinement — a zero-init
        # residual cell iterates on the fused percept before stage 1
        # (algorithmic recurrence at the earliest lived stage), trained by a
        # percept-level forward model (predict the NEXT fused percept from the
        # refined current one + efference). Percept-key invariance guardrail:
        # zero-init -> z0 byte-identical at birth.
        self.has_percept_refine = True
        self.refine_l1 = nn.Linear(cfg.d_model, cfg.motor_hidden)
        self.refine_l2 = nn.Linear(cfg.motor_hidden, cfg.d_model)
        with torch.no_grad():
            self.refine_l2.weight.zero_()
            self.refine_l2.bias.zero_()
        self.percept_fwd_l1 = nn.Linear(cfg.d_model + cfg.n_actuators, cfg.motor_hidden)
        self.percept_fwd_l2 = nn.Linear(cfg.motor_hidden, cfg.d_model)
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
        # WS5-M2 (relational binding): keyed read over recalled-episode TOKENS
        # (frozen 80-d episodic embedding layout), beside the mean-pooled
        # context vector -- five remembered situations stop entering as their
        # average. Same discipline as the slot read: zero-init ingress,
        # per-cycle constant tokens, no cross-cycle BPTT.
        from decadic.memory.embeddings import EMBEDDING_DIM as _EPI_DIM

        self._mem_tok_dim = int(_EPI_DIM)
        self.has_memory_tokens = getattr(self.faculties, "memory_tokens", False)
        if self.has_memory_tokens:
            mem_att = max(16, cfg.d_model // 4)
            self.mem_tok_query = nn.Linear(cfg.d_model, mem_att)
            self.mem_tok_key = nn.Linear(self._mem_tok_dim, mem_att)
            self.mem_tok_value = nn.Linear(self._mem_tok_dim, cfg.d_model)
            self.mem_tok_ingress = nn.Linear(cfg.d_model, cfg.d_model)
            with torch.no_grad():
                self.mem_tok_ingress.weight.zero_()
                self.mem_tok_ingress.bias.zero_()
        # WS5-M3.1: relational core. The pooled relational summary augments
        # the stage-4 risk input through a zero-init ingress (resolves the PRD
        # open decision: neither zero-concat nor a separate head -- the house
        # additive pattern keeps risk_mlp's shape and checkpoint compat).
        # Computed on DELIBERATIVE cycles only (see forward): a gate skip
        # never pays for relational deliberation, which makes it exactly the
        # compute the WS3 gate prices.
        self.has_relational_core = getattr(self.faculties, "relational_core", False)
        if self.has_relational_core:
            from decadic.nn.relational_core import RelationalCore

            self.relational = RelationalCore(d_rel=max(32, cfg.d_model // 4))
            self.rel_ingress = nn.Linear(self.relational.d_rel, cfg.d_model)
            with torch.no_grad():
                self.rel_ingress.weight.zero_()
                self.rel_ingress.bias.zero_()
        # WS6-M2.1: the voice head -- the vocal motor organ (mouth as motor,
        # not language module). Reads the SAME policy latent (lstm_h ||
        # state_mind) the motor head consumes and emits VOICE_DIM articulatory
        # params squashed by tanh in forward(). Zero-init weight AND bias =>
        # tanh(0) = exact zeros at init: the newborn does not speak. Flag-off
        # builds construct NO module, so the off state_dict is byte-identical
        # (house parity rule).
        self.has_voice = getattr(self.faculties, "voice", False)
        if self.has_voice:
            from decadic.audio.vocal_tract import VOICE_DIM

            self.voice_head = nn.Linear(pol_in, VOICE_DIM)
            with torch.no_grad():
                self.voice_head.weight.zero_()
                self.voice_head.bias.zero_()
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
        # WS-DEPTH P2: OPT-IN cap on the top-down fraction (gate is the
        # bottom-up weight; the floor is 1 - cap). Cap 1.0 (default) = no
        # clamp: full top-down is the designed occlusion fill-in (percepts
        # reconstructed from the persistent mental image — suite-pinned), and
        # the chronic-decoupling concern is watched by topdown_frac telemetry
        # (= 1 - gate_mean) and governable per-run via
        # DECADIC_PERCEPT_TOPDOWN_CAP.
        _cap = _cfg.percept_topdown_cap()
        if _cap < 1.0:
            gate = gate.clamp(min=1.0 - _cap)
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

    def inverse_action(
        self, proprio_prev: torch.Tensor, proprio_now: torch.Tensor
    ) -> torch.Tensor:
        """WS-EXPAND E10.4: infer the action behind a proprio transition.

        tanh-bounded like every motor quantity. Trained on own experience;
        applied later to observed demonstrators (BCO) once their body pose is
        perceivable.
        """
        h = F.gelu(self.inv_l1(torch.cat([proprio_prev, proprio_now], dim=-1)))
        return torch.tanh(self.inv_l2(h))

    def metacog_calibrate(self, z5: torch.Tensor) -> torch.Tensor:
        """WS-DEPTH D1: [B, 2] raw — [0] predicted next pc_loss (softplus'd by
        the caller), [1] logit P(drive improves). Zero-init -> zeros at birth."""
        return self.metacog_cal(z5)

    def refine_percept(self, z0: torch.Tensor, iters: int) -> torch.Tensor:
        """WS-DEPTH P1: iterative zero-init residual refinement of the fused
        percept. iters<=0 or untrained -> exactly z0 (invariance guardrail)."""
        z = z0
        for _ in range(max(0, int(iters))):
            z = z + self.refine_l2(F.gelu(self.refine_l1(z)))
        return z

    def percept_forward(self, z0_refined: torch.Tensor, motor_u: torch.Tensor) -> torch.Tensor:
        """WS-DEPTH P1: predict the NEXT fused percept from (refined percept,
        efference) — the training signal that makes refinement earn its keep."""
        return self.percept_fwd_l2(
            F.gelu(self.percept_fwd_l1(torch.cat([z0_refined, motor_u], dim=-1)))
        )

    def attention_schema_predict(
        self, z5: torch.Tensor, gate_state: torch.Tensor
    ) -> torch.Tensor:
        """WS-IND I1: raw schema output [B, SCHEMA_VEC_DIM].

        Layout: [0] escalate logit, [1:1+len(GATE_REASONS)] reason logits,
        [-1] predicted next gate score. Zero-init output layer -> all zeros at
        birth (sigmoid 0.5 escalate, uniform reasons, score 0).
        """
        gs = gate_state.to(device=z5.device, dtype=z5.dtype)
        if gs.dim() == 1:
            gs = gs.unsqueeze(0)
        if gs.shape[0] != z5.shape[0]:
            gs = gs.expand(z5.shape[0], -1)
        return self.schema_l2(F.gelu(self.schema_l1(torch.cat([z5, gs], dim=-1))))

    def snapshot_recurrent_state(self) -> dict[str, torch.Tensor]:
        """WS-IND I2: clone the buffers a forward pass advances, so a DRAFT
        forward can run without double-advancing the recurrent state."""
        out: dict[str, torch.Tensor] = {}
        for name in ("gru_h", "lstm_h", "lstm_c", "cpg_phase"):
            buf = getattr(self, name, None)
            if buf is not None:
                out[name] = buf.detach().clone()
        return out

    def restore_recurrent_state(self, snap: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, val in snap.items():
                buf = getattr(self, name, None)
                if buf is not None and buf.shape == val.shape:
                    buf.copy_(val)

    def motor_veto_raw(self, z5: torch.Tensor, motor_u: torch.Tensor) -> torch.Tensor:
        """WS-EXPAND E5.3: raw viability-loss prediction for (state, command).

        tanh applied by callers; zero-init output layer -> 0 at birth.
        """
        return self.veto_l2(F.gelu(self.veto_l1(torch.cat([z5, motor_u], dim=-1))))

    def intero_embedding(self, body_vec: torch.Tensor) -> torch.Tensor:
        """WS-EXPAND E8.1: felt body-state embedding over intero+touch+effort."""
        return F.gelu(self.intero_embed_l2(F.gelu(self.intero_embed_l1(body_vec))))

    def slot_relevance(
        self, wm_slots: torch.Tensor, goal_vec: torch.Tensor, floor: float
    ) -> torch.Tensor:
        """WS-EXPAND E6: per-slot pass weight in [floor, 1]; exact 1.0 at init.

        ``wm_slots``: [B, K, slot_dim]; ``goal_vec``: [GOAL_VEC_DIM] or
        [B, GOAL_VEC_DIM]. Suppression = relu(tanh(score)) grows only as the
        gate learns which slots are goal-irrelevant.
        """
        b, k, d = wm_slots.shape
        gv = goal_vec.to(device=wm_slots.device, dtype=wm_slots.dtype)
        if gv.dim() == 1:
            gv = gv.unsqueeze(0)
        gv = gv.unsqueeze(1).expand(b, k, -1)
        score = self.slot_gate(torch.cat([wm_slots, gv], dim=-1))  # [B, K, 1]
        return (1.0 - torch.relu(torch.tanh(score))).clamp(min=float(floor))

    def cached_action(self, z0: torch.Tensor) -> torch.Tensor:
        """WS-EXPAND E4.1: the habit head's action for a fused input.

        tanh-bounded; final layer zero-init. Used two ways by the pipeline:
        with gradients for the distillation loss, under no_grad for the
        skip-cycle blend.
        """
        return torch.tanh(self.cached_l2(F.gelu(self.cached_l1(z0))))

    def motor_correction(
        self, motor_u: torch.Tensor, proprio_vec: torch.Tensor
    ) -> torch.Tensor:
        """WS-EXPAND E3.1: bounded correction for (command, current proprio).

        tanh-bounded; the final layer is zero-init so this returns exactly 0
        until feedback-error learning trains it. Called from forward() (with
        the configured gain) and from the pipeline's FEL supervision term.
        """
        h = F.gelu(self.motor_corrector_l1(torch.cat([motor_u, proprio_vec], dim=-1)))
        return torch.tanh(self.motor_corrector_l2(h))

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
        mem_tokens: torch.Tensor | None = None,
        mem_tokens_mask: torch.Tensor | None = None,
        goal_vec: torch.Tensor | None = None,
        proprio_vec: torch.Tensor | None = None,
        cpg_gate: float | None = None,
        body_vec: torch.Tensor | None = None,
        schema_vec: torch.Tensor | None = None,
        draft_vec: torch.Tensor | None = None,
        other_vec: torch.Tensor | None = None,
        symbol_vec: torch.Tensor | None = None,
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
        # WS5-M2.1: keyed read over recalled-episode tokens, parallel to the
        # slot read above (query from the same pre-stage-3 latent). The
        # mean-pooled zm path is untouched -- this AUGMENTS recall with
        # structure rather than replacing the legacy signal (conservative
        # default per PRD open decisions).
        if (
            self.has_memory_tokens
            and mem_tokens is not None
            and mem_tokens_mask is not None
        ):
            mm = mem_tokens_mask.to(device=z2.device).reshape(-1).bool()
            if bool(mm.any()) and mem_tokens.shape[-1] == self._mem_tok_dim:
                ts = mem_tokens.to(device=z2.device, dtype=z2.dtype).reshape(
                    -1, self._mem_tok_dim
                )
                mq = self.mem_tok_query(z2).reshape(-1)
                mk = self.mem_tok_key(ts)
                mlogits = (mk @ mq) * (float(mk.shape[-1]) ** -0.5)
                mlogits = mlogits.masked_fill(~mm, float("-inf"))
                mattn = torch.softmax(mlogits, dim=0)
                mread = mattn.unsqueeze(0) @ self.mem_tok_value(ts)
                z2 = z2 + self.mem_tok_ingress(mread)
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
            # WS5-M3.1: relational deliberation, deliberative cycles only.
            # The summary enters the risk input via zero-init ingress; the
            # shadow tap above deliberately uses plain z3 (it measures the
            # PRE-relational counterfactual; revisit when the gate retrains).
            z3_s4 = z3
            if self.has_relational_core:
                rel = self.relational(
                    wm_slots,
                    wm_slots_mask,
                    mem_tokens,
                    mem_tokens_mask,
                    episodic_proxy,
                )
                z3_s4 = z3 + self.rel_ingress(rel)
            z4 = self.risk_mlp(z3_s4)
            risk_logit = self.risk_scalar(z4)
        mark(4)  # risk-utility evaluation
        z5a = self.stage5_enc(z4.unsqueeze(1)).squeeze(1)
        mark(5)  # pre-normative conclusion
        z5 = self.stage5_dec(z5a.unsqueeze(1)).squeeze(1)
        mark(8)  # strategy formation (decoder half)
        gru_in = torch.cat([z5, z4], dim=-1)
        # WS-EXPAND E8.1: fold the felt body-state embedding into the affect
        # path through the zero-init ingress (affect becomes a readout of body
        # state, not only the latent). No body vector supplied -> untouched.
        if body_vec is not None and getattr(self, "has_intero_embed", False):
            bv = body_vec.to(device=gru_in.device, dtype=gru_in.dtype)
            if bv.dim() == 1:
                bv = bv.unsqueeze(0)
            if bv.shape[0] != gru_in.shape[0]:
                bv = bv.expand(gru_in.shape[0], -1)
            gru_in = gru_in + self.intero_embed_ingress(self.intero_embedding(bv))
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
        # WS-FORAGE M3: fold the active-goal signal into the policy input. Zero-
        # init ingress -> exactly no-op until trained (byte-identical at birth);
        # None when goal conditioning is off or no goal is active.
        if goal_vec is not None:
            gv = goal_vec.to(device=pol_in_t.device, dtype=pol_in_t.dtype)
            if gv.dim() == 1:
                gv = gv.unsqueeze(0).expand(pol_in_t.shape[0], -1)
            pol_in_t = pol_in_t + self.goal_ingress(gv)
        # WS-IND I1.2: the attention schema's prediction re-enters the policy
        # input (the model of attention informing control). Zero-init -> no-op.
        if schema_vec is not None and getattr(self, "has_attention_schema", False):
            sv = schema_vec.to(device=pol_in_t.device, dtype=pol_in_t.dtype)
            if sv.dim() == 1:
                sv = sv.unsqueeze(0).expand(pol_in_t.shape[0], -1)
            pol_in_t = pol_in_t + self.schema_ingress(sv)
        # WS-IND I2: the draft round's conclusion re-enters the final round
        # ("think again with your draft in view"). Zero-init -> no-op.
        if draft_vec is not None and getattr(self, "has_draft_ingress", False):
            dv = draft_vec.to(device=pol_in_t.device, dtype=pol_in_t.dtype)
            if dv.dim() == 1:
                dv = dv.unsqueeze(0).expand(pol_in_t.shape[0], -1)
            pol_in_t = pol_in_t + self.draft_ingress(dv)
        # WS-EXPAND E10.3: the modeled other conditions the policy. Zero-init
        # AND all-zero when solo -> doubly inert until someone is there.
        if other_vec is not None and getattr(self, "has_other_ingress", False):
            ov = other_vec.to(device=pol_in_t.device, dtype=pol_in_t.dtype)
            if ov.dim() == 1:
                ov = ov.unsqueeze(0).expand(pol_in_t.shape[0], -1)
            pol_in_t = pol_in_t + self.other_ingress(ov)
        # WS-SYM 4.0: the previous cycle's own quantized code conditions this
        # cycle's policy -- symbols integrated INTO cognition, not a read-out.
        # Zero-init -> birth-identical; the trunk learns to use its codes.
        if symbol_vec is not None and getattr(self, "has_symbol_ingress", False):
            yv = symbol_vec.to(device=pol_in_t.device, dtype=pol_in_t.dtype)
            if yv.dim() == 1:
                yv = yv.unsqueeze(0).expand(pol_in_t.shape[0], -1)
            pol_in_t = pol_in_t + self.symbol_ingress(yv)
        pol = self.policy(pol_in_t)
        direction = torch.tanh(pol[:, :3])
        speed = torch.sigmoid(pol[:, 3:4]) * 2.0
        # Motor command: normalized PD targets in [-1, 1] (one per actuator).
        motor_u = torch.tanh(self.motor(pol_in_t))
        # WS-EXPAND E3.1: additive error-corrector. Zero-init final layer ->
        # exactly 0 at birth; true no-op when no proprio vector is supplied.
        if proprio_vec is not None and getattr(self, "has_motor_corrector", False):
            pv = proprio_vec.to(device=motor_u.device, dtype=motor_u.dtype)
            if pv.dim() == 1:
                pv = pv.unsqueeze(0)
            if pv.shape[0] != motor_u.shape[0]:
                pv = pv.expand(motor_u.shape[0], -1)
            motor_u = motor_u + _cfg.motor_corrector_gain() * self.motor_correction(
                motor_u, pv
            )
        # WS-EXPAND E3.2/E3.3: phase-generator contribution c*sin(phase). c is
        # zero-init (silent at birth). cpg_gate None -> feature untouched
        # (parity); 0.0 -> aperiodic escape (phase holds, no contribution).
        if cpg_gate is not None and getattr(self, "has_cpg", False):
            gate_f = float(cpg_gate)
            if gate_f > 0.0:
                mod = self.cpg_head(pol_in_t)
                n_a = motor_u.shape[-1]
                c_amp = torch.tanh(mod[:, :n_a])
                dfreq = torch.tanh(mod[:, n_a : 2 * n_a])
                with torch.no_grad():
                    step = _cfg.cpg_base_step() * (
                        1.0 + 0.5 * torch.nan_to_num(dfreq.detach().mean(dim=0))
                    )
                    self.cpg_phase.add_(step).remainder_(6.283185307179586)
                motor_u = motor_u + (gate_f * _cfg.cpg_amp()) * c_amp * torch.sin(
                    self.cpg_phase
                ).unsqueeze(0)
        # WS6-M2.1: vocal efference beside the motor command, from the same
        # policy latent (the mouth is one more motor organ). Zero-init head =>
        # all-zero params at init (silence-ish through the synth's energy map).
        voice_u = torch.tanh(self.voice_head(pol_in_t)) if self.has_voice else None
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
        # voice_u is present only when the voice faculty built the head, so
        # flag-off forward outputs are key-identical to the baseline.
        if voice_u is not None:
            out_shadow["voice_u"] = voice_u
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
