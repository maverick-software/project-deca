"""Owning trainable stack, frozen encoders (partially trainable proprio MLP), optimizer."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from decadic.nn.config import (
    HEAVY_PRESETS,
    NeuralArchitectureConfig,
    neural_config_from_env,
    resolve_preset,
)
from decadic.nn.faculties import CognitionFaculties
from decadic.nn.frozen_encoders import FrozenSensoryEncoders
from decadic.nn.neural_stack import NeuralCognitiveStack
from decadic.nn.optim import build_optimizer
from decadic.nn.plastic import PlasticityFlags, PlasticityRuntimeState

logger = logging.getLogger(__name__)

# fp32 weight + grad + two Adam moments per parameter, before activations.
_TRAIN_BYTES_PER_PARAM = 16


def _warn_if_heavy(preset: str, n_params: int, device: torch.device) -> None:
    """Loud warning when a define-only HEAVY tier is built.

    These tiers (250m/500m/1b) run, but training them every cognitive cycle in
    fp32 Adam can exhaust a single consumer GPU; there is no mixed-precision /
    sharded training path yet. Best-effort live VRAM check on cuda.
    """
    if preset not in HEAVY_PRESETS:
        return
    train_gb = n_params * _TRAIN_BYTES_PER_PARAM / (1024**3)
    base = (
        f"preset {preset!r} is a HEAVY/define-only tier: ~{train_gb:.1f} GB for fp32 "
        "weights+grads+Adam moments (before activations), trained every cognitive cycle. "
        "Set DECADIC_MEMORY_EFFICIENT_TRAINING=1 (8-bit Adam + bf16 forward on CUDA) to "
        "cut training memory; otherwise expect slow cycles or CUDA OOM."
    )
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            free_gb = free / (1024**3)
            if train_gb > free_gb:
                logger.warning("%s GPU has only ~%.1f GB free now -> likely to OOM.", base, free_gb)
                return
        except Exception:  # mem_get_info unsupported on this device/build
            pass
    logger.warning("%s", base)


class NeuralBundle:
    """Phase 2 cognitive weights + optimizer checkpointing."""

    def __init__(
        self,
        *,
        agent_id: str,
        device: torch.device,
        cfg: NeuralArchitectureConfig,
        encoders: FrozenSensoryEncoders,
        stack: NeuralCognitiveStack,
        optimizer: torch.optim.Optimizer,
        preset: str = "tiny",
        flags: PlasticityFlags | None = None,
        faculties: CognitionFaculties | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.device = device
        self.cfg = cfg
        self.encoders = encoders
        self.stack = stack.to(device)
        self.optimizer = optimizer
        self.preset = preset
        # Core cognitive faculties this bundle was built with (perception-feedback
        # loop, perception mode, encoder mode). Stored so reset()/load() rebuild to
        # the agent's faculties rather than silently re-reading the process env.
        self.faculties = faculties or CognitionFaculties()
        # Neuroplasticity flags + live controller state. The state is None when no
        # plastic block exists, so the per-cycle hook short-circuits (parity).
        self.flags = flags or PlasticityFlags()
        self.plasticity_state: PlasticityRuntimeState | None = (
            PlasticityRuntimeState.from_flags(self.flags) if self.stack.has_plastic else None
        )
        # Cross-cycle transition buffers for the forward model (active inference):
        # the previous cycle's state latent and realized motor command, used to
        # score the world-model prediction against the realized next state.
        self.prev_state: torch.Tensor | None = None
        self.prev_motor: torch.Tensor | None = None
        # Previous cycle's normalized reservoir vector, used to score the
        # interoceptive world model against the realized transition (homeostatic
        # drive reduction). Unused/None when that flag is off.
        self.prev_intero: torch.Tensor | None = None
        # Self-state feedback spine (self-model program): the previous cycle's
        # detached self-report (A||C||E) fed back into the stack. Like the other
        # transition buffers it is ephemeral -- never checkpointed, rebuilt as None
        # on load, and zeroed on a non-finite cycle. Stays None when the faculty
        # is off (full parity).
        self.prev_self: torch.Tensor | None = None
        # Temporal-integration window (self-model program): the accumulator + the
        # last committed "now" percept. Ephemeral (never checkpointed); created
        # lazily by the cycle when DECADIC_INTEGRATION_WINDOW_MS > 0.
        self._integration_window: Any | None = None
        self._committed_now: torch.Tensor | None = None
        # Predictive affect (self-model program): the previous cycle's actual 4-D
        # affect context, fed to the forward model to anticipate the next moment.
        # Ephemeral (never checkpointed); stays None when the faculty is off.
        self.prev_affect: torch.Tensor | None = None
        # Scene dynamics (perception organ): previous anonymous scene entities
        # used to predict the next scene. Ephemeral and label-free; never
        # checkpointed.
        self.prev_scene_features: torch.Tensor | None = None
        self.prev_scene_entity_ids: list[str] = []
        # Represented self (self-model program): the previous cycle's compact
        # self-node embedding (intero‖affect‖capability), fed back via repself_ingress.
        # Ephemeral (never checkpointed); stays None when the faculty is off.
        self.prev_repself: torch.Tensor | None = None

    @staticmethod
    def resolve_device() -> torch.device:
        pref = os.environ.get("DECADIC_DEVICE", "").strip().lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def try_build(
        cls,
        agent_id: str,
        preset: str | None = None,
        flags: PlasticityFlags | None = None,
        faculties: CognitionFaculties | None = None,
    ) -> NeuralBundle | None:
        flag = os.environ.get("DECADIC_USE_NEURAL", "1").strip().lower()
        if flag in ("0", "false", "no", "off"):
            return None
        try:
            import torch as _torch  # noqa: F401
        except ImportError:
            logger.warning("torch not installed; neural cognition disabled")
            return None

        device = cls.resolve_device()
        if device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(device)
                logger.info(
                    "neural compute device=cuda gpu=%s vram_free_gb=%.1f vram_total_gb=%.1f bf16=%s",
                    torch.cuda.get_device_name(device),
                    free / (1024**3),
                    total / (1024**3),
                    torch.cuda.is_bf16_supported(),
                )
            except Exception:
                logger.info("neural compute device=cuda gpu=%s", torch.cuda.get_device_name(device))
        else:
            logger.info("neural compute device=cpu (no CUDA available/selected; ~10-20x slower)")
        preset_name = resolve_preset(preset)
        cfg = neural_config_from_env(preset_name)
        # An explicit override (e.g. UI-set per-agent faculties / new-agent
        # defaults) wins; otherwise fall back to the process env defaults.
        faculties = faculties if faculties is not None else CognitionFaculties.from_env()
        flags = flags if flags is not None else PlasticityFlags.from_env()
        encoders = FrozenSensoryEncoders(
            mode=faculties.encoder_mode, device=device, proprio_dim_out=cfg.proprio_emb
        )
        encoders.to(device)
        stack = NeuralCognitiveStack(cfg, flags, faculties)
        params = list(stack.parameters()) + list(encoders.parameters())
        n_params = sum(p.numel() for p in params)
        # Loud warning before the (potentially large) optimizer allocation when a
        # define-only heavy tier is selected, so an oversized preset on a small GPU
        # is never silent.
        _warn_if_heavy(preset_name, n_params, device)
        lr = float(os.environ.get("DECADIC_LEARNING_RATE", "1e-4"))
        from decadic.config import memory_efficient_training_enabled

        optimizer, _opt_kind = build_optimizer(
            params,
            lr=lr,
            device=device,
            memory_efficient=memory_efficient_training_enabled(),
        )
        logger.info(
            "NeuralBundle agent_id=%s device=%s encoder_mode=%s preset=%s trainable_params=%s "
            "perception_feedback=%s perception_mode=%s plasticity=%s sparse=%s growth=%s",
            agent_id,
            device,
            faculties.encoder_mode,
            preset_name,
            n_params,
            faculties.perception_feedback,
            faculties.perception_mode,
            flags.plastic,
            flags.sparse,
            flags.growth,
        )
        return cls(
            agent_id=agent_id,
            device=device,
            cfg=cfg,
            encoders=encoders,
            stack=stack,
            optimizer=optimizer,
            preset=preset_name,
            flags=flags,
            faculties=faculties,
        )

    def reset_optimizer_state(self, params: list[torch.nn.Parameter]) -> None:
        """Drop Adam moment buffers for ``params`` so structurally-changed weights
        (rewired connections, freshly-woken neurons) learn from a clean slate
        without stale momentum. The optimizer's param groups are untouched."""
        if not params:
            return
        for p in params:
            self.optimizer.state.pop(p, None)

    def after_optimization_step(self) -> None:
        with torch.no_grad():
            self.stack.gru_h.copy_(self.stack.gru_h.detach())
            self.stack.lstm_h.copy_(self.stack.lstm_h.detach())
            self.stack.lstm_c.copy_(self.stack.lstm_c.detach())

    def train_autocast(self):
        """bf16 autocast context for the forward pass (memory-efficient training).

        Returns a real ``torch.autocast`` only when ``DECADIC_MEMORY_EFFICIENT_TRAINING``
        is on AND the device is CUDA; otherwise a ``nullcontext`` so the fp32 path is
        byte-identical (CPU / off). bf16 needs no GradScaler, so the backward pass
        stays in fp32 outside this context.
        """
        from contextlib import nullcontext

        from decadic.config import memory_efficient_training_enabled

        if self.device.type == "cuda" and memory_efficient_training_enabled():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _rebuild_stack(
        self, flags: PlasticityFlags, faculties: CognitionFaculties | None = None
    ) -> None:
        """Rebuild the stack (+ encoders + optimizer) to a different architecture.

        Used when a checkpoint was saved with different A/B/C flags or cognitive
        faculties than the current build: growable ceilings, the perception-feedback
        modules, and the slot/agency modules all change tensor shapes, so the stack
        must be reconstructed to match the saved ``state_dict`` before loading. A
        changed ``encoder_mode`` also rebuilds the frozen encoders (their submodule
        set differs between the hf and zeros paths).
        """
        self.flags = flags
        if faculties is not None and faculties.encoder_mode != self.faculties.encoder_mode:
            self.encoders = FrozenSensoryEncoders(
                mode=faculties.encoder_mode,
                device=self.device,
                proprio_dim_out=self.cfg.proprio_emb,
            ).to(self.device)
        if faculties is not None:
            self.faculties = faculties
        self.stack = NeuralCognitiveStack(self.cfg, flags, self.faculties).to(self.device)
        self.plasticity_state = (
            PlasticityRuntimeState.from_flags(flags) if self.stack.has_plastic else None
        )
        params = list(self.stack.parameters()) + list(self.encoders.parameters())
        lr = float(self.optimizer.param_groups[0]["lr"])
        from decadic.config import memory_efficient_training_enabled

        self.optimizer, _ = build_optimizer(
            params,
            lr=lr,
            device=self.device,
            memory_efficient=memory_efficient_training_enabled(),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "preset": self.preset,
            "stack": self.stack.state_dict(),
            "encoders": self.encoders.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "faculties": asdict(self.faculties),
            "plasticity_flags": asdict(self.flags),
            "plasticity_state": (
                asdict(self.plasticity_state) if self.plasticity_state is not None else None
            ),
            "plastic_arch_meta": (
                self.stack.plastic_arch_meta() if self.stack.has_plastic else None
            ),
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        saved_preset = payload.get("preset")
        if saved_preset is not None and saved_preset != self.preset:
            raise ValueError(
                f"checkpoint preset {saved_preset!r} != current preset {self.preset!r}; "
                "refusing to load mismatched architecture"
            )
        # Rebuild the stack to the checkpoint's architecture if it differs
        # (growable ceilings, the perception-feedback modules, and the slot/agency
        # modules all change tensor shapes); awake/mask/hebb buffers are then
        # restored from the state_dict. Faculties + plasticity flags are resolved
        # together so a single rebuild covers both.
        saved_flags_d = payload.get("plasticity_flags")
        saved_flags = (
            PlasticityFlags(**saved_flags_d) if saved_flags_d is not None else self.flags
        )
        saved_fac_d = payload.get("faculties")
        saved_fac = (
            CognitionFaculties(**saved_fac_d) if saved_fac_d is not None else self.faculties
        )
        if saved_flags != self.flags or saved_fac != self.faculties:
            self._rebuild_stack(saved_flags, faculties=saved_fac)
        # Drop checkpoint tensors whose shape no longer matches the live stack so
        # they keep their fresh initialization instead of crashing the load.
        # ``strict=False`` ignores missing/extra keys but still raises on a size
        # mismatch, so e.g. changing the body's actuator count (motor + forward
        # heads resize) must reinitialize exactly those heads while the rest of
        # the cognitive stack is restored verbatim.
        incoming = payload["stack"]
        own = self.stack.state_dict()
        compatible = {
            k: v
            for k, v in incoming.items()
            if k in own and getattr(v, "shape", None) == own[k].shape
        }
        reinitialized = sorted(k for k in incoming if k in own and k not in compatible)
        if reinitialized:
            logger.warning(
                "checkpoint tensors reinitialized (shape changed, kept fresh): %s",
                reinitialized,
            )
        self.stack.load_state_dict(compatible, strict=False)
        # Shape-filtered like the stack: an encoder-mode change (hf<->zeros) adds or
        # drops the frozen CLIP/Whisper submodules, so only the trainable proprio
        # MLP carries over; frozen weights keep their (deterministic) pretrained
        # init. Avoids a strict-load crash when restoring across modes.
        enc_incoming = payload["encoders"]
        enc_own = self.encoders.state_dict()
        enc_compatible = {
            k: v
            for k, v in enc_incoming.items()
            if k in enc_own and getattr(v, "shape", None) == enc_own[k].shape
        }
        self.encoders.load_state_dict(enc_compatible, strict=False)
        if "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, KeyError) as exc:  # structural drift; start optimizer fresh
                logger.warning("optimizer state incompatible (%s); reinitializing moments", exc)
        ps = payload.get("plasticity_state")
        if ps and self.plasticity_state is not None:
            for k, v in ps.items():
                if hasattr(self.plasticity_state, k):
                    setattr(self.plasticity_state, k, v)
