"""Frozen CLIP + Whisper encoders with a zero/synthetic fallback (no HF download)."""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from decadic.config import encoder_autocast_dtype
from decadic.state.body_map import (
    BODY_MAP_VECTOR_DIM,
    body_pain_vector,
    effort_vector,
    flatten_body_map,
)

logger = logging.getLogger(__name__)

CLIP_POOL_DIM = 512
WHISPER_POOL_DIM = 768
PROPRIO_BASE_DIM = 10  # position(3) + orientation(3) + velocity(3) + action hash(1)
PROPRIO_BODY_MAP_DIM = BODY_MAP_VECTOR_DIM
# Spatial patch features for object-centric slot attention (CLIP ViT-B/32 @ 224
# -> 7x7=49 patch tokens of width 768). Fixed here so a CLIP swap can't change
# the slot module's input shape (tokens are fit to this grid/width, like _fit_dim).
CLIP_PATCH_DIM = 768
CLIP_PATCH_GRID = 7
CLIP_N_PATCHES = CLIP_PATCH_GRID * CLIP_PATCH_GRID


def _ffloat(v: Any) -> float:
    """Coerce to a finite float; NaN/Inf/garbage -> 0.0.

    The body can stream non-finite proprioception (a diverging MuJoCo sim, an
    unnormalized quaternion, a ragdolling corpse), and JSON carries NaN/Infinity
    verbatim. Scrubbing here, at ingestion, stops a single bad frame from reaching
    the forward-model loss or the encoders and poisoning the network weights."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def proprio_joint_cap() -> int:
    """Max joint qpos/qvel values consumed from observation `proprioception.joints`."""
    return max(0, int(os.environ.get("DECADIC_PROPRIO_JOINT_CAP", "64")))


def proprio_contact_cap() -> int:
    """Max touch/contact values consumed from observation `proprioception.contacts`."""
    return max(0, int(os.environ.get("DECADIC_PROPRIO_CONTACT_CAP", "16")))


def proprio_input_dim() -> int:
    return PROPRIO_BASE_DIM + proprio_joint_cap() + proprio_contact_cap() + PROPRIO_BODY_MAP_DIM


def controllable_proprio_vector(obs: dict[str, Any] | None, dim: int) -> list[float]:
    """The body state the agent can change: [roll, pitch, yaw, height, vx, vy, vz] + joint qpos.

    Joints arrive interleaved (qpos, qvel) per hinge; we take the qpos channel.
    Truncated / zero-padded to exactly ``dim`` so it matches the forward-model head.
    """
    from decadic.config import CONTROLLABLE_PROPRIO_BASE

    prop = (obs or {}).get("proprioception") or {}
    ori = prop.get("orientation") or [0.0, 0.0, 0.0]
    pos = prop.get("position") or [0.0, 0.0, 0.0]
    vel = prop.get("velocity") or [0.0, 0.0, 0.0]
    base = [
        _ffloat(ori[0]) if len(ori) > 0 else 0.0,
        _ffloat(ori[1]) if len(ori) > 1 else 0.0,
        _ffloat(ori[2]) if len(ori) > 2 else 0.0,
        _ffloat(pos[2]) if len(pos) > 2 else 0.0,
        _ffloat(vel[0]) if len(vel) > 0 else 0.0,
        _ffloat(vel[1]) if len(vel) > 1 else 0.0,
        _ffloat(vel[2]) if len(vel) > 2 else 0.0,
    ]
    joints_raw = prop.get("joints")
    qpos: list[float] = []
    if isinstance(joints_raw, list):
        for v in joints_raw[0::2]:  # qpos channel of the interleaved (qpos, qvel) pairs
            qpos.append(_ffloat(v))
    vec = base[:CONTROLLABLE_PROPRIO_BASE] + qpos
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def controllable_intero_vector(homeostasis: Any | None, dim: int) -> list[float]:
    """Normalized interoceptive state the agent can influence: [hydration, energy, integrity]/100.

    Mirrors ``controllable_proprio_vector`` for the homeostatic reservoirs. A
    missing homeostasis (stub/test, no body) reads as full reservoirs (1.0), so
    the drive is zero and behavior is unchanged. Truncated/zero-padded to ``dim``.
    """
    if homeostasis is None:
        vec = [1.0, 1.0, 1.0]
    else:
        vec = [
            _ffloat(getattr(homeostasis, "hydration", 100.0)) / 100.0,
            _ffloat(getattr(homeostasis, "energy", 100.0)) / 100.0,
            _ffloat(getattr(homeostasis, "integrity", 100.0)) / 100.0,
        ]
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def controllable_tactile_vector(obs: dict[str, Any] | None, dim: int) -> list[float]:
    """Normalized per-part contact loads the agent can change by how it moves.

    Reads ``proprioception.part_loads`` (an ordered list of soft loads, one per
    touch sensor, force/body-weight). Mirrors ``controllable_proprio_vector``: a
    missing body (stub/test, no contacts) reads as all-zero (nothing touching),
    so the tactile prediction target is well-defined. Truncated/zero-padded to
    exactly ``dim`` so it lines up channel-for-channel with the tactile head.
    """
    prop = (obs or {}).get("proprioception") or {}
    raw = prop.get("part_loads")
    vec: list[float] = []
    if isinstance(raw, list):
        for v in raw:
            vec.append(_ffloat(v))
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def controllable_effort_vector(obs: dict[str, Any] | None, dim: int) -> list[float]:
    """Body-map effort/work/strain/fatigue/pain plus aggregate effort totals."""
    prop = (obs or {}).get("proprioception") or {}
    vec = effort_vector(prop.get("body_map"), prop.get("effort"))
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def controllable_body_pain_vector(obs: dict[str, Any] | None, dim: int) -> list[float]:
    """Localized body pain by deterministic body-map part order."""
    prop = (obs or {}).get("proprioception") or {}
    vec = body_pain_vector(prop.get("body_map"))
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def preferred_intero_vector(dim: int) -> list[float]:
    """The innate interoceptive setpoint: reservoirs full (1.0 == 100%).

    This is the homeostatic prior (what phylogeny fixes), with no reference to any
    external satisfier. The policy learns *how* to approach it from experience.
    """
    vec = [1.0, 1.0, 1.0]
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [1.0] * (dim - len(vec))


def intero_preference_weights(dim: int) -> list[float]:
    """Equal weights across reservoirs so the most-deprived need dominates the
    drive-reduction gradient on its own (emergent prioritization, not hardcoded)."""
    vec = [1.0, 1.0, 1.0]
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _capped_floats(raw: Any, cap: int) -> list[float]:
    """Truncate/zero-pad an observation array to exactly `cap` floats."""
    out = [0.0] * cap
    if isinstance(raw, list):
        for i, v in enumerate(raw[:cap]):
            out[i] = _ffloat(v)
    return out


def _waveform_from_obs(obs: dict[str, Any], max_seconds: float = 8.0) -> np.ndarray | None:
    """Decode ``obs.audio`` (pcm16 base64, 16 kHz mono) → float32 waveform in [-1, 1].

    Returns ``None`` when no usable audio is present so callers can fall back
    to cheap zero embeddings instead of encoding silence.
    """
    audio = obs.get("audio") or {}
    data = audio.get("data")
    if not isinstance(data, str) or not data.strip():
        return None
    encoding = str(audio.get("encoding", "pcm16_base64")).lower()
    if encoding not in ("pcm16_base64", "pcm16"):
        return None
    try:
        blob = base64.b64decode(data)
    except Exception:
        return None
    if len(blob) < 2:
        return None
    wav = np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
    sr = int(audio.get("sample_rate", 16000) or 16000)
    cap = max(1, int(max_seconds * sr))
    return wav[:cap]


class FrozenSensoryEncoders(nn.Module):
    """Vision/audio/proprio → fused vector; CLIP/Whisper frozen when ``mode=='hf'``."""

    def __init__(
        self,
        *,
        mode: str,
        device: torch.device,
        proprio_dim_out: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.device = device
        # Compute dtype for the FROZEN CLIP/Whisper forwards only. bf16 on a
        # bf16-capable CUDA device (Ampere+), fp32 everywhere else -> CPU/tests are
        # byte-identical. The trainable proprio MLP + stack always stay fp32.
        self._compute_dtype = (
            encoder_autocast_dtype() if device.type == "cuda" else torch.float32
        )
        self.joint_cap = proprio_joint_cap()
        self.contact_cap = proprio_contact_cap()
        self.proprio_in_dim = (
            PROPRIO_BASE_DIM + self.joint_cap + self.contact_cap + PROPRIO_BODY_MAP_DIM
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.proprio_in_dim, proprio_dim_out),
            nn.GELU(),
            nn.Linear(proprio_dim_out, proprio_dim_out),
        ).to(device)
        self._clip_vision: nn.Module | None = None
        self._whisper_encoder: nn.Module | None = None
        self._clip_processor = None
        self._whisper_processor = None
        # Per-observation embedding cache: avoid re-running frozen CLIP/Whisper
        # when several cycles consume the same observation.
        self._cache_key: Any = None
        self._cache_vision: torch.Tensor | None = None
        self._cache_audio: torch.Tensor | None = None
        # Separate cache for the spatial CLIP patch tokens (slot attention). Keyed
        # on the same observation identity so a frame triggers at most one CLIP
        # vision_model forward no matter how many cycles consume it.
        self._cache_patch_key: Any = None
        self._cache_patch: torch.Tensor | None = None
        if mode == "hf":
            self._load_hf()

    def _load_hf(self) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            from transformers import WhisperModel, WhisperProcessor
        except ImportError as e:
            raise RuntimeError("transformers required for DECADIC_ENCODER_MODE=hf") from e

        logger.info("loading frozen CLIP + Whisper (first run may download weights)")
        clip_name = os.environ.get("DECADIC_CLIP_MODEL", "openai/clip-vit-base-patch32")
        whisper_name = os.environ.get("DECADIC_WHISPER_MODEL", "openai/whisper-small")
        self._clip_vision = CLIPVisionModelWithProjection.from_pretrained(clip_name).to(
            self.device
        )
        self._clip_processor = CLIPImageProcessor.from_pretrained(clip_name)
        w = WhisperModel.from_pretrained(whisper_name)
        self._whisper_encoder = w.encoder.to(self.device)
        self._whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
        for p in self._clip_vision.parameters():
            p.requires_grad = False
        for p in self._whisper_encoder.parameters():
            p.requires_grad = False
        self._clip_vision.eval()
        self._whisper_encoder.eval()

    def _encoder_autocast(self):
        """Autocast the frozen encoder forwards to bf16 on capable CUDA; else no-op."""
        if self._compute_dtype == torch.float32 or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._compute_dtype)

    def _fit_dim(self, emb: torch.Tensor, dim: int) -> torch.Tensor:
        """Pad/truncate to the fixed pool dim so model swaps can't break the stack."""
        cur = emb.shape[-1]
        if cur == dim:
            return emb
        if cur > dim:
            return emb[..., :dim]
        pad = torch.zeros(*emb.shape[:-1], dim - cur, device=emb.device, dtype=emb.dtype)
        return torch.cat([emb, pad], dim=-1)

    def _decode_pixel_values(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        """CPU-only: ``obs.vision`` base64 -> PIL -> CLIP ``pixel_values`` (CPU tensor).

        This is the heavy half of vision encoding (base64 decode + RGB convert +
        resize/normalize); it touches no GPU and no shared state, so ``predecode``
        can run it off the cognitive lock. ``None`` when no decodable image present.
        """
        if self._clip_processor is None:
            return None
        from PIL import Image

        vis = (obs or {}).get("vision") or {}
        raw_b64 = vis.get("data")
        if not (isinstance(raw_b64, str) and raw_b64.strip()):
            return None
        try:
            blob = base64.b64decode(raw_b64)
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            inputs = self._clip_processor(images=im, return_tensors="pt")
            return inputs["pixel_values"]  # CPU tensor; .to(device) happens on-cycle
        except Exception:
            return None

    def _decode_audio_features(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        """CPU-only: ``obs.audio`` -> Whisper log-mel ``input_features`` (CPU tensor).

        Mirrors ``_decode_pixel_values`` for the audio path so ``predecode`` can lift
        the feature extraction off the lock. ``None`` when no usable audio present.
        """
        if self._whisper_processor is None:
            return None
        wav = _waveform_from_obs(obs or {})
        if wav is None or wav.size == 0:
            return None
        sr = int(((obs or {}).get("audio") or {}).get("sample_rate", 16000) or 16000)
        try:
            inputs = self._whisper_processor(wav, sampling_rate=sr, return_tensors="pt")
            return inputs.input_features  # CPU tensor; .to(device) happens on-cycle
        except Exception:
            return None

    def predecode(self, obs: dict[str, Any] | None) -> dict[str, Any] | None:
        """Decode the camera frame + audio into encoder-ready CPU tensors ONCE and
        stash them on ``obs['_decoded']``.

        The cognitive cycle base64-decodes + CLIP-preprocesses the same frame two to
        three times (pooled vision, patch tokens) and feature-extracts the audio once,
        all inside ``async with self.lock``. Running that pure CPU->tensor work here --
        off the lock, at observation-arrival time -- removes it from the critical
        section and dedupes the repeated decode (the pooled and patch paths now share
        one ``pixel_values``). Only ``.to(device)`` + the frozen forward stay on-cycle.

        No-op outside HF mode / when the encoders aren't loaded. Returns ``obs`` for
        chaining. Keys are always written (possibly ``None``) so the on-cycle paths
        treat a present ``_decoded`` as authoritative and never re-decode.
        """
        if not isinstance(obs, dict) or self.mode != "hf" or self._clip_vision is None:
            return obs
        cache = obs.get("_decoded")
        if not isinstance(cache, dict):
            cache = {}
            obs["_decoded"] = cache
        if "pixel_values" not in cache:
            cache["pixel_values"] = self._decode_pixel_values(obs)
        if "input_features" not in cache:
            cache["input_features"] = self._decode_audio_features(obs)
        return obs

    def _predecoded_pixel_values(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        """Pre-decoded CLIP ``pixel_values`` if ``predecode`` ran for this obs, else
        decode inline (fallback for tests / callers that skip ``predecode``)."""
        cache = (obs or {}).get("_decoded")
        if isinstance(cache, dict) and "pixel_values" in cache:
            return cache["pixel_values"]  # authoritative (CPU tensor or None)
        return self._decode_pixel_values(obs)

    def _predecoded_audio_features(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        """Pre-decoded Whisper ``input_features`` if available, else decode inline."""
        cache = (obs or {}).get("_decoded")
        if isinstance(cache, dict) and "input_features" in cache:
            return cache["input_features"]  # authoritative (CPU tensor or None)
        return self._decode_audio_features(obs)

    def _vision_embedding_hf(self, obs: dict[str, Any]) -> torch.Tensor:
        assert self._clip_vision is not None and self._clip_processor is not None
        pixel_values = self._predecoded_pixel_values(obs)
        if pixel_values is None:
            return self._vision_zeros()
        try:
            img_tensor = pixel_values.to(self.device)
        except Exception:
            return self._vision_zeros()
        with torch.no_grad(), self._encoder_autocast():
            out = self._clip_vision(pixel_values=img_tensor)
            emb = out.image_embeds
        # Back to fp32 so the downstream cat with the trainable proprio MLP (fp32)
        # has a single dtype and the stack trains in fp32.
        return self._fit_dim(emb.float(), CLIP_POOL_DIM)

    def _fit_patch_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Fit [1, N, D] patch tokens to the fixed [1, CLIP_N_PATCHES, CLIP_PATCH_DIM] grid.

        Width is padded/truncated like ``_fit_dim``; token count is bilinearly
        resampled on the square grid so a different CLIP patch size still yields a
        stable slot-attention input shape.
        """
        tokens = self._fit_dim(tokens, CLIP_PATCH_DIM)
        n = tokens.shape[1]
        if n == CLIP_N_PATCHES:
            return tokens
        g0 = max(1, int(round(n ** 0.5)))
        if g0 * g0 != n:
            # Not a square grid: pad/truncate the token axis as a fallback.
            if n > CLIP_N_PATCHES:
                return tokens[:, :CLIP_N_PATCHES, :]
            pad = torch.zeros(
                1, CLIP_N_PATCHES - n, CLIP_PATCH_DIM, device=tokens.device, dtype=tokens.dtype
            )
            return torch.cat([tokens, pad], dim=1)
        grid = tokens.reshape(1, g0, g0, CLIP_PATCH_DIM).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(
            grid, size=(CLIP_PATCH_GRID, CLIP_PATCH_GRID), mode="bilinear", align_corners=False
        )
        return grid.permute(0, 2, 3, 1).reshape(1, CLIP_N_PATCHES, CLIP_PATCH_DIM)

    def vision_patch_tokens(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        """Spatial CLIP patch features [1, CLIP_N_PATCHES, CLIP_PATCH_DIM] for slot attention.

        Returns ``None`` when not in HF mode or no decodable image is present, so
        callers fall back to the non-spatial (pooled) path. Frozen: no grad. Cached
        per observation (same key as the pooled path) so repeated cycles consuming
        one frame skip the CLIP forward; cached None is honored so a frame without a
        decodable image is not re-decoded each cycle.
        """
        if self.mode != "hf" or self._clip_vision is None or self._clip_processor is None:
            return None
        key = (obs or {}).get("timestamp") or id(obs)
        if key == self._cache_patch_key:
            return self._cache_patch
        tokens = self._compute_patch_tokens(obs)
        self._cache_patch_key = key
        self._cache_patch = tokens
        return tokens

    def _compute_patch_tokens(self, obs: dict[str, Any] | None) -> torch.Tensor | None:
        # Reuse the same pre-decoded pixel_values as the pooled path (dedupe): one
        # base64 decode + CLIP preprocess per frame, not one per encoder path.
        pixel_values = self._predecoded_pixel_values(obs)
        if pixel_values is None:
            return None
        try:
            img_tensor = pixel_values.to(self.device)
            with torch.no_grad(), self._encoder_autocast():
                vm = self._clip_vision.vision_model(pixel_values=img_tensor)
                hs = vm.last_hidden_state  # [1, 1+N, hidden] (CLS + patches)
            tokens = hs[:, 1:, :].float()  # drop CLS -> patch tokens; back to fp32
        except Exception:
            return None
        return self._fit_patch_grid(tokens)

    def _audio_embedding_hf(self, obs: dict[str, Any]) -> torch.Tensor:
        """Whisper-encode the observed waveform; zeros when no audio was sent."""
        assert self._whisper_encoder is not None and self._whisper_processor is not None
        feats_cpu = self._predecoded_audio_features(obs)
        if feats_cpu is None:
            return self._audio_zeros()
        feats = feats_cpu.to(self.device)
        with torch.no_grad(), self._encoder_autocast():
            enc_out = self._whisper_encoder(feats)
            hs = enc_out.last_hidden_state.mean(dim=1)
        return self._fit_dim(hs.float(), WHISPER_POOL_DIM)

    def _vision_zeros(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, CLIP_POOL_DIM, device=self.device)

    def _audio_zeros(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, WHISPER_POOL_DIM, device=self.device)

    def _proprio_vector(self, obs: dict[str, Any]) -> torch.Tensor:
        prop = obs.get("proprioception") or {}
        pos = prop.get("position") or [0.0, 0.0, 0.0]
        ori = prop.get("orientation") or [0.0, 0.0, 0.0]
        vel = prop.get("velocity") or [0.0, 0.0, 0.0]
        ca = prop.get("current_action")
        ca_hash = float(hash(str(ca))) % 997 / 997.0 if ca is not None else 0.0
        vec = [_ffloat(pos[i]) if i < len(pos) else 0.0 for i in range(3)]
        vec += [_ffloat(ori[i]) if i < len(ori) else 0.0 for i in range(3)]
        vec += [_ffloat(vel[i]) if i < len(vel) else 0.0 for i in range(3)]
        vec.append(ca_hash)
        vec += _capped_floats(prop.get("joints"), self.joint_cap)
        vec += _capped_floats(prop.get("contacts"), self.contact_cap)
        vec += flatten_body_map(prop.get("body_map"))
        t = torch.tensor([vec], dtype=torch.float32, device=self.device)
        return self.proprio_mlp(t)

    def forward(self, observation: dict[str, Any] | None) -> torch.Tensor:
        obs = observation or {}
        if self.mode == "hf" and self._clip_vision is not None:
            key = obs.get("timestamp") or id(observation)
            if key == self._cache_key and self._cache_vision is not None:
                v = self._cache_vision
                a = self._cache_audio
            else:
                v = self._vision_embedding_hf(obs)
                a = self._audio_embedding_hf(obs)
                self._cache_key = key
                self._cache_vision = v
                self._cache_audio = a
        else:
            v = self._vision_zeros()
            a = self._audio_zeros()
        p = self._proprio_vector(obs)
        return torch.cat([v, a, p], dim=-1)
