"""Encoder precision / GPU-perf paths.

Guarantees:
- CPU stays fp32 regardless of ``DECADIC_ENCODER_PRECISION`` -> the deterministic
  CPU test baseline is byte-identical (the no-op parity the plan promised).
- On a CUDA GPU the frozen CLIP/Whisper forwards autocast to bf16 and the result is
  cast back to fp32 (finite, right shape) so the trainable stack stays fp32.
- The CLIP patch-token cache returns the identical grid for a repeated frame and
  recomputes for a new one (the redundant-forward elimination).

The GPU tests skip without CUDA so CPU-only CI stays green and never downloads
CLIP/Whisper.
"""

from __future__ import annotations

import base64
import contextlib
import io

import pytest
import torch

from decadic.config import encoder_autocast_dtype
from decadic.nn.frozen_encoders import (
    CLIP_N_PATCHES,
    CLIP_PATCH_DIM,
    CLIP_POOL_DIM,
    WHISPER_POOL_DIM,
    FrozenSensoryEncoders,
)

_CUDA = torch.cuda.is_available()
_PROPRIO_OUT = 64
_FUSED_DIM = CLIP_POOL_DIM + WHISPER_POOL_DIM + _PROPRIO_OUT


def _png_b64(color=(120, 30, 200)) -> str:
    from PIL import Image

    im = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _obs(ts: str, *, image: bool = True) -> dict:
    o: dict = {
        "timestamp": ts,
        "proprioception": {
            "position": [0.0, 0.0, 1.2],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "idle",
            "joints": [0.0] * 34,
            "contacts": [0, 0, 0, 0],
        },
    }
    if image:
        o["vision"] = {"data": _png_b64()}
    return o


def test_encoder_autocast_dtype_explicit(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "fp32")
    assert encoder_autocast_dtype() is torch.float32
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "bf16")
    assert encoder_autocast_dtype() is torch.bfloat16
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "fp16")
    assert encoder_autocast_dtype() is torch.float16


def test_cpu_encoder_is_fp32_regardless_of_flag(monkeypatch):
    # Even when the precision flag asks for bf16, a CPU device must resolve to fp32
    # and the autocast wrapper must be a no-op -> CPU parity is unconditional.
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "bf16")
    enc = FrozenSensoryEncoders(
        mode="zeros", device=torch.device("cpu"), proprio_dim_out=_PROPRIO_OUT
    )
    assert enc._compute_dtype is torch.float32
    assert isinstance(enc._encoder_autocast(), contextlib.nullcontext)
    v = enc(_obs("t0", image=False))
    assert v.dtype is torch.float32
    assert bool(torch.isfinite(v).all())
    assert v.shape[-1] == _FUSED_DIM


@pytest.mark.skipif(not _CUDA, reason="bf16 encoder path needs a CUDA GPU")
def test_gpu_bf16_encoder_returns_finite_fp32(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "auto")
    enc = FrozenSensoryEncoders(
        mode="hf", device=torch.device("cuda"), proprio_dim_out=_PROPRIO_OUT
    )
    if enc._clip_vision is None or enc._whisper_encoder is None:
        pytest.skip("CLIP/Whisper weights unavailable")
    assert enc._compute_dtype is torch.bfloat16
    v = enc(_obs("t0"))
    assert v.dtype is torch.float32  # cast back to fp32 before the cat
    assert bool(torch.isfinite(v).all())
    assert v.shape[-1] == _FUSED_DIM


@pytest.mark.skipif(not _CUDA, reason="patch-token path needs CLIP on a CUDA GPU")
def test_patch_token_cache_skips_redundant_forward(monkeypatch):
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "auto")
    enc = FrozenSensoryEncoders(
        mode="hf", device=torch.device("cuda"), proprio_dim_out=_PROPRIO_OUT
    )
    if enc._clip_vision is None:
        pytest.skip("CLIP weights unavailable")
    o1 = _obs("frame-A")
    first = enc.vision_patch_tokens(o1)
    cached = enc.vision_patch_tokens(o1)  # same frame -> cache hit
    assert first is not None
    assert cached is first  # identical object: no recompute
    assert first.dtype is torch.float32
    assert tuple(first.shape) == (1, CLIP_N_PATCHES, CLIP_PATCH_DIM)
    assert bool(torch.isfinite(first).all())
    other = enc.vision_patch_tokens(_obs("frame-B"))  # new frame -> recompute
    assert other is not first
    # A frame with no decodable image caches None (not re-decoded each cycle).
    assert enc.vision_patch_tokens(_obs("frame-C", image=False)) is None
    assert enc.vision_patch_tokens(_obs("frame-C", image=False)) is None
