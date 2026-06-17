"""Decode-once / pre-decode: lift the camera-frame decode off the cognitive lock.

``predecode`` decodes the frame + audio into encoder-ready CPU tensors a single time and
stashes them on ``obs['_decoded']``; the pooled-vision, patch-token, and audio paths then
reuse the stash instead of base64-decoding + CLIP/Whisper-preprocessing the same frame
two-to-three times inside the cycle. Guarantees:
- It is a strict no-op in ``zeros`` mode (no stash added) -> the deterministic CPU
  baseline is byte-identical.
- On a CUDA GPU with CLIP/Whisper, the pre-decoded path produces output identical to the
  inline decode (the dedupe changes *where/how often* we decode, never the numbers).

The HF/GPU parity test skips without CUDA so CPU-only CI stays green and never downloads
CLIP/Whisper.
"""

from __future__ import annotations

import base64
import io

import pytest
import torch

from decadic.nn.frozen_encoders import (
    CLIP_N_PATCHES,
    CLIP_PATCH_DIM,
    FrozenSensoryEncoders,
)

_CUDA = torch.cuda.is_available()
_PROPRIO_OUT = 64


def _png_b64(color=(40, 160, 90)) -> str:
    from PIL import Image

    im = Image.new("RGB", (72, 72), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_predecode_is_noop_in_zeros_mode():
    enc = FrozenSensoryEncoders(
        mode="zeros", device=torch.device("cpu"), proprio_dim_out=_PROPRIO_OUT
    )
    obs = {"timestamp": "t0", "vision": {"data": _png_b64()}}
    out = enc.predecode(obs)
    assert out is obs
    assert "_decoded" not in obs  # zeros mode never decodes -> nothing stashed


@pytest.mark.skipif(not _CUDA, reason="pre-decode parity needs CLIP/Whisper on a CUDA GPU")
def test_predecode_stashes_and_matches_inline_decode():
    enc = FrozenSensoryEncoders(
        mode="hf", device=torch.device("cuda"), proprio_dim_out=_PROPRIO_OUT
    )
    if enc._clip_vision is None:
        pytest.skip("CLIP weights unavailable")
    b64 = _png_b64()

    # Inline decode (the pre-change path): no _decoded stash present.
    inline = {"timestamp": "inline", "vision": {"data": b64}}
    v_inline = enc._vision_embedding_hf(inline)
    enc._cache_patch_key = None
    p_inline = enc._compute_patch_tokens(inline)

    # Pre-decoded path: predecode stashes CPU tensors that the encoders reuse.
    pre = {"timestamp": "pre", "vision": {"data": b64}}
    enc.predecode(pre)
    assert isinstance(pre["_decoded"]["pixel_values"], torch.Tensor)
    assert pre["_decoded"]["pixel_values"].device.type == "cpu"  # decode stays on CPU
    v_pre = enc._vision_embedding_hf(pre)
    p_pre = enc._compute_patch_tokens(pre)

    assert torch.equal(v_inline, v_pre)  # pooled vision: byte-identical
    assert p_pre is not None and tuple(p_pre.shape) == (1, CLIP_N_PATCHES, CLIP_PATCH_DIM)
    assert torch.equal(p_inline, p_pre)  # patch tokens: byte-identical


def test_predecode_no_image_stashes_none_and_yields_zeros():
    enc = FrozenSensoryEncoders(
        mode="zeros", device=torch.device("cpu"), proprio_dim_out=_PROPRIO_OUT
    )
    # zeros mode: predecode is a no-op, and the pooled path falls back to zeros.
    obs = {"timestamp": "noimg", "vision": {}}
    enc.predecode(obs)
    assert "_decoded" not in obs
    v = enc(obs)
    assert bool(torch.isfinite(v).all())
