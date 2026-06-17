"""Optimizer construction with an optional memory-efficient (8-bit Adam) path.

Self-model program, Phase 6 (hardware-gated). The per-cycle training step holds,
per parameter, the fp32 weight + grad + two fp32 Adam moments (~16 bytes/param);
the moments alone are the single largest training cost and what makes the heavy
presets (250m/500m/1b) OOM on a consumer GPU.

When ``DECADIC_MEMORY_EFFICIENT_TRAINING`` is on and ``bitsandbytes`` is importable
on CUDA, this builds ``bnb.optim.Adam8bit`` (block-wise 8-bit moments, ~4x smaller
optimizer state) instead of fp32 Adam. It falls back *silently* to fp32 Adam when
the flag is off, when bitsandbytes is not installed, or when the device is not
CUDA -- so the default path is byte-identical to before and CPU/test runs are
unaffected. Returns ``(optimizer, kind)`` where ``kind`` is ``"adam8bit"`` or
``"adam"`` for logging/telemetry.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    device: torch.device | str | None = None,
    memory_efficient: bool = False,
) -> tuple[torch.optim.Optimizer, str]:
    """Build the training optimizer; 8-bit Adam when requested + available on CUDA."""
    param_list = list(params)
    dev = torch.device(device) if device is not None else None
    is_cuda = dev is not None and dev.type == "cuda"
    if memory_efficient and is_cuda:
        try:
            import bitsandbytes as bnb  # type: ignore

            opt = bnb.optim.Adam8bit(param_list, lr=lr)
            logger.info("memory-efficient training: using bitsandbytes Adam8bit (lr=%s)", lr)
            return opt, "adam8bit"
        except Exception as exc:  # not installed / unsupported build -> fall back
            logger.warning(
                "memory-efficient training requested but 8-bit Adam unavailable (%s); "
                "falling back to fp32 Adam",
                exc,
            )
    elif memory_efficient and not is_cuda:
        logger.info("memory-efficient training requested on non-CUDA device; using fp32 Adam")
    return torch.optim.Adam(param_list, lr=lr), "adam"
