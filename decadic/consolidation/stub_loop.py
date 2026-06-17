"""Phase 1 consolidation placeholder — background clock with hooks for Phase 3 replay."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)


async def consolidation_stub_loop(
    agent_id: str,
    *,
    should_continue: Callable[[], bool],
    on_tick: Callable[[], None] | None = None,
) -> None:
    """No-op consolidation heartbeat; extend with replay sampling in Phase 3."""
    interval = float(os.environ.get("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "10"))
    if interval <= 0:
        return
    while should_continue():
        await asyncio.sleep(interval)
        if not should_continue():
            break
        if on_tick is not None:
            on_tick()
        logger.info(
            "consolidation_stub_tick agent_id=%s interval_s=%.3f",
            agent_id,
            interval,
        )
