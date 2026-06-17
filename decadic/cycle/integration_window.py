"""Explicit temporal-integration window: bind a span of percepts into one "now".

Self-model program, Phase 3. Today each cognitive cycle treats the freshest
percept as the present moment; there is a recency pool that *averages* recent
percepts, but it never *commits* — there is no discrete "this is now". Conscious
experience, by contrast, integrates sensory input over a window (~tens to a few
hundred ms) and commits a single bound moment that becomes the present.

This module accumulates percepts over a wall-clock window (or a frame cap) and,
when the window closes, binds the buffered percepts into ONE committed latent.
Between commits the agent acts on the LAST committed moment (perception is held),
so manipulating the window length measurably shifts *when* "now" updates (the
P1.x signature assay). Binding here is a parameter-free mean pool; the committed
latent then flows through the stack's stage-1/stage-2 framing as usual.

Off-branch parity: ``window_ms <= 0`` commits every percept immediately, exactly
reproducing today's behaviour (the freshest percept is "now").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowResult:
    """Outcome of pushing one percept into the window."""

    committed: np.ndarray | None  # the bound "now" latent when the window just closed, else None
    buffered: int  # frames currently held in the (possibly reopened) window
    closed: bool  # did this push close the window and commit a moment?


class IntegrationWindow:
    """Accumulate percepts over a window and commit one bound moment on close."""

    def __init__(
        self,
        *,
        window_ms: float = 0.0,
        max_frames: int = 8,
        mode: str = "mean",
    ) -> None:
        self.window_ms = float(window_ms)
        self.max_frames = max(1, int(max_frames))
        self.mode = mode
        self._buf: list[np.ndarray] = []
        self._start_s: float | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._start_s = None

    def _bind(self, frames: list[np.ndarray]) -> np.ndarray:
        """Bind buffered percepts into one committed latent (parameter-free)."""
        stacked = np.stack(frames, axis=0)
        if self.mode == "last":
            return stacked[-1].astype(np.float64)
        return stacked.mean(axis=0).astype(np.float64)

    def push(self, percept, now_s: float) -> WindowResult:
        """Add ``percept`` (1-D) to the window; commit a bound moment when it closes.

        ``window_ms <= 0`` is a pass-through: the percept is committed immediately
        (today's behaviour). Otherwise the window closes when the elapsed wall time
        reaches ``window_ms`` or the frame count reaches ``max_frames``; on close
        the buffered percepts are bound into one committed latent and the window
        reopens empty.
        """
        p = np.asarray(percept, dtype=np.float64).reshape(-1)
        if self.window_ms <= 0.0:
            return WindowResult(committed=p, buffered=0, closed=True)
        if not self._buf:
            self._start_s = float(now_s)
        self._buf.append(p)
        start = self._start_s if self._start_s is not None else float(now_s)
        elapsed_ms = (float(now_s) - start) * 1000.0
        closed = elapsed_ms >= self.window_ms or len(self._buf) >= self.max_frames
        if closed:
            committed = self._bind(self._buf)
            self.reset()
            return WindowResult(committed=committed, buffered=0, closed=True)
        return WindowResult(committed=None, buffered=len(self._buf), closed=False)
