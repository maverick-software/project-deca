"""VoicePlayback -- optional monitor tee to the operator's speakers (PRD 3.7).

Playback is FOR THE OPERATOR, not for the agent: the agent's self-hearing is
the intake loopback (``AudioIntake.mix_in``), which happens whether or not any
speaker exists. This module only decides whether the room also hears it.

sounddevice is optional: without it (or without an output device) ``play``
degrades to a counted no-op after one log line. A callback-driven OutputStream
drains a small ring buffer so ``play`` never blocks the cognitive loop; on
underrun the callback emits silence, on overrun the oldest samples drop --
late audio is worse than lost audio for a live monitor.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Small ring: ~2 s of monitor audio. The synth produces ~one cycle-frame per
# cycle, so anything deeper only adds latency between the agent and the room.
PLAYBACK_RING_SECONDS = 2.0


class VoicePlayback:
    """Non-blocking speaker sink with graceful no-device degradation."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = int(sample_rate)
        self._capacity = max(1, int(PLAYBACK_RING_SECONDS * self.sample_rate))
        self._ring = np.zeros(self._capacity, dtype=np.float32)
        self._lock = threading.Lock()
        self._written = 0
        self._read = 0
        self._stream: Any = None
        self._device_active = False
        self._open_attempted = False
        self._plays = 0
        self._dropped = 0
        self._underruns = 0

    def _ensure_stream(self) -> None:
        """Open the output stream once, lazily; failure degrades to no-op."""
        if self._open_attempted:
            return
        self._open_attempted = True
        try:
            import sounddevice as sd
        except Exception:
            logger.warning(
                "voice_playback: sounddevice not installed; playback is a no-op "
                "(the loopback self-hearing path is unaffected)."
            )
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._device_active = True
        except Exception:
            self._stream = None
            logger.warning(
                "voice_playback: no usable output device; playback is a no-op."
            )

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        # Device callback thread: silence on underrun, never an exception.
        try:
            out = np.asarray(outdata)
            out[:] = 0.0
            with self._lock:
                available = min(frames, self._written - self._read)
                if available < frames:
                    self._underruns += 1
                if available <= 0:
                    return
                pos = self._read % self._capacity
                first = min(available, self._capacity - pos)
                out[:first, 0] = self._ring[pos : pos + first]
                if first < available:
                    out[first:available, 0] = self._ring[: available - first]
                self._read += available
        except Exception:
            pass

    def play(self, waveform) -> None:
        """Queue a waveform for the speakers; never blocks, never raises."""
        try:
            self._ensure_stream()
            wav = np.clip(
                np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1)),
                -1.0,
                1.0,
            )
            if wav.size == 0:
                return
            with self._lock:
                self._plays += 1
                if not self._device_active:
                    return  # counted no-op: telemetry still shows the attempt
                if wav.size >= self._capacity:
                    wav = wav[-self._capacity :]
                pos = self._written % self._capacity
                first = min(wav.size, self._capacity - pos)
                self._ring[pos : pos + first] = wav[:first]
                if first < wav.size:
                    self._ring[: wav.size - first] = wav[first:]
                self._written += wav.size
                if self._written - self._read > self._capacity:
                    self._dropped += int(
                        (self._written - self._read) - self._capacity
                    )
                    self._read = self._written - self._capacity
        except Exception:
            logger.debug("voice_playback: play failed", exc_info=True)

    def stop(self) -> None:
        """Idempotent: close the device stream (playback becomes a no-op)."""
        stream = None
        with self._lock:
            stream = self._stream
            self._stream = None
            self._device_active = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device_active": bool(self._device_active),
                "plays": int(self._plays),
                "dropped_samples": int(self._dropped),
                "underruns": int(self._underruns),
            }
