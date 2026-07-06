"""AudioIntake -- the continuous auditory intake organ (WS6-M0.5, PRD 3.8, G9).

Capture is continuous; consumption stays cycle-quantized (the cochlea never
sleeps; the cycle samples it). A background input stream (system microphone in
"mic" mode) and/or the Rig-1 mixing-bus tap (``mix_in``, both modes) fill a
rolling ring buffer; at observation ingest the runtime attaches everything
since the last read (capped ~250 ms, drop-oldest on overrun -- the perception
pipeline's own overload semantics) to ``obs.audio`` in EXACTLY the schema the
frozen Whisper path already decodes (pcm16 base64 + sample_rate; see
``decadic.nn.frozen_encoders._waveform_from_obs``), so hearing becomes a
property of the body without touching the encoder.

Precedence: client-supplied audio ALWAYS wins; the intake only fills silence.
Recorded scenarios therefore stay byte-reproducible while live rooms become
audible.

One microphone per process: agents share the module-level singleton via
``get_audio_intake()``.

sounddevice is an OPTIONAL dependency: when it is missing (or no device
exists) mic capture degrades to inert with a single log line -- the bus tap
(``mix_in``) keeps working, so the self-hearing loopback survives on a box
with no audio hardware at all. Every code path here is additive-never-fatal:
the cognitive loop must never die because a microphone hiccuped.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INTAKE_SAMPLE_RATE = 16000
# Ring capacity: ~4 s absorbs long cycle stalls (checkpoint saves, GC) without
# unbounded memory; anything older than the ring is the same audio the
# perception pipeline would have dropped under overload anyway.
RING_SECONDS = 4.0
# Per-observation attach cap (PRD 3.8): the integration window binds ~250 ms
# into one "now"; longer chunks would smear multiple moments together.
DEFAULT_MAX_CHUNK_MS = 250


def _silence_rms_threshold() -> float:
    """Energy floor below which a chunk is 'the quiet room' (skip Whisper)."""
    try:
        return max(0.0, float(os.environ.get("DECADIC_AUDIO_SILENCE_RMS", "0.01")))
    except ValueError:
        return 0.01


class AudioIntake:
    """Thread-safe ring buffer fed by mic callback and/or bus ``mix_in``."""

    def __init__(
        self,
        mode: str | None = None,
        sample_rate: int = INTAKE_SAMPLE_RATE,
        capacity_s: float = RING_SECONDS,
    ) -> None:
        if mode is None:
            from decadic import config as _cfg

            mode = _cfg.audio_intake_mode()
        self.mode = mode if mode in ("off", "mic", "bus") else "off"
        self.sample_rate = int(sample_rate)
        self._capacity = max(1, int(capacity_s * self.sample_rate))
        self._ring = np.zeros(self._capacity, dtype=np.float32)
        self._lock = threading.Lock()
        # Absolute sample counters (never wrap in practice: 2**63 samples at
        # 16 kHz outlives the machine); ring position = counter % capacity.
        self._written = 0
        self._read = 0
        self._started = False
        self._running = False
        self._device_active = False
        self._stream: Any = None
        # Silence-gate hysteresis: True after a loud chunk so the immediately
        # following quiet chunk (a word's decaying tail) still attaches.
        self._gate_open = False
        self._chunks_attached = 0
        self._silence_skips = 0
        self._mix_ins = 0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Idempotent: open the mic stream (mic mode) / arm the bus (bus mode)."""
        with self._lock:
            if self._started:
                return
            self._started = True
            if self.mode == "off":
                return
            self._running = True
        if self.mode != "mic":
            return
        # Mic capture is best-effort: no sounddevice / no device => inert mic,
        # but the intake stays running so the bus tap (loopback) still works.
        try:
            import sounddevice as sd
        except Exception:
            logger.warning(
                "audio_intake: sounddevice not installed; mic capture inert "
                "(bus tap / loopback still live). Install sounddevice for mic mode."
            )
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._mic_callback,
            )
            self._stream.start()
            self._device_active = True
        except Exception:
            self._stream = None
            logger.warning(
                "audio_intake: no usable input device; mic capture inert "
                "(bus tap / loopback still live)."
            )

    def stop(self) -> None:
        """Idempotent: close the device stream and mark not-running."""
        stream = None
        with self._lock:
            stream = self._stream
            self._stream = None
            self._started = False
            self._running = False
            self._device_active = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass  # a dying device must not take the process with it

    def _mic_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # sounddevice callback thread: keep it allocation-light and non-fatal.
        try:
            self._push(np.asarray(indata, dtype=np.float32).reshape(-1, indata.shape[-1])[:, 0])
        except Exception:
            pass  # a malformed device buffer is dropped, never raised

    # ------------------------------------------------------------------ ring

    def _push(self, wav: np.ndarray) -> None:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return
        if wav.size >= self._capacity:
            wav = wav[-self._capacity :]  # only the freshest ring-full survives
        with self._lock:
            pos = self._written % self._capacity
            first = min(wav.size, self._capacity - pos)
            self._ring[pos : pos + first] = wav[:first]
            if first < wav.size:
                self._ring[: wav.size - first] = wav[first:]
            self._written += wav.size
            # Drop-oldest: the read cursor may never lag more than one ring.
            if self._written - self._read > self._capacity:
                self._read = self._written - self._capacity

    def mix_in(self, waveform) -> None:
        """Bus tap: world mixes / loopback self-voice enter here (both modes)."""
        if self.mode == "off":
            return
        self.start()  # lazy-arm so the first loopback frame is never lost
        wav = np.clip(
            np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1)),
            -1.0,
            1.0,
        )
        if wav.size == 0:
            return
        self._push(wav)
        with self._lock:
            self._mix_ins += 1

    def read_chunk(self, max_ms: int = DEFAULT_MAX_CHUNK_MS) -> np.ndarray | None:
        """Everything since the last read, capped at max_ms (freshest wins).

        Advances the read cursor: a second read with no new audio returns
        ``None``. On overrun the OLDEST samples beyond the cap are dropped --
        the perception pipeline's own overload semantics.
        """
        cap = max(1, int(self.sample_rate * max(1, int(max_ms)) / 1000))
        with self._lock:
            available = self._written - self._read
            if available <= 0:
                return None
            if available > cap:
                self._read = self._written - cap  # drop-oldest on overrun
                available = cap
            start = self._read
            self._read = self._written
            pos = start % self._capacity
            first = min(available, self._capacity - pos)
            out = np.empty(available, dtype=np.float32)
            out[:first] = self._ring[pos : pos + first]
            if first < available:
                out[first:] = self._ring[: available - first]
            return out

    # ---------------------------------------------------------------- attach

    def attach_to_obs(self, obs: dict) -> None:
        """Fill ``obs.audio`` with the freshest chunk IF the client sent none.

        Client audio wins (recorded scenarios stay byte-reproducible). The
        silence gate skips quiet-room chunks (they would tax Whisper for
        nothing) but with hysteresis: the first quiet chunk AFTER a loud one
        still attaches, so word tails are not clipped mid-decay.
        """
        if not isinstance(obs, dict):
            return
        existing = obs.get("audio")
        if (
            isinstance(existing, dict)
            and isinstance(existing.get("data"), str)
            and existing["data"].strip()
        ):
            return  # client audio wins
        self.start()
        chunk = self.read_chunk()
        if chunk is None or chunk.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        loud = rms >= _silence_rms_threshold()
        with self._lock:
            attach = loud or self._gate_open
            self._gate_open = loud
            if not attach:
                self._silence_skips += 1
                return
            self._chunks_attached += 1
        # EXACTLY the schema _waveform_from_obs decodes: little-endian pcm16
        # base64 + sample_rate. "source" is extra telemetry (decoder ignores it).
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2")
        obs["audio"] = {
            "data": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "encoding": "pcm16_base64",
            "sample_rate": self.sample_rate,
            "source": "intake",
        }

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "running": bool(self._running),
                "device_active": bool(self._device_active),
                "chunks_attached": int(self._chunks_attached),
                "silence_skips": int(self._silence_skips),
                "mix_ins": int(self._mix_ins),
            }


# One microphone per process; agents share it. Created lazily on first use so
# merely importing decadic never opens a device.
_intake: AudioIntake | None = None
_intake_guard = threading.Lock()


def get_audio_intake() -> AudioIntake:
    """Process-wide intake singleton (mode resolved from env at first use)."""
    global _intake
    with _intake_guard:
        if _intake is None:
            _intake = AudioIntake()
        return _intake


def reset_audio_intake() -> None:
    """Tear down the singleton (tests / mode changes); next use rebuilds it."""
    global _intake
    with _intake_guard:
        old, _intake = _intake, None
    if old is not None:
        old.stop()
