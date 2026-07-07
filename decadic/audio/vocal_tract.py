"""FormantSynth -- the body-side vocal tract (WS6-M2.2, PRD 3.7 Rig 1).

A source-filter larynx for a newborn: pure numpy DSP that renders the stack's
VOICE_DIM articulatory parameters to a mono 16 kHz waveform. The loop, not the
synth quality, is the point (PRD 3.3): this exists so voice_u can become sound
that re-enters obs.audio, closing the self-hearing loop.

Design notes (why, not what):

- Stateless + deterministic. ``render`` is a pure function of
  (prev_params, params, n_samples, sample_rate): identical calls are
  bit-identical (the noise source is seeded from a hash of the inputs), so
  recorded runs replay exactly and the M2.2 determinism acceptance holds
  without threading synth state through checkpoints.

- Click-free frame boundaries WITHOUT filter state. The WBS prescribes biquad
  resonators, but recursive filters carry state across frames, and a stateless
  render with per-frame zero filter state rings at every boundary. Instead the
  voiced source is built additively (harmonics of f0 weighted by a formant
  ENVELOPE -- three resonance bumps, a bandwidth knob, a spectral-tilt knob),
  with the fundamental's phase corrected to land on an integer cycle count at
  the frame edge: every harmonic is ~0 at both ends of every frame, so the
  boundary step is no larger than an ordinary intra-frame sample step. The
  noise source is spectrally shaped by the SAME envelope (rFFT multiply) and
  edge-faded ~2 ms to zero at the frame edges for the same reason. This is the
  same acoustics as parallel bandpass resonators, realized in a form whose
  boundary behavior is provable. (Deviation from the WBS letter, recorded in
  the implementation notes; the acceptance criteria -- determinism and
  click-free interpolation -- are what this shape exists to satisfy.)

- All params LINEARLY INTERPOLATE from prev_params to params across the frame
  (per-sample for f0/energy/voicing, per-frame-midpoint for the formant
  weights, which is safe because harmonic amplitudes only matter away from the
  zero crossings that bracket each frame).

- PHONATION GATE (index 0): silence is a DECISION and the resting state, not
  a tone to suppress. The gate is the larynx's on/off (vocal-fold adduction,
  anatomically separate from pitch and articulation). A zero-init voice head
  emits all-zeros through tanh, so the gate reads 0 -> CLOSED -> the render is
  EXACT digital silence: the newborn is silent, and there is a real path where
  the agent does not speak (it is the default). Vocalizing is an active
  emission that must drive the gate above threshold. The gate param interpolates
  across the frame, so opening/closing is click-free. The old "faint hum at the
  origin" is gone: the babble-exploration curriculum (WS6-M3.1), not a constant
  carrier tone, supplies the efference->sound gradient -- exploration opens the
  gate, and only then does sound (and its learning signal) exist.
- Once phonating, energy = -1 is still EXACT silence (mouth open but no breath);
  energy maps cubically so quiet phonation stays observable for the forward
  model.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

# Frozen interface width of the voice efference channel (PRD 3.2, G3).
# Layout: [phon_gate, f0, energy, voicing, formant1..formant5], each expected
# in [-1, 1] (the stack emits them through tanh). Changing this is a versioned
# PRD amendment, never an accident -- the stack head, the action schema, and
# this synth all share it. (WS6 revision 2026-07-06: added the phonation gate
# at index 0; the newborn hum was an involuntary artifact, not a decision.)
VOICE_DIM = 9
PHON_GATE_IDX = 0


def phonation_threshold() -> float:
    """Gate param above which the larynx is voicing. Default 0.0 => a zero-init
    head (gate=tanh(0)=0) is exactly at the closed boundary: SILENT until the
    decision (or babble exploration) drives the gate strictly positive."""
    try:
        return float(os.environ.get("DECADIC_VOICE_PHONATION_THRESHOLD", "0.0"))
    except ValueError:
        return 0.0


def phonating(params, threshold: float | None = None) -> bool:
    """Is the agent vocalizing this frame? (gate strictly above threshold).

    The runtime consults this to decide whether to render + loop back at all:
    a closed gate means no sound leaves the mouth and none re-enters the ear --
    a genuinely silent cycle."""
    thr = phonation_threshold() if threshold is None else float(threshold)
    try:
        g = float(np.asarray(list(params))[PHON_GATE_IDX])
    except (IndexError, ValueError, TypeError):
        return False
    return g > thr + 1e-6

# f0 maps geometrically over the human-ish phonation range.
_F0_MIN_HZ = 80.0
_F0_MAX_HZ = 400.0
# Peak amplitude at energy=+1; headroom below 1.0 keeps the mixing bus from
# clipping when caregiver audio is summed on top of the agent's own voice.
_AMP_MAX = 0.8
# energy <= -1 + eps is EXACT silence (the mouth fully closed).
_SILENCE_EPS = 1e-4
# Three resonances stand in for the vocal-tract filter: neutral centers, the
# per-formant travel fraction driven by formant1..3, and neutral bandwidths.
_FORMANT_CENTERS = (500.0, 1500.0, 2500.0)
_FORMANT_SPAN = (0.6, 0.5, 0.3)
_FORMANT_BW = (90.0, 140.0, 220.0)
# Noise floor of the formant envelope so no parameter combination yields a
# perfect spectral null (a dead channel would stall imitation learning later).
_ENV_FLOOR = 0.01
# Edge fade for the (aperiodic) noise component, in samples at 16 kHz (~2 ms):
# long enough to zero the frame edges, short enough to be inaudible as tremolo.
_NOISE_EDGE_SAMPLES = 32
_MAX_HARMONICS = 24


def _clip_params(params) -> np.ndarray:
    """Coerce to a float64 VOICE_DIM vector in [-1, 1] (missing tail = neutral 0)."""
    vec = np.zeros(VOICE_DIM, dtype=np.float64)
    arr = np.asarray(list(params)[:VOICE_DIM], dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    vec[: arr.shape[0]] = np.clip(arr, -1.0, 1.0)
    return vec


def _f0_hz(p) -> np.ndarray:
    """Map a [-1, 1] pitch param geometrically onto [~80, ~400] Hz."""
    frac = (np.clip(np.asarray(p, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0
    return _F0_MIN_HZ * (_F0_MAX_HZ / _F0_MIN_HZ) ** frac


def _amplitude(e) -> np.ndarray:
    """Cubic energy->amplitude map: exactly 0 at e=-1, 'very quiet' at e=0."""
    frac = (np.clip(np.asarray(e, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0
    return _AMP_MAX * frac**3


def _formant_envelope(freqs: np.ndarray, formants: np.ndarray) -> np.ndarray:
    """Spectral magnitude of the 'vocal tract' at the given frequencies.

    formants[0..2] shift the three resonance centers, formants[3] scales all
    bandwidths (0.5x .. 2x), formants[4] tilts the spectrum (dark .. bright).
    """
    f = np.asarray(freqs, dtype=np.float64)
    bw_scale = 2.0 ** float(np.clip(formants[3], -1.0, 1.0))
    env = np.zeros_like(f)
    for i, (center, span, bw) in enumerate(
        zip(_FORMANT_CENTERS, _FORMANT_SPAN, _FORMANT_BW)
    ):
        fc = center * (1.0 + span * float(np.clip(formants[i], -1.0, 1.0)))
        sigma = max(20.0, bw * bw_scale)
        env = env + np.exp(-0.5 * ((f - fc) / sigma) ** 2)
    tilt = float(np.clip(formants[4], -1.0, 1.0))
    env = env * (1.0 + f / 1000.0) ** (0.8 * tilt)
    return env + _ENV_FLOOR


def _noise_seed(p0: np.ndarray, p1: np.ndarray, n: int, sr: int) -> int:
    """Deterministic noise seed from the render inputs (identical call =>
    bit-identical output; the aperiodic source is repeatable, not random)."""
    payload = np.concatenate([p0, p1, [float(n), float(sr)]]).tobytes()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


class FormantSynth:
    """Stateless source-filter renderer; callers keep prev_params themselves."""

    def render(
        self,
        prev_params,
        params,
        n_samples: int,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Render one cycle frame: float32 mono in [-1, 1], length n_samples."""
        n = int(n_samples)
        sr = int(sample_rate)
        if n <= 0 or sr <= 0:
            return np.zeros(max(0, n), dtype=np.float32)
        p0 = _clip_params(prev_params)
        p1 = _clip_params(params)
        thr = phonation_threshold()
        # Phonation gate CLOSED across the whole frame => exact digital silence.
        # This is the newborn resting state (zero-init gate = 0 <= threshold)
        # and the "does not speak" path -- reached by default, every cycle,
        # until cognition (or babble) drives the gate open.
        if max(p0[PHON_GATE_IDX], p1[PHON_GATE_IDX]) <= thr + 1e-6:
            return np.zeros(n, dtype=np.float32)
        # Mouth open but no breath across the whole frame => also silence.
        if max(p0[2], p1[2]) <= -1.0 + _SILENCE_EPS:
            return np.zeros(n, dtype=np.float32)

        # Per-sample linear interpolation ramp. It ends exactly on p1 (so the
        # next frame, which starts interpolating FROM p1, is continuous) and
        # starts one sample past p0 (the previous frame already emitted p0).
        t = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
        # Phonation gate as a per-sample amplitude multiplier in [0, 1]: a
        # threshold crossing mid-frame ramps amplitude smoothly from 0, so
        # opening/closing the larynx is click-free (like every other param).
        gate = p0[PHON_GATE_IDX] + (p1[PHON_GATE_IDX] - p0[PHON_GATE_IDX]) * t
        phon = np.clip((gate - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
        f0 = _f0_hz(p0[1] + (p1[1] - p0[1]) * t)
        amp = _amplitude(p0[2] + (p1[2] - p0[2]) * t) * phon
        voiced_mix = (np.clip(p0[3] + (p1[3] - p0[3]) * t, -1.0, 1.0) + 1.0) / 2.0

        # Voiced source: harmonics of the (interpolated) fundamental. The
        # accumulated phase is rescaled so the frame ends on an INTEGER cycle
        # count: every harmonic is ~0 at both frame edges, which is what makes
        # stateless per-frame rendering click-free at the boundaries.
        dt = 1.0 / float(sr)
        cycles = np.cumsum(f0) * dt
        total = float(cycles[-1])
        target = max(1.0, float(np.round(total)))
        cycles = cycles * (target / max(total, 1e-9))
        phase = 2.0 * np.pi * cycles

        f0_mid = float(_f0_hz(0.5 * (p0[1] + p1[1])))
        formants_mid = 0.5 * (p0[4:] + p1[4:])
        n_h = int(max(1, min(_MAX_HARMONICS, np.floor(0.45 * sr / f0_mid))))
        harmonics = np.arange(1, n_h + 1, dtype=np.float64)
        weights = _formant_envelope(harmonics * f0_mid, formants_mid) / harmonics
        weights = weights / max(1e-9, float(np.sqrt(np.sum(weights**2))))
        voiced = np.sin(phase[:, None] * harmonics[None, :]) @ weights

        # Aperiodic source: deterministic noise shaped by the same envelope in
        # the frequency domain, then edge-faded to zero so frame boundaries
        # stay click-free without carrying filter state between frames.
        rng = np.random.default_rng(_noise_seed(p0, p1, n, sr))
        noise = rng.standard_normal(n)
        spectrum = np.fft.rfft(noise)
        spectrum = spectrum * _formant_envelope(np.fft.rfftfreq(n, dt), formants_mid)
        noise = np.fft.irfft(spectrum, n)
        noise = noise / max(1e-9, float(np.sqrt(np.mean(noise**2))))
        noise = noise * 0.5  # noise sits below the voiced source at equal mix
        edge = int(min(_NOISE_EDGE_SAMPLES, max(1, n // 4)))
        fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge))
        envelope = np.ones(n, dtype=np.float64)
        envelope[:edge] = fade
        envelope[n - edge :] = fade[::-1]
        noise = noise * envelope

        out = amp * (voiced_mix * voiced + (1.0 - voiced_mix) * noise)
        return np.clip(out, -1.0, 1.0).astype(np.float32)
