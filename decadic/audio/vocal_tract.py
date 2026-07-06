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

- All-zero params (a zero-init voice head through tanh) map to NEAR-silence,
  not full loudness: the energy mapping is cubic in (energy+1)/2, so energy=0
  gives (0.5)^3 = 12.5% of max amplitude -- a quiet breathy hum. Why: the
  newborn must not scream at birth (zero-init = silence-ish), but a strictly
  zero output would give babble learning a dead gradient plateau at the
  origin; a faint hum keeps the efference->sound contingency observable from
  the very first cycle. energy = -1 is EXACT digital silence.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Frozen interface width of the voice efference channel (PRD 3.2, G3).
# Layout: [f0, energy, voicing, formant1..formant5], each expected in [-1, 1]
# (the stack emits them through tanh). Changing this is a versioned PRD
# amendment, never an accident -- the stack head, the action schema, and this
# synth all share it.
VOICE_DIM = 8

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
        # Mouth fully closed across the whole frame => exact digital silence.
        if max(p0[1], p1[1]) <= -1.0 + _SILENCE_EPS:
            return np.zeros(n, dtype=np.float32)

        # Per-sample linear interpolation ramp. It ends exactly on p1 (so the
        # next frame, which starts interpolating FROM p1, is continuous) and
        # starts one sample past p0 (the previous frame already emitted p0).
        t = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
        f0 = _f0_hz(p0[0] + (p1[0] - p0[0]) * t)
        amp = _amplitude(p0[1] + (p1[1] - p0[1]) * t)
        voiced_mix = (np.clip(p0[2] + (p1[2] - p0[2]) * t, -1.0, 1.0) + 1.0) / 2.0

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

        f0_mid = float(_f0_hz(0.5 * (p0[0] + p1[0])))
        formants_mid = 0.5 * (p0[3:] + p1[3:])
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
