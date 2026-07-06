"""WS6 speech loop: M0.5 continuous auditory intake + M2.1 voice head +
M2.2 formant vocal tract.

House style (test_ws5_binding.py): zero-init parity, content sensitivity,
flag-off ignores. Every test here must pass WITHOUT sounddevice installed and
without a microphone/speaker: mic/playback degrade paths are forced by
poisoning sys.modules, never by touching real devices.
"""

import base64
import sys

import numpy as np
import pytest

from decadic.audio.intake import AudioIntake, get_audio_intake, reset_audio_intake
from decadic.audio.playback import VoicePlayback
from decadic.audio.vocal_tract import VOICE_DIM, FormantSynth


@pytest.fixture(autouse=True)
def _fresh_intake_singleton():
    """The intake singleton is process-wide; keep tests hermetic."""
    reset_audio_intake()
    yield
    reset_audio_intake()


def _sine(n=1600, freq=440.0, amp=0.3, sr=16000):
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ------------------------------------------------------------ FormantSynth


def test_synth_shape_dtype_range():
    synth = FormantSynth()
    params = [0.3, 0.4, 0.8, 0.1, -0.2, 0.0, 0.3, -0.1]
    wav = synth.render([0.0] * VOICE_DIM, params, n_samples=1600)
    assert wav.shape == (1600,) and wav.dtype == np.float32
    assert float(np.abs(wav).max()) <= 1.0
    assert synth.render(params, params, n_samples=0).shape == (0,)


def test_synth_determinism_bit_identical():
    synth = FormantSynth()
    p0 = [0.1, 0.2, -0.3, 0.4, 0.0, -0.5, 0.2, 0.9]
    p1 = [0.2, 0.5, 0.1, -0.4, 0.3, 0.0, -0.2, 0.1]
    a = synth.render(p0, p1, n_samples=1600)
    b = synth.render(p0, p1, n_samples=1600)
    assert np.array_equal(a, b)  # includes the seeded noise path (voicing<1)


def test_synth_exact_silence_at_energy_floor():
    synth = FormantSynth()
    quiet = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    wav = synth.render(quiet, quiet, n_samples=1600)
    assert float(np.abs(wav).sum()) == 0.0  # mouth closed = digital silence


def test_synth_zero_params_near_silent():
    """tanh(zero-init head) = all-zero params: the cubic energy map keeps the
    newborn's output a faint hum (audible contingency, not a scream)."""
    synth = FormantSynth()
    wav = synth.render([0.0] * VOICE_DIM, [0.0] * VOICE_DIM, n_samples=1600)
    rms = float(np.sqrt(np.mean(np.square(wav))))
    assert 0.0 < rms < 0.08
    # Loud params really are louder -- the energy channel is live.
    loud = synth.render([0.0] * VOICE_DIM, [0.0, 1.0] + [0.0] * 6, n_samples=1600)
    assert float(np.sqrt(np.mean(np.square(loud)))) > 3.0 * rms


def test_synth_click_free_frame_boundaries():
    """Consecutive frames with continuous params: the boundary step must be no
    larger than an ordinary intra-frame step (WBS M2.2 acceptance)."""
    synth = FormantSynth()
    # Voiced case (voicing=1 -> pure harmonic source).
    pa = [0.20, 0.50, 1.0, 0.10, -0.10, 0.00, 0.20, 0.00]
    pb = [0.25, 0.55, 1.0, 0.15, -0.05, 0.05, 0.20, 0.00]
    pc = [0.30, 0.50, 1.0, 0.20, 0.00, 0.10, 0.20, 0.00]
    f1 = synth.render(pa, pb, n_samples=1600)
    f2 = synth.render(pb, pc, n_samples=1600)
    boundary = abs(float(f2[0]) - float(f1[-1]))
    intra = max(float(np.abs(np.diff(f1)).max()), float(np.abs(np.diff(f2)).max()))
    assert boundary <= 1.5 * intra + 1e-4
    # Aperiodic case (voicing=-1 -> pure noise, edge-faded to zero).
    na = [0.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    g1 = synth.render(na, na, n_samples=1600)
    g2 = synth.render(na, na, n_samples=1600)
    assert abs(float(g2[0]) - float(g1[-1])) < 1e-4


def test_synth_param_interpolation_reaches_endpoint():
    """The frame's last sample must reflect the END params (the next frame
    interpolates FROM them -- continuity across the seam)."""
    synth = FormantSynth()
    up = synth.render(
        [0.0, -1.0, 1.0, 0, 0, 0, 0, 0], [0.0, 1.0, 1.0, 0, 0, 0, 0, 0], n_samples=1600
    )
    head = float(np.sqrt(np.mean(np.square(up[:160]))))
    tail = float(np.sqrt(np.mean(np.square(up[-160:]))))
    assert tail > 5.0 * head  # energy ramped up across the frame


# ------------------------------------------------------------- AudioIntake


def test_intake_bus_mix_in_then_read():
    intake = AudioIntake(mode="bus")
    wav = _sine(1600)
    intake.mix_in(wav)
    chunk = intake.read_chunk(max_ms=250)
    assert chunk is not None and chunk.shape == (1600,)
    assert np.allclose(chunk, wav, atol=1e-6)
    # Cursor advanced: nothing new => nothing returned.
    assert intake.read_chunk(max_ms=250) is None
    intake.mix_in(wav[:100])
    assert intake.read_chunk(max_ms=250).shape == (100,)
    stats = intake.stats()
    assert stats["mode"] == "bus" and stats["running"] is True
    assert stats["device_active"] is False and stats["mix_ins"] == 2


def test_intake_read_cap_keeps_freshest():
    intake = AudioIntake(mode="bus")
    long_wav = np.concatenate([np.full(4000, 0.1, np.float32), np.full(4000, 0.2, np.float32)])
    intake.mix_in(long_wav)
    chunk = intake.read_chunk(max_ms=250)  # cap = 4000 samples @16k
    assert chunk.shape == (4000,)
    assert np.allclose(chunk, 0.2)  # oldest dropped on overrun
    assert intake.read_chunk(max_ms=250) is None


def test_intake_ring_drop_oldest_on_overflow():
    intake = AudioIntake(mode="bus", capacity_s=0.1)  # 1600-sample ring
    intake.mix_in(np.full(1200, 0.1, np.float32))
    intake.mix_in(np.full(1200, 0.2, np.float32))
    chunk = intake.read_chunk(max_ms=1000)
    assert chunk.shape == (1600,)  # only one ring-full survives
    assert np.allclose(chunk[:400], 0.1) and np.allclose(chunk[400:], 0.2)


def test_intake_attach_client_audio_wins():
    intake = AudioIntake(mode="bus")
    intake.mix_in(_sine(1600))
    obs = {"audio": {"data": "Y2xpZW50", "encoding": "pcm16_base64", "sample_rate": 16000}}
    intake.attach_to_obs(obs)
    assert obs["audio"]["data"] == "Y2xpZW50"  # untouched
    # The chunk was NOT consumed: a silent-client obs still gets it.
    obs2: dict = {}
    intake.attach_to_obs(obs2)
    assert obs2["audio"]["source"] == "intake"


def test_intake_silence_gate_with_hysteresis(monkeypatch):
    monkeypatch.setenv("DECADIC_AUDIO_SILENCE_RMS", "0.01")
    intake = AudioIntake(mode="bus")
    quiet = np.full(1600, 0.001, np.float32)
    loud = _sine(1600, amp=0.5)

    obs1: dict = {}
    intake.mix_in(quiet)
    intake.attach_to_obs(obs1)
    assert "audio" not in obs1  # quiet room skipped (no Whisper tax)

    obs2: dict = {}
    intake.mix_in(loud)
    intake.attach_to_obs(obs2)
    assert obs2["audio"]["encoding"] == "pcm16_base64"

    obs3: dict = {}
    intake.mix_in(quiet)
    intake.attach_to_obs(obs3)
    assert "audio" in obs3  # hysteresis: the word tail still attaches

    obs4: dict = {}
    intake.mix_in(quiet)
    intake.attach_to_obs(obs4)
    assert "audio" not in obs4  # gate closed again

    stats = intake.stats()
    assert stats["chunks_attached"] == 2 and stats["silence_skips"] == 2


def test_intake_attach_schema_roundtrips_through_frozen_encoders():
    """The attached obs.audio must decode through the EXISTING Whisper-path
    helper unchanged -- the intake speaks the frozen schema exactly."""
    pytest.importorskip("torch")  # frozen_encoders imports torch at module level
    from decadic.nn.frozen_encoders import _waveform_from_obs

    intake = AudioIntake(mode="bus")
    wav = _sine(1600, amp=0.3)
    intake.mix_in(wav)
    obs: dict = {}
    intake.attach_to_obs(obs)
    assert obs["audio"]["sample_rate"] == 16000
    decoded = _waveform_from_obs(obs)
    assert decoded is not None and decoded.shape == (1600,)
    assert np.allclose(decoded, wav, atol=2.0 / 32768.0)  # pcm16 quantization
    # Sanity on the wire format itself: base64 of little-endian int16.
    raw = np.frombuffer(base64.b64decode(obs["audio"]["data"]), dtype="<i2")
    assert raw.shape == (1600,)


def test_intake_mic_mode_degrades_without_sounddevice(monkeypatch):
    # Poison the import so the degrade path runs even where sounddevice exists.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    intake = AudioIntake(mode="mic")
    intake.start()
    intake.start()  # idempotent
    stats = intake.stats()
    assert stats["running"] is True and stats["device_active"] is False
    # The bus tap (loopback) survives a missing microphone.
    intake.mix_in(_sine(800, amp=0.4))
    obs: dict = {}
    intake.attach_to_obs(obs)
    assert obs["audio"]["source"] == "intake"
    intake.stop()
    intake.stop()  # idempotent
    assert intake.stats()["running"] is False


def test_intake_off_mode_is_inert():
    intake = AudioIntake(mode="off")
    intake.start()
    intake.mix_in(_sine(1600))
    assert intake.read_chunk() is None
    obs: dict = {}
    intake.attach_to_obs(obs)
    assert "audio" not in obs
    stats = intake.stats()
    assert stats["running"] is False and stats["mix_ins"] == 0


def test_intake_singleton_shared_and_resettable(monkeypatch):
    monkeypatch.setenv("DECADIC_AUDIO_INTAKE", "bus")
    reset_audio_intake()
    a = get_audio_intake()
    b = get_audio_intake()
    assert a is b and a.mode == "bus"
    reset_audio_intake()
    assert get_audio_intake() is not a


# ------------------------------------------------------------ VoicePlayback


def test_playback_degrades_without_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    playback = VoicePlayback()
    playback.play(_sine(1600))  # must not raise, must not block
    playback.play(np.zeros(0, np.float32))
    stats = playback.stats()
    assert stats["device_active"] is False and stats["plays"] == 1
    playback.stop()
    playback.stop()  # idempotent


# ------------------------------------------------------ config + faculties


def test_config_defaults_and_validation(monkeypatch):
    from decadic import config as C

    # conftest pins the suite OFF; production defaults are ON.
    assert C.audio_intake_mode() == "off"
    assert C.voice_enabled() is False
    monkeypatch.delenv("DECADIC_AUDIO_INTAKE", raising=False)
    monkeypatch.delenv("DECADIC_VOICE", raising=False)
    assert C.audio_intake_mode() == "mic"
    assert C.voice_enabled() is True
    monkeypatch.setenv("DECADIC_AUDIO_INTAKE", "bogus")
    assert C.audio_intake_mode() == "mic"  # invalid falls back to the default
    monkeypatch.setenv("DECADIC_AUDIO_INTAKE", "bus")
    assert C.audio_intake_mode() == "bus"
    monkeypatch.delenv("DECADIC_VOICE_PLAYBACK", raising=False)
    assert C.voice_playback_mode() == "auto"
    monkeypatch.setenv("DECADIC_VOICE_PLAYBACK", "device")
    assert C.voice_playback_mode() == "device"
    monkeypatch.setenv("DECADIC_VOICE_PLAYBACK", "bogus")
    assert C.voice_playback_mode() == "auto"


def test_voice_faculty_defaults(monkeypatch):
    pytest.importorskip("torch")  # decadic.nn.__init__ imports the bundle
    from decadic.nn.faculties import CognitionFaculties

    assert CognitionFaculties().voice is True  # an organ, not an experiment
    assert CognitionFaculties.from_env().voice is False  # conftest pins 0
    monkeypatch.setenv("DECADIC_VOICE", "1")
    assert CognitionFaculties.from_env().voice is True


# ---------------------------------------------- M2.1: voice head in the stack


def _tiny_stack(monkeypatch, voice: bool):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_VOICE", "1" if voice else "0")
    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack

    torch.manual_seed(21)
    stack = NeuralCognitiveStack(neural_config_from_env("tiny"))
    stack.eval()
    return torch, stack, neural_config_from_env("tiny")


def test_stack_flag_off_builds_no_voice_modules(monkeypatch):
    torch, stack, cfg = _tiny_stack(monkeypatch, voice=False)
    assert stack.has_voice is False
    assert not hasattr(stack, "voice_head")
    assert not any("voice" in k for k in stack.state_dict())  # byte-identical build
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    with torch.no_grad():
        stack.reset_recurrent_state()
        out = stack(z0, ep, mem)
    assert "voice_u" not in out


def test_stack_voice_head_zero_init_then_live(monkeypatch):
    torch, stack, cfg = _tiny_stack(monkeypatch, voice=True)
    assert stack.has_voice and hasattr(stack, "voice_head")
    z0 = torch.randn(1, cfg.d_model)
    ep = torch.rand(1, 4)
    mem = torch.randn(1, cfg.memory_context_dim)
    with torch.no_grad():
        stack.reset_recurrent_state()
        out = stack(z0, ep, mem)
    # Zero-init head + tanh => exact zeros: the newborn does not speak.
    assert out["voice_u"].shape == (1, VOICE_DIM)
    assert int(torch.count_nonzero(out["voice_u"])) == 0

    with torch.no_grad():
        stack.voice_head.weight.normal_(0.0, 0.5)
        stack.voice_head.bias.normal_(0.0, 0.1)
        stack.reset_recurrent_state()
        live_a = stack(z0, ep, mem)
        stack.reset_recurrent_state()
        live_b = stack(z0, ep, mem)
    assert int(torch.count_nonzero(live_a["voice_u"])) > 0
    assert torch.equal(live_a["voice_u"], live_b["voice_u"])  # deterministic
    assert float(live_a["voice_u"].abs().max()) <= 1.0  # tanh range
    # The voice head is a pure readout: it must not perturb cognition itself.
    with torch.no_grad():
        stack.reset_recurrent_state()
        again = stack(z0, ep, mem)
    assert torch.equal(out["z5"], again["z5"])
