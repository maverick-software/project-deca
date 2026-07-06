"""WS6 speech-loop acoustic organs: intake (ears), vocal tract (mouth),
playback (operator monitor tee). Pure numpy + optional sounddevice; importing
this package never opens a device (everything is lazy)."""

from decadic.audio.intake import (
    AudioIntake,
    get_audio_intake,
    reset_audio_intake,
)
from decadic.audio.playback import VoicePlayback
from decadic.audio.vocal_tract import VOICE_DIM, FormantSynth

__all__ = [
    "VOICE_DIM",
    "AudioIntake",
    "FormantSynth",
    "VoicePlayback",
    "get_audio_intake",
    "reset_audio_intake",
]
