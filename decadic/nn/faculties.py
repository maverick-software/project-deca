"""Core cognitive faculties — the inherent, on-by-default mechanisms of the mind.

Unlike the A/B/C neuroplasticity switches in ``plastic.py`` (which default OFF so
the stack is byte-identical to a clean baseline for ablation studies), these are
not optional experiments: they are the faculties that make the architecture what
it is. They therefore default ON.

Each faculty changes the model's module set / ``state_dict`` shape (the top-down
perception loop adds modules; discovered perception builds the slot/agency
modules; the encoder mode decides whether real CLIP/Whisper weights are loaded),
so a change takes effect on the next brain rebuild — exactly like a preset or an
A/B/C flag switch. They are threaded per-agent (mirroring ``PlasticityFlags``)
instead of read from the process env at build time, so the dashboard can toggle
them per agent rather than via launch flags.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_PERCEPTION_MODES = ("oracle", "discovered")
VALID_ENCODER_MODES = ("zeros", "hf")


@dataclass
class CognitionFaculties:
    """Build-time switches for the inherent cognitive faculties (all on = full mind).

    - ``perception_feedback``: the top-down predictive-perception loop (a learned
      prediction of the percept blended with the bottom-up encode under a learned
      precision gate) plus perceptual-similarity episodic recall.
    - ``perception_mode``: ``"discovered"`` builds the slot-attention object /
      agency modules so the world graph emerges from the agent's own camera;
      ``"oracle"`` takes the simulator-given entity graph (eval scaffold).
    - ``encoder_mode``: ``"hf"`` loads real frozen CLIP + Whisper (required for
      discovered perception's patch tokens); ``"zeros"`` uses the cheap synthetic
      fallback (no download, but discovered perception is inert).
    """

    perception_feedback: bool = True
    perception_mode: str = "discovered"
    encoder_mode: str = "hf"

    def __post_init__(self) -> None:
        mode = str(self.perception_mode).strip().lower()
        self.perception_mode = mode if mode in VALID_PERCEPTION_MODES else "discovered"
        enc = str(self.encoder_mode).strip().lower()
        self.encoder_mode = enc if enc in VALID_ENCODER_MODES else "hf"
        self.perception_feedback = bool(self.perception_feedback)

    @property
    def discovered(self) -> bool:
        """True when the slot/agency (discovered-perception) modules should build."""
        return self.perception_mode == "discovered"

    @classmethod
    def from_env(cls) -> "CognitionFaculties":
        from decadic import config as _cfg

        return cls(
            perception_feedback=_cfg.perception_feedback_enabled(),
            perception_mode=_cfg.perception_mode(),
            encoder_mode=_cfg.encoder_mode(),
        )
