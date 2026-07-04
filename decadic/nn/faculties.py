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
    - ``self_model_feedback``: the self-state feedback spine (self-model program).
      When on the stack builds a zero-init ``self_ingress`` projection that
      injects the previous cycle's self-report (A‖C‖E) into the stage-3 fuse, so
      the self-model shapes subsequent processing. Defaults OFF (unlike the other
      faculties): it is a research pathway that must prove it raises integration
      before becoming a default, and OFF is byte-identical to the baseline.
    - ``predictive_affect``: the affect forward model (self-model program). When on
      the stack builds a zero-init ``AffectPredictor`` that predicts the next-step
      affective context and colours the episodic proxy with it, so the agent
      perceives in light of how it expects to feel. Defaults OFF (research pathway,
      byte-identical when off).
    - ``represented_self``: the represented self (self-model program). When on the
      stack builds a zero-init ``repself_ingress`` and the cycle writes
      interoception/affect/capability onto the self-node, adds "controls" edges to
      the body parts, and feeds the self-node embedding back through that ingress.
      Defaults OFF (research pathway, byte-identical when off).
    """

    perception_feedback: bool = True
    perception_mode: str = "discovered"
    encoder_mode: str = "hf"
    self_model_feedback: bool = False
    predictive_affect: bool = False
    represented_self: bool = False
    # WS5-M1 (relational binding): working memory enters the stack as a slot
    # TENSOR read by keyed cross-attention (zero-init ingress => parity until
    # learned). Defaults OFF (research pathway; flags-off is byte-identical).
    wm_slot_tensor: bool = False
    # WS5-M2 (relational binding): recalled episodes enter as TOKENS read by
    # keyed cross-attention beside the mean-pooled context vector. Defaults
    # OFF (research pathway; zero-init ingress, byte-identical when off).
    memory_tokens: bool = False

    def __post_init__(self) -> None:
        mode = str(self.perception_mode).strip().lower()
        self.perception_mode = mode if mode in VALID_PERCEPTION_MODES else "discovered"
        enc = str(self.encoder_mode).strip().lower()
        self.encoder_mode = enc if enc in VALID_ENCODER_MODES else "hf"
        self.perception_feedback = bool(self.perception_feedback)
        self.self_model_feedback = bool(self.self_model_feedback)
        self.predictive_affect = bool(self.predictive_affect)
        self.represented_self = bool(self.represented_self)
        self.wm_slot_tensor = bool(self.wm_slot_tensor)
        self.memory_tokens = bool(self.memory_tokens)

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
            self_model_feedback=_cfg.self_model_feedback_enabled(),
            predictive_affect=_cfg.predictive_affect_enabled(),
            represented_self=_cfg.represented_self_enabled(),
            wm_slot_tensor=_cfg.wm_slot_tensor_enabled(),
            memory_tokens=_cfg.memory_tokens_enabled(),
        )
