"""Architecture presets for Phase 2 neural stack (brief-aligned, scalable).

Presets are defined as a data-driven table (``_PRESET_SPECS``) rather than an
if/elif chain so the ladder can grow to many tiers without bloating this module
(house PHILOSOPHY #1, <=500 lines). Each spec carries only the *scalable* widths;
the State-Bus interface widths (``emotion_out``/``state_mind_out``/
``narrative_out``/``metacog_out``/``memory_context_dim``) are fixed across every
preset because they are tied to the persistent cognitive state-element dims and
must NOT scale with network size. ``n_actuators``/``forward_pred_dim`` are derived
from the connected body at build time.

Approximate weight-connection counts (``sum(p.numel() for p in stack.parameters()
if p.dim() >= 2)``, measured via ``brain_map.brain_topology``) are noted per tier
and surfaced as the dashboard labels. The 250m/500m/1b tiers are *define-only*:
they instantiate and run a forward pass, but training them every cognitive cycle
in fp32 Adam will exhaust a single consumer GPU (see README "heavy tiers / memory
cliff"). The guardrail in ``decadic.nn.bundle.NeuralBundle.try_build`` warns when a
heavy tier is selected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from decadic.config import CONTROLLABLE_PROPRIO_BASE, DEFAULT_N_ACTUATORS


@dataclass(frozen=True)
class NeuralArchitectureConfig:
    d_model: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff: int
    risk_hidden: int
    encoder_decoder_layers: int
    gru_hidden: int
    lstm_hidden: int
    emotion_out: int
    state_mind_out: int
    narrative_out: int
    metacog_out: int
    proprio_emb: int
    dropout: float
    memory_context_dim: int
    # Embodied motor control (active inference):
    n_actuators: int  # PD targets emitted by the motor head (== body model.nu)
    motor_hidden: int  # motor head hidden width
    forward_pred_dim: int  # controllable-proprio vector the forward model predicts


# Fixed interface widths shared by EVERY preset. Tied to the State Bus element
# dims (B/A/C/E) and the memory-context width; scaling these would break
# checkpoint/State-Bus compatibility, so they never change with network size.
_INTERFACE: dict[str, int] = {
    "emotion_out": 32,
    "state_mind_out": 64,
    "narrative_out": 48,
    "metacog_out": 24,
    "memory_context_dim": 32,
}

# Scalable widths per tier, ordered small -> large (drives VALID_PRESETS order,
# the dashboard option order, and validation). The "~conn" comment is the
# measured cognitive-stack weight-connection count (dim>=2 matrices, faculties
# off), asserted by tests/test_neural_presets.py. NOTE: the dashboard label now
# advertises TOTAL trainable parameters (stack + the shared ~177M frozen HF
# sensory encoder) with this stack count shown alongside -- see
# dashboard/src/neuralPresets.ts. "conn" here is the scalable stack only.
_PRESET_SPECS: dict[str, dict] = {
    # ~1.1M conn -- fastest cycles (default). This is the practical floor: the
    # fixed motor/PC/projection heads keep connections near 1M even at d_model=96,
    # so tiny doubles as the "~1M" tier (no separate sub-tier below it).
    "tiny": {
        "d_model": 96,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "transformer_ff": 192,
        "risk_hidden": 64,
        "encoder_decoder_layers": 2,
        "gru_hidden": 64,
        "lstm_hidden": 64,
        "proprio_emb": 32,
        "motor_hidden": 64,
        "dropout": 0.05,
    },
    # ~2.5M conn.
    "2_5m": {
        "d_model": 144,
        "transformer_layers": 3,
        "transformer_heads": 8,
        "transformer_ff": 576,
        "risk_hidden": 112,
        "encoder_decoder_layers": 2,
        "gru_hidden": 128,
        "lstm_hidden": 128,
        "proprio_emb": 40,
        "motor_hidden": 96,
        "dropout": 0.1,
    },
    # ~5M conn.
    "5m": {
        "d_model": 200,
        "transformer_layers": 3,
        "transformer_heads": 8,
        "transformer_ff": 800,
        "risk_hidden": 160,
        "encoder_decoder_layers": 3,
        "gru_hidden": 160,
        "lstm_hidden": 160,
        "proprio_emb": 44,
        "motor_hidden": 112,
        "dropout": 0.1,
    },
    # ~8.3M conn -- balanced.
    "medium": {
        "d_model": 256,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "transformer_ff": 1024,
        "risk_hidden": 192,
        "encoder_decoder_layers": 3,
        "gru_hidden": 192,
        "lstm_hidden": 192,
        "proprio_emb": 48,
        "motor_hidden": 128,
        "dropout": 0.1,
    },
    # ~10M conn.
    "10m": {
        "d_model": 280,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "transformer_ff": 1120,
        "risk_hidden": 208,
        "encoder_decoder_layers": 3,
        "gru_hidden": 208,
        "lstm_hidden": 208,
        "proprio_emb": 52,
        "motor_hidden": 144,
        "dropout": 0.1,
    },
    # ~25M conn.
    "full": {
        "d_model": 384,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 1536,
        "risk_hidden": 256,
        "encoder_decoder_layers": 4,
        "gru_hidden": 256,
        "lstm_hidden": 256,
        "proprio_emb": 64,
        "motor_hidden": 192,
        "dropout": 0.1,
    },
    # ~50M conn (about 2x full). Same 6/4/2 transformer layout as full, scaled
    # in width (d_model + proportional hidden dims).
    "xl": {
        "d_model": 544,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 2176,
        "risk_hidden": 384,
        "encoder_decoder_layers": 4,
        "gru_hidden": 384,
        "lstm_hidden": 384,
        "proprio_emb": 96,
        "motor_hidden": 272,
        "dropout": 0.1,
    },
    # ~75M conn (about 3x full).
    "xxl": {
        "d_model": 664,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 2656,
        "risk_hidden": 448,
        "encoder_decoder_layers": 4,
        "gru_hidden": 448,
        "lstm_hidden": 448,
        "proprio_emb": 112,
        "motor_hidden": 320,
        "dropout": 0.1,
    },
    # ~100M conn (about 4x full). Heaviest of the original ladder: runs every
    # cognitive cycle, so watch Cycle wall ms / VRAM after switching.
    "ultra": {
        "d_model": 768,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 3072,
        "risk_hidden": 512,
        "encoder_decoder_layers": 4,
        "gru_hidden": 512,
        "lstm_hidden": 512,
        "proprio_emb": 128,
        "motor_hidden": 384,
        "dropout": 0.1,
    },
    # ~250M conn -- HEAVY/define-only. d ~= 384*sqrt(250M/25M).
    "250m": {
        "d_model": 1216,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 4864,
        "risk_hidden": 768,
        "encoder_decoder_layers": 4,
        "gru_hidden": 768,
        "lstm_hidden": 768,
        "proprio_emb": 192,
        "motor_hidden": 576,
        "dropout": 0.1,
    },
    # ~500M conn -- HEAVY/define-only.
    "500m": {
        "d_model": 1720,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 6880,
        "risk_hidden": 1024,
        "encoder_decoder_layers": 4,
        "gru_hidden": 1024,
        "lstm_hidden": 1024,
        "proprio_emb": 256,
        "motor_hidden": 768,
        "dropout": 0.1,
    },
    # ~1B conn -- HEAVY/define-only. Will OOM training every cycle on a single
    # consumer GPU; see README "heavy tiers / memory cliff".
    "1b": {
        "d_model": 2432,
        "transformer_layers": 6,
        "transformer_heads": 8,
        "transformer_ff": 9728,
        "risk_hidden": 1280,
        "encoder_decoder_layers": 4,
        "gru_hidden": 1280,
        "lstm_hidden": 1280,
        "proprio_emb": 320,
        "motor_hidden": 1024,
        "dropout": 0.1,
    },
}

VALID_PRESETS = tuple(_PRESET_SPECS.keys())

# Tiers whose fp32 Adam training footprint can exhaust a single consumer GPU.
# Used by the build-time guardrail (bundle.try_build) and surfaced in the UI.
HEAVY_PRESETS = frozenset({"250m", "500m", "1b"})


def resolve_preset(preset: str | None = None) -> str:
    """Explicit preset wins over DECADIC_NEURAL_PRESET; unknown names -> tiny."""
    name = (preset or os.environ.get("DECADIC_NEURAL_PRESET", "tiny")).strip().lower()
    return name if name in VALID_PRESETS else "tiny"


def neural_config_from_env(preset: str | None = None) -> NeuralArchitectureConfig:
    preset = resolve_preset(preset)
    nu = max(1, int(os.environ.get("DECADIC_N_ACTUATORS", str(DEFAULT_N_ACTUATORS))))
    fwd_dim = CONTROLLABLE_PROPRIO_BASE + nu
    cfg = NeuralArchitectureConfig(
        **_INTERFACE,
        **_PRESET_SPECS[preset],
        n_actuators=nu,
        forward_pred_dim=fwd_dim,
    )
    mem_dim = int(os.environ.get("DECADIC_MEMORY_CONTEXT_DIM", str(cfg.memory_context_dim)))
    return replace(cfg, memory_context_dim=max(4, mem_dim))


def viability_pe_scale() -> float:
    return float(os.environ.get("DECADIC_VIABILITY_PE_SCALE", "0.015"))
