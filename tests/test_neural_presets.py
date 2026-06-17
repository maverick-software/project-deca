"""Preset ladder: refactor parity, validity, and measured connection counts.

The size presets were refactored from an if/elif chain into a data-driven table
(decadic/nn/config.py). These tests pin the six legacy presets byte-identically so
old preset-tagged checkpoints keep loading, validate the whole ladder's shape, and
assert the advertised connection counts stay accurate.

The heavy tiers (250m/500m/1b) are only built when DECADIC_TEST_HEAVY=1 (building
1b allocates ~4 GB of fp32 weights), so the default suite stays fast.
"""

from __future__ import annotations

import os

import pytest

from decadic.nn.config import (
    HEAVY_PRESETS,
    VALID_PRESETS,
    NeuralArchitectureConfig,
    neural_config_from_env,
    resolve_preset,
)

# State-Bus interface widths that must be identical across EVERY preset.
INTERFACE = {
    "emotion_out": 32,
    "state_mind_out": 64,
    "narrative_out": 48,
    "metacog_out": 24,
    "memory_context_dim": 32,
}

# The six original presets, transcribed from the pre-refactor if/elif chain. The
# refactor must reproduce these scalable widths exactly (checkpoint compat).
LEGACY = {
    "tiny": dict(d_model=96, transformer_layers=2, transformer_heads=4, transformer_ff=192,
                 risk_hidden=64, encoder_decoder_layers=2, gru_hidden=64, lstm_hidden=64,
                 proprio_emb=32, motor_hidden=64, dropout=0.05),
    "medium": dict(d_model=256, transformer_layers=4, transformer_heads=8, transformer_ff=1024,
                   risk_hidden=192, encoder_decoder_layers=3, gru_hidden=192, lstm_hidden=192,
                   proprio_emb=48, motor_hidden=128, dropout=0.1),
    "full": dict(d_model=384, transformer_layers=6, transformer_heads=8, transformer_ff=1536,
                 risk_hidden=256, encoder_decoder_layers=4, gru_hidden=256, lstm_hidden=256,
                 proprio_emb=64, motor_hidden=192, dropout=0.1),
    "xl": dict(d_model=544, transformer_layers=6, transformer_heads=8, transformer_ff=2176,
               risk_hidden=384, encoder_decoder_layers=4, gru_hidden=384, lstm_hidden=384,
               proprio_emb=96, motor_hidden=272, dropout=0.1),
    "xxl": dict(d_model=664, transformer_layers=6, transformer_heads=8, transformer_ff=2656,
                risk_hidden=448, encoder_decoder_layers=4, gru_hidden=448, lstm_hidden=448,
                proprio_emb=112, motor_hidden=320, dropout=0.1),
    "ultra": dict(d_model=768, transformer_layers=6, transformer_heads=8, transformer_ff=3072,
                  risk_hidden=512, encoder_decoder_layers=4, gru_hidden=512, lstm_hidden=512,
                  proprio_emb=128, motor_hidden=384, dropout=0.1),
}

# Measured BASELINE weight-connection counts (sum of params with dim>=2, every
# cognitive faculty OFF, matching tests/conftest.py and the dashboard labels).
# Building each preset must stay within tolerance of these.
EXPECTED_CONN = {
    "tiny": 0.76e6,
    "2_5m": 2.4e6,
    "5m": 4.8e6,
    "medium": 8.4e6,
    "10m": 10.0e6,
    "full": 25.4e6,
    "xl": 50.9e6,
    "xxl": 75.2e6,
    "ultra": 100.3e6,
    "250m": 248.8e6,
    "500m": 493.9e6,
    "1b": 976.4e6,
}

# Light enough to build in the default suite (<= ~10M connections).
LIGHT_PRESETS = ("tiny", "2_5m", "5m", "medium", "10m")

HEAVY_ENABLED = os.environ.get("DECADIC_TEST_HEAVY", "").strip().lower() in ("1", "true", "yes")


def test_valid_presets_cover_the_ladder():
    assert VALID_PRESETS == (
        "tiny", "2_5m", "5m", "medium", "10m", "full", "xl", "xxl", "ultra",
        "250m", "500m", "1b",
    )
    assert HEAVY_PRESETS == {"250m", "500m", "1b"}
    assert all(h in VALID_PRESETS for h in HEAVY_PRESETS)


def test_resolve_preset_unknown_falls_back_to_tiny():
    assert resolve_preset("does-not-exist") == "tiny"
    assert resolve_preset("1B") == "1b"  # case-insensitive
    assert resolve_preset(None) in VALID_PRESETS


@pytest.mark.parametrize("name", list(LEGACY))
def test_legacy_presets_byte_identical_after_refactor(name):
    cfg = neural_config_from_env(name)
    for field, value in LEGACY[name].items():
        assert getattr(cfg, field) == value, f"{name}.{field}"


@pytest.mark.parametrize("name", list(VALID_PRESETS))
def test_interface_dims_fixed_across_presets(name):
    cfg = neural_config_from_env(name)
    for field, value in INTERFACE.items():
        assert getattr(cfg, field) == value, f"{name}.{field} must not scale with size"


@pytest.mark.parametrize("name", list(VALID_PRESETS))
def test_heads_divide_d_model(name):
    cfg = neural_config_from_env(name)
    assert isinstance(cfg, NeuralArchitectureConfig)
    assert cfg.d_model % cfg.transformer_heads == 0, name
    assert cfg.transformer_ff > 0 and cfg.d_model > 0


def _assert_conn_in_tolerance(name, *, tol=0.2):
    import torch

    from decadic.nn.brain_map import brain_topology
    from decadic.nn.neural_stack import NeuralCognitiveStack

    with torch.no_grad():
        stack = NeuralCognitiveStack(neural_config_from_env(name))
        totals = brain_topology(stack, preset=name)["totals"]
    conn = totals["connections"]
    expected = EXPECTED_CONN[name]
    assert abs(conn - expected) <= tol * expected, (
        f"{name}: connections={conn:,} not within {tol:.0%} of label {expected:,.0f}"
    )
    assert totals["preset"] == name
    assert totals["neurons"] > 0


@pytest.mark.parametrize("name", list(LIGHT_PRESETS))
def test_light_presets_build_and_match_labels(name):
    _assert_conn_in_tolerance(name)


@pytest.mark.skipif(not HEAVY_ENABLED, reason="set DECADIC_TEST_HEAVY=1 to build heavy tiers")
@pytest.mark.parametrize("name", ["full", "xl", "xxl", "ultra", "250m", "500m", "1b"])
def test_heavy_presets_build_and_match_labels(name):
    _assert_conn_in_tolerance(name)
