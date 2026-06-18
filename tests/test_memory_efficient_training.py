"""Memory-efficient training path + integration sweep harness (Phase 6).

Phase 6 is hardware-gated (8-bit Adam + bf16 forward only bite on CUDA with
bitsandbytes installed). These tests verify the CPU/off path is byte-identical:
the optimizer falls back to fp32 Adam, the autocast context is a nullcontext, and
the integration sweep harness runs end-to-end with the zero-init spine giving an
exact zero PCI delta (parity) and a learned spine producing a finite reading.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_config_default_on(monkeypatch):
    from decadic import config as C

    # Ships ON by default; on CPU/no-bnb it falls back to fp32 so it stays
    # byte-identical regardless (conftest does not need to pin it).
    monkeypatch.delenv("DECADIC_MEMORY_EFFICIENT_TRAINING", raising=False)
    assert C.memory_efficient_training_enabled() is True


def test_build_optimizer_default_is_fp32_adam():
    from decadic.nn.optim import build_optimizer

    p = [torch.nn.Parameter(torch.zeros(3))]
    opt, kind = build_optimizer(p, lr=1e-4, device="cpu", memory_efficient=False)
    assert kind == "adam"
    assert isinstance(opt, torch.optim.Adam)


def test_build_optimizer_cpu_falls_back_when_requested():
    from decadic.nn.optim import build_optimizer

    p = [torch.nn.Parameter(torch.zeros(3))]
    # memory_efficient on a non-CUDA device must silently use fp32 Adam.
    opt, kind = build_optimizer(p, lr=1e-4, device="cpu", memory_efficient=True)
    assert kind == "adam"
    assert isinstance(opt, torch.optim.Adam)


def test_train_autocast_is_nullcontext_on_cpu(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_MEMORY_EFFICIENT_TRAINING", "1")
    from decadic.nn.bundle import NeuralBundle
    from decadic.nn.faculties import CognitionFaculties

    b = NeuralBundle.try_build(
        "mem-eff",
        faculties=CognitionFaculties(
            perception_feedback=False, perception_mode="oracle", encoder_mode="zeros"
        ),
    )
    # CPU device => nullcontext even with the flag on (no bf16 autocast on CPU).
    assert isinstance(b.train_autocast(), type(nullcontext()))
    # And the optimizer fell back to fp32 Adam.
    assert isinstance(b.optimizer, torch.optim.Adam)


def test_sweep_zero_sigma_is_exact_parity():
    from scripts.integration_sweep import _measure

    row = _measure("tiny", seed=0, learned_sigma=0.0)
    assert row["pci_delta"] == 0.0  # zero-init spine => on == off
    assert row["pci_off"] == row["pci_on"]


def test_sweep_learned_sigma_produces_finite_reading():
    import math

    from scripts.integration_sweep import _measure

    row = _measure("tiny", seed=1, learned_sigma=0.4)
    assert math.isfinite(row["pci_on"]) and math.isfinite(row["pci_off"])
    assert set(row) >= {"preset", "seed", "pci_off", "pci_on", "pci_delta"}
