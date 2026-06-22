from __future__ import annotations

import pytest


def test_cuda_required_rejects_cpu_device(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_REQUIRE_CUDA", "1")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.bundle import NeuralBundle

    with pytest.raises(RuntimeError, match="DECADIC_REQUIRE_CUDA=1"):
        NeuralBundle.resolve_device()


def test_cuda_not_required_allows_cpu_device(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DECADIC_REQUIRE_CUDA", "0")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.nn.bundle import NeuralBundle

    assert NeuralBundle.resolve_device().type == "cpu"

