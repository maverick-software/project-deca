import pytest


@pytest.fixture(autouse=True)
def _baseline_faculties(monkeypatch):
    """Pin the pre-flip baseline for every test.

    The core cognitive faculties (perception-feedback loop, discovered perception,
    hf encoders), the A/B/C neuroplasticity subsystems, and the autonomous-learning
    subsystems (need-gated curiosity, dual-network consolidation) now default ON in
    production. The loss-landscape probe is pinned off here too (visualization-only,
    expensive).     The performance knobs (GPU/bf16 encoders, write-behind episodic +
    write-behind LTM consolidation) are pinned to the CPU/fp32/synchronous baseline so
    the suite stays byte-identical and never downloads CLIP/Whisper. The cycle-affect
    knobs are pinned to their pre-real-affect baseline too: the intrinsic
    drive-reduction reward is OFF (the legacy periodic placeholder is used instead)
    and the PE stub weight is held at its legacy 0.25, so the neural affect path is
    byte-identical to before. The existing suite was written against the dense,
    oracle, no-loop baseline, so we restore that here for determinism. Tests that
    exercise a faculty or subsystem set its env / pass explicit flags in the test
    body, which override these (this autouse fixture is set up before the
    explicitly-requested ones).
    """
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_PERCEPTION_MODE", "oracle")
    monkeypatch.setenv("DECADIC_PERCEPTION_FEEDBACK_ENABLED", "0")
    monkeypatch.setenv("DECADIC_PLASTICITY_ENABLED", "0")
    monkeypatch.setenv("DECADIC_SPARSE_ENABLED", "0")
    monkeypatch.setenv("DECADIC_GROWTH_ENABLED", "0")
    monkeypatch.setenv("DECADIC_CURIOSITY_ENABLED", "0")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_ENABLED", "0")
    monkeypatch.setenv("DECADIC_LANDSCAPE_ENABLED", "0")
    monkeypatch.setenv("DECADIC_ENCODER_PRECISION", "fp32")
    monkeypatch.setenv("DECADIC_EPISODIC_ASYNC", "0")
    monkeypatch.setenv("DECADIC_LTM_ASYNC", "0")
    monkeypatch.setenv("DECADIC_CYCLE_PROFILE", "0")
    monkeypatch.setenv("DECADIC_DRIVE_REWARD_ENABLED", "0")
    monkeypatch.setenv("DECADIC_PE_STUB_WEIGHT", "0.25")


@pytest.fixture
def api_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DECADIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("DECADIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.02")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    # Disable the wall-clock metabolic loop so passive drain never races API
    # assertions; reservoir routing via the fast path is exercised explicitly.
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    from decadic.api.app import create_app

    return create_app()


@pytest.fixture
def api_app_neural(tmp_path, monkeypatch):
    """Neural stack enabled, but every plasticity subsystem OFF by default.

    Used to verify the UI-set new-agent defaults turn plasticity on for agents
    created *after* the toggle, without touching agents created before it. Growth
    intervals/gate are tightened so a toggled-on agent grows within a short run.
    """
    monkeypatch.setenv("DECADIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("DECADIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.02")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    # Plasticity enable flags intentionally left UNSET -> defaults resolve to off.
    monkeypatch.setenv("DECADIC_SPARSE_REWIRE_INTERVAL", "3")
    monkeypatch.setenv("DECADIC_GROWTH_INTERVAL", "3")
    monkeypatch.setenv("DECADIC_GROWTH_STEP", "8")
    monkeypatch.setenv("DECADIC_GROWTH_PCLOSS_THRESHOLD", "0")
    from decadic.api.app import create_app

    return create_app()


@pytest.fixture
def api_app_plastic(tmp_path, monkeypatch):
    """API app with the neural stack + all three plasticity subsystems enabled.

    Intervals are tightened and the growth pc-loss gate dropped to zero so a
    short test run actually exercises rewiring and neuron growth.
    """
    monkeypatch.setenv("DECADIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DECADIC_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("DECADIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DECADIC_CYCLE_INTERVAL_S", "0.02")
    monkeypatch.setenv("DECADIC_CONSOLIDATION_STUB_INTERVAL_S", "0")
    monkeypatch.setenv("DECADIC_METABOLIC_TICK_S", "0")
    monkeypatch.setenv("DECADIC_USE_NEURAL", "1")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_NEURAL_PRESET", "tiny")
    monkeypatch.setenv("DECADIC_DEVICE", "cpu")
    monkeypatch.setenv("DECADIC_PLASTICITY_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_ENABLED", "1")
    monkeypatch.setenv("DECADIC_GROWTH_ENABLED", "1")
    monkeypatch.setenv("DECADIC_SPARSE_DENSITY", "0.5")
    monkeypatch.setenv("DECADIC_SPARSE_REWIRE_INTERVAL", "3")
    monkeypatch.setenv("DECADIC_GROWTH_INTERVAL", "3")
    monkeypatch.setenv("DECADIC_GROWTH_STEP", "8")
    monkeypatch.setenv("DECADIC_GROWTH_PCLOSS_THRESHOLD", "0")
    monkeypatch.setenv("DECADIC_MAX_NEURONS", "160")
    monkeypatch.setenv("DECADIC_GROWABLE_HIDDEN_CEILING", "256")
    from decadic.api.app import create_app

    return create_app()
