import os
import tempfile
from pathlib import Path

import pytest


def _writable_tmp_root() -> Path:
    """Return a Windows-safe pytest temp root before ``tmp_path`` initializes."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    override = os.environ.get("DECADIC_TEST_TMP_ROOT")
    if override:
        candidates.append(Path(override))
    if os.name == "nt":
        candidates.append(Path("C:/tmp/decadic_pytest"))
    candidates.append(repo_root / ".pytest_tmp")
    candidates.append(Path(tempfile.gettempdir()) / "decadic_pytest")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / f".write_probe_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable pytest temp root found")


_TEST_TMP_ROOT = _writable_tmp_root()
os.environ["TMP"] = str(_TEST_TMP_ROOT)
os.environ["TEMP"] = str(_TEST_TMP_ROOT)
os.environ["TMPDIR"] = str(_TEST_TMP_ROOT)
tempfile.tempdir = str(_TEST_TMP_ROOT)


def pytest_configure(config):
    """Use a per-process basetemp so stale Windows locks cannot poison tmp_path."""
    if getattr(config.option, "basetemp", None):
        return
    config.option.basetemp = str(_TEST_TMP_ROOT / f"run-{os.getpid()}")


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
    monkeypatch.setenv("DECADIC_REQUIRE_CUDA", "0")
    monkeypatch.setenv("DECADIC_ENCODER_MODE", "zeros")
    monkeypatch.setenv("DECADIC_PERCEPTION_MODE", "oracle")
    monkeypatch.setenv("DECADIC_PERCEPTION_FEEDBACK_ENABLED", "0")
    monkeypatch.setenv("DECADIC_SELF_MODEL_FEEDBACK", "0")
    monkeypatch.setenv("DECADIC_PREDICTIVE_AFFECT", "0")
    monkeypatch.setenv("DECADIC_REPRESENTED_SELF", "0")
    monkeypatch.setenv("DECADIC_GWT_ENABLED", "0")
    monkeypatch.setenv("DECADIC_INTEGRATION_WINDOW_MS", "0")
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
    # WS3/WS5 faculties: default ON in production since 2026-07-04 (owner
    # decision -- every validated upgrade runs in a full cognition run).
    # Pinned OFF here for the deterministic byte-identical baseline, exactly
    # like the faculties above; tests that exercise them set env explicitly.
    monkeypatch.setenv("DECADIC_GATE_ENABLED", "0")
    monkeypatch.setenv("DECADIC_WM_SLOT_TENSOR", "0")
    monkeypatch.setenv("DECADIC_MEMORY_TOKENS", "0")
    monkeypatch.setenv("DECADIC_RELATIONAL_CORE", "0")
    # WS6 speech loop: production defaults ON (hearing + voice are organs);
    # pinned OFF for determinism and device-free CI, same as the flags above.
    monkeypatch.setenv("DECADIC_VOICE", "0")
    monkeypatch.setenv("DECADIC_AUDIO_INTAKE", "off")
    # WS-FORAGE: production ships the normalized, longer-horizon SF regime ON,
    # but the suite is pinned to the ORIGINAL SF params so existing numeric
    # expectations stay deterministic (the gate/binding-faculty pattern:
    # prod-on, tests-pinned). Dedicated WS-FORAGE tests set these explicitly.
    monkeypatch.setenv("DECADIC_SF_NORMALIZE_RETURNS", "0")
    monkeypatch.setenv("DECADIC_SF_GAMMA", "0.97")
    monkeypatch.setenv("DECADIC_SF_LAMBDA", "0.9")
    monkeypatch.setenv("DECADIC_SF_VALUE_WEIGHT", "0.3")
    # Goal-conditioned policy is prod-on (zero-init, birth-identical) but pinned
    # OFF in the suite so trained-cycle tests with an active goal can't diverge;
    # the dedicated M3 tests enable it explicitly.
    monkeypatch.setenv("DECADIC_GOAL_CONDITIONED_POLICY", "0")
    monkeypatch.setenv("DECADIC_GOAL_BEARING", "0")
    monkeypatch.setenv("DECADIC_TYPE2_SEARCH", "0")
    # WS-EXPAND E2: multi-channel learning control is prod-on (neutral until
    # warmup -> birth-identical) but pinned OFF in the suite so plasticity/gate
    # numeric expectations stay deterministic; test_learning_control.py enables
    # it explicitly.
    monkeypatch.setenv("DECADIC_LEARN_CONTROL_MULTI", "0")
    # WS-EXPAND E1: cognitive map is prod-on (behavior-identical until a
    # measured stall evidences a blockage) but pinned OFF in the suite;
    # test_cognitive_map.py enables it explicitly.
    monkeypatch.setenv("DECADIC_COGNITIVE_MAP", "0")
    # WS-EXPAND E1.6: rollout planner is prod-on (inert until the SF value
    # ramp opens) but pinned OFF in the suite; test_action_planner.py tests
    # the helper directly.
    monkeypatch.setenv("DECADIC_PLANNER", "0")


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
    # Governance gates off (0 = disabled) so short runs grow on schedule.
    monkeypatch.setenv("DECADIC_GROWTH_MIN_PROGRESS", "0")
    monkeypatch.setenv("DECADIC_GROWTH_MIN_GAIN", "0")
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
    # Governance gates off (0 = disabled) so short runs grow on schedule.
    monkeypatch.setenv("DECADIC_GROWTH_MIN_PROGRESS", "0")
    monkeypatch.setenv("DECADIC_GROWTH_MIN_GAIN", "0")
    monkeypatch.setenv("DECADIC_MAX_NEURONS", "160")
    monkeypatch.setenv("DECADIC_GROWABLE_HIDDEN_CEILING", "256")
    from decadic.api.app import create_app

    return create_app()
