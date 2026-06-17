"""Faithfulness guardrails: the curriculum shapes the world; it never touches the loss.

These tests statically assert the experiment's core invariant - that the walking
curriculum is observation/config/world-only and that its eval-only telemetry never
leaks into the cognitive (gradient) path. If a future change wires a curriculum
symbol into the loss or feeds a gait metric into cognition, one of these fails.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "decadic" / "cycle" / "neural_pipeline.py"
_STACK = _ROOT / "decadic" / "nn" / "neural_stack.py"
_SUPERVISOR = _ROOT / "decadic" / "curriculum" / "supervisor.py"

# The only agent attributes/methods the supervisor is allowed to touch: reads,
# live config, world shaping, and checkpointing. Nothing that computes a gradient.
_ALLOWED_AGENT_CALLS = {
    "lock",
    "metrics",
    "viability",
    "status",
    "has_body",
    "configure",
    "queue_body_command",
    "checkpoint_payload",
    "save_brain",
    "revive",
}

# Eval-only telemetry the curriculum reads; must never appear in cognition source.
_EVAL_ONLY_METRICS = (
    "distance_traveled",
    "net_displacement",
    "fall_rate",
    "gait_regularity",
    "consume_events",
)


def _strip_comments_and_strings(src: str) -> str:
    """Crude strip of triple-quoted docstrings, line comments, and string literals.

    Good enough to keep prose mentions (e.g. "never touches the loss") out of the
    code-only forbidden-token scan below.
    """
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    src = re.sub(r'"[^"\n]*"', '""', src)
    src = re.sub(r"'[^'\n]*'", "''", src)
    return src


def test_gradient_path_has_no_curriculum_import():
    # The cognitive cycle must not import the walking-curriculum package. (The
    # file legitimately mentions the unrelated joint-brace "ROM curriculum" and
    # the pre-existing `curriculum_mode` assist knob, so we forbid the *import*,
    # not the word.)
    src = _PIPELINE.read_text(encoding="utf-8")
    assert "decadic.curriculum" not in src
    assert "CurriculumSupervisor" not in src


def test_supervisor_never_invokes_cognition_or_loss():
    # Strip comments/docstrings so a doc mention of "the loss" isn't a false hit;
    # we only forbid actual gradient-path *code*.
    src = _SUPERVISOR.read_text(encoding="utf-8")
    code = _strip_comments_and_strings(src)
    for forbidden in (
        "run_neural_cycle",
        "run_stub_cycle",
        ".backward(",
        "optimizer",
        "import torch",
    ):
        assert forbidden not in code, f"supervisor must not reference {forbidden!r}"


def test_supervisor_agent_call_surface_is_read_config_world_only():
    src = _SUPERVISOR.read_text(encoding="utf-8")
    # Every `agent.<name>` (and `self._current_agent()`-bound) access must be in
    # the allow-list. We scan the generic `agent.<attr>` pattern used throughout.
    used = set(re.findall(r"\bagent\.([A-Za-z_][A-Za-z0-9_]*)", src))
    leaked = used - _ALLOWED_AGENT_CALLS
    assert not leaked, f"supervisor touches disallowed agent surface: {sorted(leaked)}"


def test_eval_only_metrics_never_feed_cognition():
    for path in (_PIPELINE, _STACK):
        src = path.read_text(encoding="utf-8")
        for key in _EVAL_ONLY_METRICS:
            assert key not in src, f"{key!r} (eval-only) leaked into {path.name}"


def test_live_overrides_default_to_env_parity():
    """When no curriculum override is set, the cycle must read the env default."""
    from decadic.config import motor_exploration_sigma

    # sigma_max=None -> env default branch (parity with pre-curriculum behaviour).
    env_default = motor_exploration_sigma(drive=0.7, fwd_error=0.2)
    same = motor_exploration_sigma(drive=0.7, fwd_error=0.2, sigma_max=None)
    assert env_default == same
    # An explicit override changes the ceiling (curriculum knob takes effect).
    overridden = motor_exploration_sigma(drive=0.7, fwd_error=0.2, sigma_max=0.0)
    assert overridden == 0.0


def test_cycle_context_overrides_default_none():
    """CycleContext curriculum knobs default to None (env fallback / parity)."""
    from decadic.cycle.types import CycleContext

    f = CycleContext.__dataclass_fields__
    for name in ("ai_intero_pref_weight", "drive_priority_gain", "motor_babble_sigma"):
        assert name in f
        assert f[name].default is None
