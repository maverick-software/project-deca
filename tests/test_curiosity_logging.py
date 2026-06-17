"""Curiosity event logging + the JSON formatter timestamp.

Two layers, both driven directly so they need neither torch nor the async loop:
  1. ``JsonFormatter`` now stamps every line with an ISO-8601 UTC ``time`` field
     so the server log answers "when", not just "what".
  2. ``AgentRuntime._apply_cycle_diagnostics`` emits edge-triggered
     ``curiosity_investigate_enter`` / ``curiosity_investigate_exit`` lines on the
     transitions into and out of the curiosity-driven "investigate" priority --
     never per-cycle, and silent while the agent never investigates.
"""

import json
import logging
from datetime import datetime, timedelta


def _runtime(monkeypatch, agent_id: str, cycle: int):
    """An AgentRuntime with no torch bundle (use_neural off) at a known cycle."""
    monkeypatch.setenv("DECADIC_USE_NEURAL", "0")
    from decadic.agents.runtime import AgentRuntime

    logging.getLogger("decadic.agents.runtime").propagate = True
    rt = AgentRuntime(agent_id)
    rt.state_bus.cycle_index = cycle
    return rt


def _curiosity_diag(drive=0.4, lp=0.3, pleasure=0.4):
    return {
        "curiosity_drive": drive,
        "curiosity_learning_progress": lp,
        "curiosity_pleasure": pleasure,
    }


# --- 1. JSON formatter timestamp --------------------------------------------


def test_json_formatter_stamps_iso_utc_time():
    from decadic.logging.json_logging import JsonFormatter

    rec = logging.LogRecord(
        "decadic.test", logging.INFO, __file__, 1, "hello world", None, None
    )
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["level"] == "INFO"
    assert obj["logger"] == "decadic.test"
    assert obj["message"] == "hello world"
    # "time" is present, ISO-8601, and in UTC (zero offset).
    parsed = datetime.fromisoformat(obj["time"])
    assert parsed.utcoffset() == timedelta(0)


# --- 2. Edge-triggered curiosity event log ----------------------------------


def test_logs_investigate_enter(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-CUR", 321)
    rt.state_bus.priority_label = "investigate"
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(_curiosity_diag(drive=0.42, lp=0.31))
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "curiosity_investigate_enter" in m
        and "agent_id=agent-CUR" in m
        and "cycle=321" in m
        and "drive=0.4200" in m
        for m in msgs
    )
    assert rt._curiosity_investigating is True


def test_enter_is_edge_triggered_not_per_cycle(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-EDGE", 10)
    rt.state_bus.priority_label = "investigate"
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(_curiosity_diag())
        rt.state_bus.cycle_index = 11
        rt._apply_cycle_diagnostics(_curiosity_diag())  # still investigating
    enters = [m for m in (r.getMessage() for r in caplog.records) if "curiosity_investigate_enter" in m]
    assert len(enters) == 1, "a sustained investigate state must log enter only once"


def test_logs_investigate_exit(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-EXIT", 50)
    rt.state_bus.priority_label = "investigate"
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(_curiosity_diag())  # enter
        rt.state_bus.cycle_index = 51
        rt.state_bus.priority_label = "explore"
        rt._apply_cycle_diagnostics(_curiosity_diag())  # exit
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "curiosity_investigate_exit" in m and "agent_id=agent-EXIT" in m and "cycle=51" in m
        for m in msgs
    )
    assert rt._curiosity_investigating is False


def test_no_curiosity_log_when_never_investigating(monkeypatch, caplog):
    rt = _runtime(monkeypatch, "agent-QUIET", 7)
    rt.state_bus.priority_label = "explore"
    with caplog.at_level(logging.INFO, logger="decadic.agents.runtime"):
        rt._apply_cycle_diagnostics(_curiosity_diag(drive=0.0, lp=0.0, pleasure=0.0))
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("curiosity_investigate" in m for m in msgs)
