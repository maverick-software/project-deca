"""WS-FREEZE — FreezeWatchdog unit tests.

Drives `check_once` deterministically with an injected clock (via the probe's
`now`) and an injected dump, so no real hang or real agent is needed. Verifies:
fresh heartbeat is silent; a stall reports once + dumps; the report attributes
H1/H2 correctly; re-emits are throttled; a missing heartbeat is safe.
"""

from __future__ import annotations

from decadic.agents.freeze_watchdog import FreezeWatchdog


def _mk(state, now, dumps):
    def probe():
        return {
            "now": now[0],
            "cognition": {
                "hb_cycle_s": state["hb_cycle_s"],
                "phase": state["phase"],
                "cycle_index": state["cycle_index"],
            },
            "write_behind": {
                "in_job": state["wb_in_job"],
                "job_start_s": state["wb_job_start_s"],
                "jobs_completed": state["jobs"],
                "last_worker_ms": state["last_ms"],
                "queue_size": state["qsize"],
            },
            "flusher": {
                "alive": state["fl_alive"],
                "last_batch_s": state["fl_last_batch_s"],
                "backlog": state["backlog"],
            },
        }

    return FreezeWatchdog(
        probe, agent_id="t", stall_s=20.0, check_s=1.0, repeat_s=60.0,
        dump=lambda: dumps.__setitem__("n", dumps["n"] + 1),
    )


def _state():
    return {
        "hb_cycle_s": 1000.0, "phase": "idle", "cycle_index": 100,
        "wb_in_job": False, "wb_job_start_s": None, "jobs": 5, "last_ms": 40.0, "qsize": 0,
        "fl_alive": True, "fl_last_batch_s": 999.0, "backlog": 0,
    }


def test_fresh_heartbeat_is_silent():
    now, dumps, st = [1005.0], {"n": 0}, _state()
    wd = _mk(st, now, dumps)
    assert wd.check_once() is None
    assert dumps["n"] == 0 and wd.reports == 0


def test_stall_reports_and_dumps():
    now, dumps, st = [1005.0], {"n": 0}, _state()
    wd = _mk(st, now, dumps)
    now[0] = 1040.0
    st["hb_cycle_s"] = 1010.0  # 30s stale
    r = wd.check_once()
    assert r is not None and r["stalled_s"] == 30.0
    assert dumps["n"] == 1 and wd.reports == 1


def test_report_attributes_h1_lock_hold():
    now, dumps, st = [1040.0], {"n": 0}, _state()
    st["hb_cycle_s"] = 1010.0
    st["phase"] = "neural"
    st["wb_in_job"] = True
    st["wb_job_start_s"] = 1012.0  # holding the graph RLock ~28s
    wd = _mk(st, now, dumps)
    line = wd.check_once()["line"]
    assert "H1_wb_in_job=True" in line
    assert "H1_wb_lock_held=28.0s" in line
    assert "phase=neural" in line
    assert "H2_flusher_alive=True" in line


def test_report_attributes_h2_flusher_dead():
    now, dumps, st = [2000.0], {"n": 0}, _state()
    st["hb_cycle_s"] = 1960.0
    st["fl_alive"] = False
    wd = _mk(st, now, dumps)
    line = wd.check_once()["line"]
    assert "H2_flusher_alive=False" in line
    assert "H1_wb_in_job=False" in line  # H1 cleared -> points elsewhere


def test_reemit_is_throttled_then_repeats():
    now, dumps, st = [1040.0], {"n": 0}, _state()
    st["hb_cycle_s"] = 1010.0
    wd = _mk(st, now, dumps)
    assert wd.check_once() is not None  # first report
    now[0] = 1050.0
    assert wd.check_once() is None      # within repeat_s -> throttled
    assert wd.reports == 1
    now[0] = 1110.0                     # past repeat_s
    assert wd.check_once() is not None
    assert wd.reports == 2


def test_dead_agent_labeled_not_hang_no_dump():
    now, dumps, st = [2000.0], {"n": 0}, _state()
    st["hb_cycle_s"] = 1960.0  # 40s "stalled"
    wd = _mk(st, now, dumps)
    # inject dead status into the probe snapshot
    base = wd._probe
    wd._probe = lambda: {**base(), "cognition": {**base()["cognition"], "status": "dead"}}
    r = wd.check_once()
    assert r is not None and r.get("dead") is True
    assert "AGENT_DEAD" in r["line"] and "NOT a hang" in r["line"]
    assert dumps["n"] == 0  # no stack dump for an expected-dead idle


def test_missing_heartbeat_is_safe():
    wd = FreezeWatchdog(lambda: {"now": 1.0, "cognition": {"hb_cycle_s": None}}, dump=lambda: None)
    assert wd.check_once() is None


def test_empty_probe_is_safe():
    wd = FreezeWatchdog(lambda: {}, dump=lambda: None)
    assert wd.check_once() is None


def test_start_stop_is_clean():
    wd = FreezeWatchdog(lambda: {"now": 0.0, "cognition": {"hb_cycle_s": 0.0}}, check_s=0.05, dump=lambda: None)
    wd.start()
    wd.start()  # idempotent
    wd.stop()   # joins cleanly, no error
