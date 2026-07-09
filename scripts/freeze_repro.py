"""WS-FREEZE — fast reproduction helper (snapshot near the freeze, reload, iterate).

The freeze is store-size dependent and takes ~100 min to reach fresh. Instead of
paying that every iteration, snapshot a full agent (weights + episodic + LTM
graph) near the freeze once, then reload that snapshot and run a short pass with
the watchdog live. Each fix iteration becomes a reload + short run.

Runbook
-------
1. Start the server + body normally (scripts/run_body_diag.ps1 or run_decadic_server).
   Let it run toward the freeze zone (~cycle 22k). Watchdog is default-on.
2. Snapshot near the freeze (one-time priming cost):
     python scripts/freeze_repro.py save
   -> prints SAVE_ID. This persists weights + episodic DB + the whole LTM graph.
3. Reload and reproduce (repeat per fix iteration):
     python scripts/freeze_repro.py load <SAVE_ID>
   -> spins up a fresh agent carrying the large store; attach the body to it and
      it should freeze within minutes. Watch server.err.log for FREEZE_REPORT +
      the faulthandler stack dump.
4. Lower the stall threshold if you want a faster catch:
     set DECADIC_WATCHDOG_STALL_S=15

Notes
-----
- checkpoint/restore alone is NOT enough (it omits the graph + episodic, which
  is exactly the state the freeze depends on) -- this uses the full save/load.
- All HTTP; no extra deps (urllib). Override host with DECADIC_BASE_URL.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("DECADIC_BASE_URL", "http://127.0.0.1:8765")


def _req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def _first_agent_id() -> str | None:
    agents = _req("GET", "/agents").get("agents") or []
    return agents[0]["agent_id"] if agents else None


def cmd_save() -> int:
    aid = _first_agent_id()
    if not aid:
        print("no running agent found", file=sys.stderr)
        return 1
    out = _req("POST", f"/agent/{aid}/save")
    print(json.dumps(out, indent=2))
    sid = out.get("save_id") or out.get("id") or out.get("saveId")
    if sid:
        print(f"\nSAVE_ID={sid}\n  reload with: python scripts/freeze_repro.py load {sid}")
    return 0


def cmd_load(save_id: str) -> int:
    out = _req("POST", f"/saved-agents/{save_id}/load")
    print(json.dumps(out, indent=2))
    print("\nloaded -- attach the body and watch server.err.log for FREEZE_REPORT")
    return 0


def cmd_list() -> int:
    print(json.dumps(_req("GET", "/saved-agents"), indent=2))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "save":
        return cmd_save()
    if cmd == "load" and len(argv) > 1:
        return cmd_load(argv[1])
    if cmd == "list":
        return cmd_list()
    print(f"usage: freeze_repro.py [save | load <SAVE_ID> | list]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
