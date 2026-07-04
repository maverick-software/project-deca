"""WS5-M5: drive the binding probe against a live server and sample telemetry.

Spawns the synthetic client in scenario mode, polls /agent/<id>/metrics while
it runs, writes one JSONL row per sample (priority_scalar is the verdict
signal; exported per-cycle by the pipeline). The PS1 wrapper owns server/env.

Usage: python scripts/binding_probe_run.py --agent-id <id> --scenario <json>
           [--port 8767] [--steps 0=auto] [--out reports/x/samples.jsonl]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

KEEP = ("cycle", "cycles_completed", "cycle_index", "priority_scalar", "pain_scalar", "priority_label")


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scen = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    steps = args.steps or int(scen.get("total_steps_hint", 4000))
    base = f"http://{args.host}:{args.port}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = subprocess.Popen(
        [
            sys.executable,
            "scripts/synthetic_ws_client.py",
            "--port", str(args.port),
            "--agent-id", args.agent_id,
            "--steps", str(steps),
            "--rate", str(args.rate),
            "--log-every", "0",
            "--binding-scenario", args.scenario,
        ]
    )
    n = 0
    t0 = time.time()
    try:
        with out.open("w", encoding="utf-8") as f:
            while client.poll() is None:
                m = _get(f"{base}/agent/{args.agent_id}/metrics")
                if isinstance(m, dict):
                    src = m.get("metrics") if isinstance(m.get("metrics"), dict) else m
                    row = {k: src.get(k) for k in KEEP if src.get(k) is not None}
                    row["t"] = round(time.time() - t0, 2)
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    if n % 120 == 0:
                        f.flush()
                        print(
                            f"[probe] t={row['t']:.0f}s samples={n} "
                            f"prio={row.get('priority_scalar')}",
                            flush=True,
                        )
                time.sleep(args.poll_s)
    finally:
        if client.poll() is None:
            client.terminate()
    print(f"[probe] done: {n} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
