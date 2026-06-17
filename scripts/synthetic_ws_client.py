"""Synthetic observation stream over WebSocket (Phase 1 plan / brief harness).

Requires ``websockets`` (included with ``pip install -e ".[dev]"``).

Example::

    python scripts/synthetic_ws_client.py --host 127.0.0.1 --port 8765 --steps 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

import websockets


def _post_agent(base_http: str) -> str:
    req = urllib.request.Request(f"{base_http.rstrip('/')}/agent", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["agent_id"])


def _observation(step: int) -> dict:
    return {
        "timestamp": f"2026-05-07T12:{step % 60:02d}:00Z",
        "proprioception": {
            "position": [float(step) * 0.01, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.1, 0.0, 0.0],
            "current_action": "walking_forward",
        },
        "events": (
            [{"type": "collision", "intensity": 0.85, "source": "wall_test"}]
            if step == 5
            else []
        ),
        "world_state": {"nearby_entities": [], "agent_inventory": []},
    }


async def _run(host: str, port: int, steps: int, ssl: bool) -> None:
    scheme_http = "https" if ssl else "http"
    scheme_ws = "wss" if ssl else "ws"
    base_http = f"{scheme_http}://{host}:{port}"
    aid = _post_agent(base_http)
    ws_url = f"{scheme_ws}://{host}:{port}/agent/{aid}/cycle"
    async with websockets.connect(ws_url, max_size=None) as ws:
        for i in range(steps):
            await ws.send(json.dumps(_observation(i)))
            raw = await ws.recv()
            msg = json.loads(raw)
            print(json.dumps({"step": i, "action": msg.get("action")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream synthetic observations to Decadic server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--ssl", action="store_true", help="Use wss/https URLs")
    args = parser.parse_args()
    asyncio.run(_run(args.host, args.port, args.steps, args.ssl))


if __name__ == "__main__":
    main()
