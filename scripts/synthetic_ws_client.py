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


async def _run(
    host: str,
    port: int,
    steps: int,
    ssl: bool,
    agent_id: str | None = None,
    rate_s: float = 0.0,
    log_every: int = 1,
) -> None:
    scheme_http = "https" if ssl else "http"
    scheme_ws = "wss" if ssl else "ws"
    base_http = f"{scheme_http}://{host}:{port}"
    aid = agent_id or _post_agent(base_http)
    print(json.dumps({"agent_id": aid}), flush=True)
    ws_url = f"{scheme_ws}://{host}:{port}/agent/{aid}/cycle"

    # Sender and receiver are deliberately decoupled: the server does NOT
    # guarantee one action message per observation (serial prefetch commits
    # cycles on its own schedule), so a lock-step send/recv client deadlocks
    # around the point the pipeline stops replying 1:1.
    async with websockets.connect(ws_url, max_size=None) as ws:
        recv_count = 0

        async def _receiver() -> None:
            nonlocal recv_count
            async for raw in ws:
                msg = json.loads(raw)
                if log_every > 0 and recv_count % log_every == 0:
                    print(
                        json.dumps({"recv": recv_count, "action": msg.get("action")}),
                        flush=True,
                    )
                recv_count += 1

        async def _sender() -> None:
            for i in range(steps):
                await ws.send(json.dumps(_observation(i)))
                if log_every > 0 and i > 0 and i % log_every == 0:
                    print(json.dumps({"sent": i, "received": recv_count}), flush=True)
                await asyncio.sleep(rate_s if rate_s > 0 else 0.01)

        recv_task = asyncio.create_task(_receiver())
        try:
            await _sender()
        finally:
            recv_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream synthetic observations to Decadic server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--ssl", action="store_true", help="Use wss/https URLs")
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Attach to an existing agent instead of creating a new one",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Seconds to sleep between observations (0 = lock-step as fast as the server replies)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print every Nth action (0 = silent stream)",
    )
    args = parser.parse_args()
    asyncio.run(
        _run(
            args.host,
            args.port,
            args.steps,
            args.ssl,
            agent_id=args.agent_id,
            rate_s=args.rate,
            log_every=args.log_every,
        )
    )


if __name__ == "__main__":
    main()
