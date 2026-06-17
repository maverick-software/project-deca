#!/usr/bin/env python3
"""Optional MineRL / Gymnasium → Decadic WebSocket bridge.

Produces the same JSON observation shape as ``mujoco_decadic_adapter`` / ``ObservationMessage``.
MineRL and Minecraft assets are heavy to install; use ``--dry-run`` to verify Decadic only.

Example::

    python scripts/minerl_decadic_adapter.py --dry-run --steps 30

With a real Gymnasium env (you wire ``make_env()``)::

    python scripts/minerl_decadic_adapter.py --env-id MineRLNavigateDense-v0 --steps 100

Requires ``websockets`` (dev extra). MineRL itself is optional and must be installed separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import urllib.request
from typing import Any


def _post_agent(base_http: str) -> str:
    req = urllib.request.Request(f"{base_http.rstrip('/')}/agent", method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["agent_id"])


def _dry_observation(step: int) -> dict[str, Any]:
    """Synthetic miner-ish observation for connectivity tests."""
    t = step * 0.05
    pos = [math.cos(t), 64.0, math.sin(t)]
    return {
        "timestamp": "2026-05-07T12:00:00Z",
        "proprioception": {
            "position": pos,
            "orientation": [0.0, math.degrees(t) % 360.0, 0.0],
            "velocity": [0.01, 0.0, 0.01],
            "current_action": "minerl_stub",
        },
        "events": [],
        "world_state": {
            "agent": {"id": "self", "position": pos, "orientation": [0.0, 0.0, 0.0]},
            "entities": [
                {
                    "id": "decadic-entity-block_stub",
                    "kind": "block",
                    "position": [pos[0] + 2.0, pos[1], pos[2]],
                    "relative": [2.0, 0.0, 0.0],
                }
            ],
            "nearby_entities": [],
            "agent_inventory": [],
        },
    }


def _observation_from_gym(obs: Any, step: int) -> dict[str, Any]:
    """Map a Gymnasium observation to Decadic JSON (heuristic; tune per env-id)."""
    _ = obs  # Extend: MineRL-specific POV → vision.base64_jpeg, compass → orientation, etc.

    pos = [float(step) * 0.01, 64.0, 0.0]
    return {
        "timestamp": "2026-05-07T12:00:00Z",
        "proprioception": {
            "position": pos,
            "orientation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "current_action": "minerl_gym",
        },
        "events": [],
        "world_state": {
            "agent": {"id": "self", "position": pos, "orientation": [0.0, 0.0, 0.0]},
            "entities": [],
            "nearby_entities": [],
            "agent_inventory": [],
        },
    }


async def _run_ws_loop(
    *,
    ws_url: str,
    steps: int,
    dry_run: bool,
    env_id: str | None,
) -> None:
    import websockets

    async with websockets.connect(ws_url, max_size=None) as ws:
        if dry_run or env_id is None:
            for i in range(steps):
                await ws.send(json.dumps(_dry_observation(i)))
                await ws.recv()
            return

        try:
            import gymnasium as gym
        except ImportError as e:
            raise RuntimeError("Install gymnasium (and MineRL) for --env-id mode") from e

        env = gym.make(env_id)
        obs, _ = env.reset()
        for i in range(steps):
            await ws.send(json.dumps(_observation_from_gym(obs, i)))
            raw = await ws.recv()
            msg = json.loads(raw)
            action = msg.get("action") or {}
            _ = action  # discrete mapping would go here: env.step(discrete_idx)
            obs, _reward, term, trunc, _info = env.step(0)
            if term or trunc:
                obs, _ = env.reset()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MineRL-style bridge to Decadic WebSocket.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="No Gym/MineRL; synthetic obs")
    parser.add_argument("--env-id", default=None, help="Gymnasium env id if MineRL installed")
    args = parser.parse_args()

    scheme_http = "https" if args.ssl else "http"
    scheme_ws = "wss" if args.ssl else "ws"
    base_http = f"{scheme_http}://{args.host}:{args.port}"
    aid = _post_agent(base_http)
    ws_url = f"{scheme_ws}://{args.host}:{args.port}/agent/{aid}/cycle"

    asyncio.run(
        _run_ws_loop(
            ws_url=ws_url,
            steps=max(1, args.steps),
            dry_run=args.dry_run,
            env_id=args.env_id,
        )
    )


if __name__ == "__main__":
    main()
