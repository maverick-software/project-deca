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


NOVEL_BURST_STEPS = 25  # how many steps a "novel" episode perturbs the stream


def parse_events(spec: str | None) -> dict[int, str]:
    """Parse an event spec like "collision:600,novel:1200,revisit:2400"
    into {step: kind}. Kinds: collision (fast-path threat), novel
    (out-of-distribution burst at a target UNIQUE to its start step),
    revisit (same burst dynamics at the FIRST novel event's target -- the
    agent has genuinely seen this place before, so a healthy memory-backed
    novelty channel must NOT spike; habituation-across-events is the
    assertion, probe redesign 2026-07-04). Used by the gate_probe scenario."""
    out: dict[int, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kind, _, step_s = part.partition(":")
        kind = kind.strip().lower()
        if kind in ("collision", "novel", "revisit") and step_s.strip().isdigit():
            out[int(step_s.strip())] = kind
    return out


def _novel_target(seed_step: int) -> tuple[list[float], list[float], list[float]]:
    """Deterministic OOD (position0, per-step stride, velocity), unique per
    start step. Distinct novel events must land in distinct regions --
    otherwise the second event is a de-facto revisit, and correct episodic
    memory (full-corpus recall) rightly reads it as familiar, which is what
    made the pre-redesign probe unpassable."""
    import random as _random

    rng = _random.Random(9700 + int(seed_step))
    pos0 = [
        rng.choice((-1.0, 1.0)) * rng.uniform(300.0, 900.0),
        rng.uniform(5.0, 60.0),
        rng.choice((-1.0, 1.0)) * rng.uniform(150.0, 600.0),
    ]
    stride = [rng.uniform(1.5, 4.5), 0.0, rng.uniform(2.0, 6.0)]
    vel = [rng.uniform(-4.0, 4.0), rng.uniform(0.5, 3.0), rng.uniform(-4.0, 4.0)]
    return pos0, stride, vel


LAP_STEPS = 200  # closed patrol loop: the world repeats, so monotony is real


# --- WS5-M0.4: binding-probe scenario (scripted multi-entity world) ----------


def load_binding_scenario(path: str | None) -> dict | None:
    """Load a binding scenario file: controlled entities + relation schedule.

    Shape (see scripts/gen_binding_scenario.py):
      entities: [{id, kind, appearance[16], home[3], orbit, period}]
      schedule: [{start, steps, pair:[a,b], gap, event:{type,intensity,every}}]
    The schedule is the serialized relation ground truth (WBS M0.4
    acceptance); train/holdout pairing metadata rides in the same file for
    the M5 verdict script and is ignored here.
    """
    if not path:
        return None
    import pathlib

    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _binding_world(step: int, scenario: dict) -> tuple[list[dict], list[dict]]:
    """Entity list + extra events for this step under the relation schedule.

    Default motion: each entity orbits its home slowly (distinct, boring,
    familiar). During a scheduled phase the pair's second entity is placed
    ADJACENT to the first (gap units away) -- the spatial relation whose
    generalization the probe tests -- and the phase's event fires every
    ``every`` steps, sourced to the first entity of the pair.
    """
    import math as _math

    ents = {e["id"]: e for e in scenario.get("entities", [])}
    positions: dict[str, list[float]] = {}
    for i, (eid, e) in enumerate(sorted(ents.items())):
        home = e.get("home", [0.0, 0.0, 0.0])
        orbit = float(e.get("orbit", 2.0))
        period = max(8, int(e.get("period", 400)))
        ph = 2.0 * _math.pi * ((step + i * 37) % period) / period
        positions[eid] = [
            home[0] + orbit * _math.cos(ph),
            home[1],
            home[2] + orbit * _math.sin(ph),
        ]

    events: list[dict] = []
    for phase in scenario.get("schedule", []):
        start = int(phase.get("start", -1))
        length = int(phase.get("steps", 0))
        if not (start <= step < start + length):
            continue
        a, b = phase.get("pair", [None, None])[:2]
        if a in positions and b in positions:
            gap = float(phase.get("gap", 1.5))
            pa = positions[a]
            positions[b] = [pa[0] + gap, pa[1], pa[2]]
            ev = phase.get("event") or {}
            every = max(1, int(ev.get("every", 10)))
            if ev.get("type") and (step - start) % every == 0:
                events.append(
                    {
                        "type": str(ev["type"]),
                        "intensity": float(ev.get("intensity", 0.6)),
                        "entity": str(a),
                        "source": str(a),
                    }
                )
        break  # phases are non-overlapping by construction

    out_entities = []
    for eid, e in sorted(ents.items()):
        out_entities.append(
            {
                "id": eid,
                "kind": str(e.get("kind", "entity")),
                "position": positions[eid],
                "appearance": list(e.get("appearance", [])),
            }
        )
    return out_entities, events


def _observation(
    step: int,
    events: dict[int, str] | None = None,
    scenario: dict | None = None,
) -> dict:
    import math as _math

    obs_events: list[dict] = []
    scenario_entities: list[dict] = []
    if scenario is not None:
        scenario_entities, scen_events = _binding_world(step, scenario)
        obs_events.extend(scen_events)
    # Bounded circular walk instead of an unbounded line: an infinite straight
    # walk makes every observation mildly novel forever (episodic similarity
    # never rises), which defeats habituation and any steady-state-calm
    # measurement. On a loop, experience recurs and novelty genuinely decays.
    phase = 2.0 * _math.pi * (step % LAP_STEPS) / LAP_STEPS
    position = [10.0 * _math.cos(phase), 0.0, 10.0 * _math.sin(phase)]
    velocity = [-_math.sin(phase) * 0.3, 0.0, _math.cos(phase) * 0.3]
    action = "walking_forward"
    if events:
        if events.get(step) == "collision":
            obs_events.append(
                {"type": "collision", "intensity": 0.9, "source": "probe_injected"}
            )
        # A novel episode perturbs the stream for a window of steps: the body
        # is suddenly somewhere else, moving differently. Each novel event
        # uses a target UNIQUE to its start step (first exposure -> novelty
        # spike); a revisit event replays the FIRST novel target (seen before
        # -> a healthy memory-backed channel stays quiet).
        for start, kind in sorted(events.items()):
            if kind in ("novel", "revisit") and start <= step < start + NOVEL_BURST_STEPS:
                if kind == "revisit":
                    novel_starts = sorted(
                        s for s, kd in events.items() if kd == "novel"
                    )
                    seed = novel_starts[0] if novel_starts else start
                else:
                    seed = start
                pos0, stride, velocity = _novel_target(seed)
                k = step - start
                position = [
                    pos0[0] + stride[0] * k,
                    pos0[1],
                    pos0[2] + stride[2] * ((k * 7) % 11),
                ]
                action = "falling"
                break
    elif step == 5:
        # Legacy default: one collision early (kept for existing harness runs).
        obs_events.append({"type": "collision", "intensity": 0.85, "source": "wall_test"})
    return {
        "timestamp": f"2026-05-07T12:{step % 60:02d}:00Z",
        "proprioception": {
            "position": position,
            "orientation": [0.0, 0.0, 0.0],
            "velocity": velocity,
            "current_action": action,
        },
        "events": obs_events,
        # NB: the oracle seam reads world_state.entities (world_graph.py);
        # "nearby_entities" was never consumed and is kept only for schema
        # compatibility with old harness recordings.
        "world_state": {
            "entities": scenario_entities,
            "nearby_entities": [],
            "agent_inventory": [],
        },
    }


async def _run(
    host: str,
    port: int,
    steps: int,
    ssl: bool,
    agent_id: str | None = None,
    rate_s: float = 0.0,
    log_every: int = 1,
    events: dict[int, str] | None = None,
    scenario: dict | None = None,
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
    #
    # Long-run resilience: the server event loop can block for seconds under
    # load (heavy /state snapshots, consolidation bursts), which trips the
    # default 20s keepalive and kills the stream. Use a generous ping timeout
    # and reconnect with backoff instead of dying - a soak driver must behave
    # like a real environment and keep offering observations.
    recv_count = 0
    sent = 0
    reconnects = 0
    while sent < steps and reconnects < 100:
        try:
            async with websockets.connect(
                ws_url, max_size=None, ping_interval=20, ping_timeout=120
            ) as ws:

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
                    nonlocal sent
                    while sent < steps:
                        await ws.send(json.dumps(_observation(sent, events, scenario)))
                        sent += 1
                        if log_every > 0 and sent % log_every == 0:
                            print(
                                json.dumps({"sent": sent, "received": recv_count}), flush=True
                            )
                        await asyncio.sleep(rate_s if rate_s > 0 else 0.01)

                recv_task = asyncio.create_task(_receiver())
                try:
                    await _sender()
                finally:
                    recv_task.cancel()
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            reconnects += 1
            print(
                json.dumps({"reconnect": reconnects, "after_sent": sent, "reason": str(e)[:120]}),
                flush=True,
            )
            await asyncio.sleep(min(30.0, 1.0 * reconnects))


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
    parser.add_argument(
        "--events",
        default=None,
        help='Injected stimuli for the gate probe, e.g. "collision:600,novel:1200,collision:1800"',
    )
    parser.add_argument(
        "--binding-scenario",
        default=None,
        help="WS5-M0.4: path to a binding scenario JSON (scripted entities + "
        "relation schedule; see scripts/gen_binding_scenario.py)",
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
            events=parse_events(args.events),
            scenario=load_binding_scenario(args.binding_scenario),
        )
    )


if __name__ == "__main__":
    main()
