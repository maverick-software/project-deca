"""Correlate resource feeding against cognitive tempo and deliberation.

Reads a diag/soak run directory and answers the question the throughput curve
can only hint at: when a resource was GRANTED and later RETRIEVED, did the
agent's cycle rate recover and did it stop escalating into full deliberation
("inquisitive moments")?

Inputs it looks for in <run_dir> (all optional; it uses what's present):
  - decadic_server.jsonl / server.err.log  -> "resource_provision ... cycle=N",
        "nourishment ... cycle=N ... viability=V", and
        "cycle_completed ... cycle=N wall_ms=MS" lines.
  - gate_decisions_*.jsonl                  -> per-cycle {cycle, escalate,
        viability, drive, pc_ema} (only present when DECADIC_GATE_LOG=1, i.e.
        watched runs).

Outputs, per feeding event, a before/after window comparison of cycles-per-
second (from cycle wall_ms), escalation rate, and viability -- plus an overall
viability<->escalation correlation. Pure stdlib.

Usage:  python scripts/analyze_feeding.py reports/bodydiag_kuzu_<stamp> [--window 150]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

_RE_PROVISION = re.compile(
    r"resource_provision .*?cycle=(\d+) kind=(\w+) mode=(\w+)"
)
_RE_NOURISH = re.compile(
    r"nourishment .*?cycle=(\d+).*?viability=([0-9.]+)"
)
_RE_CYCLE = re.compile(r"cycle_completed .*?cycle=(\d+) wall_ms=([0-9.]+)")


def _iter_message_lines(run_dir: str):
    """Yield log 'message' strings from the server logs (JSONL or raw)."""
    for name in ("decadic_server.jsonl", "server.err.log", "server.out.log"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Structured logs carry the text under "message"; raw logs are
                # already the text.
                if line.startswith("{"):
                    try:
                        yield json.loads(line).get("message", "")
                        continue
                    except json.JSONDecodeError:
                        pass
                yield line


def _load_events_and_tempo(run_dir: str):
    grants = []  # (cycle, kind, mode)
    retrievals = []  # (cycle, viability)
    wall_by_cycle = {}  # cycle -> wall_ms
    for msg in _iter_message_lines(run_dir):
        m = _RE_PROVISION.search(msg)
        if m:
            grants.append((int(m.group(1)), m.group(2), m.group(3)))
            continue
        m = _RE_NOURISH.search(msg)
        if m:
            retrievals.append((int(m.group(1)), float(m.group(2))))
            continue
        m = _RE_CYCLE.search(msg)
        if m:
            wall_by_cycle[int(m.group(1))] = float(m.group(2))
    return grants, retrievals, wall_by_cycle


def _load_gate(run_dir: str):
    """cycle -> {escalate, viability, drive, pc_ema} from the gate log."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "gate_decisions_*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = r.get("cycle")
                if c is not None:
                    out[int(c)] = r
    return out


def _rate_in_window(wall_by_cycle, lo, hi):
    """Mean cycles/s from wall_ms over cycles in [lo, hi)."""
    ms = [wall_by_cycle[c] for c in range(lo, hi) if c in wall_by_cycle]
    if not ms:
        return None
    mean_ms = statistics.fmean(ms)
    return 1000.0 / mean_ms if mean_ms > 0 else None


def _escalation_rate(gate, lo, hi):
    vals = [gate[c].get("escalate") for c in range(lo, hi) if c in gate]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _mean_field(gate, lo, hi, field):
    vals = [gate[c].get(field) for c in range(lo, hi) if c in gate]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.fmean(vals) if vals else None


def _fmt(x, suffix=""):
    return "n/a" if x is None else f"{x:.2f}{suffix}"


def _delta(before, after):
    if before is None or after is None:
        return ""
    d = after - before
    arrow = "up" if d > 0 else ("down" if d < 0 else "flat")
    return f"  ({arrow} {d:+.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="reports/bodydiag_<...> directory")
    ap.add_argument(
        "--window", type=int, default=150,
        help="cycles before/after each event to compare (default 150)",
    )
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        print(f"not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    grants, retrievals, wall = _load_events_and_tempo(args.run_dir)
    gate = _load_gate(args.run_dir)
    W = args.window

    print(f"== feeding analysis: {os.path.basename(args.run_dir)} ==")
    print(
        f"grants={len(grants)}  retrievals={len(retrievals)}  "
        f"cycles_with_timing={len(wall)}  gate_rows={len(gate)}  window=±{W}\n"
    )
    if not grants and not retrievals:
        print("No resource_provision/nourishment events found.")
        print("(Feeding needs DECADIC_ALLOW_EXTERNAL_BODY_PROVISION=1 and the")
        print(" dashboard Give buttons; retrievals need the agent to consume.)")

    events = [(c, f"GRANT {k}/{m}") for (c, k, m) in grants]
    events += [(c, f"RETRIEVE (viab={v:.1f})") for (c, v) in retrievals]
    events.sort()
    for cyc, label in events:
        r_before = _rate_in_window(wall, cyc - W, cyc)
        r_after = _rate_in_window(wall, cyc, cyc + W)
        e_before = _escalation_rate(gate, cyc - W, cyc)
        e_after = _escalation_rate(gate, cyc, cyc + W)
        v_before = _mean_field(gate, cyc - W, cyc, "viability")
        v_after = _mean_field(gate, cyc, cyc + W, "viability")
        print(f"cycle {cyc:>6}  {label}")
        print(f"    cycles/s : {_fmt(r_before)} -> {_fmt(r_after)}{_delta(r_before, r_after)}")
        print(f"    escalate%: {_fmt(None if e_before is None else e_before*100,'%')} -> "
              f"{_fmt(None if e_after is None else e_after*100,'%')}"
              f"{_delta(None if e_before is None else e_before*100, None if e_after is None else e_after*100)}")
        print(f"    viability: {_fmt(v_before)} -> {_fmt(v_after)}{_delta(v_before, v_after)}\n")

    # Overall: does deliberation track (inverse) viability across the whole run?
    if gate:
        pts = [(g.get("viability"), g.get("escalate")) for g in gate.values()]
        pts = [(v, e) for (v, e) in pts if isinstance(v, (int, float)) and e is not None]
        if len(pts) > 50:
            vs = [v for v, _ in pts]
            es = [1.0 if e else 0.0 for _, e in pts]
            try:
                r = statistics.correlation(vs, es)  # py>=3.10
                print(f"overall viability<->escalate correlation: r={r:+.3f}  "
                      f"(negative = lower viability -> more deliberation, as theorized)")
            except (ValueError, AttributeError):
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
