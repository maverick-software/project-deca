"""WS3: novelty-channel distribution from a gate-probe sample file.

Prints ambient novelty percentiles (post-warmup), the max, and the novelty
trace in a window around each injected event cycle -- enough to tell WHETHER
the injected events moved the channel at all (verdict said 0 high-novelty
samples) and how much headroom the ambient baseline leaves.

Usage: .venv\\Scripts\\python.exe scripts\\inspect_gate_novelty.py <samples.jsonl> [event_cycles...]
       (event cycles default to 700 1400 2100 2800)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WARMUP_CYCLES = 300
WINDOW = 60  # cycles of context on each side of an event

NOVELTY_KEYS = ("gate_i_novelty", "novelty", "gate_novelty")
CYCLE_KEYS = ("cycle_index", "cycle", "cycles")


def _get(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
        m = d.get("metrics")
        if isinstance(m, dict) and k in m:
            return m[k]
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    events = [int(x) for x in sys.argv[2:]] or [700, 1400, 2100, 2800]

    rows: list[tuple[int, float]] = []
    other_keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        cyc = _get(d, CYCLE_KEYS)
        nov = _get(d, NOVELTY_KEYS)
        if cyc is None:
            continue
        if nov is None:
            src = d.get("metrics") if isinstance(d.get("metrics"), dict) else d
            other_keys.update(k for k in src if "novel" in k.lower() or k.startswith("gate_"))
            continue
        rows.append((int(cyc), float(nov)))

    if not rows:
        print("no novelty samples found; gate-ish keys seen:", sorted(other_keys))
        return 1

    rows.sort()
    post = [n for c, n in rows if c >= WARMUP_CYCLES]
    pool = sorted(post if post else [n for _, n in rows])

    def pct(p: float) -> float:
        return pool[min(len(pool) - 1, int(p * (len(pool) - 1)))]

    print(f"samples={len(rows)} post-warmup={len(post)} (warmup<{WARMUP_CYCLES})")
    print(
        "ambient novelty: "
        f"p05={pct(0.05):.4f} p25={pct(0.25):.4f} p50={pct(0.50):.4f} "
        f"p75={pct(0.75):.4f} p95={pct(0.95):.4f} p99={pct(0.99):.4f} "
        f"max={pool[-1]:.4f}"
    )
    for ev in events:
        win = [(c, n) for c, n in rows if ev - WINDOW <= c <= ev + 2 * WINDOW]
        if not win:
            print(f"event @{ev}: no samples in window")
            continue
        peak_c, peak_n = max(win, key=lambda t: t[1])
        pre = [n for c, n in win if c < ev]
        base = sum(pre) / len(pre) if pre else float("nan")
        print(
            f"event @{ev}: window n={len(win)} pre-mean={base:.4f} "
            f"peak={peak_n:.4f} @cycle {peak_c} (delta={peak_n - base:+.4f})"
        )
    # Also useful: where the gate stood (top-5 novelty samples overall).
    top = sorted(rows, key=lambda t: -t[1])[:5]
    print("top-5 novelty samples:", ", ".join(f"{n:.4f}@{c}" for c, n in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
