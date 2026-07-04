"""WS3: offline replay of the gate's percept-novelty computation.

For every stored episode (cycle c), computes novelty = 1 - max cosine of its
percept key against all keys older than c - HORIZON -- exactly what the gate
sees online. Prints ambient percentiles, event-window peaks, and the
manifold-saturation curve (how the per-cycle best-match similarity grows as
the corpus fills in). Decides between: (a) events genuinely have no old
neighbor (channel fixable by calibration/thresholds) vs (b) the key manifold
saturates within warmup and max-similarity novelty is structurally dead under
synthetic input.

Usage: .venv\\Scripts\\python.exe scripts\\simulate_gate_novelty.py <agent-id> [horizon] [data-dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVENTS = (700, 1400, 2100, 2800)
WINDOW = 60


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    aid = sys.argv[1]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data")

    import lancedb

    hits = list(data_dir.glob(f"*{aid}*episodes*.lance"))
    if not hits:
        print(f"no lance table for agent {aid} under {data_dir.resolve()}")
        return 1
    db = lancedb.connect(str(hits[0]))
    at = db.open_table("episodes").to_arrow()
    keys = np.asarray(
        at["percept_key"].combine_chunks().flatten(), dtype=np.float32
    ).reshape(-1, 16)
    cycles = np.asarray(at["cycle_index"], dtype=np.int64)
    order = np.argsort(cycles)
    keys, cycles = keys[order], cycles[order]
    norms = np.linalg.norm(keys, axis=1)
    live = norms > 1e-8
    keys, cycles = keys[live], cycles[live]
    kn = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    n = len(kn)
    print(f"rows={n} horizon={horizon}")

    # Full pairwise similarity (n~3k -> ~9M floats, fine), then per-row max
    # over strictly-older-than-horizon rows.
    sims = kn @ kn.T
    nov = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        old = cycles < (cycles[i] - horizon)
        if old.any():
            nov[i] = 1.0 - float(sims[i, old].max())
        else:
            nov[i] = 1.0  # nothing beyond horizon: fully novel (gate: None)
    valid = ~np.isnan(nov)
    post = valid & (cycles >= 300)
    pool = np.sort(nov[post]) if post.any() else np.sort(nov[valid])

    def pct(p: float) -> float:
        return float(pool[min(len(pool) - 1, int(p * (len(pool) - 1)))])

    print(
        "post-warmup novelty: "
        f"p50={pct(0.5):.4f} p95={pct(0.95):.4f} p99={pct(0.99):.4f} max={pool[-1]:.4f}"
    )
    for ev in EVENTS:
        w = post & (cycles >= ev) & (cycles <= ev + WINDOW)
        if not w.any():
            print(f"event @{ev}: no rows")
            continue
        peak = float(nov[w].max())
        peak_c = int(cycles[w][int(np.argmax(nov[w]))])
        print(f"event @{ev}: peak novelty {peak:.4f} @cycle {peak_c} (rows {int(w.sum())})")

    # Saturation curve: best old-match similarity as the corpus grows.
    print("saturation (per-cycle best old-match similarity, medians per band):")
    for lo, hi in ((300, 600), (600, 1000), (1000, 1500), (1500, 2200), (2200, 3400)):
        band = valid & (cycles >= lo) & (cycles < hi)
        if band.any():
            best_sim = 1.0 - nov[band]
            print(
                f"  cycles {lo:>4}-{hi:<4}: p50={np.median(best_sim):.5f} "
                f"p05={np.percentile(best_sim, 5):.5f} min={best_sim.min():.5f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
