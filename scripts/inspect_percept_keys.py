"""WS3: are stored percept keys actually distinct, or (near-)constant?

Loads the agent's lance episodes table and reports the 16-d percept-key
geometry: norms, per-dimension std, pairwise cosine of random pairs, and
consecutive-cycle cosine. Novelty == 0 everywhere (probe 20260704) implies
best similarity ~= 1.0 at every cycle; if the keys here are near-identical,
the channel's input is degenerate BEFORE any search/gate logic runs.

Usage: .venv\\Scripts\\python.exe scripts\\inspect_percept_keys.py <agent-id> [data-dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    aid = sys.argv[1]
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")

    import lancedb

    hits = list(data_dir.glob(f"*{aid}*episodes*.lance"))
    if not hits:
        print(f"no lance table dir matching agent {aid} under {data_dir.resolve()}")
        print("candidates:", [p.name for p in data_dir.glob("*.lance")][:10])
        return 1
    # The *.lance directory IS the lancedb database (one per agent store),
    # holding a single table named "episodes" -- see LanceEpisodicStore.
    tdir = hits[0]
    db = lancedb.connect(str(tdir))
    tbl = db.open_table("episodes")
    at = tbl.to_arrow()
    keys = np.asarray(at["percept_key"].combine_chunks().flatten(), dtype=np.float32)
    keys = keys.reshape(-1, 16)
    cycles = np.asarray(at["cycle_index"], dtype=np.int64)
    has = np.asarray(at["has_embedding"], dtype=bool)
    order = np.argsort(cycles)
    keys, cycles, has = keys[order], cycles[order], has[order]

    n = len(keys)
    norms = np.linalg.norm(keys, axis=1)
    live = norms > 1e-8
    print(f"rows={n} has_embedding={int(has.sum())} nonzero-key rows={int(live.sum())}")
    if live.sum() < 2:
        print("keys are (near-)all zero -- the percept slice is never populated.")
        return 0

    k = keys[live]
    kn = k / np.linalg.norm(k, axis=1, keepdims=True)
    print(
        f"key norms: min={norms[live].min():.4f} p50={np.median(norms[live]):.4f} "
        f"max={norms[live].max():.4f}"
    )
    dim_std = k.std(axis=0)
    print(
        "per-dim std: "
        + " ".join(f"{s:.4f}" for s in dim_std)
        + f"  (mean {dim_std.mean():.5f})"
    )

    rng = np.random.default_rng(3)
    m = min(2000, len(kn))
    a = kn[rng.integers(0, len(kn), m)]
    b = kn[rng.integers(0, len(kn), m)]
    pair_cos = np.sum(a * b, axis=1)
    print(
        f"random-pair cosine: p05={np.percentile(pair_cos, 5):.5f} "
        f"p50={np.percentile(pair_cos, 50):.5f} min={pair_cos.min():.5f}"
    )
    consec = np.sum(kn[1:] * kn[:-1], axis=1)
    print(
        f"consecutive cosine: p05={np.percentile(consec, 5):.5f} "
        f"p50={np.percentile(consec, 50):.5f} min={consec.min():.5f} "
        f"@cycle {cycles[live][1:][int(np.argmin(consec))]}"
    )
    # Event windows: most-dissimilar key vs the 50 cycles before it.
    for ev in (700, 1400, 2100, 2800):
        w = (cycles[live] >= ev - 50) & (cycles[live] <= ev + 100)
        pre = (cycles[live] >= ev - 50) & (cycles[live] < ev)
        if w.sum() < 3 or pre.sum() < 3:
            print(f"event @{ev}: too few rows")
            continue
        ref = kn[pre].mean(axis=0)
        ref /= np.linalg.norm(ref)
        cos_to_ref = kn[w] @ ref
        cyc_w = cycles[live][w]
        i = int(np.argmin(cos_to_ref))
        print(
            f"event @{ev}: min cosine-to-pre-baseline {cos_to_ref.min():.5f} "
            f"@cycle {cyc_w[i]} (window p50 {np.median(cos_to_ref):.5f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
