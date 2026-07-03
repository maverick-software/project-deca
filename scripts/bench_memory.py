"""WS4-M0.4/M2: memory benchmark harness (torch-free).

Generates synthetic episodes into a backend selected via the WS4 factory seam
and measures add throughput, search_similar / search_similar_percept latency
percentiles, and process RSS. With ``--graph-backend`` it also benchmarks the
semantic graph: node add rate and identity-match (``match``) latency p50/p95
at ``--n`` observations. Writes ``reports/ws4_bench_<...>_<n>.json`` and
prints a one-line summary per section.

Usage (from the repo root, inside the project venv):

    python scripts/bench_memory.py --backend sqlite  --n 10000
    python scripts/bench_memory.py --backend lancedb --n 100000
    python scripts/bench_memory.py --graph-backend sqlite --graph-only --n 10000
    python scripts/bench_memory.py --graph-backend kuzu   --graph-only --n 10000

Notes:
- DB retention pruning is disabled by default so row counts are stable across
  backends and sizes; pass --retention to benchmark with production pruning.
- The sqlite backend's recall path is the in-memory cache (recent+salient
  caps), which is exactly what production measures today; the JSON records the
  cache stats so the curves can be interpreted honestly.
- The graph match path is likewise fronted by the in-memory match cache in
  production; pass --no-graph-match-cache to measure the raw engine path
  (sqlite linear scan vs the kuzu native vector index).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

GRAPH_APPEARANCE_DIM = 16  # production appearance-fingerprint dim (perception/organ.py)


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }


def _bench_episodic(args: argparse.Namespace, base: Path) -> dict:
    """Episodic add/search bench (WS4-M0.4; unchanged behavior)."""
    from decadic.memory.embeddings import EMBEDDING_DIM, PERCEPT_KEY_SLICE
    from decadic.memory.episodic_store import EpisodicRecord
    from decadic.memory.factory import make_episodic_store

    key_dim = PERCEPT_KEY_SLICE.stop - PERCEPT_KEY_SLICE.start
    rng = np.random.default_rng(args.seed)
    n = int(args.n)
    db_path = base / f"bench_{args.backend}_{n}.sqlite"

    store = make_episodic_store(db_path)
    print(
        f"[bench_memory] backend={args.backend} n={n} dim={EMBEDDING_DIM} "
        f"db={db_path}",
        flush=True,
    )

    # ---- add throughput -------------------------------------------------
    chunk = 10_000
    added = 0
    t0 = time.perf_counter()
    while added < n:
        m = min(chunk, n - added)
        embs = rng.standard_normal((m, EMBEDDING_DIM)).astype(np.float32)
        keys = embs[:, PERCEPT_KEY_SLICE]
        norms = np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
        embs[:, PERCEPT_KEY_SLICE] = keys / norms  # unit percept keys (as production)
        saliences = rng.uniform(0.0, 1.0, size=m)
        for j in range(m):
            i = added + j
            store.append(
                EpisodicRecord(
                    cycle_index=i,
                    summary={"i": i},
                    salience=float(saliences[j]),
                    embedding=embs[j].tolist(),
                )
            )
        added += m
        if added % 100_000 == 0 or added == n:
            print(f"[bench_memory]   added {added}/{n}", flush=True)
    flush = getattr(store, "flush", None)
    if callable(flush):
        flush()
    add_s = time.perf_counter() - t0
    add_rate = n / add_s if add_s > 0 else 0.0

    # ---- search_similar latency -----------------------------------------
    full_ms: list[float] = []
    for _ in range(int(args.queries)):
        q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        t = time.perf_counter()
        store.search_similar(q, top_k=int(args.top_k))
        full_ms.append((time.perf_counter() - t) * 1000.0)

    # ---- search_similar_percept latency ----------------------------------
    percept_ms: list[float] = []
    for _ in range(int(args.queries)):
        key = rng.standard_normal(key_dim).astype(np.float32)
        key /= max(1e-8, float(np.linalg.norm(key)))
        t = time.perf_counter()
        store.search_similar_percept(key, top_k=int(args.top_k))
        percept_ms.append((time.perf_counter() - t) * 1000.0)

    metrics = {}
    try:
        metrics = dict(store.persistence_metrics())
    except Exception:
        pass
    cache_stats = {}
    try:
        cache_stats = dict(store.recall_cache_stats())
    except Exception:
        pass

    section = {
        "backend": args.backend,
        "n_episodes": n,
        "embedding_dim": int(EMBEDDING_DIM),
        "percept_key_dim": int(key_dim),
        "add": {
            "total_s": add_s,
            "rows_per_s": add_rate,
        },
        "search_similar": _percentiles(full_ms),
        "search_similar_percept": _percentiles(percept_ms),
        "persistence_metrics": metrics,
        "recall_cache_stats": cache_stats,
    }

    full = section["search_similar"]
    percept = section["search_similar_percept"]
    print(
        f"[bench_memory] backend={args.backend} n={n} "
        f"add={add_rate:.0f} rows/s | "
        f"search p50={full['p50_ms']:.2f}ms p95={full['p95_ms']:.2f}ms | "
        f"percept p50={percept['p50_ms']:.2f}ms p95={percept['p95_ms']:.2f}ms"
    )

    close = getattr(store, "close", None)
    if callable(close):
        close()
    return section


def _bench_graph(args: argparse.Namespace, base: Path) -> dict:
    """Semantic-graph bench (WS4-M2): node add rate + match latency p50/p95."""
    from decadic.memory.factory import make_semantic_graph

    rng = np.random.default_rng(int(args.seed) + 1)
    n = int(args.n)
    dim = GRAPH_APPEARANCE_DIM
    db_path = base / f"bench_graph_{args.graph_backend}_{n}.sqlite"
    graph = make_semantic_graph(db_path)
    print(
        f"[bench_memory] graph backend={args.graph_backend} n={n} dim={dim} "
        f"db={db_path}",
        flush=True,
    )

    # ---- node add (observation stream of distinct unit appearances) ------
    apps = rng.standard_normal((n, dim)).astype(np.float32)
    apps /= np.maximum(np.linalg.norm(apps, axis=1, keepdims=True), 1e-8)
    chunk = 1000
    added = 0
    t0 = time.perf_counter()
    while added < n:
        m = min(chunk, n - added)
        with graph.write_batch():
            for j in range(m):
                i = added + j
                graph.upsert_node(apps[i], kind="obj", cycle=i)
        added += m
        if added % 100_000 == 0 or added == n:
            print(f"[bench_memory]   graph observations {added}/{n}", flush=True)
    add_s = time.perf_counter() - t0

    # ---- node-match latency (half revisits, half novel directions) -------
    match_ms: list[float] = []
    match_hits = 0
    for qi in range(int(args.queries)):
        if qi % 2 == 0:
            q = apps[int(rng.integers(n))] + rng.normal(0.0, 0.02, size=dim).astype(
                np.float32
            )
        else:
            q = rng.standard_normal(dim).astype(np.float32)
        q /= max(1e-8, float(np.linalg.norm(q)))
        t = time.perf_counter()
        nid = graph.match(q)
        match_ms.append((time.perf_counter() - t) * 1000.0)
        if nid is not None:
            match_hits += 1

    metrics = {}
    try:
        metrics = dict(graph.persistence_metrics())
    except Exception:
        pass
    cache_stats = {}
    try:
        cache_stats = dict(graph.match_cache_stats())
    except Exception:
        pass

    section = {
        "backend": args.graph_backend,
        "n_observations": n,
        "node_count": int(graph.counts()[0]),
        "appearance_dim": int(dim),
        "queries": int(args.queries),
        "match_hits": int(match_hits),
        "match_cache_enabled": bool(cache_stats.get("enabled", False)),
        "add": {
            "total_s": add_s,
            "rows_per_s": n / add_s if add_s > 0 else 0.0,
        },
        "match": _percentiles(match_ms),
        "match_cache_stats": cache_stats,
        "persistence_metrics": metrics,
    }

    match = section["match"]
    print(
        f"[bench_memory] graph backend={args.graph_backend} n={n} "
        f"nodes={section['node_count']} add={section['add']['rows_per_s']:.0f} rows/s | "
        f"match p50={match['p50_ms']:.2f}ms p95={match['p95_ms']:.2f}ms | "
        f"hits={match_hits}/{args.queries}"
    )

    close = getattr(graph, "close", None)
    if callable(close):
        close()
    return section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WS4 memory benchmark")
    parser.add_argument("--backend", choices=("sqlite", "lancedb"), default="sqlite")
    parser.add_argument(
        "--graph-backend",
        choices=("sqlite", "kuzu"),
        default=None,
        help="also benchmark the semantic graph on this backend (WS4-M2)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="skip the episodic section (requires --graph-backend)",
    )
    parser.add_argument(
        "--no-graph-match-cache",
        action="store_true",
        help="disable the in-memory match cache to measure the raw engine "
        "match path (sqlite linear scan vs kuzu vector index)",
    )
    parser.add_argument("--n", type=int, default=10_000, help="episodes/nodes to insert")
    parser.add_argument("--queries", type=int, default=200, help="search queries to time")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="directory for the store files (default: fresh temp dir, deleted after)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(REPO_ROOT / "reports"), help="report directory"
    )
    parser.add_argument(
        "--retention",
        action="store_true",
        help="keep production DB retention/pruning enabled during the run",
    )
    args = parser.parse_args(argv)
    if args.graph_only and not args.graph_backend:
        parser.error("--graph-only requires --graph-backend")

    os.environ["DECADIC_MEMORY_BACKEND"] = args.backend
    if not args.retention:
        os.environ["DECADIC_EPISODIC_DB_RETENTION_ENABLED"] = "0"
    if args.graph_backend:
        os.environ["DECADIC_GRAPH_BACKEND"] = args.graph_backend
        if not args.retention:
            os.environ["DECADIC_LTM_RETENTION_ENABLED"] = "0"
        if args.no_graph_match_cache:
            os.environ["DECADIC_LTM_MATCH_CACHE_ENABLED"] = "0"

    tmp_dir: str | None = None
    if args.db_dir:
        base = Path(args.db_dir)
        base.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="ws4_bench_")
        base = Path(tmp_dir)

    n = int(args.n)
    rss_before = _rss_mb()

    report = {
        "workstream": "WS4-M0.4" if not args.graph_backend else "WS4-M0.4/M2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "graph_backend": args.graph_backend,
        "n_episodes": n,
        "queries": int(args.queries),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "retention_enabled": bool(args.retention),
        "rss_mb_before": rss_before,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    if not args.graph_only:
        episodic = _bench_episodic(args, base)
        # Keep the original top-level report layout (M0.4 consumers).
        report["embedding_dim"] = episodic["embedding_dim"]
        report["percept_key_dim"] = episodic["percept_key_dim"]
        report["add"] = episodic["add"]
        report["search_similar"] = episodic["search_similar"]
        report["search_similar_percept"] = episodic["search_similar_percept"]
        report["persistence_metrics"] = episodic["persistence_metrics"]
        report["recall_cache_stats"] = episodic["recall_cache_stats"]

    if args.graph_backend:
        report["graph"] = _bench_graph(args, base)

    report["rss_mb_after"] = _rss_mb()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [] if args.graph_only else [args.backend]
    if args.graph_backend:
        parts.append(f"graph-{args.graph_backend}")
    out_path = out_dir / f"ws4_bench_{'_'.join(parts)}_{n}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rss_after = report["rss_mb_after"]
    rss_txt = f"{rss_after:.1f}MB" if rss_after is not None else "n/a (pip install psutil)"
    print(f"[bench_memory] rss={rss_txt} | report={out_path}")

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
