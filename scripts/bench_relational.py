"""WS5-M3.2: measured cycle cost of the relational core on a real preset.

Builds the stack twice (relational off / on), runs N forwards with realistic
token loads (K=6 slots, k=5 memory tokens), reports p50/p95 forward ms and
the delta. The 2x2-vs-3x4 sizing decision comes from THESE numbers against
the 70-90 ms cycle envelope (the ANN lesson from WS4: measure, never assume).

Usage:
    .venv\\Scripts\\python.exe scripts\\bench_relational.py [--preset full]
        [--device cuda] [--n 200] [--layers 2 --heads 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bench(relational: bool, args) -> tuple[float, float]:
    os.environ["DECADIC_RELATIONAL_CORE"] = "1" if relational else "0"
    os.environ["DECADIC_WM_SLOT_TENSOR"] = "1" if relational else "0"
    os.environ["DECADIC_MEMORY_TOKENS"] = "1" if relational else "0"
    os.environ["DECADIC_RELATIONAL_LAYERS"] = str(args.layers)
    os.environ["DECADIC_RELATIONAL_HEADS"] = str(args.heads)
    os.environ["DECADIC_DEVICE"] = args.device

    import torch

    from decadic.nn.config import neural_config_from_env
    from decadic.nn.neural_stack import NeuralCognitiveStack
    from decadic.state.working_memory import SLOT_TENSOR_DIM

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(3)
    cfg = neural_config_from_env(args.preset)
    stack = NeuralCognitiveStack(cfg).to(dev)
    stack.eval()

    z0 = torch.randn(1, cfg.d_model, device=dev)
    ep = torch.rand(1, 4, device=dev)
    mem = torch.randn(1, cfg.memory_context_dim, device=dev)
    slots = torch.randn(6, SLOT_TENSOR_DIM, device=dev)
    smask = torch.ones(6, dtype=torch.bool, device=dev)
    toks = torch.randn(5, 80, device=dev)
    tmask = torch.ones(5, dtype=torch.bool, device=dev)
    kw = (
        dict(wm_slots=slots, wm_slots_mask=smask, mem_tokens=toks, mem_tokens_mask=tmask)
        if relational
        else {}
    )

    times = []
    with torch.no_grad():
        for i in range(args.n + 20):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            stack(z0, ep, mem, **kw)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if i >= 20:  # warmup excluded
                times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[int(0.95 * len(times)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    args = ap.parse_args()

    p50_off, p95_off = _bench(False, args)
    p50_on, p95_on = _bench(True, args)
    print(
        f"[bench_relational] preset={args.preset} device={args.device} "
        f"layers={args.layers} heads={args.heads}\n"
        f"  off: p50={p50_off:.2f}ms p95={p95_off:.2f}ms\n"
        f"  on : p50={p50_on:.2f}ms p95={p95_on:.2f}ms\n"
        f"  delta p50={p50_on - p50_off:+.2f}ms "
        f"({(p50_on - p50_off) / max(0.01, p50_off) * 100:+.1f}%) -- envelope 70-90ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
