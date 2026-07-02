"""Prediction-error trend check over an eval samples JSONL (WS1 verification).

Reads the ``training_eval_*.jsonl`` samples file written by
``scripts/run_training_eval.py``, extracts ``neural_pc_loss_last`` (and a few
companion metrics) per cycle, writes a CSV, fits a least-squares slope, and
prints a verdict. Optionally renders a PNG if matplotlib is installed.

Usage::

    python scripts/plot_pc_trend.py reports/training_eval_ws1_learning_run_<ts>.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

METRICS = (
    "neural_pc_loss_last",
    "loss_total",
    "prediction_error_ema",
    "viability",
    "energy",
    "hydration",
    "integrity",
)


def load_samples(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            m = s.get("metrics") or {}
            row = {"cycle": s.get("cycle"), "t_s": s.get("t_s")}
            for k in METRICS:
                v = m.get(k)
                row[k] = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
            rows.append(row)
    return rows


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples", help="path to training_eval_*.jsonl samples file")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    path = Path(args.samples)
    rows = load_samples(path)
    if len(rows) < 10:
        print(f"FAIL: only {len(rows)} samples in {path} — run too short or agent stalled.")
        return 1

    csv_path = Path(args.out_csv) if args.out_csv else path.with_suffix(".pc_trend.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cycle", "t_s", *METRICS])
        w.writeheader()
        w.writerows(rows)

    pts = [(r["cycle"], r["neural_pc_loss_last"]) for r in rows if r["neural_pc_loss_last"] is not None]
    if len(pts) < 10:
        print("FAIL: neural_pc_loss_last present in fewer than 10 samples — neural path not active?")
        return 1

    xs = [float(c) for c, _ in pts]
    ys = [y for _, y in pts]
    n = len(ys)
    half = n // 2
    first_half = sum(ys[:half]) / half
    second_half = sum(ys[half:]) / (n - half)
    s = slope(xs, ys)
    nonfinite = sum(1 for r in rows if r["neural_pc_loss_last"] is None)

    print(f"samples:            {n} (cycles {xs[0]:.0f} -> {xs[-1]:.0f})")
    print(f"pc_loss first/last: {ys[0]:.6f} -> {ys[-1]:.6f}  (delta {ys[-1]-ys[0]:+.6f})")
    print(f"half means:         {first_half:.6f} -> {second_half:.6f}  (delta {second_half-first_half:+.6f})")
    print(f"lsq slope/cycle:    {s:+.3e}")
    print(f"nonfinite samples:  {nonfinite}")
    print(f"csv:                {csv_path}")

    if args.out_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(xs, ys, lw=0.8, alpha=0.7, label="neural_pc_loss_last")
            k = max(1, n // 50)
            smooth = [sum(ys[max(0, i - k):i + 1]) / len(ys[max(0, i - k):i + 1]) for i in range(n)]
            ax.plot(xs, smooth, lw=2.0, label=f"rolling mean (k={k})")
            ax.set_xlabel("cycle")
            ax.set_ylabel("predictive-coding loss")
            ax.set_title(f"PC loss trend — slope {s:+.2e}/cycle")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out_png, dpi=120)
            print(f"png:                {args.out_png}")
        except ImportError:
            print("png:                skipped (matplotlib not installed)")

    ok = second_half < first_half and s < 0
    print("verdict:            " + ("PASS — PC loss trending down" if ok else "FAIL — no downward trend"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
