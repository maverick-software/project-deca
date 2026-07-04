"""WS5-M1: build a GateNet training dataset from gate decision logs.

Input: one or more ``gate_decisions_*.jsonl`` files (or run directories
containing them) produced by ``DECADIC_GATE_LOG=1`` (WS5-M0.1). Each input
file is treated as one RUN; splits downstream must be by run, never by row
(rows within a run are autocorrelated).

Labels (M1.2, PRD ws5 3.2): both shadow kinds measure the same quantity --
how much fresh stage-4 deliberation diverged from the cheap substitute
(decayed precedent). So one formula covers both:

    should_escalate = sigmoid(alpha * (divergence - cost))

- skip rows:      divergence = shadow_regret_z4 (high -> should have thought)
- escalation rows: divergence = shadow_waste_z4 (low -> thought for nothing)

Outcome sharpening: a pain rise inside the forward horizon after the decision
pushes the label up (attention that precedes pain was warranted):

    label = clamp01(label + beta * max(0, pain_delta))

alpha / cost / beta / horizon are dataset hyperparameters recorded in the
manifest -- never silently baked in.

Output: ``<out>/gate_dataset_<stamp>/data.npz`` + ``manifest.json``.
npz keys: X (n,8 float32; FEATURES order), y (n float32; NaN when unlabeled),
run_id (n int16), cycle (n int32), escalate (n int8), shadow_kind
(n int8: 0=none 1=skip 2=esc), pain_delta / viability_delta / pc_delta
(n float32; NaN when the horizon ran off the log's end).

Usage:
    python scripts/build_gate_dataset.py reports/gateprobe_*/ reports/soak_*/ \
        [--horizon 50] [--alpha 40] [--cost 0.05] [--beta 0.5] [--out reports]
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

FEATURES = (
    "novelty",
    "pe",
    "affect",
    "priority",
    "drive",
    "esc_rate",
    "latch",
    "precedent_age",
)
AGE_BUCKETS = ((0, 4), (5, 16), (17, 64), (65, 10**9))


def find_logs(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        for m in sorted(globmod.glob(t)) or [t]:
            p = Path(m)
            if p.is_dir():
                out.extend(sorted(p.glob("gate_decisions_*.jsonl")))
            elif p.is_file():
                out.append(p)
    # De-duplicate, preserve order.
    seen: set[Path] = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "cycle" in r and "novelty" in r:
            rows.append(r)
    rows.sort(key=lambda r: int(r["cycle"]))
    return rows


def join_outcomes(rows: list[dict], horizon: int) -> None:
    """Attach pain_delta / viability_delta / pc_delta over (c, c+horizon].

    pain_delta uses the window MAX (a spike anywhere in the horizon counts);
    viability/pc use the endpoint (drift measures). NaN when the horizon runs
    past the end of the log -- those rows keep their divergence-only label.
    """
    by_cycle: dict[int, dict] = {int(r["cycle"]): r for r in rows}
    max_cycle = max(by_cycle) if by_cycle else 0
    for r in rows:
        c = int(r["cycle"])
        end = c + horizon
        if end > max_cycle:
            r["pain_delta"] = r["viability_delta"] = r["pc_delta"] = float("nan")
            continue
        window = [by_cycle[k] for k in range(c + 1, end + 1) if k in by_cycle]
        if not window:
            r["pain_delta"] = r["viability_delta"] = r["pc_delta"] = float("nan")
            continue
        r["pain_delta"] = max(float(w.get("pain", 0.0)) for w in window) - float(
            r.get("pain", 0.0)
        )
        last = window[-1]
        r["viability_delta"] = float(last.get("viability", 0.0)) - float(
            r.get("viability", 0.0)
        )
        pc0, pc1 = r.get("pc_ema"), last.get("pc_ema")
        r["pc_delta"] = (
            float(pc1) - float(pc0) if pc0 is not None and pc1 is not None else float("nan")
        )


def label_row(r: dict, *, alpha: float, cost: float, beta: float) -> float | None:
    """Unified divergence label; None when the row carries no shadow sample."""
    kind = r.get("shadow_kind")
    if kind == "skip":
        div = float(r.get("shadow_regret_z4", 0.0))
    elif kind == "esc":
        div = float(r.get("shadow_waste_z4", 0.0))
    else:
        return None
    label = 1.0 / (1.0 + math.exp(-alpha * (div - cost)))
    pain_delta = float(r.get("pain_delta", float("nan")))
    if beta > 0 and pain_delta == pain_delta:  # not NaN
        label = min(1.0, max(0.0, label + beta * max(0.0, pain_delta)))
    return label


def build(
    log_paths: list[Path], *, horizon: int, alpha: float, cost: float, beta: float
) -> tuple[dict[str, np.ndarray], dict]:
    X, y, run_id, cycle, escalate, shadow_kind = [], [], [], [], [], []
    pain_d, via_d, pc_d = [], [], []
    per_run_stats = []
    kind_code = {None: 0, "skip": 1, "esc": 2}

    for ri, path in enumerate(log_paths):
        rows = load_rows(path)
        join_outcomes(rows, horizon)
        n_lab = 0
        labels_this = []
        for r in rows:
            X.append([float(r.get(f, 0.0) or 0.0) for f in FEATURES])
            lab = label_row(r, alpha=alpha, cost=cost, beta=beta)
            y.append(float("nan") if lab is None else lab)
            if lab is not None:
                n_lab += 1
                labels_this.append(lab)
            run_id.append(ri)
            cycle.append(int(r["cycle"]))
            escalate.append(int(bool(r.get("escalate"))))
            shadow_kind.append(kind_code.get(r.get("shadow_kind"), 0))
            pain_d.append(float(r.get("pain_delta", float("nan"))))
            via_d.append(float(r.get("viability_delta", float("nan"))))
            pc_d.append(float(r.get("pc_delta", float("nan"))))

        # Label-circularity check (PRD ws5 risk 1): regret conditioned on
        # precedent age -- if regret only grows with age, the label mostly
        # restates the decay clock, not stimulus-driven need.
        age_cond = {}
        for lo, hi in AGE_BUCKETS:
            vals = [
                float(r.get("shadow_regret_z4", 0.0))
                for r in rows
                if r.get("shadow_kind") == "skip" and lo <= int(r.get("precedent_age", -1)) <= hi
            ]
            age_cond[f"age_{lo}_{hi if hi < 10**9 else 'inf'}"] = {
                "n": len(vals),
                "mean_regret": (sum(vals) / len(vals)) if vals else None,
            }
        per_run_stats.append(
            {
                "run": path.name,
                "rows": len(rows),
                "labeled": n_lab,
                "label_mean": (sum(labels_this) / n_lab) if n_lab else None,
                "label_pos_frac": (
                    sum(1 for v in labels_this if v > 0.5) / n_lab if n_lab else None
                ),
                "escalation_rate": (
                    sum(1 for r in rows if r.get("escalate")) / len(rows) if rows else None
                ),
                "regret_by_precedent_age": age_cond,
            }
        )

    arrays = {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "run_id": np.asarray(run_id, dtype=np.int16),
        "cycle": np.asarray(cycle, dtype=np.int32),
        "escalate": np.asarray(escalate, dtype=np.int8),
        "shadow_kind": np.asarray(shadow_kind, dtype=np.int8),
        "pain_delta": np.asarray(pain_d, dtype=np.float32),
        "viability_delta": np.asarray(via_d, dtype=np.float32),
        "pc_delta": np.asarray(pc_d, dtype=np.float32),
    }
    lab_mask = ~np.isnan(arrays["y"])
    manifest = {
        "workstream": "WS5-M1",
        "features": list(FEATURES),
        "hyperparameters": {
            "horizon": horizon,
            "alpha": alpha,
            "cost": cost,
            "beta": beta,
        },
        "runs": [p.name for p in log_paths],
        "totals": {
            "rows": int(arrays["X"].shape[0]),
            "labeled": int(lab_mask.sum()),
            "labeled_pos_frac": (
                float((arrays["y"][lab_mask] > 0.5).mean()) if lab_mask.any() else None
            ),
        },
        "per_run": per_run_stats,
        "split_rule": "BY RUN ONLY (rows within a run are autocorrelated)",
    }
    return arrays, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="log files, run dirs, or globs")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    logs = find_logs(args.inputs)
    if not logs:
        print("no gate_decisions_*.jsonl found in inputs")
        return 1
    print(f"[build_gate_dataset] runs={len(logs)}")
    for p in logs:
        print(f"  - {p}")
    arrays, manifest = build(
        logs, horizon=args.horizon, alpha=args.alpha, cost=args.cost, beta=args.beta
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"gate_dataset_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "data.npz", **arrays)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    t = manifest["totals"]
    print(
        f"[build_gate_dataset] rows={t['rows']} labeled={t['labeled']} "
        f"pos_frac={t['labeled_pos_frac']} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
