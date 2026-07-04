"""WS3B-M2.2: offline GateNet trainer.

Input: a gate dataset directory (data.npz + manifest.json) from
``build_gate_dataset.py``. Trains on the labeled subset (shadow-sampled
rows), soft BCE against the sigmoid labels, positives up-weighted by
inverse frequency (labels > 0.5 are rare on synthetic data by construction).

Split is BY RUN: ``--val-run <index>`` holds one run out. With a single-run
dataset the trainer proceeds but stamps the report VAL=NONE -- train metrics
only, never promotion evidence (PRD ws3b 3.4).

Output: ``reports/gate_training_<stamp>/{gate_net.pt, report.json}`` with
AUC, accuracy, calibration bins, and the regret-vs-rate frontier vs the
recorded heuristic decisions.

Usage:
    python scripts/train_gate.py reports/gate_dataset_<stamp> \
        [--val-run 1] [--hidden 16] [--epochs 300] [--lr 0.01] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from decadic.nn.gate_net import GateNet, normalize  # noqa: E402


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Rank-based AUC (no sklearn). None when one class is absent."""
    pos = y_prob[y_true > 0.5]
    neg = y_prob[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([neg, pos]), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg) :].sum()
    n_pos, n_neg = float(len(pos)), float(len(neg))
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (y_prob >= lo) & (y_prob < hi if b < bins - 1 else y_prob <= hi)
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_pred": float(y_prob[m].mean()) if m.any() else None,
                "mean_label": float(y_true[m].mean()) if m.any() else None,
            }
        )
    return out


def regret_rate_frontier(
    probs: np.ndarray, regret: np.ndarray, heuristic_escalate: np.ndarray
) -> dict:
    """At the heuristic's own escalation rate, what fraction of total regret
    mass does each policy's escalation set capture? (Higher = attention goes
    where deliberation actually changes something.)"""
    total = float(regret.sum())
    rate = float(heuristic_escalate.mean())
    k = max(1, int(round(rate * len(probs))))
    top = np.argsort(-probs)[:k]
    captured_net = float(regret[top].sum())
    captured_heur = float(regret[heuristic_escalate > 0.5].sum())
    return {
        "matched_rate": rate,
        "total_regret_mass": total,
        "heuristic_capture": captured_heur / total if total > 0 else None,
        "gatenet_capture": captured_net / total if total > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="gate_dataset_<stamp> directory")
    ap.add_argument("--val-run", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ds = Path(args.dataset)
    data = np.load(ds / "data.npz")
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    lab = ~np.isnan(data["y"])
    X = normalize(data["X"][lab])
    y = data["y"][lab].astype(np.float32)
    runs = data["run_id"][lab]
    # Regret proxy for the frontier: the divergence that produced the label.
    kind = data["shadow_kind"][lab]
    # Recover divergence from the label is lossy; carry it via pain-free proxy:
    # for frontier purposes use the label itself as regret mass (monotone map).
    regret_mass = y.copy()
    heur_esc = data["escalate"][lab].astype(np.float32)

    if args.val_run is not None:
        tr = runs != args.val_run
        va = runs == args.val_run
        if not va.any():
            print(f"val run {args.val_run} has no labeled rows")
            return 1
    else:
        tr = np.ones(len(y), dtype=bool)
        va = np.zeros(len(y), dtype=bool)
        if len(set(runs.tolist())) == 1:
            print("[train_gate] WARNING: single-run dataset -> VAL=NONE "
                  "(train metrics only; never promotion evidence)")

    pos = float((y[tr] > 0.5).sum())
    neg = float((y[tr] <= 0.5).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)

    net = GateNet(hidden=args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.as_tensor(X[tr])
    yt = torch.as_tensor(y[tr])
    net.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xt), yt)
        loss.backward()
        opt.step()
    net.eval()

    def _metrics(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"n": 0}
        with torch.no_grad():
            p = torch.sigmoid(net(torch.as_tensor(X[mask]))).numpy()
        yy = y[mask]
        return {
            "n": int(mask.sum()),
            "pos": int((yy > 0.5).sum()),
            "auc": auc_score(yy, p),
            "acc_at_0.5": float(((p > 0.5) == (yy > 0.5)).mean()),
            "calibration": calibration_bins(yy, p),
        }

    with torch.no_grad():
        p_all = torch.sigmoid(net(torch.as_tensor(X))).numpy()
    report = {
        "workstream": "WS3B-M2.2",
        "dataset": str(ds),
        "dataset_manifest_totals": manifest.get("totals"),
        "hyperparameters": {
            "hidden": args.hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "pos_weight": float(pos_weight[0]),
            "val_run": args.val_run,
        },
        "train": _metrics(tr),
        "val": _metrics(va) if va.any() else "NONE (single run or no holdout)",
        "frontier_labeled_rows": regret_rate_frontier(p_all, regret_mass, heur_esc),
        "final_train_loss": float(loss.detach()),
        "shadow_kind_mix": {
            "skip": int((kind == 1).sum()),
            "esc": int((kind == 2).sum()),
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"gate_training_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    net.save(out / "gate_net.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    tr_m, va_m = report["train"], report["val"]
    print(
        f"[train_gate] n_labeled={len(y)} train_auc={tr_m.get('auc')} "
        f"val={va_m if isinstance(va_m, str) else va_m.get('auc')} "
        f"frontier: heuristic={report['frontier_labeled_rows']['heuristic_capture']} "
        f"gatenet={report['frontier_labeled_rows']['gatenet_capture']}"
    )
    print(f"[train_gate] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
