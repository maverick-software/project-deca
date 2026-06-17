"""Offline trainer for interpretability probes.

Reads a capture JSONL produced with ``DECADIC_PROBE_CAPTURE=1`` (rows of
``{latents, targets}``) and fits a cheap probe per (latent, target): ridge
regression for continuous targets, logistic regression for binary ones. For each
target it records every latent's held-out quality (R^2 / accuracy) and marks the
best latent, then writes a JSON "probe bank" that the runtime applies read-only.

The probes are *measurement only*: their supervision comes from the eval-only
ground-truth channels and their weights never enter the cognitive optimizer.

Example::

    DECADIC_PROBE_CAPTURE=1 DECADIC_PROBE_CAPTURE_PATH=probe_capture.jsonl <run a session>
    python scripts/train_probes.py --capture probe_capture.jsonl --out probes.json
    DECADIC_PROBE_PATH=probes.json <run a session: the Cognition panel shows read-outs>
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

LATENT_KEYS = ["emotion", "state_mind", "metacognition", "narrative", "z5"]


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _stack_latent(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, X) for one latent: mask marks rows with the modal dimension."""
    vecs = [r.get("latents", {}).get(key) or [] for r in rows]
    lens = [len(v) for v in vecs]
    dim = max(lens) if lens else 0
    if dim == 0:
        return np.zeros(len(rows), dtype=bool), np.zeros((0, 0))
    mask = np.array([len(v) == dim for v in vecs], dtype=bool)
    X = np.array([vecs[i] for i in range(len(rows)) if mask[i]], dtype=np.float64)
    return mask, X


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    mu_x = X.mean(axis=0)
    mu_y = float(y.mean())
    Xc = X - mu_x
    yc = y - mu_y
    n_feat = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(n_feat)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = mu_y - float(mu_x @ w)
    return w, b


def _fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 600, lr: float = 0.5) -> tuple[np.ndarray, float]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gerr = p - y
        gw = Xs.T @ gerr / n + lam * w
        gb = float(gerr.mean())
        w -= lr * gw
        b -= lr * gb
    # Fold standardization back so the bank can apply raw (vec @ w + b).
    w_raw = w / sd
    b_raw = b - float(mu @ w_raw)
    return w_raw, b_raw


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _train_target(
    rows: list[dict[str, Any]],
    target: str,
    *,
    lam: float,
    holdout: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    have = [i for i, r in enumerate(rows) if isinstance(r.get("targets", {}).get(target), (int, float))]
    if len(have) < 8:
        return None
    y_all = np.array([float(rows[i]["targets"][target]) for i in have], dtype=np.float64)
    is_binary = bool(np.all(np.isin(np.unique(y_all), [0.0, 1.0])))
    kind = "classification" if is_binary else "regression"

    per_latent: dict[str, Any] = {}
    for key in LATENT_KEYS:
        sub = [rows[i] for i in have]
        mask, X = _stack_latent(sub, key)
        if X.shape[0] < 8 or X.shape[1] == 0:
            continue
        y = y_all[mask]
        n = X.shape[0]
        idx = rng.permutation(n)
        n_test = max(1, int(round(holdout * n))) if n >= 4 else 0
        test_idx = idx[:n_test]
        train_idx = idx[n_test:] if n_test else idx
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = (X[test_idx], y[test_idx]) if n_test else (X, y)
        if kind == "classification":
            if len(np.unique(ytr)) < 2:
                continue
            w, b = _fit_logistic(Xtr, ytr, lam)
            p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
            score = float(np.mean((p > 0.5).astype(np.float64) == yte))
        else:
            w, b = _fit_ridge(Xtr, ytr, lam)
            score = _r2(yte, Xte @ w + b)
        per_latent[key] = {
            "w": [round(float(v), 6) for v in w],
            "b": round(float(b), 6),
            "dim": int(X.shape[1]),
            "score": round(float(score), 4),
            "n": int(n),
        }
    if not per_latent:
        return None
    best = max(per_latent, key=lambda k: per_latent[k]["score"])
    return {"kind": kind, "best_latent": best, "per_latent": per_latent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="probe_capture.jsonl", help="input capture JSONL")
    ap.add_argument("--out", default="probes.json", help="output probe-bank JSON")
    ap.add_argument("--ridge", type=float, default=1.0, help="L2 regularization strength")
    ap.add_argument("--holdout", type=float, default=0.25, help="held-out fraction for scoring")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _load_rows(args.capture)
    if not rows:
        raise SystemExit(f"no usable rows in {args.capture}")
    rng = np.random.default_rng(args.seed)

    target_names = sorted({t for r in rows for t in (r.get("targets") or {})})
    targets: dict[str, Any] = {}
    for t in target_names:
        spec = _train_target(rows, t, lam=args.ridge, holdout=args.holdout, rng=rng)
        if spec is not None:
            targets[t] = spec

    bank = {
        "meta": {
            "rows": len(rows),
            "ridge": args.ridge,
            "holdout": args.holdout,
            "latents": LATENT_KEYS,
        },
        "targets": targets,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2)

    print(f"trained {len(targets)} target(s) from {len(rows)} rows -> {args.out}")
    for t, spec in targets.items():
        best = spec["best_latent"]
        sc = spec["per_latent"][best]["score"]
        kind = "acc" if spec["kind"] == "classification" else "R2"
        print(f"  {t:>20s}: best={best:<14s} {kind}={sc}")


if __name__ == "__main__":
    main()
