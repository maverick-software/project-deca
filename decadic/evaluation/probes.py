"""Read probe-bank quality as eval-only report data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_probe_bank(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "reason": "no_probe_bank"}
    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": "missing_probe_bank", "path": str(p)}
    with p.open(encoding="utf-8") as f:
        bank = json.load(f)
    targets = bank.get("targets") or {}
    summary: dict[str, Any] = {
        "available": True,
        "path": str(p),
        "target_count": len(targets),
        "targets": {},
        "best_score": None,
    }
    best_score: float | None = None
    for name, spec in targets.items():
        best = spec.get("best_latent")
        per = spec.get("per_latent") or {}
        row = per.get(best) if best else None
        score = None if row is None else float(row.get("score", 0.0))
        summary["targets"][name] = {
            "kind": spec.get("kind", ""),
            "best_latent": best,
            "score": score,
            "n": int(row.get("n", 0)) if row else 0,
        }
        if score is not None:
            best_score = score if best_score is None else max(best_score, score)
    summary["best_score"] = best_score
    return summary

