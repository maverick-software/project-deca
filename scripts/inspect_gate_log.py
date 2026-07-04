"""WS3B-M0 acceptance: summarize a gate decision log.

Reports coverage (rows vs cycle span), decision mix, shadow sampling rate,
and the first regret/waste distributions -- the raw material for M1 labels.

Usage: .venv\\Scripts\\python.exe scripts\\inspect_gate_log.py <run_dir_or_jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1)))]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    path = target
    if target.is_dir():
        cands = sorted(target.glob("gate_decisions_*.jsonl"))
        if not cands:
            print(f"no gate_decisions_*.jsonl under {target}")
            return 1
        path = cands[-1]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        print("no rows")
        return 1

    cycles = [int(r.get("cycle", 0)) for r in rows]
    span = max(cycles) - min(cycles) + 1
    esc = sum(1 for r in rows if r.get("escalate"))
    fast = sum(1 for r in rows if r.get("fast_path"))
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[str(r.get("reason"))] = reasons.get(str(r.get("reason")), 0) + 1
    shadow = [r for r in rows if "shadow_kind" in r]
    skips = [r for r in shadow if r.get("shadow_kind") == "skip"]
    escs = [r for r in shadow if r.get("shadow_kind") == "esc"]

    print(f"file: {path.name}")
    print(
        f"rows={len(rows)} cycle-span={min(cycles)}..{max(cycles)} "
        f"coverage={len(rows) / max(1, span):.1%}"
    )
    print(
        f"escalations={esc} ({esc / len(rows):.1%}) fast_path={fast} "
        f"reasons={reasons}"
    )
    print(
        f"shadow rows={len(shadow)} ({len(shadow) / len(rows):.2%} of decisions; "
        f"skip={len(skips)} esc={len(escs)})"
    )
    if skips:
        rr = sorted(float(r.get("shadow_regret_risk", 0.0)) for r in skips)
        rz = sorted(float(r.get("shadow_regret_z4", 0.0)) for r in skips)
        print(
            "  skip-regret risk: "
            f"p50={_pct(rr, 0.5):.4f} p90={_pct(rr, 0.9):.4f} max={rr[-1]:.4f} | "
            f"z4-RMS p50={_pct(rz, 0.5):.4f} p90={_pct(rz, 0.9):.4f} max={rz[-1]:.4f}"
        )
    if escs:
        wr = sorted(float(r.get("shadow_waste_risk", 0.0)) for r in escs)
        wz = sorted(float(r.get("shadow_waste_z4", 0.0)) for r in escs)
        print(
            "  esc-waste risk:   "
            f"p50={_pct(wr, 0.5):.4f} p90={_pct(wr, 0.9):.4f} max={wr[-1]:.4f} | "
            f"z4-RMS p50={_pct(wz, 0.5):.4f} p90={_pct(wz, 0.9):.4f} max={wz[-1]:.4f}"
        )
    # The M1 question in one line: does regret vary enough to learn from?
    if skips:
        rz = sorted(float(r.get("shadow_regret_z4", 0.0)) for r in skips)
        spread = _pct(rz, 0.9) - _pct(rz, 0.1)
        print(
            f"signal check: skip-regret z4 p10-p90 spread = {spread:.4f} "
            f"({'learnable structure plausible' if spread > 0.01 else 'near-flat -- expected on synthetic (PRD risk G5)'})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
