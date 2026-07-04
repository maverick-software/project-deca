"""WS5-M5 verdict: does risk/priority track the RELATION on unseen pairings?

Reads probe samples (binding_probe_run.py) + the scenario file. Logic:

1. The probe segment is everything after the last pain spike (train phases
   carry threat events; probe phases are eventless BY DESIGN -- pain cannot
   be the carrier of any probe-segment response).
2. Baseline = priority_scalar statistics over the pre-train warmup window.
3. Deflection episode = a contiguous run where priority deflects toward
   "avoid" (below baseline mean - sigma*std). Episodes are mapped to probe
   phases BY ORDER (phases are sequential and widely separated; the mapping
   caveat is documented).
4. Verdicts:
   - BINDING (PASS): every probe_novel phase deflects -- the relation
     generalized to never-seen pairings.
   - MEMORIZER: probe_trained deflects, probe_novel does not.
   - BLIND: no probe-segment deflections (the flags-off expectation: a
     pooled representation cannot carry which-entity-is-adjacent-to-which).

--expect pass|fail makes this usable for both ablation legs: the flags-off
run SUCCEEDS (exit 0) when the probe correctly FAILS.

Usage: python scripts/check_binding_probe.py <samples.jsonl> <scenario.json>
           [--sigma 3.0] [--expect pass|fail]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAIN_SPIKE = 0.05
MIN_RUN = 2  # samples; at 0.5 s polling a 12 s phase is ~24 samples
MERGE_GAP = 4


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return [r for r in rows if r.get("priority_scalar") is not None]


def episodes(flags: list[bool]) -> list[tuple[int, int]]:
    """Contiguous True runs (>= MIN_RUN), merging gaps <= MERGE_GAP."""
    runs, start = [], None
    for i, v in enumerate(flags + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r))
    return [(a, b) for a, b in merged if b - a + 1 >= MIN_RUN]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples")
    ap.add_argument("scenario")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--expect", choices=("pass", "fail"), default=None)
    args = ap.parse_args()

    rows = load(Path(args.samples))
    scen = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    order = scen.get("probe_order", [])
    n_novel = sum(1 for p in order if p["kind"] == "probe_novel")
    n_trained = sum(1 for p in order if p["kind"] == "probe_trained")
    if len(rows) < 50 or not order:
        print(f"FAIL: unusable inputs (samples={len(rows)}, probe phases={len(order)})")
        return 1

    prio = [float(r["priority_scalar"]) for r in rows]
    pain = [float(r.get("pain_scalar", 0.0) or 0.0) for r in rows]

    # Baseline: pre-train warmup = samples before the FIRST pain spike.
    first_pain = next((i for i, p in enumerate(pain) if p > PAIN_SPIKE), len(rows) // 4)
    base = prio[: max(10, first_pain)]
    mean = sum(base) / len(base)
    var = sum((x - mean) ** 2 for x in base) / len(base)
    std = max(0.01, var**0.5)
    thresh = mean - args.sigma * std

    # Probe segment: after the LAST pain spike, plus a recovery margin.
    last_pain = max(
        (i for i, p in enumerate(pain) if p > PAIN_SPIKE), default=first_pain
    )
    seg_start = min(len(rows) - 1, last_pain + 20)
    seg = [p < thresh for p in prio[seg_start:]]
    eps = episodes(seg)

    # Order-based mapping to the interleaved probe phases.
    hits = {"probe_trained": 0, "probe_novel": 0}
    for i, _ in enumerate(eps):
        if i < len(order):
            hits[order[i]["kind"]] += 1
    total = len(eps)

    if total == 0:
        verdict = "BLIND"
    elif hits["probe_novel"] >= n_novel:
        verdict = "BINDING"
    elif hits["probe_trained"] > 0 and hits["probe_novel"] < n_novel:
        verdict = "MEMORIZER"
    else:
        verdict = "PARTIAL"

    print(
        f"baseline: mean={mean:.4f} std={std:.4f} thresh={thresh:.4f} "
        f"(warmup n={len(base)})"
    )
    print(
        f"probe segment: samples={len(seg)} episodes={total} "
        f"(expected {n_trained} trained + {n_novel} novel) "
        f"hits: trained={hits['probe_trained']}/{n_trained} "
        f"novel={hits['probe_novel']}/{n_novel}"
    )
    print(f"BINDING_PROBE: {verdict}")

    if args.expect == "pass":
        return 0 if verdict == "BINDING" else 1
    if args.expect == "fail":
        return 0 if verdict != "BINDING" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
