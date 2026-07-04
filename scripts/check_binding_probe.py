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

    # v2 (2026-07-04, after the first off-leg run): segmentation by SCENARIO
    # STEP WINDOWS, not pain anchors -- threat_near raises anticipatory
    # stress, not pain, so the pain-spike anchor collapsed and swallowed the
    # train segment. step_est (t/rate) maps each sample onto the schedule;
    # drop_oldest keeps the server current with the freshest frame, so the
    # clock holds within a phase-gap tolerance.
    def _step(r: dict) -> float:
        if r.get("step_est") is not None:
            return float(r["step_est"])
        return float(r.get("t", 0.0)) / 0.1  # legacy samples: rate was 0.1

    prio_by_step = [(_step(r), float(r["priority_scalar"])) for r in rows]
    phase_len = int(scen["schedule"][0].get("steps", 120)) if scen.get("schedule") else 120

    # Baseline: samples inside probe-segment GAPS (between probe phases) --
    # local to the segment, immune to slow drift across the run.
    probe_start = min(int(p["start"]) for p in order)
    gap_vals = [
        v
        for s, v in prio_by_step
        if s >= probe_start - 200
        and not any(
            int(p["start"]) - 20 <= s <= int(p["start"]) + phase_len + 60 for p in order
        )
    ]
    if len(gap_vals) < 20:
        print(f"FAIL: too few gap-baseline samples ({len(gap_vals)})")
        return 1
    mean = sum(gap_vals) / len(gap_vals)
    var = sum((x - mean) ** 2 for x in gap_vals) / len(gap_vals)
    std = max(0.01, var**0.5)
    thresh = mean - args.sigma * std

    # Per-phase decision: does priority deflect toward avoid INSIDE the
    # phase's own window? (No order mapping, no episode counting.)
    hits = {"probe_trained": 0, "probe_novel": 0}
    details = []
    for p in order:
        lo, hi = int(p["start"]) + 5, int(p["start"]) + phase_len + 40
        w = [v for s, v in prio_by_step if lo <= s <= hi]
        if not w:
            details.append((p["kind"], p["pair"], None, False))
            continue
        wmin = min(w)
        deflected = wmin < thresh
        if deflected:
            hits[p["kind"]] += 1
        details.append((p["kind"], p["pair"], round(wmin, 4), deflected))
    total = sum(hits.values())
    eps = []  # retained for the summary line's episode count semantics
    for kind, pair, wmin, ok in details:
        print(f"  {kind:14s} {str(pair):24s} min={wmin} deflected={ok}")

    if total == 0:
        verdict = "BLIND"
    elif hits["probe_novel"] >= n_novel:
        verdict = "BINDING"
    elif hits["probe_trained"] > 0 and hits["probe_novel"] < n_novel:
        verdict = "MEMORIZER"
    else:
        verdict = "PARTIAL"

    print(
        f"gap baseline: mean={mean:.4f} std={std:.4f} thresh={thresh:.4f} "
        f"(n={len(gap_vals)})"
    )
    print(
        f"probe phases deflected: {total}/{len(order)} "
        f"(trained={hits['probe_trained']}/{n_trained} "
        f"novel={hits['probe_novel']}/{n_novel})"
    )
    print(f"BINDING_PROBE: {verdict}")

    if args.expect == "pass":
        return 0 if verdict == "BINDING" else 1
    if args.expect == "fail":
        return 0 if verdict != "BINDING" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
