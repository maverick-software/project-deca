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
    ap.add_argument(
        "--flag-on",
        action="store_true",
        help="this is the flags-ON leg: distinguishes LEARNING_FAILED (training "
        "signal too weak) from BLIND (representation cannot bind)",
    )
    ap.add_argument(
        "--auroc-pass",
        type=float,
        default=0.75,
        help="novel-only + threat-vs-safe AUROC at/above this => BINDING "
        "(primary verdict driver, v4)",
    )
    args = ap.parse_args()
    flag_on_hint = args.flag_on

    rows = load(Path(args.samples))
    scen = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    order = scen.get("probe_order", [])
    kinds = sorted({p["kind"] for p in order})
    n_of = {k: sum(1 for p in order if p["kind"] == k) for k in kinds}
    if len(rows) < 50 or not order:
        print(f"FAIL: unusable inputs (samples={len(rows)}, probe phases={len(order)})")
        return 1

    # Validity guard (2026-07-06, run B): a dead or arrested agent yields a
    # stale, flat readout that the binding logic would mis-report as BLIND.
    # Detect it FIRST and return DIED/STALLED, never a representational verdict.
    cyc = [r["cycles_completed"] for r in rows if r.get("cycles_completed") is not None]
    if cyc:
        # Frozen-TAIL detection, not flat-whole: run B advanced 10->1268 and
        # THEN arrested, so a max-min test misses it. Count trailing samples
        # sharing the final counter value; a long frozen tail (>=120 samples
        # ~= 60 s at 0.5 s polling) means the cycle worker died mid-run.
        tail = 0
        for v in reversed(cyc):
            if v == cyc[-1]:
                tail += 1
            else:
                break
        if tail >= 120:
            print(f"BINDING_PROBE: STALLED (cycles_completed frozen at {cyc[-1]} "
                  f"for the final {tail} samples -- cognitive arrest; verdict "
                  f"is not evaluable)")
            return 1
    via = [r["viability"] for r in rows if isinstance(r.get("viability"), (int, float))]
    if via and min(via) <= 0.0:
        # Locate death onset in step space to distinguish "died before probe"
        # (invalid) from "died after the probe segment" (probe still valid).
        first_dead = next(
            (float(r.get("step_est", 0)) for r in rows
             if isinstance(r.get("viability"), (int, float)) and r["viability"] <= 0.0),
            None,
        )
        probe_start = min(int(p["start"]) for p in order)
        if first_dead is not None and first_dead < probe_start:
            print(f"BINDING_PROBE: DIED (viability->0 at step~{first_dead:.0f}, before "
                  f"probe start {probe_start}; the curriculum killed the agent -- "
                  f"lower combat_hit dose. Verdict not evaluable)")
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
    # Exclusion margins scale with the scheduled inter-phase gap (compressed
    # schedules shrink the gap from 180 to 100 steps; fixed 20/60 margins
    # would swallow the entire baseline window between probe phases).
    sched_gap = 100
    if len(order) >= 2:
        starts = sorted(int(p["start"]) for p in order)
        sched_gap = max(20, min(b - a for a, b in zip(starts, starts[1:])) - phase_len)
    lead, tail = max(5, sched_gap // 10), max(20, sched_gap // 2)
    gap_vals = [
        v
        for s, v in prio_by_step
        if s >= probe_start - 200
        and not any(
            int(p["start"]) - lead <= s <= int(p["start"]) + phase_len + tail
            for p in order
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
    hits = {k: 0 for k in kinds}
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

    # Distribution-level discrimination statistic (2026-07-07, run C): the
    # per-phase min-vs-3sigma criterion is an extreme-value test against a
    # non-stationary baseline (sigma varied 0.010 -> 0.071 across runs) and is
    # underpowered for the ~0.05-0.13 conditioned-response effect sizes
    # observed. Secondary readout: phase-MEAN priority per probe phase,
    # threat vs safe, rank-sum AUROC. AUROC > 0.5 means threat-pair adjacency
    # lowers priority relative to safe-pair adjacency -- the binding signal --
    # regardless of absolute threshold placement.
    def phase_mean(p) -> float | None:
        lo, hi = int(p["start"]) + 5, int(p["start"]) + phase_len + 40
        w = [v for s, v in prio_by_step if lo <= s <= hi]
        return (sum(w) / len(w)) if w else None

    threat_means = [
        m for p in order if p["kind"].endswith("_threat")
        for m in [phase_mean(p)] if m is not None
    ]
    safe_means = [
        m for p in order if p["kind"].endswith("_safe")
        for m in [phase_mean(p)] if m is not None
    ]
    auroc = None
    nov_auroc = None
    if threat_means and safe_means:
        # Mann-Whitney U by direct rank comparison (n is tiny; exact count).
        u = sum(1 for t in threat_means for s in safe_means if t < s)
        u += 0.5 * sum(1 for t in threat_means for s in safe_means if t == s)
        auroc = u / (len(threat_means) * len(safe_means))
        # Same statistic restricted to NOVEL pairs only: the generalization
        # signal proper (trained pairs could be memorized).
        nt_means = [
            m for p in order if p["kind"] == "probe_novel_threat"
            for m in [phase_mean(p)] if m is not None
        ]
        ns_means = [
            m for p in order if p["kind"] == "probe_novel_safe"
            for m in [phase_mean(p)] if m is not None
        ]
        nov_auroc = None
        if nt_means and ns_means:
            nu = sum(1 for t in nt_means for s in ns_means if t < s)
            nu += 0.5 * sum(1 for t in nt_means for s in ns_means if t == s)
            nov_auroc = nu / (len(nt_means) * len(ns_means))
        print(
            f"discrimination: threat-vs-safe phase-mean AUROC={auroc:.3f} "
            f"(n={len(threat_means)}v{len(safe_means)}; 0.5=chance, 1.0=perfect) "
            f"| novel-only AUROC={nov_auroc if nov_auroc is None else round(nov_auroc, 3)}"
        )

    # LEARNING CHECK (added 2026-07-06): did the aversive relation get learned
    # AT ALL? Measure priority deflection DURING train_threat phases (threat
    # actively firing). If training itself never deflects, the probe's BLIND is
    # a training-signal failure, not a binding failure -- report it as such so
    # the two are never conflated again (the first ablation read BLIND on both
    # legs because collision damage was graced to nothing).
    train_threat_phases = [
        p for p in scen.get("schedule", []) if p.get("kind") == "train_threat"
    ]
    tt_min = None
    for p in train_threat_phases:
        lo, hi = int(p["start"]) + 5, int(p["start"]) + phase_len
        w = [v for s, v in prio_by_step if lo <= s <= hi]
        if w:
            m = min(w)
            tt_min = m if tt_min is None else min(tt_min, m)
    trained_response = tt_min is not None and tt_min < thresh
    print(
        f"learning check: train_threat min priority={round(tt_min, 4) if tt_min is not None else None} "
        f"vs thresh={thresh:.4f} -> relation {'LEARNED' if trained_response else 'NOT learned'}"
    )

    # v4 verdict (2026-07-07): the AUROC rank statistic is PRIMARY; the
    # per-phase 3sigma count above is diagnostic only. It was shown
    # underpowered for the measured ~0.05-0.13 effect -- it read
    # LEARNING_FAILED on a run whose threat-vs-safe AUROC was 0.806 and
    # novel-only 0.778 (exact one-sided Mann-Whitney p~0.047). The
    # generalization claim rests on the NOVEL-ONLY statistic: held-out
    # pairings discriminated threat from safe. --auroc-pass (default 0.75)
    # is the decision boundary (the "measure, don't assume" discipline
    # applied to the detector itself).
    if nov_auroc is None or auroc is None:
        verdict = "UNEVALUABLE"  # a probe category is missing; no test forms
    elif nov_auroc >= args.auroc_pass and auroc >= args.auroc_pass:
        verdict = "BINDING"  # generalization AND full contrast both discriminate
    elif auroc >= args.auroc_pass and nov_auroc < 0.5:
        verdict = "MEMORIZER"  # trained pairs separate, novel do not
    elif auroc <= 1.0 - args.auroc_pass:
        verdict = "BLIND"  # at/below chance (flags-off leg lands here ~0.167)
    else:
        verdict = "INCONCLUSIVE"  # signal present, under the confirmatory bar

    print(
        f"gap baseline: mean={mean:.4f} std={std:.4f} thresh={thresh:.4f} "
        f"(n={len(gap_vals)})"
    )
    print(
        "probe phases deflected: "
        + " ".join(f"{k}={hits[k]}/{n_of[k]}" for k in kinds)
        + f" (total {total}/{len(order)})"
    )
    print(f"BINDING_PROBE: {verdict}")

    if args.expect == "pass":
        return 0 if verdict == "BINDING" else 1
    if args.expect == "fail":
        return 0 if verdict != "BINDING" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
