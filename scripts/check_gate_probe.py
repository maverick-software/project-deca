"""WS3-P1 verdict: analyze gate_probe samples for reflex correctness.

Three checks, sample-stream based (no fragile cycle/step alignment):
1. Threat reflex — every fast_path_hits increment must coincide with a gate
   escalation within +/- LATENCY_SAMPLES samples (the reflex that must never
   be suppressed).
2. Novelty response — high-novelty samples after warmup must escalate within
   LATENCY_SAMPLES samples.
3. Steady-state calm — in quiet samples (no events, low novelty), the
   escalated fraction must stay at or below --max-quiet-rate.

Usage:
    python scripts/check_gate_probe.py <samples.jsonl> [--max-quiet-rate 0.10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LATENCY_SAMPLES = 2
# Calibrated 2026-07-04 (probe redesign): with percept-source novelty + the
# recency horizon, ambient sits at ~0.001 (p99) while a genuine first
# exposure measures ~0.35 (the encoder partially assimilates even a teleport,
# so the old 0.8 bar was unreachable by construction). 0.20 is ~200x ambient:
# a clean separator on measured data, not a hopeful one.
NOVELTY_HIGH = 0.20
WARMUP_CYCLES = 500
# Calm is judged only outside a recovery window after any stimulus: post-threat
# vigilance is designed behavior (pain decays slowly; prediction error stays
# elevated after out-of-distribution episodes). You don't measure resting
# heart rate mid-startle.
RECOVERY_CYCLES = 300


def load(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("metrics") if isinstance(r.get("metrics"), dict) else r
            if isinstance(m, dict) and "gate_escalations" in m:
                m = dict(m)
                m.setdefault("cycle", r.get("cycle"))
                rows.append(m)
    return rows


def _nov(m: dict) -> float:
    """Novelty signal for verdicts: the rolling-window peak when available
    (spikes last ~1-3 cycles; the sampler reads every ~6 -- the raw value
    misses them by construction), else the raw per-cycle input."""
    v = m.get("gate_i_novelty_peak")
    if v is None:
        v = m.get("gate_i_novelty")
    return float(v or 0.0)


def check(
    rows: list[dict], max_quiet_rate: float, expect_novel: int | None = None
) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def esc_delta(i: int, j: int) -> float:
        a = rows[max(0, i)].get("gate_escalations", 0) or 0
        b = rows[min(len(rows) - 1, j)].get("gate_escalations", 0) or 0
        return float(b) - float(a)

    # 1. threat reflex
    threat_events = 0
    threat_hits = 0
    for i in range(1, len(rows)):
        prev_fp = rows[i - 1].get("fast_path_hits", 0) or 0
        cur_fp = rows[i].get("fast_path_hits", 0) or 0
        if cur_fp > prev_fp:
            threat_events += 1
            if esc_delta(i - LATENCY_SAMPLES, i + LATENCY_SAMPLES) >= 1:
                threat_hits += 1
    results.append(
        (
            "threat reflex",
            threat_events > 0 and threat_hits == threat_events,
            f"{threat_hits}/{threat_events} injected threats answered"
            + ("" if threat_events else " - NO threats observed (event injection failed?)"),
        )
    )

    # 2. novelty response. When --expect-novel is given, the burst COUNT is
    # part of the contract: too few means a first exposure went unnoticed,
    # too many means either ambient noise or -- the redesign's key assertion
    # -- a REVISIT spiked that correct episodic memory should have recognized
    # (habituation-across-events is a feature under test, not a nuisance).
    novel_events = 0
    novel_hits = 0
    i = 0
    while i < len(rows):
        m = rows[i]
        cyc = m.get("cycle") or m.get("cycles_completed") or 0
        if _nov(m) >= NOVELTY_HIGH and cyc > WARMUP_CYCLES:
            novel_events += 1
            if esc_delta(i - 1, i + LATENCY_SAMPLES) >= 1:
                novel_hits += 1
            # skip the rest of this burst
            while i < len(rows) and _nov(rows[i]) >= NOVELTY_HIGH:
                i += 1
        else:
            i += 1
    count_ok = novel_events > 0 if expect_novel is None else novel_events == expect_novel
    expect_note = "" if expect_novel is None else f" (expected exactly {expect_novel})"
    results.append(
        (
            "novelty response",
            count_ok and novel_hits == novel_events,
            f"{novel_hits}/{novel_events} novelty bursts answered{expect_note}"
            + ("" if novel_events else " - NO high-novelty samples after warmup"),
        )
    )

    # 3. steady-state calm (outside recovery windows)
    def _cycle(m: dict) -> float:
        return float(m.get("cycle") or m.get("cycles_completed") or 0)

    stimulus_cycles: list[float] = []
    for i in range(1, len(rows)):
        if (rows[i].get("fast_path_hits", 0) or 0) > (rows[i - 1].get("fast_path_hits", 0) or 0):
            stimulus_cycles.append(_cycle(rows[i]))
    for m in rows:
        if _nov(m) >= NOVELTY_HIGH:
            stimulus_cycles.append(_cycle(m))

    def in_recovery(cyc: float) -> bool:
        return any(0 <= cyc - s <= RECOVERY_CYCLES for s in stimulus_cycles)

    quiet = [
        m
        for m in rows
        if _cycle(m) > WARMUP_CYCLES
        and _nov(m) < NOVELTY_HIGH
        and not in_recovery(_cycle(m))
    ]
    if quiet:
        rate = sum(1 for m in quiet if m.get("gate_escalated")) / len(quiet)
        results.append(
            (
                "steady-state calm",
                rate <= max_quiet_rate,
                f"quiet-sample escalation {rate:.3f} vs limit {max_quiet_rate:.3f} ({len(quiet)} samples)",
            )
        )
    else:
        results.append(("steady-state calm", False, "no quiet samples found"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples")
    ap.add_argument("--max-quiet-rate", type=float, default=0.10)
    ap.add_argument(
        "--expect-novel",
        type=int,
        default=None,
        help="Exact number of novelty bursts expected (count the novel: events "
        "in the spec; revisit: events must NOT add to this count -- that is "
        "the habituation assertion)",
    )
    args = ap.parse_args()

    rows = load(args.samples)
    if len(rows) < 20:
        print(f"FAIL: only {len(rows)} usable samples (gate telemetry missing? gate not enabled?)")
        return 1
    results = check(rows, args.max_quiet_rate, expect_novel=args.expect_novel)
    ok = all(passed for _, passed, _ in results)
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print("GATE_PROBE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
