"""WS3-P2 offline gate tuner: replay recorded runs against config grids.

Replays the gate decision function over recorded samples (which carry the
raw normalized inputs gate_i_*; older runs fall back to backing inputs out
of the weighted contributions gate_c_*). Zero live machine time.

Caveats (documented, acceptable for calibration):
- Samples are ~0.5-2 s apart (several cycles each), so replayed rates are
  sample-level approximations of per-cycle rates, and hysteresis operates on
  samples rather than cycles here. Rankings are still monotone in the
  quantities that matter (relative sensitivity of configs).

Usage:
    python scripts/tune_gate.py reports/<run>/training_eval_*.jsonl [more.jsonl ...]
    python scripts/tune_gate.py reports/soak_*/harness_samples.jsonl --target 0.05
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decadic.cycle.attention_gate import (  # noqa: E402
    DEFAULT_GATE_WEIGHTS,
    AttentionGate,
    GateInputs,
)

WEIGHT_PRESETS = {
    "default": DEFAULT_GATE_WEIGHTS,  # (novelty, pe, affect, priority)
    "novelty_heavy": (0.50, 0.20, 0.20, 0.10),
    "pe_heavy": (0.20, 0.50, 0.20, 0.10),
    "affect_heavy": (0.25, 0.20, 0.45, 0.10),
    "balanced": (0.25, 0.25, 0.25, 0.25),
}
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
BUDGET_GAINS = [0.5, 1.0]
NOVELTY_HIGH = 0.8


def load_inputs(paths: list[str]) -> list[GateInputs]:
    """Extract per-sample GateInputs from eval or harness sample files."""
    seq: list[GateInputs] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                m = r.get("metrics") if isinstance(r.get("metrics"), dict) else r
                if not isinstance(m, dict):
                    continue
                if "gate_i_novelty" in m:
                    seq.append(
                        GateInputs(
                            novelty=float(m.get("gate_i_novelty") or 0),
                            prediction_error=float(m.get("gate_i_prediction_error") or 0),
                            affect=float(m.get("gate_i_affect") or 0),
                            priority_investigate=float(m.get("gate_i_priority") or 0),
                            fast_path_threat=bool(m.get("gate_i_fast_path")),
                        )
                    )
                elif "gate_c_novelty" in m:
                    # Back out raw inputs from contributions recorded under the
                    # run's weights (assume defaults - true for untuned runs).
                    w = DEFAULT_GATE_WEIGHTS
                    wt = sum(w)
                    seq.append(
                        GateInputs(
                            novelty=min(1.0, float(m.get("gate_c_novelty") or 0) * wt / w[0]),
                            prediction_error=min(
                                1.0, float(m.get("gate_c_prediction_error") or 0) * wt / w[1]
                            ),
                            affect=min(1.0, float(m.get("gate_c_affect") or 0) * wt / w[2]),
                            priority_investigate=min(
                                1.0, float(m.get("gate_c_priority") or 0) * wt / w[3]
                            ),
                            fast_path_threat=bool(m.get("gate_i_fast_path")),
                        )
                    )
    return seq


def replay(seq: list[GateInputs], *, threshold: float, weights, budget_gain: float,
           target_rate: float) -> dict:
    gate = AttentionGate(
        threshold=threshold,
        weights=weights,
        target_rate=target_rate,
        hysteresis_k=3,
        rate_window=500,
        budget_gain=budget_gain,
    )
    escalated = []
    novelty_events = 0
    novelty_answered = 0
    in_burst = False
    for idx, inp in enumerate(seq):
        d = gate.decide(inp)
        escalated.append(d.escalate)
        if inp.novelty >= NOVELTY_HIGH and idx > len(seq) // 10:
            if not in_burst:
                novelty_events += 1
                if d.escalate or (idx + 1 < len(seq)):
                    # answered if this or next decision escalates
                    pass
                in_burst = True
                burst_start = idx
            if in_burst and d.escalate and idx <= burst_start + 2:
                novelty_answered += 1
                burst_start = -(10**9)  # count once per burst
        else:
            in_burst = False
    n = len(escalated)
    half = n // 2
    return {
        "rate_overall": sum(escalated) / max(1, n),
        "rate_settled": sum(escalated[half:]) / max(1, n - half),
        "novelty_events": novelty_events,
        "novelty_answered": novelty_answered,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples", nargs="+", help="sample jsonl files (globs ok)")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.samples:
        paths.extend(globmod.glob(pattern))
    seq = load_inputs(paths)
    if len(seq) < 50:
        print(f"FAIL: only {len(seq)} samples with gate inputs across {len(paths)} file(s). "
              "Run with DECADIC_GATE_ENABLED=1 to record gate telemetry.")
        return 1
    print(f"replaying {len(seq)} samples from {len(paths)} file(s)\n")

    rows = []
    for preset_name, weights in WEIGHT_PRESETS.items():
        for th in THRESHOLDS:
            for bg in BUDGET_GAINS:
                r = replay(seq, threshold=th, weights=weights, budget_gain=bg,
                           target_rate=args.target)
                miss = abs(r["rate_settled"] - args.target)
                responded = (
                    r["novelty_events"] == 0 or r["novelty_answered"] >= r["novelty_events"]
                )
                rows.append((miss, responded, preset_name, th, bg, r))

    # responsive configs first, then closest settled rate to target
    rows.sort(key=lambda x: (not x[1], x[0]))
    print(f"{'weights':<14} {'thresh':>6} {'budget':>6} {'settled':>8} {'overall':>8} {'novelty':>9}")
    for miss, responded, preset, th, bg, r in rows[: args.top]:
        nov = f"{r['novelty_answered']}/{r['novelty_events']}"
        print(f"{preset:<14} {th:>6.2f} {bg:>6.1f} {r['rate_settled']:>8.3f} "
              f"{r['rate_overall']:>8.3f} {nov:>9}")

    best = rows[0]
    _, _, preset, th, bg, r = best
    w = WEIGHT_PRESETS[preset]
    print("\nrecommended config:")
    print(f'  $env:DECADIC_GATE_THRESHOLD="{th}"')
    print(f'  $env:DECADIC_GATE_WEIGHTS="{w[0]},{w[1]},{w[2]},{w[3]}"')
    print(f'  $env:DECADIC_GATE_BUDGET_GAIN="{bg}"')
    print(f"  (settled rate {r['rate_settled']:.3f} vs target {args.target}, "
          f"novelty {r['novelty_answered']}/{r['novelty_events']})")
    print("\nnote: sample-level replay approximates per-cycle rates; confirm with "
          "scripts/check_gate_probe.py on a live gate_probe run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
