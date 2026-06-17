"""Integration sweep harness — the self-model program's falsification track.

A phase that adds a feedback pathway must *prove* it raises integration. This
script sweeps the Phase-0 perturbational-complexity proxy (PCI / Phi-style; see
``decadic.metrics.integration``) across configurations and reports PCI so the
claim "the self-model spine raises integration" is falsifiable rather than
asserted.

For each preset and each seed it builds ONE spine-capable stack and probes it
twice on the *same weights* -- once with the loop severed (spine not fed) and once
with the loop closed -- and records ``pci_on - pci_off``. Measuring one stack both
ways (rather than two separate builds, whose random init would differ) isolates
the loop as the only variable. The spine is zero-init, so the as-built delta is
*exactly* 0 (parity); pass ``--learned-sigma S`` to fill the spine ingress with
N(0, S) weights, emulating a *learned* loop -- the regime where a real closing of
the feedback loop should lift PCI.

Usage (CPU, no downloads -- encoder_mode is pinned to "zeros"):

    python scripts/integration_sweep.py --presets tiny --seeds 0,1,2 \
        --learned-sigma 0.3 --out logs/integration_sweep.jsonl

Writes one JSON object per (preset, seed) to ``--out`` (JSONL) and prints a table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _build_spine_stack(preset: str, *, learned_sigma: float):
    import torch

    from decadic.nn.config import neural_config_from_env, resolve_preset
    from decadic.nn.faculties import CognitionFaculties
    from decadic.nn.neural_stack import NeuralCognitiveStack

    cfg = neural_config_from_env(resolve_preset(preset))
    fac = CognitionFaculties(
        perception_feedback=False,
        perception_mode="oracle",
        encoder_mode="zeros",
        self_model_feedback=True,
    )
    stack = NeuralCognitiveStack(cfg, faculties=fac)
    if learned_sigma > 0.0 and hasattr(stack, "self_ingress"):
        with torch.no_grad():
            stack.self_ingress.weight.normal_(0.0, learned_sigma)
            stack.self_ingress.bias.normal_(0.0, learned_sigma * 0.3)
    return stack


def _measure(preset: str, seed: int, learned_sigma: float) -> dict[str, Any]:
    from decadic.metrics.integration import perturbational_complexity

    # One stack, probed both ways: the loop is the only variable (same weights).
    stack = _build_spine_stack(preset, learned_sigma=learned_sigma)
    stack.has_self_model_feedback = False  # sever the loop (spine not fed)
    off = perturbational_complexity(stack, seed=seed)
    stack.has_self_model_feedback = True  # close the loop
    on = perturbational_complexity(stack, seed=seed)
    return {
        "preset": preset,
        "seed": seed,
        "learned_sigma": learned_sigma,
        "pci_off": round(off.pci, 6),
        "pci_on": round(on.pci, 6),
        "pci_delta": round(on.pci - off.pci, 6),
        "active_off": round(off.active_fraction, 6),
        "active_on": round(on.active_fraction, 6),
        "persistence_off": round(off.persistence, 6),
        "persistence_on": round(on.persistence, 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--presets", default="tiny", help="comma-separated preset names")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated integer seeds")
    ap.add_argument(
        "--learned-sigma",
        type=float,
        default=0.0,
        help="stddev for emulating a learned spine ingress (0 = as-built, zero-init)",
    )
    ap.add_argument("--out", default=None, help="optional JSONL output path")
    args = ap.parse_args()

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    for preset in presets:
        for seed in seeds:
            rows.append(_measure(preset, seed, args.learned_sigma))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    print(f"{'preset':<10} {'seed':>4}  {'pci_off':>9} {'pci_on':>9} {'delta':>9}")
    print("-" * 46)
    for r in rows:
        print(
            f"{r['preset']:<10} {r['seed']:>4}  "
            f"{r['pci_off']:>9.4f} {r['pci_on']:>9.4f} {r['pci_delta']:>9.4f}"
        )
    for preset in presets:
        deltas = [r["pci_delta"] for r in rows if r["preset"] == preset]
        if deltas:
            mean = statistics.fmean(deltas)
            verdict = "RAISES" if mean > 1e-6 else "no-change" if abs(mean) <= 1e-6 else "LOWERS"
            print(f"\n{preset}: mean PCI delta = {mean:+.4f} over {len(deltas)} seeds -> {verdict}")
    if args.learned_sigma == 0.0:
        print(
            "\nNote: spine is zero-init -> on==off by construction. Re-run with "
            "--learned-sigma 0.3 to probe a learned loop."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
