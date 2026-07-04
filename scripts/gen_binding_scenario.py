"""WS5-M0.4: generate a binding-probe scenario file.

Entities carry controlled 16-d unit appearance vectors (seeded, near-
orthogonal by construction at these dims/counts). The relation under test is
threat-adjacency: during a scheduled phase, pair (a, b) becomes adjacent and
threat_near events fire, sourced to `a`. Train phases expose a subset of
pairs; `holdout_pairs` (never scheduled here) are reserved for the M5 probe's
novel-pairing test phase. Every entity appears in BOTH some train pair and
some holdout pair, so marginal entity statistics are uninformative -- the
leakage control from PRD ws5 risk 5, finalized at M5.1.

Usage:
    python scripts/gen_binding_scenario.py [--entities 6] [--seed 42] \
        [--out docs/eval_scenarios/binding_probe.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entities", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase-steps", type=int, default=120)
    ap.add_argument("--phase-gap", type=int, default=280)
    ap.add_argument("--start", type=int, default=600, help="first phase start (post-warmup)")
    # NOT docs/eval_scenarios: the /eval/scenarios route parses everything
    # there as an EvalSpec (different schema).
    ap.add_argument("--out", default="docs/binding_scenarios/binding_probe.json")
    args = ap.parse_args()

    n = max(4, int(args.entities))
    rng = np.random.default_rng(args.seed)
    ids = [f"ent-{chr(ord('A') + i)}" for i in range(n)]

    apps = rng.normal(size=(n, 16)).astype(np.float32)
    apps /= np.linalg.norm(apps, axis=1, keepdims=True)
    homes = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        homes.append([float(14.0 * np.cos(ang)), 0.0, float(14.0 * np.sin(ang))])

    entities = [
        {
            "id": ids[i],
            "kind": "entity",
            "appearance": [round(float(v), 6) for v in apps[i]],
            "home": homes[i],
            "orbit": 2.0,
            "period": 400,
        }
        for i in range(n)
    ]

    # Pair split: round-robin over all pairs, alternating train/holdout, then
    # repair so every entity has >= 1 of each (balance = leakage control).
    all_pairs = list(itertools.combinations(ids, 2))
    rng.shuffle(all_pairs)
    train, hold = [], []
    for i, p in enumerate(all_pairs):
        (train if i % 2 == 0 else hold).append(list(p))

    def _covered(pairs: list[list[str]]) -> set[str]:
        return {e for p in pairs for e in p}

    for eid in ids:
        if eid not in _covered(train):
            for p in hold:
                if eid in p:
                    train.append(p)
                    hold.remove(p)
                    break
        if eid not in _covered(hold):
            for p in train:
                if eid in p and len([q for q in train if eid in q]) > 1:
                    hold.append(p)
                    train.remove(p)
                    break

    schedule = []
    t = int(args.start)
    for pair in train:
        schedule.append(
            {
                "start": t,
                "steps": int(args.phase_steps),
                "pair": pair,
                "kind": "train",
                "gap": 1.5,
                "event": {"type": "threat_near", "intensity": 0.6, "every": 10},
            }
        )
        t += int(args.phase_steps) + int(args.phase_gap)

    # Probe segment (M5.1): EVENTLESS adjacency phases after training.
    # probe_trained re-presents trained pairs (memorization check);
    # probe_novel presents held-out pairings (the generalization criterion).
    # Interleaved deterministically so drift cannot masquerade as either.
    # A learned relation deflects priority on BOTH kinds; a memorizer only on
    # probe_trained; a pooled (flags-off) system on NEITHER (no events -> no
    # pain -> nothing else can carry the adjacency information).
    probe_order = []
    trained_probe = train[: len(hold)] if len(train) >= len(hold) else train
    pairs_interleaved = []
    for i in range(max(len(trained_probe), len(hold))):
        if i < len(trained_probe):
            pairs_interleaved.append(("probe_trained", trained_probe[i]))
        if i < len(hold):
            pairs_interleaved.append(("probe_novel", hold[i]))
    for kind, pair in pairs_interleaved:
        schedule.append(
            {
                "start": t,
                "steps": int(args.phase_steps),
                "pair": pair,
                "kind": kind,
                "gap": 1.5,
                # no "event": adjacency is the ONLY signal in the probe segment
            }
        )
        probe_order.append({"start": t, "kind": kind, "pair": pair})
        t += int(args.phase_steps) + int(args.phase_gap)

    scenario = {
        "workstream": "WS5-M5.1",
        "relation": "threat_adjacency",
        "seed": args.seed,
        "entities": entities,
        "schedule": schedule,
        "holdout_pairs": hold,
        "probe_order": probe_order,
        "total_steps_hint": t + 400,
        "leakage_controls": (
            "every entity appears in >=1 train AND >=1 holdout pair (marginal "
            "entity statistics uninformative); probe phases eventless (pain "
            "cannot carry the signal); probe_trained/probe_novel interleaved "
            "(drift cannot masquerade as generalization); pairs balanced by "
            "round-robin split of all C(n,2) pairings"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(
        f"[gen_binding_scenario] entities={n} train_pairs={len(train)} "
        f"holdout_pairs={len(hold)} phases_end~step {t} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
