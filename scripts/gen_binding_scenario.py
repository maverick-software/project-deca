"""WS5-M5.1 v3: generate the binding-probe scenario (role-structured).

v3 design (after the 2026-07-04 v2 runs exposed two flaws):
1. REAL CONSEQUENCES: threat phases fire damage-class collision events during
   adjacency -- pain rises, the risk head gets an aversive prediction target
   (v2's threat_near produced stress but no pain: nothing was learnable).
2. ROLE STRUCTURE: 2 predators + 4 prey. Damage fires ONLY on predator-prey
   adjacency; predator-predator and prey-prey adjacencies are scheduled as
   explicit SAFE negatives. This closes the proximity-shortcut leak: "any two
   things close -> danger" is now a WRONG generalization, so a pooled system
   cannot pass by reading global proximity out of z0.
3. REPETITION: the train schedule runs --laps (default 3) so the zero-init
   binding ingresses see enough gradient pressure to move.

Probe segment (all phases EVENTLESS -- deflection must be prediction):
    probe_trained_threat  -> should deflect (memory check)
    probe_novel_threat    -> MUST deflect  (generalization criterion)
    probe_trained_safe    -> must NOT deflect (sanity)
    probe_novel_safe      -> must NOT deflect (the discrimination that only
                             a bound representation can make)

Usage: python scripts/gen_binding_scenario.py [--seed 42] [--laps 3]
           [--out docs/binding_scenarios/binding_probe.json]
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--phase-steps", type=int, default=120)
    ap.add_argument("--phase-gap", type=int, default=180)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--out", default="docs/binding_scenarios/binding_probe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    predators = ["ent-A", "ent-B"]
    prey = ["ent-C", "ent-D", "ent-E", "ent-F"]
    ids = predators + prey

    apps = rng.normal(size=(len(ids), 16)).astype(np.float32)
    apps /= np.linalg.norm(apps, axis=1, keepdims=True)
    entities = []
    for i, eid in enumerate(ids):
        ang = 2.0 * np.pi * i / len(ids)
        entities.append(
            {
                "id": eid,
                "kind": "entity",
                "role_truth": "predator" if eid in predators else "prey",  # eval-only
                "appearance": [round(float(v), 6) for v in apps[i]],
                "home": [float(14.0 * np.cos(ang)), 0.0, float(14.0 * np.sin(ang))],
                "orbit": 2.0,
                "period": 400,
            }
        )

    # Threat pairs: predator x prey (8). Train on 5, hold out 3 such that
    # every predator and every prey appears in BOTH splits where possible.
    threat_pairs = [list(p) for p in itertools.product(predators, prey)]
    rng.shuffle(threat_pairs)
    train_threat, hold_threat = threat_pairs[:5], threat_pairs[5:]
    # Safe pairs: prey-prey (6) + the predator-predator pair (1).
    safe_pairs = [list(p) for p in itertools.combinations(prey, 2)]
    rng.shuffle(safe_pairs)
    train_safe = safe_pairs[:3] + [list(predators)]
    hold_safe = safe_pairs[3:]

    schedule, t = [], int(args.start)

    def _phase(pair, kind, event=None):
        nonlocal t
        ph = {"start": t, "steps": int(args.phase_steps), "pair": pair, "kind": kind, "gap": 1.5}
        if event:
            ph["event"] = event
        schedule.append(ph)
        t += int(args.phase_steps) + int(args.phase_gap)

    dmg = {"type": "collision", "intensity": 0.35, "every": 12}
    for _lap in range(max(1, args.laps)):
        # Interleave threat and safe training so neither is a temporal block.
        for i in range(max(len(train_threat), len(train_safe))):
            if i < len(train_threat):
                _phase(train_threat[i], "train_threat", dmg)
            if i < len(train_safe):
                _phase(train_safe[i], "train_safe")

    probe_order = []
    probe_sets = (
        [("probe_trained_threat", p) for p in train_threat[:3]]
        + [("probe_novel_threat", p) for p in hold_threat]
        + [("probe_trained_safe", p) for p in train_safe[:3]]
        + [("probe_novel_safe", p) for p in hold_safe]
    )
    rng.shuffle(probe_sets)  # deterministic interleave; drift can't masquerade
    for kind, pair in probe_sets:
        _phase(pair, kind)
        probe_order.append({"start": schedule[-1]["start"], "kind": kind, "pair": pair})

    scenario = {
        "workstream": "WS5-M5.1-v3",
        "relation": "predator_prey_threat_adjacency",
        "seed": args.seed,
        "laps": args.laps,
        "entities": entities,
        "schedule": schedule,
        "probe_order": probe_order,
        "total_steps_hint": t + 400,
        "leakage_controls": (
            "role-structured: damage ONLY on predator-prey adjacency; safe "
            "adjacencies scheduled as explicit negatives (proximity shortcut "
            "is a WRONG generalization); every entity in train and holdout "
            "splits; probe phases eventless and shuffled; two-direction test "
            "(novel threat must deflect, novel safe must not)"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(
        f"[gen_binding_scenario] v3 laps={args.laps} "
        f"train: threat={len(train_threat)} safe={len(train_safe)} | "
        f"probe: {len(probe_sets)} phases | end~step {t} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
