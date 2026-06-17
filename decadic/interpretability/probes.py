"""Interpretability probes for the cognitive State Bus latents.

Two halves, both deliberately decoupled from cognition:

1. Capture (offline supervision). When ``DECADIC_PROBE_CAPTURE`` is on, each cycle
   appends ``{latents, targets}`` to a JSONL. The *targets* are drawn ONLY from
   the eval-only ground-truth channels (the oracle entities the discovery system
   retains for evaluation, the sensed body pose, the reservoirs/viability) - never
   from any cognitive output. This is the same purity contract the discovery
   subsystem uses: labels are for measurement, never for learning behaviour.

2. Read-out (online, read-only). When ``DECADIC_PROBE_PATH`` points at a trained
   probe bank, we decode the current latents into predicted interpretable
   variables and report which latent (and which axis of it) best tracks each
   variable, along with the probe's held-out quality (R^2 / accuracy). The probe
   weights are a fixed linear map applied with numpy; they are never added to the
   torch graph or the optimizer, so behaviour is unchanged whether or not a probe
   bank is present.
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from decadic import config as C

if TYPE_CHECKING:
    from decadic.cycle.types import CycleContext

# Ordered latent sources decoded by the probes. Each maps to a State Bus /
# cycle vector available at trace-assembly time.
LATENT_KEYS = ["emotion", "state_mind", "metacognition", "narrative", "z5"]

# Targets whose true values come only from eval-only channels.
BINARY_TARGETS = {"contact"}


def latents_from_cycle(ctx: "CycleContext") -> dict[str, list[float]]:
    """Snapshot the cognitive latents that probes decode (read-only)."""
    sb = ctx.state_bus
    z5 = ctx.latents.get("z5_snapshot") if isinstance(ctx.latents, dict) else None

    def _vec(x: Any) -> list[float]:
        return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]

    return {
        "emotion": _vec(sb.emotion_physio),
        "state_mind": _vec(sb.state_of_mind),
        "metacognition": _vec(sb.metacognition),
        "narrative": _vec(sb.narrative_emb),
        "z5": _vec(z5) if isinstance(z5, list) else [],
    }


def targets_from_truth(ctx: "CycleContext", obs: dict[str, Any] | None) -> dict[str, float]:
    """Derive interpretable targets ONLY from eval-only ground-truth channels."""
    out: dict[str, float] = {}
    perc = ctx.perceptual
    pos = list(perc.proprio_position) if perc and perc.proprio_position else None
    if pos and len(pos) >= 3:
        out["height"] = float(pos[2])
    orient = getattr(perc, "proprio_orientation", None) if perc else None
    if isinstance(orient, list) and len(orient) >= 2:
        tilt = math.sqrt(float(orient[0]) ** 2 + float(orient[1]) ** 2)
        out["upright"] = float(math.cos(min(math.pi, tilt)))
    # Nearest object distance from the eval-only oracle entities.
    truth = getattr(perc, "oracle_truth", None) if perc else None
    if pos and isinstance(truth, list) and truth:
        dists: list[float] = []
        for e in truth:
            p = e.get("position") if isinstance(e, dict) else None
            if isinstance(p, list) and len(p) >= 3:
                dists.append(math.dist(pos[:3], [float(x) for x in p[:3]]))
        if dists:
            out["nearest_object_dist"] = float(min(dists))
    if ctx.viability is not None:
        out["viability"] = float(ctx.viability.value) / 100.0
    h = ctx.homeostasis
    if h is not None:
        for name in ("hydration", "energy", "integrity"):
            v = getattr(h, name, None)
            if v is not None:
                out[name] = float(v) / 100.0
    contacts = None
    if isinstance(obs, dict):
        contacts = (obs.get("proprioception") or {}).get("contacts")
    if isinstance(contacts, list) and contacts:
        out["contact"] = 1.0 if max(abs(float(c)) for c in contacts) > 1e-6 else 0.0
    return out


def maybe_capture(ctx: "CycleContext", obs: dict[str, Any] | None) -> None:
    """Append one ``{latents, targets}`` row to the capture JSONL when enabled."""
    capture_on = ctx.probe_capture if ctx.probe_capture is not None else C.probe_capture_enabled()
    if not capture_on:
        return
    targets = targets_from_truth(ctx, obs)
    if not targets:
        return
    rec = {
        "cycle": int(ctx.state_bus.cycle_index),
        "latents": latents_from_cycle(ctx),
        "targets": targets,
    }
    # Serialize on-thread (cheap, keeps content identical); hand the file append to the
    # background JSONL writer so the disk write never blocks the cognitive cycle.
    try:
        line = json.dumps(rec)
    except (TypeError, ValueError):
        return
    from decadic.io import get_jsonl_writer

    get_jsonl_writer().append(C.probe_capture_path(), line)


class ProbeBank:
    """A trained set of linear/logistic probes, applied as a pure read-out."""

    def __init__(self, data: dict[str, Any]):
        self.data = data or {}

    @property
    def targets(self) -> dict[str, Any]:
        return self.data.get("targets", {})

    def readout(self, latents: dict[str, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for target, spec in self.targets.items():
            best = spec.get("best_latent")
            per = (spec.get("per_latent") or {}).get(best)
            if not per:
                continue
            vec = np.asarray(latents.get(best, []), dtype=np.float64).reshape(-1)
            w = np.asarray(per.get("w", []), dtype=np.float64)
            if vec.size == 0 or w.size != vec.size:
                continue
            raw = float(vec @ w + float(per.get("b", 0.0)))
            kind = spec.get("kind", "regression")
            predicted = 1.0 / (1.0 + math.exp(-raw)) if kind == "classification" else raw
            axis = int(np.argmax(np.abs(w))) if w.size else -1
            out[target] = {
                "predicted": round(predicted, 4),
                "kind": kind,
                "best_latent": best,
                "axis": axis,
                "score": round(float(per.get("score", 0.0)), 4),
                "score_kind": "accuracy" if kind == "classification" else "r2",
            }
        return out


_BANK_CACHE: dict[tuple[str, float], ProbeBank] = {}


def load_probe_bank(path: str) -> ProbeBank | None:
    """Load (and cache by mtime) a trained probe bank from JSON."""
    if not path or not os.path.exists(path):
        return None
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    cached = _BANK_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    bank = ProbeBank(data)
    _BANK_CACHE.clear()
    _BANK_CACHE[key] = bank
    return bank


def readout_for_cycle(ctx: "CycleContext") -> dict[str, Any] | None:
    """Decode the current latents through the trained bank, if one is configured."""
    bank = load_probe_bank(C.probe_path())
    if bank is None:
        return None
    try:
        result = bank.readout(latents_from_cycle(ctx))
    except Exception:
        return None
    return result or None
