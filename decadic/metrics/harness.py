"""WS2 measurement harness: REST sampler, rollups, and soak gates.

Design (see docs/ws2_measurement_harness_prd.md):

- Sampling is REST-polling based, not log-scraping: WS1 showed the API stays
  responsive even mid-stall, while the JSONL file log rotates away.
- The sampler captures the ENTIRE numeric surface of ``/agent/{id}/metrics``
  (252 keys at time of writing) plus selected labels, state-bus vector norms
  from ``/agent/{id}/state``, GPU stats via nvidia-smi, and host-side gauges.
- Stdlib only. This module must import without torch so tooling and tests can
  run anywhere.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# Non-numeric metrics worth keeping per sample.
LABEL_KEYS = ("priority_label", "status", "loss_canary_state", "preset", "encoder_mode")

# State-bus vector fields -> short column prefix (element letter).
STATE_VECTOR_KEYS = {
    "state_of_mind": "a",
    "emotion_physio": "b",
    "narrative_emb": "c",
    "metacognition": "e",
}
STATE_SCALAR_KEYS = ("pain_scalar", "pleasure_scalar", "priority_scalar", "cycle_index")


def _http_get_json(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None


def _find_key(obj: Any, key: str, max_depth: int = 6) -> Any:
    """Depth-first search for ``key`` anywhere in a nested dict payload."""
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key, max_depth - 1)
            if found is not None:
                return found
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for v in obj:
            found = _find_key(v, key, max_depth - 1)
            if found is not None:
                return found
    return None


def _l2(vec: Any) -> float | None:
    if not isinstance(vec, (list, tuple)) or not vec:
        return None
    try:
        return math.sqrt(sum(float(x) * float(x) for x in vec))
    except (TypeError, ValueError):
        return None


def _nvidia_smi() -> dict[str, float]:
    """GPU memory-used (MiB) and utilization (%); empty dict if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = out.stdout.strip().splitlines()[0]
        mem, util = (float(x.strip()) for x in first.split(",")[:2])
        return {"gpu_mem_used_mib": mem, "gpu_util_pct": util}
    except Exception:
        return {}


class HarnessSampler:
    """Polls a Decadic agent and appends flat JSON samples to a file."""

    def __init__(
        self,
        base_url: str,
        agent_id: str,
        out_path: str | Path,
        *,
        interval_s: float = 2.0,
        run_dir: str | Path | None = None,
        collect_gpu: bool = True,
        state_every: int = 5,
        disk_every: int = 5,
        http_get: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.out_path = Path(out_path)
        self.interval_s = max(0.2, float(interval_s))
        self.run_dir = Path(run_dir) if run_dir else self.out_path.parent
        self.collect_gpu = collect_gpu
        # /state rebuilds the full perceptual + LTM-graph payload per request
        # (cost grows with the graph) ON the server event loop. Polling it at
        # the base interval starves the cycle loop and can stall keepalive
        # pongs long enough to drop the observation websocket (shakedown
        # soak_20260702_165403). Sample it, and the run-dir disk walk, only
        # every Nth poll.
        self.state_every = max(1, int(state_every))
        self.disk_every = max(1, int(disk_every))
        self._http_get = http_get or _http_get_json
        self._prev_norms: dict[str, float] = {}
        self.samples_written = 0
        self.errors = 0
        self._poll_index = 0

    # -- single sample ----------------------------------------------------
    def sample_once(self) -> dict[str, Any] | None:
        t0 = time.time()
        metrics_resp = self._http_get(f"{self.base_url}/agent/{self.agent_id}/metrics")
        if metrics_resp is None:
            self.errors += 1
            return None
        metrics = metrics_resp.get("metrics") or metrics_resp

        row: dict[str, Any] = {"ts": round(t0, 3)}
        for k, v in metrics.items():
            if isinstance(v, bool):
                row[k] = int(v)
            elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                row[k] = v
            elif isinstance(v, (int, float)):
                row[k] = None
                row["nonfinite_metric_seen"] = row.get("nonfinite_metric_seen", 0) + 1
            elif k in LABEL_KEYS and isinstance(v, str):
                row[k] = v

        state_resp = None
        if self._poll_index % self.state_every == 0:
            state_resp = self._http_get(f"{self.base_url}/agent/{self.agent_id}/state")
        if state_resp is not None:
            payload = state_resp.get("payload") or state_resp
            for field, prefix in STATE_VECTOR_KEYS.items():
                norm = _l2(_find_key(payload, field))
                if norm is not None:
                    row[f"{prefix}_norm"] = round(norm, 6)
                    prev = self._prev_norms.get(prefix)
                    if prev is not None:
                        row[f"{prefix}_norm_drift"] = round(abs(norm - prev), 6)
                    self._prev_norms[prefix] = norm
            for field in STATE_SCALAR_KEYS:
                val = _find_key(payload, field)
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    row[f"state_{field}"] = val

        if self.collect_gpu:
            row.update(_nvidia_smi())
        if self._poll_index % self.disk_every == 0:
            try:
                usage = shutil.disk_usage(self.run_dir)
                row["disk_free_gb"] = round(usage.free / 1e9, 3)
                row["run_dir_mb"] = round(
                    sum(f.stat().st_size for f in self.run_dir.glob("**/*") if f.is_file()) / 1e6,
                    3,
                )
            except OSError:
                pass
        self._poll_index += 1
        row["sample_ms"] = round((time.time() - t0) * 1000.0, 2)
        return row

    def write_sample(self) -> dict[str, Any] | None:
        row = self.sample_once()
        if row is None:
            return None
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.samples_written += 1
        return row

    def run(self, stop: Callable[[], bool], *, max_seconds: float | None = None) -> None:
        """Blocking sampling loop; call from a thread. ``stop()`` ends it."""
        t_start = time.time()
        while not stop():
            if max_seconds is not None and time.time() - t_start >= max_seconds:
                break
            self.write_sample()
            time.sleep(self.interval_s)


# -- rollups ---------------------------------------------------------------

def load_samples(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def _bucket(ts: float, seconds: int) -> int:
    return int(ts // seconds) * seconds


def rollup(
    samples: list[dict[str, Any]],
    *,
    bucket_seconds: int = 60,
) -> list[dict[str, Any]]:
    """Aggregate flat samples into fixed time buckets.

    Numeric keys -> mean/min/max/last; ``cycles_completed`` additionally
    yields ``cycle_rate_hz`` (delta over wall time); label keys -> mode.
    """
    if not samples:
        return []
    buckets: dict[int, list[dict[str, Any]]] = {}
    for s in samples:
        ts = s.get("ts")
        if isinstance(ts, (int, float)):
            buckets.setdefault(_bucket(float(ts), bucket_seconds), []).append(s)

    numeric_keys: set[str] = set()
    label_keys: set[str] = set()
    for s in samples:
        for k, v in s.items():
            if k == "ts":
                continue
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
            elif isinstance(v, str):
                label_keys.add(k)

    out: list[dict[str, Any]] = []
    prev_last_cycle: float | None = None
    prev_bucket_ts: int | None = None
    for b_ts in sorted(buckets):
        rows = sorted(buckets[b_ts], key=lambda r: r.get("ts", 0))
        agg: dict[str, Any] = {"bucket_ts": b_ts, "n_samples": len(rows)}
        for k in sorted(numeric_keys):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            agg[f"{k}__mean"] = round(statistics.fmean(vals), 6)
            agg[f"{k}__min"] = min(vals)
            agg[f"{k}__max"] = max(vals)
            agg[f"{k}__last"] = vals[-1]
        for k in sorted(label_keys):
            vals = [r[k] for r in rows if isinstance(r.get(k), str)]
            if vals:
                agg[f"{k}__mode"] = statistics.mode(vals)
        last_cycle = agg.get("cycles_completed__last")
        if isinstance(last_cycle, (int, float)):
            if prev_last_cycle is not None and prev_bucket_ts is not None and b_ts > prev_bucket_ts:
                agg["cycle_rate_hz"] = round(
                    max(0.0, (last_cycle - prev_last_cycle)) / (b_ts - prev_bucket_ts), 4
                )
            prev_last_cycle = last_cycle
            prev_bucket_ts = b_ts
        out.append(agg)
    return out


def write_rollup_csv(rollup_rows: list[dict[str, Any]], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for r in rollup_rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rollup_rows)
    return str(p)


# -- soak gates --------------------------------------------------------------

DEFAULT_SOAK_GATES: dict[str, Any] = {
    # Thresholds are PRD 5.4 estimates; calibrate during the 1-hour shakedown.
    "max_stall_events": 0,
    "cycle_rate_floor_fraction": 0.5,  # hourly mean vs first-hour mean
    "max_nan_recovery_events": 0,
    "max_rss_growth_fraction": 0.20,  # reserved; RSS gauge lands with soak_run
    "require_pc_loss_decrease": True,
    "note_growth_events": True,
}


def evaluate_soak_gates(
    samples: list[dict[str, Any]],
    *,
    stall_events: int = 0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate PRD 5.4 gates over raw samples. Returns verdict structure."""
    cfg = dict(DEFAULT_SOAK_GATES)
    if config:
        cfg.update(config)
    gates: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        gates.append({"name": name, "ok": ok, "detail": detail})

    add(
        "no stalls",
        stall_events <= int(cfg["max_stall_events"]),
        f"stall_events={stall_events}",
    )

    hourly = rollup(samples, bucket_seconds=3600)
    rates = [r["cycle_rate_hz"] for r in hourly if isinstance(r.get("cycle_rate_hz"), (int, float))]
    if len(rates) >= 2:
        floor = rates[0] * float(cfg["cycle_rate_floor_fraction"])
        worst = min(rates[1:])
        add(
            "cycle rate holds",
            worst >= floor,
            f"first-hour={rates[0]:.2f}Hz worst-later={worst:.2f}Hz floor={floor:.2f}Hz",
        )
    else:
        add("cycle rate holds", None, "run shorter than 2 rollup hours - not evaluated")

    nan_vals = [
        s["nan_recovery_events"]
        for s in samples
        if isinstance(s.get("nan_recovery_events"), (int, float))
    ]
    nan_total = int(nan_vals[-1]) if nan_vals else 0
    add(
        "no NaN recoveries",
        nan_total <= int(cfg["max_nan_recovery_events"]),
        f"nan_recovery_events={nan_total}",
    )

    pc = [
        (s.get("ts", 0.0), s["neural_pc_loss_last"])
        for s in samples
        if isinstance(s.get("neural_pc_loss_last"), (int, float))
    ]
    if cfg["require_pc_loss_decrease"] and len(pc) >= 10:
        ys = [y for _, y in pc]
        half = len(ys) // 2
        first_half = statistics.fmean(ys[:half])
        second_half = statistics.fmean(ys[half:])
        add(
            "pc loss decreases",
            second_half < first_half,
            f"half-means {first_half:.4f} -> {second_half:.4f}",
        )
    elif cfg["require_pc_loss_decrease"]:
        add("pc loss decreases", False, f"only {len(pc)} pc-loss samples")

    if cfg["note_growth_events"]:
        growth = [
            s["growth_events"] for s in samples if isinstance(s.get("growth_events"), (int, float))
        ]
        total = int(growth[-1]) if growth else 0
        # Informational gate: ok either way, the REPORT must state which case
        # occurred (closes the WS1 growth caveat).
        add(
            "growth events observed (informational)",
            True,
            f"growth_events={total}"
            + ("" if total > 0 else " - pc-loss threshold never crossed or growth idle"),
        )

    passed = all(g["ok"] is not False for g in gates)
    return {"passed": passed, "gates": gates, "config": cfg}


# -- overhead check ----------------------------------------------------------

def measure_overhead(
    base_url: str,
    agent_id: str,
    *,
    window_s: float = 60.0,
    interval_s: float = 2.0,
) -> dict[str, Any]:
    """A4: cycle-rate with sampler off vs on, two back-to-back windows."""

    def cycles_now() -> float | None:
        resp = _http_get_json(f"{base_url.rstrip('/')}/agent/{agent_id}/metrics")
        if resp is None:
            return None
        m = resp.get("metrics") or resp
        v = m.get("cycles_completed")
        return float(v) if isinstance(v, (int, float)) else None

    def window(sample: bool) -> float | None:
        c0 = cycles_now()
        t0 = time.time()
        if c0 is None:
            return None
        sampler = None
        if sample:
            sampler = HarnessSampler(
                base_url, agent_id, Path(os.devnull), interval_s=interval_s, collect_gpu=False
            )
        while time.time() - t0 < window_s:
            if sampler is not None:
                sampler.sample_once()
            time.sleep(interval_s)
        c1 = cycles_now()
        if c1 is None:
            return None
        return (c1 - c0) / (time.time() - t0)

    off = window(False)
    on = window(True)
    delta = None
    if off and on and off > 0:
        delta = round((off - on) / off, 4)
    return {"rate_off_hz": off, "rate_on_hz": on, "overhead_fraction": delta}
