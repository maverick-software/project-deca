"""Run a live training evaluation against a Decadic server.

The script is intentionally CLI-first and read-only with respect to cognition:
it creates/observes agents through public REST endpoints and writes an eval
report. It does not inject rewards, labels, or training targets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decadic.evaluation.runner import (  # noqa: E402
    build_report,
    load_eval_spec,
    report_path_for,
    write_samples_jsonl,
)
from decadic.evaluation.sampling import normalize_eval_metrics, target_end_cycle  # noqa: E402
from decadic.evaluation.types import EvalSample  # noqa: E402


class HttpClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def get(self, path: str) -> dict[str, Any]:
        return self.request("GET", path)

    def post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", path, body)


def _try_get(client: HttpClient, path: str) -> dict[str, Any] | None:
    try:
        return client.get(path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _create_agent(client: HttpClient, preset: str | None) -> str:
    path = "/agent"
    if preset:
        path += "?" + urllib.parse.urlencode({"preset": preset})
    resp = client.post(path)
    return str(resp["agent_id"])


def _start_dojo(client: HttpClient, agent_id: str, skill_id: str) -> dict[str, Any]:
    return client.post("/dojo/start", {"agent_id": agent_id, "skill_id": skill_id})


def _collect_live(
    *,
    client: HttpClient,
    agent_id: str,
    cycles: int,
    poll_interval_s: float,
    timeout_s: float,
    include_dojo: bool,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    last_cycle = -1
    start_cycle: int | None = None
    end_cycle: int | None = None
    while time.perf_counter() < deadline:
        metrics_resp = client.get(f"/agent/{agent_id}/metrics")
        metrics = dict(metrics_resp.get("metrics") or {})
        cycle = int(metrics.get("cycles_completed", 0) or 0)
        discovery = _try_get(client, f"/agent/{agent_id}/discovery")
        dojo = _try_get(client, "/dojo/status") if include_dojo else None
        metrics = normalize_eval_metrics(metrics, discovery, dojo)
        if cycle != last_cycle or not samples:
            if start_cycle is None:
                start_cycle = cycle
                end_cycle = target_end_cycle(start_cycle, cycles)
            samples.append(
                EvalSample(
                    cycle=cycle,
                    t_s=round(time.perf_counter() - t0, 6),
                    metrics=metrics,
                    discovery=discovery,
                    dojo=dojo,
                )
            )
            last_cycle = cycle
        if end_cycle is not None and cycle >= end_cycle:
            break
        time.sleep(max(0.05, poll_interval_s))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, help="scenario name under docs/eval_scenarios or JSON path")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--dojo-skill", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--probe-bank", default=None)
    ap.add_argument("--poll-interval", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument(
        "--agent-id",
        default=None,
        help="Attach to an existing agent instead of creating a new one "
        "(the eval is read-only; an observation stream must already be driving the agent)",
    )
    args = ap.parse_args()

    spec = load_eval_spec(args.scenario)
    if args.cycles is not None:
        spec.cycles = int(args.cycles)
    if args.seeds:
        spec.seeds = [int(x) for x in args.seeds]
    if args.preset is not None:
        spec.agent_preset = args.preset
    if args.dojo_skill is not None:
        spec.dojo_skill_id = args.dojo_skill
    if args.baseline is not None:
        spec.baseline = args.baseline
    if args.poll_interval is not None:
        spec.poll_interval_s = float(args.poll_interval)
    if args.timeout is not None:
        spec.timeout_s = float(args.timeout)

    client = HttpClient(args.base_url)
    agent_id = args.agent_id or _create_agent(client, spec.agent_preset)
    if spec.dojo_skill_id:
        _start_dojo(client, agent_id, spec.dojo_skill_id)

    samples = _collect_live(
        client=client,
        agent_id=agent_id,
        cycles=spec.cycles,
        poll_interval_s=spec.poll_interval_s,
        timeout_s=spec.timeout_s,
        include_dojo=bool(spec.dojo_skill_id),
    )

    out_path = report_path_for(spec.scenario, args.out_dir)
    sample_path = out_path.with_suffix(".jsonl")
    write_samples_jsonl(samples, sample_path)
    report = build_report(
        spec=spec,
        samples=samples,
        agent_id=agent_id,
        samples_path=str(sample_path),
        probe_bank=args.probe_bank,
    )
    start_cycle = samples[0].cycle if samples else 0
    observed = max(0, samples[-1].cycle - start_cycle) if samples else 0
    if samples and observed < spec.cycles:
        report.status = "fail"
        report.failures.append(
            f"target cycles not reached: observed {observed}/{spec.cycles}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"{report.status.upper()} {spec.scenario} agent={agent_id} report={out_path}")
    for failure in report.failures:
        print(f"  - {failure}")
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
