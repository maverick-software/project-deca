"""WS2 soak runner: full-lifecycle long-run harness (PRD section 5.2).

Owns server + agent + observation client + sampler for --hours N, with a
stall watchdog (auto /debug/tasks capture), disk guard, optional hourly
checkpoints, and end-of-run rollups + gate evaluation + report generation.

Usage (from repo root, inside the venv):

    python scripts/soak_run.py --hours 1                # shakedown
    python scripts/soak_run.py --hours 12               # the real soak
    python scripts/soak_run.py --hours 1 --consolidation-off
    python scripts/soak_run.py --overhead-check         # A4 measurement
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decadic.metrics.harness import (  # noqa: E402
    HarnessSampler,
    evaluate_soak_gates,
    load_samples,
    rollup,
    write_rollup_csv,
)

SOAK_ENV = {
    "DECADIC_SELF_HOST": "127.0.0.1",
    "DECADIC_NEURAL_PRESET": "full",
    "DECADIC_PLASTICITY_ENABLED": "1",
    "DECADIC_SPARSE_ENABLED": "1",
    "DECADIC_GROWTH_ENABLED": "1",
    "DECADIC_ENCODER_MODE": "hf",
    "DECADIC_DEVICE": "cuda",
    "DECADIC_EPISODIC_ASYNC": "1",
    "DECADIC_N_ACTUATORS": "21",
    "DECADIC_CURRICULUM_MODE": "legacy",
    "DECADIC_SELF_MODEL_FEEDBACK": "1",
    "DECADIC_GWT_ENABLED": "1",
    "DECADIC_INTEGRATION_WINDOW_MS": "200",
    "DECADIC_PREDICTIVE_AFFECT": "1",
    "DECADIC_REPRESENTED_SELF": "1",
    "DECADIC_MEMORY_EFFICIENT_TRAINING": "1",
    # WS1: block policy deadlocks intake; drop_oldest + tombstone fix is safe.
    "DECADIC_PREFETCH_OVERLOAD_POLICY": "drop_oldest",
}


def _http(method: str, url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None


def _py() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv if venv.is_file() else sys.executable)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class SoakRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT / "reports" / f"soak_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.server: subprocess.Popen | None = None
        self.client: subprocess.Popen | None = None
        self.agent_id: str | None = None
        self.stall_events = 0
        self.stop_flag = False
        self.result = "unknown"

    # -- lifecycle ---------------------------------------------------------
    def start_server(self) -> None:
        env = dict(os.environ)
        env.update(SOAK_ENV)
        env["DECADIC_SELF_PORT"] = str(self.args.port)
        env["DECADIC_LOG_DIR"] = str(self.run_dir)
        if self.args.consolidation_off:
            env["DECADIC_CONSOLIDATION_ENABLED"] = "0"
        manifest = {
            "started": datetime.now().isoformat(),
            "git_sha": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
            ).stdout.strip(),
            "hours": self.args.hours,
            "obs_rate_s": self.args.rate,
            "sample_interval_s": self.args.interval,
            "stall_policy": self.args.stall_policy,
            "env": {k: v for k, v in env.items() if k.startswith("DECADIC_")},
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        _log(f"starting server on port {self.args.port} (run dir: {self.run_dir.name})")
        self.server = subprocess.Popen(
            [_py(), "-m", "uvicorn", "decadic.api.app:app", "--host", "127.0.0.1",
             "--port", str(self.args.port)],
            cwd=ROOT,
            env=env,
            stdout=(self.run_dir / "server.out.log").open("w"),
            stderr=(self.run_dir / "server.err.log").open("w"),
        )
        deadline = time.time() + 240
        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("server exited during startup - see server.err.log")
            if _http("GET", f"{self.base_url}/agents", timeout=3) is not None:
                _log("server ready")
                return
            time.sleep(2)
        raise RuntimeError("server not ready after 240s")

    def start_agent_and_client(self) -> None:
        # Full-preset + hf-encoder bundle construction can exceed the default
        # 10 s _http timeout (observed 2026-07-04: creation aborted at the
        # edge). Generous bound -- the readiness loop above already proved
        # the server is alive.
        resp = _http("POST", f"{self.base_url}/agent", timeout=180)
        if not resp or "agent_id" not in resp:
            raise RuntimeError("agent creation failed")
        self.agent_id = str(resp["agent_id"])
        _log(f"agent: {self.agent_id}")
        self.client = subprocess.Popen(
            [_py(), "scripts/synthetic_ws_client.py", "--port", str(self.args.port),
             "--agent-id", self.agent_id, "--steps", "100000000",
             "--rate", str(self.args.rate), "--log-every", "5000"],
            cwd=ROOT,
            stdout=(self.run_dir / "client.out.log").open("w"),
            stderr=(self.run_dir / "client.err.log").open("w"),
        )

    def teardown(self) -> None:
        for proc, name in ((self.client, "client"), (self.server, "server")):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _log(f"{name} stopped")

    # -- watchdog + guards ---------------------------------------------------
    def cycles_now(self) -> float | None:
        resp = _http("GET", f"{self.base_url}/agent/{self.agent_id}/metrics", timeout=5)
        if resp is None:
            return None
        m = resp.get("metrics") or resp
        v = m.get("cycles_completed")
        return float(v) if isinstance(v, (int, float)) else None

    def capture_stall(self, cycle: float | None) -> None:
        self.stall_events += 1
        tag = f"stall_{self.stall_events:02d}"
        _log(f"STALL detected at cycle {cycle} - capturing {tag}")
        tasks = _http("GET", f"{self.base_url}/debug/tasks", timeout=30)
        if tasks is not None:
            (self.run_dir / f"{tag}_tasks.json").write_text(
                json.dumps(tasks, indent=1), encoding="utf-8"
            )
        metrics = _http("GET", f"{self.base_url}/agent/{self.agent_id}/metrics", timeout=30)
        if metrics is not None:
            (self.run_dir / f"{tag}_metrics.json").write_text(
                json.dumps(metrics, indent=1), encoding="utf-8"
            )

    def checkpoint(self) -> None:
        t0 = time.time()
        resp = _http("POST", f"{self.base_url}/agent/{self.agent_id}/checkpoint", timeout=120)
        _log(f"checkpoint {'ok' if resp is not None else 'FAILED'} ({time.time() - t0:.1f}s)")

    def main_loop(self) -> None:
        end = time.time() + self.args.hours * 3600
        last_cycle: float | None = None
        last_change = time.time()
        last_checkpoint = time.time()
        last_status = 0.0
        while time.time() < end:
            time.sleep(5)
            cycle = self.cycles_now()
            now = time.time()
            if cycle is not None and cycle != last_cycle:
                last_cycle = cycle
                last_change = now
            elif cycle is not None and now - last_change >= self.args.stall_seconds:
                self.capture_stall(cycle)
                if self.args.stall_policy == "abort":
                    self.result = f"aborted_on_stall_at_cycle_{int(cycle)}"
                    return
                last_change = now  # record-and-continue: rearm
            free_gb = shutil.disk_usage(self.run_dir).free / 1e9
            if free_gb < self.args.min_free_gb:
                self.result = f"aborted_low_disk_{free_gb:.1f}GB"
                return
            if self.args.checkpoint_hours > 0 and now - last_checkpoint >= self.args.checkpoint_hours * 3600:
                self.checkpoint()
                last_checkpoint = now
            if now - last_status >= 60:
                remaining = (end - now) / 3600
                _log(f"cycle {int(cycle) if cycle else '?'} | {remaining:.2f}h remaining | disk free {free_gb:.0f}GB")
                last_status = now
        self.result = "completed"

    # -- post-run -------------------------------------------------------------
    def finalize(self) -> None:
        samples_path = self.run_dir / "harness_samples.jsonl"
        summary: dict[str, Any] = {
            "result": self.result,
            "stall_events": self.stall_events,
            "finished": datetime.now().isoformat(),
        }
        if samples_path.exists():
            samples = load_samples(samples_path)
            summary["n_samples"] = len(samples)
            write_rollup_csv(rollup(samples, bucket_seconds=60), self.run_dir / "rollup_1m.csv")
            if self.args.hours >= 2:
                write_rollup_csv(
                    rollup(samples, bucket_seconds=3600), self.run_dir / "rollup_1h.csv"
                )
            verdict = evaluate_soak_gates(samples, stall_events=self.stall_events)
            summary["gates"] = verdict
            (self.run_dir / "gates.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
        (self.run_dir / "soak_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        _log(f"summary written: result={self.result} stalls={self.stall_events}")
        report = subprocess.run(
            [_py(), "scripts/generate_run_report.py", str(self.run_dir)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        _log("report: " + (report.stdout.strip().splitlines()[-1] if report.stdout.strip() else report.stderr.strip()[-200:]))

    def run(self) -> int:
        try:
            self.start_server()
            self.start_agent_and_client()
            sampler = HarnessSampler(
                self.base_url,
                self.agent_id or "",
                self.run_dir / "harness_samples.jsonl",
                interval_s=self.args.interval,
                run_dir=self.run_dir,
            )
            t = threading.Thread(
                target=sampler.run, kwargs={"stop": lambda: self.stop_flag}, daemon=True
            )
            t.start()
            self.main_loop()
        except KeyboardInterrupt:
            self.result = "interrupted"
        except Exception as e:  # any lifecycle failure must still clean up
            self.result = f"error: {e}"
        finally:
            self.stop_flag = True
            time.sleep(max(1.0, self.args.interval))
            if self.agent_id:
                _http("DELETE", f"{self.base_url}/agent/{self.agent_id}", timeout=10)
            self.teardown()
            self.finalize()
        ok = self.result == "completed" and self.stall_events == 0
        _log(f"RESULT: {self.result} (artifacts in {self.run_dir})")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rate", type=float, default=0.1, help="seconds between observations")
    ap.add_argument("--interval", type=float, default=2.0, help="sampler poll interval seconds")
    ap.add_argument("--stall-seconds", type=int, default=60)
    ap.add_argument("--stall-policy", choices=["abort", "record-and-continue"], default="abort")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--checkpoint-hours", type=float, default=1.0, help="0 disables checkpoints")
    ap.add_argument("--consolidation-off", action="store_true")
    ap.add_argument("--overhead-check", action="store_true", help="run A4 measurement against a live agent and exit")
    ap.add_argument("--agent-id", default=None, help="agent id for --overhead-check")
    args = ap.parse_args()

    if args.overhead_check:
        from decadic.metrics.harness import measure_overhead

        if not args.agent_id:
            print("--overhead-check requires --agent-id of a running, driven agent")
            return 2
        result = measure_overhead(f"http://127.0.0.1:{args.port}", args.agent_id)
        print(json.dumps(result, indent=1))
        return 0

    return SoakRun(args).run()


if __name__ == "__main__":
    sys.exit(main())
