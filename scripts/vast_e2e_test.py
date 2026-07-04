"""End-to-end test of the Vast.ai deploy pipeline: rent -> deploy -> verify -> destroy.

SPENDS REAL MONEY (a rented GPU, typically well under $0.50 for a full run at
default settings). Safety properties:
  - Hard wall-clock cap (--max-minutes, default 40): the script self-terminates.
  - The rented instance is destroyed in a finally block on EVERY exit path
    (success, failure, Ctrl+C), with retries; destroy is verified before exit.
  - A cost cap (--max-dph) bounds the hourly rate of the offer it will pick.
  - Interactive confirmation before renting unless --yes is passed.

What it verifies (each is a named PASS/FAIL check):
  1. preflight     - server up, API key stored, vastai CLI available, no active deployment
  2. offer_search  - the tiered offer search returns rentable offers under the cap
  3. deploy_ready  - full provisioning pipeline reaches phase "ready" in time
  4. remote_alive  - the proxied remote agent answers /agents and /agent/{id}/metrics
  5. model_runs    - cycles_completed advances on the remote agent (the model is
                     actually thinking on the rented GPU), and neural_pc_loss is finite
  6. teardown      - destroy returns the deployment to idle (instance gone = billing stopped)

Artifacts land in reports/vast_e2e_<stamp>/ (deployment log, metrics samples, verdict).

Usage:
    .venv\\Scripts\\python.exe scripts\\vast_e2e_test.py --yes
    .venv\\Scripts\\python.exe scripts\\vast_e2e_test.py --max-dph 0.30 --observe-minutes 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CHECKS: list[tuple[str, bool, str]] = []


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def check(name: str, passed: bool, detail: str) -> bool:
    CHECKS.append((name, passed, detail))
    log(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def _req(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    return json.loads(text) if text.strip() else {}


def get(base: str, path: str, timeout: float = 30.0) -> dict:
    return _req("GET", f"{base}{path}", timeout=timeout)


def post(base: str, path: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    return _req("POST", f"{base}{path}", body=body, timeout=timeout)


def destroy_and_verify(base: str, attempts: int = 5) -> bool:
    """Destroy the active deployment and confirm it is gone. Retries: this is
    the step that stops billing, so it must not give up on a transient error."""
    for i in range(1, attempts + 1):
        try:
            post(base, "/vast/deployment/destroy", timeout=180.0)
        except Exception as exc:  # noqa: BLE001
            log(f"destroy attempt {i}/{attempts} errored: {exc}")
        time.sleep(3.0)
        try:
            d = get(base, "/vast/deployment")
            if d.get("instance_id") is None and d.get("phase") in ("idle", "error", "stopped"):
                return True
        except Exception as exc:  # noqa: BLE001
            log(f"destroy verify {i}/{attempts} errored: {exc}")
        time.sleep(5.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--gpu-name", default="RTX_4090", help="Model filter; empty = any GPU")
    ap.add_argument("--min-gpu-ram", type=float, default=16.0)
    ap.add_argument("--max-dph", type=float, default=0.45, help="Max $/hr offer the test may rent")
    ap.add_argument("--preset", default="tiny", help="Brain preset (tiny = fastest/cheapest)")
    ap.add_argument("--encoder", default="zeros", help="zeros = no 1GB encoder download (faster)")
    ap.add_argument("--scene", default="forage", help="Built-in preset id; the remote body drives cycles")
    ap.add_argument("--disk", type=int, default=25)
    ap.add_argument("--deploy-timeout-minutes", type=float, default=25.0)
    ap.add_argument("--observe-minutes", type=float, default=4.0, help="How long to watch the remote model run")
    ap.add_argument("--min-cycle-advance", type=int, default=50, help="Cycles that must elapse while observing")
    ap.add_argument("--max-minutes", type=float, default=40.0, help="Hard wall-clock cap for the whole test")
    ap.add_argument("--yes", action="store_true", help="Skip the interactive rent confirmation")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    started = time.monotonic()
    deadline = started + args.max_minutes * 60.0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / f"vast_e2e_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"artifacts: {out_dir}")
    log(f"hard cap: {args.max_minutes:.0f} minutes; instance auto-destroyed on every exit path")

    rented = False
    metrics_samples: list[dict] = []
    try:
        # ---- 1. preflight ---------------------------------------------------
        try:
            settings = get(base, "/vast/settings")
        except Exception as exc:  # noqa: BLE001
            check("preflight", False, f"server not reachable at {base}: {exc}")
            return 1
        dep = get(base, "/vast/deployment")
        ok = (
            settings.get("has_api_key") is True
            and settings.get("cli_available") is True
            and dep.get("instance_id") is None
            and not dep.get("busy")
        )
        if not check(
            "preflight",
            ok,
            f"key={settings.get('has_api_key')} cli={settings.get('cli_available')} "
            f"phase={dep.get('phase')} instance={dep.get('instance_id')}",
        ):
            return 1

        # ---- 2. offer search -------------------------------------------------
        q = urllib.parse.urlencode(
            {
                "gpu_name": args.gpu_name,
                "num_gpus": 1,
                "min_gpu_ram": args.min_gpu_ram,
                "verified": "true",
                "limit": 50,
            }
        )
        res = get(base, f"/vast/offers?{q}", timeout=120.0)
        offers = [
            o
            for o in res.get("offers", [])
            if o.get("dph_total") is not None and float(o["dph_total"]) <= args.max_dph
        ]
        offers.sort(key=lambda o: float(o["dph_total"]))
        if not check(
            "offer_search",
            len(offers) > 0,
            f"{len(offers)} rentable offers <= ${args.max_dph}/hr "
            f"(of {len(res.get('offers', []))} returned)",
        ):
            return 1
        offer = offers[0]
        est_hourly = float(offer["dph_total"])
        est_total = est_hourly * (args.max_minutes / 60.0)
        log(
            f"selected offer {offer['id']}: {offer.get('num_gpus')}x {offer.get('gpu_name')} "
            f"{offer.get('gpu_ram_gb')}GB @ ${est_hourly:.3f}/hr "
            f"(worst-case spend this run: ~${est_total:.2f} + small disk/bandwidth fees)"
        )
        (out_dir / "selected_offer.json").write_text(json.dumps(offer, indent=2), encoding="utf-8")

        if not args.yes:
            answer = input(f"Rent offer {offer['id']} at ${est_hourly:.3f}/hr? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                log("aborted before renting; nothing was charged")
                return 1

        # ---- 3. deploy -------------------------------------------------------
        post(
            base,
            "/vast/deploy",
            {
                "offer_id": offer["id"],
                "preset": args.preset,
                "encoder": args.encoder,
                "scene": args.scene,
                "disk": args.disk,
            },
        )
        rented = True
        log("deploy started; polling phases (billing has begun)")

        phase_deadline = min(time.monotonic() + args.deploy_timeout_minutes * 60.0, deadline)
        last_phase = None
        last_log_len = 0
        ready = False
        while time.monotonic() < phase_deadline:
            d = get(base, "/vast/deployment")
            phase = d.get("phase")
            for line in d.get("log", [])[last_log_len:]:
                log(f"  remote> {line}")
            last_log_len = len(d.get("log", []))
            if phase != last_phase:
                log(f"phase: {phase} (elapsed {d.get('elapsed_s', 0):.0f}s, est ${d.get('est_cost_usd') or 0:.3f})")
                last_phase = phase
            if phase == "ready":
                ready = True
                break
            if phase == "error":
                check("deploy_ready", False, f"deploy errored: {d.get('error')}")
                (out_dir / "deployment_error.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
                return 1
            time.sleep(10.0)
        if not check(
            "deploy_ready",
            ready,
            f"phase=ready in {(time.monotonic() - started) / 60.0:.1f} min"
            if ready
            else f"not ready within {args.deploy_timeout_minutes:.0f} min (last phase: {last_phase})",
        ):
            return 1
        (out_dir / "deployment_ready.json").write_text(
            json.dumps(get(base, "/vast/deployment"), indent=2), encoding="utf-8"
        )

        # ---- 4. remote agent alive --------------------------------------------
        d = get(base, "/vast/deployment")
        agent_id = d.get("agent_id")
        if not agent_id:
            agents = get(base, "/agents")
            ids = agents.get("agents") or agents.get("agent_ids") or []
            agent_id = (ids[0].get("agent_id") if ids and isinstance(ids[0], dict) else ids[0]) if ids else None
        try:
            m0 = get(base, f"/agent/{agent_id}/metrics", timeout=60.0)
            alive = isinstance(m0.get("metrics"), dict)
        except Exception as exc:  # noqa: BLE001
            m0, alive = {}, False
            log(f"metrics fetch failed: {exc}")
        if not check("remote_alive", alive and agent_id is not None, f"agent {agent_id} answers through the tunnel proxy"):
            return 1

        # ---- 5. the model actually runs ---------------------------------------
        c0 = int(m0["metrics"].get("cycles_completed", 0) or 0)
        observe_until = min(time.monotonic() + args.observe_minutes * 60.0, deadline - 120.0)
        log(f"observing remote model for up to {args.observe_minutes:.0f} min (start cycles={c0})")
        c_now, pc_loss = c0, None
        while time.monotonic() < observe_until:
            time.sleep(15.0)
            try:
                m = get(base, f"/agent/{agent_id}/metrics", timeout=60.0)["metrics"]
            except Exception as exc:  # noqa: BLE001
                log(f"metrics poll error (continuing): {exc}")
                continue
            c_now = int(m.get("cycles_completed", 0) or 0)
            pc_loss = m.get("neural_pc_loss")
            metrics_samples.append({"t": time.time(), "cycles": c_now, "neural_pc_loss": pc_loss})
            log(f"  cycles={c_now} (+{c_now - c0}) pc_loss={pc_loss}")
            if c_now - c0 >= args.min_cycle_advance:
                break
        advanced = c_now - c0
        loss_ok = pc_loss is None or (isinstance(pc_loss, (int, float)) and math.isfinite(float(pc_loss)))
        check(
            "model_runs",
            advanced >= args.min_cycle_advance and loss_ok,
            f"{advanced} cycles advanced on the rented GPU (needed {args.min_cycle_advance}); "
            f"pc_loss={pc_loss} ({'finite/absent' if loss_ok else 'NOT finite'})",
        )
        (out_dir / "metrics_samples.jsonl").write_text(
            "\n".join(json.dumps(s) for s in metrics_samples), encoding="utf-8"
        )

    except KeyboardInterrupt:
        log("interrupted - proceeding to teardown")
    except Exception as exc:  # noqa: BLE001
        log(f"UNEXPECTED ERROR: {exc} - proceeding to teardown")
        check("unexpected_error", False, str(exc))
    finally:
        # ---- 6. teardown: ALWAYS runs; this is what stops billing --------------
        if rented:
            log("destroying instance (stops billing)...")
            gone = destroy_and_verify(base)
            check("teardown", gone, "instance destroyed and deployment idle" if gone else "COULD NOT CONFIRM DESTROY - CHECK cloud.vast.ai/instances NOW")
        summary = {
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in CHECKS],
            "all_passed": all(p for _, p, _ in CHECKS),
            "wall_minutes": round((time.monotonic() - started) / 60.0, 1),
        }
        (out_dir / "verdict.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log("---- summary ----")
        for n, p, detail in CHECKS:
            log(f"[{'PASS' if p else 'FAIL'}] {n}: {detail}")
        log(f"VAST_E2E: {'PASS' if summary['all_passed'] else 'FAIL'} ({summary['wall_minutes']} min)")

    return 0 if all(p for _, p, _ in CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
