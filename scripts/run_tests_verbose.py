"""Verbose, hang-diagnosable pytest runner.

Pytest's normal progress UI is too opaque for this project: when a long-running
test freezes, the operator sees a percentage but not the active test. This runner
collects node ids, then runs each test in its own pytest process while printing
the exact test before it starts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _test_tmp_root() -> Path:
    override = os.environ.get("DECADIC_TEST_TMP_ROOT")
    root = Path(override) if override else Path("C:/tmp/decadic_pytest")
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return root
    except OSError:
        fallback = ROOT / ".pytest_tmp_fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _env() -> dict[str, str]:
    env = dict(os.environ)
    tmp = str(_test_tmp_root())
    env["TMP"] = tmp
    env["TEMP"] = tmp
    env["TMPDIR"] = tmp
    return env


def _python() -> str:
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv_py if venv_py.is_file() else Path(sys.executable))


def collect_tests(pytest_args: list[str]) -> list[str]:
    cmd = [_python(), "-m", "pytest", "--collect-only", "-q", *pytest_args]
    print("=== COLLECT ===", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    tests = [
        line.strip()
        for line in proc.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]
    print(f"=== COLLECTED {len(tests)} TESTS ===", flush=True)
    return tests


def run_one(nodeid: str, index: int, total: int, timeout_s: int, passthrough: list[str]) -> bool:
    started = time.perf_counter()
    print(f"\n=== RUN {index}/{total}: {nodeid} ===", flush=True)
    cmd = [
        _python(),
        "-m",
        "pytest",
        nodeid,
        "-vv",
        "-s",
        "--tb=short",
        "-ra",
        *passthrough,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        print(exc.stdout or "", end="")
        print(f"\n=== TIMEOUT after {elapsed:.1f}s: {nodeid} ===", flush=True)
        return False
    elapsed = time.perf_counter() - started
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode == 0:
        print(f"=== PASS {index}/{total} in {elapsed:.1f}s: {nodeid} ===", flush=True)
        return True
    print(f"=== FAIL {index}/{total} exit={proc.returncode} in {elapsed:.1f}s: {nodeid} ===", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with per-test progress and timeout.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-test timeout in seconds.")
    parser.add_argument("--start-at", default="", help="Resume at a nodeid containing this text.")
    parser.add_argument("pytest_args", nargs="*", help="Optional pytest selection args, e.g. tests/test_app.py")
    args = parser.parse_args()

    tests = collect_tests(args.pytest_args)
    if args.start_at:
        for i, nodeid in enumerate(tests):
            if args.start_at in nodeid:
                tests = tests[i:]
                break
        else:
            print(f"start marker not found: {args.start_at}", file=sys.stderr)
            return 2
    total = len(tests)
    if total == 0:
        print("No tests collected.")
        return 5
    passed = 0
    suite_started = time.perf_counter()
    for index, nodeid in enumerate(tests, 1):
        if not run_one(nodeid, index, total, args.timeout, []):
            print(f"\n=== STOPPED: {passed} passed before failure/timeout ===", flush=True)
            return 1
        passed += 1
    elapsed = time.perf_counter() - suite_started
    print(f"\n=== ALL {passed} TESTS PASSED in {elapsed:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
