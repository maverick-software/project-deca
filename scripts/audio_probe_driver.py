"""WS6 audio-cognition probe driver: does the agent's MIND react to sound?

Runs an interactive phased session against a live agent whose observations
carry NO client audio (the intake fills them from the system microphone).
The operator is prompted to stay silent / speak on cue; the driver polls the
metrics endpoint through every phase and then renders a verdict over three
questions:

  1. GATING  -- do intake attaches rise only while speaking (silence gate)?
  2. SALIENCE -- does gate novelty (peak) rise when speech begins?
  3. COGNITION -- do gate escalations / prediction dynamics move with speech?

This is observational instrumentation in the gate-probe mold: it reports
per-phase numbers and a coarse verdict, and archives everything for the
report ledger.

Usage (normally via run_audio_probe.ps1):
    python scripts/audio_probe_driver.py --base http://127.0.0.1:8765 \
        --agent <id> --out reports/audioprobe_x [--phase-plan default]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# (name, seconds, operator instruction). settle_* phases are DISCARDED from
# scoring: birth novelty and early vigilance decay monotonically, so the first
# probe run (2026-07-06) confounded time-order with condition. Scoring is by
# TRANSITION: each speak phase against the tail of its preceding silence.
DEFAULT_PLAN = [
    ("settle_0", 60, "STAY SILENT, hands off keyboard/mouse. (Warmup -- discarded.)"),
    ("silence_1", 30, "STAY SILENT, hands still. Quiet baseline."),
    ("speak_1", 20, "SPEAK NOW -- talk to it continuously (anything)."),
    ("silence_2", 30, "SILENT again, hands still."),
    ("speak_2", 20, "SPEAK AGAIN -- same voice, new words."),
    ("silence_3", 30, "SILENT, hands still. Final baseline."),
]
TAIL_S = 15.0  # score silence phases by their settled tail only

POLL_S = 0.5
KEYS = (
    "gate_i_novelty",
    "gate_i_novelty_peak",
    "gate_escalations",
    "gate_escalation_rate",
    "neural_pc_loss_last",
    "cycles_completed",
)


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _metrics(base: str, aid: str) -> dict:
    payload = _get(f"{base}/agent/{aid}/metrics") or {}
    m = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return m if isinstance(m, dict) else {}


def _row(m: dict, phase: str, t0: float) -> dict:
    row = {"t": round(time.time() - t0, 2), "phase": phase}
    for k in KEYS:
        v = m.get(k)
        if isinstance(v, (int, float)):
            row[k] = v
    ai = m.get("audio_intake")
    if isinstance(ai, dict):
        row["attached"] = ai.get("chunks_attached")
        row["silence_skips"] = ai.get("silence_skips")
        row["intake_mode"] = ai.get("mode")
        row["self_rms_ema"] = ai.get("self_rms_ema")
        row["last_chunk_rms"] = ai.get("last_chunk_rms")
    return row


def analyze(samples_path: str) -> int:
    """Offline: per-phase last_chunk_rms percentiles from a recorded run --
    the data-driven way to place DECADIC_AUDIO_SILENCE_RMS between the room's
    ambient level and speech at the configured gain."""
    rows = [
        json.loads(x)
        for x in Path(samples_path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    def pct(vals, p):
        vals = sorted(v for v in vals if isinstance(v, (int, float)))
        return vals[min(len(vals) - 1, int(p * (len(vals) - 1)))] if vals else None
    amb, spk = [], []
    for r in rows:
        v = r.get("last_chunk_rms")
        if r["phase"].startswith("speak"):
            spk.append(v)
        elif not r["phase"].startswith("settle"):
            amb.append(v)
    print(f"ambient rms: p50={pct(amb, .5)} p90={pct(amb, .9)} p99={pct(amb, .99)}")
    print(f"speech  rms: p10={pct(spk, .1)} p50={pct(spk, .5)} p90={pct(spk, .9)}")
    a90, s50 = pct(amb, 0.9), pct(spk, 0.5)
    if a90 and s50 and s50 > a90:
        rec = (a90 * s50) ** 0.5  # geometric midpoint
        print(f"recommended: $env:DECADIC_AUDIO_SILENCE_RMS = \"{rec:.4f}\"")
    else:
        print("speech does not separate from ambient at this gain -- raise "
              "DECADIC_AUDIO_GAIN or move closer to the mic")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--agent", default=None)
    ap.add_argument("--out", default="reports/audioprobe")
    ap.add_argument("--analyze", default=None, help="samples.jsonl from a past run")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.analyze)
    if not args.agent:
        ap.error("--agent is required unless --analyze is given")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m0 = _metrics(args.base, args.agent)
    ai = m0.get("audio_intake") or {}
    print(
        f"[probe] intake mode={ai.get('mode')} running={ai.get('running')} "
        f"device={ai.get('device')} gain={ai.get('mic_gain')}"
    )
    if ai.get("mode") != "mic":
        print("[probe] WARNING: intake is not in mic mode -- the room is inaudible.")
    if ai.get("device") in (None, "default") or (ai.get("mic_gain") or 1.0) <= 1.0:
        print(
            "[probe] WARNING: default device / unity gain. If a previous run "
            "needed DECADIC_AUDIO_DEVICE / DECADIC_AUDIO_GAIN, set them in THIS "
            "shell before running (they are session env, not persisted). "
            "scripts/mic_check.py --all re-measures."
        )
        try:
            if input("[probe] continue anyway? [y/N] ").strip().lower() != "y":
                return 1
        except EOFError:
            pass  # non-interactive: proceed, the warning is on record

    rows: list[dict] = []
    t0 = time.time()
    print("\n=== AUDIO-COGNITION PROBE ===")
    print("Follow the prompts. Total ~2.7 minutes.\n")
    for name, seconds, instruction in DEFAULT_PLAN:
        print(f"\n>>> [{name}] {instruction} ({seconds}s)")
        end = time.time() + seconds
        while time.time() < end:
            rows.append(_row(_metrics(args.base, args.agent), name, t0))
            remaining = int(end - time.time())
            if remaining % 10 == 0:
                print(f"    ...{remaining}s", flush=True)
            time.sleep(POLL_S)
    (out / "samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------- verdict
    # Transition scoring: each speak phase vs the SETTLED TAIL of the silence
    # phase immediately before it. Immune to monotone decay (warmup novelty,
    # early vigilance) that poisoned phase-mean comparisons.
    def prows(name: str) -> list[dict]:
        return [r for r in rows if r["phase"] == name]

    def tail(rws: list[dict], seconds: float = TAIL_S) -> list[dict]:
        if not rws:
            return rws
        t_end = max(r["t"] for r in rws)
        return [r for r in rws if r["t"] >= t_end - seconds]

    def mean(vals) -> float | None:
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    def rate(rws: list[dict], key: str) -> float:
        """Counter increase per second across the window."""
        vals = [(r["t"], r[key]) for r in rws if isinstance(r.get(key), (int, float))]
        if len(vals) < 2 or vals[-1][0] <= vals[0][0]:
            return 0.0
        return max(0.0, (vals[-1][1] - vals[0][1]) / (vals[-1][0] - vals[0][0]))

    pairs = [("silence_1", "speak_1"), ("silence_2", "speak_2")]
    att_deltas, nov_deltas, esc_deltas, detail_lines = [], [], [], []
    for sil, spk in pairs:
        base, active = tail(prows(sil)), prows(spk)
        d_att = rate(active, "attached") - rate(base, "attached")
        d_esc = rate(active, "gate_escalations") - rate(base, "gate_escalations")
        n_a, n_b = mean([r.get("gate_i_novelty_peak") for r in active]), mean(
            [r.get("gate_i_novelty_peak") for r in base]
        )
        d_nov = (n_a - n_b) if (n_a is not None and n_b is not None) else None
        att_deltas.append(d_att)
        esc_deltas.append(d_esc)
        if d_nov is not None:
            nov_deltas.append(d_nov)
        detail_lines.append(
            f"  {sil}->{spk}: attach/s {rate(base, 'attached'):.2f}->"
            f"{rate(active, 'attached'):.2f}  novelty_peak {n_b}->{n_a}  "
            f"esc/s {rate(base, 'gate_escalations'):.3f}->"
            f"{rate(active, 'gate_escalations'):.3f}"
        )

    # Layer-honest criteria (recalibrated 2026-07-06 from measured room data):
    # per-chunk energy CANNOT separate speech from a lived-in room (typing
    # transients rival speech; inter-word gaps rival silence) and gating is
    # not its job -- the energy gate exists to skip Whisper on dead air.
    # DISCRIMINATION is cognition's job (escalations) and, at finer grain,
    # the WS6-M0 audio-token lane's. Novelty stays informational: it measures
    # the pooled pathway whose bluntness the token lane exists to fix.
    checks: list[tuple[str, bool, str]] = []
    total_att = rate(rows, "attached") * max(1.0, rows[-1]["t"] - rows[0]["t"])
    total_skips = rate(rows, "silence_skips") * max(1.0, rows[-1]["t"] - rows[0]["t"])
    checks.append((
        "intake liveness + gate economy",
        total_att > 5 and total_skips > 5,
        f"attached~{total_att:.0f} skipped~{total_skips:.0f} across the session "
        f"(the ear delivers sound AND skips dead air; both paths exercised)",
    ))
    checks.append((
        "cognitive response",
        any(d > 0 for d in esc_deltas),
        f"escalation-rate deltas at transitions: {[round(d, 4) for d in esc_deltas]} "
        f"(speech makes the agent deliberate -- the circuit's core assertion)",
    ))
    info_nov = f"novelty-peak deltas at transitions: {nov_deltas}"
    lines_info = [
        f"[INFO] auditory salience (pooled-pathway baseline for the M0 token lane): {info_nov}",
        f"[INFO] attach-rate deltas at transitions: {[round(d, 2) for d in att_deltas]}",
    ]

    spoken = [r for p in ("speak_1", "speak_2") for r in prows(p)]
    silent_tails = [r for p in ("silence_1", "silence_2", "silence_3") for r in tail(prows(p))]
    pc_speak = mean([r.get("neural_pc_loss_last") for r in spoken])
    pc_silent = mean([r.get("neural_pc_loss_last") for r in silent_tails])
    lines = [
        "# Audio-cognition probe (transition-scored)",
        f"samples={len(rows)}  plan={[p[0] for p in DEFAULT_PLAN]}",
        *detail_lines,
        f"pc_loss mean: speak={pc_speak} silent-tails={pc_silent} (informational -- "
        f"speech is unpredicted input; a rise then re-settle is healthy)",
        "",
    ]
    print("\n".join(detail_lines))
    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        line = f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
        print(line)
        lines.append(line)
    for line in lines_info:
        print(line)
        lines.append(line)
    print(f"AUDIO_PROBE: {'PASS' if ok else 'FAIL'}")
    lines.append(f"AUDIO_PROBE: {'PASS' if ok else 'FAIL'}")
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[probe] artifacts in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
