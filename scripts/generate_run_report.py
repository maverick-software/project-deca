"""WS2 report generator: run dir -> report.md + plots (PRD section 5.3).

Sections map 1:1 to the five PoC success criteria. Verdict block comes from
gates.json (written by soak_run.py). Plots require matplotlib (dev extra);
without it the report is still generated, text-only.

Usage:
    python scripts/generate_run_report.py reports/soak_<stamp>
    python scripts/generate_run_report.py --compare reports/soak_A reports/soak_B
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decadic.metrics.harness import load_samples, rollup  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# -- data access ---------------------------------------------------------------

class RunData:
    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        self.name = self.dir.name
        self.manifest = self._json("manifest.json")
        self.summary = self._json("soak_summary.json")
        self.gates = self._json("gates.json")
        samples_path = self.dir / "harness_samples.jsonl"
        self.samples = load_samples(samples_path) if samples_path.exists() else []
        self.minutes = rollup(self.samples, bucket_seconds=60)

    def _json(self, name: str) -> dict[str, Any]:
        p = self.dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                return {}
        return {}

    def series(self, key: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for s in self.samples:
            v = s.get(key)
            c = s.get("cycles_completed")
            if isinstance(v, (int, float)) and isinstance(c, (int, float)):
                xs.append(float(c))
                ys.append(float(v))
        return xs, ys

    def last(self, key: str) -> Any:
        for s in reversed(self.samples):
            if key in s:
                return s[key]
        return None

    def first(self, key: str) -> Any:
        for s in self.samples:
            if key in s:
                return s[key]
        return None


# -- helpers -------------------------------------------------------------------

def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _trend_line(run: RunData, key: str) -> str:
    xs, ys = run.series(key)
    if len(ys) < 4:
        return f"- `{key}`: insufficient samples"
    half = len(ys) // 2
    fh, sh = statistics.fmean(ys[:half]), statistics.fmean(ys[half:])
    return (
        f"- `{key}`: {ys[0]:.4f} -> {ys[-1]:.4f} (half-means {fh:.4f} -> {sh:.4f}, "
        f"{'down' if sh < fh else 'up'})"
    )


def _plot(run: RunData, keys: list[str], title: str, fname: str, *, logy: bool = False) -> str | None:
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6))
    plotted = False
    for key in keys:
        xs, ys = run.series(key)
        if len(ys) >= 4:
            ax.plot(xs, ys, lw=0.9, alpha=0.85, label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("cycle")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = run.dir / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out.name


def _plot_compare(a: RunData, b: RunData, key: str, title: str, out_dir: Path, fname: str) -> str | None:
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.6))
    plotted = False
    for run, style in ((a, "-"), (b, "--")):
        xs, ys = run.series(key)
        if len(ys) >= 4:
            ax.plot(xs, ys, style, lw=1.0, alpha=0.85, label=f"{run.name}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("cycle")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out.name


def _img(name: str | None) -> str:
    return f"\n![]({name})\n" if name else "\n*(plot unavailable - matplotlib not installed or too few samples)*\n"


# -- single-run report -----------------------------------------------------------

def render_report(run: RunData) -> str:
    m = run.manifest
    s = run.summary
    lines: list[str] = []
    add = lines.append

    add(f"# Run Report - {run.name}")
    add("")
    add(f"**Result:** {s.get('result', 'unknown')} · **Git:** `{m.get('git_sha', '?')}` · "
        f"**Duration target:** {m.get('hours', '?')} h · **Samples:** {len(run.samples)} · "
        f"**Stalls:** {s.get('stall_events', '?')}")
    add("")

    # Verdict block
    add("## Verdict")
    add("")
    gates = run.gates.get("gates", [])
    if gates:
        add(f"**Overall: {'PASS' if run.gates.get('passed') else 'FAIL'}**")
        add("")
        for g in gates:
            mark = "PASS" if g["ok"] else ("SKIP" if g["ok"] is None else "FAIL")
            add(f"- [{mark}] {g['name']} - {g['detail']}")
    else:
        add("*(no gates.json in run dir)*")
    add("")
    add("> Standing caveat: the dominant-loss canary misfires on synthetic "
        "proprioception-only input (PC loss legitimately dominates). "
        "Re-evaluate under MuJoCo embodied input.")
    add("")

    # 1 Stability
    add("## 1. Stability")
    add("")
    c0, c1 = run.first("cycles_completed"), run.last("cycles_completed")
    add(f"- cycles: {_fmt(c0, 0)} -> {_fmt(c1, 0)}")
    rates = [r["cycle_rate_hz"] for r in run.minutes if isinstance(r.get("cycle_rate_hz"), (int, float))]
    if rates:
        add(f"- cycle rate (per-minute): mean {statistics.fmean(rates):.2f} Hz, "
            f"min {min(rates):.2f}, max {max(rates):.2f}")
    add(f"- frames received/dropped: {_fmt(run.last('frames_received'), 0)} / {_fmt(run.last('frames_dropped'), 0)}")
    add(f"- nan recoveries: {_fmt(run.last('nan_recovery_events'), 0)}")
    add(f"- gpu mem (MiB) first -> last: {_fmt(run.first('gpu_mem_used_mib'), 0)} -> {_fmt(run.last('gpu_mem_used_mib'), 0)}")
    add(f"- run dir size: {_fmt(run.last('run_dir_mb'), 1)} MB · disk free: {_fmt(run.last('disk_free_gb'), 1)} GB")
    add(_img(_plot(run, ["approx_cycles_per_sec"], "Cycle rate", "plot_cycle_rate.png")))
    add(_img(_plot(run, ["stage_pipeline_active_sessions", "prefetch_queue_depth", "queue_depth"],
                   "Queues and sessions", "plot_queues.png")))

    # 2 Learning
    add("## 2. Learning")
    add("")
    add(_trend_line(run, "neural_pc_loss_last"))
    add(_trend_line(run, "loss_total"))
    for key in ("forward_model_error", "intero_pred_error", "tactile_pred_error",
                "effort_pred_error", "consolidator_loss"):
        add(_trend_line(run, key))
    add(f"- rewire events: {_fmt(run.last('rewire_events'), 0)} · growth events: {_fmt(run.last('growth_events'), 0)}")
    add(f"- plasticity freezes/thaws: {_fmt(run.last('plasticity_freeze_count'), 0)}/{_fmt(run.last('plasticity_thaw_count'), 0)}")
    add(_img(_plot(run, ["neural_pc_loss_last", "loss_total"], "Predictive-coding loss", "plot_pc_loss.png", logy=True)))
    add(_img(_plot(run, ["forward_model_error", "intero_pred_error", "tactile_pred_error"],
                   "Per-head errors", "plot_heads.png", logy=True)))

    # 3 State coherence
    add("## 3. State coherence (A-F)")
    add("")
    for prefix, label in (("a", "A state-of-mind"), ("b", "B emotion"), ("c", "C narrative"), ("e", "E metacognition")):
        add(_trend_line(run, f"{prefix}_norm"))
    add(f"- viability first -> last: {_fmt(run.first('viability'), 2)} -> {_fmt(run.last('viability'), 2)}")
    labels: dict[str, int] = {}
    for smp in run.samples:
        lab = smp.get("priority_label")
        if isinstance(lab, str):
            labels[lab] = labels.get(lab, 0) + 1
    if labels:
        total = sum(labels.values())
        dist = ", ".join(f"{k}: {100 * v / total:.1f}%" for k, v in sorted(labels.items(), key=lambda x: -x[1]))
        add(f"- priority distribution: {dist}")
    add(_img(_plot(run, ["a_norm", "b_norm", "c_norm", "e_norm"], "State-bus norms", "plot_state_norms.png")))
    add(_img(_plot(run, ["viability", "energy", "hydration", "integrity"], "Viability and reservoirs", "plot_viability.png")))
    add(_img(_plot(run, ["state_pain_scalar", "state_pleasure_scalar"], "Pain / pleasure", "plot_affect.png")))

    # 4 Memory / consolidation
    add("## 4. Memory and consolidation")
    add("")
    add(f"- episodic rows: {_fmt(run.first('episodic_db_rows'), 0)} -> {_fmt(run.last('episodic_db_rows'), 0)} "
        f"({_fmt((run.last('episodic_db_bytes') or 0) / 1e6, 1)} MB)")
    add(f"- LTM db: {_fmt((run.last('ltm_db_bytes') or 0) / 1e6, 1)} MB · property beliefs: {_fmt(run.last('ltm_property_beliefs'), 0)}")
    add(f"- replay buffer: {_fmt(run.last('replay_buffer_size'), 0)} · replays: {_fmt(run.last('replay_count'), 0)}")
    add(f"- recall cache hits/misses: {_fmt(run.last('memory_recall_cache_hits'), 0)}/{_fmt(run.last('memory_recall_cache_misses'), 0)}")
    add(_img(_plot(run, ["episodic_db_rows", "replay_buffer_size"], "Memory growth", "plot_memory.png")))

    # 5 Distinctness
    add("## 5. Distinctness")
    add("")
    add("*Single-run report - populated in comparison mode (`--compare run_a run_b`) "
        "once the baseline agent exists.*")
    add("")
    return "\n".join(lines)


# -- comparison report -------------------------------------------------------------

COMPARE_KEYS = (
    "neural_pc_loss_last",
    "loss_total",
    "viability",
    "approx_cycles_per_sec",
    "episodic_db_rows",
    "state_pleasure_scalar",
)


def render_compare(a: RunData, b: RunData, out_dir: Path) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Comparison Report - {a.name} vs {b.name}")
    add("")
    add(f"| | {a.name} | {b.name} |")
    add("|---|---|---|")
    add(f"| result | {a.summary.get('result')} | {b.summary.get('result')} |")
    add(f"| gates | {'PASS' if a.gates.get('passed') else 'FAIL'} | {'PASS' if b.gates.get('passed') else 'FAIL'} |")
    add(f"| samples | {len(a.samples)} | {len(b.samples)} |")
    add(f"| consolidation | {a.manifest.get('env', {}).get('DECADIC_CONSOLIDATION_ENABLED', 'default')} "
        f"| {b.manifest.get('env', {}).get('DECADIC_CONSOLIDATION_ENABLED', 'default')} |")
    for key in COMPARE_KEYS:
        row = [f"| {key} (first -> last)"]
        for run in (a, b):
            xs, ys = run.series(key)
            row.append(f" {ys[0]:.4f} -> {ys[-1]:.4f} " if len(ys) >= 2 else " n/a ")
        add("|".join(row) + "|")
    add("")
    add("## Overlaid trends")
    for key in COMPARE_KEYS:
        img = _plot_compare(a, b, key, key, out_dir, f"compare_{key}.png")
        add(f"### {key}")
        add(_img(img))
    add("")
    add("Simple effect size (second-half mean delta, b - a):")
    add("")
    for key in COMPARE_KEYS:
        halves = []
        for run in (a, b):
            _, ys = run.series(key)
            halves.append(statistics.fmean(ys[len(ys) // 2:]) if len(ys) >= 4 else None)
        if None not in halves:
            add(f"- `{key}`: {halves[1] - halves[0]:+.4f}")
    add("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="run directory (single-run mode)")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--out", default=None, help="output path override")
    args = ap.parse_args()

    if args.compare:
        a, b = (RunData(p) for p in args.compare)
        out_dir = Path(args.out).parent if args.out else a.dir
        text = render_compare(a, b, out_dir)
        out = Path(args.out) if args.out else a.dir / f"compare_{a.name}_vs_{b.name}.md"
    elif args.run_dir:
        run = RunData(args.run_dir)
        text = render_report(run)
        out = Path(args.out) if args.out else run.dir / "report.md"
    else:
        ap.error("provide a run_dir or --compare RUN_A RUN_B")
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report written: {out}")
    if not HAVE_MPL:
        print("note: matplotlib not installed - text-only report (pip install -e .[dev])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
