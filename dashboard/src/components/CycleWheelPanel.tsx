import { useEffect, useRef, useState } from "react";
import type { CycleTrace } from "../api";
import { usePersistentState } from "../usePersistentState";
import Info from "./Info";

/**
 * Animated ten-stage Decadic wheel. Replays the most recent completed cycle
 * as a slowed sweep: each segment lights in order, dwell proportional to the
 * measured per-stage compute time, brightness from real latent activity.
 */

const STAGES = [
  { n: 1, short: "Perception", full: "Sensory Perception" },
  { n: 2, short: "Framing", full: "Experience Framing" },
  { n: 3, short: "Memory", full: "Memory Retrieval (CFM)" },
  { n: 4, short: "Risk-Utility", full: "Risk-Utility Evaluation" },
  { n: 5, short: "Pre-Norm", full: "Pre-Normative Conclusion" },
  { n: 6, short: "Emotion", full: "Emotional / Physiological Experience" },
  { n: 7, short: "Reprioritize", full: "Reprioritization & State of Mind Update" },
  { n: 8, short: "Strategy", full: "Strategy Formation" },
  { n: 9, short: "Response", full: "Behavioral Response" },
  { n: 10, short: "Memory Map", full: "Normative Memory Mapping" },
];

const CX = 200;
const CY = 195;
const R_OUT = 138;
const R_IN = 96;
const GAP_DEG = 4;

function polar(r: number, deg: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

function sectorPath(a0: number, a1: number): string {
  const [x0, y0] = polar(R_OUT, a0);
  const [x1, y1] = polar(R_OUT, a1);
  const [x2, y2] = polar(R_IN, a1);
  const [x3, y3] = polar(R_IN, a0);
  return `M${x0},${y0} A${R_OUT},${R_OUT} 0 0 1 ${x1},${y1} L${x2},${y2} A${R_IN},${R_IN} 0 0 0 ${x3},${y3} Z`;
}

type Anim = {
  trace: CycleTrace;
  start: number;
  bounds: number[]; // cumulative seconds at which each stage ends
};

export default function CycleWheelPanel(props: { trace: CycleTrace | null | undefined }) {
  const [duration, setDuration] = usePersistentState("decadic.cycle.sweep", 0.75);
  const [paused, setPaused] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [display, setDisplay] = useState<CycleTrace | null>(null);

  const pendingRef = useRef<CycleTrace | null>(null);
  const animRef = useRef<Anim | null>(null);
  const lastCycleRef = useRef(-1);

  // Queue the newest cycle delivered by polling (multiple cycles pass between
  // polls; we replay the most recent one).
  useEffect(() => {
    if (props.trace && props.trace.cycle !== lastCycleRef.current) {
      pendingRef.current = props.trace;
    }
  }, [props.trace]);

  useEffect(() => {
    let raf = 0;
    const tick = (tMs: number) => {
      raf = requestAnimationFrame(tick);
      if (paused) return;
      let anim = animRef.current;
      if (!anim) {
        const next = pendingRef.current;
        if (!next) return;
        pendingRef.current = null;
        lastCycleRef.current = next.cycle;
        const times = next.stages.map((s) => Math.max(0.3, s.payload.timing_ms ?? 0.3));
        const total = times.reduce((a, b) => a + b, 0);
        const bounds: number[] = [];
        let acc = 0;
        for (const x of times) {
          acc += (x / total) * duration;
          bounds.push(acc);
        }
        anim = animRef.current = { trace: next, start: tMs, bounds };
        setDisplay(next);
      }
      const el = (tMs - anim.start) / 1000;
      const idx = anim.bounds.findIndex((b) => el < b);
      if (idx === -1) {
        animRef.current = null;
        setActiveIdx(-1);
      } else {
        setActiveIdx(idx);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [duration, paused]);

  const stages = display?.stages ?? [];
  const maxAct = stages.reduce((m, s) => Math.max(m, s.payload.activity ?? 0), 1e-6);
  const maxPc = stages.reduce((m, s) => Math.max(m, s.payload.pc_part ?? 0), 1e-6);

  const segDeg = 360 / STAGES.length;

  return (
    <div className="panel span-7">
      <h2>
        Decadic Cycle — Live Replay
        <Info tip="Replays the brain's most recent completed cycle as a slowed sweep (the real cycle finishes in well under a second). Each segment lights in stage order; dwell time is the measured compute time of that stage's network block, brightness is its real latent activity, and a red rim marks predictive-coding surprise at that stage." />
      </h2>

      <div className="wheel-wrap">
        <svg viewBox="0 0 400 390" className="wheel-svg">
          {/* faint center rings echoing the State Bus layers */}
          {[78, 62, 46, 30].map((r) => (
            <circle key={r} cx={CX} cy={CY} r={r} className="wheel-ring" />
          ))}
          {STAGES.map((st, i) => {
            const a0 = i * segDeg + GAP_DEG / 2;
            const a1 = (i + 1) * segDeg - GAP_DEG / 2;
            const mid = (a0 + a1) / 2;
            const s = stages[i];
            const act = s ? (s.payload.activity ?? 0) / maxAct : 0;
            const pc = s ? (s.payload.pc_part ?? 0) / maxPc : 0;
            const isActive = i === activeIdx;
            const swept = activeIdx >= 0 && i < activeIdx;
            const fillL = isActive ? 58 : swept ? 22 + act * 22 : display ? 14 + act * 14 : 12;
            const [nx, ny] = polar((R_OUT + R_IN) / 2, mid);
            const [lx, ly] = polar(R_OUT + 16, mid);
            const sx = Math.sin((mid * Math.PI) / 180);
            const anchor = Math.abs(sx) < 0.35 ? "middle" : sx > 0 ? "start" : "end";
            const ms = s?.payload.timing_ms;
            const actVal = s?.payload.activity;
            const pcVal = s?.payload.pc_part;
            return (
              <g key={st.n}>
                <path
                  d={sectorPath(a0, a1)}
                  fill={`hsl(215 80% ${fillL}%)`}
                  stroke={
                    pc > 0.02 ? `hsla(355, 80%, 60%, ${0.15 + 0.85 * pc})` : "#232a3a"
                  }
                  strokeWidth={isActive ? 2.5 : 1.2}
                  className={isActive ? "wheel-seg active" : "wheel-seg"}
                >
                  <title>
                    {`${st.n}. ${st.full}${ms != null ? ` — ${ms.toFixed(2)} ms` : ""}${
                      actVal != null ? ` · activity ${actVal.toFixed(3)}` : ""
                    }${pcVal != null ? ` · PC error ${pcVal.toFixed(4)}` : ""}`}
                  </title>
                </path>
                <text x={nx} y={ny + 4} textAnchor="middle" className="wheel-num">
                  {st.n}
                </text>
                <text x={lx} y={ly + 3} textAnchor={anchor} className="wheel-label">
                  {st.short}
                </text>
              </g>
            );
          })}
          <text x={CX} y={CY - 6} textAnchor="middle" className="wheel-cycle">
            {display ? `#${display.cycle}` : "—"}
          </text>
          <text x={CX} y={CY + 14} textAnchor="middle" className="wheel-caption">
            {activeIdx >= 0 ? STAGES[activeIdx].short.toUpperCase() : "CYCLE"}
          </text>
        </svg>

        <div className="wheel-side">
          <div className="wheel-controls">
            <label>
              Sweep
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
              >
                <option value={0.75}>0.75 s</option>
                <option value={1.5}>1.5 s</option>
                <option value={3}>3 s</option>
              </select>
            </label>
            <button className="btn" onClick={() => setPaused((p) => !p)}>
              {paused ? "\u25B6 Resume" : "\u275A\u275A Pause"}
            </button>
          </div>
          <div className="wheel-stages">
            {STAGES.map((st, i) => {
              const s = stages[i];
              return (
                <div
                  key={st.n}
                  className={`wheel-stage-row${i === activeIdx ? " active" : ""}`}
                >
                  <span className="num">{st.n}</span>
                  <span className="name">{st.full}</span>
                  <span className="ms">
                    {s?.payload.timing_ms != null
                      ? `${s.payload.timing_ms.toFixed(2)} ms`
                      : "—"}
                  </span>
                  <span className="act">
                    {s?.payload.activity != null
                      ? s.payload.activity.toFixed(2)
                      : s?.payload.salience != null
                        ? `s ${s.payload.salience.toFixed(2)}`
                        : ""}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="wheel-legend">
            brightness = latent activity · red rim = PC surprise · dwell = compute time
          </div>
        </div>
      </div>
    </div>
  );
}
