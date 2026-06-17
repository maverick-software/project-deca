import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Metrics } from "../api";
import type { HistorySample } from "../usePolling";
import Info from "./Info";

function Chart(props: {
  data: HistorySample[];
  dataKey: keyof HistorySample;
  color: string;
  domain?: [number, number];
}) {
  return (
    <ResponsiveContainer width="100%" height={110}>
      <LineChart data={props.data} margin={{ top: 4, right: 6, bottom: 0, left: -18 }}>
        <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#7c8499" }} stroke="#232a3a" />
        <YAxis
          tick={{ fontSize: 10, fill: "#7c8499" }}
          stroke="#232a3a"
          domain={props.domain ?? ["auto", "auto"]}
          width={58}
        />
        <Tooltip
          contentStyle={{
            background: "#131722",
            border: "1px solid #232a3a",
            borderRadius: 6,
            fontSize: 12,
          }}
          labelFormatter={(t) => `cycle ${t}`}
        />
        <Line
          type="monotone"
          dataKey={props.dataKey}
          stroke={props.color}
          strokeWidth={1.6}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function CyclePanel(props: {
  metrics: Metrics | null;
  history: HistorySample[];
}) {
  const m = props.metrics;
  return (
    <div className="panel span-8">
      <h2>
        Cognitive Cycle
        <Info tip="One Decadic cycle = ten stages run in sequence: perceive, fuse, recall, appraise, predict, deliberate, narrate, prioritize, meta-monitor, act. The neural stack executes all ten and trains itself by predictive coding as it goes." />
      </h2>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginBottom: 8 }}>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Cycles
            <Info tip="Full ten-stage cycles completed since the agent started (or was reset)." />
          </span>
          <span className="v">{m?.cycles_completed ?? "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Cycles/s
            <Info tip="Average cycle rate since start. Determined by the configured cycle interval plus how long each cycle's compute takes." />
          </span>
          <span className="v">{m ? m.approx_cycles_per_sec.toFixed(1) : "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            LR
            <Info tip="Learning rate of the optimizer training the neural cognitive stack. Modulated upward by pain/salience so significant experiences are learned faster." />
          </span>
          <span className="v">{m ? m.learning_rate.toExponential(1) : "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Fast-path hits
            <Info tip="Reflexes: events intense enough (e.g. a hard collision) to trigger an immediate response that bypasses the full cycle, like pulling your hand off a hot stove before 'thinking'." />
          </span>
          <span className="v">{m?.fast_path_hits ?? "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Cycle wall ms
            <Info tip="Wall-clock time the last cycle took to compute (encoders + ten stages + learning step). Lower is better; this is where the GPU helps." />
          </span>
          <span className="v">{m ? m.last_cycle_wall_ms.toFixed(1) : "—"}</span>
        </div>
      </div>

      <div className="strip-label">
        <span>
          Predictive-coding loss
          <Info tip="How badly each stage predicted the activity of the next stage. This is the network's main training signal — a falling line means its internal world-model is improving; spikes mean surprise (something unexpected happened or weights were just reset)." />
        </span>
        <span>{m ? m.neural_pc_loss_last.toFixed(4) : "—"}</span>
      </div>
      <Chart data={props.history} dataKey="pcLoss" color="#5aa9ff" />

      <div className="strip-label" style={{ marginTop: 8 }}>
        <span>
          Viability
          <Info tip="The Vitals gauge over time. Slow decline = ongoing stress or prediction error; drops = damage events; recovery = calm, well-predicted stretches." />
        </span>
        <span>{m ? m.viability.toFixed(1) : "—"}</span>
      </div>
      <Chart data={props.history} dataKey="viability" color="#4fd683" domain={[0, 100]} />
    </div>
  );
}
