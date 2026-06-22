import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { STANCES, type Metrics } from "../api";
import type { HistorySample } from "../usePolling";
import Info from "./Info";

function MotorBars(props: { ctrl: number[] }) {
  const { ctrl } = props;
  if (!ctrl.length) {
    return <div className="empty">no motor command yet - connect a body</div>;
  }
  return (
    <div className="motor-bars">
      {ctrl.map((u, i) => {
        const mag = Math.min(1, Math.abs(u));
        const pos = u >= 0;
        return (
          <div className="motor-bar" key={i} title={`actuator ${i}: ${u.toFixed(3)}`}>
            <div className="motor-bar-track">
              <div
                className={`motor-bar-fill ${pos ? "pos" : "neg"}`}
                style={{
                  height: `${mag * 100}%`,
                  alignSelf: pos ? "flex-end" : "flex-start",
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function metric(value: number | undefined | null, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function BodyMapMini(props: { metrics: Metrics | null }) {
  const bm = props.metrics?.body_map;
  const parts = bm?.parts ?? [];
  const contact = bm?.contact_load ?? [];
  const pain = bm?.pain ?? [];
  const fatigue = bm?.fatigue ?? [];
  const rows = parts
    .map((part, i) => ({
      part,
      contact: contact[i] ?? 0,
      pain: pain[i] ?? 0,
      fatigue: fatigue[i] ?? 0,
    }))
    .sort((a, b) => Math.max(b.contact, b.pain, b.fatigue) - Math.max(a.contact, a.pain, a.fatigue))
    .slice(0, 6);
  if (!rows.length) return null;
  return (
    <div className="body-map-mini">
      {rows.map((r) => {
        const mag = Math.max(r.contact, r.pain, r.fatigue);
        return (
          <div className="body-map-row" key={r.part} title={`${r.part}: contact ${metric(r.contact, 2)}, fatigue ${metric(r.fatigue, 2)}, pain ${metric(r.pain, 2)}`}>
            <span>{r.part.replaceAll("_", " ")}</span>
            <div className="body-map-track">
              <div className="body-map-fill" style={{ width: `${Math.min(1, mag) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function MotorPanel(props: {
  metrics: Metrics | null;
  history: HistorySample[];
  agentId: string | null;
}) {
  const m = props.metrics;
  const ctrl = m?.motor_command ?? [];
  const hasFwd = props.history.some((h) => h.fwdErr != null);
  const romMean = m?.rom_mean ?? 0;
  const braceEngaged = m?.brace_engaged ?? 0;
  const bracesEnabled = m?.braces_enabled ?? false;
  const stance = m?.stance ?? "stand";
  const stancePhase = m?.stance_phase ?? 0;
  const activeStance = STANCES.find((s) => s.name === stance);
  const movementHold = m?.movement_hold ?? false;

  return (
    <div className="panel span-5">
      <h2>
        Motor / Active Inference
        <Info tip="Read-only motor telemetry. Manual brace and stance controls live in Skill Dojo's Body Scaffold panel; Skill Dojo does not command braces." />
      </h2>

      <div className="dojo-scaffold-stats" style={{ marginBottom: 10 }}>
        <div className="statrow">
          <span className="k">Joint braces</span>
          <span className="v">{bracesEnabled ? "on" : "off"}</span>
        </div>
        <div className="statrow">
          <span className="k">Stance</span>
          <span className="v">
            {activeStance?.label ?? stance}
            {activeStance?.motion ? ` (${(stancePhase * 100).toFixed(0)}%)` : ""}
          </span>
        </div>
        <div className="statrow">
          <span className="k">Movement hold</span>
          <span className="v">{movementHold ? "on" : "off"}</span>
        </div>
        <div className="statrow">
          <span className="k">Contacts</span>
          <span className="v">
            F {metric(m?.foot_load_l, 2)}/{metric(m?.foot_load_r, 2)} H{" "}
            {metric(m?.hand_load_l, 2)}/{metric(m?.hand_load_r, 2)}
          </span>
        </div>
      </div>

      <div className="strip-label">
        <span>
          Brace ROM release
          <Info tip="Mean per-joint ROM across all hinges. Brace engaged is the inverse weld tightness." />
        </span>
        <span>
          {(romMean * 100).toFixed(0)}% ROM - {(braceEngaged * 100).toFixed(0)}% braced
        </span>
      </div>
      <div className="assist-track">
        <div className="assist-fill" style={{ width: `${Math.min(1, romMean) * 100}%` }} />
      </div>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "10px 0 4px" }}>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Forward-model error
            <Info tip="MSE between predicted next proprioceptive state and what actually happened." />
          </span>
          <span className="v">{metric(m?.forward_model_error, 4)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Babble sigma
            <Info tip="Motor exploration noise added to commands." />
          </span>
          <span className="v">{metric(m?.motor_babble_sigma, 3)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Activity RMS
            <Info tip="Root-mean-square of per-actuator command magnitudes." />
          </span>
          <span className="v">{metric(m?.motor_activity_rms, 3)}</span>
        </div>
      </div>

      <div className="strip-label" style={{ marginTop: 6 }}>
        <span>
          Effort / body map
          <Info tip="Body-localized effort, fatigue, and pain from actuator work and contact. Body part names are sensor addresses; external objects remain anonymous." />
        </span>
        <span>{m?.most_pained_part ? `${m.most_pained_part.replaceAll("_", " ")} ${metric(m?.most_pained_part_pain, 2)}` : "no localized pain"}</span>
      </div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", margin: "6px 0 4px" }}>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Effort</span><span className="v">{metric(m?.effort_total, 3)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Work</span><span className="v">{metric(m?.work_total, 3)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Fatigue</span><span className="v">{metric(m?.fatigue_total, 3)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Pain</span><span className="v">{metric(m?.pain_total, 3)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Energy drain</span><span className="v">{metric(m?.effort_energy_delta, 4)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Net energy</span><span className="v">{metric(m?.net_energy_return, 4)}</span></div>
        <div className="statrow" style={{ gap: 8 }}><span className="k">Effort PE</span><span className="v">{metric(m?.effort_pred_error, 4)}</span></div>
      </div>
      <BodyMapMini metrics={m} />

      <div className="strip-label" style={{ marginTop: 6 }}>
        <span>
          Goals & long-horizon value
          <Info tip="Current goal and value-head telemetry for drive-reduction learning." />
        </span>
        <span>{m?.goal && m.goal !== "none" ? `${m.goal} (${m?.goal_status ?? "idle"})` : "idle"}</span>
      </div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "6px 0 4px" }}>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">Goal dwell</span>
          <span className="v">{m?.goal_dwell ?? 0}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">Episodes</span>
          <span className="v">
            {m?.episodes_closed ?? 0}
            {m?.goal_last_outcome ? ` - ${m.goal_last_outcome}` : ""}
          </span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">Last return</span>
          <span className="v">{metric(m?.episode_last_return, 4)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">SF value</span>
          <span className="v">{metric(m?.sf_value, 4)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">Value weight</span>
          <span className="v">{metric(m?.sf_value_weight, 3)}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">HER / seed</span>
          <span className="v">
            {m?.her_relabels ?? 0} / {m?.resource_seed ?? -1}
          </span>
        </div>
      </div>

      {hasFwd && (
        <>
          <div className="strip-label" style={{ marginTop: 6 }}>
            <span>Forward-model error trend</span>
            <span>{metric(m?.forward_model_error, 4)}</span>
          </div>
          <ResponsiveContainer width="100%" height={96}>
            <LineChart data={props.history} margin={{ top: 4, right: 6, bottom: 0, left: -18 }}>
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#7c8499" }} stroke="#232a3a" />
              <YAxis
                tick={{ fontSize: 10, fill: "#7c8499" }}
                stroke="#232a3a"
                domain={[0, "auto"]}
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
                dataKey="fwdErr"
                stroke="#e0a23b"
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      <div className="strip-label" style={{ marginTop: 6 }}>
        <span>
          Per-actuator command
          <Info tip="The motor head's normalized PD target for each body actuator." />
        </span>
        <span>{ctrl.length} actuators</span>
      </div>
      <MotorBars ctrl={ctrl} />
    </div>
  );
}
