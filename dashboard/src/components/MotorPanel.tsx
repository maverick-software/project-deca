import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { resetBraces, setBracesEnabled, setMovementHold, setStance, STANCES, type Metrics } from "../api";
import type { HistorySample } from "../usePolling";
import Info from "./Info";

/** Per-joint motor command bars: how hard, and which way, each actuator is driven. */
function MotorBars(props: { ctrl: number[] }) {
  const { ctrl } = props;
  if (!ctrl.length) {
    return <div className="empty">no motor command yet — connect a body</div>;
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

export default function MotorPanel(props: {
  metrics: Metrics | null;
  history: HistorySample[];
  agentId: string | null;
}) {
  const m = props.metrics;
  const ctrl = m?.motor_command ?? [];
  const hasFwd = props.history.some((h) => h.fwdErr != null);
  const romMean = m?.rom_mean ?? 0;
  const braceEngaged = m?.brace_engaged ?? 1;
  const bracesEnabled = m?.braces_enabled ?? true;
  const stance = m?.stance ?? "stand";
  const stancePhase = m?.stance_phase ?? 0;
  const activeStance = STANCES.find((s) => s.name === stance);
  const movementHold = m?.movement_hold ?? false;

  const onResetRom = () => {
    if (!props.agentId) return;
    resetBraces(props.agentId).catch(() => {});
  };

  const onToggleBraces = () => {
    if (!props.agentId) return;
    setBracesEnabled(props.agentId, !bracesEnabled).catch(() => {});
  };

  const onSetStance = (name: string) => {
    if (!props.agentId) return;
    setStance(props.agentId, name).catch(() => {});
  };

  const onToggleHold = () => {
    if (!props.agentId) return;
    setMovementHold(props.agentId, !movementHold).catch(() => {});
  };

  return (
    <div className="panel span-5">
      <h2>
        Motor / Active Inference
        <Info tip="The agent's efferent loop. A motor head emits one PD target per actuator; the body tracks them with a fast reflex-like loop. Actions are chosen to minimize the forward model's predicted deviation from a survival prior (upright, standing, stable) — no scripted locomotion. There is no external support: every joint is braced toward the stand pose by an internal joint spring, so the feet always carry the full weight and all travel must come from real limb push-off." />
      </h2>

      <div className="assist-control">
        <span className="assist-control-label">
          Joint braces {bracesEnabled ? "(on)" : "(off — free body)"}
          <Info tip="Master on/off for the internal joint orthosis. On: every hinge is braced toward the upright stand pose (starts welded) and earns range of motion as the brain's per-joint forward-model error falls — no external force, so it cannot skate. Off: hinges relax to their native springs and the brain alone holds the body up, so it can fall (useful for observing raw behavior). Earned ROM is preserved and resumes when switched back on." />
        </span>
        <div className="assist-buttons">
          <button
            type="button"
            className={`assist-btn ${bracesEnabled ? "active" : ""}`}
            disabled={!props.agentId}
            onClick={onToggleBraces}
            title={
              bracesEnabled
                ? "Switch the joint braces OFF (free body — it can fall)"
                : "Switch the joint braces ON (brace toward the stand pose)"
            }
          >
            {bracesEnabled ? "On" : "Off"}
          </button>
          <button
            type="button"
            className="assist-btn"
            disabled={!props.agentId || !bracesEnabled}
            onClick={onResetRom}
            title="Re-weld every joint brace and restart the ROM curriculum from zero"
          >
            Reset ROM
          </button>
        </div>
      </div>

      <div className="assist-control">
        <span className="assist-control-label">
          Stance
          <Info tip="The braced posture the body is held in. Selecting a stance re-poses the body into that stance's start pose and re-welds every joint brace, so the agent re-learns range of motion for the new posture from fully braced. Static stances (stand, all-fours, kneel) hold a fixed pose; motion stances (crawl, rise) drive a trajectory the braces track, so travel comes from real contact with the floor — never an external force." />
        </span>
        <div className="assist-buttons">
          {STANCES.map((s) => (
            <button
              key={s.name}
              type="button"
              className={`assist-btn ${stance === s.name ? "active" : ""}`}
              disabled={!props.agentId}
              onClick={() => onSetStance(s.name)}
              title={s.motion ? `${s.label} — a braced motion trajectory` : `${s.label} — a static braced pose`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="assist-control">
        <span className="assist-control-label">
          Hold movement {movementHold ? "(on)" : "(off)"}
          <Info tip="Run the active movement until you turn this off. On: every joint brace stays fully welded — the ROM curriculum is suspended (no range-of-motion release) — and motion stances (crawl, rise) loop continuously, so the body is driven through the selected movement precisely on repeat. Off: the per-joint ROM curriculum resumes (joints re-earn range as prediction error falls). Has no effect while the joint braces are off." />
        </span>
        <div className="assist-buttons">
          <button
            type="button"
            className={`assist-btn ${movementHold ? "active" : ""}`}
            disabled={!props.agentId || !bracesEnabled}
            onClick={onToggleHold}
            title={
              movementHold
                ? "Stop holding — resume the ROM curriculum (joints re-earn range)"
                : "Hold the active movement welded and looping until you turn it off"
            }
          >
            {movementHold ? "On" : "Off"}
          </button>
        </div>
      </div>

      {activeStance?.motion && (
        <div className="strip-label">
          <span>
            Stance phase
            <Info tip="Progress through the current motion stance's trajectory (loops for crawl, one-shot for rise). The stiff braces track this moving reference; the limbs push the floor, so motion is contact-driven." />
          </span>
          <span>{(stancePhase * 100).toFixed(0)}%</span>
        </div>
      )}

      <div className="strip-label">
        <span>
          Range of motion earned
          <Info tip="Mean per-joint ROM across all hinges (0% = fully welded into the stand pose, 100% = native/free). Ratchets open monotonically as prediction errors fall. Brace engaged = mean weld tightness (the inverse)." />
        </span>
        <span>
          {(romMean * 100).toFixed(0)}% ROM · {(braceEngaged * 100).toFixed(0)}% braced
        </span>
      </div>
      <div className="assist-track">
        <div className="assist-fill" style={{ width: `${Math.min(1, romMean) * 100}%` }} />
      </div>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "10px 0 4px" }}>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Forward-model error
            <Info tip="How wrong the agent's body-prediction was: MSE between the predicted next proprioceptive state and what actually happened. Falling = the agent is learning the consequences of its own motor commands (a working body model)." />
          </span>
          <span className="v">{m?.forward_model_error != null ? m.forward_model_error.toFixed(4) : "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Babble σ
            <Info tip="Motor exploration noise added to commands, decaying over training. High early = the body 'babbles' to discover what its actuators do (like an infant's spontaneous movements); near zero late = deliberate control." />
          </span>
          <span className="v">{m?.motor_babble_sigma != null ? m.motor_babble_sigma.toFixed(3) : "—"}</span>
        </div>
        <div className="statrow" style={{ gap: 8 }}>
          <span className="k">
            Activity (RMS)
            <Info tip="Root-mean-square of the per-actuator command magnitudes — overall how vigorously the body is being driven right now." />
          </span>
          <span className="v">{m?.motor_activity_rms != null ? m.motor_activity_rms.toFixed(3) : "—"}</span>
        </div>
      </div>

      {hasFwd && (
        <>
          <div className="strip-label" style={{ marginTop: 6 }}>
            <span>Forward-model error trend</span>
            <span>{m?.forward_model_error != null ? m.forward_model_error.toFixed(4) : "—"}</span>
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
          <Info tip="The motor head's normalized PD target for each of the body's actuators (up = positive, down = negative). Watch for stable, structured patterns emerging from initial noise — that's coordination forming." />
        </span>
        <span>{ctrl.length} actuators</span>
      </div>
      <MotorBars ctrl={ctrl} />
    </div>
  );
}
