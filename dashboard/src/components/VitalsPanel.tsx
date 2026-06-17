import type { AgentState, Metrics } from "../api";
import Info from "./Info";

function Gauge(props: { value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(100, (props.value / props.max) * 100));
  return (
    <div className="gauge-track">
      <div
        className="gauge-fill"
        style={{ width: `${pct}%`, backgroundColor: props.color }}
      />
    </div>
  );
}

export default function VitalsPanel(props: { state: AgentState; metrics: Metrics | null }) {
  const { state } = props;
  const viability = state.viability.value;
  const pain = state.state_bus.B_pain_scalar;
  const pleasure = state.state_bus.B_pleasure_scalar;
  const label = state.state_bus.D_priority_label;

  const viaColor = viability > 60 ? "#4fd683" : viability > 30 ? "#ffb74a" : "#ff5a6e";

  return (
    <div className="panel span-4">
      <h2>
        Vitals
        <Info tip="The agent's motivational core. Everything it learns is ultimately in service of keeping viability up — there is no external reward function." />
      </h2>
      <div className="strip-label">
        <span>
          Viability
          <Info tip="Survival resource (0–100), the minimum of the homeostasis reservoirs (hydration, energy, integrity). Thirst, hunger, and damage lower it; food, water, and healing restore it. Hitting 0 means the agent is no longer viable. See the Homeostasis panel for the breakdown." />
        </span>
        <span>{viability.toFixed(1)} / 100</span>
      </div>
      <Gauge value={viability} max={100} color={viaColor} />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Pain
          <Info tip="Generated from drops in viability (body damage, surprise). Spikes on impact, decays over time. High pain pushes priority toward 'avoid' and scales up learning on the offending experience." />
        </span>
        <span>{pain.toFixed(3)}</span>
      </div>
      <Gauge value={pain} max={5} color="#ff5a6e" />

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Pleasure
          <Info tip="Generated from gains in viability (recovery, successful prediction). Reinforces whatever the agent was just doing and biases priority toward 'explore'." />
        </span>
        <span>{pleasure.toFixed(3)}</span>
      </div>
      <Gauge value={pleasure} max={5} color="#4fd683" />

      <div className="statrow" style={{ marginTop: 14 }}>
        <span className="k">
          Priority (D)
          <Info tip="State Bus element D — the agent's current motivational stance, derived from pain/pleasure balance. 'explore' when comfortable, 'avoid' when pain or threat dominates." />
        </span>
        <span className={`badge ${label === "avoid" ? "avoid" : "explore"}`}>{label}</span>
      </div>
      <div className="statrow">
        <span className="k">
          Priority scalar
          <Info tip="Urgency of the current priority (0–1). Higher means behavior is more dominated by that single motive." />
        </span>
        <span className="v">{state.state_bus.D_priority_scalar.toFixed(3)}</span>
      </div>
      <div className="statrow">
        <span className="k">
          Current action
          <Info tip="What the body reports it is doing, as source:control-mode. 'mujoco_humanoid:root_assist' = the MuJoCo body is being steered at the pelvis while a PD controller keeps it standing." />
        </span>
        <span className="v">{state.perceptual.current_action_observed ?? "—"}</span>
      </div>
    </div>
  );
}
