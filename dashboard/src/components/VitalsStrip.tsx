import type { AgentState } from "../api";

/** Always-visible survival readout for the topbar; the full VitalsPanel lives on the Overview tab. */
export default function VitalsStrip(props: { state: AgentState }) {
  const { state } = props;
  const viability = state.viability.value;
  const label = state.state_bus.D_priority_label;
  const color = viability > 60 ? "#4fd683" : viability > 30 ? "#ffb74a" : "#ff5a6e";

  return (
    <div className="vitals-strip" title="Viability (0-100) and current motivational priority">
      <span className="vitals-strip-dot" style={{ backgroundColor: color }} />
      <span className="vitals-strip-value" style={{ color }}>
        {viability.toFixed(1)}
      </span>
      <span className="vitals-strip-unit">/ 100</span>
      <span className={`badge ${label === "avoid" ? "avoid" : "explore"}`}>{label}</span>
    </div>
  );
}
