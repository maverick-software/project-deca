import type { AgentSummary } from "../api";

export default function AgentPicker(props: {
  agents: AgentSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const { agents, selected, onSelect } = props;
  if (agents.length === 0) {
    return <span className="sub">no agents</span>;
  }
  return (
    <select value={selected ?? ""} onChange={(e) => onSelect(e.target.value)}>
      {agents.map((a) => (
        <option key={a.agent_id} value={a.agent_id}>
          {a.agent_id.slice(0, 8)} — {a.neural_enabled ? "neural" : "stub"} ({a.cycles_completed}{" "}
          cycles){a.has_body === false ? " — no body" : ""}
          {a.status === "dead" ? " — dead" : a.paused ? " — paused" : ""}
        </option>
      ))}
    </select>
  );
}
