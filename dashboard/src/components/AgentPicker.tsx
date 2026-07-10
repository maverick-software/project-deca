import type { AgentSummary, SavedAgent } from "../api";

// The top-bar agent picker. Lists LIVE (in-memory) agents plus the SAVED brain
// library. Live agents switch instantly; picking a saved brain restores it as a
// brand-new live agent (spawns via onLoadSaved, comes up body-less). Option
// values are prefixed live:/saved: so the two namespaces never collide.
export default function AgentPicker(props: {
  agents: AgentSummary[];
  saved?: SavedAgent[];
  selected: string | null;
  onSelect: (id: string) => void;
  onLoadSaved?: (saveId: string) => void;
  loadingSaved?: boolean;
}) {
  const { agents, selected, onSelect } = props;
  const saved = props.saved ?? [];
  const hasLive = agents.length > 0;
  const hasSaved = saved.length > 0;
  if (!hasLive && !hasSaved) {
    return <span className="sub">no agents</span>;
  }
  const handleChange = (v: string) => {
    if (v.startsWith("saved:")) {
      props.onLoadSaved?.(v.slice("saved:".length));
    } else if (v.startsWith("live:")) {
      onSelect(v.slice("live:".length));
    }
  };
  return (
    <>
      <select
        value={selected ? `live:${selected}` : ""}
        onChange={(e) => handleChange(e.target.value)}
        disabled={props.loadingSaved}
      >
        {!selected && (
          <option value="" disabled>
            select agent…
          </option>
        )}
        {hasLive && (
          <optgroup label="Live agents">
            {agents.map((a) => (
              <option key={a.agent_id} value={`live:${a.agent_id}`}>
                {a.agent_id.slice(0, 8)} — {a.neural_enabled ? "neural" : "stub"} (
                {a.cycles_completed} cycles){a.has_body === false ? " — no body" : ""}
                {a.status === "dead" ? " — dead" : a.paused ? " — paused" : ""}
              </option>
            ))}
          </optgroup>
        )}
        {hasSaved && (
          <optgroup label="Saved brains (load restores as new agent)">
            {saved.map((s) => (
              <option key={s.save_id} value={`saved:${s.save_id}`}>
                {s.name} ({s.cycle_index ?? "—"} cycles)
              </option>
            ))}
          </optgroup>
        )}
      </select>
      {props.loadingSaved && <span className="sub"> loading…</span>}
    </>
  );
}
