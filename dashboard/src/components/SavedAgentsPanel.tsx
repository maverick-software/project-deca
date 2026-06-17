import { useCallback, useEffect, useState } from "react";
import {
  deleteSavedAgent,
  listSavedAgents,
  loadSavedAgent,
  type SavedAgent,
} from "../api";
import Info from "./Info";

function fmtDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/**
 * Saved Agents library: list durable saves (stored under saved_agents/,
 * separate from the auto-pruned backups/ checkpoints) and load any one back
 * into a brand-new live agent. onLoaded receives the freshly created agent id.
 */
export default function SavedAgentsPanel(props: {
  onLoaded: (agentId: string) => void;
}) {
  const { onLoaded } = props;
  const [saves, setSaves] = useState<SavedAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSaves(await listSavedAgents());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const load = async (saveId: string) => {
    setBusyId(saveId);
    setError(null);
    try {
      const agentId = await loadSavedAgent(saveId);
      onLoaded(agentId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (saveId: string) => {
    if (confirmId !== saveId) {
      setConfirmId(saveId);
      window.setTimeout(() => setConfirmId((c) => (c === saveId ? null : c)), 3000);
      return;
    }
    setConfirmId(null);
    setBusyId(saveId);
    setError(null);
    try {
      await deleteSavedAgent(saveId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="panel">
      <h2>
        Saved Agents
        <Info tip="Durable saves stored under saved_agents/, kept separate from the auto-pruned backups/ checkpoints. Each save bundles the agent's brain, internal state, and episodic memory. Load spins up a brand-new live agent restored from the save; a body is not included, so start one from the Environment controls afterwards." />
      </h2>
      <div className="controls" style={{ marginBottom: 8 }}>
        <button className="btn" disabled={loading} onClick={() => void refresh()}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
        {error && <span className="ctrl-error">{error}</span>}
      </div>
      {saves.length === 0 && !loading ? (
        <p style={{ color: "var(--text-dim)" }}>
          No saved agents yet. Use the <b>Save</b> button in the top bar to capture the
          selected agent.
        </p>
      ) : (
        <table className="disc-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Preset</th>
              <th>Encoder</th>
              <th>Viability</th>
              <th>Cycle</th>
              <th>Memory</th>
              <th>Saved</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {saves.map((s) => (
              <tr key={s.save_id}>
                <td className="row-part" title={s.notes ?? undefined}>
                  {s.name}
                </td>
                <td>{s.preset ?? "—"}</td>
                <td>{s.encoder_mode ?? "—"}</td>
                <td>{s.viability != null ? s.viability.toFixed(1) : "—"}</td>
                <td>{s.cycle_index ?? "—"}</td>
                <td>{s.has_memory ? "yes" : "no"}</td>
                <td>{fmtDate(s.created_at)}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button
                    className="btn start"
                    disabled={busyId != null}
                    title="Load this save into a brand-new live agent"
                    onClick={() => void load(s.save_id)}
                  >
                    {busyId === s.save_id ? "Loading…" : "Load"}
                  </button>{" "}
                  <button
                    className={`btn ${confirmId === s.save_id ? "danger" : "reset"}`}
                    disabled={busyId != null}
                    title="Delete this save permanently"
                    onClick={() => void remove(s.save_id)}
                  >
                    {confirmId === s.save_id ? "Confirm?" : "Delete"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
