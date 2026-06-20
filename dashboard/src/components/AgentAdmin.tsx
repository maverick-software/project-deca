import { useEffect, useRef, useState } from "react";
import {
  createAgent,
  deleteAgent,
  saveAgent,
  startEnvironment,
  type AgentPreset,
  type ScenarioDraft,
} from "../api";
import {
  NEURAL_PRESET_INFO,
  NEURAL_PRESET_LABELS,
  heavyPresetWarning,
} from "../neuralPresets";
import Info from "./Info";

/**
 * Topbar admin controls: pick a preset (a named scenario/body config), spawn a
 * fresh agent from the shared draft, save the selected agent to the library, or
 * delete the selected one. The preset dropdown loads its config into the shared
 * draft (reflected/edited in the Environment tab); "+ New agent" starts the
 * current draft.
 */
export default function AgentAdmin(props: {
  selected: string | null;
  presets: AgentPreset[];
  selectedPresetId: string | null;
  draft: ScenarioDraft;
  creationPreset: string;
  onSelectPreset: (id: string) => void;
  onCreationPresetChange: (preset: string) => void;
  onCreated: (agentId: string) => void;
  onDeleted: () => void;
}) {
  const {
    selected,
    presets,
    selectedPresetId,
    draft,
    creationPreset,
    onSelectPreset,
    onCreationPresetChange,
    onCreated,
    onDeleted,
  } = props;
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const revertTimer = useRef<number | null>(null);

  // Leave confirm mode when the selection changes or on unmount.
  useEffect(() => {
    setConfirming(false);
    return () => {
      if (revertTimer.current != null) window.clearTimeout(revertTimer.current);
    };
  }, [selected]);

  const create = async () => {
    const heavyWarning = heavyPresetWarning(creationPreset);
    if (
      heavyWarning &&
      !window.confirm(
        `Create a fresh agent with the "${creationPreset}" architecture?` +
          heavyWarning,
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (!draft.mindOnly && draft.elements.length > 0) {
        // Embodied: the server creates the agent and spawns a body bound to it.
        // replace=true supersedes any running body (single body slot) instead of
        // erroring, keeping the previous mind alive but bodiless.
        const status = await startEnvironment({
          elements: draft.elements,
          vision: draft.vision,
          audio: draft.audio,
          braces: draft.braces,
          replace: true,
          preset: creationPreset,
        });
        if (status.agent_id) onCreated(status.agent_id);
        else throw new Error("Body started but no agent id was returned.");
      } else {
        const id = await createAgent(creationPreset);
        onCreated(id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!selected) return;
    const name = window.prompt("Save name for this agent:");
    if (name == null) return; // cancelled
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Save name cannot be empty.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await saveAgent(selected, { name: trimmed });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!selected) return;
    if (!confirming) {
      setConfirming(true);
      if (revertTimer.current != null) window.clearTimeout(revertTimer.current);
      revertTimer.current = window.setTimeout(() => setConfirming(false), 3000);
      return;
    }
    if (revertTimer.current != null) window.clearTimeout(revertTimer.current);
    setConfirming(false);
    setBusy(true);
    setError(null);
    try {
      await deleteAgent(selected);
      onDeleted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const embodied = !draft.mindOnly && draft.elements.length > 0;

  return (
    <div className="controls">
      <select
        className="body-select"
        value={selectedPresetId ?? ""}
        disabled={busy}
        title="Choose a preset to load into the Environment tab. Edit it there, then start it with '+ New agent' or the Environment tab's Start button."
        onChange={(e) => {
          if (e.target.value) onSelectPreset(e.target.value);
        }}
      >
        {selectedPresetId == null && <option value="">Custom (unsaved)…</option>}
        {presets.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
      <span className="preset-picker create-preset-picker">
        <select
          value={creationPreset}
          disabled={busy}
          title="Neural network size for the next new agent"
          aria-label="Neural network size for the next new agent"
          onChange={(e) => onCreationPresetChange(e.target.value)}
        >
          {Object.entries(NEURAL_PRESET_LABELS).map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
        <Info tip={`New agent neural size. ${NEURAL_PRESET_INFO}`} />
      </span>
      <button
        className="btn start"
        disabled={busy}
        title={
          embodied
            ? "Spawn a fresh agent with a MuJoCo body from the current draft (Environment tab). Supersedes any running body."
            : "Spawn a fresh disembodied mind (no body until an adapter connects)"
        }
        onClick={() => void create()}
      >
        {busy ? "Starting…" : "+ New agent"}
      </button>
      {selected && (
        <button
          className="btn"
          disabled={busy || saving}
          title="Save this agent (brain + internal state + episodic memory) to the durable Saved Agents library"
          onClick={() => void save()}
        >
          {saving ? "Saving…" : "Save"}
        </button>
      )}
      {selected && (
        <button
          className={`btn ${confirming ? "danger" : "reset"}`}
          disabled={busy}
          title="Terminate this agent and remove it from the server"
          onClick={() => void remove()}
        >
          {confirming ? "Confirm delete?" : "Delete"}
        </button>
      )}
      <Info tip="Pick a preset to load its scenario into the Environment tab, tweak it there, then '+ New agent' spawns a fresh agent from the current draft (an embodied MuJoCo body unless the preset is 'Mind only'). There is a single body slot, so starting a new body supersedes a running one (the previous mind keeps living, just without a body). Delete terminates the selected agent permanently." />
      {error && <span className="ctrl-error">{error}</span>}
    </div>
  );
}
