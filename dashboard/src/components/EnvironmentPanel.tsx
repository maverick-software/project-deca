import { useEffect, useState } from "react";
import {
  deleteEnvironment,
  fetchEnvironment,
  pauseEnvironment,
  resumeEnvironment,
  setParentPaused as apiSetParentPaused,
  startEnvironment,
  stopEnvironment,
  type AgentPreset,
  type EnvironmentStatus,
  type ScenarioDraft,
} from "../api";
import { heavyPresetWarning } from "../neuralPresets";
import { usePolling } from "../usePolling";
import Info from "./Info";

const ELEMENT_LABELS: Record<string, string> = {
  house: "House",
  food: "Food",
  water: "Water",
  bear: "Bear (threat)",
  ball: "Ball",
  obstacles: "Obstacles",
  npc: "Parent (NPC)",
  crowd: "Village (8 NPCs)",
};

/**
 * Compose a scenario and start / pause / stop / delete the body+world.
 *
 * The editable config is the shared "draft" owned by App and selected from the
 * top-bar preset dropdown; this tab edits it (elements, senses, manual braces)
 * and can either Save it as a new preset or Start it now. When a body is
 * running the same fields show the live config read-only.
 */
export default function EnvironmentPanel(props: {
  draft: ScenarioDraft;
  updateDraft: (patch: Partial<ScenarioDraft>) => void;
  selectedPreset: AgentPreset | null;
  creationPreset: string;
  onSavePreset: (name: string) => Promise<AgentPreset>;
  onStarted?: (agentId: string) => void;
}) {
  const { draft, updateDraft, selectedPreset, onSavePreset } = props;
  const { data: env } = usePolling<EnvironmentStatus>(fetchEnvironment, 1000);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Parent-freeze state is tracked locally: the body process resets it to
  // "walking" on every (re)start, so we clear it whenever the agent changes.
  const [parentFrozen, setParentFrozen] = useState(false);

  const available =
    env?.available_elements && env.available_elements.length > 0
      ? env.available_elements
      : Object.keys(ELEMENT_LABELS);
  const running = env?.running ?? false;
  const paused = env?.paused ?? false;
  const state = env?.state ?? "stopped";
  const agentId = env?.agent_id ?? null;
  const hasParent = running && !!env?.elements?.includes("npc");
  const editable = !running && !busy;
  const mindOnly = draft.mindOnly;

  useEffect(() => {
    setParentFrozen(false);
  }, [agentId, running]);

  const toggleParentFrozen = async () => {
    if (!agentId) return;
    const next = !parentFrozen;
    setBusy(true);
    setError(null);
    try {
      await apiSetParentPaused(agentId, next);
      setParentFrozen(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggleElement = (el: string) => {
    const has = draft.elements.includes(el);
    updateDraft({
      elements: has ? draft.elements.filter((x) => x !== el) : [...draft.elements, el],
      // Choosing world elements implies an embodied scenario.
      mindOnly: false,
    });
  };

  const run = async (
    fn: () => Promise<EnvironmentStatus>,
    onOk?: (s: EnvironmentStatus) => void,
  ) => {
    setBusy(true);
    setError(null);
    try {
      const s = await fn();
      onOk?.(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const confirmCreationPreset = () => {
    const heavyWarning = heavyPresetWarning(props.creationPreset);
    return (
      !heavyWarning ||
      window.confirm(
        `Create a fresh agent with the "${props.creationPreset}" architecture?` +
          heavyWarning,
      )
    );
  };

  const onStart = () => {
    if (!confirmCreationPreset()) return;
    void run(
      () =>
        startEnvironment({
          elements: draft.elements,
          vision: draft.vision,
          audio: draft.audio,
          braces: draft.braces,
          preset: props.creationPreset,
        }),
      (s) => {
        if (s.agent_id) props.onStarted?.(s.agent_id);
      },
    );
  };

  const onSave = async () => {
    const name = window.prompt("Name this preset:");
    if (name == null) return; // cancelled
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Preset name cannot be empty.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onSavePreset(trimmed);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Live (running) values fall back to the draft when stopped.
  const showVision = running ? !!env?.options?.vision : draft.vision;
  const showAudio = running ? !!env?.options?.audio : draft.audio;
  const showBraces = running ? env?.options?.braces === true : draft.braces;

  return (
    <div className="panel env-panel">
      <h2>
        Environment
        <Info tip="Compose a scenario from world elements, senses, and optional manual braces, then either Save it as a preset or Start it now. Braces default off; use them only as a manual body scaffold." />
      </h2>

      <div className={`env-status env-${state}`}>
        <span className="env-dot" />
        <b>{state}</b>
        {env?.agent_id && (
          <span className="env-meta">agent {env.agent_id.slice(0, 8)}</span>
        )}
        {env?.pid != null && <span className="env-meta">pid {env.pid}</span>}
        {running && env?.elements?.length ? (
          <span className="env-meta">{env.elements.join(" + ")}</span>
        ) : null}
      </div>

      {!running && (
        <div className="env-editing">
          Editing preset: <b>{selectedPreset?.name ?? "Custom (unsaved)"}</b>
        </div>
      )}

      {!running && mindOnly ? (
        <div className="env-warn">
          This preset is <b>Mind only</b> (no body). Spawn it with{" "}
          <b>+ New agent</b> in the top bar, or pick an element below to make it an
          embodied scenario.
        </div>
      ) : null}

      <div className="env-section-label">Elements</div>
      <div className="env-chips">
        {available.map((el) => {
          const on = running ? !!env?.elements.includes(el) : draft.elements.includes(el);
          return (
            <label key={el} className={`env-chip ${on ? "on" : ""}`}>
              <input
                type="checkbox"
                checked={on}
                disabled={!editable}
                onChange={() => toggleElement(el)}
              />
              {ELEMENT_LABELS[el] ?? el}
            </label>
          );
        })}
      </div>

      <div className="env-section-label">Senses</div>
      <div className="env-chips">
        <label className={`env-chip ${showVision ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={showVision}
            disabled={!editable}
            onChange={(e) => updateDraft({ vision: e.target.checked })}
          />
          Vision
        </label>
        <label className={`env-chip ${showAudio ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={showAudio}
            disabled={!editable}
            onChange={(e) => updateDraft({ audio: e.target.checked })}
          />
          Audio
        </label>
      </div>

      <div className="env-section-label">
        Joint braces
        <Info tip="Whether the manual joint-brace orthosis is engaged when the body spawns. Off is the default free body. On starts scaffolded for debugging or setup; Skill Dojo does not command braces." />
      </div>
      <div className="env-chips">
        <label className={`env-chip ${showBraces ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={showBraces}
            disabled={!editable}
            onChange={(e) => updateDraft({ braces: e.target.checked })}
          />
          Braces engaged at spawn
        </label>
      </div>

      <div className="env-controls">
        {!running ? (
          <>
            <button
              className="btn start"
              disabled={busy || mindOnly || draft.elements.length === 0}
              title={
                mindOnly
                  ? "Mind-only presets have no body; use + New agent in the top bar"
                  : "Create an agent and spawn the body bound to it from this config"
              }
              onClick={onStart}
            >
              &#9654; Start now
            </button>
            <button
              className="btn"
              disabled={busy || (!mindOnly && draft.elements.length === 0)}
              title="Save this config as a named preset in the top-bar dropdown"
              onClick={() => void onSave()}
            >
              &#128190; Save as preset
            </button>
          </>
        ) : paused ? (
          <button
            className="btn start"
            disabled={busy}
            title="Resume brain and world"
            onClick={() => void run(resumeEnvironment)}
          >
            &#9654; Resume
          </button>
        ) : (
          <button
            className="btn stop"
            disabled={busy}
            title="Freeze brain and world together"
            onClick={() => void run(pauseEnvironment)}
          >
            &#10073;&#10073; Pause
          </button>
        )}

        {running && (
          <button
            className="btn reset"
            disabled={busy}
            title="Terminate the body process (the agent/brain is kept)"
            onClick={() => void run(stopEnvironment)}
          >
            &#9632; Stop
          </button>
        )}

        {hasParent && (
          <button
            className={`btn ${parentFrozen ? "start" : "reset"}`}
            disabled={busy || !agentId}
            title={
              parentFrozen
                ? "Let the parent walk, forage, and offer again"
                : "Freeze the parent where it stands (stops moving, foraging, and offering)"
            }
            onClick={() => void toggleParentFrozen()}
          >
            {parentFrozen ? "\u25B6 Resume parent" : "\u23F8 Pause parent"}
          </button>
        )}

        <button
          className="btn reset"
          disabled={busy || (!running && !env?.agent_id)}
          title="Stop the body and delete the bound agent (the whole scenario)"
          onClick={() => {
            if (
              window.confirm(
                "Delete the scenario? The body stops and its agent (brain) is removed.",
              )
            ) {
              void run(deleteEnvironment);
            }
          }}
        >
          &#128465; Delete scenario
        </button>

        {busy && <span className="strip-label">working…</span>}
        {error && <span className="ctrl-error">{error}</span>}
      </div>

      {state === "crashed" && (
        <div className="env-warn">
          The body process exited unexpectedly (returncode {env?.returncode}).
          {env?.log_path ? ` See ${env.log_path}.` : ""} A common cause is a wrong
          server port: set DECADIC_SELF_PORT to the port this server runs on.
        </div>
      )}
    </div>
  );
}
