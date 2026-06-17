import { useState } from "react";
import type { AgentDefaults, CapacityConfig, Metrics } from "../api";
import { configureAgent } from "../api";
import Info from "./Info";

// All cognitive-faculty + observation toggles, surfaced in the UI so nothing is
// controlled only by launch flags. The three faculties (perception-feedback loop,
// perception mode, sensory encoder) change the model's architecture, so toggling
// one rebuilds the brain with fresh weights (episodic + working memory survive).
// The observation toggles (cognition trace, probe capture) apply live.
export default function CognitionTogglesPanel(props: {
  agentId: string | null;
  capacity?: CapacityConfig;
  metrics: Metrics | null;
  defaults: AgentDefaults | null;
  onCommitDefaults: (partial: Partial<AgentDefaults>) => void;
}) {
  const { agentId, capacity, metrics, defaults, onCommitDefaults } = props;
  const [busy, setBusy] = useState(false);

  const cap = capacity;
  const perceptionMode = cap?.perception_mode ?? metrics?.perception_mode ?? "discovered";
  const encoderMode = cap?.encoder_mode ?? "hf";
  const feedback = cap?.perception_feedback ?? metrics?.perception_feedback ?? true;
  const trace = cap?.cognition_trace ?? true;
  const probe = cap?.probe_capture ?? false;
  const episodicAsync = cap?.episodic_async ?? true;
  const ltmAsync = cap?.ltm_async ?? true;

  const commit = async (patch: Partial<CapacityConfig>) => {
    if (!agentId) return;
    setBusy(true);
    try {
      await configureAgent(agentId, patch);
    } catch {
      // server unreachable; optimistic UI, next poll re-syncs
    } finally {
      setBusy(false);
    }
  };

  // Discovered perception needs real CLIP patch tokens, so selecting it also
  // forces the encoder to hf (the inert discovered+zeros combo is never sent).
  const onPerceptionMode = (mode: string) => {
    const patch: Partial<CapacityConfig> = { perception_mode: mode };
    if (mode === "discovered" && encoderMode !== "hf") patch.encoder_mode = "hf";
    void commit(patch);
  };

  const discoveredInert = perceptionMode === "discovered" && encoderMode !== "hf";

  return (
    <div className="defaults-section">
      <h3 className="cap-subhead">
        Cognition faculties
        <Info tip="The inherent mechanisms of the mind. Each changes the model's architecture, so toggling one rebuilds the brain with fresh weights (like Reset) — episodic and working memory are kept. Defaults below apply to NEW agents; the live controls apply to the selected agent." />
        {busy && <span className="strip-label">rebuilding…</span>}
      </h3>

      <div className="strip-label">New agent defaults</div>

      <label className="cap-row toggle-row">
        <span>
          Perception-feedback loop
          <Info tip="Top-down predictive perception: a learned prediction of the percept (from history) is blended with the bottom-up senses under a learned precision gate, plus perceptual-similarity episodic recall. Core faculty; default on." />
        </span>
        <input
          type="checkbox"
          checked={defaults?.perception_feedback ?? true}
          disabled={!defaults}
          onChange={(e) => onCommitDefaults({ perception_feedback: e.target.checked })}
        />
      </label>

      <label className="cap-row">
        <span>
          Perception mode
          <Info tip="discovered: the world graph emerges from the agent's own camera (slot-attention object discovery) + proprioception + memory. oracle: the simulator hands the agent the entity graph (eval scaffold). Discovered needs the hf encoder." />
        </span>
        <select
          value={defaults?.perception_mode ?? "discovered"}
          disabled={!defaults}
          onChange={(e) => onCommitDefaults({ perception_mode: e.target.value })}
        >
          <option value="discovered">discovered</option>
          <option value="oracle">oracle</option>
        </select>
      </label>

      <label className="cap-row">
        <span>
          Sensory encoder
          <Info tip="hf: real frozen CLIP + Whisper (downloads ~1 GB on first run; gives the patch tokens discovered perception needs). zeros: cheap synthetic fallback (fast, but discovered perception is inert)." />
        </span>
        <select
          value={defaults?.encoder_mode ?? "hf"}
          disabled={!defaults}
          onChange={(e) => onCommitDefaults({ encoder_mode: e.target.value })}
        >
          <option value="hf">hf (CLIP + Whisper)</option>
          <option value="zeros">zeros (synthetic)</option>
        </select>
      </label>

      {!agentId && (
        <div className="strip-label">Select an agent for live faculty controls.</div>
      )}

      {agentId && (
        <>
          <div className="strip-label">
            Selected agent (rebuilds the mind on change)
          </div>

          <label className="cap-row toggle-row">
            <span>
              Perception-feedback loop
              <Info tip="Toggle the top-down predictive-perception loop for this agent. Rebuilds the brain with fresh weights." />
            </span>
            <input
              type="checkbox"
              checked={feedback}
              onChange={(e) => void commit({ perception_feedback: e.target.checked })}
            />
          </label>

          <label className="cap-row">
            <span>Perception mode</span>
            <select value={perceptionMode} onChange={(e) => onPerceptionMode(e.target.value)}>
              <option value="discovered">discovered (from perception)</option>
              <option value="oracle">oracle (sim-given)</option>
            </select>
          </label>

          <label className="cap-row">
            <span>Sensory encoder</span>
            <select
              value={encoderMode}
              onChange={(e) => void commit({ encoder_mode: e.target.value })}
            >
              <option value="hf">hf (CLIP + Whisper)</option>
              <option value="zeros">zeros (synthetic)</option>
            </select>
          </label>

          {discoveredInert && (
            <div className="strip-label warn">
              Discovered perception is inert with the zeros encoder — switch the
              encoder to hf for it to do anything.
            </div>
          )}

          <div className="strip-label">Observation (live; never feeds cognition)</div>

          <label className="cap-row toggle-row">
            <span>
              Cognition trace
              <Info tip="Assemble the per-cycle, human-readable 'why' explanation. Read-only; turning it off saves a little compute." />
            </span>
            <input
              type="checkbox"
              checked={trace}
              onChange={(e) => void commit({ cognition_trace: e.target.checked })}
            />
          </label>

          <label className="cap-row toggle-row">
            <span>
              Probe capture
              <Info tip="Append {latents, targets} rows to the interpretability-probe training file each cycle. Developer data capture; off by default." />
            </span>
            <input
              type="checkbox"
              checked={probe}
              onChange={(e) => void commit({ probe_capture: e.target.checked })}
            />
          </label>

          <div className="strip-label">Performance (live)</div>

          <label className="cap-row toggle-row">
            <span>
              Async episodic memory
              <Info tip="Write-behind persistence: the per-cycle episodic record is saved to SQLite on a background worker so the disk commit never blocks the cognitive cycle. No memory is lost (it falls back to a synchronous write under backpressure). Default on; turn off to write each episode synchronously inside the cycle." />
            </span>
            <input
              type="checkbox"
              checked={episodicAsync}
              onChange={(e) => void commit({ episodic_async: e.target.checked })}
            />
          </label>

          <label className="cap-row toggle-row">
            <span>
              Async LTM consolidation
              <Info tip="Write-behind consolidation: stage 10's working-memory -> long-term-graph commit runs on a background worker so the SQLite fsync never blocks the cognitive cycle. No consolidation is lost (order-preserving synchronous fallback under backpressure). The graph is read on the next cycle, so the ~one-cycle visibility lag is immaterial. Default on; turn off to consolidate synchronously inside the cycle." />
            </span>
            <input
              type="checkbox"
              checked={ltmAsync}
              onChange={(e) => void commit({ ltm_async: e.target.checked })}
            />
          </label>
        </>
      )}
    </div>
  );
}
