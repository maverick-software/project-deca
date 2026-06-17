import { useEffect, useState } from "react";
import type { AgentStatus } from "../api";
import { pauseAgent, resetAgent, resumeAgent, reviveAgent } from "../api";

/** Start / Stop / Reset / Revive controls for the selected neural net. */
export default function AgentControls(props: {
  agentId: string;
  paused: boolean;
  status?: AgentStatus;
}) {
  const { agentId } = props;
  const dead = props.status === "dead";
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Optimistic paused state so the button flips instantly; cleared once polling agrees.
  const [optimistic, setOptimistic] = useState<boolean | null>(null);
  const paused = optimistic ?? props.paused;

  useEffect(() => {
    setOptimistic(null);
  }, [props.paused, agentId]);

  const run = async (fn: (id: string) => Promise<void>, nextPaused: boolean | null) => {
    setBusy(true);
    setError(null);
    try {
      await fn(agentId);
      if (nextPaused !== null) setOptimistic(nextPaused);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="controls">
      {dead ? (
        <button
          className="btn revive"
          disabled={busy}
          title="Restore viability and resume the same mind (weights + memory retained)"
          onClick={() => run((id) => reviveAgent(id), null)}
        >
          &#10010; Revive
        </button>
      ) : paused ? (
        <button
          className="btn start"
          disabled={busy}
          title="Resume the cognitive cycle loop"
          onClick={() => run(resumeAgent, false)}
        >
          &#9654; Start
        </button>
      ) : (
        <button
          className="btn stop"
          disabled={busy}
          title="Pause the cognitive cycle loop (state and weights retained)"
          onClick={() => run(pauseAgent, true)}
        >
          &#10073;&#10073; Stop
        </button>
      )}
      <button
        className="btn reset"
        disabled={busy}
        title={
          dead
            ? "Reincarnate: a fresh mind with new weights and wiped memory"
            : "Fresh mind: new weights, zeroed state bus / viability, wiped episodic memory"
        }
        onClick={() => {
          const msg = dead
            ? "Reincarnate this neural net? A brand-new mind replaces the dead one."
            : "Reset this neural net? Weights, state, and memory will be wiped.";
          if (window.confirm(msg)) {
            void run(resetAgent, null);
          }
        }}
      >
        &#8635; {dead ? "Reincarnate" : "Reset"}
      </button>
      {dead && <span className="badge dead">dead</span>}
      {!dead && paused && <span className="badge paused">paused</span>}
      {error && <span className="ctrl-error">{error}</span>}
    </div>
  );
}
