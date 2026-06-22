import type { AgentState } from "../api";
import { usePersistentState } from "../usePersistentState";
import PerceptionPanel from "./PerceptionPanel";
import MindsEyePanel from "./MindsEyePanel";

type View = "camera" | "minds-eye";

/** Perception card with a Camera / Mind's Eye view switch: the same panel shows
 *  either the head-camera sensory view or the read-out of the modeled scene. */
export default function PerceptionCard(props: { agentId: string; state: AgentState }) {
  const { agentId, state } = props;
  const [view, setView] = usePersistentState<View>("decadic.perception.view", "camera");

  return (
    <div className="panel span-5">
      <div className="view-toggle" role="tablist" aria-label="Perception view">
        <button
          type="button"
          role="tab"
          aria-selected={view === "camera"}
          className={`seg${view === "camera" ? " active" : ""}`}
          onClick={() => setView("camera")}
        >
          Camera
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "minds-eye"}
          className={`seg${view === "minds-eye" ? " active" : ""}`}
          onClick={() => setView("minds-eye")}
        >
          Mind's Eye
        </button>
      </div>

      {view === "camera" ? (
        <PerceptionPanel agentId={agentId} state={state} embedded />
      ) : (
        <MindsEyePanel agentId={agentId} state={state} embedded />
      )}
    </div>
  );
}
