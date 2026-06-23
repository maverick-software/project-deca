import { useEffect, useRef, useState } from "react";
import type { AgentState } from "../api";
import { audioUrl, openBodyViewer, recenterBody, visionUrl } from "../api";
import { usePersistentState } from "../usePersistentState";
import Info from "./Info";

const CAMERA_LABELS: Record<string, string> = {
  egocentric: "Egocentric (head)",
  track: "Follow — back",
  front: "Follow — front",
  side: "Follow — side",
  top: "Top-down",
};

function Bars(props: { values: number[]; scale?: number }) {
  const scale =
    props.scale ?? props.values.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-6);
  return (
    <div className="bars">
      {props.values.map((v, i) => (
        <div
          key={i}
          title={v.toFixed(3)}
          style={{ height: `${Math.min(100, (Math.abs(v) / scale) * 100)}%` }}
        />
      ))}
    </div>
  );
}

export default function PerceptionPanel(props: {
  agentId: string;
  state: AgentState;
  embedded?: boolean;
}) {
  const { agentId, state } = props;
  const p = state.perceptual;
  const [visionOk, setVisionOk] = useState(true);
  const [camera, setCamera] = usePersistentState("decadic.perception.camera", "track");
  const [recentering, setRecentering] = useState(false);
  const [viewerBusy, setViewerBusy] = useState(false);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // integration_ticks advances with each observation; reusing it busts the img cache
  const tick = p.integration_ticks;

  const views = state.vision_views?.length ? state.vision_views : ["egocentric"];

  useEffect(() => {
    setVisionOk(true);
  }, [agentId, camera]);

  // If the selected camera disappears (e.g. agent swap), fall back to the
  // preferred follow-back view if present, else the first available view.
  useEffect(() => {
    if (!views.includes(camera)) {
      setCamera(views.includes("track") ? "track" : views[0]);
    }
  }, [views, camera]);

  const onListen = () => {
    const el = audioRef.current;
    if (!el) return;
    el.src = audioUrl(agentId, tick);
    setPlaying(true);
    void el.play().catch(() => setPlaying(false));
  };

  const onRecenter = async () => {
    setRecentering(true);
    try {
      await recenterBody(agentId);
    } catch {
      // body not connected; nothing to recenter
    } finally {
      setTimeout(() => setRecentering(false), 600);
    }
  };

  const onOpenViewer = async () => {
    setViewerBusy(true);
    setViewerError(null);
    try {
      await openBodyViewer(agentId);
    } catch (err) {
      setViewerError(err instanceof Error ? err.message : "Unable to open live window");
    } finally {
      setTimeout(() => setViewerBusy(false), 600);
    }
  };

  const content = (
    <>
      <h2>
        Perception
        <Info tip="The agent's senses, as last integrated into its perceptual state: egocentric vision, body proprioception (joints + touch), and the world-graph of things it knows about around it." />
      </h2>

      <div className="camera-bar">
        <select value={camera} onChange={(e) => setCamera(e.target.value)}>
          {views.map((v) => (
            <option key={v} value={v}>
              {CAMERA_LABELS[v] ?? v}
            </option>
          ))}
        </select>
        <Info tip="Camera to display. 'Egocentric' is the brain's own eye (what gets encoded into perception); the others are spectator cameras that track the body for your benefit only." />
        <div style={{ flex: 1 }} />
        <button
          className="btn"
          disabled={recentering}
          title="Teleport the body back to the stage origin (props stay put)"
          onClick={() => void onRecenter()}
        >
          &#8982; Recenter body
        </button>
        <button
          className="btn"
          disabled={viewerBusy}
          title="Open the native MuJoCo viewer window (opens on the machine running the body). Closing it won't stop the agent."
          onClick={() => void onOpenViewer()}
        >
          &#10697; Live window
        </button>
      </div>

      <div className="strip-label" style={{ marginBottom: 4 }}>
        <span>{CAMERA_LABELS[camera] ?? camera}</span>
        <span>{p.vision_resolution ? p.vision_resolution.join("×") : ""}</span>
      </div>
      {/* Keep the img mounted even after an error: each tick changes the src,
          so the browser retries and recovers once frames are available again. */}
      <img
        className="vision-img"
        style={visionOk ? undefined : { display: "none" }}
        src={visionUrl(agentId, tick, camera)}
        alt={`${camera} vision`}
        onError={() => setVisionOk(false)}
        onLoad={() => setVisionOk(true)}
      />
      {!visionOk && (
        <div className="empty">
          no vision frames yet (run the body adapter with <code>--vision</code>)
        </div>
      )}
      {viewerError && <div className="empty">{viewerError}</div>}

      {p.scene_health && (
        <>
          <div className="strip-label" style={{ marginTop: 12 }}>
            <span>
              Scene Workspace
              <Info tip="Persistent anonymous scene model built before cognition: visible, occluded, and focused scene entities. Labels never enter this layer." />
            </span>
            <span>
              {p.scene_health.visible_count} visible / {p.scene_health.entity_count} total
            </span>
          </div>
          <div className="stat-grid mini">
            <div>
              <strong>{p.scene_health.focus_count}</strong>
              <span>focus</span>
            </div>
            <div>
              <strong>{p.scene_health.candidate_count ?? 0}</strong>
              <span>candidates</span>
            </div>
            <div>
              <strong>{p.scene_health.occluded_count}</strong>
              <span>occluded</span>
            </div>
            <div>
              <strong>{p.scene_health.stable_count}</strong>
              <span>stable</span>
            </div>
            <div>
              <strong>
                {p.scene_health.prediction_error != null
                  ? p.scene_health.prediction_error.toFixed(3)
                  : "n/a"}
              </strong>
              <span>scene PE</span>
            </div>
            <div>
              <strong>{p.scene_health.reidentified_count ?? 0}</strong>
              <span>re-id</span>
            </div>
            <div>
              <strong>{p.scene_health.prediction_unstable_count ?? 0}</strong>
              <span>unstable</span>
            </div>
          </div>
          {p.scene_prediction && (
            <div className="strip-label" style={{ marginTop: 8 }}>
              <span>
                Scene dynamics: {p.scene_prediction.model_active ? "learned" : "fallback"}
              </span>
              <span>
                {p.scene_prediction.prediction_count ?? 0} predicted · loss{" "}
                {p.scene_prediction.loss != null ? p.scene_prediction.loss.toFixed(4) : "n/a"}
              </span>
            </div>
          )}
          {p.scene_health.active_drive_deficits && (
            <div className="strip-label" style={{ marginTop: 8 }}>
              <span>
                Drive attention: energy {Number(p.scene_health.active_drive_deficits.energy ?? 0).toFixed(2)} /
                hydration {Number(p.scene_health.active_drive_deficits.hydration ?? 0).toFixed(2)} /
                integrity {Number(p.scene_health.active_drive_deficits.integrity ?? 0).toFixed(2)}
              </span>
              <span>{String(p.scene_health.active_drive_deficits.priority ?? "explore")}</span>
            </div>
          )}
          {p.workspace_ignition && (
            <div className="strip-label" style={{ marginTop: 8 }}>
              <span>Global workspace</span>
              <span>
                {p.workspace_ignition.ignited ? "ignited" : "not ignited"} ·{" "}
                {p.workspace_ignition.n_candidates} candidates
              </span>
            </div>
          )}
        </>
      )}

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Hearing (RMS)
          <Info tip="Loudness of the last audio window the agent heard — a procedural soundscape of its own footsteps, impacts, growls, and chimes synthesized from physics. Encoded by frozen Whisper when pretrained encoders are enabled. Press Listen to hear it yourself." />
        </span>
        <span>
          {p.audio_duration_s != null ? `${p.audio_duration_s.toFixed(1)}s window` : "no audio"}
        </span>
      </div>
      <div className="audio-row">
        <div className="gauge-track" style={{ flex: 1 }}>
          <div
            className="gauge-fill"
            style={{
              width: `${Math.min(100, ((p.audio_rms ?? 0) / 0.25) * 100)}%`,
              backgroundColor: "var(--accent)",
            }}
          />
        </div>
        <button
          className="btn"
          disabled={p.audio_duration_s == null || playing}
          title="Play the agent's most recent audio window"
          onClick={onListen}
        >
          {playing ? "..." : "\u25B8 Listen"}
        </button>
        <audio ref={audioRef} onEnded={() => setPlaying(false)} onError={() => setPlaying(false)} />
      </div>

      <div className="strip-label" style={{ marginTop: 12 }}>
        <span>
          Joints (qpos/qvel)
          <Info tip="Joint proprioception streamed from the body: each bar is one joint angle (qpos) or joint velocity (qvel). Bar height = magnitude relative to the largest. This is how the agent 'feels' its own posture and movement." />
        </span>
        <span>{p.proprio_joints?.length ?? 0}</span>
      </div>
      {p.proprio_joints && p.proprio_joints.length > 0 ? (
        <Bars values={p.proprio_joints} />
      ) : (
        <div className="empty">no joint proprioception</div>
      )}

      <div className="strip-label" style={{ marginTop: 10 }}>
        <span>
          Touch contacts (N)
          <Info tip="Force in Newtons on the body's touch sensors — left sole, right sole, left palm, right palm. Standing puts ~300 N on each foot; only forces well above normal footfall register as damaging collisions." />
        </span>
        <span>{p.proprio_contacts?.map((c) => c.toFixed(0)).join(" · ") ?? "—"}</span>
      </div>
      {p.proprio_contacts && p.proprio_contacts.length > 0 && (
        <Bars values={p.proprio_contacts} scale={400} />
      )}

      <div className="strip-label" style={{ marginTop: 12, marginBottom: 4 }}>
        <span>
          Egocentric nodes
          <Info tip="The agent's world-graph, centered on itself: SELF is its own body, ENTITY rows are objects it perceives (with positions relative to it, in meters), CONTEXT rows are situational facts like body posture or current region." />
        </span>
        <span>{p.egocentric_nodes.length}</span>
      </div>
      <div className="nodes">
        {p.egocentric_nodes.slice(0, 10).map((n, i) => (
          <div className="node-row" key={i}>
            <span className={`node-role ${n.role}`}>{n.role}</span>
            <span>{n.kind ?? n.id}</span>
            {n.relative && (
              <span style={{ color: "var(--text-dim)" }}>
                rel [{n.relative.map((x) => x.toFixed(1)).join(", ")}]
              </span>
            )}
            {n.standing !== undefined && (
              <span style={{ color: "var(--text-dim)" }}>
                {n.standing ? "standing" : "fallen"}
                {n.moving ? " · moving" : ""}
              </span>
            )}
          </div>
        ))}
      </div>
    </>
  );

  if (props.embedded) return content;
  return <div className="panel span-5">{content}</div>;
}
