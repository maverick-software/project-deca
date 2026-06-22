import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { AgentState, SceneEntitySnapshot } from "../api";
import { usePersistentState } from "../usePersistentState";
import { visionUrl } from "../api";
import Info from "./Info";

function toThree(p: number[]): [number, number, number] {
  return [p[0] ?? 0, p[2] ?? 0, -(p[1] ?? 0)];
}

function clamp01(v: number | null | undefined): number {
  if (typeof v !== "number" || !Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function entityPosition(e: SceneEntitySnapshot): number[] | null {
  if (e.relative && e.relative.length >= 3) return e.relative;
  if (e.depth != null && e.centroid_uv && e.centroid_uv.length >= 2) {
    const x = (e.centroid_uv[0] - 0.5) * Math.max(1, e.depth) * 2;
    const y = Math.max(0.2, e.depth);
    const z = (0.5 - e.centroid_uv[1]) * Math.max(1, e.depth);
    return [x, y, z];
  }
  return null;
}

function entityColor(e: SceneEntitySnapshot, focused: boolean): string {
  if (e.kind_hint === "body_part_candidate") return "#b982ff";
  if (e.kind_hint === "stuff") return "#69738a";
  if (focused) return "#7fc3ff";
  const err = clamp01((e.prediction_error ?? 0) * 2);
  const hue = Math.round(205 - 65 * err);
  return `hsl(${hue} 75% 68%)`;
}

function EntityGeometry({ entity }: { entity: SceneEntitySnapshot }) {
  if (entity.kind_hint === "stuff") return <boxGeometry args={[3.5, 0.03, 3.5]} />;
  if (entity.kind_hint === "body_part_candidate") return <capsuleGeometry args={[0.11, 0.28, 5, 10]} />;
  const size = 0.12 + 0.28 * clamp01(entity.confidence || entity.persistence || 0.5);
  return <sphereGeometry args={[size, 18, 18]} />;
}

function PredictionGhost({ entity }: { entity: SceneEntitySnapshot }) {
  const pos = entityPosition(entity);
  const pred = entity.predicted_relative ?? null;
  if (!pos || !pred || pred.length < 3) return null;
  const a = toThree(pos);
  const b = toThree(pred);
  const points = [new THREE.Vector3(...a), new THREE.Vector3(...b)];
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({
    color: "#ffd36a",
    transparent: true,
    opacity: 0.45,
  });
  return (
    <group>
      <primitive object={new THREE.Line(geom, mat)} />
      <mesh position={b}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshBasicMaterial color="#ffd36a" transparent opacity={0.35} />
      </mesh>
    </group>
  );
}

function FocusHalo({ radius }: { radius: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((s) => {
    if (!ref.current) return;
    const k = 1 + 0.08 * Math.sin(s.clock.elapsedTime * 5);
    ref.current.scale.set(k, k, k);
  });
  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius, radius + 0.035, 32]} />
      <meshBasicMaterial color="#7fc3ff" transparent opacity={0.72} side={THREE.DoubleSide} />
    </mesh>
  );
}

function SceneEntityMesh({ entity, focused }: { entity: SceneEntitySnapshot; focused: boolean }) {
  const pos = entityPosition(entity);
  if (!pos) return null;
  const p = toThree(pos);
  const color = entityColor(entity, focused);
  const salience = clamp01(entity.salience || entity.persistence || entity.confidence);
  const opacity = entity.occluded ? 0.22 : entity.kind_hint === "stuff" ? 0.18 : 0.35 + 0.6 * salience;
  const uncertainty = clamp01(entity.prediction_uncertainty ?? 0);
  const radius = 0.28 + 0.24 * uncertainty;

  return (
    <group position={p}>
      <mesh castShadow={entity.kind_hint !== "stuff"} receiveShadow={entity.kind_hint === "stuff"}>
        <EntityGeometry entity={entity} />
        <meshStandardMaterial
          color={color}
          emissive={focused ? new THREE.Color("#234f82") : new THREE.Color("#000000")}
          emissiveIntensity={focused ? 0.8 : 0}
          transparent
          opacity={opacity}
          wireframe={entity.occluded}
          depthWrite={!entity.occluded}
        />
      </mesh>
      {focused && <FocusHalo radius={radius} />}
      {entity.predicted_relative && <PredictionGhost entity={entity} />}
    </group>
  );
}

type Pose = { position: number[]; yaw: number; pitch: number };

function FirstPersonRig({ pose, enabled }: { pose: Pose; enabled: boolean }) {
  const { camera } = useThree();
  useFrame(() => {
    if (!enabled) return;
    const [sx, sy, sz] = pose.position;
    camera.position.set(sx, sz + 0.35, -sy);
    const fx = Math.cos(pose.yaw);
    const fy = Math.sin(pose.pitch);
    const fz = -Math.sin(pose.yaw);
    camera.lookAt(camera.position.x + fx, camera.position.y + fy, camera.position.z + fz);
  });
  return null;
}

function RenderedScene({
  entities,
  focusIds,
  pose,
  orbit,
}: {
  entities: SceneEntitySnapshot[];
  focusIds: Set<string>;
  pose: Pose;
  orbit: boolean;
}) {
  return (
    <>
      <color attach="background" args={["#0d111a"]} />
      <fog attach="fog" args={["#0d111a", 8, 48]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 24, 8]} intensity={0.9} />
      <Grid
        args={[80, 80]}
        cellSize={1}
        cellThickness={0.55}
        cellColor="#1d2536"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#2b3650"
        fadeDistance={45}
        fadeStrength={1.5}
        infiniteGrid
        position={[0, 0, 0]}
      />
      {entities.map((entity) => (
        <SceneEntityMesh
          key={entity.entity_id}
          entity={entity}
          focused={focusIds.has(entity.entity_id)}
        />
      ))}
      <FirstPersonRig pose={pose} enabled={!orbit} />
      {orbit && <OrbitControls makeDefault target={toThree(pose.position) as never} />}
    </>
  );
}

function RawLens({
  agentId,
  tick,
  entities,
  focusIds,
}: {
  agentId: string;
  tick: number;
  entities: SceneEntitySnapshot[];
  focusIds: Set<string>;
}) {
  const [ok, setOk] = useState(true);
  useEffect(() => setOk(true), [agentId]);
  const visible = entities.filter((e) => e.visible && e.centroid_uv && e.centroid_uv.length >= 2);
  return (
    <div className="me-lens me-raw-lens">
      <div className="me-lens-head">
        <span>Raw egocentric camera</span>
        <span>{visible.length} overlays</span>
      </div>
      <div className="me-camera-frame">
        <img
          src={visionUrl(agentId, tick, "egocentric")}
          alt="egocentric vision"
          onError={() => setOk(false)}
          onLoad={() => setOk(true)}
        />
        {!ok && <div className="me-no-frame">no vision frame</div>}
        {visible.map((entity) => {
          const uv = entity.centroid_uv ?? [0.5, 0.5];
          const focused = focusIds.has(entity.entity_id);
          const err = clamp01((entity.prediction_error ?? 0) * 2);
          return (
            <div
              key={entity.entity_id}
              className={`me-uv-marker ${focused ? "focused" : ""} ${entity.kind_hint}`}
              style={{
                left: `${uv[0] * 100}%`,
                top: `${uv[1] * 100}%`,
                opacity: 0.35 + 0.65 * clamp01(entity.confidence),
                boxShadow: `0 0 ${4 + err * 14}px rgba(255, 211, 106, ${err})`,
              }}
              title={`${entity.entity_id} ${entity.kind_hint}`}
            />
          );
        })}
      </div>
    </div>
  );
}

function SceneLatentStrip({ preview, rms }: { preview: (number | null)[]; rms: number | null }) {
  const values = preview.map((v) =>
    typeof v === "number" && Number.isFinite(v) ? v : 0,
  );
  const maxAbs = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const rmsFinite = typeof rms === "number" && Number.isFinite(rms) ? rms : null;
  return (
    <div style={{ marginTop: 8 }}>
      <div className="strip-label">
        <span>
          Scene latent
          <Info tip="Sub-symbolic persisting percept folded from the neural scene latent. The lenses above are the visual/object read-out; this strip is the latent trace." />
        </span>
        <span>{rmsFinite != null ? `rms ${rmsFinite.toFixed(3)}` : ""}</span>
      </div>
      <div className="heatstrip">
        {values.map((v, i) => {
          const n = maxAbs < 1e-9 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
          const light = Math.round(Math.abs(n) * 70 + 12);
          return (
            <div
              key={i}
              style={{ backgroundColor: n >= 0 ? `hsl(355 75% ${light}%)` : `hsl(215 75% ${light}%)` }}
              title={v.toFixed(4)}
            />
          );
        })}
      </div>
    </div>
  );
}

function moodGradient(viability: number, pain: number): string {
  const harm = Math.max(0, Math.min(1, 1 - viability / 100));
  const r = Math.round(180 + 60 * harm);
  const edge = 0.12 + 0.5 * harm + 0.3 * Math.min(1, pain);
  return `radial-gradient(ellipse at center, rgba(0,0,0,0) 45%, rgba(${r},40,50,${edge.toFixed(
    3,
  )}) 100%)`;
}

export default function MindsEyePanel(props: {
  agentId: string;
  state: AgentState;
  embedded?: boolean;
}) {
  const { agentId, state } = props;
  const [orbit, setOrbit] = usePersistentState("decadic.mindseye.orbit", false);
  const scene = state.perceptual.scene_workspace;
  const prediction = state.perceptual.scene_prediction;
  const workspace = state.perceptual.workspace_ignition;
  const wm = state.perceptual.working_memory;
  const entities = useMemo(() => scene?.entities ?? [], [scene]);
  const focusIds = useMemo(() => new Set(scene?.focus_ids ?? state.perceptual.focus?.ids ?? []), [scene, state.perceptual.focus]);

  const pos = state.perceptual.proprio_position ?? [0, 0, 1.4];
  const ori = state.perceptual.proprio_orientation ?? [0, 0, 0];
  const pose: Pose = { position: pos, yaw: ori[2] ?? 0, pitch: ori[1] ?? 0 };
  const viability = state.viability?.value ?? 100;
  const pain = state.state_bus?.B_pain_scalar ?? 0;
  const tick = state.perceptual.integration_ticks;
  const visible = entities.filter((e) => e.visible).length;
  const occluded = entities.filter((e) => e.occluded).length;

  const content = (
    <>
      <h2>
        Mind's Eye
        <Info tip="Dual lens into the agent: raw egocentric camera on the left, anonymous scene workspace rendered from the agent's own pose on the right. This is a read-only observer view; no labels or rewards feed cognition." />
      </h2>

      <div className="me-bar">
        <span className="strip-label">
          {visible} visible · {occluded} occluded · {focusIds.size} focused
        </span>
        <div style={{ flex: 1 }} />
        <button
          className="btn"
          title={orbit ? "Return to the agent's first-person rendered scene" : "Detach the camera to inspect the rendered scene"}
          onClick={() => setOrbit((v) => !v)}
        >
          {orbit ? "First-person" : "Inspect"}
        </button>
      </div>

      <div className="me-dual">
        <RawLens agentId={agentId} tick={tick} entities={entities} focusIds={focusIds} />
        <div className="me-lens">
          <div className="me-lens-head">
            <span>Rendered scene workspace</span>
            <span>{prediction?.model_active ? "learned dynamics" : "fallback dynamics"}</span>
          </div>
          <div className="me-stage">
            <Canvas shadows camera={{ fov: 75, near: 0.1, far: 100, position: [0, 3, 7] }}>
              <RenderedScene entities={entities} focusIds={focusIds} pose={pose} orbit={orbit} />
            </Canvas>
            <div className="me-vignette" style={{ background: moodGradient(viability, pain) }} />
            {!orbit && <div className="me-reticle" />}
            <div className="me-hud me-hud-tl">
              <div className="me-via">
                viability <b>{viability.toFixed(0)}</b>
              </div>
              {pain > 0.05 && <div className="me-pain">pain {pain.toFixed(2)}</div>}
            </div>
            <div className="me-hud me-hud-tr">
              <div>PE {prediction?.error != null ? prediction.error.toFixed(3) : "n/a"}</div>
              <div>unc {prediction?.uncertainty != null ? prediction.uncertainty.toFixed(3) : "n/a"}</div>
              <div>{workspace?.ignited ? "GWT ignited" : "GWT quiet"}</div>
            </div>
          </div>
        </div>
      </div>

      {wm?.scene_preview && wm.scene_preview.length > 0 && (
        <SceneLatentStrip preview={wm.scene_preview} rms={wm.scene_latent_rms ?? null} />
      )}

      <div className="me-legend">
        <span><i className="me-dot object" /> object</span>
        <span><i className="me-dot stuff" /> stuff</span>
        <span><i className="me-dot body" /> body candidate</span>
        <span><i className="me-dot focus" /> focused</span>
        <span><i className="me-dot ghost" /> occluded/predicted</span>
      </div>

      {entities.length === 0 && (
        <div className="empty">no scene workspace entities yet; connect a vision body in discovered perception</div>
      )}
    </>
  );

  if (props.embedded) return content;
  return <div className="panel span-7">{content}</div>;
}
