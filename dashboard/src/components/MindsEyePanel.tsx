import { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { AgentState, WorkingMemorySlot } from "../api";
import { usePersistentState } from "../usePersistentState";
import Info from "./Info";

// Sim is z-up (x east, y north, z up); three.js is y-up. Map (x,y,z) -> (x, z, -y).
function toThree(p: number[]): [number, number, number] {
  return [p[0], p[2], -p[1]];
}

type Shape = "capsule" | "sphere" | "box" | "cylinder" | "octa";
type Proto = { color: string; shape: Shape; size: [number, number, number] };

// Working memory stores a kind label, not geometry, so we render canonical prototypes.
const PROTO: Record<string, Proto> = {
  bear: { color: "#7a4a22", shape: "capsule", size: [0.34, 0.9, 0] },
  food: { color: "#f28c1a", shape: "sphere", size: [0.16, 0, 0] },
  box: { color: "#c0392b", shape: "box", size: [0.5, 0.5, 0.5] },
  sphere: { color: "#2ecc71", shape: "sphere", size: [0.4, 0, 0] },
  pillar: { color: "#2f6fdb", shape: "cylinder", size: [0.3, 1.2, 0] },
};
const DEFAULT_PROTO: Proto = { color: "#8a8f9c", shape: "octa", size: [0.3, 0, 0] };

function Geometry({ proto }: { proto: Proto }) {
  const [a, b] = proto.size;
  switch (proto.shape) {
    case "capsule":
      return <capsuleGeometry args={[a, b, 6, 12]} />;
    case "box":
      return <boxGeometry args={proto.size} />;
    case "cylinder":
      return <cylinderGeometry args={[a, a, b, 16]} />;
    case "octa":
      return <octahedronGeometry args={[a]} />;
    default:
      return <sphereGeometry args={[a, 18, 18]} />;
  }
}

function EntityMesh({ slot }: { slot: WorkingMemorySlot }) {
  const proto = PROTO[slot.kind] ?? DEFAULT_PROTO;
  if (!slot.position) return null;
  const [x, y, z] = toThree(slot.position);

  const salience = Math.max(0, Math.min(1, slot.salience));
  const affect = slot.affective_weight ?? 0;
  const audio = slot.audio_intensity ?? 0;
  const inView = slot.in_view;

  const emissive = useMemo(() => {
    if (affect < 0) return new THREE.Color(1.0, 0.18, 0.18); // threat
    if (affect > 0) return new THREE.Color(0.2, 1.0, 0.4); // reward
    return new THREE.Color(0, 0, 0);
  }, [affect]);

  const opacity = 0.2 + 0.8 * salience;
  const emissiveIntensity = Math.min(1, Math.abs(affect)) * 0.9;

  return (
    <group position={[x, y, z]}>
      <mesh castShadow>
        <Geometry proto={proto} />
        <meshStandardMaterial
          color={proto.color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={opacity}
          wireframe={!inView}
          depthWrite={inView}
        />
      </mesh>
      {audio > 0.05 && <AudioPulse intensity={audio} />}
      {slot.heading != null && <HeadingArrow yaw={slot.heading} />}
    </group>
  );
}

function AudioPulse({ intensity }: { intensity: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((s) => {
    if (!ref.current) return;
    const k = 1 + 0.15 * Math.sin(s.clock.elapsedTime * 6);
    ref.current.scale.set(k, k, k);
  });
  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
      <ringGeometry args={[0.5 + intensity * 0.3, 0.62 + intensity * 0.4, 24]} />
      <meshBasicMaterial color="#ffd36a" transparent opacity={0.25 + 0.5 * intensity} side={THREE.DoubleSide} />
    </mesh>
  );
}

function HeadingArrow({ yaw }: { yaw: number }) {
  // sim yaw in XY -> rotate cone about three Y so it points along heading
  return (
    <mesh position={[Math.cos(yaw) * 0.6, 0.05, -Math.sin(yaw) * 0.6]} rotation={[0, -yaw + Math.PI / 2, -Math.PI / 2]}>
      <coneGeometry args={[0.08, 0.25, 8]} />
      <meshBasicMaterial color="#cfd6e6" transparent opacity={0.7} />
    </mesh>
  );
}

type Pose = { position: number[]; yaw: number; pitch: number };

function FirstPersonRig({ pose, enabled }: { pose: Pose; enabled: boolean }) {
  const { camera } = useThree();
  useFrame(() => {
    if (!enabled) return;
    const [sx, sy, sz] = pose.position;
    camera.position.set(sx, sz + 0.4, -sy);
    const fx = Math.cos(pose.yaw);
    const fy = Math.sin(pose.pitch);
    const fz = -Math.sin(pose.yaw);
    camera.lookAt(camera.position.x + fx, camera.position.y + fy, camera.position.z + fz);
  });
  return null;
}

function Scene({ slots, pose, orbit }: { slots: WorkingMemorySlot[]; pose: Pose; orbit: boolean }) {
  return (
    <>
      <color attach="background" args={["#0d111a"]} />
      <fog attach="fog" args={["#0d111a", 7, 48]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 24, 8]} intensity={0.85} />
      <Grid
        args={[80, 80]}
        cellSize={1}
        cellThickness={0.6}
        cellColor="#1d2536"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#2b3650"
        fadeDistance={45}
        fadeStrength={1.5}
        infiniteGrid
        position={[0, 0, 0]}
      />
      {slots.map((s) => (
        <EntityMesh key={s.entity_id} slot={s} />
      ))}
      <FirstPersonRig pose={pose} enabled={!orbit} />
      {orbit && <OrbitControls makeDefault target={toThree(pose.position) as never} />}
    </>
  );
}

/** Signed value → blue (negative) / dark (zero) / red (positive). */
function heatColor(v: number, maxAbs: number): string {
  if (maxAbs < 1e-9) return "#1d2230";
  const n = Math.max(-1, Math.min(1, v / maxAbs));
  const intensity = Math.round(Math.abs(n) * 70 + 12);
  return n >= 0 ? `hsl(355 75% ${intensity}%)` : `hsl(215 75% ${intensity}%)`;
}

function SceneLatentStrip({ preview, rms }: { preview: (number | null)[]; rms: number | null }) {
  // The server can serialize NaN/None as null (unstable neural state); coerce to
  // finite numbers so a single bad bucket can't crash the whole Mind's Eye panel.
  const values = preview.map((v) =>
    typeof v === "number" && Number.isFinite(v) ? v : 0,
  );
  const maxAbs = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const rmsFinite = typeof rms === "number" && Number.isFinite(rms) ? rms : null;
  return (
    <div style={{ marginTop: 8 }}>
      <div className="strip-label">
        <span>
          Scene latent (persisting percept)
          <Info tip="The literal image in the neural layer: every cycle the pooled multi-frame percept (vision + audio + body, fused by the frozen encoders) is EMA-blended into this vector, so it persists and drifts across cycles instead of being replaced. The 3D scene above is the symbolic read-out; this strip is the sub-symbolic one. Folded to 32 buckets for display." />
        </span>
        <span>{rmsFinite != null ? `rms ${rmsFinite.toFixed(3)}` : ""}</span>
      </div>
      <div className="heatstrip">
        {values.map((v, i) => (
          <div key={i} style={{ backgroundColor: heatColor(v, maxAbs) }} title={v.toFixed(4)} />
        ))}
      </div>
    </div>
  );
}

function moodGradient(viability: number, pain: number): string {
  // healthy -> faint, dying -> red closing in; pain pulses the rim
  const harm = Math.max(0, Math.min(1, 1 - viability / 100));
  const r = Math.round(180 + 60 * harm);
  const edge = 0.12 + 0.5 * harm + 0.3 * Math.min(1, pain);
  return `radial-gradient(ellipse at center, rgba(0,0,0,0) 45%, rgba(${r},40,50,${edge.toFixed(
    3,
  )}) 100%)`;
}

export default function MindsEyePanel(props: { state: AgentState; embedded?: boolean }) {
  const { state } = props;
  const [orbit, setOrbit] = usePersistentState("decadic.mindseye.orbit", false);
  const wm = state.perceptual.working_memory;
  const slots = useMemo(() => (wm?.slots ?? []).filter((s) => s.position), [wm]);

  const pos = state.perceptual.proprio_position ?? [0, 0, 1.4];
  const ori = state.perceptual.proprio_orientation ?? [0, 0, 0];
  const pose: Pose = { position: pos, yaw: ori[2] ?? 0, pitch: ori[1] ?? 0 };

  const viability = state.viability?.value ?? 100;
  const pain = state.state_bus?.B_pain_scalar ?? 0;
  const contacts = state.perceptual.proprio_contacts ?? [];
  const contactLabels = ["R foot", "L foot", "R hand", "L hand"];

  const inView = slots.filter((s) => s.in_view).length;
  const remembered = slots.length - inView;
  const events = slots
    .filter((s) => (s.audio_intensity ?? 0) > 0.05 && s.last_event)
    .sort((a, b) => (b.audio_intensity ?? 0) - (a.audio_intensity ?? 0))
    .slice(0, 4);

  const content = (
    <>
      <h2>
        Mind's Eye
        <Info tip="The world as the agent models it, rendered from working memory + proprioception (not the head camera). Objects sit at their remembered coordinates; they fade as salience decays (object permanence), glow red when threatening / green when rewarding, and pulse when they make a sound. The view is from the agent's own pose and heading. CLIP is untouched - this is a read-out of the bound mental scene, not a photo." />
      </h2>

      <div className="me-bar">
        <span className="strip-label">
          {inView} in view · {remembered} remembered
        </span>
        <div style={{ flex: 1 }} />
        <button
          className="btn"
          title={orbit ? "Return to the agent's first-person view" : "Detach the camera to inspect the scene"}
          onClick={() => setOrbit((v) => !v)}
        >
          {orbit ? "First-person" : "Inspect"}
        </button>
      </div>

      <div className="me-stage">
        <Canvas shadows camera={{ fov: 75, near: 0.1, far: 100, position: [0, 3, 7] }}>
          <Scene slots={slots} pose={pose} orbit={orbit} />
        </Canvas>

        {/* HUD overlays: interoception + multimodal binding made visible */}
        <div className="me-vignette" style={{ background: moodGradient(viability, pain) }} />
        {!orbit && <div className="me-reticle" />}

        <div className="me-hud me-hud-tl">
          <div className="me-via">
            viability <b>{viability.toFixed(0)}</b>
          </div>
          {pain > 0.05 && <div className="me-pain">pain {pain.toFixed(2)}</div>}
        </div>

        <div className="me-hud me-hud-bl">
          {contactLabels.map((lab, i) => {
            const f = contacts[i] ?? 0;
            return (
              <div className="me-contact" key={lab} title={`${f.toFixed(0)} N`}>
                <span>{lab}</span>
                <span className="me-contact-track">
                  <span className="me-contact-fill" style={{ width: `${Math.min(100, (f / 400) * 100)}%` }} />
                </span>
              </div>
            );
          })}
        </div>

        {events.length > 0 && (
          <div className="me-hud me-hud-tr">
            {events.map((e) => (
              <div className="me-event" key={e.entity_id}>
                <span className="me-event-kind">{e.kind}</span> {e.last_event}
              </div>
            ))}
          </div>
        )}
      </div>

      {wm?.scene_preview && wm.scene_preview.length > 0 && (
        <SceneLatentStrip preview={wm.scene_preview} rms={wm.scene_latent_rms ?? null} />
      )}

      <div className="me-legend">
        <span><i className="me-dot threat" /> threatening</span>
        <span><i className="me-dot reward" /> rewarding</span>
        <span><i className="me-dot ghost" /> remembered (out of view)</span>
        <span><i className="me-dot audio" /> sounding</span>
      </div>

      {slots.length === 0 && (
        <div className="empty">nothing in working memory yet — connect a body in a populated scene</div>
      )}
    </>
  );

  if (props.embedded) return content;
  return <div className="panel span-7">{content}</div>;
}
