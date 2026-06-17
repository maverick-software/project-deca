import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { BrainLandscape } from "../api";
import { fetchBrainLandscape } from "../api";
import Info from "./Info";

const EXTENT = 5; // half-width of the rendered plane (alpha/beta axes)
const HEIGHT = 3.4; // vertical exaggeration of the normalized loss
const POLL_MS = 4000;

// Diverging RdYlBu (reversed): deep blue = low loss (valley), red = high (ridge).
const STOPS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"].map(
  (h) => new THREE.Color(h),
);

function colormap(t: number, out: THREE.Color): THREE.Color {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  return out.copy(STOPS[i]).lerp(STOPS[i + 1], x - i);
}

/** Build the loss surface mesh geometry from the z[][] grid. */
function buildSurface(s: BrainLandscape): THREE.BufferGeometry | null {
  const z = s.z;
  if (!z || z.length < 2) return null;
  const n = z.length;
  const zmin = s.z_min ?? 0;
  const zmax = s.z_max ?? 1;
  const range = zmax - zmin || 1;
  const pos: number[] = [];
  const col: number[] = [];
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = (i / (n - 1) - 0.5) * 2 * EXTENT;
      const zc = (j / (n - 1) - 0.5) * 2 * EXTENT;
      const v = z[i][j];
      const t = Number.isFinite(v) ? (v - zmin) / range : 0;
      pos.push(x, t * HEIGHT, zc);
      colormap(t, c);
      col.push(c.r, c.g, c.b);
    }
  }
  const idx: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - 1; j++) {
      const a = i * n + j;
      const b = i * n + j + 1;
      const d = (i + 1) * n + j;
      const e = (i + 1) * n + j + 1;
      idx.push(a, d, b, b, d, e);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function Surface({ surface }: { surface: BrainLandscape }) {
  const geom = useMemo(() => buildSurface(surface), [surface]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom) return null;
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.55}
        metalness={0.05}
        flatShading
      />
    </mesh>
  );
}

/** Marker sphere at the current weights theta* (center of the slice). */
function CenterMarker({ surface }: { surface: BrainLandscape }) {
  const zmin = surface.z_min ?? 0;
  const zmax = surface.z_max ?? 1;
  const range = zmax - zmin || 1;
  const t = ((surface.center_loss ?? zmin) - zmin) / range;
  const y = t * HEIGHT;
  return (
    <mesh position={[0, y + 0.12, 0]}>
      <sphereGeometry args={[0.16, 16, 16]} />
      <meshBasicMaterial color="#ffffff" toneMapped={false} />
    </mesh>
  );
}

function fmt(n: number | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 100) return n.toFixed(0);
  return n.toFixed(3);
}

export default function LandscapePanel(props: { agentId: string }) {
  const { agentId } = props;
  const [surface, setSurface] = useState<BrainLandscape | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let gone = false;
    let timer: number | undefined;
    const poll = () => {
      fetchBrainLandscape(agentId)
        .then((s) => {
          if (gone) return;
          setFailed(false);
          setSurface(s.ready ? s : null);
        })
        .catch(() => {
          if (!gone) setFailed(true);
        })
        .finally(() => {
          if (!gone) timer = window.setTimeout(poll, POLL_MS);
        });
    };
    poll();
    return () => {
      gone = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [agentId]);

  const ready = surface?.ready === true;

  return (
    <div className="panel span-7">
      <h2>
        Loss Landscape
        <Info tip="A live 2D slice of the agent's actual weight space, rendered as the loss surface from filter-normalized random directions (Li et al., 2018). Height and color are the real predictive-coding + forward-model objective evaluated on a replay batch of the agent's own experience — blue valleys are low loss, red ridges high. The white sphere is the agent's current weights θ*; watch the bowl reshape as it learns. This is a projection of a huge weight space, so the geometry is qualitative. Requires DECADIC_LANDSCAPE_ENABLED=1. Drag to orbit, scroll to zoom." />
      </h2>

      <div className="bm-stage">
        {ready && surface && (
          <Canvas camera={{ fov: 50, near: 0.1, far: 100, position: [7, 6, 7] }}>
            <color attach="background" args={["#0d111a"]} />
            <ambientLight intensity={0.65} />
            <directionalLight position={[6, 12, 4]} intensity={0.85} />
            <directionalLight position={[-6, 4, -6]} intensity={0.25} />
            <Surface surface={surface} />
            <CenterMarker surface={surface} />
            <OrbitControls autoRotate autoRotateSpeed={0.5} enablePan={false} />
          </Canvas>
        )}

        {ready && surface && (
          <div className="me-hud me-hud-tl">
            <div className="bm-totals">
              θ* loss <b>{fmt(surface.center_loss)}</b> · range{" "}
              <b>
                {fmt(surface.z_min)}–{fmt(surface.z_max)}
              </b>{" "}
              · grid <b>{surface.grid}</b> · batch <b>{surface.batch}</b>
              {surface.cycle != null ? (
                <>
                  {" "}
                  · cycle <b>{surface.cycle}</b>
                </>
              ) : null}
              {surface.wall_ms != null ? (
                <>
                  {" "}
                  · <b>{surface.wall_ms.toFixed(0)}ms</b>
                </>
              ) : null}
            </div>
          </div>
        )}

        {!ready && (
          <div className="empty bm-empty">
            {failed
              ? "loss landscape unavailable (unknown agent)"
              : "warming up — enable DECADIC_LANDSCAPE_ENABLED=1 and let the replay buffer fill"}
          </div>
        )}
      </div>
    </div>
  );
}
