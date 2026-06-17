import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { AgentState, BrainLayer, BrainTopology, Metrics } from "../api";
import { fetchBrainTopology } from "../api";
import Info from "./Info";

const MAX_SPHERES = 256; // visual cap per layer; real unit count shown in HUD
const RING_RADIUS = 6.5;
const SPACING = 0.15;

const STAGE_COLORS: Record<number, string> = {
  1: "#5aa9ff",
  2: "#4fd1c5",
  3: "#b08cff",
  4: "#ffb74a",
  5: "#7ce38b",
  6: "#ff6b81",
  7: "#ffd36a",
  8: "#62d0ff",
  9: "#ff9f43",
  10: "#8a8f9c",
};
const DIM = new THREE.Color("#1a2233");

type LayerLayout = {
  layer: BrainLayer;
  center: THREE.Vector3;
  tangent: THREE.Vector3;
  count: number;
  cols: number;
};

function buildLayouts(topo: BrainTopology): LayerLayout[] {
  const n = topo.layers.length;
  return topo.layers.map((layer, i) => {
    const theta = (i / n) * Math.PI * 2;
    const count = Math.min(layer.units, MAX_SPHERES);
    return {
      layer,
      center: new THREE.Vector3(
        Math.sin(theta) * RING_RADIUS,
        0,
        Math.cos(theta) * RING_RADIUS,
      ),
      tangent: new THREE.Vector3(Math.cos(theta), 0, -Math.sin(theta)),
      count,
      cols: Math.min(8, Math.max(1, Math.ceil(Math.sqrt(count)))),
    };
  });
}

const UP = new THREE.Vector3(0, 1, 0);

function spherePos(l: LayerLayout, unit: number, out: THREE.Vector3): THREE.Vector3 {
  const idx = unit % l.count;
  const rows = Math.ceil(l.count / l.cols);
  const col = idx % l.cols;
  const row = Math.floor(idx / l.cols);
  out.copy(l.center);
  out.addScaledVector(l.tangent, (col - (l.cols - 1) / 2) * SPACING);
  out.addScaledVector(UP, (row - (rows - 1) / 2) * SPACING);
  return out;
}

function LayerCloud(props: {
  layout: LayerLayout;
  activations: number[] | null;
  onHover: (layer: BrainLayer | null) => void;
}) {
  const { layout, activations, onHover } = props;
  const ref = useRef<THREE.InstancedMesh>(null);
  const base = useMemo(
    () => new THREE.Color(STAGE_COLORS[layout.layer.stage] ?? "#8a8f9c"),
    [layout.layer.stage],
  );

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const v = new THREE.Vector3();
    for (let i = 0; i < layout.count; i++) {
      spherePos(layout, i, v);
      dummy.position.copy(v);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, base);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [layout, base]);

  // Live coloring: each displayed sphere samples the 32-bin activation profile
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const c = new THREE.Color();
    let max = 0;
    if (activations) for (const a of activations) max = Math.max(max, a);
    for (let i = 0; i < layout.count; i++) {
      let t = 0.35;
      if (activations && max > 1e-9) {
        const bin = Math.min(31, Math.floor((i * 32) / layout.count));
        t = 0.15 + 0.85 * Math.min(1, activations[bin] / max);
      }
      c.copy(DIM).lerp(base, t);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [activations, layout, base]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, layout.count]}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(layout.layer);
      }}
      onPointerOut={() => onHover(null)}
    >
      <sphereGeometry args={[0.055, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/** All exported strongest weights drawn as individual fibers. */
function Fibers({ topo, layouts }: { topo: BrainTopology; layouts: LayerLayout[] }) {
  const geom = useMemo(() => {
    const byId = new Map(layouts.map((l) => [l.layer.id, l]));
    const pos: number[] = [];
    const col: number[] = [];
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const neg = new THREE.Color("#3d6fd1");
    const posi = new THREE.Color("#ff9f43");
    for (const e of topo.edges) {
      const src = byId.get(e.src);
      const dst = byId.get(e.dst);
      if (!src || !dst) continue;
      for (const f of e.fibers) {
        spherePos(src, f.si, a);
        spherePos(dst, f.di, b);
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const c = f.w < 0 ? neg : posi;
        col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    return g;
  }, [topo, layouts]);

  useEffect(() => () => geom.dispose(), [geom]);

  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial vertexColors transparent opacity={0.22} toneMapped={false} />
    </lineSegments>
  );
}

/** Faint trunk between connected blocks (summarizes the unexported weights). */
function Trunks({ topo, layouts }: { topo: BrainTopology; layouts: LayerLayout[] }) {
  const geom = useMemo(() => {
    const byId = new Map(layouts.map((l) => [l.layer.id, l]));
    const pos: number[] = [];
    for (const e of topo.edges) {
      const src = byId.get(e.src);
      const dst = byId.get(e.dst);
      if (!src || !dst) continue;
      pos.push(src.center.x, src.center.y, src.center.z, dst.center.x, dst.center.y, dst.center.z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, [topo, layouts]);

  useEffect(() => () => geom.dispose(), [geom]);

  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color="#2b3650" transparent opacity={0.5} toneMapped={false} />
    </lineSegments>
  );
}

function fmt(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

export default function BrainMapPanel(props: {
  agentId: string;
  state: AgentState;
  metrics: Metrics | null;
}) {
  const { agentId, state, metrics } = props;
  const [topo, setTopo] = useState<BrainTopology | null>(null);
  const [failed, setFailed] = useState(false);
  const [hovered, setHovered] = useState<BrainLayer | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  // Weights change on reset/preset switch — detect the cycle counter dropping
  const prevCycles = useRef<number | null>(null);
  const cycles = metrics?.cycles_completed ?? null;
  useEffect(() => {
    if (cycles != null && prevCycles.current != null && cycles < prevCycles.current) {
      setReloadKey((k) => k + 1);
    }
    prevCycles.current = cycles;
  }, [cycles]);

  // Neuron growth (C) changes the topology mid-run — reload the ring when the
  // awake-neuron count moves so the Brain Map shows the newly woken units.
  const prevAwake = useRef<number | null>(null);
  const awake = metrics?.awake_neurons ?? null;
  useEffect(() => {
    if (awake != null && prevAwake.current != null && awake !== prevAwake.current) {
      setReloadKey((k) => k + 1);
    }
    prevAwake.current = awake;
  }, [awake]);

  const preset = metrics?.preset ?? null;
  useEffect(() => {
    let gone = false;
    setFailed(false);
    fetchBrainTopology(agentId)
      .then((t) => {
        if (!gone) setTopo(t);
      })
      .catch(() => {
        if (!gone) {
          setTopo(null);
          setFailed(true);
        }
      });
    return () => {
      gone = true;
    };
  }, [agentId, preset, reloadKey]);

  const layouts = useMemo(() => (topo ? buildLayouts(topo) : []), [topo]);

  const stageActs = useMemo(() => {
    const m = new Map<number, number[]>();
    for (const s of state.last_cycle_trace?.stages ?? []) {
      if (s.payload.activations) m.set(s.stage, s.payload.activations);
    }
    return m;
  }, [state.last_cycle_trace]);

  const totals = topo?.totals;

  return (
    <div className="panel span-7">
      <h2>
        Brain Map
        <Info tip="The agent's actual neural network: every block of the cognitive stack as a cluster of neurons (capped at 256 spheres per block; real counts in the overlay), arranged around the Decadic cycle. Lines are the strongest learned weights — orange excitatory, blue inhibitory. Clusters light up with the real activations of the last cycle, so you watch the thought move around the ring. Drag to orbit, scroll to zoom." />
      </h2>

      <div className="bm-stage">
        {topo && (
          <Canvas camera={{ fov: 55, near: 0.1, far: 120, position: [0, 6, 13] }}>
            <color attach="background" args={["#0d111a"]} />
            <fog attach="fog" args={["#0d111a", 14, 40]} />
            <Trunks topo={topo} layouts={layouts} />
            <Fibers topo={topo} layouts={layouts} />
            {layouts.map((l) => (
              <LayerCloud
                key={l.layer.id}
                layout={l}
                activations={stageActs.get(l.layer.stage) ?? null}
                onHover={setHovered}
              />
            ))}
            <OrbitControls autoRotate autoRotateSpeed={0.6} enablePan={false} />
          </Canvas>
        )}

        {totals && (
          <div className="me-hud me-hud-tl">
            <div className="bm-totals">
              <b>{fmt(totals.neurons)}</b> neurons · <b>{fmt(totals.connections)}</b> connections ·{" "}
              <b>{fmt(totals.params)}</b> params
              {totals.preset ? (
                <>
                  {" "}
                  · preset <b>{totals.preset}</b>
                </>
              ) : null}
              {totals.awake_neurons != null && totals.allocated_neurons != null ? (
                <>
                  {" "}
                  · awake <b>{fmt(totals.awake_neurons)}</b>/{fmt(totals.allocated_neurons)}
                </>
              ) : null}
            </div>
          </div>
        )}

        <div className="me-hud me-hud-bl">
          {hovered ? (
            <div className="bm-layerinfo">
              <b>{hovered.label}</b> — stage {hovered.stage} · {fmt(hovered.units)} units ·{" "}
              {fmt(hovered.params)} params
            </div>
          ) : (
            <div className="bm-layerinfo dim">hover a cluster for details</div>
          )}
        </div>

        {!topo && (
          <div className="empty bm-empty">
            {failed
              ? "no neural stack (agent runs stub cognition)"
              : "loading brain topology..."}
          </div>
        )}
      </div>
    </div>
  );
}
