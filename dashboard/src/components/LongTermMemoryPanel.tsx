import { useMemo, useState } from "react";
import type { AgentState, LtmEdge, LtmGraphSnapshot, LtmNode } from "../api";
import Info from "./Info";

// The persistent, unbounded long-term knowledge graph. Promoted entity nodes
// remain the compatibility view; the semantic counters show provisional
// Framework-style evidence from moment one.

const W = 380;
const H = 320;
const CX = W / 2;
const CY = H / 2;
const PAD = 18;
const SEED_R = 120;
const ITER = 160;
const REP = 1500; // repulsion strength
const EDGE_LEN = 72; // preferred visible distance between linked LTM nodes
const EDGE_STRENGTH = 0.08;
const COOL = 15; // max per-step displacement at iteration 0
const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

type XY = { x: number; y: number };

function layout(nodes: LtmNode[], edges: LtmEdge[]): Map<string, XY> {
  const pos = new Map<string, XY>();
  const n = nodes.length;
  if (n === 0) return pos;
  if (n === 1) {
    pos.set(nodes[0].id, { x: CX, y: CY });
    return pos;
  }
  nodes.forEach((node, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    pos.set(node.id, { x: CX + Math.cos(a) * SEED_R, y: CY + Math.sin(a) * SEED_R });
  });
  const idx = new Map(nodes.map((nd, i) => [nd.id, i]));
  const ex = edges.filter((e) => idx.has(e.source) && idx.has(e.target));

  for (let it = 0; it < ITER; it++) {
    const disp: XY[] = nodes.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const pi = pos.get(nodes[i].id)!;
        const pj = pos.get(nodes[j].id)!;
        let dx = pi.x - pj.x;
        let dy = pi.y - pj.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 0.01) {
          dx = (i - j) * 0.13 + 0.1;
          dy = (j - i) * 0.07 + 0.1;
          d2 = dx * dx + dy * dy + 0.01;
        }
        const d = Math.sqrt(d2);
        const f = REP / d2;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        disp[i].x += fx;
        disp[i].y += fy;
        disp[j].x -= fx;
        disp[j].y -= fy;
      }
    }
    for (const e of ex) {
      const i = idx.get(e.source)!;
      const j = idx.get(e.target)!;
      const pi = pos.get(e.source)!;
      const pj = pos.get(e.target)!;
      const dx = pi.x - pj.x;
      const dy = pi.y - pj.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const target = EDGE_LEN * (1 + 0.35 * (1 - Math.min(1, e.weight)));
      const f = ((d - target) / target) * EDGE_STRENGTH;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      disp[i].x -= fx;
      disp[i].y -= fy;
      disp[j].x += fx;
      disp[j].y += fy;
    }
    const tmax = COOL * (1 - it / ITER) + 0.5;
    nodes.forEach((nd, i) => {
      const p = pos.get(nd.id)!;
      let dx = disp[i].x;
      let dy = disp[i].y;
      const dl = Math.sqrt(dx * dx + dy * dy) + 1e-6;
      const cap = Math.min(dl, tmax);
      dx = (dx / dl) * cap;
      dy = (dy / dl) * cap;
      p.x += dx + (CX - p.x) * 0.01;
      p.y += dy + (CY - p.y) * 0.01;
      p.x = Math.max(PAD, Math.min(W - PAD, p.x));
      p.y = Math.max(PAD, Math.min(H - PAD, p.y));
    });
  }
  return pos;
}

function nodeRadius(n: LtmNode): number {
  const seen = Math.log2(1 + Math.max(0, n.seen_count)) * 1.8;
  const deg = Math.max(0, n.degree) * 0.5;
  return Math.max(4, Math.min(13, 4 + seen + deg));
}

function nodeFill(n: LtmNode): string {
  // Hue is the object's identity (appearance), brightness tracks salience.
  const hue = ((n.appearance_hash % 360) + 360) % 360;
  const lum = 40 + Math.round(26 * Math.min(1, Math.max(0, n.salience)));
  return `hsl(${hue} 68% ${lum}%)`;
}

function nodeStroke(n: LtmNode): string {
  // A faint warm/cool rim encodes the consolidated affect (red = aversive,
  // green = rewarding) without naming anything.
  if (n.affect > 0.05) return "rgba(90,220,140,0.9)";
  if (n.affect < -0.05) return "rgba(255,90,90,0.9)";
  return "rgba(0,0,0,0.45)";
}

export default function LongTermMemoryPanel(props: { state: AgentState }) {
  const [zoom, setZoom] = useState(1);
  const ltm: LtmGraphSnapshot | null | undefined = props.state.perceptual.ltm_graph;
  const nodes = ltm?.nodes ?? [];
  const edges = ltm?.edges ?? [];
  const totalNodes = ltm?.total_nodes ?? 0;
  const totalEdges = ltm?.total_edges ?? 0;
  const health = props.state.perceptual.discovery_health;
  const ltmStatus = props.state.perceptual.ltm_consolidation;
  const totalBeliefs = ltm?.total_property_beliefs ?? 0;
  const unstableBeliefs = ltm?.unstable_property_count ?? 0;
  const semantic = ltm?.semantic;

  // Recompute the force layout only when the graph structure actually changes,
  // not on every cycle tick (keeps the view stable and cheap).
  const sig = useMemo(
    () => `${nodes.map((n) => n.id).join(",")}|${edges.length}|${totalNodes}`,
    [nodes, edges.length, totalNodes],
  );
  const pos = useMemo(() => layout(nodes, edges), [sig]); // eslint-disable-line react-hooks/exhaustive-deps
  const viewW = W / zoom;
  const viewH = H / zoom;
  const viewBox = `${CX - viewW / 2} ${CY - viewH / 2} ${viewW} ${viewH}`;
  const zoomBy = (factor: number) => {
    setZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, +(z * factor).toFixed(2))));
  };

  return (
    <div className="panel span-5">
      <h2>
        Long-term Memory
        <Info tip="The persistent semantic graph. Provisional anonymous percepts enter immediately as evidence; promotion into stable nodes is stricter and precision-based. Runtime records remain label-free: entities, events, relationships, correlations, conclusions, and contextual values." />
      </h2>

      <div className="strip-label">
        <span className="ltm-counter">
          {totalNodes.toLocaleString()} nodes · {totalEdges.toLocaleString()} edges
          {" · "}
          {totalBeliefs.toLocaleString()} beliefs
        </span>
        <span>
          {ltmStatus?.status ?? (nodes.length < totalNodes ? `showing ${nodes.length}` : "long-term index")}
        </span>
      </div>
      {health && health.status !== "healthy" && (
        <div className={`health-strip ${health.collapsed ? "bad" : "warn"}`}>
          <span>{health.reason}</span>
          <span>LTM {health.ltm_write}</span>
          <span>{health.object_files} object files</span>
          <span>spread {health.centroid_spread.toFixed(3)}</span>
          <span>flow {health.flow_confidence.toFixed(3)}</span>
        </div>
      )}
      {semantic && (
        <div className="health-strip ok">
          <span>entities {semantic.entities}</span>
          <span>events {semantic.events}</span>
          <span>relationships {semantic.relationships}</span>
          <span>correlations {semantic.correlations}</span>
          <span>conclusions {semantic.conclusions}</span>
          <span>values {semantic.values}</span>
        </div>
      )}
      {ltmStatus?.semantic_update && (
        <div className="health-strip ok">
          <span>last semantic write</span>
          <span>e {ltmStatus.semantic_update.entities ?? 0}</span>
          <span>ev {ltmStatus.semantic_update.events ?? 0}</span>
          <span>rel {ltmStatus.semantic_update.relationships ?? 0}</span>
          <span>corr {ltmStatus.semantic_update.correlations ?? 0}</span>
          <span>conc {ltmStatus.semantic_update.conclusions ?? 0}</span>
          <span>val {ltmStatus.semantic_update.values ?? 0}</span>
        </div>
      )}
      {(ltmStatus?.property_update || totalBeliefs > 0) && (
        <div className={`health-strip ${unstableBeliefs > 0 ? "warn" : "ok"}`}>
          <span>beliefs {totalBeliefs}</span>
          <span>avg conf {(ltmStatus?.avg_property_confidence ?? 0).toFixed(3)}</span>
          <span>unstable {unstableBeliefs}</span>
          <span>relations {ltmStatus?.relationship_update ? "updated" : "gated"}</span>
        </div>
      )}

      {nodes.length === 0 ? (
        <div className="ltm-empty">
          No promoted long-term nodes yet. Provisional semantic evidence should
          still appear in the counters above as soon as percepts enter working memory.
        </div>
      ) : (
        <div className="ltm-graph-wrap">
          <div className="ltm-zoom-controls" aria-label="LTM graph zoom controls">
            <button type="button" className="icon-btn" title="Zoom out" onClick={() => zoomBy(1 / 1.25)}>
              -
            </button>
            <span>{zoom.toFixed(1)}x</span>
            <button type="button" className="icon-btn" title="Zoom in" onClick={() => zoomBy(1.25)}>
              +
            </button>
            <button type="button" className="btn" title="Reset LTM graph zoom" onClick={() => setZoom(1)}>
              Reset
            </button>
          </div>
          <svg
          className="graph-svg ltm-graph-svg"
          viewBox={viewBox}
          preserveAspectRatio="xMidYMid meet"
          onWheel={(e) => {
            e.preventDefault();
            zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12);
          }}
        >
          {edges.map((e, i) => {
            const a = pos.get(e.source);
            const b = pos.get(e.target);
            if (!a || !b) return null;
            const w = Math.min(1, Math.max(0, e.weight));
            return (
              <line
                key={`le${i}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={`rgba(150,165,210,${0.12 + 0.5 * w})`}
                strokeWidth={0.8 + 1.6 * w}
              />
            );
          })}
          {nodes.map((n, i) => {
            const p = pos.get(n.id);
            if (!p) return null;
            const r = nodeRadius(n);
            return (
              <circle
                key={`ln${i}`}
                cx={p.x}
                cy={p.y}
                r={r}
                fill={nodeFill(n)}
                stroke={nodeStroke(n)}
                strokeWidth={1.2}
              >
                <title>
                  {`${n.id} · seen ${n.seen_count}× · ${n.degree} links` +
                    (Math.abs(n.affect) > 0.05 ? ` · affect ${n.affect.toFixed(2)}` : "")}
                </title>
              </circle>
            );
          })}
          </svg>
        </div>
      )}

      {nodes.length > 0 && (
        <div className="belief-list">
          {nodes.slice(0, 4).map((n) => {
            const beliefs = n.property_beliefs ?? [];
            if (beliefs.length === 0) return null;
            return (
              <div className="belief-row" key={`belief-${n.id}`}>
                <span className="belief-node">{n.id}</span>
                <span className="belief-items">
                  {beliefs.slice(0, 4).map((b) => (
                    <span className={b.unstable ? "belief-chip unstable" : "belief-chip"} key={`${n.id}-${b.property_key}`}>
                      {b.property_key} {(b.confidence * 100).toFixed(0)}% · n={b.evidence_count.toFixed(0)}
                    </span>
                  ))}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <div className="graph-legend">
        <span><i className="lg-dot fresh" /> appearance = hue</span>
        <span><i className="lg-dot bigger" /> bigger = seen more</span>
        <span><i className="lg-line reward" /> rewarding rim</span>
        <span><i className="lg-line threat" /> aversive rim</span>
      </div>
    </div>
  );
}
