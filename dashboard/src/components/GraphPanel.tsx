import type { AgentState, EgoEdge, EgoGraph, EgoNode } from "../api";
import Info from "./Info";

const W = 360;
const H = 300;
const CX = W / 2;
const CY = H / 2;
const MIN_R = 46;
const MAX_R = 128;

type Placed = { node: EgoNode; x: number; y: number };

function spatialDistances(edges: EgoEdge[], selfId: string): Map<string, number> {
  const out = new Map<string, number>();
  for (const e of edges) {
    if (e.kind === "spatial" && e.source === selfId && e.distance != null) {
      out.set(e.target, e.distance);
    }
  }
  return out;
}

function place(graph: EgoGraph): { selfId: string; placed: Map<string, Placed> } {
  const placed = new Map<string, Placed>();
  const self = graph.nodes.find((n) => n.role === "self");
  const selfId = self?.id ?? "self";
  if (self) placed.set(selfId, { node: self, x: CX, y: CY });

  const others = graph.nodes.filter((n) => n.role !== "self");
  const dists = spatialDistances(graph.edges, selfId);
  const maxDist = Math.max(1, ...[...dists.values()]);

  others.forEach((n, i) => {
    const angle = (i / Math.max(1, others.length)) * Math.PI * 2 - Math.PI / 2;
    const d = dists.get(n.id ?? "");
    const r =
      n.role === "context"
        ? MAX_R + 8
        : d != null
          ? MIN_R + (d / maxDist) * (MAX_R - MIN_R)
          : (MIN_R + MAX_R) / 2;
    placed.set(n.id ?? `n${i}`, {
      node: n,
      x: CX + Math.cos(angle) * r,
      y: CY + Math.sin(angle) * r,
    });
  });
  return { selfId, placed };
}

function edgeStyle(e: EgoEdge): { stroke: string; width: number; dash?: string } {
  if (e.kind === "affective") {
    const v = e.weight;
    const mag = Math.min(1, Math.abs(v));
    return {
      stroke: v < 0 ? `rgba(255,90,90,${0.45 + 0.55 * mag})` : `rgba(90,220,140,${0.45 + 0.55 * mag})`,
      width: 1.5 + 3.5 * mag,
    };
  }
  if (e.kind === "proximity") {
    return { stroke: "rgba(120,160,220,0.5)", width: 1, dash: "4 3" };
  }
  if (e.kind === "context") {
    return { stroke: "rgba(150,150,170,0.25)", width: 1, dash: "2 4" };
  }
  if (e.kind === "agency") {
    // The learned "this is mine" relation (self -> discovered body part).
    const mag = Math.min(1, Math.abs(e.weight));
    return { stroke: `rgba(200,130,255,${0.5 + 0.5 * mag})`, width: 2 + 2.5 * mag };
  }
  // spatial: closeness brightness
  return { stroke: `rgba(150,160,180,${0.2 + 0.5 * Math.min(1, e.weight)})`, width: 1.2 };
}

function nodeFill(n: EgoNode): string {
  if (n.role === "self") return "var(--accent)";
  if (n.role === "context") return "rgba(180,160,90,0.85)";
  // Discovered body part ("mine"): violet, set apart from external objects.
  if (n.kind === "self_part") return "rgba(200,130,255,0.95)";
  const s = n.salience ?? 1;
  // brightness tracks salience: bright when fresh, dim as it fades out of view
  const lum = 30 + Math.round(55 * Math.min(1, Math.max(0, s)));
  return `hsl(205 70% ${lum}%)`;
}

export default function GraphPanel(props: { state: AgentState }) {
  const p = props.state.perceptual;
  const graph: EgoGraph = p.egocentric_graph ?? {
    nodes: p.egocentric_nodes,
    edges: p.egocentric_edges ?? [],
  };
  const { placed } = place(graph);
  const nodeCount = graph.nodes.length;
  const edgeCount = graph.edges.length;
  const affective = graph.edges.filter((e) => e.kind === "affective").length;
  const health = p.discovery_health;

  return (
    <div className="panel span-7">
      <h2>
        Self-Indexed Graph
        <Info tip="The agent's logical layer made visible: a relational graph centered on the SELF node. Spokes are spatial relations (brighter = closer), dashed blue links are entity-to-entity proximity, and red/green links are affective edges tying a thing to the self's survival concerns (red = threatening, green = rewarding). Node brightness = working-memory salience, so objects fade as they leave view rather than vanishing." />
      </h2>

      <div className="strip-label">
        <span>{nodeCount} nodes · {edgeCount} edges · {affective} affective</span>
        <span>{health ? health.status : "self-centered"}</span>
      </div>
      {health && health.status !== "healthy" && (
        <div className={`health-strip ${health.collapsed ? "bad" : "warn"}`}>
          <span>{health.reason}</span>
          <span>{health.object_files} object files</span>
          <span>spread {health.centroid_spread.toFixed(3)}</span>
          <span>flow {health.flow_confidence.toFixed(3)}</span>
          <span>loom {health.looming_count}</span>
        </div>
      )}

      <svg className="graph-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {graph.edges.map((e, i) => {
          const a = placed.get(e.source);
          const b = placed.get(e.target);
          if (!a || !b) return null;
          const st = edgeStyle(e);
          return (
            <line
              key={`e${i}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={st.stroke}
              strokeWidth={st.width}
              strokeDasharray={st.dash}
            />
          );
        })}
        {[...placed.values()].map((pl, i) => {
          const isSelf = pl.node.role === "self";
          const isCtx = pl.node.role === "context";
          const isPart = pl.node.kind === "self_part";
          const r = isSelf ? 11 : isPart ? 8 : isCtx ? 5 : 7;
          const k = pl.node.kind;
          // Discovered objects coin anonymous ids; prefer them over "unknown".
          const label = k && k !== "unknown" ? k : (pl.node.id ?? "");
          return (
            <g key={`n${i}`}>
              <circle
                cx={pl.x}
                cy={pl.y}
                r={r}
                fill={nodeFill(pl.node)}
                stroke="rgba(0,0,0,0.4)"
                strokeWidth={isSelf ? 2 : 1}
              >
                <title>
                  {label}
                  {pl.node.salience != null ? ` (salience ${pl.node.salience.toFixed(2)})` : ""}
                </title>
              </circle>
              <text
                x={pl.x}
                y={pl.y - r - 3}
                textAnchor="middle"
                className="graph-label"
              >
                {isSelf ? "SELF" : label}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="graph-legend">
        <span><i className="lg-line spatial" /> spatial</span>
        <span><i className="lg-line proximity" /> proximity</span>
        <span><i className="lg-line threat" /> threat</span>
        <span><i className="lg-line reward" /> reward</span>
        <span><i className="lg-line agency" /> mine (agency)</span>
        <span><i className="lg-dot fresh" /> in view</span>
        <span><i className="lg-dot faded" /> remembered</span>
        <span><i className="lg-dot bodypart" /> body part</span>
      </div>
    </div>
  );
}
