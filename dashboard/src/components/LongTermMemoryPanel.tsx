import { useMemo } from "react";
import type { AgentState, LtmEdge, LtmGraphSnapshot, LtmNode } from "../api";
import Info from "./Info";

// The persistent, unbounded long-term knowledge graph (the "hippocampal index"):
// one permanent node per consolidated object, keyed by its learned appearance and
// colored by an appearance-derived hue (no semantic labels). Working memory stays
// bounded; this graph grows without limit, so the counters are the headline.

const W = 380;
const H = 320;
const CX = W / 2;
const CY = H / 2;
const PAD = 18;
const SEED_R = 120;
const ITER = 160;
const REP = 1500; // repulsion strength
const SPRING = 190; // edge spring length scale
const COOL = 15; // max per-step displacement at iteration 0

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
      const f = ((d * d) / SPRING) * (0.5 + 0.5 * Math.min(1, e.weight));
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
  const ltm: LtmGraphSnapshot | null | undefined = props.state.perceptual.ltm_graph;
  const nodes = ltm?.nodes ?? [];
  const edges = ltm?.edges ?? [];
  const totalNodes = ltm?.total_nodes ?? 0;
  const totalEdges = ltm?.total_edges ?? 0;

  // Recompute the force layout only when the graph structure actually changes,
  // not on every cycle tick (keeps the view stable and cheap).
  const sig = useMemo(
    () => `${nodes.map((n) => n.id).join(",")}|${edges.length}|${totalNodes}`,
    [nodes, edges.length, totalNodes],
  );
  const pos = useMemo(() => layout(nodes, edges), [sig]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="panel span-5">
      <h2>
        Long-term Memory
        <Info tip="The persistent, unbounded long-term knowledge graph - the agent's lifelong relational memory. Each node is an object the agent has stably perceived and consolidated out of working memory, keyed by its learned appearance (color = appearance, not a label). Edges form when objects are seen together. Working memory is a bounded 'now' buffer; this graph grows without limit, so the headline counters show its real size while the view shows a recent window." />
      </h2>

      <div className="strip-label">
        <span className="ltm-counter">
          {totalNodes.toLocaleString()} nodes · {totalEdges.toLocaleString()} edges
        </span>
        <span>
          {nodes.length < totalNodes ? `showing ${nodes.length}` : "long-term index"}
        </span>
      </div>

      {nodes.length === 0 ? (
        <div className="ltm-empty">
          No long-term nodes yet. The graph grows as the agent stably perceives
          objects and consolidates them out of working memory.
        </div>
      ) : (
        <svg
          className="graph-svg"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
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
