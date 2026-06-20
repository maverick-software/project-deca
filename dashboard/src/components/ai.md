# dashboard/src/components — AI navigation notes

React panels for the Decadic dashboard. Each panel is a self-contained view fed
by `AgentState` / `Metrics` from `../api`. Conventions: a `.panel` wrapper with an
`<h2>` + `<Info tip=... />`, an optional `.strip-label` header row, an SVG body,
and a `.graph-legend`. Panels are mounted per-tab in `../App.tsx` and wrapped in
an `ErrorBoundary`.

## Memory / graph panels

- `GraphPanel.tsx` — the **bounded "now"** self-indexed egocentric graph
  (working-memory slots around the SELF node). Spatial/proximity/affective/agency
  edges; node brightness = salience.
- `LongTermMemoryPanel.tsx` — the **persistent, unbounded** long-term knowledge
  graph (`perceptual.ltm_graph`, the hippocampal index). Force-directed relational
  view; node color = a deterministic hue of the appearance embedding (NO semantic
  labels, by design), node size = times seen, rim = consolidated affect (green
  rewarding / red aversive). Headline `total_nodes / total_edges` counters make
  growth observable; the rendered nodes/edges are a recent-window read-out, not
  the whole graph. Layout is memoized on a structural signature so it only
  recomputes when the graph changes, not every cycle tick. Both panels share the
  Self-Indexed Graph tab.

## Other panels (overview)

Vitals/Homeostasis/Cycle/StateBus/Events/CycleWheel/Discovery/Eval/Cognition/
Capacity/Motor/Locomotion/BrainMap/Environment/SkillDojo/Deployment/SavedAgents
— see `../App.tsx` for tab wiring and which props each receives.

When adding a panel: add its types to `../api.ts`, keep the file < 500 lines,
reuse the shared `.panel`/`.graph-svg`/`.graph-legend`/`.strip-label` styles in
`../style.css`, mount it in `../App.tsx` inside an `ErrorBoundary`.
