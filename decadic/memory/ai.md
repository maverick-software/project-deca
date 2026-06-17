# decadic/memory — AI navigation notes

Memory subsystems for the Decadic cognitive cycle. Two complementary stores
(Complementary-Learning-Systems framing):

- `episodic_store.py` — `EpisodicStore`: SQLite-backed (in-memory fallback)
  per-cycle *log*. Stores cycle summaries + fixed-size embeddings; vector-addressed
  similarity recall (`search_similar`, `retrieval_context_vector`). This is the
  "diary", not a graph.
- `embeddings.py` — fixed-size episode fingerprints (`EMBEDDING_DIM`,
  `episode_embedding_from_cycle`, `query_vector_from_state_bus`, `perceptual_key`).
- `semantic_graph.py` — `LongTermGraph`: the persistent, **unbounded** relational
  long-term memory (the "hippocampal index"). One node per consolidated object,
  keyed by its learned appearance embedding; edges accumulate from co-presence.
  Key methods:
  - `match(appearance, threshold)` — cosine re-identification -> node id | None.
  - `upsert_node(appearance, kind, position, affect, cycle)` — coin `ent-NNNNN`
    or EMA-update a matched node. Node count grows without cap.
  - `bump_edge(src, dst, kind, weight, cycle)` — undirected co-occurrence edges.
  - `consolidate(slots, affect, cycle, min_seen)` — commit stable working-memory
    slots + link co-present ones (no-op for appearance-less oracle slots).
  - `snapshot(limit)` — windowed read-out for the dashboard + `total_nodes/edges`.
  - `clear()`, `backup_to(path)`, `restore_from(path)` — reset + SQLite persistence.

## How it wires into the cycle

- Constructed per agent in `agents/registry.py` as `agent_<id>_graph.sqlite`,
  held on `AgentRuntime.ltm_graph`, injected into every `CycleContext`.
- ON by default (`config.ltm_graph_enabled()`; env `DECADIC_LTM_GRAPH=0` disables
  for parity tests, in which case `ltm_graph` is None and all paths fall back).
- Consolidation write: `cycle/stages/stage_10.py` calls `consolidate(...)` over
  `working_memory.active_slots()` (discovered mode + stability gate).
- Reinstatement read: `state/working_memory.integrate_discovered(reidentify=...)`
  rebinds a re-seen appearance to its `ent-NNNNN` id (passed from
  `cycle/neural_pipeline.py` as `ltm_graph.match`).
- Persistence across saves: `api/saved_agents/{store,routes}.py` copy `graph.sqlite`.

Invariants: working memory stays bounded (the "now"); the long-term graph is the
unbounded growth. Re-identification reuses ids (no duplicate nodes). The no-LTM /
oracle path is byte-identical (reidentify=None).
