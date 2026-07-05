# PRD + WBS: WS8 — The Agent Library (Unified, Versioned, Durable)

**Version:** 1.0 — 2026-07-04 · **Status:** Draft for review
**Scope note:** smaller than WS5–WS7; PRD and WBS share this document.

**Thesis:** one library, two kinds of entries. **Templates** are birth configurations (spawn = a fresh mind). **Masters** are trained minds (spawn = a clone with a biography). A saved agent can be *promoted* into the library — as a new Master entry or as the next version of an existing one — and nothing in the library can be lost by accident: overwrite supersedes, never destroys; deletion tombstones, never erases; every artifact is hash-verified against a registry database.

---

## 1. Gap analysis (2026-07-04 code inventory)

| Capability | Where | State |
|---|---|---|
| Template agents | `decadic/api/presets/` — JSON store, seeded built-ins, `/agent-presets` CRUD, user presets persisted, built-in migration logic | **Built**, but isolated: knows nothing of trained minds |
| Saved (trained) agents | `decadic/api/saved_agents/` — filesystem bundles (`manifest.json`, `brain.pt`, `state.json`, episodes/graph snapshots), save/load/delete routes, route-proven on lance+kuzu | **Built**, but flat: no versioning, no promotion, `DELETE` is a real `rmtree`, durability = "the folder still exists" |
| The bridge (promotion) | — | **Missing** entirely: a trained mind cannot become a template-like library citizen |
| Durability | — | **Missing**: no integrity hashes, no registry, no tombstones, no export/backup story; one bad `rmtree` or disk hiccup loses a trained mind silently |
| Display | Two separate dashboard panels (presets tab, saved-agents tab) | **Fragmented**: no unified view, no lineage, no versions, no certification status |
| Provenance hooks | Saved-agent manifest carries faculties/preset/cycle (WS5-M4.2 made it self-describing) | **Partial**: no ancestry chain, no skill/certification tags (WS7 will produce these) |

## 2. Design

### 2.1 One registry, two entry kinds
`library.sqlite` (WAL) — the metadata database. Tables: `entries` (id, kind `template|master`, name, description, current_version), `versions` (entry_id, version, created_at, provenance: source agent id + parent save id + ancestry entry id, faculties/preset snapshot, artifact manifest with **sha256 per file**, status `active|superseded|tombstoned`), `certifications` (version_id, battery id, verdict, report path — the WS7 hook). SQLite because metadata is small, transactional, and boring; the minds themselves do not go in blobs — see 2.2.

### 2.2 Content-addressed artifact store (the durability mechanism)
`library/objects/<sha256-prefix>/<sha256>` holds every artifact file exactly once (brains, state JSONs, zipped episode/graph snapshots — lance/kuzu snapshot *directories* are zipped at promote time so every artifact is a hashable file). Registry rows reference artifacts by hash. Consequences, each load-bearing:
- **Integrity:** every read verifies sha256; corruption is detected, named, and non-silent.
- **Dedup:** two versions sharing a brain store it once.
- **Overwrite = supersede:** promoting onto an existing entry writes a NEW version and flips `current_version`; the old version's artifacts remain addressable. Rollback is a registry update, not a restore.
- **Tombstone deletes:** `DELETE` marks status; artifacts are only physically removed by an explicit `library gc` command that refuses to collect anything referenced by ANY non-tombstoned version, and prints what it would remove before doing it.
- **Export/import:** `library export <entry>` produces one portable archive (registry rows + objects); import verifies hashes. This is also the off-machine backup story.

### 2.3 Promotion flow
`POST /library/promote {save_id, mode: "new"|"overwrite", entry_id?, name?, notes?}`: reads the saved-agent bundle → zips directory snapshots → hashes everything into the object store → writes a `master` version row with full provenance (source agent, save id, ancestry: which library entry it was originally spawned from, carried in the agent/save manifest from now on) → flips current_version if overwriting. The reverse path (spawn) already exists: templates spawn via preset routes; masters spawn via the saved-agents load path, now reading from the object store. Existing `saved_agents/` bundles migrate in with a one-shot `library migrate` command; the legacy routes keep working against the library backend.

### 2.4 Display — one Library panel
Card grid, one card per entry: kind badge (Template / Master), name, version chip with history dropdown, lineage line ("cloned from Walker-Master v3 → trained 2026-07-04"), certification badges from WS7 verdicts (skill + battery date), storage size, integrity status (last verify). Actions per card: **Spawn** (template: fresh; master: clone), **Promote** (visible on live/saved agents, targets this panel), **Export**, **History** (version list with rollback), **Tombstone**. Filters: kind, skill tag, certification status. The two existing tabs collapse into this panel; `/agent-presets` and `/saved-agents` REST surfaces stay as delegating aliases (the /dojo→/dream pattern).

### 2.5 What this unlocks downstream
This registry IS the commercial registry from the WS7 pipeline: master images with provenance and certification verdicts are exactly what provisioning clones from. WS7's `provision.py` reads from and promotes into this library; the Dream Trainer's package registry can share the same object store later.

## 3. Success criteria

1. Promote round trip: live agent → save → promote (new) → spawn from library → identical mind (WS5-M4.2 route test pattern extended to the library path).
2. Overwrite supersedes: promote onto an entry twice; both versions listed; rollback to v1 restores byte-identical artifacts; nothing deleted.
3. Integrity: flip one byte in an object file → next load FAILS with a named-artifact diagnostic (never a silent bad mind).
4. Tombstone + gc: deleted entry recoverable before gc; gc refuses referenced objects; gc prints its plan first.
5. Crash safety: kill the process mid-promote → registry shows no partial version; object store may hold orphans (gc-collectable), never dangling references.
6. Migration: existing `saved_agents/` and presets appear in the unified panel; legacy routes still green in the suite.
7. Export/import round trip across a simulated fresh machine (fresh temp dirs).

## 4. WBS (dependency order)

- **M0 Registry + object store.** `decadic/library/` — schema, content-addressed writes (temp-file + rename for atomicity), sha256 verify-on-read, tombstones, `gc` with plan-print, export/import. Unit tests incl. criteria 3–5 (corruption, crash-mid-write via injected failure, gc refusal).
- **M1 Promotion + migration.** Promote route (new/overwrite), ancestry carried in agent/save manifests, `library migrate` one-shot, spawn-from-library path; criteria 1–2 route tests; legacy delegating aliases.
- **M2 Unified `/library` API.** List/detail/history/rollback/export endpoints; certifications table + WS7 attach hook (verdict reports link to versions).
- **M3 Library panel (dashboard).** Card grid per 2.4; the two old tabs redirect. ⚙ eyeball pass on the live dashboard.
- **M4 Durability drills + docs.** Scripted drill (`scripts/library_drill.py`) running criteria 3/4/5/7 against a scratch library — becomes a suite test AND an operator runbook (`docs/runbooks/library-recovery.md`, WS7-M7 house style).

Dependencies: consumes Saved Agents + presets (built) and WS4 snapshot mechanics (built). Feeds WS7 provisioning. Independent of WS5/WS6 stack work — schedulable whenever infra capacity exists.

## 5. Risks

- **Zipping large snapshots at promote time** (multi-GB masters): stream-zip with progress + resumable temp staging; measured before defaults, per house rule.
- **Two sources of truth during migration:** the migrate command is one-way and idempotent; legacy dirs become read-only after a successful migrate (marker file), preventing drift.
- **Registry/object-store divergence:** writes are ordered (objects first, registry row last, atomic rename); the drill's crash test pins this.
