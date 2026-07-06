# Gap Analysis — Preset-Activated Foraging Curriculum in the Environment Tab

**Goal.** Turn the foraging-curriculum *behaviour* (currently the external
`scripts/run_foraging_curriculum.ps1`) into a **server-side loop that a scenario
preset switches on**, selectable and tunable from the dashboard's **Environment
tab** — no terminal, no external script.

**One-line answer to "can a preset turn on a loop?"** Yes. A preset is static
config, but it can carry a `curriculum` flag; the `EnvironmentSupervisor` (which
already owns the body lifecycle) reads that flag on start and runs an asyncio
background loop for as long as the scenario is alive. The preset is the *switch*;
the supervisor is the *engine*.

---

## 1. Target behaviour

Selecting **"Foraging trainer"** in the Environment tab (or the top-bar preset
dropdown) and pressing Start should:

1. Spawn a food+water(+medical) body scenario in **metabolic** mode (hunger →
   deficit → the deficit-gated foraging drive; immortal would zero it).
2. Run **contact consumption** so meals are earned by a reach (already the
   default).
3. Start a **server-side loop** that, on a cadence: places the *most-depleted*
   resource **within reach**, and **rescues** any reservoir that nears the fatal
   floor (the survival net that keeps a metabolic agent alive without pinning it
   full).
4. Surface live progress (earned meals, rescues, `successor_value`) and let the
   user tune cadence / rescue floor / contact radius, and toggle the curriculum
   off, from the Environment tab.
5. Stop cleanly when the scenario is stopped/replaced/deleted (no orphan loop).

---

## 2. Current state (what already exists)

| Piece | Where | State |
|---|---|---|
| Preset store (JSON, seeded builtins, migration) | `decadic/api/presets/store.py` (`PresetStore`, `BUILTIN_PRESETS`, `SCHEMA_VERSION=2`) | present; schema = `{id,name,elements,vision,audio,braces,mind_only,builtin}` — **no curriculum field** |
| Built-in "forage" scene preset (food+water) | `store.py` BUILTIN_PRESETS `id:"forage"` | present, scene-only |
| Single-slot body lifecycle | `decadic/api/environment.py` (`EnvironmentSupervisor.start/stop/delete/pause/resume`) | present; **no background curriculum task** |
| Start route + request | `app.py:947 start_environment`, `EnvironmentStartRequest` schema | present; carries elements/senses/braces only |
| Within-reach placement | adapter `give_within_reach` + body cmd `give_{res}_reach` (WS-FORAGE M0) | present |
| Contact consumption (earned meals) | adapter `touched_now`, `DECADIC_BODY_CONSUME_MODE=contact` (default) | present |
| Admin rescue credit | `agent.give_resource(res, amount)` (async) | present |
| Reservoir + progress metrics | `agent.metrics` → `hydration/energy/integrity/viability/consume_events`; `successor_value` in metrics | present |
| Metabolic/immortal toggle | Homeostasis panel → `configureAgent(viability_mode)`; per-agent attr | present (live toggle, not a preset field) |
| Environment tab UI (compose/start/save preset) | `dashboard/src/components/EnvironmentPanel.tsx`; `App.tsx presetToDraft`, `ScenarioDraft`; `api.ts` types + `startEnvironment` | present; **no curriculum control** |

**Reading:** every *primitive* the loop needs already exists (within-reach
placement, contact consumption, rescue credit, reservoir metrics, a body
lifecycle to hang the loop on). Nothing new is needed at the mechanism level —
the gaps are (a) a flag on the preset, (b) a loop in the supervisor, and (c) a
control + readout in the Environment tab.

---

## 3. Design decision (surface first, so the gaps are unambiguous)

Support **both** entry points, because the user asked for a preset *and* the
Environment tab:

- The **preset** carries a *default* `curriculum` (so "Foraging trainer" starts
  the loop with sensible defaults on selection).
- The **Environment tab** exposes a `curriculum` selector (`none | foraging`) +
  its parameters, editable when composing a scenario and savable into a
  user preset — so it isn't locked to one built-in.

`curriculum` is a small typed record: `{ kind: "none"|"forage", place_every_s,
rescue_floor, rescue_to, contact_radius, reach_distance }`. Absent/`none`
everywhere = today's behaviour (full back-compat).

---

## 4. Gaps, component by component

### G1 — Preset schema (backend) · `decadic/api/presets/store.py`
- Add `curriculum` to `_normalise` (default `{"kind":"none"}`), preserving it
  through read/migrate/create.
- Add a `forage_trainer` built-in: `elements:["food","water"]`,
  `curriculum:{"kind":"forage", …defaults}`.
- Bump `SCHEMA_VERSION` → 3 so `_migrate` re-seeds builtins (adds the new preset)
  while preserving user presets. **Risk:** migration must not drop the new field
  on user presets — extend `_normalise` before bumping.

### G2 — Supervisor curriculum loop (backend) · `decadic/api/environment.py`
- `start(...)` gains `curriculum: dict | None`; store it; if `kind=="forage"`,
  **force the agent to metabolic** (override an immortal env default) and launch
  `self._curriculum_task = asyncio.create_task(self._run_forage_curriculum(agent_id, params))`.
- `_run_forage_curriculum`: every `place_every_s` — read `agent.metrics`, pick the
  lowest of hydration/energy/integrity, `agent.queue_body_command(f"give_{res}_reach")`;
  for any reservoir `< rescue_floor`, `await agent.give_resource(res, rescue_to - v)`.
  Guard every tick with `self._agent_id == agent_id` and tolerate a not-yet-ready
  body (queue failures are non-fatal).
- Cancel the task in `_terminate_locked` (covers stop/replace/delete); include
  `curriculum` + a small live counter (meals earned since start, rescues) in
  `_status_dict`.
- **Also pass `DECADIC_BODY_CONSUME_MODE=contact` explicitly to the adapter
  subprocess** for a foraging curriculum, so a proximity override in the server
  env can't silently defeat "earned meals."

### G3 — Start route + request (backend) · `decadic/api/app.py`
- `EnvironmentStartRequest` gains `curriculum: dict | None`; `start_environment`
  passes it to `sup.start(...)`. Preset-list route already returns whatever the
  store holds, so `curriculum` surfaces once G1 lands.

### G4 — Environment tab + types (frontend)
- `api.ts`: add `curriculum` to `AgentPreset`, `CreateAgentPresetRequest`,
  `ScenarioDraft`, `EnvironmentStartRequest`; `startEnvironment` forwards it.
- `App.tsx presetToDraft`: carry `curriculum` from preset → draft.
- `EnvironmentPanel.tsx`: a **Curriculum** control (select none/foraging + a few
  number inputs for cadence/floor/radius, shown only when foraging), wired via
  `updateDraft`; Save-preset persists it; Start sends it. A **live readout**
  (earned meals / rescues from `status`, and `successor_value` from metrics) so
  the user can watch learning.
- **Guard:** if `curriculum.kind=="forage"` and the user picks Immortal, warn /
  auto-switch to Metabolic (immortal zeroes the drive).

### G5 — Telemetry / verdict
- Reuse the script's headline signals in the tab: `consume_events` delta (earned
  meals climbing = it's reaching food) and `successor_value` (should crawl off
  ~0 as it learns). Optional: persist a per-run forage summary under `reports/`.

---

## 5. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Orphan loop acting on a dead/replaced agent | Task holds `agent_id`; checks `self._agent_id==agent_id` each tick; cancelled in `_terminate_locked`. |
| R2 | Immortal + foraging (no drive) | Force metabolic on a forage curriculum; UI warns/auto-switches. |
| R3 | Proximity override defeats "earned meals" | Supervisor passes `DECADIC_BODY_CONSUME_MODE=contact` to the adapter for forage curricula. |
| R4 | Preset migration drops the new field / breaks user presets | Extend `_normalise` first; version bump only after; unit-test round-trip incl. user presets. |
| R5 | Body not ready when the loop first ticks | Tolerate `queue_body_command`/metrics failures; first successful tick just starts later. |
| R6 | Frontend/backend contract drift | `curriculum` optional everywhere; missing ⇒ `none` ⇒ today's behaviour (back-compat). |
| R7 | Loop cost on the server event loop | Cadence is seconds; work per tick is one metrics read + one queued command — negligible, off the cognition path. |
| R8 | Starvation before it can reach (calibration) | Rescue floor/`to` + contact radius are tunable in the tab; sane defaults (floor 12 → 35, radius 0.35). |

## 6. Sequencing

1. **G1** preset schema + `forage_trainer` builtin (back-compat, unit-tested).
2. **G2** supervisor loop + lifecycle + metabolic/contact enforcement.
3. **G3** route/request passthrough.
4. **G4** Environment-tab control + live readout + preset save/apply.
5. **G5** telemetry polish.
Each backend step is testable without the frontend; the frontend lands last.

## 7. Test plan

- **Unit (backend):** preset round-trip incl. `curriculum` + user-preset
  preservation across the version bump (`test_environment_supervisor.py` /
  presets test); a supervisor-loop test with a **fake registry/agent** asserting
  it places the most-depleted resource and rescues below the floor, and that the
  task is cancelled on stop/replace (no orphan).
- **Contract:** `EnvironmentStartRequest` accepts/forwards `curriculum`; absent ⇒
  `none` ⇒ byte-identical start path (parity).
- **Frontend:** type-check; a render test that the Curriculum control appears and
  Start sends the field.
- **Live smoke:** select "Foraging trainer" → metabolic, food appears within
  reach on cadence, `consume_events` increments on an earned reach, a near-fatal
  reservoir gets rescued, and Stop cancels the loop.

## 8. Effort / surface summary

Backend: 3 files (`store.py`, `environment.py`, `app.py`), all additive and
back-compatible. Frontend: 3 files (`api.ts`, `App.tsx`, `EnvironmentPanel.tsx`),
one new control + type threading. No new subsystem — it wires existing primitives
(within-reach, contact, rescue, metrics) onto the existing body lifecycle, gated
by one preset field. The external `run_foraging_curriculum.ps1` becomes the
reference/back-up path; the tab becomes the primary surface.
