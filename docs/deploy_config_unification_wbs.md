# WBS: Deploy/GPU tab config unification

**Version:** 1.0 — 2026-07-03 · **Trigger:** `DeploymentPanel.tsx` (Deploy/GPU tab) hardcodes its own
disconnected copies of scene, brain-preset, encoder, and Whisper-model lists instead of pulling
from the canonical sources the rest of the dashboard already uses. Investigation notes below;
implementation not yet started.

**Convention:** 1 d = one focused dev-day. ⚙ = needs Charles's machine (server/dashboard restart
to verify). Everything else is buildable/verifiable off-box (sandbox has the repo mounted,
`tsc --noEmit` and `pytest` both run there).

**Assumptions (clarifying questions were asked but the answer never came back — proceeding on the
recommended option from each, flagged below so they're easy to override):**
- Scene fix: full unification, not a minimal 3→6 patch. (Q1 recommended option)
- Brain preset: expose all 12 tiers with the existing heavy-tier warning, not a curated subset.
  (Q2 recommended option)
- DB-loss: tooltip **and** a visible warning banner near Stop/Destroy, not tooltip-only. (Q3
  recommended option)
- Legacy CLI `--scene` flag in `mujoco_decadic_adapter.py` (`LEGACY_SCENES`): left alone — it's a
  standalone manual-testing convenience, not part of either dashboard flow. (Q4 recommended option)

If any of these are wrong, say so before Phase U starts — U1–U3 are the ones that touch shared
code paths and are more annoying to unwind after the fact.

---

## Phase U — Unify source-of-truth lists (est. 1.5 d)

**U1. Scene dropdown → `/agent-presets`** (0.5 d)
`RunConfig.tsx`: replace the hardcoded `SCENES = { none, bear, food }` map with a fetch against
the same `GET /agent-presets` endpoint `AgentAdmin.tsx` already uses (`listAgentPresets()` in
`api.ts`). Render all 6 presets (`calm/forage/parent/village/predator/mind`) using their existing
`name` field as the label — same text as the top dropdown, verbatim.
*Acceptance:* GPU tab scene dropdown and top dropdown show identical option sets from a single
network call site (reuse `listAgentPresets`, don't duplicate the fetch logic). `tsc --noEmit`
clean.

**U2. Backend: deploy path resolves scenes via preset elements** (0.5 d)
`decadic/api/vast/controller.py::VastController._scene_elements()` currently hardcodes
`"bear" → [house, bear]`, `"food" → [house, food, water]`, and falls back to raw CSV-split for
anything else. Replace the two special cases with a lookup against the same `PresetStore` (or the
elements list handed over from the frontend directly, since U1 already resolves the preset
client-side) — one element-resolution path shared with `environment.py`'s local-agent flow instead
of three.
*Acceptance:* new unit test (there are currently zero tests covering `_scene_elements` or the vast
controller at all — this is the first) asserting each of the 6 presets resolves to the same
element list `PresetStore` would return locally, plus `none`/`mind` still resolves to `[]`.
Depends on U1 (deciding whether frontend sends preset id or elements CSV).

**U3. Brain preset dropdown → `neuralPresets.ts`** (0.5 d)
`RunConfig.tsx`: replace hardcoded `PRESETS = ["tiny","medium","full","xl"]` with
`NEURAL_PRESET_LABELS` from `dashboard/src/neuralPresets.ts` (already the canonical mirror of
`decadic/nn/config.py`, already consumed by `PresetPicker.tsx` and `AgentAdmin.tsx`). Add the
`Info` tooltip using `NEURAL_PRESET_INFO`, and call `heavyPresetWarning(preset)` to show the
GPU-cost warning inline when a 250m/500m/1b tier is selected — this matters more here than
anywhere else in the app, since selecting one starts real $/hr billing.
*Acceptance:* all 12 tiers selectable and match `PresetPicker.tsx`'s option set exactly (existing
`tests/test_neural_presets.py` stays the single backend source of truth; no new backend test
needed here, this is a frontend-only swap).

## Phase X — Explanatory tooltips (est. 1 d)

**X1. Shared explainer copy** (0.5 d)
Pull the encoder explanation out of `CognitionTogglesPanel.tsx:137` (currently a one-off inline
string) into a small shared module, e.g. `dashboard/src/explainers.ts`, alongside new copy for
Whisper model and disk sizing (written in X-something below). `CognitionTogglesPanel.tsx` switches
to importing it too, so there's one copy of the encoder explanation instead of two the moment this
lands.
*Acceptance:* `tsc --noEmit` clean; grep confirms no second hardcoded copy of the encoder
explanation string remains.

**X2. Wire `Info` tooltips into `RunConfig.tsx`** (0.5 d)
Add the existing `?`-badge `Info` component next to: Brain preset (already covered by U3), Encoder
(reuse X1 copy), Whisper model (new copy: what it trades off — download size / VRAM / transcription
latency vs. accuracy, and that it only matters when Encoder = hf and audio sensing is on), Scene
(new copy: what "elements" a scene actually spawns, linking back to the same wording the preset
picker uses), Disk (new copy, see D1).
*Acceptance:* visual check — every field on the Run configuration panel has a hoverable `?` with
non-empty, accurate text. No dead copy — reuse X1's constants, don't hand-write a second version of
anything already explained elsewhere in the app.

## Phase D — Disk sizing guidance + database-loss warning (est. 1 d)

**D1. Disk guidance tooltip, grounded in real measurements** (0.5 d)
Wrote this from actually measuring what's on disk today, not guessing:
base CUDA/PyTorch image + deps ≈ 6–10GB · hf encoder weights (CLIP+Whisper, downloaded once) ≈
1–2GB · brain checkpoint (`_brain.pt`) scales with preset — measured 0.7–1.9GB across `full`/`xl`
-tier local checkpoints; the 250m/500m/1b heavy tiers were never actually run to completion, so
their checkpoint size is an extrapolation (proportional to param count), not a measurement — flag
it as such in the copy · episodic SQLite grows unboundedly with session length (measured 4KB up to
83MB across current local agents; a multi-hour soak on a rented box should budget for tens of MB,
more for very long runs) · logs, small. Default 40GB has headroom for anything up to `xl`; call out
that `ultra` and above may need more.
*Acceptance:* tooltip text reviewed against this doc's numbers (no invented figures); flagged
uncertainty stays flagged in the UI copy, not silently presented as fact.

**D2. Database-loss warning** (0.5 d)
Backend fact already confirmed: `VastController`'s payload builder excludes `.sqlite`/
`.sqlite-journal` from both the upload and any download/checkpoint-capture path — only
`_checkpoint.json` + `_brain.pt` survive a Destroy. Add (a) an `Info` tooltip near the
Agent/restore field in `RunConfig.tsx` stating this plainly, and (b) per the Q3 assumption above, a
persistent (non-dismissable-by-accident) warning line in `ActiveDeployment.tsx` near the
Stop/Destroy buttons: destroying this instance permanently deletes its episodic memory; only the
neural weights are kept locally.
*Acceptance:* warning text present and visible without hovering in the active-deployment view;
wording confirms this is *current* behavior, not a hypothetical, since Charles may otherwise assume
memory synced by default.

## Phase V — Validation (est. 0.5 d dev + no paid machine time)

**V1. Regression pass** (0.25 d)
`tsc --noEmit` (dashboard) + `pytest -q` (touches `_scene_elements`, presets, settings_store —
none of the changes above touch neural/runtime code, so the full suite should be unaffected;
targeted run of `tests/test_agent_presets.py` + the new controller test from U2 at minimum).
*Acceptance:* both green.

**V2. ⚙ Manual UI check on Charles's machine** (0.25 d, no deploy $ spent)
Restart server + hard-refresh dashboard (same step needed for the earlier VastCredentials fix,
can be combined into one restart). Open Deploy/GPU tab, confirm: scene dropdown shows all 6 presets
matching top dropdown; brain preset shows all 12 tiers with heavy-tier warning on the last 3;
tooltips present on every field; DB-loss warning visible. **Does not require actually renting a
GPU** — everything here is checkable from the idle form state before hitting Search/Rent.
*Acceptance:* Charles confirms visually; no real Vast.ai spend needed for this pass.

---

## Totals and sequencing

Dev effort: **~4 focused days**. No paid GPU machine time required for this workstream (all
validation is UI/local-form-state — the first real-money test is whichever deploy Charles chooses
to run afterward using the corrected panel).

```
U1 -> U2 (needs U1's frontend decision: preset id vs elements CSV)
U1 -> U3 (independent of U2)
U1,U2,U3 -> X1 -> X2
X2, D1, D2 -> V1 -> V2
```

Critical path: U1 → U2 → X2 → V1 → V2. U3, D1, D2 can run in parallel with U2 once U1 lands.

## Explicitly out of scope

- **Legacy CLI `--scene` shim** (`LEGACY_SCENES` in `mujoco_decadic_adapter.py`) — left untouched
  per the Q4 assumption; it's a manual-testing convenience independent of both dashboard flows.
- **Dynamic/computed disk-size estimator** (a backend endpoint that inspects the chosen preset +
  any local checkpoint being restored and proposes an exact disk number) — D1 ships static
  guidance copy instead; worth a follow-up if the static numbers turn out wrong in practice.
- **Whisper model list expansion** (medium/large tiers) — out of scope; X2 documents the existing
  3-option list (tiny/base/small) as-is, doesn't add new models.
- **Audio toggle on the Deploy tab** — noted during investigation that none of the 6 built-in
  presets enable audio, so the Whisper-model field is currently always inert on this flow
  regardless of encoder choice; flagged here for awareness, not fixed as part of this WBS.
