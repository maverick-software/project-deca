# WBS: GPU offer search — tiered VRAM filter + Vast-style results

**Version:** 1.0 — 2026-07-03 · **Trigger:** the "Find a GPU" filter bar exposes `# GPUs` and
`Max $/hr` as raw inputs and a freeform `Min GPU RAM (GB)` number, and results render as a dense
table instead of the grouped card layout Vast's own site uses. Field names below are sourced from
Vast's public API docs (see WBS drafting response for links), not guessed — two items are flagged
explicitly as unconfirmed and gated behind Phase D0.

**Convention:** 1 d = one focused dev-day. ⚙ = needs Charles's machine (live Vast.ai API
key/CLI — the sandbox has neither, so nothing in this workstream that touches real marketplace
data can be verified here beyond synthetic unit tests).

**Locked-in decisions from the planning discussion (no objection raised — proceeding on these):**
- `# GPUs` is removed entirely; every search hardcodes `num_gpus=1`. The neural stack has no
  multi-GPU/sharded training path, so renting more than one GPU has no use today, and locking to
  1 GPU also makes `gpu_ram` (per-card) and `gpu_total_ram` (aggregate) identical — sidestepping
  the ambiguity that produced the odd 140GB/179GB numbers in the first place.
- `Max $/hr` is dropped as a hard filter. Default sort stays perf-per-dollar first (existing
  `dlperf_usd-` order), and price stays visible on every result, so cost-consciousness isn't lost,
  just not gated on a preset number.
- `Min GPU RAM (GB)` becomes a tiered "at least" select: 10, 12, 16, 24, 32, 48, 64, 80, 96, 128,
  240 GB (Charles's list plus 48/80, common real datacenter-card sizes he may have just omitted —
  trivial to trim in F1 if unwanted).
- Results become a card list matching Vast's own grouping (icon+model header, spec line, CPU/network
  line, badges, price+Rent), not a 1:1 visual clone, just the same information density/grouping
  adapted to the dashboard's dark theme.

---

## Phase D0 — Discovery (est. 0.1 d dev + a few minutes of Charles's machine time)

**D0. ⚙ Capture one real raw offers sample**
Run (on Charles's machine, using the existing `/vast/offers` endpoint or `vastai search offers
"rentable=true" --raw --limit 5`) to resolve the two things the public API docs didn't pin down
precisely enough to code against blind:
1. **`inet_up`/`inet_down` units.** Docs say MB/s; Vast's own displayed numbers (screenshot: "↑3047
   Mbps") look ~8x smaller than a straight MB/s→display mapping would produce, suggesting the site
   shows Mbps. Confirm by comparing a raw sample's `inet_up` value against what Vast's own site
   shows for the same offer.
2. **`hosting_type` meaning and a "days remaining" source.** The docs example shows `hosting_type: 0`
   with no legend; confirm whether 0/1 (or another encoding) maps to community/datacenter. No
   `max_duration` field exists in the documented schema — `end_date` (unix timestamp) is documented
   and is the likely source for Vast's "Max Duration" badge (`end_date - now`); confirm this
   produces sane day counts against a real offer.
*Acceptance:* both answers written into this doc (or B1's code comments) before Phase B starts.
Nothing else in this WBS is blocked on D0 except the exact transform in B1 — F1 (filter bar) has
no dependency on it and can proceed in parallel.

## Phase F — Filter bar (est. 0.5 d)

**F1. `GpuSearch.tsx`: drop `# GPUs`/`Max $/hr`, tiered VRAM select**
Remove the `# GPUs` and `Max $/hr` `<input>`s and their state; `searchOffers()` call always passes
`num_gpus: 1` and omits `max_dph`. Replace the `Min GPU RAM (GB)` number `<input>` with a
`<select>` offering `Any` plus the tier ladder above, each option's value being the plain GB
number; `onChange` still calls `searchOffers({ min_gpu_ram: value || undefined, ... })` exactly as
today (the backend's `gpu_ram>=N` filter is unchanged — this is a frontend input-type swap only,
per the earlier investigation).
*Acceptance:* `tsc --noEmit` clean; manual check that the existing (pre-redesign) results table
still populates correctly with the new filter shape, so this phase can ship independently of R1 if
needed.

## Phase B — Backend field passthrough (est. 0.5 d dev; final numeric transforms depend on D0)

**B1. Extend `_normalize_offer()` and add a unit test**
Add to the normalized offer dict: `total_flops`, `inet_up`/`inet_down` (unit-corrected per D0's
finding), `cpu_name`, `cpu_cores`, `direct_port_count`, `host_id`, `machine_id`, `is_datacenter`
(derived from `hosting_type` per D0), `days_remaining` (derived from `end_date` if present, else
`None` — never invented). New unit test in `tests/test_vast_routes_normalize.py` (first test file
for this function) feeding a synthetic raw dict shaped like Vast's own documented example and
asserting every new field maps correctly, including the `None` fallback paths when a field is
missing from the raw payload (real offers won't always populate every optional field).
*Acceptance:* new test passes (runnable in the sandbox — pure dict transform, no torch/CLI
dependency, verified the same way `test_vast_controller_scene.py` was in the prior workstream).

**B2. `VastOffer` TS type + `searchOffers()` params**
Add the new fields to `VastOffer` in `vastApi.ts`. Drop `num_gpus`/`max_dph` from the
`searchOffers()` params object the frontend actually sends (backend keeps accepting them
un-wired, for compatibility and manual/curl testing — no backend signature break).
*Acceptance:* `tsc --noEmit` clean.

## Phase R — Results card redesign (est. 1 d)

**R1. New card component**
Replace the `<table>` in `GpuSearch.tsx` with a card per offer (own component if it keeps
`GpuSearch.tsx` readable): header row (hardware icon + "Nx `gpu_name`" + verified/datacenter badge +
days remaining + Rent button, right-aligned like Vast's own layout), a spec line (TFLOPS · VRAM ·
GPU memory bandwidth), a CPU/network line (`cpu_name` + `cpu_cores` cores · ↑/↓ bandwidth · port
count), a metrics line (DLPerf · DLPerf/$ · reliability %), and the price prominently next to Rent.
Dark-theme styling consistent with the rest of the dashboard — not a literal light-mode reskin of
Vast's site, just the same grouping/density.
*Acceptance:* visual review against a real search (V2) — this is the one piece of this workstream
that most needs live data to judge properly, since column alignment/wrapping with real GPU names
and multi-line CPU names can't be fully judged from synthetic data alone.

**R2. Wire into `GpuSearch.tsx`**
Swap the table render for the card list; keep the existing empty/error states.
*Acceptance:* `tsc --noEmit` clean.

## Phase V — Validation (est. 0.4 d)

**V1. Regression pass**
`tsc --noEmit` (dashboard) + the new `test_vast_routes_normalize.py` (and the existing
`test_vast_controller_scene.py`, unaffected by this workstream but cheap to re-run alongside it).
*Acceptance:* both green.

**V2. ⚙ Live check on Charles's machine**
Restart server, hard-refresh dashboard, run a real search with the API key already on file.
Confirms: D0's two unit/field assumptions held, the card layout reads well against real GPU
names/CPU names of varying length, and the VRAM tier filter actually narrows results the way "at
least N GB" should.
*Acceptance:* Charles confirms visually; no rental required to validate this (Search only, not
Rent).

---

## Totals and sequencing

Dev effort: **~2.5 focused days**. Machine time: a few minutes for D0's sample capture + V2's
live-search check — no paid GPU rental needed anywhere in this workstream.

```
D0 -----------------------\
F1 (independent of D0) ---+--> B1 -> B2 -> R1 -> R2 -> V1 -> V2
```

Critical path: D0/F1 → B1 → B2 → R1 → R2 → V1 → V2. F1 can ship on its own first if you want the
filter-bar improvement live before the card redesign is ready.

## Explicitly out of scope

- **Multi-GPU rental UI** — locked out by design (`num_gpus=1` always); revisit only if the
  project gains a sharded/multi-GPU training path.
- **Price sort toggle / advanced sort control** — Vast's screenshot didn't show one and it wasn't
  requested; default perf-per-dollar ordering stays as today. Easy follow-up if wanted later.
- **`type` param (ondemand/bid/reserved)** — stays on-demand only; interruptible "bid" pricing
  isn't relevant to keeping one agent's brain running continuously.
- **Bandwidth/storage cost fields** (`inet_up_cost`, `inet_down_cost`, `storage_cost`) — not
  surfaced; the existing Disk (GB) tooltip from the prior workstream already covers storage cost
  context at a high level.
