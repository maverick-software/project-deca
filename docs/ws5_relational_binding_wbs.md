# WBS: WS5 — Relational Binding (Slots Across the Neural Boundary)

**Version:** 1.0 — 2026-07-04 · **Companion PRD:** `ws5_relational_binding_prd.md`
**Convention:** ordered by dependency only. ⚙ = needs Charles's machine (GPU run or live stack). Everything else buildable and testable off-box against seams + fixtures.

---

## M0 — Ground truth and adapters (no stack changes)

**M0.1 Slot/WM inventory freeze**
Document the exact `MemorySlot` field set, slot lifecycle (integrate/decay/evict), and the three pooling chokepoints (scene fold, memory mean-pool, fused z3) with file/line references; extract any magic dims into named constants.
*Acceptance:* inventory note in `docs/`; constants + layout unit test; no behavior change.

**M0.2 `WorkingMemory.slot_tensor()`**
Read-side adapter: salience-ranked deterministic ordering, fixed-dim per-slot projection (appearance ⊕ motion ⊕ scalars, zero-pad), (K, D_slot) float32 + validity mask. Pure function of existing slot state.
*Acceptance:* unit tests — determinism, mask correctness, stable order under ties, empty-WM shape. Depends on M0.1.

**M0.3 `retrieval_context_tokens()`**
Episodic adapter returning top-k hits as token matrix + mask (ranked hits already exist; no search changes). k=1 must reproduce today's best-hit semantics exactly.
*Acceptance:* unit + equivalence tests on both backends. Depends on M0.1.

**M0.4 Binding probe scaffold (world side)**
Synthetic client grows scripted `world_state.nearby_entities` with controlled appearance vectors and relation schedules (threat-adjacency patterns); WM-seam injection path for object files (bypasses the starved discovery pipeline; PRD 5.5).
*Acceptance:* entities appear as WM slots in a live run ⚙ (smoke, minutes); relation schedule serialized into the scenario file.

## M1 — Slot tensor into the stack (flag: `DECADIC_WM_SLOT_TENSOR`)

**M1.1 Stack plumbing**
`forward`/`top_down_perceive` accept optional slot tensor + mask; flag off ⇒ inputs identical to today (byte-identical baseline).
*Acceptance:* full suite green flags-off; shape/mask unit tests flags-on. Depends on M0.2.

**M1.2 WM keyed read**
Cross-attention block, query = f(z3_pre), keys/values = slot tensor; augments (not replaces) the scene-latent input per PRD open-decision default. Detach discipline matches existing loop conventions.
*Acceptance:* unit tests incl. gradient-isolation asserts (no new cross-cycle BPTT path); parity flags-off. Depends on M1.1.

## M2 — Memory tokens into the stack (same flag family)

**M2.1 Memory keyed read**
Cross-attention over `retrieval_context_tokens` replaces mean-pooling when on; k configurable, default = today's top-k.
*Acceptance:* unit tests; k=1 equivalence; parity flags-off; both backends. Depends on M0.3, M1.1.

## M3 — Relational core (flag: `DECADIC_RELATIONAL_CORE`)

**M3.1 Module**
Small pre-norm transformer over [slot tokens; memory tokens; interoceptive token]; pooled summary into the risk head input; flag-off weight-compatibility strategy decided and tested here (PRD open decision).
*Acceptance:* unit tests (masking, empty-slot degeneracy, determinism); full suite green both flag states. Depends on M1.2, M2.1.

**M3.2 ⚙ Cycle-cost measurement**
WS2 overhead harness on the full preset (CUDA), relational core on vs off; sizing decision (2×2 vs 3×4) made from measured numbers.
*Acceptance:* cost recorded in the PRD; fits the 70–90 ms envelope; sizing decision logged. Depends on M3.1.

## M4 — Graph keying + checkpoint compatibility

**M4.1 Entity-keyed slots**
On WM integrate, identity-matched slots key on the graph entity's appearance embedding (existing Kuzu match path); `entity_id` carried in slot metadata.
*Acceptance:* unit test — re-encountered entity re-binds to the same key across an occlusion gap (object-permanence assertion). Depends on M0.2; graph side is WS4-closed machinery.

**M4.2 Checkpoint/save-load compatibility**
Flags-off saves load flags-on (zero-init new blocks) and vice versa (drop); exercised through the WS4-M4 REST route tests.
*Acceptance:* round-trip tests in both directions on lance+kuzu defaults. Depends on M3.1.

## M5 — Binding probe and validation

**M5.1 Probe scenario + verdict script**
Train/test relation schedules with novel-pairing holdout, balanced so marginal statistics are uninformative (PRD risk: probe leakage); verdict script in the `check_gate_probe.py` mold — PASS = generalization to unseen pairings, FAIL = memorized pairs only.
*Acceptance:* scenario + verdict script reviewed against the leakage checklist; runs end-to-end on a stub agent. Depends on M0.4.

**M5.2 ⚙ Probe runs**
Flags-on vs flags-off (the built-in ablation): flags-off must fail novel pairings (it cannot represent them), flags-on must pass. Both runs archived like gate-probe artifacts.
*Acceptance:* the discriminating result, both directions; report in `reports/`. Depends on M3.2, M4.1, M5.1.

**M5.3 ⚙ Regression sweep + default decision**
Full suite both flag states; gate probe re-run (WS3 redesign spec) with flags on — the relational core must not perturb threat/calm/novelty verdicts; 1-h soak flags-on if defaults are to flip.
*Acceptance:* evidence recorded; default decision logged in the PRD (defaults stay off absent probe + soak evidence). Depends on M5.2.

---

## Dependency graph

```
M0.1 -> M0.2 -> M1.1 -> M1.2 --\
M0.1 -> M0.3 ---------> M2.1 ---+-> M3.1 -> M3.2 --\
M0.2 -> M4.1 -------------------------------------- +-> M5.2 -> M5.3
M3.1 -> M4.2 --------------------------------------/
M0.4 -> M5.1 --------------------------------------/
```

Critical path: M0.2 → M1.1 → M1.2/M2.1 → M3.1 → M3.2 → M5.2 → M5.3. M0.4/M5.1 (probe world-side) and M4.1 (graph keying) parallelize against the stack work.

## Explicit dependencies on other workstreams

- **Requires (partial):** nothing hard — WS5 is buildable and probe-testable on synthetic input via the WM-seam injection (M0.4). Full perception-side validation (slot attention over real visual input, discovery pipeline exercised) is **MuJoCo-era** and explicitly out of scope.
- **Feeds:** the learned gate (relational deliberation defines the true cost/benefit the gate must learn to price — sequence the gate after M3.2's cost numbers exist); theory-of-mind-era work (entity slots + self-model are its substrate).
- **Consumes:** WS4 (closed) — slot keys ride the Kuzu identity match; memory tokens ride the mirror's ranked hits. WS3 (redesigned probe) — M5.3 reuses it as a regression gate.
