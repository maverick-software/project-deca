# Serial Cognition + Prefetch Perception — Gap Analysis & WBS

**Status:** as-built vs. intended; remediation plan
**Scope:** the observation-ingestion / "parallel processing" path feeding the Decadic cognitive cycle
**Key files:** `decadic/cycle/stage_pipeline.py`, `decadic/agents/runtime.py` (perception workers + cycle loop), `decadic/cycle/neural_pipeline.py` (`run_neural_cycle`, `pool_fused`), `decadic/perception/scene_workspace.py`, `decadic/config.py` (`processing_mode`)

---

## 1. Target specification (the system you described)

| ID | Requirement | Plain statement |
|----|-------------|-----------------|
| **R1** | Serial logical pipeline | Exactly **one** observation is deep-processed by the cognitive cycle at a time, as a single coherent logical pass. (You do **not** want concurrent cognition across stages.) |
| **R2** | Persistent reception | Observations are received continuously; the receiver never blocks waiting on cognition. |
| **R3** | No information loss | **Every** observation's perceptual evidence is folded into the mental model, in arrival order — "ready to add its information to the equation." |
| **R4** | Prefetch / latency hiding | While observation *N* is in cognition, observation *N+1* is **decoded/encoded concurrently** so it is ready the instant cognition frees up. |
| **R5** | Correct order & timing | Observations are integrated in arrival/timestamp order — "arriving at the correct time." |
| **R6** | No wasted time | The serial mind is never idle waiting on decode; the producer never stalls on cognition. |

The mental model: a **serial consumer** (the logical pipeline) fed by a **persistent, lossless, prefetching producer** (perception). "Observation 2 waits behind observation 1, ready" = a staged, ordered ready-queue, with the next frame already prepared.

---

## 2. As-built (current implementation)

Default mode is `stage_pipeline` (`DECADIC_STAGE_PIPELINING_ENABLED` defaults on → `processing_mode()` returns `PROCESSING_STAGE_PIPELINE`).

**Stage pipeline (`stage_pipeline.py`).** Each observation becomes a `DecadicSession` and advances through ten async stage-queue workers (1→10), then enters a `ready` pool. A `DecadicCommitArbiter` selects **one** ready session by `(urgency, salience, oldest)`; the runtime then runs the real `run_neural_cycle` on that **single selected** observation.

**Perception workers (`runtime.py`).** A pool of `_perception_worker_loop` coroutines pull observations off a queue, call `predecode(obs)`, and commit object-files to the scene workspace in sequence order under `self.lock`.

**Scene workspace (`scene_workspace.py`).** A stateful, decaying, re-identifying egocentric scene model, updated per committed frame.

**What is genuinely good and reusable:** the `DecadicSession` snapshot/immutability machinery, the ordered-commit sequencing (`_perception_next_commit`), the scene workspace itself, and the rich per-stage/per-session telemetry. The *structure* of "sessions queue behind one another" is the right shape.

**What it actually does, in substance:**

1. The "stage workers" do **placeholder bookkeeping**, not cognition. `_run_stage_candidate` does `await asyncio.sleep(0)` and emits candidate metadata (`frame_seen`, `event_count`, `urgency`, `salience`, `"defer_to_commit_cycle"`). Module docstring: *"Stage work is candidate-only and snapshot-based."* The real ten-stage neural cognition runs **once, serially**, on the single arbiter-selected observation.
2. It is **selection-with-drop**, not lossless ingestion. The arbiter picks one ready session; others that sit longer than `stale_after_s = 2.0` go **stale and are discarded** whole; queues drop on capacity overflow. Un-selected observations' information **never enters the model**.
3. There is **no real time-overlap**. Stage workers do ~zero work; the heavy `predecode` runs **synchronously** (no `asyncio.to_thread` — the only `to_thread` in `runtime.py` is in the metabolic loop), so decode and cognition share one event loop and cannot run at the same wall-clock instant.

---

## 3. Gap analysis

| Req | As-built | Gap | Severity |
|-----|----------|-----|----------|
| **R1** Serial cognition | `run_neural_cycle` runs serially on one observation (no pooling in stage mode) | **Met.** Keep as-is. | — |
| **R2** Persistent reception | Bounded queue exists; ingestion enqueues sessions | Partial — producer work runs on the cognition event loop, so heavy ingestion still contends with cognition | Medium |
| **R3** No information loss | Arbiter selects one; stale (2 s) + overflow drops discard whole sessions without folding their evidence | **Not met.** Frames are lost by design. This is the primary gap. | **High** |
| **R4** Prefetch overlap | `predecode` is synchronous on the event loop; no producer thread | **Not met.** Decode of *N+1* does not overlap cognition of *N*; no latency hiding. | **High** |
| **R5** Order & timing | Perception path commits in seq order; stage arbiter selects by salience, not order | Partial — deep-processing order is salience-driven, not arrival order | Medium |
| **R6** No wasted time | Decode not overlapped; stale frames discarded | **Not met.** Time is wasted (serialized decode) and frames are wasted (stale-dropped). | **High** |
| **—** Stages do cognition | Stages emit placeholder metadata | Misaligned scaffolding: a *systolic-cognition* pipeline was built where a *prefetch buffer feeding serial cognition* was intended | Medium |

**One-line verdict:** the serial consumer (R1) is correct; the producer side (R3, R4, R6) is not — it neither overlaps decode with cognition nor preserves every frame's information, and the elaborate per-stage pipeline is solving a problem you didn't ask for.

---

## 4. Target architecture (corrected)

```
            ┌─────────────────────────── PRODUCER (own thread) ───────────────────────────┐
 obs stream │  decode + frozen-encode (CLIP/Whisper)  →  fold evidence into SceneWorkspace │
   ───────► │      (runs CONCURRENTLY with cognition; native ops release the GIL)          │
            └───────────────┬───────────────────────────────────┬─────────────────────────┘
                            │ every frame folded (R3, no loss)   │ decoded frame enqueued
                            ▼                                     ▼
                    Persistent SceneWorkspace            Ordered ready-queue (bounded)
                    (the always-current model)           "obs N+1 waiting, pre-decoded"
                            ▲                                     │ pull next in order
                            │ read maintained model               ▼
            ┌───────────────┴──────────── CONSUMER (serial) ──────────────────────────────┐
            │   run_neural_cycle on ONE observation at a time, in order  (R1, R5)          │
            └─────────────────────────────────────────────────────────────────────────────┘
```

**Two layers, mirroring the brain you described:**

- **Pre-attentive producer (lossless):** decodes/encodes every frame off the cognition thread and folds its evidence into the persistent `SceneWorkspace` immediately and in order. Nothing is lost to the model even if not every frame gets a full cognitive pass.
- **Serial attentional consumer:** the existing serial cycle, pulling the next prepared observation in order.

**The decision you must make explicitly (Phase 0).** Input rate and processing rate cannot both be honored without limit: if frames arrive faster than the serial mind can deep-process, you must choose one of —

- **(A) Strict every-frame-deep-processed, in order.** No drops; the consumer falls behind under load (latency grows). Only safe if input ≤ processing rate (e.g. throttle the camera to the cycle rate).
- **(B) Lossless model, attentional deep-processing (recommended, brain-faithful).** The producer folds *every* frame into the scene model (R3 satisfied — no information loss), while the serial consumer deep-processes as many as it can keep up with, **coalescing** skipped frames rather than discarding them. Under overload you lose *cognitive passes*, never *information*.

The rest of this WBS targets **(B)** with **(A)** available as a config-flagged throttle.

---

## 5. Work Breakdown Structure

Effort is rough order-of-magnitude in developer-days. Dependencies in brackets.

### Phase 0 — Decision & baseline (1–2 d)
- **0.1** Confirm the R1–R6 spec with stakeholder; choose policy (A) or (B). *Deliverable:* signed-off one-pager. *Acceptance:* explicit chosen policy + target input/cycle rates.
- **0.2** Instrument a baseline run: record `committed/dropped/stale` (stage pipeline `metrics()`), `commit_lag_ms`, cycle Hz, decode ms. *Deliverable:* baseline numbers. *Acceptance:* a reproducible before-snapshot to measure the fix against.

### Phase 1 — Real prefetch producer (4–6 d) [0]
- **1.1** Move `predecode` (and, ideally, the frozen CLIP/Whisper encode) into a worker that runs via `asyncio.to_thread` / a dedicated `ThreadPoolExecutor`, so it overlaps cognition. *Files:* `_perception_worker_loop`, `frozen_encoders.predecode`. *Acceptance:* with `parallel_sessions ≥ 2`, decode wall-time overlaps a busy cognitive cycle (measured: producer-active time during `run_neural_cycle` > 0).
- **1.2** Make the producer→model handoff thread-safe: the scene-workspace mutation must be the only locked section; the heavy encode must be **outside** `self.lock`. *Files:* `_commit_perception_observation_locked`, `_drain_ready_perception`. *Acceptance:* no data race under a 2-thread stress test; scene snapshots remain consistent.
- **1.3** Add an **ordered ready-queue** of fully-decoded observations the consumer pulls from (the "obs N+1 waiting, ready" buffer). *Acceptance:* the consumer never calls `predecode` itself; on cycle start the next frame is already decoded (decode-on-consume time ≈ 0).

### Phase 2 — No-loss ingestion (5–7 d) [1]
- **2.1** Implement **fold-before-drop**: before any session is dropped (capacity or staleness), its perceptual evidence is integrated into the `SceneWorkspace` / object-files / LTM. *Files:* `stage_pipeline.py` (`enqueue_observation`, stale path), `scene_workspace.update`. *Acceptance:* `information_loss == 0` — every ingested frame increments a `frames_folded` counter even when not deep-processed.
- **2.2** Replace the salience-only `DecadicCommitArbiter` with an **order-respecting** selector: default FIFO (arrival order, R5); urgency may pre-empt (fast-path threats) but pre-empted frames are still folded, never lost. *Files:* `DecadicCommitArbiter.select`. *Acceptance:* under steady load, deep-processing order == arrival order; under threat, urgent frames jump the queue but folded counts stay complete.
- **2.3** Remove unconditional 2 s `stale` whole-discard; replace with "fold + coalesce." *Acceptance:* `stale_sessions` that lose information drops to 0; a new `coalesced_sessions` counter accounts for skipped-but-folded frames.

### Phase 3 — Ordered serial consumption (2–3 d) [2]
- **3.1** Drive the serial cycle from the ordered ready-queue, one observation per cycle, in order (already true in stage mode for the *selected* obs — make it the *next* obs). *Files:* `_cycle_loop` (stage-pipeline branch, ~`runtime.py:2187`). *Acceptance:* frame_seq committed is monotonic (modulo urgent pre-emption).
- **3.2** Ensure the consumer reads the **maintained scene model** (updated by the producer) rather than re-deriving perception. *Acceptance:* cognition input = current `SceneWorkspace` + the one fresh observation; no per-cycle re-decode.

### Phase 4 — Backpressure & coalescing policy (3–4 d) [2,3]
- **4.1** Implement the chosen policy from 0.1: (B) coalescing under overload, or (A) input throttle to cycle rate (config flag `DECADIC_INGEST_POLICY`). *Acceptance:* at 3× input rate, policy (B) keeps `frames_folded == frames_received` with bounded `commit_lag_ms`; policy (A) throttles input and never drops.
- **4.2** Coalescing rule for skipped frames (e.g. evidence-weighted merge into the next deep-processed observation, or scene-only fold). *Acceptance:* documented, deterministic, unit-tested.

### Phase 5 — Simplify the misaligned stage scaffold (2–4 d) [3]
- **5.1** Collapse the 10 placeholder stage-queues. The intent needs **one** producer stage (decode + fold) + the ordered ready-queue, not a systolic cognition pipeline. Either (a) retire stages 2–10 of `stage_pipeline.py`, keeping session/snapshot/telemetry, or (b) keep `stage_pipeline` purely as the producer/scheduler with a single real stage. *Acceptance:* no placeholder `"defer_to_commit_cycle"` stages remain; LOC reduced; behavior unchanged or improved.
- **5.2** (Optional, only if you later want *concurrent cognition*) document that running real neural stages concurrently is a separate project with weight-staleness implications — out of scope for this intent. *Acceptance:* decision recorded.

### Phase 6 — Telemetry, tests, acceptance (3–4 d) [1–5]
- **6.1** New metrics: `frames_received`, `frames_folded`, `frames_deep_processed`, `coalesced_sessions`, `producer_overlap_ratio`, `decode_on_consume_ms`, `information_loss`. *Acceptance:* surfaced in `metrics()` and the dashboard.
- **6.2** Tests: `test_prefetch_overlap` (decode overlaps cognition), `test_no_information_loss` (`frames_folded == frames_received`), `test_order_preserved`, `test_overload_coalesces` (no information loss at 3× rate), `test_serial_consumer_single_obs`. *Acceptance:* all green; parity test for the legacy `batching` mode preserved.

**Indicative total:** ~20–30 developer-days, sequenced 0 → 1 → 2 → 3 → 4 → 5 → 6. Phases 1 and 2 are the critical path and deliver the bulk of the user-visible fix (real prefetch + no loss).

---

## 6. Acceptance criteria (how you'll know it's fixed)

| Req | Pass condition |
|-----|----------------|
| R1 | Exactly one observation in `run_neural_cycle` at a time; `frames_deep_processed` increments once per cycle. |
| R2 | Producer never blocks on cognition; `frames_received` keeps climbing while the consumer is busy. |
| R3 | `information_loss == 0`; `frames_folded == frames_received` across a long run. |
| R4 | `decode_on_consume_ms ≈ 0` (next frame pre-decoded); `producer_overlap_ratio > 0`. |
| R5 | Committed `frame_seq` is monotonic except for explicit urgent pre-emption, and pre-empted frames are still folded. |
| R6 | Consumer idle-on-decode time ≈ 0; `stale_sessions` losing information == 0. |

---

## 7. Risks & honest tradeoffs

- **The rate conflict is fundamental, not a bug.** No design can both deep-process every frame *and* never fall behind when input > processing rate. Policy (B) resolves it the way the brain does — lossless pre-attentive model, attentional deep-processing — but it means *not every frame gets a full cognitive pass*. If you literally require every frame deep-processed, you must throttle input (policy A).
- **Single-GPU reality.** Real producer/consumer overlap helps because the frozen encoders release the GIL during native compute; but on one GPU the encode and the cognitive forward still contend for the same device. Expect latency-hiding for the CPU/decode portion, modest gains for GPU-bound encode. Big wins need separate devices.
- **Thread-safety is the main new hazard.** Moving encode off `self.lock` (1.2) introduces concurrency between producer and consumer over the scene workspace; this must be carefully scoped (lock only the mutation) and stress-tested.
- **Scope discipline.** Resist rebuilding the 10-stage pipeline as concurrent *cognition* (Phase 5.2). That's a different, much larger project (it reopens the recurrent-state / weight-staleness issues) and is not what the stated intent requires.
