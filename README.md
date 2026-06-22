# Self-Determination Model — Decadic Cycle server

Phase **2** cognitive architecture server: FastAPI + WebSocket, State Bus (A–F), **PyTorch** Decadic pipeline (multimodal fusion transformer, risk MLP, narrative encoder–decoder stack, GRU/LSTM, policy head), **predictive coding** losses, frozen **CLIP + Whisper** encoders when `DECADIC_ENCODER_MODE=hf`, a need-gated **curiosity** drive (`decadic/state/curiosity.py`), **dual-network memory consolidation** (`decadic/consolidation/`), a pre-cognitive **perception organ** with anonymous object files and LTM write gates (`decadic/perception/`), reusable **Skill Dojo / Perception Dojo** curricula (`decadic/training/`), and structured logging (`decadic/logging/`).

## Recent implementation snapshot

The current branch includes several major upgrades beyond the original Phase 2 server:

- **Fly/human-inspired perception organ** - live camera frames now produce anonymous object files through retinotopic contrast, local motion, flow, looming, stuff/background, and body-candidate cues before Working Memory or LTM.
- **Retinotopic bootstrap repair** - if CLIP patch tokens or SlotAttention are not yet producing usable proposals, the perception organ falls back to label-free image-region proposals. This fixes the "camera sees objects but graph/LTM shows 0 objects" starvation path without weakening LTM gates.
- **Persistent 3D Scene Workspace** - object files feed a persistent egocentric rendered space with visible/occluded entities, spatial relations, focus selection, object permanence, and prediction health. Working Memory is now the small focus cache, not the whole scene.
- **Persistent Parallel Perceptual Processing** - incoming frames enter a default-on, 10-session perceptual pipeline. Perception can process frames concurrently, but commits them into the persistent scene workspace in timestamp order; Decadic cognition remains one serialized decision loop. The old recent-frame batching path is still available as **Batching Perceptual Observations**.
- **Optional/default-on scene dynamics** - a trainable perception-side head predicts next anonymous entity state from prior scene state plus efference copy, with constant-velocity fallback when disabled.
- **Anonymous property-belief LTM** - repeated perceptual evidence strengthens beliefs on one anonymous entity instead of duplicating nodes for the same property. Beliefs track evidence count, confidence, variance/instability, and remain label-free.
- **Stricter memory write gates** - Stage 10 accepts only stable, confident, non-collapsed object files; bad perception skips permanent writes with explicit reasons.
- **Skill Dojo and Perception Dojo** - reusable skill specs, adaptive teacher assist, attempt/retry lifecycle, perception-first graduation gates, and a packaged Stand and Recover curriculum.
- **Dashboard/API diagnostics** - Discovery, Perception, Self-Indexed Graph, LTM, Skill Dojo, and camera panels expose object-file health, scene health, LTM write reasons, property beliefs, flow/looming/stuff/body counts, and live viewer controls.
- **Runtime/process tooling** - Windows launch/restart scripts, managed environment subprocess control, spectator camera views, hand-feeding/admin resource delivery, NPC village/caregiver scaffolding, saved agents, and Vast.ai deployment support.

## The cognitive architecture

The server runs the **Decadic Cycle of Expression** — a ten-stage model of how cognition
unfolds from perception to behavior — as a continuous, asynchronous process. Each stage is a
small trainable PyTorch module; there is no pretrained LLM in the cognitive core. See
[`decadic_project_brief.md`](decadic_project_brief.md) for the full theory.

**The ten stages** (`decadic/cycle/stages/`, orchestrated by `decadic/cycle/neural_pipeline.py`):

1. Sensory perception
2. Experience framing & multisensory fusion (transformer)
3. Heuristic assessment & memory correlation
4. Risk-utility evaluation + curiosity arbitration (MLP)
5. Pre-normative conclusion (encoder–decoder)
6. Emotional / physiological update (GRU)
7. Reprioritization & state-of-mind update (LSTM)
8. Strategy formation (policy head)
9. Behavioral response (action emitted to the body)
10. Normative memory mapping (feeds the next cycle)

**The persistent State Bus (A–F)** (`decadic/state/state_bus.py`) carries continuous state
across cycles:

| Element | Meaning |
|---------|---------|
| **A** | State of mind |
| **B** | Emotional / physiological state (pain / pleasure / curiosity affect) |
| **C** | Internal narrative |
| **D** | Current priority (`explore` / `investigate` / `avoid` / …) |
| **E** | Metacognition |
| **F** | Action history / efference copy |

Learning is **online and self-supervised**: predictive-coding losses across the stages plus
forward-model errors (proprioceptive, interoceptive, tactile) drive an Adam step every cycle.
There is no external reward and no labeled data — the only innate signals are the homeostatic
drive and curiosity (below). Set `DECADIC_USE_NEURAL=0` to swap the trainable stack for the
fast numpy stub pipeline (`decadic/cycle/pipeline.py`) used by the test suite.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

PyTorch is a core dependency (CPU wheel by default; GPU if your platform provides it).

## Run

```powershell
python -m uvicorn decadic.api.app:app --host 0.0.0.0 --port 8765
```

## Synthetic streaming client (plan / brief harness)

With the server running:

```bash
python scripts/synthetic_ws_client.py --host 127.0.0.1 --port 8765 --steps 20
```

## MuJoCo humanoid body (persistent embodiment)

A physical body with hands, feet, joint proprioception, and palm/sole touch sensing
([assets/humanoid_body.xml](assets/humanoid_body.xml)). Decadic `move` actions steer it
via root-assist (PD standing hold + pelvis drive); full-body senses stream back as
`proprioception.joints` / `proprioception.contacts`, impact/fall `events` hit the viability
fast path, and a `world_state.body` blob feeds the egocentric graph.

```bash
pip install -e ".[body]"

# Contract check (no MuJoCo needed; vision is auto-skipped in --dry-run)
python scripts/mujoco_decadic_adapter.py --dry-run --steps 30 --port 8765

# Persistent embodied run with soundscape. Vision is ON by default (the encoder
# mode is 'hf'); pass --no-vision for a fast, headless proprioception-only body.
python scripts/mujoco_decadic_adapter.py --steps 0 --audio --port 8765

# Scenarios: a chasing bear (threat → pain/avoid) or scattered food (eat → pleasure)
python scripts/mujoco_decadic_adapter.py --steps 0 --audio --view --scene bear --port 8765
python scripts/mujoco_decadic_adapter.py --steps 0 --audio --view --scene food --port 8765
```

`--audio` synthesizes a procedural soundscape from physics — footstep thuds scaled by
sole force, collision impacts, fall thumps, bear growls (tremolo, louder as it closes
in), and food chimes — shipped as 0.8 s pcm16 windows in `observation.audio`.

| Variable | Purpose |
|----------|---------|
| `DECADIC_BODY_OBS_INTERVAL_MS` | Adapter observation throttle (default `80`) |
| `DECADIC_PROPRIO_JOINT_CAP` | Joint qpos/qvel values consumed by the proprio encoder (default `64`) |
| `DECADIC_PROPRIO_CONTACT_CAP` | Touch/contact values consumed by the proprio encoder (default `16`) |

### Real vision + audio in the forward pass

By default `DECADIC_ENCODER_MODE=hf`: the egocentric camera frame is encoded by frozen
CLIP (512-d) and the audio window by the frozen Whisper encoder (768-d) every observation,
so the brain actually sees and hears (first run downloads ~1 GB of frozen weights).
Embeddings are cached per observation timestamp so repeated cycles on the same observation
cost nothing extra. For a fast, no-download run — vision/audio embeddings become zero
tensors and only proprioception reaches the network — set the cheap fallback:

```bash
$env:DECADIC_ENCODER_MODE="zeros"   # PowerShell; no download, faster cycles, brain is blind/deaf
.\.venv\Scripts\python.exe -m uvicorn decadic.api.app:app --host 127.0.0.1 --port 8765
```

Expect the PC loss to spike and re-converge when flipping modes (input statistics
change); checkpoints stay loadable since embedding dims are fixed.

| Variable | Purpose |
|----------|---------|
| `DECADIC_ENCODER_MODE` | `hf` (default; frozen CLIP + Whisper) or `zeros` (no download, brain blind/deaf) |
| `DECADIC_CLIP_MODEL` | Vision encoder (default `openai/clip-vit-base-patch32`) |
| `DECADIC_WHISPER_MODEL` | Audio encoder (default `openai/whisper-small`; `openai/whisper-tiny` is faster) |

Measured on an RTX 3080: `whisper-small` ≈ 80 ms/cycle (~7.5 cycles/s, ~1.6 GB VRAM);
`whisper-tiny` ≈ 30–60 ms/cycle (~9.3 cycles/s, ~0.45 GB VRAM). Non-default encoder
widths are padded/truncated to the fixed 512/768-d slots, so swaps are checkpoint-safe.

Note: widening the proprio encoder changed its input shape — brain checkpoints
(`agent_*_brain.pt`) saved before this change can no longer be loaded.

### Perception organ and anonymous object files

Discovered perception now runs through a pre-cognitive **perception organ**
(`decadic/perception/`) before Working Memory and LTM. The purpose is to give the agent
a fly/human-inspired sensory substrate without changing the Decadic Cycle itself:

`egocentric camera -> CLIP patch tokens + retinotopic maps + bootstrap regions -> motion/contrast/body cues -> anonymous object files -> Scene Workspace -> Working Memory focus -> gated LTM`

The live discovered-perception path now inserts a persistent **Scene Workspace**
between object files and cognition:

`anonymous object files -> Scene Workspace -> Working Memory focus cache -> Global Workspace -> Decadic Cycle -> episodic + LTM`

The Scene Workspace keeps an egocentric, label-free scene model across frames:
visible entities, temporarily occluded entities, stuff/background regions,
body-part candidates, spatial relations, salience, focus ids, and a lightweight
constant-velocity scene prediction error. Working Memory is therefore the small
attention/focus cache, not the whole world model. The Global Workspace receives
focused coalitions derived from the scene instead of the entire perceptual field.

The perception organ adds:

- **Retinotopic feature maps** that preserve image-space location instead of collapsing
  immediately into one global embedding.
- **Retinotopic bootstrap proposals** from contrast, brightness discontinuity, and
  frame-difference motion when learned SlotAttention proposals are absent or immature.
  This prevents a real camera frame from being treated as "no objects" simply because
  the slot learner has not yet stabilized.
- **Contrast, edge, and local motion channels** from the live camera frame.
- **Frame-difference flow diagnostics** that separate global camera motion from local
  independently moving regions.
- **Looming estimates** for expansion / near-collision perception.
- **Foreground/stuff/body-candidate hints** so floor/walls/large uniform regions do not
  poison object memory, and visually self-moving regions can become body-part candidates.
- **Stable anonymous object files** with `object_id`, centroid, appearance, motion, flow,
  contrast, looming, persistence, agency, confidence, and `kind_hint`.

The bootstrap path is deliberately sensory-only. It does not infer names, resource kinds,
simulator classes, or task labels; it only creates anonymous region candidates so the
same object-file, scene-workspace, Working Memory, and LTM health gates can decide whether
the percept is trustworthy. If the frame is blank, uniform, stale, or low confidence, the
health gate still skips memory writes.

Runtime cognition remains label-free. Live object files may say `object`, `stuff`, or
`body_part_candidate`, but they must not contain semantic labels such as food, water,
hand, wall, or building. Offline bootstrap/evaluation may use MuJoCo truth, depth, flow,
or segmentation-teacher masks to train the perception organ, but those labels are stripped
before the Decadic stages, Working Memory, LTM, replay records, and dashboard object-file
payloads. See `docs/perception_organ_contract.md` for the runtime contract.

Perception health is computed every discovered cycle:

- `healthy`
- `low_confidence`
- `collapsed`
- `no_objects`
- `teacher_only`
- `stale_frame`

Health metrics include centroid spread, appearance diversity, mask entropy/diversity,
active proposal count, stable tracked object count, flow confidence, looming count,
stuff count, body-candidate count, bootstrap proposal count, and the latest LTM write
result. The system treats bad perception as a reason to skip permanent memory writes
rather than storing corrupted object memories.

Scene diagnostics are exposed on `/agent/{id}/state` and `/agent/{id}/discovery`:
scene entity count, visible/occluded/stable counts, focus ids, relation count,
duplicate-identity count, Global Workspace ignition metadata, and scene prediction
error. The dashboard shows these on the Perception and Discovery panels so a failure
can be localized to camera/body, object files, scene tracking, focus selection, GWT,
or LTM consolidation.

Scene prediction has two layers. The Scene Workspace is the persistent rendered
space: it holds anonymous visible and occluded entities, egocentric position,
relations, focus candidates, and property-belief evidence. The optional/default-on
Scene Dynamics head is the predictive stabilizer: in discovered perception it learns
next-frame anonymous entity UV/relative position, motion, visibility, persistence,
and uncertainty from prior scene state plus efference copy. The same anonymous matched
scene features are stored in replay and trained during consolidation when present.
Disable it with `DECADIC_SCENE_DYNAMICS_ENABLED=0` to fall back to constant-velocity diagnostics.
The learned head is perception-only; no labels, rewards, or simulator classes enter
the Decadic stages.

## Watch it live (body viewer + dashboard)

**One click (Windows):** double-click the desktop shortcut **Decadic** (or run
`scripts\Launch Decadic.cmd` / `scripts\launch_decadic.ps1`) — it starts the server and
the web UI in their own windows and opens the dashboard in your browser. To recreate the
shortcut after moving the repo:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\install_desktop_shortcut.ps1"
```

(Re-running it is safe: it reuses a server/UI already listening instead of spawning
duplicates, and installs the dashboard's npm deps on first run. The body viewer is a
separate, optional process — start it with the command below when you want a body.)

### Starting, stopping, and restarting the local processes

Preferred start path:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\launch_decadic.ps1"
```

That starts the backend on `127.0.0.1:8765` and the web UI on
`127.0.0.1:5173`. If the ports are already occupied, the launcher reuses those
processes.

To stop only the Decadic backend and web UI, kill the processes listening on
those two ports:

```powershell
$listeners = netstat -ano |
  Select-String -Pattern 'LISTENING' |
  Select-String -Pattern ':8765|:5173'

$pids = foreach ($line in $listeners) {
  $parts = ($line.ToString() -split '\s+') | Where-Object { $_ }
  if ($parts.Length -ge 5) { [int]$parts[-1] }
}

$pids | Select-Object -Unique | ForEach-Object {
  Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}
```

Clean restart sequence: stop listeners on `8765` and `5173`, wait a second or
two, then run `scripts\launch_decadic.ps1` again.

If the launcher cannot find `uvicorn`, create/update the local environment once:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv run --extra body python -m uvicorn --version
```

After that, manual backend starts can use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn decadic.api.app:app --host 127.0.0.1 --port 8765
```

Manual web UI starts:

```powershell
cd dashboard
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

If a background start is needed, send output to the repo logs:
`logs\decadic_server_stderr.log`, `logs\decadic_server_stdout.log`,
`logs\decadic_ui_stdout.log`, and `logs\decadic_ui_stderr.log`.

Or start the three processes by hand:

```powershell
# 1. Server
.\.venv\Scripts\python.exe -m uvicorn decadic.api.app:app --host 127.0.0.1 --port 8765

# 2. Body — native 3D viewer window + egocentric vision + soundscape
.\.venv\Scripts\python.exe scripts\mujoco_decadic_adapter.py --steps 0 --vision --audio --view --port 8765

# 3. Mind — React dashboard (first run: npm install)
cd dashboard
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

The **viewer window** shows the humanoid being driven by the brain's `move` actions.
The **dashboard** polls `GET /agents`, `/agent/{id}/state`, `/agent/{id}/metrics`, and
`/agent/{id}/vision`, showing: viability gauge + pain/pleasure, priority (D) badge,
PC-loss and viability time series, A/B/C/E heat strips, F action stream, the egocentric
camera frame, joint/touch proprioception, egocentric graph nodes, and recent events
(collisions/falls hit the viability fast path in real time). Set `VITE_DECADIC_HTTP`
in `dashboard/.env` if the server is not on `127.0.0.1:8765`.

The dashboard is organized into tabs: **Overview**, **Environment**, **Training**,
**Self-Indexed Graph**, **Discovery**, **Cognition / Why**, **Motor / Active Inference**,
**Brain Map**, **Loss Landscape**, **Events + State Bus**, **Agent Settings**, **Deploy / GPU**,
and **Saved Agents** — most are described in the sections below.

The **Discovery**, **Self-Indexed Graph**, **Long-Term Memory**, and **Skill Dojo** panels
surface perception health directly: object-file count, stable tracked count, centroid
spread, appearance diversity, flow confidence, looming count, stuff count, body-candidate
count, bootstrap proposal count, scene entity/focus counts, property-belief totals, and
the latest LTM accepted/skipped reason. If perception collapses or starves before memory,
the dashboard shows that as a perception/memory failure instead of making it look like a
motor or cognition failure.

Each neural net has **Start / Stop / Reset** controls in the dashboard topbar, backed by
`POST /agent/{id}/resume`, `POST /agent/{id}/pause`, and `POST /agent/{id}/reset`:

- **Stop** freezes the cognitive cycle loop — state, weights, and memory are retained,
  and incoming observations can still update the perception pipeline / scene workspace.
- **Start** resumes cycling from exactly where it left off.
- **Reset** gives the agent a fresh mind: re-initialized neural weights, zeroed State
  Bus / viability / perception, and wiped episodic memory (confirmation required).

### Brain Map & network size

The **Brain Map** panel renders the agent's actual neural network in 3D: each block of
the cognitive stack is a cluster of neurons arranged around the Decadic ring, the lines
are its strongest learned weights (orange excitatory / blue inhibitory, from
`GET /agent/{id}/brain/topology`), and clusters light up with the real per-stage
activations of the last cycle. Drag to orbit, scroll to zoom, hover a cluster for unit
and parameter counts.

The topbar **preset selector** switches the network size per agent via
`POST /agent/{id}/preset?preset=...` (also settable at boot with
`DECADIC_NEURAL_PRESET`):

| Preset | Neurons | Connections | Notes |
|--------|---------|-------------|-------|
| `tiny` (default) | ~3.9k | ~0.76M | fastest cycles; practical floor (~1M) |
| `2_5m` | ~8k | ~2.4M | |
| `5m` | ~12k | ~4.8M | |
| `medium` | ~17k | ~8.4M | balanced |
| `10m` | ~18k | ~10M | |
| `full` | ~34k | ~25M | |
| `xl` | ~49k | ~51M | |
| `xxl` | ~59k | ~75M | |
| `ultra` | ~69k | ~100M | watch cycle wall ms |
| `250m` | ~108k | ~249M | heavy / define-only |
| `500m` | ~153k | ~494M | heavy / define-only |
| `1b` | ~215k | ~976M | heavy / define-only |

Counts are the baseline (cognitive faculties off); production agents (perception
feedback + discovered perception on) are somewhat larger. Switching rebuilds the brain
from scratch (architectures cannot share weights), like a reset. Saved brain checkpoints
are tagged with their preset and refuse to load into a mismatched architecture.

**Heavy tiers / memory cliff.** `250m`, `500m`, and `1b` are *define-only*: they build
and run a forward pass, but the stack is trained **every cognitive cycle** with fp32 Adam
(~16 bytes/param for weights + grads + optimizer moments, before activations) — roughly
4 GB at 250M, 8 GB at 500M, and ~16 GB at 1B. Continuous training of `500m`/`1b` will
slow cycles sharply or OOM on a single consumer GPU (24–32 GB), especially alongside
CLIP/Whisper and the MuJoCo body. Running them well would need a mixed-precision (bf16) +
8-bit/fused-Adam (and likely sharded) training path that is **not yet implemented**;
`NeuralBundle.try_build` logs a warning when a heavy tier is selected.

### Loss Landscape (live)

The **Loss Landscape** panel renders a live 2D slice of the agent's *actual* weight
space as a 3D loss surface (`GET /agent/{id}/brain/landscape`). A background probe
(`decadic/consolidation/landscape.py`) clones the live stack, perturbs it along two
**filter-normalized** random directions (Li et al., 2018, *Visualizing the Loss
Landscape of Neural Nets* — each weight filter's direction is rescaled to that
filter's norm, so the surface is not an artifact of weight scale), and evaluates the
agent's real predictive-coding + forward-model objective on a replay batch of its own
experience at every grid point. Blue valleys are low loss, red ridges high; the white
sphere is the current weights `θ*`. The directions are seeded/persisted so the bowl is
comparable frame-to-frame — *watch it reshape as the agent learns*.

The probe is **OFF by default** (`DECADIC_LANDSCAPE_ENABLED=1` to enable): it is purely
a visualization, costs `grid × grid × batch` no-grad forward passes per refresh, reuses
the consolidation replay buffer (so consolidation must be on to feed it), runs on a worker
thread off the cognitive critical path, and **never touches the live weights** — enabling
it is byte-identical to baseline cognition. Each refresh logs a `landscape_compute` event
(cycle / grid / batch / center / min / max / wall ms) to `logs/decadic_server.jsonl`.
The slice is a projection of a huge weight space, so the geometry is qualitative.

### Cameras & body recenter

When the adapter runs with `--vision`, it also renders spectator cameras (`track`,
`front`, `side`, `top` — all tracking the body's center of mass) and ships them as a
`debug_views` field next to the egocentric frame. The server strips `debug_views`
before cognition — only the egocentric eye is ever encoded into perception — and
serves them via `GET /agent/{id}/vision?camera=<name>`. The Perception panel has a
camera dropdown to switch between the brain's eye and the spectator angles. If the
selected camera disappears after an agent/body swap, the UI falls back to an available
view. The **Live window** button asks the managed environment process to open the native
MuJoCo viewer on the machine running the body; if no managed body is running, start one
from the Environment tab or run the adapter with `--view`.

If the body wanders off, **Recenter body** (`POST /agent/{id}/body/recenter`) sends a
`body_command` over the WebSocket; the adapter teleports the root back to the stage
origin in its default standing pose (props stay put). The adapter also auto-recenters
whenever the body crosses an 18 m radius fence, so it can no longer walk off the
20 m × 20 m floor into the void.

## Curiosity & memory consolidation (autonomous learning)

Two intrinsic-learning subsystems, both **ON by default** in production (the test suite
pins them off for a deterministic baseline; set the env flag to `0` to disable):

- **Need-gated curiosity** (`decadic/state/curiosity.py`, `DECADIC_CURIOSITY_ENABLED=1`).
  An autonomous epistemic drive that rewards the **reduction** of forward-model
  prediction error — *learning progress*, not raw surprise, so it sidesteps the
  noisy-TV trap. It folds into element B as a pleasure-side affect, can flip the
  priority label to `investigate`, and relaxes motor babble as it is satisfied. It is
  **need-gated**: a threatened or deprived agent suppresses it; a safe, sated one
  expresses it. Telemetry: `curiosity_drive`, `curiosity_pleasure`,
  `curiosity_learning_progress` on `GET /agent/{id}/metrics`.

- **Dual-network consolidation** (`decadic/consolidation/`, `DECADIC_CONSOLIDATION_ENABLED=1`).
  A second, cloned cognitive stack replays salience-prioritized transitions from a
  bounded `ReplayBuffer` (lowest-salience evicted → built-in forgetting) on its own
  optimizer, then periodically **Polyak soft-syncs** its weights back into the live
  stack under the agent lock. Replay runs in a thread executor so the cognitive cycle
  is never blocked. Telemetry: `replay_count`, `replay_buffer_size`,
  `consolidator_loss`, `last_sync_cycle`.

Beyond per-cycle telemetry, both subsystems write a low-volume event timeline to the
structured server log (`logs/decadic_server.jsonl`): consolidation emits
`consolidation_start` (once) and `consolidation_sync` (per soft-sync, with cycle / replay
steps / loss); curiosity emits edge-triggered `curiosity_investigate_enter` /
`curiosity_investigate_exit` only when it takes or releases the `investigate` priority
(never per cycle). Every log line now carries an ISO-8601 UTC `time` field, so the log
answers both *what* happened and *when*.

## Embodiment: stances, manual joint braces & Skill Dojo

New embodied agents now spawn as free bodies by default. The joint-brace orthosis remains
available as a **manual scaffold** for debugging and body setup, but it is no longer part of
Skill Dojo training. Teacher targets are the dojo scaffold; braces are operator controlled.
When enabled manually, every hinge uses a stiff MuJoCo-native spring toward the active stance
reference and can ratchet ROM open as prediction error falls. The torso external wrench remains
zero, so movement has to come from real body contacts rather than an invisible support force.

- **Master toggle** - `POST /agent/{id}/body/braces?enabled=bool` runs the body free
  or re-engages the braces without losing earned ROM.
- **Reset ROM** - `POST /agent/{id}/body/reset_braces` re-welds every joint to start over.
- **Stances** - `GET /body/stances` lists the posture library and
  `POST /agent/{id}/body/stance?name=...` re-poses without changing brace state: `stand`,
  `all_fours`, `kneel_left`, `kneel_right`, `kneel_upright` (static) and `crawl`,
  `sit_to_stand`, `kneel_to_stand` (timed motions). Single source of truth:
  `decadic/embodiment/stances.py`.
- **Hold movement** - `POST /agent/{id}/body/movement_hold?enabled=bool` pins braces fully
  welded and loops the active motion stance indefinitely. It is a manual scaffold control,
  not a Skill Dojo command.

The **Skill Dojo** tab exposes the manual Body Scaffold controls; the **Motor / Active
Inference** and **Locomotion** panels show read-only brace, touch, gait, and forward-model
telemetry.

### Skill Dojo (episode-based skill practice)

The **Skill Dojo** tab runs named, reusable skill curricula around the live Decadic loop
(`decadic/training/`, `POST /dojo/{start,pause,resume,stop,phase}`). It is a supervisor,
not a replacement policy: it reads eval-only metrics, configures training knobs, queues
safe stance/world commands, records demo metadata, and gates phase promotion. The live cognitive
cycle remains self-supervised; teacher targets enter replay/consolidation metadata only,
and final evaluation phases run with autonomous teacher assist at `0`.

Teacher assistance is adaptive. A phase's `teacher_weight` is only the initial/default
compatibility value; during a run the supervisor computes live `teacher_assist` from posture,
fall, stall, and prediction-error metrics. Assist rises when the student is losing control,
fades after stable dwell, and is recorded into replay as the current `demo_weight`.

Each skill is a sequence of phases. Each phase now contains explicit **attempts**:

- A success gate uses all criteria with AND semantics plus `min_samples` and `min_dwell_s`.
- `failure_criteria` use OR semantics as fail-fast conditions, for example root height too
  low, torso tilt too high, or fall rate too high.
- `timeout_s` closes an attempt that is not making progress.
- `reset_commands` restore only the phase start state (`set_stance:*`, `recenter`, and safe
  world commands); they do not wipe the agent's weights, memory, or replay buffer.
- `auto_retry` and `max_attempts` control whether the phase restarts after a failed or
  timed-out attempt. Exhausted retries end the dojo run as `failed`.
- Manual braces or movement hold block phase graduation and are reported as
  `manual_scaffold_active`; they do not trigger retries by themselves.
- Embodied built-in skills keep `viability_mode=metabolic` during training.
  The Skill Dojo caregiver scaffold monitors hydration, energy, and integrity;
  below threshold it requests the visible parent NPC to deliver food/water/care
  through normal world objects instead of pinning reservoirs full.
- The dashboard shows the live teacher-assist meter, assist reason, and whether current
  samples are `self`, `dagger`, or `demo`.

Built-in skills include:

- `stand_and_recover` - adaptive teacher-guided standing, small perturbation recovery,
  reduced assistance, and autonomous balance evaluation.
- `perception_object_files` - perception-first dojo for static separation, enter/exit and
  reappearance, motion/parallax, looming/stuff rejection, body-candidate correlation, and
  autonomous LTM growth. Object-dependent skills should not be considered valid if this
  perception curriculum is failing.
- `developmental_locomotion` and `affective_locomotion` - the legacy walking curriculum
  migrated into Skill Dojo phases.

Uploadable skills live as JSON files and can be added from the Skill Dojo tab or
`POST /dojo/skills/upload`; uploaded specs are listed with built-ins and can be deleted
without touching built-in skills. The packaged example
`docs/dojo_skills/stand_up_from_floor_balance.json` trains from upright kneeling through
`kneel_to_stand`, then requires autonomous standing balance. See
`docs/skill_dojo_methodology.md` for the full SOP, JSON schema, caregiver scaffold,
and WBS.

## Motivation & long-horizon learning

The agent has exactly two innate motivational substrates; everything else is learned by
association from its own experience.

- **Homeostatic drive.** Viability is derived from three reservoirs — hydration, energy,
  integrity (`decadic/state/viability.py`). A depleted reservoir is felt as innate, convex
  deprivation **pain**; its positive complement is a phasic **relief** reward — pleasure
  proportional to the per-cycle *reduction* in that drive (homeostatic RL à la Keramati & Gutkin;
  `drive_reduction_reward`, `DECADIC_DRIVE_REWARD_ENABLED=1`) — so moving back toward the setpoint
  feels good, with no external or labeled satisfier. A learned interoceptive world model lets the
  policy act to reduce its *predicted* internal drive toward the full setpoint. The satisfier
  (food/water) is never labeled — it is discovered from experienced transitions. Damage events
  (collisions, falls) are event-driven and hit element B on the **fast path**. The cycle's other
  phasic affect is the agent's genuine predictive-coding surprise; the legacy cycle-counter PE
  oscillation is removed by default (`DECADIC_PE_STUB_WEIGHT=0`). `DECADIC_VIABILITY_MODE=immortal`
  remains an admin/debug mode, but embodied Skill Dojo runs use metabolic mode plus visible
  caregiver delivery so survival pressure remains part of training.
- **Need-gated curiosity.** See *Curiosity & memory consolidation* above — rewards learning
  progress (PE reduction), gated by survival urgency.

Closing the gap between "act until a drive is accidentally relieved" and "see a resource, value
it, walk to it, and learn the path that worked" is a set of **distal credit-assignment**
mechanisms. True to the experiment, each starts **naive** (zero/identity init, influence ramped
from 0) so a fresh agent is byte-identical to the one-step baseline until its own experience
grows them:

- **Goal lifecycle** — an explicit `GoalState` latches the dominant deficit as the active goal
  at onset, holds it while pursued, and closes it on achievement or abandonment. The closed
  `[onset → close]` window is the episode the return-based learner trains on.
- **Successor-features (SF) value** — the consolidator computes λ-returns over ordered
  goal-episodes and trains an SF head `ψ(state, action)` predicting the discounted sum of future
  controllable-intero features. A scalar value `v = ψ · w` composes the learned (reward-free)
  prediction with the **innate** drive weights, so a seen resource inherits value from the
  future relief it predicts — "the object becomes a goal." Policy-shaping influence ramps in
  over `DECADIC_SF_VALUE_RAMP_CYCLES`.
- **Hindsight relabeling (HER)** — a goal episode that ends without achievement is relabeled
  with the terminal feature it *did* reach and re-pushed, turning near-misses into positive
  signal ("the journey still taught me").
- **Imagined replay (Dreamer-lite, OFF by default)** — optionally rolls out short imagined
  trajectories from the agent's own forward models during consolidation and trains the value
  targets on them too. Bounded + trust-weighted to limit hallucinated value.
- **Per-life resource randomization (anti-camping)** — food/water are re-scattered each life so
  the *location* of relief is never memorizable; only the *skill* of seeking-and-reaching
  transfers (`arena` or `zone` placement).

## Memory: episodic store + long-term knowledge graph

Two complementary stores (a Complementary-Learning-Systems framing; see `decadic/memory/ai.md`):

- **Episodic store** (`decadic/memory/episodic_store.py`) — a SQLite-backed per-cycle *diary*
  of cycle summaries + fixed-size embeddings, with vector-addressed similarity recall. Query it
  via `GET /agent/{id}/memory` and `GET /agent/{id}/memory/similar`.
- **Long-term knowledge graph** (`decadic/memory/semantic_graph.py`) — the persistent,
  **unbounded** relational "hippocampal index". One node per consolidated object (keyed by its
  learned appearance embedding), with edges accumulating from co-presence. Working memory stays
  bounded (the "now"); stage 10 consolidates stable slots into the graph and stage 3
  re-identifies re-seen objects against it, reusing stable `ent-NNNNN` ids. Watch it grow on the
  **Self-Indexed Graph** tab. ON by default (`DECADIC_LTM_GRAPH=0` proves the no-LTM path is
  byte-identical).

Both persist per agent under `DECADIC_DATA_DIR` and are bundled when an agent is saved.

The LTM now stores **anonymous property beliefs** on consolidated entities. When the agent
sees the same anonymous object again, numeric perceptual evidence such as area, compactness,
roundness, edge strength, brightness contrast, depth, bearing, local motion, looming, and
experienced consequence predictors strengthens the existing belief instead of creating a new
node. Each belief tracks running mean, evidence count, confidence, variance, first/last cycle,
source, and instability. Semantic keys and simulator labels are stripped; allowed consequence
beliefs are still anonymous/survival-indexed, for example `predicts_pain`,
`predicts_integrity_loss`, `predicts_energy_relief`, or localized body-pain predictors.
The LTM dashboard shows total beliefs, per-node belief chips, confidence, evidence counts,
and unstable belief counts.

Stage 10 now gates LTM writes on perception health. It accepts stable, confident,
non-collapsed object files and records a write reason such as:

- `accepted`
- `skipped_no_objects`
- `skipped_perception_collapsed`
- `skipped_low_confidence`
- `skipped_prediction_unstable`

The graph also avoids merging simultaneous distinct object files into the same LTM node
unless appearance plus spatial/temporal evidence says they are the same tracked entity.
This is intentionally conservative: a skipped write is preferable to poisoning permanent
memory with collapsed perception. Fresh objects normally appear in the self-indexed graph
first; LTM follows only after the configured minimum stable sightings.

## Discovery API and perception diagnostics

`GET /agent/{id}/discovery` returns the live discovered-perception payload:

- `object_files`
- `discovery_health`
- `ltm_consolidation`
- `perception_organ`
- `retinotopic_map`
- `scene_workspace`
- `scene_prediction`
- `discovery`

`GET /agent/{id}/state` also includes the current perception organ diagnostics and
object-file snapshots. These payloads are diagnostic only; they expose what the sensory
system produced without feeding semantic labels back into cognition.

The useful debugging read is:

- `perception_organ.frame_seen` — whether a decodable egocentric camera frame reached perception.
- `perception_organ.bootstrap_proposal_count` — anonymous image-region proposals created before learned slots are reliable.
- `discovery_health.object_files` — healthy foreground object files after stuff/low-confidence filtering.
- `discovery_health.reason` and `ltm_consolidation.status` — why LTM accepted or skipped the cycle.
- `scene_health.entity_count`, `visible_count`, `occluded_count`, and `focus_count` — whether the persistent scene space is tracking entities even when LTM has not written yet.

## Environments & scenarios

Beyond the standalone CLI body, the server can **manage the body subprocess** for you. The
**Environment** tab composes a scenario from elements (house, food, water, a chasing bear, the
NPC village) plus senses (vision/audio) and starts/stops the MuJoCo adapter:
`GET/POST /environment`, `POST /environment/{pause,resume,stop}`, `DELETE /environment`.

- **Scenes** — the standalone adapter also takes `--scene bear` (threat → pain/avoid) or
  `--scene food` (eat → pleasure).
- **NPC village** (`decadic/embodiment/`) — a small society of 8 scripted, collisionless,
  kinematically-animated NPCs confined to their own habitats, each running a per-zone behavior,
  with co-located respawning food/water and one **parent** that provisions the learner on a need
  threshold. Adding the crowd never changes the agent's 21 actuators or 42-value proprioception.
- **Hand-feeding** — `POST /agent/{id}/give` drops a resource near the body for testing.

## Saving & loading agents

Three persistence mechanisms, from ephemeral to durable:

- **Checkpoints** — `POST /agent/{id}/checkpoint` writes JSON state plus **`agent_{id}_brain.pt`**
  (stack + proprio encoder + optimizer) when neural mode is on; `POST /agent/{id}/restore`
  reloads both. Ephemeral (keyed by the volatile agent id, pruned on death).
- **Saved Agents library** (`decadic/api/saved_agents/`) — the **Saved Agents** tab saves an
  agent under a stable id (brain + episodic memory + long-term graph always bundled) that
  survives restarts and loads into a fresh agent: `POST /agent/{id}/save`, `GET /saved-agents`,
  `POST /saved-agents/{id}/load`, `DELETE /saved-agents/{id}` (`DECADIC_SAVED_DIR`).
- **Scenario presets** (`decadic/api/presets/`) — named scenario drafts (elements + senses +
  joint braces) behind the top-bar dropdown: `GET/POST/DELETE /agent-presets`. Built-ins:
  calm / forage / parent / village / predator / mind.

## Interpretability: cognition trace / why

Every cycle assembles a read-only, human-readable explanation of the agent's behavior from
tensors it already computes — survival-intent decomposition, self-model surprise, episodic
grounding, optional input attribution, counterfactual rollouts, and a templated/LM narrative
(`decadic/cycle/cognition_trace.py`, `decadic/interpretability/`). Nothing here feeds back into
cognition. Read it on the **Cognition / Why** tab or via `GET /agent/{id}/explain`. Knobs:
`DECADIC_COGNITION_TRACE` (default on), `DECADIC_COGNITION_ATTRIBUTION_INTERVAL`,
`DECADIC_NARRATIVE_MODE` (`off`/`template`/`lm`), and `DECADIC_PROBE_PATH`.

## Cloud GPU deployment (Vast.ai)

The **Deploy / GPU** tab makes renting a cloud GPU terminal-free (`decadic/api/vast/`,
`deploy/vast/`). Paste a Vast.ai API key once, search the live marketplace, click Rent, and the
local server provisions a box, ships the brain + a headless MuJoCo body, runs them on CUDA, opens
an ssh tunnel, and reverse-proxies agent traffic so the existing panels show the **remote** agent
learning live. Stop/Destroy from the UI. The `vastai` client is bundled as a dependency; you
still need an ssh client on the host and an SSH key registered with Vast.ai (see
`deploy/vast/README.md`).

## Tests

```bash
python -m pytest -q
```

Integration tests disable neural weights (`DECADIC_USE_NEURAL=0`) for speed. `tests/test_neural_cycle.py` exercises the torch path. The whole suite is pinned to `cpu` / `zeros` / `fp32` / synchronous episodic writes (`tests/conftest.py`), so it is byte-identical regardless of the GPU/precision/async knobs below.

For the current perception/memory stack, the focused regression sweep is:

```powershell
$env:TMP = "$PWD\.pytest_tmp"
$env:TEMP = $env:TMP
.\.venv\Scripts\python.exe -m pytest `
  tests/test_perception_organ.py `
  tests/test_object_files.py `
  tests/test_scene_workspace.py `
  tests/test_property_beliefs.py `
  tests/test_perception_discovery.py `
  tests/test_pipeline.py `
  tests/test_neural_cycle.py `
  tests/test_scene_dynamics.py `
  tests/test_working_memory.py `
  tests/test_ltm_write_behind.py `
  tests/test_world_graph.py `
  tests/test_global_workspace.py `
  tests/test_persistent_mental_image.py -q
```

That set covers retinotopic bootstrap proposals, object-file health gates, scene tracking,
anonymous property-belief LTM, the neural cycle, scene dynamics replay fields, write-behind
LTM, and Global Workspace integration. Setting `TMP`/`TEMP` inside the repo avoids Windows
temp-folder ACL issues on some machines.

## Performance / GPU

The `full` preset trains a ~25M-param stack **and** runs frozen CLIP + Whisper **every cycle**. On CPU that is ~1 Hz; on an NVIDIA GPU it is an order of magnitude faster. Cognition is device-aware end to end (the consolidation clone and loss-landscape probe inherit the GPU for free), so the only thing standing between you and a 10–20 Hz cycle is turning the GPU on and trimming a little CPU work.

1. **Install a CUDA build of PyTorch** (skip if `python -c "import torch; print(torch.cuda.is_available())"` already prints `True`). The default wheel is CPU-only; the GPU wheel bundles its own CUDA runtime (no CUDA Toolkit needed), e.g. for an Ampere card (RTX 30-series, driver ≥ 525):

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126   # fallback: cu124
```

2. **Pin the device.** `scripts/launch_decadic.ps1` sets `DECADIC_DEVICE=cuda` so an inherited shell env can't silently fall back to the slow CPU path (it still auto-falls-back to CPU if CUDA is absent). On startup the bundle logs `neural compute device=cuda gpu=… vram_free_gb=… bf16=…` so you can confirm the GPU and headroom.

3. **bf16 frozen encoders.** With `DECADIC_ENCODER_PRECISION=auto` (default) the CLIP/Whisper forwards autocast to bf16 on a bf16-capable CUDA device and are cast back to fp32 before fusion — the trainable stack stays fp32 (no GradScaler, deterministic training). CPU is always fp32.

4. **Profile first.** `DECADIC_CYCLE_PROFILE=1` logs the per-cycle split (`encoders_ms`, `fwd_ms`, `bwd_ms`, `mem_recall_ms`, `stage10_ms`, `gpu_mem_mb`) so you can see where the time goes before reaching for more levers.

5. **Write-behind episodic memory (on by default).** The per-cycle episodic record is persisted to SQLite on a background worker so the disk commit never blocks the cognitive lock. No write is ever lost (it falls back to a synchronous write under backpressure); a just-stored episode becomes queryable ~one cycle later, which is immaterial for associative recall. `DECADIC_EPISODIC_ASYNC` sets the birth default for new agents, and **Agent Settings → Performance (live)** has a per-agent toggle that flips it at runtime (turning it off drains the queue and writes each episode synchronously).

6. **Write-behind LTM consolidation (on by default).** Stage 10's working-memory → long-term-graph consolidation ends in a per-cycle SQLite `commit` (an fsync). The same write-behind contract as episodic moves it to a background worker: no consolidation is lost (order-preserving synchronous fallback under backpressure, flushed before backup/restore/clear), and the graph is read on the *next* cycle (stage 3 re-identification) so the ~one-cycle visibility lag is immaterial. `DECADIC_LTM_ASYNC` sets the birth default; **Agent Settings → Performance (live)** has a per-agent toggle.

7. **Less CPU on the lock (always on, no toggle).** Two pure wins run unconditionally: **(a)** logging is non-blocking — the root logger only enqueues to a `QueueHandler`, and a background `QueueListener` owns the stream/rotating-file handlers (same JSON output, same order), so a per-cycle `logger.info` never blocks on a slow console; **(b)** the camera frame is **decoded once** — `predecode` base64-decodes + CLIP/Whisper-preprocesses each frame a single time at observation-arrival (off the cognitive lock) and stashes the CPU tensors on the observation, so the pooled-vision and patch-token paths share one decode instead of three inside the cycle (byte-identical output). Trace/probe file writes likewise leave the cycle via a background JSONL writer (inherits the existing `DECADIC_CYCLE_TRACE_EVERY` / probe-capture gating).

8. **Persistent Parallel Perceptual Processing (default on).** `DECADIC_PARALLEL_SESSIONS`
   now defaults to `10` and means perception pipeline capacity in the default mode:
   each incoming frame gets a sequence number and enters a bounded perceptual pipeline.
   Workers may stage frames concurrently, but commits into the anonymous scene workspace
   are ordered, so frame `N+1` cannot overwrite the world model before frame `N`.
   The Decadic cycle remains serialized and samples the latest coherent scene. Set
   `DECADIC_PERCEPTUAL_PROCESSING_MODE=batching_observations` or
   `DECADIC_PERSISTENT_PARALLEL_PERCEPTION=0` to restore the legacy behavior where up
   to `K` recent observations are encoded together and recency-pooled into one cycle.
   In **Agent Settings → Workspace capacity**, the segmented control switches between
   **Persistent Parallel Perceptual Processing** and **Batching Perceptual Observations**;
   the `K` slider is labeled as pipeline sessions or batched frames accordingly, and
   live readouts show queue depth, in-flight sessions, committed percepts/sec, dropped
   frames, and scene sample age.

> **Cognition vs. wall clock.** Speeding the cycle from ~1 Hz to 10–20 Hz means the agent *thinks* 10–20× more per real second, while wall-clock-timed processes (metabolic drain, consolidation sync interval, landscape refresh) keep their real-time cadence. The science is unchanged — the agent simply gets far more cognition per unit of the same world time. Adjust the `*_INTERVAL_S` / metabolic constants if you want to preserve the old cognition-to-event ratio.

## Environment — Phase 1 + Phase 2

| Variable | Purpose |
|----------|---------|
| `DECADIC_USE_NEURAL` | `1` (default) trainable stack + PC loss; `0` stub numpy stages (tests) |
| `DECADIC_ENCODER_MODE` | `hf` (default) — frozen CLIP ViT-B/32 + Whisper-small encoder; `zeros` — no HF download, vision/audio off |
| `DECADIC_NEURAL_PRESET` | `tiny` (default) … `1b` — width-scaled ladder; see the preset table (`250m`/`500m`/`1b` are heavy/define-only) |
| `DECADIC_DEVICE` | `cpu` or `cuda` (auto picks CUDA if available when unset) |
| `DECADIC_LEARNING_RATE` | Adam LR (default `1e-4`) |
| `DECADIC_VIABILITY_PE_SCALE` | Maps mean PC loss into viability penalty scale |
| `DECADIC_PAIN_PC_WEIGHT` | Pain scales PC loss (motivational scaffolding) |
| `DECADIC_DATA_DIR` | SQLite episodic stores (default `./data`) |
| `DECADIC_BACKUPS_DIR` | Checkpoint JSON + `agent_{id}_brain.pt` |
| `DECADIC_LOG_DIR` | Rotating JSON logs + optional cycle traces |
| `DECADIC_CYCLE_INTERVAL_S` | Cycle tick interval |
| `DECADIC_PARALLEL_SESSIONS` | Perception pipeline capacity in default persistent mode; batched-frame count in legacy batching mode (default `10`, max `16`) |
| `DECADIC_PERCEPTUAL_PROCESSING_MODE` | `persistent_parallel` (default) or `batching_observations` (legacy recent-frame encode+pool path) |
| `DECADIC_PERSISTENT_PARALLEL_PERCEPTION` | Boolean alias for the default-on mode (`1` default; set `0` to use `batching_observations` when the explicit mode var is unset) |
| `DECADIC_SESSION_RECENCY` | Recency-pooling decay used only by `batching_observations` mode (default `0.7`) |
| `DECADIC_CONSOLIDATION_STUB_INTERVAL_S` | Heartbeat cadence for the consolidation runner when consolidation is OFF (`0` disables) |
| `DECADIC_CYCLE_TRACE_EVERY` | Cycle trace JSONL every N cycles (`0` off) |
| `DECADIC_PE_STUB_EMA_ALPHA` | EMA on the prediction-error magnitude surfaced as a metric (`DECADIC_PE_EMA_ALPHA` alias) |
| `DECADIC_PE_STUB_WEIGHT` | Blend weight of the legacy cycle-counter PE oscillation in the cycle's PE affect. Default `0.0` (removed — the predictive-coding loss is the genuine surprise); tests pin `0.25` for a byte-identical neural baseline |
| `DECADIC_CURIOSITY_ENABLED` | Need-gated curiosity drive (default `1` on; `0` → byte-identical no-curiosity cycle) |
| `DECADIC_CURIOSITY_GAIN` | Pleasure scale of a fully-permitted, fully-learning state (default `1.0`) |
| `DECADIC_CURIOSITY_PROGRESS_WINDOW` | Forward-model PE samples used to estimate learning progress (default `8`) |
| `DECADIC_CURIOSITY_SAFETY_SHARPNESS` | `>1`: curiosity falls off faster as threat/deprivation rises (default `2.0`) |
| `DECADIC_CONSOLIDATION_ENABLED` | Dual-network replay consolidation (default `1` on; `0` → no-op stub heartbeat) |
| `DECADIC_REPLAY_BUFFER_SIZE` | Max transitions retained; lowest-salience evicted (default `2048`) |
| `DECADIC_CONSOLIDATION_REPLAY_BATCH` | Transitions per replay gradient step (default `32`) |
| `DECADIC_CONSOLIDATION_STEPS_PER_BURST` | Replay steps per wake-up before a sync (default `4`) |
| `DECADIC_CONSOLIDATION_SYNC_TAU` | Polyak blend rate `live ← (1-τ)·live + τ·consolidator` (default `0.05`) |
| `DECADIC_CONSOLIDATION_SYNC_INTERVAL_S` | Seconds between replay+sync bursts (default `30`) |
| `DECADIC_CONSOLIDATION_PRUNE_MIN_SALIENCE` | Transitions below this salience are never stored (default `0.0`) |
| `DECADIC_LANDSCAPE_ENABLED` | Live loss-landscape probe (default `0` off; visualization-only, needs the replay buffer) |
| `DECADIC_LANDSCAPE_GRID` | Surface resolution (`grid × grid` points; default `15`, capped at `41`) |
| `DECADIC_LANDSCAPE_SPAN` | Half-width of the α/β sweep in filter-normalized units (default `1.0`) |
| `DECADIC_LANDSCAPE_BATCH` | Replay transitions scored at each grid point (default `8`) |
| `DECADIC_LANDSCAPE_INTERVAL_S` | Seconds between surface refreshes (default `20`) |
| `DECADIC_LANDSCAPE_SEED` | Fixes the random direction basis so refreshes are comparable (default `0`) |
| `DECADIC_ENCODER_PRECISION` | Compute dtype for the **frozen** CLIP/Whisper forwards: `auto` (default → bf16 on Ampere+ CUDA, else fp32), `bf16`, `fp16`, `fp32`. CPU is always fp32; the trainable stack is always fp32 |
| `DECADIC_CYCLE_PROFILE` | `1` logs a per-section cycle split (`cycle_profile … encoders_ms/fwd_ms/bwd_ms/mem_recall_ms/stage10_ms/gpu_mem_mb`) for diagnosing the bottleneck (default `0`) |
| `DECADIC_EPISODIC_ASYNC` | Write-behind episodic persistence — moves the per-cycle SQLite write to a background worker so it never blocks the cognitive lock (default `1` on; the **birth** default for new agents, also a live per-agent toggle in **Agent Settings**; reads lag by ~one cycle — see Performance) |
| `DECADIC_LTM_ASYNC` | Write-behind long-term-graph consolidation — moves stage 10's WM→LTM SQLite commit to a background worker so it never blocks the cognitive lock (default `1` on; **birth** default for new agents, also a live per-agent toggle in **Agent Settings**; graph reads lag by ~one cycle — see Performance) |

### Motivation, goals & long-horizon credit

| Variable | Purpose |
|----------|---------|
| `DECADIC_VIABILITY_MODE` | `metabolic` (default; wall-clock reservoir drain + death) or `immortal` (reservoirs pinned full for long learning runs) |
| `DECADIC_DRIVE_REWARD_ENABLED` | Intrinsic homeostatic-relief reward — phasic pleasure ∝ the per-cycle *reduction* in drive pressure (the positive complement to deprivation pain; default `1` on, pinned off in tests where the legacy periodic placeholder is used) |
| `DECADIC_DRIVE_REWARD_GAIN` | Scale of the drive-reduction relief, clamped to `[0,1]` (default `1.0`, symmetric with `DECADIC_DRIVE_PAIN_GAIN`) |
| `DECADIC_GOAL_ONSET_DEFICIT` | Weighted deficit (`1 − reservoir`) above which a goal latches (default `0.15`) |
| `DECADIC_GOAL_SATISFY_LEVEL` | Reservoir level (0..1) at/above which a goal is achieved (default `0.92`) |
| `DECADIC_GOAL_ABANDON_CYCLES` | Cycles the dominant deficit may differ before the goal is abandoned (default `40`) |
| `DECADIC_GOAL_MAX_CYCLES` | Hard cap on an open goal episode so returns always resolve (default `4000`) |
| `DECADIC_SF_ENABLED` | Successor-features λ-return value head in the consolidator (default `1` on) |
| `DECADIC_SF_GAMMA` | Discount on future features (horizon ≈ `1/(1−γ)`; default `0.97`) |
| `DECADIC_SF_LAMBDA` | Eligibility-trace / λ-return decay (credit smear over the journey; default `0.9`) |
| `DECADIC_SF_LOSS_WEIGHT` | Weight of the SF TD(λ) loss in the consolidator (default `1.0`) |
| `DECADIC_SF_VALUE_WEIGHT` | Max weight of the value-advantage policy-shaping term, post-ramp (default `0.3`) |
| `DECADIC_SF_VALUE_RAMP_CYCLES` | Cycles over which the shaping weight climbs `0 → max` (default `2000`) |
| `DECADIC_HER_ENABLED` | Hindsight relabeling of failed goal episodes (default `1` on) |
| `DECADIC_HER_RELABEL_K` | Relabeled copies pushed per failed episode (default `1`) |
| `DECADIC_IMAGINATION_ENABLED` | Model-based imagined replay during consolidation (default `0` off) |
| `DECADIC_IMAGINATION_HORIZON` | Imagined steps rolled out per sampled start state (default `5`) |
| `DECADIC_IMAGINATION_WEIGHT` | Weight of the imagined-rollout SF loss vs real replay (default `0.25`) |
| `DECADIC_RANDOMIZE_RESOURCES` | Re-scatter food/water each life so location isn't memorizable (default `1` on) |
| `DECADIC_RESOURCE_PLACEMENT_MODE` | `arena` (uniform in the arena disc; default) or `zone` (random within each habitat) |
| `DECADIC_RESOURCE_MIN_DIST` | Keep scattered resources ≥ this far (m) from the spawn origin (default `3.0`) |
| `DECADIC_RESOURCE_FENCE_MARGIN` | Keep resources this far (m) inside the arena fence (default `1.5`) |

### Perception, memory & interpretability

| Variable | Purpose |
|----------|---------|
| `DECADIC_PERCEPTION_MODE` | `discovered` (default; perception-organ + slot-attention object/self discovery from the camera) or `oracle` (entities handed in by the sim — eval scaffold) |
| `DECADIC_PERCEPTION_FEEDBACK_ENABLED` | Top-down predictive perception (precision-gated blend of prediction + encode; default `1` on) |
| `DECADIC_SLOTS_K` | Number of competing anonymous object slots in discovered perception (default `7`) |
| `DECADIC_SLOT_PRESENCE_THRESHOLD` | Minimum slot presence before a proposal can become an object file (default `0.12`) |
| `DECADIC_SLOT_RECON_WEIGHT` | Self-supervised slot feature-reconstruction loss weight (default `0.5`) |
| `DECADIC_SLOT_DIVERSITY_WEIGHT` | Anti-collapse loss discouraging multiple slots from claiming one region (default `0.02`) |
| `DECADIC_SLOT_ENTROPY_WEIGHT` | Encourages confident per-patch slot assignment (default `0.01`) |
| `DECADIC_SLOT_SPATIAL_SEPARATION_WEIGHT` | Discourages center-collapsed object centroids (default `0.02`) |
| `DECADIC_ASSOC_APPEARANCE_WEIGHT` | Appearance-vs-position share of working-memory object-file matching (default `0.6`) |
| `DECADIC_ASSOC_MATCH_THRESHOLD` | Minimum association score to bind a proposal to an existing object file (default `0.35`) |
| `DECADIC_SCENE_WORKSPACE_ENABLED` | Persistent anonymous scene workspace between object files and Working Memory focus (default `1`) |
| `DECADIC_SCENE_ENTITY_TTL_CYCLES` | Cycles an occluded scene entity persists before expiring (default `12`) |
| `DECADIC_SCENE_RELATION_ENABLED` | Anonymous scene relation extraction (`near`, `left_of`, `co_visible`, etc.; default `1`) |
| `DECADIC_SCENE_PREDICTION_ENABLED` | Expose scene prediction-error diagnostics from constant-velocity entity tracking (default `1`) |
| `DECADIC_SCENE_DYNAMICS_ENABLED` | Trainable anonymous scene-dynamics head for next-entity prediction in discovered perception (default `1`; set `0` for constant-velocity fallback) |
| `DECADIC_SCENE_DYNAMICS_WEIGHT` | Live self-supervised scene-dynamics loss weight (default `0.05`) |
| `DECADIC_SCENE_DYNAMICS_MAX_ENTITIES` | Max scene entities predicted/replayed per cycle (default `12`) |
| `DECADIC_SCENE_DYNAMICS_MATCH_THRESHOLD` | Prediction-position confidence threshold for prediction-assisted re-identification (default `0.35`) |
| `DECADIC_SCENE_DYNAMICS_UNCERTAINTY_WEIGHT` | Uncertainty calibration term weight inside the scene-dynamics loss (default `0.05`) |
| `DECADIC_ATTENTION_FOCUS_CAPACITY` | Max scene entities selected into the focus cache per cycle (default `7`) |
| `DECADIC_LTM_MATCH_THRESHOLD` | Appearance threshold for LTM re-identification (default `0.6`) |
| `DECADIC_LTM_MIN_SEEN` | Cycles an object file must persist before LTM consolidation can accept it (default `2`) |
| `DECADIC_LTM_SNAPSHOT_LIMIT` | Max LTM nodes returned to dashboard snapshots; the graph itself remains unbounded (default `64`) |
| `DECADIC_SELF_MODEL_FEEDBACK` | Self-state feedback spine (self-model program): the previous cycle's self-report (A state-of-mind ‖ C narrative ‖ E metacognition) is injected back into the stack so internal state shapes the next cycle. Default `1` on; zero-init projection ⇒ on is byte-identical until it learns. Rebuilds the brain on toggle. (Pinned `0` in tests.) |
| `DECADIC_GWT_ENABLED` | Real global workspace (self-model program): replaces the working-memory EMA blend into A with a capacity-limited winner-take-all competition + ignition threshold + broadcast (to A, the self-model spine, the episodic salience, and the narrative). Default `1` on (set `0` for the legacy EMA blend). Live toggle (no rebuild). Tuned by `DECADIC_GWT_IGNITION_THRESHOLD` (default `0.5` share of salience mass), `DECADIC_GWT_CAPACITY` (coalition size, default `1`), `DECADIC_GWT_TEMPERATURE` (default `1.0`), `DECADIC_GWT_SALIENCE_BOOST` (episodic-salience lift, default `1.0`). (Pinned `0` in tests.) |
| `DECADIC_INTEGRATION_WINDOW_MS` | Explicit temporal-integration window (self-model program): bind a span of bottom-up percepts into one committed "now". The agent acts on the last committed moment until the window (this many wall-clock ms, or `DECADIC_INTEGRATION_WINDOW_MAX_FRAMES` cycles, default `8`) closes and a new now is bound — so longer windows shift when perception updates. Default `200` ms (set `0` for the freshest percept always = now). Live setting (no rebuild). (Pinned `0` in tests.) |
| `DECADIC_PREDICTIVE_AFFECT` | Predictive affect (self-model program): a small forward model predicts the next-step affective context (viability/pain/pleasure/priority) from the previous cycle's actual affect, and the predicted delta colours the episodic proxy before it is projected into the stack — so the agent perceives in light of how it expects to feel. Default `1` on; zero-init output ⇒ on is byte-identical until it learns (it rides the main prediction-error graph). Scaled by `DECADIC_PREDICTIVE_AFFECT_GAIN` (default `1.0`). Rebuilds the brain on toggle. (Pinned `0` in tests.) |
| `DECADIC_REPRESENTED_SELF` | Represented self (self-model program): the agent's interoception (reservoirs), affect, and capability (its discovered body schema) are written as content onto the egocentric self-node, "controls" edges bind the self to its learned body parts, and a compact self-node embedding is fed back through a dedicated zero-init spine ingress — so the self becomes a represented object the agent models. Default `1` on; zero-init ingress ⇒ on is byte-identical until it learns. Rebuilds the brain on toggle. (Pinned `0` in tests.) |
| `DECADIC_MEMORY_EFFICIENT_TRAINING` | Memory-efficient training path (self-model program, hardware-gated): the per-cycle train step uses 8-bit Adam (via `bitsandbytes`) + a bf16 autocast forward **on CUDA**, cutting optimizer-moment + activation memory so the heavy 250m/500m/1b presets can fit a single consumer GPU. Default `1` on; falls back silently to fp32 Adam (no `bitsandbytes` / CPU) so the standard/test path is byte-identical. See `scripts/integration_sweep.py` for the integration (PCI) falsification harness. |
| `DECADIC_LTM_GRAPH` | Persistent long-term knowledge graph (default `1` on; `0` proves the no-LTM path is byte-identical) |
| `DECADIC_COGNITION_TRACE` | Per-cycle read-only "why" explanation (default `1` on) |
| `DECADIC_NARRATIVE_MODE` | Element-C narrative: `off`, `template` (default), or `lm` |
| `DECADIC_PROBE_PATH` | Path to trained interpretability probes (empty disables read-out) |

> These tables cover the headline knobs. Nearly every accessor in `decadic/config.py` reads a
> `DECADIC_*` environment variable with the documented default (homeostasis timing, impact-damage
> calibration, neuroplasticity, slot-attention, and brace tuning live there too); consult
> `decadic/config.py` for the complete set.

Default log directory is `./logs/` when `DECADIC_LOG_DIR` is unset (see app lifespan).

### Checkpoints

`POST /agent/{id}/checkpoint` writes JSON state plus **`agent_{id}_brain.pt`** (stack + proprio encoder + optimizer) when neural mode is on. `POST /agent/{id}/restore` reloads both.
