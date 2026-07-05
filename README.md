# Project Deca — The Decadic Cycle Cognitive Architecture

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Commercial license available](https://img.shields.io/badge/commercial%20license-available-green.svg)](./LICENSING.md)

**A continuously-running, embodied cognitive architecture that learns from its own
experience — with no large language model anywhere in the loop.**

Deca is a mind built as a *process*, not a text predictor. It perceives through a camera
and body, remembers, attends, feels its own viability (hunger, damage, curiosity), and
learns online from prediction error — one ten-stage cognitive cycle at a time, forever.
Nothing about it is pretrained on human text; everything it knows, it lived.

This repository is the working software realization of the **Decadic Cycle of Expression**,
the framework introduced in *[The Architecture of Awareness: Decoding Consciousness](https://www.amazon.com/dp/1963092066)*
by Charles R. W. Sears. It is a research instrument: a testbed for specific, falsifiable
claims from psychology and neuroscience about how a mind might work.

---

## Table of contents

- [What this is — and what it is *not*](#what-this-is--and-what-it-is-not)
- [Why an ML researcher might care](#why-an-ml-researcher-might-care)
- [The scientific thesis](#the-scientific-thesis)
- [Architecture at a glance](#architecture-at-a-glance)
- [What it is built from](#what-it-is-built-from)
- [The research program (falsifiable by design)](#the-research-program-falsifiable-by-design)
- [Quickstart](#quickstart)
- [Watch it think](#watch-it-think)
- [Repository map](#repository-map)
- [Status and roadmap](#status-and-roadmap)
- [Theoretical lineage](#theoretical-lineage)
- [License, citation, trademarks](#license-citation-trademarks)

---

## What this is — and what it is *not*

Deca is an **agent-first cognitive architecture**. The unit of computation is a *cognitive
cycle*, not a token. It runs as a live server: a body streams observations over a WebSocket,
the mind runs its ten-stage loop, and an action goes back to the body — indefinitely, while
learning every cycle.

### How it differs from a large language model

| | Large language model | Project Deca |
|---|---|---|
| **What it models** | The distribution of human text | A single agent's ongoing sensorimotor experience |
| **Where knowledge comes from** | Pretraining on a corpus | Lived interaction — no corpus, no labels, no pretrained weights in the cognitive core |
| **Objective** | Next-token likelihood | Predictive-coding error + forward-model error, minimized online every cycle |
| **Motivation** | None intrinsic (RLHF is external) | Two innate drives only: homeostatic viability (pain/pleasure) and need-gated curiosity |
| **Temporality** | One forward pass per prompt | A persistent process with memory, mood, and metacognition that carry across cycles |
| **Scale story** | Capability scales with parameters + data | Capability scales with *richer experience*; the architecture is the lever, not the parameter count |
| **Size** | Billions of parameters | ~0.8M (default `tiny`) to ~1B (heavy tier); the flagship runs happily at ~25M |

There is **no LLM in the cognitive loop.** Frozen CLIP and Whisper encoders can turn pixels
and audio into vectors (an optional sensory front-end), but the cognition — the ten trainable
stages — is a small, purpose-built neural stack trained from scratch by the agent's own life.
An optional templated/LM narrator can *describe* what the agent is doing for interpretability,
but it never feeds back into cognition.

### What it is *not*

It is not a chatbot, not a benchmark-chaser, and not (yet) a claim that anything is conscious.
It is an attempt to build the *mechanisms* that theories of mind say a cognitive system needs —
and then to test, falsifiably, whether those mechanisms produce the behaviors those theories
predict.

## Why an ML researcher might care

- **A concrete, runnable alternative to the LLM paradigm.** If you are interested in embodied
  cognition, world models, active inference, intrinsic motivation, or continual learning, this
  is a full system you can start in minutes and watch learn in real time.
- **Everything is online and self-supervised.** No dataset, no reward function to design, no
  fine-tuning. The agent's only teachers are prediction error and its own body.
- **Falsifiable by construction.** Every major faculty ships with an ablation: a flag turns it
  off, and with it off the system is *byte-identical* to the baseline. Claims are tested with
  probes that the flags-off system must fail and the flags-on system must pass — not asserted.
- **Fully instrumented.** A live dashboard renders the actual network in 3D, the loss landscape
  of the agent's real weights, per-stage timings, the memory graph growing, and a human-readable
  "why" trace of every decision.
- **Small and legible.** The flagship cognition is ~25M parameters. You can read the whole
  forward pass. Interpretability is a design constraint, not an afterthought.

## The scientific thesis

Deca is an engine for testing a specific stack of ideas from cognitive science and
neuroscience. Each is implemented as a real mechanism and paired with an experiment.

- **Predictive coding / the Bayesian brain** (Rao & Ballard; Friston). Cognition is prediction;
  learning is the minimization of prediction error. Deca's every-cycle objective is a
  predictive-coding loss across the stages plus forward-model errors (proprioceptive,
  interoceptive, tactile). There is no other training signal.
- **Homeostasis as the root of value** (Damasio; homeostatic RL, Keramati & Gutkin). Value is
  not given — it is felt. Three reservoirs (hydration, energy, integrity) define viability;
  depletion is convex *pain*, and moving back toward setpoint is phasic *pleasure*. The
  satisfier (food, water) is never labeled; it is discovered from experienced transitions.
- **Intrinsic motivation as learning progress** (Oudeyer & Kaplan; Schmidhuber). Curiosity
  rewards the *reduction* of forward-model error — learning progress, not raw surprise — so it
  sidesteps the "noisy-TV" trap, and it is need-gated: a threatened agent stops exploring.
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly). A fast episodic store
  (a per-cycle diary) and a slow semantic graph (a hippocampal-style index) with dual-network
  replay consolidation and Polyak soft-sync between a live and a sleeping stack.
- **Global Workspace Theory** (Baars; Dehaene). A capacity-limited winner-take-all competition
  with an ignition threshold and broadcast, replacing a naive attention blend.
- **The self-model program** (Metzinger; Seth). A self-state feedback spine, predictive affect,
  and a represented self — the agent modeling itself as an object — each zero-initialized so
  "on" is byte-identical to "off" until experience moves it.
- **Variable binding and systematicity** (Fodor & Pylyshyn; Treisman). The current frontier:
  carrying discrete entity "slots" across the neural boundary so the system can represent
  *relations* ("the wolf is behind the rock" ≠ "the rock is behind the wolf") — the prerequisite
  for compositional thought, tested by a novel-combination generalization probe.

The framing question behind all of it — from the originating book — is whether a system built
this way develops the functional signatures that theories associate with awareness. Deca does
not claim to answer that. It is built to make the question *empirical*.

## Architecture at a glance

The mind runs the **Decadic Cycle of Expression**: ten stages from perception to behavior,
each a small trainable module, orchestrated by `decadic/cycle/neural_pipeline.py`.

```
 1 Sensory perception            →  6 Emotional / physiological update (GRU)
 2 Experience framing & fusion   →  7 Reprioritization & state-of-mind (LSTM)
 3 Memory retrieval / heuristics →  8 Strategy formation (policy head)
 4 Risk-utility + curiosity gate →  9 Behavioral response (action to the body)
 5 Pre-normative conclusion      → 10 Normative memory mapping (feeds next cycle)
                         ↑______________________________________|
```

A persistent **State Bus** carries continuous state across cycles — the substrate that makes it
a *process* rather than a function call:

| Element | Meaning |
|---|---|
| **A** | State of mind |
| **B** | Emotional / physiological state (pain / pleasure / curiosity affect) |
| **C** | Internal narrative |
| **D** | Current priority (`explore` / `investigate` / `avoid` / …) |
| **E** | Metacognition |
| **F** | Action history / efference copy |

Around this core sit the faculties the thesis calls for: a pre-cognitive **perception organ**
that discovers anonymous object files from raw camera frames (no labels ever reach cognition);
a bounded **working memory** and unbounded **semantic graph**; an **attention gate** that decides
per-cycle whether a percept deserves deliberate stage-4 thought; a **relational core** that makes
entity relations computable; and the **memory backends** (LanceDB vectors + Kuzu graph) that keep
recall sub-millisecond at full-corpus fidelity.

Learning is **online, self-supervised, reward-free.** One Adam step per cycle on the
predictive-coding + forward-model objective. Set `DECADIC_USE_NEURAL=0` to swap the trainable
stack for a fast numpy stub (used by the test suite).

## What it is built from

- **Python 3.11+, PyTorch** — the trainable ten-stage stack (fusion transformer, risk MLP,
  narrative encoder–decoder, GRU/LSTM, policy head), trained from scratch.
- **FastAPI + WebSocket** — the agent runs as a live server; bodies and dashboards attach over HTTP/WS.
- **MuJoCo** — an optional physical humanoid body with hands, feet, joint proprioception, and
  touch sensing, driven by the mind's actions and streaming its senses back.
- **Frozen CLIP + Whisper** (optional, `DECADIC_ENCODER_MODE=hf`) — sensory front-end only; they
  turn pixels/audio into vectors and are never trained. A `zeros` mode runs with no download.
- **LanceDB + Kuzu** — episodic vector store (with a full-mirror in-RAM cache) and semantic
  knowledge graph, both off the cognitive critical path.
- **React/Vite dashboard** — live 3D brain map, loss-landscape probe, memory graph, cognition
  trace, and body viewer.

## The research program (falsifiable by design)

Development proceeds as workstreams, each ending in an experiment whose result could refute the
mechanism. A representative slice:

- **Learning is real.** Predictive-coding loss falls ~90%+ over a run from the agent's own
  experience; verified, root-caused, regression-tested.
- **Attention gate.** A per-cycle decision to think hard or coast, validated by a "startle"
  probe: threat reflex fires 100%, ambient novelty stays calm, genuinely novel stimuli spike —
  and a *revisited* location correctly does **not** spike, because the agent remembers it.
- **Memory at scale.** Full-corpus episodic recall in <1 ms via a write-through mirror; the
  semantic graph's writes run entirely off the critical path so an embodied agent thinks at the
  cycle's own ceiling.
- **Relational binding (current frontier).** Slots crossing the neural boundary so relations
  become computable, with a built-in ablation: flags-off must fail novel entity pairings (it
  structurally cannot represent them); flags-on must generalize.
- **On the roadmap.** A learned attention gate; a full speech loop (a mouth that babbles, hears
  itself, and learns to speak the way an infant does — no TTS, no text); embodied validation of
  every probe under MuJoCo.

Design docs and PRDs for each live in [`docs/`](./docs/); results and benchmark reports in
[`reports/`](./reports/).

## Quickstart

```bash
# 1. Environment
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  Unix:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# 2. Run the mind (CPU works; a CUDA build of PyTorch is ~10–20× faster)
python -m uvicorn decadic.api.app:app --host 127.0.0.1 --port 8765

# 3. Give it something to experience — a synthetic stream…
python scripts/synthetic_ws_client.py --host 127.0.0.1 --port 8765 --steps 200

# …or a physical body (installs MuJoCo)
pip install -e ".[body]"
python scripts/mujoco_decadic_adapter.py --steps 0 --audio --port 8765
```

With no environment variables set, you get the **canonical configuration**: every validated
faculty on — full predictive-coding cognition, homeostatic drive, curiosity, dual-network
consolidation, global workspace, the self-model program, relational binding, the attention gate
at its validated operating point, and the LanceDB + Kuzu memory stack. The mind is not
configurable à la carte; env vars exist for *ablation and diagnosis*, not for assembling
cognition. On a fresh run the agent starts life knowing nothing and learns from the first cycle.

> **Blind-and-deaf fast mode:** `DECADIC_ENCODER_MODE=zeros` skips the ~1 GB CLIP/Whisper
> download — only proprioception reaches the network, but cycles are much faster.

Full operational procedures — the desktop launcher, the web dashboard, starting/stopping bodies,
cloud-GPU deployment, the complete environment-variable reference, and the tuning guide — live in
**[docs/operations_guide.md](./docs/operations_guide.md)**.

## Watch it think

Start the dashboard (`cd dashboard && npm run dev`) and open `127.0.0.1:5173`. You get:

- a **3D brain map** of the actual network, clusters lighting up with the last cycle's real activations;
- the **loss landscape** of the agent's live weights, reshaping as it learns (filter-normalized, Li et al. 2018);
- the **viability gauge**, pain/pleasure, priority, PC-loss and A–F state strips in real time;
- the **semantic graph** growing as the agent consolidates what it re-encounters;
- a human-readable **"why" trace** of every decision (`GET /agent/{id}/explain`);
- and, with a MuJoCo body running, the **humanoid** being driven by the mind's own actions.

Nothing on the dashboard feeds back into cognition — it is a window, not an input.

## Repository map

```
decadic/
  cycle/        the ten-stage Decadic pipeline + the attention gate & relational core
  state/        State Bus (A–F), working memory, viability, curiosity, self-model
  perception/   pre-cognitive perception organ, anonymous object files, scene workspace
  memory/       episodic store (LanceDB) + semantic graph (Kuzu) + consolidation
  nn/           the trainable neural stack, frozen encoders, faculties, presets
  consolidation/ dual-network replay, successor features, loss-landscape probe
  embodiment/   MuJoCo body integration, stances, NPC village
  api/          FastAPI server, routes, saved agents, cloud deploy
  training/     Skill Dojo / Perception Dojo curricula
scripts/        body adapter, synthetic client, diagnostic & benchmark harnesses
dashboard/      React/Vite live UI
docs/           PRDs, WBS, design contracts, the operations guide
reports/        benchmark & probe results
tests/          ~830 tests; the suite is byte-identical across GPU/precision/async knobs
```

## Status and roadmap

Deca is an active research preview under continuous development. Cognition, memory, perception,
attention, and the embodied stack are all live and instrumented; the current frontier is
relational binding (compositional thought), followed by a learned attention gate and an
embodied speech loop. The system is designed to be operated by its successors: every pipeline is
runnable from documented runbooks and diagnostic harnesses, and every faculty carries a
flags-off parity guarantee so the baseline is always one environment variable away.

```bash
python -m pytest -q      # ~830 tests, pinned to cpu / zeros / fp32 for determinism
```

## Theoretical lineage

The architecture originates in *The Architecture of Awareness: Decoding Consciousness*
(Sears, ISBN 1963092066). The mechanisms it implements draw on, and are used to test, a
literature including: Rao & Ballard and Friston (predictive coding, active inference);
Damasio and Keramati & Gutkin (homeostasis and value); Oudeyer & Kaplan and Schmidhuber
(intrinsic motivation); McClelland, McNaughton & O'Reilly (complementary learning systems);
Baars and Dehaene (global workspace); Metzinger and Seth (self-models); Fodor & Pylyshyn and
Treisman (systematicity and feature binding); and Li et al. 2018 (loss-landscape visualization).
Deca does not adjudicate these theories — it makes them *runnable*, so their predictions can be
tested against the behavior of a single, continuously-living artificial agent.

## License, citation, trademarks

**Dual-licensed.** Open source under [GNU AGPL-3.0-or-later](./LICENSE): free to use, study,
modify, and share — with the AGPL §13 condition that running a modified version as a network
service obliges you to share your modified source with that service's users. Because Deca is
built to be served over an API, that clause is deliberate. A separate **commercial license**
without the copyleft/network-source obligations is available — see [LICENSING.md](./LICENSING.md).

All third-party dependencies are permissive (MIT/BSD/Apache-2.0/HPND) — none copyleft; their
attribution notices are in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

**Citing.** If you use this software or the architecture, please cite **both** the software and
the originating book; a machine-readable citation for both is in [CITATION.cff](./CITATION.cff).

- **Software:** Sears, Charles Richard Wayne. *Decadic Cycle Cognitive Architecture* (software), 2026. AGPL-3.0-or-later.
- **Framework:** Sears, Charles R. W. *The Architecture of Awareness: Decoding Consciousness.* ISBN 1963092066.

**Trademarks.** *"Decadic Cycle of Expression"*, *"Project Deca"*, and *"Deca"* are trademarks of
Charles Richard Wayne Sears and are **not** granted by the code license. You may state that your
work is "based on" the Decadic architecture, but may not name a derivative in a way that implies
it is the original or is endorsed.

**Contact** — Charles Richard Wayne Sears: [contact form](https://www.charlesrsears.com/#connect)
· charles.r.sears@gmail.com · [LinkedIn](https://www.linkedin.com/in/charlesrsears/)

---

© 2026 Charles Richard Wayne Sears. The Decadic Cycle of Expression framework originates in
*The Architecture of Awareness: Decoding Consciousness*.
