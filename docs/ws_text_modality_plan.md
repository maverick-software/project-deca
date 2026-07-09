# WS-TEXT — Text as a Sensory Modality and Output Vector

**Status:** design draft (2026-07-08). Companion to the LLM-parent language
program. Mirrors the WS6 audio path (ears + mute-at-birth mouth) for text.

## 1. Goal

Give the agent a **text sense** (it can perceive text the way it perceives
audio) and a **text output vector** (it can emit text the way it emits voice),
so that: an external LLM-parent or a human can type to the agent; the agent's
text output is captured over the API; and a chat window in the GUI shows both
sides. Text is treated as *just another I/O vector* — a different substrate, the
same mechanism as vision and audio.

**Two invariants carried over from the language discussion:**
- **Grounded, not injected.** Text arrives through the senses (encoded,
  perceived) and the agent grounds its meaning itself. Text is *never* written
  into the belief store as a label (the `semantic_graph.py` firewall stands).
  "This is water" as a text percept is fine; `belief.label = "water"` is not.
- **Mute-at-birth, unfolds over cycles.** The text head is zero-initialized like
  the voice head — the agent starts unable to type and must *learn* to. Output
  is one token per cognitive cycle, an utterance accumulating over cycles (like
  phonation accumulates a sound), closed through the loop — never a whole string
  emitted in one cycle (that would be an un-embodied LLM call bolted on).

## 2. Current interfaces this plugs into (file:line)

- **Frozen encoder fusion:** `FrozenEncoders.forward(obs)` returns
  `torch.cat([v, a, p], dim=-1)` (`frozen_encoders.py:510`) — vision
  (`CLIP_POOL_DIM=512`), audio (`WHISPER_POOL_DIM=768`), proprio (~90). Absent
  modality → zero vector (`_vision_zeros`/`_audio_zeros`, `:507-508`). Only the
  CLIP **vision** tower is loaded (no text tower today).
- **Observation schema:** `ObservationMessage` (`schemas.py:33`) has
  `vision`/`audio`/`proprioception`/`events`/`world_state` and `extra="allow"`,
  so a `text` field drops in cleanly.
- **Policy latent + output heads:** `pol_in = lstm_hidden + state_mind_out`
  (`neural_stack.py:164`); `voice_head = nn.Linear(pol_in, VOICE_DIM=9)`
  zero-init at birth (`:552-555`); the runtime latent is
  `pol_in_t = cat([h, state_mind])` (`:1187`). **Additive ingress pattern** for
  lateral/top-down vectors: `other_ingress = nn.Linear(OTHER_VEC_DIM, pol_in)`
  added into `pol_in_t` (`:375, :1213-1216`) — the non-breaking seam.
- **Voice emit (output template):** `Agent._emit_voice` (`runtime.py:3360`)
  reads the per-cycle voice params, checks the phonation gate (index 0,
  `vocal_tract.py`), renders, and surfaces `metrics["voice_params"]`,
  `metrics["voice_phonating"]`. `GET /agent/{id}/audio` (`app.py:673`).
- **Discrete code substrate:** FSQ symbol codes (`symbol.py`) — an *internal*
  abstraction, not a controllable emission; text uses a **dedicated head**, not
  this.
- **Faculty gating:** `CognitionFaculties.voice` / `DECADIC_VOICE`
  (`faculties.py:81`) gates whether the voice head is built/run — the pattern to
  mirror for `DECADIC_TEXT`.
- **Server I/O:** obs in + metrics out on `WS /agent/{id}/cycle`
  (`app.py:1179`, `sender()`); external input template `POST /agent/{id}/give`
  (`app.py:867`); dashboard is a Vite app on :5173 consuming the cycle ws.

## 3. Architecture

### 3a. Text INPUT (the sense)

- **Wire-in:** add `text: dict | None` to `ObservationMessage` —
  `{"data": "<utf-8>", "source": "parent"|"user", "ts": ...}`. Intermittent:
  present only when someone speaks, absent (None) otherwise → zero vector, the
  same as audio silence.
- **Encoder:** a new **frozen text encoder** producing a fixed `TEXT_POOL_DIM`
  vector, added to the fusion. Which encoder is the key grounding decision (§4).
- **Entry point (v1, non-breaking): additive ingress.** Rather than widening the
  frozen-encoder concat (which changes the stack's input width and breaks
  checkpoints), route the text embedding through a zero-init
  `text_ingress = nn.Linear(TEXT_POOL_DIM, pol_in)` added into `pol_in_t`, exactly
  like `other_ingress`. Zero-init ⇒ existing checkpoints and the no-text case are
  bit-for-bit unchanged until text is learned. Flag-gated on `DECADIC_TEXT`.
  - *Target (later): perception concat.* The semantically-pure home for a sense
    is the encoder fusion (`cat([v, a, p, t])`), so text co-enters with the other
    senses. That changes the trunk input dim (a checkpoint migration), so it's a
    deliberate v2 upgrade, not v1.
- **Absence:** no `text` field → zero embedding → text_ingress contributes
  nothing that cycle. Natural, like audio silence.

### 3b. Text OUTPUT (the vector)

- **Head:** `text_head = nn.Linear(pol_in, 1 + V)` zero-init (mute-at-birth),
  mirroring `voice_head`. Index 0 = **emit gate** (the phonation-gate analog:
  emit vs stay silent this cycle); indices `1..V` = logits over a small token
  vocabulary. Per cycle: if `gate > threshold`, emit `token = argmax/sample` over
  the V logits; else silent.
- **Vocabulary (V):** start small and fixed — e.g. a ~40-symbol grapheme set
  (a–z, space, a few marks) or ~64–256 subword units. The agent learns to
  *sequence* tokens; meaning is grounded by which sequences earn responses, not
  assigned. `DECADIC_TEXT_VOCAB` sets V.
- **Per-cycle decode — `Agent._emit_text`** (clone of `_emit_voice`): read the
  head output from the neural cycle, apply the gate, pick the token, append to an
  **utterance buffer**. Surface `metrics["text_token"]` (this cycle),
  `metrics["text_emitting"]` (gate open), `metrics["text_utterance"]` (buffer so
  far). Closed-loop: the emitted text can also re-enter as `obs.text` next cycle
  (self-hearing analog), so the agent perceives what it "said."

### 3c. Turn / utterance assembly

- An utterance is a run of gate-open cycles. **Turn boundary** = gate closed for
  `DECADIC_TEXT_TURN_SILENCE_CYCLES` cycles (default e.g. 3) *or* a reserved
  end-of-utterance token. On boundary, finalize the buffer →
  `metrics["text_last_utterance"]`, clear the buffer. This mirrors how a spoken
  utterance is a run of phonation between silences.

### 3d. API

- **Egress (read the agent):** the `text_*` metrics ride the existing cycle-ws
  `sender()` — the chat GUI streams `text_utterance` live and commits
  `text_last_utterance` on turn end. Plus `GET /agent/{id}/text` returns the last
  utterance for polling clients.
- **Ingress (send to the agent):** `POST /agent/{id}/text?source=user|parent`
  with `{"text": "..."}`, mirroring `/give` — it queues the string to be attached
  to the next cycle's `obs["text"]` (same shape as the audio-intake attach). The
  GUI chat box and the LLM-parent both use this one endpoint.

### 3e. GUI chat window

- A new **chat panel** in the Vite dashboard: renders agent utterances from the
  cycle-ws telemetry (streaming `text_utterance`, committed `text_last_utterance`)
  as "agent" bubbles, and a text input box that POSTs to `/agent/{id}/text` as
  "user" bubbles. Optionally an audio-record button that posts `obs.audio` too, so
  the same window drives both senses. Framework hook: wherever the dashboard
  already subscribes to the cycle ws and renders metric panels (confirm React vs
  Vue under `dashboard/`).

### 3f. Faculty + config

- `CognitionFaculties.text` gated by `DECADIC_TEXT` (default off) — builds the
  text encoder + `text_head` + `text_ingress` only when enabled; zero-init means
  ON-but-untrained == parity. Knobs: `DECADIC_TEXT_ENCODER` (§4),
  `DECADIC_TEXT_VOCAB`, `DECADIC_TEXT_GATE_THRESHOLD`,
  `DECADIC_TEXT_TURN_SILENCE_CYCLES`.

## 4. Key design decisions

1. **Text-input encoder (the grounding fork).** Options, least→most pre-installed
   meaning: (a) **frozen char/byte encoder** (blank-slate; agent grounds text
   from scratch — most faithful, slowest); (b) **frozen small text embedder**,
   non-vision-aligned (consistent with the Whisper precedent — gives "text
   features" the way Whisper gives "audio features," agent still grounds
   meaning); (c) **CLIP text tower** (shared space with vision, so "water" text
   sits near water images — fast grounding but pre-installs cross-modal meaning,
   least faithful to the thesis). Recommend **(a) or (b)**; treat (c) as an
   explicit experiment, not the default.
2. **Entry point:** additive `text_ingress` (non-breaking, v1) vs perception
   concat (pure sense, checkpoint migration, v2). Recommend ingress first.
3. **Output vocabulary:** grapheme set vs subword. Recommend a small grapheme
   set first — maximally learnable, human-readable, and it lets the agent invent
   its own "words" as token sequences rather than being handed morphology.

## 5. Gap analysis

| # | Capability | Current | Gap | Effort |
|---|---|---|---|---|
| T1 | Text in the obs schema | absent; `extra="allow"` | add `text` field + ingest attach | S |
| T2 | Frozen text encoder | none (CLIP vision only) | new encoder → `TEXT_POOL_DIM` vec; zero on absence | M |
| T3 | Text enters cognition | no path | zero-init `text_ingress` into `pol_in_t` | S |
| T4 | Text output head | none (voice only) | zero-init `text_head = Linear(pol_in, 1+V)` + gate | M |
| T5 | Per-cycle decode + utterance buffer | none | `_emit_text` clone of `_emit_voice`; turn assembly | M |
| T6 | API ingress/egress | `/give`, cycle-ws metrics | `POST /agent/{id}/text`, `GET /agent/{id}/text`, `text_*` metrics | S |
| T7 | GUI chat window | none | dashboard panel: stream + input box | M |
| T8 | Faculty + config | `DECADIC_VOICE` pattern | `DECADIC_TEXT` faculty + knobs | S |
| T9 | Grounding measurement | FSQ codes exist | verdict: do symbol codes / `predicts_*` cluster on a token seq when the referent appears? | M |

## 6. WBS

Effort S/M/L; no calendar (work lands in minutes-to-hours); deps by ID.

- **1.0 Input sense**
  - 1.1 (S) `text` field on `ObservationMessage` + runtime read of `obs["text"]`.
  - 1.2 (M) Frozen text encoder (decision §4.1) → `TEXT_POOL_DIM`; zero on absence. Dep: 1.1.
  - 1.3 (S) `text_ingress` zero-init into `pol_in_t`, gated on `DECADIC_TEXT`. Dep: 1.2.
  - 1.4 (S) Tests: no-text == parity (zero-init); a text obs perturbs `pol_in_t`. Dep: 1.3.
- **2.0 Output vector**
  - 2.1 (M) `text_head` (1+V) zero-init + emit gate. Dep: none.
  - 2.2 (M) `_emit_text` per-cycle decode + utterance buffer + turn boundary. Dep: 2.1.
  - 2.3 (S) `text_token`/`text_emitting`/`text_utterance`/`text_last_utterance` metrics. Dep: 2.2.
  - 2.4 (S) Tests: mute-at-birth (zero-init → no emission); gate/turn assembly. Dep: 2.3.
- **3.0 API**
  - 3.1 (S) `POST /agent/{id}/text` (queue → next `obs.text`), mirror `/give`. Dep: 1.1.
  - 3.2 (S) `GET /agent/{id}/text` + ensure `text_*` metrics ride the cycle ws. Dep: 2.3.
- **4.0 GUI chat window**
  - 4.1 (S) Confirm dashboard framework; locate cycle-ws subscription + panel pattern.
  - 4.2 (M) Chat panel: agent bubbles from telemetry, user input box → `POST /text`. Dep: 3.1, 3.2, 4.1.
- **5.0 Faculty/config/validation**
  - 5.1 (S) `DECADIC_TEXT` faculty + knobs. Dep: 1.3, 2.1.
  - 5.2 (M) Grounding verdict (T9): symbol-code clustering on a taught token seq. Dep: 4.2.
  - 5.3 (S) Parity run: `DECADIC_TEXT=0` bit-identical to today. Dep: 5.1.

**Critical path:** 1.1 → 1.2 → 1.3 → (2.x) → 3.x → 4.2 → 5.2.

## 7. Risks

- **Grounding leakage via the encoder.** A vision-aligned or large pretrained
  text encoder pre-installs meaning (decision §4.1). Keep the encoder blank/
  non-aligned to keep the learning honest.
- **The agent may never learn to type without scaffolding.** Zero-init + a hard
  task = silence forever. The LLM-parent's serve-and-return (respond to any
  emission) is the curriculum that bootstraps it, same as babble bootstraps
  voice.
- **Checkpoint width.** Perception-concat (v2) changes input dim; the ingress v1
  avoids it. Don't do the concat migration until text is proven via ingress.
- **Turn latency.** One token/cycle at ~4 Hz means slow "typing." That's correct
  (it's learning to speak, not calling an LLM), but the GUI should stream partial
  utterances so it doesn't look hung.
