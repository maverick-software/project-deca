# PRD: WS6 — The Speech Loop (Ears That Parse, a Mouth That Learns)

**Version:** 1.0 — 2026-07-04
**Status:** Draft for review (sequenced AFTER WS5 closes; M2+ pairs naturally with the MuJoCo era but M0–M1 run on the desk rig)
**Companion:** `ws6_speech_loop_wbs.md`
**Numbering note:** WS6 was unclaimed; if theory-of-mind work later claims a number, it slots after this — protoword comprehension/production (this WS) does not require ToM; conversation does.

**Thesis:** language must be lived, not downloaded. No text corpus, no TTS, no LLM. The mouth is a motor organ that starts incompetent; the ears already exist but pool away everything a word is; meaning arrives only through contingent interaction. Success is defined by falsifiable developmental probes, in the WS3/WS5 house style.

---

## 1. Why (and why now)

WS5 built the prerequisite that makes language *learnable in principle*: compositional internal states (slots = referents, relations = predicates). What remains between Deca and receptive protolanguage is interfacial and environmental, not representational. The audit below found the same disease WS5 cured three times — pooling at a boundary — plus two genuinely missing organs (a production channel and a self-hearing loop) and one missing world property (communicative contingency).

The developmental target ladder, each stage falsifiable and valuable alone:
1. **Word-form familiarity** (statistical segmentation of repeated speech) — infants at 8 months.
2. **Receptive protowords** (spoken label retrieves/boosts the named WM slot) — dog-level comprehension.
3. **Babble + vocal agency** ("this sound is ME") — infant 2–6 months, after the mouth exists.
4. **Vocal forward model** (articulation→sound prediction; compensates under altered feedback).
5. **Imitation** (inverse model reproduces a heard target).
6. **Protoword production** (deprivation elicits the historically-effective label — a word said because it is hungry).

Syntax, conversation, and pragmatics are explicitly POST-ToM and out of scope.

## 2. Gap analysis (2026-07-04 code inventory)

### 2.1 What exists

- **Ears (hardware):** `FrozenSensoryEncoders` (decadic/nn/frozen_encoders.py) — frozen Whisper-small encoder over `obs.audio` (base64 wav → log-mel → encoder hidden states), CPU-predecoded in the prefetch workers, per-observation embedding cache. `WHISPER_POOL_DIM=768`, fused with CLIP+proprio into the stack input.
- **Symbolic audio:** `perceptual_state` keeps `audio_duration_s`/`audio_rms`; `MemorySlot.audio_intensity` with fast `AUDIO_DECAY`; `_bind_events_discovered` attributes events to the most salient in-view slot — **joint attention already exists** as the associative substrate for word learning.
- **Motor mouth-mount point:** the action schema (`neural_pipeline.py` ~L1971) is a parameter dict — `{"type": "motor", "parameters": {"ctrl": [...], "babble_sigma": ..., ...}}` — with per-actuator PD targets as the real efferent output and **motor babbling already a first-class concept** (`babble_sigma`, `motor_babble_sigma` curriculum override).
- **Agency machinery:** `update_agency` (efference↔motion contingency, EMA, promotion to `self_part`, touch cross-check) — the exact mechanism vocal self-discovery needs, currently visual-motion-only.
- **From WS5:** token lanes + keyed reads + relational core; the binding scenario scaffold (`docs/binding_scenarios/`, oracle-seam injection, `check_gate_probe.py`-style verdict pattern); frozen interface-dim discipline.

### 2.2 The gaps

| # | Gap | Current state | Required |
|---|---|---|---|
| G1 | **Audio temporal structure destroyed** | Whisper hidden states are mean-pooled to one 768-d vector per observation (`_fit_dim`); a word's identity lives in the SEQUENCE | Audio token lane: T downsampled Whisper frames (est. 4–8 tokens/obs) exposed to the stack — the WS5 pattern, 4th application |
| G2 | **Audio pooled into the fused percept** | 768-d audio ⊕ CLIP ⊕ proprio → one z0; no audio-specific familiarity signal | Keyed read over audio tokens + an audio sub-key (percept-key discipline) so word-forms can habituate/novelty-spike independently |
| G3 | **No production channel** | Action parameters carry locomotion only | `voice_u` head (est. 6–8 dims: f0, energy, voicing, 3–5 articulator params), zero-init, emitted per cycle beside `ctrl` |
| G4 | **No self-hearing loop** | The world never mixes agent output into `obs.audio` | Body/world renders `voice_u` through a source-filter articulatory synth (interpolated between ~9 Hz cycle frames ≈ syllable rate) and mixes it into the audio observation; desk rig = speaker + microphone, sim = direct mix |
| G5 | **No vocal forward model** | PC heads predict percept/intero/tactile/effort; nothing predicts auditory consequences of vocal efference | Zero-init forward head: predict next-cycle audio tokens from `voice_u`; rides the existing PC training graph (trained for free, like the affect predictor) |
| G6 | **Mouth absent from the body schema** | Body map/agency know limbs; no articulator proprioception | Articulator positions return as proprio fields; vocal efference↔audio-onset contingency feeds `update_agency` (self-voice = "mine") |
| G7 | **No communicative pressure** | Resources arrive ambiently; nothing is ever gained by signaling | Request-game scenario: a scripted caregiver grants resources contingent on vocalization (any call → shaped calls → the label) |
| G8 | **No speaking partner** | Synthetic client emits no speech | Scenario-scheduled caregiver utterances (pre-rendered or mic-live) with naming contingencies tied to entity salience — extends the WS5 binding scenario, same oracle seam |

## 3. Design (settled shapes; sizing open until measured)

- **3.1 Audio tokens (G1/G2):** `FrozenSensoryEncoders` exposes `audio_tokens(obs) -> (T, D_a)` — adaptive-pooled Whisper frames (T fixed, est. 6; D_a fit from 768, est. 64 via frozen projection), frozen interface dims per house rule. Stack gains a keyed read (flag `DECADIC_AUDIO_TOKENS`, zero-init ingress, identical discipline to WS5-M1/M2). An audio familiarity key (16-d, percept-key style) joins episodic storage — layout amendment, versioned like the WS4 embedding freeze.
- **3.2 Voice head (G3):** stack emits `voice_u` (flag `DECADIC_VOICE`, zero-init head ⇒ silence at init — the newborn does not speak); action schema gains `"voice": [...]`; babble applies `babble_sigma`-style exploration noise on the voice channel during a babble curriculum window.
- **3.3 Vocal tract (G4):** body-side source-filter synthesizer (formant synth first; articulatory model optional later) renders voice params to waveform, world mixes into `obs.audio` with location/attenuation. Desk rig: literal speaker+mic. **The loop, not the synth quality, is the point.**
- **3.4 Forward model + agency (G5/G6):** zero-init `voice_forward` head predicting next audio tokens from (voice_u, current audio tokens); its error joins `pc_parts`. Vocal agency: onset-correlation between voice efference and audio energy feeds the existing agency EMA; mouth registers in the body map.
- **3.5 Scenarios (G7/G8):** `gen_speech_scenario.py` extends the binding scenario format with `utterances` (schedule: wav/label id, tied to entity salience windows) and `request_phases` (resource granted iff vocal energy/label emitted). Verdict scripts in the `check_gate_probe.py` mold.
- **3.6 Parity and cost:** every flag off ⇒ byte-identical (suite-enforced); all new heads zero-init; audio-token and voice-head cycle cost measured with `bench_relational.py`'s pattern before defaults move; checkpoint compat via the versioned-bundle discipline (WS3B GateNet precedent).

## 4. Success criteria (the probe ladder — each falsifiable, flags-off must fail where marked)

1. **P0 parity/cost:** suite green flags-off and flags-on; measured cost inside the cycle envelope.
2. **P1 word-form familiarity:** repeated caregiver word-forms habituate (audio-key novelty falls); a novel word-form spikes. *Flags-off (pooled audio) must fail the discrimination* — the built-in ablation.
3. **P2 receptive protoword:** after naming-game exposure, the spoken label ALONE (entity absent) measurably boosts the named slot's salience/retrieval vs. control labels — word→referent. Balanced novel-label controls (WS5 leakage discipline).
4. **P3 vocal agency:** with voice on, self-voice binds as "mine" (agency ≥ threshold); MUTE TEST — efference without rendered sound collapses the binding.
5. **P4 forward model:** PC error on self-audio falls with babble experience; pitch-shifted feedback produces measurable compensation (altered-auditory-feedback paradigm).
6. **P5 imitation:** played vowel target reproduced with formant error under threshold, from the inverse of the learned forward model.
7. **P6 protoword production:** in request games, deprivation elicits the historically-effective label above chance and above yoked-control rates — production grounded in need.

## 5. Risks

- **Bandwidth honesty (the WS3 lesson):** if Whisper-small pooled frames cannot discriminate the caregiver's word-forms at T=6 tokens, P1 fails for input reasons, not mechanism reasons — measure discriminability offline FIRST (M0 acceptance) before building upward.
- **Cycle-rate vs speech timescale:** 9 Hz control ≈ syllable rate; sub-syllabic articulation is out of reach by design at this stage (protowords are syllabic; fine).
- **Feedback loudness/self-masking:** the agent's own voice could swamp caregiver audio; mix with attenuation + efference-aware suppression (biology does the same).
- **Reward hacking in request games:** any-vocalization shaping must tighten to label-specific contingency, or the agent learns to scream, not speak (shaping schedule is part of the scenario spec).
- **Scope creep toward syntax:** explicitly out; the WS ends at P6.

## 6. Dependencies

Consumes WS5 (token lanes, keyed-read pattern, relational core, binding scenario scaffold, frozen-interface discipline) and WS4 (episodic key layout versioning). Independent of the learned gate (WS3B). MuJoCo improves the rig but M0–M1 (ears + receptive naming) run on the desk with the synthetic client; M2+ (the mouth) needs only the body-side synth, which is sim-or-desk, not MuJoCo-gated.
