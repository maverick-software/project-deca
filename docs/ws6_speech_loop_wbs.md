# WBS: WS6 — The Speech Loop

**Companion:** `ws6_speech_loop_prd.md` · Dependency order only. ⚙ = needs the live rig (GPU/mic/speaker).
**Hard gate:** M0.4's measured word-form discriminability decides whether anything above M0 proceeds on Whisper-small or the encoder question reopens first (PRD risk 1 — the WS3 bandwidth lesson, paid up front this time).

---

## M0 — Ears: the audio token lane

**M0.1 Audio inventory freeze.** Document the exact audio path (obs.audio → predecode → Whisper → `_fit_dim` pooling → fusion) with refs; freeze the token-lane interface dims (T tokens × D_a, est. 6×64) and the 16-d audio familiarity key as named constants + layout test (the WS5-M0.1 pattern).
**M0.2 `audio_tokens()` adapter.** FrozenSensoryEncoders exposes T downsampled Whisper frames (frozen projection); zeros-mode fallback preserved; per-observation cache extended.
*Accept:* unit tests — shapes, determinism, zeros-mode parity, cache hit.
**M0.3 Stack keyed read** (flag `DECADIC_AUDIO_TOKENS`): zero-init ingress, masked cross-attention, pipeline plumbing — verbatim WS5-M1 discipline.
*Accept:* off-ignores / zero-init parity / content sensitivity; full suite green both states.
**M0.4 ⚙ Discriminability measurement (THE GATE).** Offline: record ~20 word-forms (caregiver vocabulary), run the token lane, measure pairwise separability (linear probe / cosine margins) at the frozen dims.
*Accept:* separability report in `reports/`; go/no-go recorded in the PRD. No-go ⇒ encoder follow-up before M1.
**M0.5 ⚙ Continuous auditory intake service (PRD 3.8, G9).** `AudioIntake` background worker: callback input stream (system mic and/or Rig-1 bus tap) → ring buffer → freshest chunk (≤250 ms, drop-oldest) attached at observation ingest; client-audio-wins precedence; energy-threshold silence gate before Whisper encode; `DECADIC_AUDIO_INTAKE=off|mic|bus` (default off).
*Accept:* flag-off parity (byte-identical, suite-enforced); unit tests — ring-buffer continuity across cycle jitter, precedence, silence-gate hysteresis; live smoke ⚙ — with the client sending NO audio, speaking into the room moves `audio_rms`/the audio key while silence does not; encode-cost delta measured with the intake on.
Parallel to M0.4 (does not gate on it; hearing the room needs no word discrimination).

## M1 — Receptive naming (desk-runnable)

**M1.1 Speech scenario format + generator.** Extend the binding scenario: `utterances` schedule (label ↔ entity-salience windows, balanced novel-label controls per the WS5 leakage checklist) + pre-rendered wav bank; `gen_speech_scenario.py`.
**M1.2 Client + seam.** Synthetic client attaches scheduled wavs to `obs.audio`; mic-live mode is FIRST-CLASS via M0.5's intake (Charles speaks the labels instead of the wav bank — same probe, live caregiver); audio familiarity key written to episodic storage (versioned layout amendment).
**M1.3 ⚙ P1 word-form familiarity probe.** Habituation/novelty verdict over the audio key; **flags-off ablation must fail the discrimination**.
**M1.4 ⚙ P2 receptive protoword probe.** Label alone boosts the named slot vs controls; verdict script in the gate-probe mold.
*Accept (M1):* P1 and P2 verdicts archived; ablation direction confirmed.

## M2 — Mouth: hardware and silence

**M2.1 `voice_u` head** (flag `DECADIC_VOICE`): zero-init (silence at init), emitted in the action dict as `"voice"`; checkpoint compat via versioned bundle key.
**M2.2 Vocal tract + mixing bus (Rig 1, PRD 3.7).** Body/world-side formant synthesizer (pure numpy, ~1,600 samples/cycle at 16 kHz, param interpolation across cycle frames); world-side mixing bus sums agent voice + scheduled utterances with distance attenuation into the next `obs.audio` chunk. Optional monitor tee to operator speakers.
*Accept:* unit tests — synth determinism, click-free interpolation at frame boundaries, bus attenuation math; loopback smoke ⚙ (`run_binding_smoke.ps1` pattern) shows the rendered waveform re-entering observation on the next cycle.
**M2.3 Mouth in the body schema.** Articulator positions as proprio fields; body-map registration.
**M2.4 ⚙ Desk rig (Rig 2, PRD 3.7).** Callback-driven output stream + ring buffer to a physical speaker; the INPUT side reuses M0.5's intake service (mic mode) unchanged; efference-aware self-masking hook.
*Accept:* measured end-to-end loop latency (synth→speaker→mic→obs) under one cycle period; the M2.2 loopback smoke passes UNCHANGED on the physical rig — the rig-indifference principle, enforced. Not required for M3–M6 training (Rig 1 suffices); required before any live naming-game session.
*Accept (M2 overall):* suite green both flag states.

## M3 — Babble and vocal agency

**M3.1 Voice babble curriculum.** `babble_sigma`-style exploration on the voice channel, curriculum-windowed.
**M3.2 Vocal agency binding.** Voice-efference↔audio-onset correlation feeds `update_agency`; self-voice promotes to "mine".
**M3.3 ⚙ P3 probe + mute test.** Agency binds with the loop closed; muting the render collapses it.

## M4 — Vocal forward model

**M4.1 `voice_forward` head** (zero-init): predict next audio tokens from (voice_u, audio tokens); error joins `pc_parts` (trained for free on the main graph).
**M4.2 ⚙ P4 probe.** Self-audio PC error falls over babble experience; pitch-shifted feedback ⇒ compensation (altered-auditory-feedback paradigm).

## M5 — Imitation

**M5.1 Inverse-model probe rig.** Target vowel played; agent's reproduction scored by formant error (uses M4's forward model; no new training machinery — inversion via the existing active-inference pull).
**M5.2 ⚙ P5 verdict.**

## M6 — Protoword production

**M6.1 Request-game scenario.** Caregiver grants resources contingent on vocalization; shaping schedule any-call → label-specific (anti-scream design, PRD risk 4); deprivation drives the need.
**M6.2 ⚙ P6 verdict.** Deprivation elicits the historically-effective label above chance and yoked controls. **This is the workstream's headline result: a word uttered because it is hungry.**

## Cross-cutting

- Parity: every flag off ⇒ byte-identical; suite-enforced at each milestone.
- Cost: token lane + voice head measured (`bench_relational.py` pattern) before any default flips.
- Checkpoint/save-load: new heads ride the versioned bundle (WS3B/WS5 precedent); route-level round-trip test extended once per new module family.
- Artifacts: every probe run archived under `reports/` with verdict logs, like gateprobe/bindsmoke.

## Dependency graph

```
M0.1 -> M0.2 -> M0.3 -> M0.4 (GATE)
M0.1 -> M0.5 (continuous intake; parallel to the gate, reused by M1.2 and M2.4)
M0.4 -> M1.1 -> M1.2 -> M1.3 -> M1.4
M0.4 -> M2.1 -> M2.2 -> M2.3 -> M3.1 -> M3.2 -> M3.3 -> M4.1 -> M4.2 -> M5 -> M6
M2.2 -> M2.4 (desk rig: gates LIVE sessions only; M3-M6 train on Rig 1)
M1 and M2+ parallelize after the M0.4 gate.
```

### Revision 2026-07-06: phonation gate (silence is a decision)

First live probe finding, root cause: there was NO path where the agent does
not speak. The voice head emitted every cycle, all-zero params mapped to a
deliberate "faint hum" (dead-gradient avoidance), and the runtime rendered
unconditionally -- an involuntary permanent carrier tone that held the
intake's silence gate open via loopback. Fixes, in layers:

- VOICE_DIM 8 -> 9: index 0 is now a PHONATION GATE (vocal-fold adduction,
  separate from pitch/articulation). Zero-init head => gate closed => EXACT
  digital silence. Vocalizing requires driving the gate above
  DECADIC_VOICE_PHONATION_THRESHOLD (default 0.0). Gate interpolates across
  the frame (click-free open/close).
- Runtime consults phonating() BEFORE rendering: a closed gate renders
  nothing, loops back nothing, plays nothing (metrics: voice_phonating).
- The "hum for gradient" rationale is retired: the efference->sound gradient
  belongs to the babble curriculum (M3.1), which explores by OPENING the
  gate -- not to a constant carrier tone.
- The intake's efference-aware self-mask (self_rms_ema floor) stays as
  defense-in-depth for when the agent genuinely vocalizes.

### Live validation session 2026-07-06 (audio-cognition probe, 4 runs)

The first end-to-end test of ears+mouth against a live human. Findings, in
discovery order, each now fixed and regression-covered:

1. Involuntary hum -> phonation gate (revision above). Confirmed live:
   self_rms_ema = 0.0 across the whole session -- the agent is truly silent
   by default.
2. Probe time-order confound -> transition scoring with a discarded warmup.
3. Host default mic delivered digital silence -> DECADIC_AUDIO_DEVICE
   selection + scripts/mic_check.py sweep (working device found at index 4).
4. Working mic at ~0.002 RMS speech (below the gate) -> DECADIC_AUDIO_GAIN
   capture stage (mic path only; loopback stays true-level).
5. Measured room reality: per-chunk energy CANNOT separate speech from a
   lived-in room (typing transients ~ speech level; inter-word gaps ~
   silence level; ambient p90 0.107 vs speech p50 0.009 at gain 30).
   Design conclusion recorded: the energy gate's job is dead-air economy
   (skip Whisper on nothing), NOT discrimination -- selection belongs to
   cognition, and finer-grain hearing to the M0 audio-token lane. Probe
   verdict recalibrated to layer-honest criteria.

VALIDATED RESULT: cognitive response PASS -- operator speech raised gate
escalation rate at both silence->speak transitions (0.53->1.90 esc/s at the
second), through the full circuit mic -> intake -> Whisper -> fused percept ->
salience -> deliberation. First confirmed instance of a human voice reaching
and moving the agent's cognition. Pooled-pathway novelty response measured
blunt as predicted (the M0.2/M0.3 before-baseline, archived in
reports/audioprobe_20260706_200702).
r frame, and recursive filters with per-frame zero state ring at
  every boundary, which would fail the click-free acceptance this milestone
  exists to meet. The fundamental's phase is corrected to land on integer
  cycles at frame edges and the noise is edge-faded, so boundaries are
  click-free by construction. Same acoustics, provable seams.
- **All-zero voice params are near-silent, not digitally silent.** The cubic
  energy map puts tanh(0)=0 at ~12.5% of max amplitude (a faint hum): exact
  zero would give babble learning a dead plateau at the origin; energy=-1 is
  exact digital silence. The zero-init head therefore whispers breath, it does
  not speak.
- **M2.2's world-side mixing bus (caregiver utterances + distance
  attenuation) is NOT in this change**; the loopback path implemented here is
  the bus's first client (`mix_in`), and the scenario-scheduled utterance
  mixing lands with M1.1/M1.2.
- The intake defaults `mic` and voice defaults ON (owner decision 2026-07-04),
  not the `off` written in the M0.5 line above; parity is preserved by the
  suite pinning both off and by zero-init/no-module discipline.
