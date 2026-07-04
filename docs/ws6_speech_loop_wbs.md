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

## M1 — Receptive naming (desk-runnable)

**M1.1 Speech scenario format + generator.** Extend the binding scenario: `utterances` schedule (label ↔ entity-salience windows, balanced novel-label controls per the WS5 leakage checklist) + pre-rendered wav bank; `gen_speech_scenario.py`.
**M1.2 Client + seam.** Synthetic client attaches scheduled wavs to `obs.audio` (mic-live mode optional ⚙); audio familiarity key written to episodic storage (versioned layout amendment).
**M1.3 ⚙ P1 word-form familiarity probe.** Habituation/novelty verdict over the audio key; **flags-off ablation must fail the discrimination**.
**M1.4 ⚙ P2 receptive protoword probe.** Label alone boosts the named slot vs controls; verdict script in the gate-probe mold.
*Accept (M1):* P1 and P2 verdicts archived; ablation direction confirmed.

## M2 — Mouth: hardware and silence

**M2.1 `voice_u` head** (flag `DECADIC_VOICE`): zero-init (silence at init), emitted in the action dict as `"voice"`; checkpoint compat via versioned bundle key.
**M2.2 Vocal tract.** Body/world-side formant synthesizer, interpolation between cycle frames, mixed into `obs.audio` with attenuation; desk rig = speaker+mic path documented.
**M2.3 Mouth in the body schema.** Articulator positions as proprio fields; body-map registration.
*Accept:* suite green both flag states; a live run ⚙ shows rendered audio re-entering observation (loopback smoke, `run_binding_smoke.ps1` pattern).

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
M0.4 -> M1.1 -> M1.2 -> M1.3 -> M1.4
M0.4 -> M2.1 -> M2.2 -> M2.3 -> M3.1 -> M3.2 -> M3.3 -> M4.1 -> M4.2 -> M5 -> M6
M1 and M2+ parallelize after the M0.4 gate.
```
