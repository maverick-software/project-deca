# Audio-cognition probe (transition-scored)
samples=288  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.00->0.00  novelty_peak 8.942307692307692e-05->0.0001506296296296296  esc/s 0.269->0.629
  silence_2->speak_2: attach/s 0.00->0.00  novelty_peak 9.611764705882353e-05->0.0001284054054054054  esc/s 0.554->0.615
pc_loss mean: speak=1.006358795799315 silent-tails=0.9098070161683219 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] intake liveness + gate economy: attached~0 skipped~1326 across the session (the ear delivers sound AND skips dead air; both paths exercised)
[PASS] cognitive response: escalation-rate deltas at transitions: [0.3599, 0.0611] (speech makes the agent deliberate -- the circuit's core assertion)
[INFO] auditory salience (pooled-pathway baseline for the M0 token lane): novelty-peak deltas at transitions: [6.120655270655268e-05, 3.228775834658187e-05]
[INFO] attach-rate deltas at transitions: [0.0, 0.0]
AUDIO_PROBE: FAIL
