# Audio-cognition probe (transition-scored)
samples=282  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.00->0.00  novelty_peak 0.00015044->0.00022123076923076925  esc/s 0.601->4.232
  silence_2->speak_2: attach/s 0.00->0.00  novelty_peak 0.00012793333333333332->0.00013462162162162163  esc/s 1.108->1.208
pc_loss mean: speak=0.9207630285194942 silent-tails=1.0238366837010664 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] intake liveness + gate economy: attached~0 skipped~1326 across the session (the ear delivers sound AND skips dead air; both paths exercised)
[PASS] cognitive response: escalation-rate deltas at transitions: [3.6312, 0.1004] (speech makes the agent deliberate -- the circuit's core assertion)
[INFO] auditory salience (pooled-pathway baseline for the M0 token lane): novelty-peak deltas at transitions: [7.079076923076924e-05, 6.688288288288316e-06]
[INFO] attach-rate deltas at transitions: [0.0, 0.0]
AUDIO_PROBE: FAIL
