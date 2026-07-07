# Audio-cognition probe (transition-scored)
samples=337  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.00->0.00  novelty_peak 1.1111111111111112e-05->6.755263157894736e-05  esc/s 0.000->0.000
  silence_2->speak_2: attach/s 0.00->0.00  novelty_peak 0.000176->0.0002923888888888889  esc/s 0.000->0.000
pc_loss mean: speak=1.113179946267927 silent-tails=1.124779595107567 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] silence gate (self-masked): attach-rate deltas at silence->speak transitions: [0.0, 0.0] (speech must raise the attach rate; the self-mask keeps the hum from holding the gate open)
[PASS] auditory salience: novelty-peak deltas at transitions: [5.644152046783625e-05, 0.00011638888888888891]
[FAIL] cognitive response: escalation-rate deltas at transitions: [0.0, 0.0] (at least one transition should make your voice worth thinking about)
AUDIO_PROBE: FAIL
