# Audio-cognition probe (transition-scored)
samples=337  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.00->0.00  novelty_peak 6.611111111111112e-05->0.00011475675675675676  esc/s 0.000->0.000
  silence_2->speak_2: attach/s 0.00->0.00  novelty_peak 3.6892857142857144e-05->1.4756756756756757e-05  esc/s 0.000->0.000
pc_loss mean: speak=1.1214532513876219 silent-tails=1.1307125263903515 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] silence gate (self-masked): attach-rate deltas at silence->speak transitions: [0.0, 0.0] (speech must raise the attach rate; the self-mask keeps the hum from holding the gate open)
[FAIL] auditory salience: novelty-peak deltas at transitions: [4.8645645645645637e-05, -2.2136100386100385e-05]
[FAIL] cognitive response: escalation-rate deltas at transitions: [0.0, 0.0] (at least one transition should make your voice worth thinking about)
AUDIO_PROBE: FAIL
