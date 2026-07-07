# Audio-cognition probe (transition-scored)
samples=321  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.00->0.00  novelty_peak 0.00026836->0.0001704857142857143  esc/s 0.000->0.000
  silence_2->speak_2: attach/s 0.00->0.00  novelty_peak 3.7833333333333336e-05->0.00020927027027027028  esc/s 0.000->0.000
pc_loss mean: speak=1.1246476752890482 silent-tails=1.132312098616048 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] silence gate (self-masked): attach-rate deltas at silence->speak transitions: [0.0, 0.0] (speech must raise the attach rate; the self-mask keeps the hum from holding the gate open)
[FAIL] auditory salience: novelty-peak deltas at transitions: [-9.787428571428569e-05, 0.00017143693693693696]
[FAIL] cognitive response: escalation-rate deltas at transitions: [0.0, 0.0] (at least one transition should make your voice worth thinking about)
AUDIO_PROBE: FAIL
