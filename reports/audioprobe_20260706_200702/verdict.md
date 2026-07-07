# Audio-cognition probe (transition-scored)
samples=135  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.84->1.02  novelty_peak 0.0032708181818181816->0.001958142857142857  esc/s 0.561->0.646
  silence_2->speak_2: attach/s 1.13->1.37  novelty_peak 0.00288825->0.0099614  esc/s 0.533->1.896
pc_loss mean: speak=0.9565419760243646 silent-tails=0.9564566629273551 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] silence gate (self-masked): attach-rate deltas at silence->speak transitions: [0.18, 0.24] (speech must raise the attach rate; the self-mask keeps the hum from holding the gate open)
[FAIL] auditory salience: novelty-peak deltas at transitions: [-0.0013126753246753245, 0.00707315]
[PASS] cognitive response: escalation-rate deltas at transitions: [0.0852, 1.3624] (at least one transition should make your voice worth thinking about)
AUDIO_PROBE: FAIL
