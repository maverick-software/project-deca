# Audio-cognition probe (transition-scored)
samples=161  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 0.97->1.54  novelty_peak 0.002327714285714286->0.009405142857142858  esc/s 1.381->1.010
  silence_2->speak_2: attach/s 1.66->2.00  novelty_peak 0.004260454545454546->0.0046735  esc/s 1.106->0.410
pc_loss mean: speak=0.9622444595609393 silent-tails=0.7112492503121842 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[PASS] intake liveness + gate economy: attached~268 skipped~336 across the session (the ear delivers sound AND skips dead air; both paths exercised)
[FAIL] cognitive response: escalation-rate deltas at transitions: [-0.3717, -0.6957] (speech makes the agent deliberate -- the circuit's core assertion)
[INFO] auditory salience (pooled-pathway baseline for the M0 token lane): novelty-peak deltas at transitions: [0.007077428571428573, 0.00041304545454545413]
[INFO] attach-rate deltas at transitions: [0.57, 0.34]
AUDIO_PROBE: FAIL
