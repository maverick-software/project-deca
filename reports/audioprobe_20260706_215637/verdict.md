# Audio-cognition probe (transition-scored)
samples=141  plan=['settle_0', 'silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
  silence_1->speak_1: attach/s 1.16->2.12  novelty_peak 0.003276818181818182->0.005421307692307692  esc/s 3.486->0.584
  silence_2->speak_2: attach/s 1.45->2.03  novelty_peak 0.0025477272727272727->0.004378533333333334  esc/s 0.552->0.811
pc_loss mean: speak=1.093055859208107 silent-tails=0.9722907183801427 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[PASS] intake liveness + gate economy: attached~309 skipped~140 across the session (the ear delivers sound AND skips dead air; both paths exercised)
[PASS] cognitive response: escalation-rate deltas at transitions: [-2.9024, 0.2581] (speech makes the agent deliberate -- the circuit's core assertion)
[INFO] auditory salience (pooled-pathway baseline for the M0 token lane): novelty-peak deltas at transitions: [0.0021444895104895103, 0.001830806060606061]
[INFO] attach-rate deltas at transitions: [0.96, 0.58]
AUDIO_PROBE: PASS
