# Audio-cognition probe
samples=117  plan=['silence_1', 'speak_1', 'silence_2', 'speak_2', 'silence_3']
pc_loss mean: speak=1.095668209010157 silent=1.093233582648364 (informational -- speech is unpredicted input; a rise then re-settle is healthy)

[FAIL] silence gate: chunks attached: speak=95 silent=275 (speech should dominate; ambient noise may leak a little)
[FAIL] auditory salience: novelty peak: speak=0.003713931034482759 silent=0.03869717045454545
[FAIL] cognitive response: escalations during speech=0 vs silence=12 (your voice should be worth thinking about at least once)
AUDIO_PROBE: FAIL
