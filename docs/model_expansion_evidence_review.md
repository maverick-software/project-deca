# Capability Expansion — Evidence Review

Each of the 13 proposals from `model_expansion_integration.md` researched against the
ML/RL/robotics literature: would it benefit the model, how, and what breaks.
Sources were gathered by five parallel research passes and the load-bearing claims
were adversarially fact-checked (14/16 verified against primary sources, 2 partly
verified, 0 refuted). Verdict scale: **BUILD** (strong evidence of benefit, design
matches what worked), **BUILD WITH GUARDRAILS**, **TEST FIRST** (plausible,
unproven — ship behind an ablation), **DEFER** (evidence says wrong time or wrong
reason).

## Scorecard

| # | Proposal | Verdict | Evidence strength |
|---|----------|---------|-------------------|
| 1 | Spatial state + model-based planning | **BUILD** | Strong (3 independent lines) |
| 2 | Symbolic abstraction (VQ codebook) | **SPLIT**: discrete bottleneck BUILD; language loop DEFER | Strong / absent |
| 3 | Multi-channel learning control | **BUILD WITH GUARDRAILS** (clamp the γ channel) | Strong per channel |
| 4 | Input routing gate | **BUILD** | Strong |
| 5 | Motor corrector + phase timing | **BUILD** (best-supported of all 13) | Strong |
| 6 | Interoceptive embedding → affect | **TEST FIRST** | Weak (principle yes, architecture unproven) |
| 7 | Threat prediction + valence replay | **SPLIT**: avoidance bearing BUILD; replay weighting blend-only | Strong / moderate |
| 8 | Scheduled rest consolidation | **BUILD WITH GUARDRAILS** | Strong (structure), unvalidated (trigger) |
| 9 | Other-agent modeling | **DEFER** until adaptive others exist | Strong when social, zero when solo |
| 10 | Goal hierarchy + veto | **SPLIT**: veto BUILD; hierarchy PROVE-IT | Weak (hierarchy) / strong (veto) |
| 11 | Cached vs deliberate dual control | **BUILD** | Strong |
| 12 | Salience view control | **SPLIT**: WM gating BUILD; view command guarded | Strong / fragile |
| 13 | Global phase clock | **DEFER** (measurement value only) | Absent for task benefit |

---

## 1. Spatial state estimation + model-based planning — BUILD

All three sub-mechanisms have direct quantitative support, and our integration is
*more conservative* than the published versions.

**Self-localization + positional code.** Banino et al., *Nature* 2018 (verified):
an agent given a learned path-integration embedding navigated better than
raw-input/landmark-code baselines and took novel shortcuts — the exact
dead-reckoning + learned positional embedding we propose. Known cost: pose
integration drifts; our landmark re-sighting correction is the standard remedy.

**Waypoint planning over a coarse adjacency graph.** Savinov et al. (topological
memory, ICLR 2018) and Eysenbach et al. (*Search on the Replay Buffer*, NeurIPS
2019, verified): graph search feeding *next-waypoint* subgoals to a local policy
solves 100+-step sparse-reward navigation that flat goal-conditioned RL fails.
Our "bearing points to the next waypoint" substitution is exactly how both
systems drive the low-level policy. **The documented risk is graph hygiene, not
the planner**: aliased locations create spurious edges that let plans "warp"
across the map; unweighted edges degrade path quality. Mitigation: weight edges
with a learned distance, prune aggressively.

**Online rollout action selection.** TD-MPC2 (ICLR 2024, verified) is our exact
loop — sample K short action sequences, roll a learned latent model, score by a
terminal learned value — and beats SAC and DreamerV3 across 104 continuous-control
tasks with one hyperparameter set. The load-bearing tricks are ones we already
have: **short horizons truncated by a value function** (our successor-features
value is the terminal score). Compounding model error is the known failure; it is
controlled precisely by short rollouts + value truncation. Two design points in
our favor vs the literature norm: we plan only on the gated deliberate path
(cheaper), and we *bias* motor output rather than override it (robust to model
error).

**Benefit for Deca now:** this is the direct consumer of the WS-FORAGE spine —
it converts "I remember where water is" into "I can get there around obstacles,"
and it finally makes `imagination.py` earn its keep online. Highest-leverage item
on the list.

## 2. Symbolic abstraction — SPLIT

**The discrete bottleneck: BUILD.** Discrete latents demonstrably help world
models and generalization: DreamerV2's categorical latents beat Gaussian on 42/55
Atari tasks (verified); IRIS (VQ tokens + autoregressive next-token dynamics —
architecturally our proposal) reached superhuman on 10/26 Atari-100k games from
2 hours of experience; discrete-valued inter-module communication improves
out-of-distribution generalization (Liu et al., NeurIPS 2021). Counter-evidence
we must respect: TD-MPC2's *continuous* latents beat discrete-latent DreamerV3 on
continuous control — for a MuJoCo body, discreteness is beneficial for
*abstraction/memory*, not required for *control*. Implementation note: classic VQ
suffers codebook collapse and straight-through-estimator instability; **FSQ**
(finite scalar quantization, ICLR 2024) removes the whole failure class with
~100% code utilization and no auxiliary losses — use FSQ, not a growing learned
codebook (open-endedly growing codebooks have no published RL result).

**The language loop (recurrent token feedback + audio expression): DEFER.** Every
demonstrated symbol-grounding result requires a *listener* and communicative
pressure (Lazaridou et al.; Mordatch & Abbeel). Without one, emergent codes are
anti-efficient absent a length cost (Chaabouni, NeurIPS 2019, verified) and
compositionality is uncorrelated with generalization (Chaabouni, ACL 2020,
verified). Non-LLM "inner speech helps" evidence is a single small result
(Granato et al. 2020). For a solo forager this is machinery without payoff;
becomes live the moment proposal 9's precondition (other agents) is met.

## 3. Multi-channel learning control — BUILD WITH GUARDRAILS

Each channel is independently validated; the vector as a whole mirrors a real
division of labor (Yu & Dayan 2005: expected uncertainty vs surprise as distinct
control signals scaling learning rate and attention).

- **Reward channel** = current behavior; Backpropamine (ICLR 2019, verified)
  is the validated baseline for a learned scalar gating local plasticity.
- **Expected-uncertainty → learning rate**: supported, but Piray & Daw (*Nat.
  Comm.* 2021) prove the critical subtlety: learning rate must rise with
  *volatility* and **fall** with irreducible noise. A naive "surprise → boost LR"
  chases aleatoric noise. Our channel must separate the two (e.g., compare
  `pc_slope_ema` trend vs `pc_ema` variance).
- **Surprise → transient boost + escalation**: same caveat; route through the
  volatility test.
- **Horizon channel (viability → γ): the risky one.** Meta-Gradient RL (NeurIPS
  2018, verified) shows online-adapted γ helps at scale, but later work documents
  catastrophic instability from gradient spikes on lucky high-reward streaks —
  and "thriving → longer horizon → bigger value targets" is a positive-feedback
  loop of exactly that shape. **Clamp γ to a narrow band, rate-limit changes,
  and log every move.**

Integration fit: init-maps-to-current-scalar is exactly how the literature
bootstraps these (verified pattern). Cheapest sharpening of all learning in the
system; one seam.

## 4. Input routing gate — BUILD

Mott et al. (NeurIPS 2019, verified): top-down (goal-conditioned) attention gave
a large gain over an equivalent bottom-up-only mechanism — direct support for
combining salience with the goal vector. Object/slot-masked inputs match
raw-input performance while gaining large robustness to distractors (OCCAM 2025;
SOLD 2024; learned masks on the Distracting Control Suite). Known failure: a gate
trained on current-task relevance can blind the agent to *newly* relevant
percepts. Fix is already in our architecture: **re-open the gate on
prediction-error spikes** (proposal 3's surprise channel), and identity-init so
it starts permissive. Benefit: cheaper deep-network cycles and distraction
robustness as scenes get busier.

## 5. Motor corrector + phase timing — BUILD (strongest evidence of all 13)

Three independent validated lines converge on exactly our design:

- **Residual/additive correction**: Silver et al. 2018 + Johannink et al., ICRA
  2019 (verified) — an additive learned correction on a base controller gives
  order-of-magnitude sample-efficiency gains vs learning from scratch; zero-init
  residual is the documented practice.
- **Error-driven feedforward correction**: feedback-error learning (Kawato) —
  training a corrector head on the motor prediction-error signal is a
  decades-validated control scheme.
- **Per-actuator phase timing**: CPG-RL (Bellegarda & Ijspeert 2022, verified) —
  policy modulating oscillator setpoints achieved sim-to-real quadruped walking
  robust to a 115%-body-mass load; Siekmann et al. 2021 (verified) learned all
  common bipedal gaits on Cassie from a periodic phase clock.

One guardrail: train the corrector on prediction error as a **supervised
target**, never as a reward it optimizes — otherwise it learns to exploit
forward-model inaccuracies. Watch for phase machinery constraining aperiodic
recovery motions (fall recovery). This is the unlock for locomotion, which is
currently our binding constraint.

## 6. Interoceptive embedding → affect — TEST FIRST

The honest verdict: the *principle* is solid, the *architecture* is unproven.
Grounding reward/value in internal-state distance-from-setpoint is formalized and
validated (Keramati & Gutkin, NeurIPS 2011 / eLife 2014 — drive-reduction reward
matches behavior and reward-signal data). But **no published controlled ablation
shows that a separate learned body-state embedding feeding the affect head beats
deriving affect from the main latent** (adversarially checked; no
counter-evidence found, but it's an absence-of-evidence negative). Risk is
redundancy: if the latent already encodes reservoirs, this adds parameters
without signal. It's nearly free to build (zero-init, signals already exist), so:
build it flag-gated, and **require an A/B on affect-prediction quality and
viability maintenance before declaring it kept.**

## 7. Threat prediction + valence-weighted replay — SPLIT

**Avoidance bearing: BUILD.** Two-factor avoidance (learned cue→pain association
driving an instrumental avoidance response) is a standard, well-modeled
computational template (Maia 2010 actor-critic formulation). It reuses M3/M4/M5
with a sign flip — near-zero cost. **One mandatory guardrail from the
literature: persistent avoidance blocks extinction.** If the agent always steers
away, it never learns the cue is no longer dangerous, and the threat belief
becomes permanent. Add decay/occasional re-test of `predicts_pain` beliefs (our
belief-confidence machinery already supports decay).

**|valence|-weighted replay: blend, don't replace.** Priority-replay variants
using reward-magnitude signals help, but the strongest results *fuse* value
magnitude with TD-error rather than replacing it, and any non-uniform sampling
needs importance-sampling correction or it biases the learned value function.
Implement as `salience × (α·|valence| + (1−α)·|td_error|)` with IS correction.

## 8. Scheduled rest consolidation — BUILD WITH GUARDRAILS

The two-phase structure is directly validated. Wake-Sleep Consolidated Learning
(Sorrenti et al. 2024, verified with minor number caveat): a wake phase + a
two-stage offline phase (replay of real episodes, then generative rollouts)
improved continual-learning accuracy ~11-12pp on CIFAR-10 and — the striking
part — **flipped forward transfer from negative to positive, and the ablation
shows the generative second stage is what does it.** Real-replay-then-generative
is not redundant; each stage contributes differently. Offline low-activity
consolidation phases also reduce forgetting in the local-plasticity setting
(Tadros et al., *Nat. Comm.* 2022) — notable because our plasticity is local too.

Two caveats. (a) **The trigger is our research bet**: published systems trigger
on idle time or task boundaries; a prediction-error-load accumulator has no
head-to-head validation. A/B it against always-on. (b) **Value drift**: the
robot-sleep literature names the exact risk — after long active periods, replayed
value estimates extrapolate beyond the current policy's distribution. Bound rest
onset by *time-since-last-rest*, not load alone. Downtime is a real cost once
threats exist; fine in the current arena.

## 9. Other-agent modeling — DEFER (right design, wrong time)

The architecture is validated: a single meta-learned observer models other agents
from behavior alone and passes false-belief tests (ToMnet, ICML 2018, verified) —
supporting our reuse-the-self-model plan; imitation from observation without
action labels works (BCO 2018, verified), supporting generalizing the teacher
path — though later analyses show observation-only imitation carries a provably
higher sample cost. **But every demonstrated benefit is conditional on adaptive
others being present.** No paper shows agent-modeling helping a solo agent; for
us today it is parameters, latency, and graph writes for zero policy benefit,
plus spurious beliefs about scripted props. Gate activation on perceiving an
adaptive agent-entity. The observation-imitation path is the one piece worth
wiring early (useful even with scripted demonstrators).

## 10. Goal hierarchy + veto — SPLIT

**The veto head: BUILD, with one design change.** Learned safety layers are a
validated pattern (Dalal et al. 2018; conservative safety critics, ICLR 2021),
and zero-init (no suppression at birth) matches how they're bootstrapped. But the
literature is clear that **minimal correction beats multiplicative zeroing**:
tight suppression cripples task performance ("over-conservatism"), and a veto
driven by a wrong forward model can lock the agent into a do-nothing attractor.
Implement as smallest-perturbation attenuation, uncertainty-weighted, never a
hard zero.

**The goal-stack hierarchy: PROVE-IT.** The skeptical anchor is Nachum et al.
2019 (verified): *most measured benefit of hierarchical RL is attributable to
improved exploration*, reproducible by a flat policy with an exploration bonus;
option-style hierarchies also degenerate to one-step primitives without
regularization. Our goal conditioning is already continuous and arbitrated —
the burden of proof is on the stack. Acceptance test before keeping it: beat
flat-policy-plus-exploration-bonus on a multi-step task (e.g., detour-then-forage).

## 11. Cached vs deliberate dual control — BUILD

Strong match on all three components. Fast head amortizing a slow deliberate
process is the Expert Iteration / AlphaZero pattern (Anthony et al., NeurIPS
2017); policy distillation retains teacher performance at up to 15× compression
and supports *online* distillation continually tracking an evolving teacher
(Rusu et al., ICLR 2016, verified); uncertainty-based arbitration between
cached and deliberate controllers is the classic validated account of dual
control (Daw et al. 2005) — and our gate is precisely that arbitrator. The two
documented failure modes both have fixes already wired into other proposals:
**stale habits** (distilled head degrades when the world changes — a named,
studied failure) are caught by proposal 3's surprise channel forcing escalation;
**arbitration thrash** is caught by the existing Type-2 refractory/hysteresis.
Requirements: distill from the deliberate teacher's outputs (never the cached
head's own), continually. Payoff: the deliberate path's compute is reserved for
novelty — directly attacks the 55%-perseveration class of problem.

## 12. Salience view control — SPLIT

**Working-memory admission weighting: BUILD.** Adding goal relevance to
bottom-up salience significantly improves relevance prediction (Tanner & Itti
2019); foveated/masked perception matches full-frame performance with better
distractor robustness and lower compute. Soft, zero-init — degrades gracefully.

**The view-orientation motor command: BUILD, but guarded.** Here the honest read
of the evidence matters: the headline gaze-control numbers (e.g., 95.5% vs 85.9%
task success) come from **imitation of human gaze**, which we don't have. The
pure-RL evidence (SUGARL, NeurIPS 2023, verified) shows joint gaze+motor learning
is feasible but hard: non-stationary observations destabilize TD targets, and it
only converged with an added **intrinsic sensorimotor reward** (reward for views
that improve prediction). Prescription: make the view command **soft/continuous**
(never a discrete hard-attention action), zero-init, and give it an intrinsic
term — reward views that reduce prediction error on the current goal target. This
still fixes the "camera at the sky" failure; it just doesn't pretend the
imitation-learning numbers transfer.

## 13. Global phase clock — DEFER

The oscillation-adjacent ANN successes (Kuramoto-style binding, ICLR 2025,
verified; rotating features, NeurIPS 2023) are all **perception/binding
representation** results on static benchmarks — none uses a global phase clock to
schedule *when* an acting agent binds or deliberates, and no task benefit for
that use exists in the literature (adversarially checked; no counter-evidence).
Meanwhile agents demonstrably *learn* internal timing when tasks demand it,
suggesting a hand-imposed clock may be redundant. Keep as instrumentation for
integration-measurement work only; revisit if percept binding itself becomes the
bottleneck (in which case adopt phase as a *representation*, not a scheduler).

---

## Revised build order (evidence-weighted)

The original doc's ordering survives contact with the literature almost intact,
with two changes: the veto and dual control move up (stronger evidence than
expected), the interoceptive head moves from "nearly free win" to "free but
prove it."

1. **Spatial state + planning (1)** — strongest capability unlock, direct
   consumer of the foraging spine, three verified evidence lines.
2. **Multi-channel learning control (3)** — one seam, sharpens everything;
   clamp γ, separate volatility from noise.
3. **Motor corrector + phase timing (5)** — best-supported proposal on the
   list; unblocks locomotion.
4. **Dual control (11)** — validated pattern; its failure modes are already
   solved by 3 + the existing refractory.
5. **Avoidance bearing (7a)** — sign-flip reuse, add belief decay for
   extinction. Veto head (10a) rides the same forward-model read.
6. **Routing gate (4) + WM salience weighting (12a)** — robustness + compute,
   both low-risk.
7. **Rest consolidation (8)** — validated structure; A/B our load trigger.
8. **Interoceptive head (6), valence replay blend (7b), view command (12b)** —
   flag-gated, each behind an explicit A/B.
9. **Discrete bottleneck via FSQ (2a)** — abstraction/memory payoff; the
   language loop (2b) waits for other agents.
10. **Deferred:** other-agent modeling (9) until adaptive others exist; goal
    stack (10b) until it beats flat+exploration; phase clock (13) indefinitely.

## Key sources

Banino et al., Nature 2018 — https://www.nature.com/articles/s41586-018-0102-6
· Eysenbach et al., SoRB, NeurIPS 2019 — https://arxiv.org/abs/1906.05253
· Savinov et al., ICLR 2018 — https://arxiv.org/abs/1803.00653
· Hansen et al., TD-MPC2, ICLR 2024 — https://arxiv.org/abs/2310.16828
· Hafner et al., DreamerV2, ICLR 2021 — https://arxiv.org/abs/2010.02193
· Micheli et al., IRIS, ICLR 2023 — https://arxiv.org/abs/2209.00588
· Liu et al., Discrete-Valued Neural Communication, NeurIPS 2021 — https://arxiv.org/abs/2107.02367
· Mentzer et al., FSQ, ICLR 2024 — https://arxiv.org/abs/2309.15505
· Chaabouni et al., NeurIPS 2019 — https://arxiv.org/abs/1905.12561; ACL 2020 — https://aclanthology.org/2020.acl-main.407/
· Miconi et al., Backpropamine, ICLR 2019 — https://openreview.net/pdf?id=r1lrAiA5Ym
· Xu et al., Meta-Gradient RL, NeurIPS 2018 — https://arxiv.org/abs/1805.09801
· Piray & Daw, Nat. Comm. 2021 — https://www.nature.com/articles/s41467-021-26731-9
· Yu & Dayan, 2005 — https://www.sciencedirect.com/science/article/pii/S0896627305003624
· Nachum et al., 2019 — https://arxiv.org/abs/1909.10618
· Dalal et al., 2018 — https://arxiv.org/abs/1801.08757
· Anthony et al., NeurIPS 2017 — https://arxiv.org/abs/1705.08439
· Rusu et al., Policy Distillation, ICLR 2016 — https://arxiv.org/abs/1511.06295
· Daw et al., 2005 — https://www.nature.com/articles/nn1560
· Mott et al., NeurIPS 2019 — https://arxiv.org/abs/1906.02500
· Silver et al., 2018 — https://arxiv.org/abs/1812.06298; Johannink et al., ICRA 2019 — https://arxiv.org/abs/1812.03201
· Bellegarda & Ijspeert, CPG-RL 2022 — https://arxiv.org/abs/2211.00458
· Siekmann et al., ICRA 2021 — https://arxiv.org/abs/2011.01387
· Keramati & Gutkin, eLife 2014 — https://elifesciences.org/articles/04811
· Maia, 2010 — https://tiagomaia.org/wp-content/uploads/2015/02/maia_2010_lb.pdf
· Schaul et al., PER 2016 — https://arxiv.org/abs/1511.05952
· Sorrenti et al., WSCL 2024 — https://arxiv.org/abs/2401.08623
· Shin et al., NeurIPS 2017 — https://arxiv.org/abs/1705.08690
· Tadros et al., Nat. Comm. 2022 — https://www.nature.com/articles/s41467-022-34938-7
· Rabinowitz et al., ToMnet, ICML 2018 — https://arxiv.org/abs/1802.07740
· Torabi et al., BCO 2018 — https://arxiv.org/abs/1805.01954
· Shang et al., SUGARL, NeurIPS 2023 — https://arxiv.org/abs/2306.00975
· Tanner & Itti, J. Vision 2019 — https://jov.arvojournals.org/article.aspx?articleid=2720949
· Miyato et al., AKOrN, ICLR 2025 — https://arxiv.org/abs/2410.13821
