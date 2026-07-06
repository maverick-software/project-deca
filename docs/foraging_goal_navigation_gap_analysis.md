# Gap Analysis — Goal-Directed Foraging & Memory-Guided Navigation

**Question.** A thirsty agent, standing where a resource is out of view, does not
recall the resource, orient/search for it, and navigate to it. Why not — and is
anything *vital* missing, or is this a matter of experience, motor development,
and tuning? Includes a feasibility study of **extending the credit horizon from
seconds to minutes** without destabilizing the system.

**Verdict up front.** The core reinforcement machinery for this behavior is
present and unusually complete (need-modulated reward, successor-feature value,
temporal credit assignment, actor-critic-style policy shaping, neuromodulated
plasticity, a symbolic goal latch, and a motor-learning curriculum). The agent
is **not** missing a "mind" component (GWT/IIT are about access and integration,
not action learning, and the workspace/gate side exists). What blocks the
behavior is, in priority order: (A) a curriculum that starves the value machinery
of any *completed* approach→relief trajectory; (B) a credit horizon (~8 s) far
shorter than a real pursuit; (C) one genuine architectural gap — value is
**cue-driven, not memory-driven**, so the agent can approach what it *sees* but
cannot navigate to what it *remembers*; (D) immature locomotion; (E) uncontrolled
gaze. (B) is fixable and (C) is the interesting frontier; they must move together.

---

## 1. Target behavior decomposed

"Go get the water" requires this loop to close and be reinforced:

1. **Perceive** the resource and distinguish it (food vs water vs other).
2. **Feel the need** (interoception: hydration deficit).
3. **Associate** the percept with relief (Pavlovian cue → outcome).
4. **Value it *now*** — the cue's pull scales with the current deficit
   (incentive salience).
5. **Hold the target** when it leaves view (working/spatial memory as a goal).
6. **Act** — produce locomotion that reduces distance to the target.
7. **Get relief**, and **assign credit** back over the *whole* approach so the
   moves that led there are reinforced (temporal credit assignment).

The agent's failure is not at one point but at the seams between 4↔5↔6 and in the
time-span of 7.

---

## 2. What the system already has (inventory, with code)

| Capability | Where | State |
|---|---|---|
| Perception (frozen CLIP/Whisper) → percept, scene entities | `nn/frozen_encoders.py`, `perception/organ.py` | present |
| Interoception / homeostatic reservoirs + viability | `state/viability.py`, homeostasis | present |
| Working memory slots (Baddeley-style, cap 12) | `state/working_memory.py` (`DEFAULT_WORKING_MEMORY_SLOTS=12`) | present, decays |
| Object permanence (brief slot persistence) | WS5 relational binding | present, short |
| LTM graph: entities + spatial relations, positions | `memory/*graph.py` (`scene_near/above/left_of/...`) | present |
| Pavlovian association (`predicts_energy_relief`) | property beliefs | forming |
| **Reward = deficit-gated drive reduction (incentive salience)** | `neural_pipeline.py:1104` `w_gated=(w_pref_i*deficit_i)` | present |
| **Successor features** `psi` = discounted future reservoir-change | `nn/successor_features.py` (zero-init) | present |
| **TD(λ) SF training** over lived episodes | `consolidation/consolidator.py:95-109`; `returns.py` | present |
| Imagined SF rollouts | `consolidator.py:216-223`, `consolidation/imagination.py` | present |
| Hindsight relabeling (learn from partial success) | `episodes.py:128-157`, `HER_RELABEL_K` | present |
| **Actor-critic-style policy shaping** (maximize value) | `neural_pipeline.py:1100-1108` `loss -= sf_value_w*value` | present, ramped |
| Neuromodulated (pleasure−pain) Hebbian plasticity | `neural_pipeline.py:402` | present |
| **Symbolic goal latch** (episode boundary) | `state/goal_lifecycle.py` `GoalState` | present |
| Motor-learning curriculum (teacher, babble, braces) | `training/skills.py` (`stand_teacher`, Skill Dojo) | present |
| Global-workspace gate / ignition | `cycle/attention_gate.py`, `gwt_capacity()` | present |
| Neural pipeline is what runs the agent (not the stub stages) | `agents/runtime.py:2894` (`run_neural_cycle`) | — |

**Reading of the inventory:** items 1–7 of the target loop each have a mechanism.
The reward is even need-modulated, which most hand-built agents lack. The problem
is not absence of machinery.

---

## 3. Gaps, in priority order

### Gap A — Curriculum starvation (dominant, near-term) — the value machinery has never seen a completed approach

The SF value only learns "approaching water leads to relief" from **trajectories
in which the agent actually approached and got relief, inside the credit
horizon**. Today:

- *Give directly* → relief arrives with **no approach** to credit.
- *Place nearby (far)* → an approach that **never completes** (immature
  locomotion), and — even if it did — a resource more than ~8 s away sits
  **beyond the credit horizon** (Gap B), so terminal relief cannot propagate to
  the first steps.

So the agent has never once experienced a short, completed approach→relief. This
is a **data problem, not an architecture problem**.

**Fix (owner's instinct, correct):** a graded placement ladder that guarantees a
*completable* success and shrinks it toward autonomy:

- **Place within reach** — resource ~0.5–1 m, satisfiable by a lean/reach, **no
  locomotion required**. Closes the value loop before crawling exists.
- **Place nearby** — a few steps, short locomotion **inside the horizon**.
- **Place far / give directly** — retained for association upkeep and stress
  tests, not for teaching approach.

This is the single highest-leverage change and needs no new machinery — only a new
placement distance (the `give_near` seam already relocates a prop; a "within
reach" variant is a distance parameter).

### Gap B — Credit horizon too short (the explicit ask) — feasible to extend; see §4

`SF_GAMMA=0.97` ⇒ effective horizon `1/(1−γ) ≈ 33` steps ≈ **~8 s** at ~4
cycles/s. A real pursuit is tens of seconds to minutes. Terminal relief cannot
reach the postural/locomotor commands that preceded it by more than ~8 s.
**Feasible to extend to minutes** — the plumbing already supports it (§4).

### Gap C — Cue-driven, not memory-driven value (the deepest gap) — approaches what it *sees*, cannot navigate to what it *remembers*

The value is `successor_predict(z5, motor_u)` on the **current** latent
(`neural_pipeline.py:1103`). `z5` absorbs working memory and the memory-context
summary, so it is *tinted* by memory — but the influence is **associative, not a
persistent spatial goal**. Consequences:

- Incentive salience operates on a resource **in the current percept**. Remove it
  from view (agent looks at sky) and there is no cue to value, so no pull.
- The `GoalState` latch **does** persist "pursuing water" across the whole
  pursuit — but it is used only to define the **credit-assignment episode**, not
  fed to the **policy**. Nothing converts "water existed at bearing θ" into a
  motor goal.
- Therefore "remember it, look around, go to it" is unsupported: the agent is
  reactive to the visible cue, not directed by a remembered target.

This is the one place a **new architectural pathway** is required, and it is the
necessary partner to Gap B: a minutes-long *value* is inert unless the agent also
**holds the target over those minutes**. See §5.

### Gap D — Motor competence (prerequisite) — can't crawl yet

The value gradient can only reinforce approach motion that *occurs*. Locomotion is
immature (ragdoll behavior). The scaffolding exists (`training/skills.py`:
teacher-guided stand → perturbation recovery → reduced assistance → autonomy, with
`motor_babble_sigma` and braces), so this is **curriculum progression, not missing
machinery**. Gap A's "within reach" placement also lowers the motor bar so the
value loop can close on a reach before full locomotion.

### Gap E — Uncontrolled gaze / brief object permanence (enabling)

Head/camera pose is emergent motor output with no gaze-stabilization reflex, so
the agent stares at the sky and the resource falls out of frame — nothing to
perceive, nothing to value. Object permanence is brief and WM slots are bounded
(12) and decay. Attention (the WS3 gate) selects *what to deliberate on*, not
*where to move* — attending to a ball at the edge of view does not orient the body
toward it. These are enabling constraints on Gaps A/C rather than the core blocker.

---

## 4. Credit horizon → minutes: feasibility and risks

**Feasible.** The horizon is set by the discount, not by the buffers:

- Episode length cap `GOAL_MAX_CYCLES=4000` and `EpisodeAccumulator.max_steps=4096`
  ≈ **~17 minutes** at ~4 cycles/s — already far beyond "minutes."
- λ-returns are computed over the **entire** episode (`episodes.py:on_close` →
  `returns.lambda_returns[_vec]`), so a longer horizon is a matter of weighting,
  not capacity.
- The binding constraint is purely `SF_GAMMA` (and `SF_LAMBDA`).

**Horizon vs γ** (steps `≈1/(1−γ)`, seconds at ~4 cyc/s):

| γ | horizon (steps) | ≈ time | note |
|---|---|---|---|
| 0.97 (current) | ~33 | ~8 s | today |
| 0.99 | ~100 | ~25 s | |
| 0.995 | ~200 | ~50 s | ~1 completable approach |
| 0.998 | ~500 | ~2.1 min | |
| 0.999 | ~1000 | ~4.2 min | |
| ~0.9997 | ~4000 | ~17 min | episode-length-bound |

**What could break (and mitigations):**

1. **Target magnitude growth / instability.** `returns.py` applies **no
   normalization or clipping**; with higher γ the λ-return and SF targets sum over
   more steps and grow. The zero-init SF head then chases larger targets → larger
   gradients. *Mitigate:* normalize SF targets by `(1−γ)` (turns a discounted
   *sum* into a discounted *average*, magnitude-invariant to γ), and/or scale the
   SF learning rate. Low-risk, localized to the consolidator/returns.
2. **Variance.** TD(λ) variance rises with γ (more Monte-Carlo-like). *Mitigate:*
   lower `SF_LAMBDA` as γ rises so the learned-value **bootstrap** carries the long
   range with less variance (the recursion already supports `values=` bootstrap).
3. **Credit smearing — the real tradeoff.** A minutes-long horizon dilutes
   *which* action gets credit: a step three minutes before relief receives nearly
   the same credit as the decisive approach. Overshooting the horizon **weakens**
   the approach signal. *Mitigate:* match γ to realistic approach durations
   (start ~30–60 s, `γ≈0.995`), not the maximum; successor **features** (per
   channel, not a scalar) preserve more structure than a scalar return.
4. **Goal must persist over the horizon (couples to Gap C).** Raising γ lets the
   *value* span minutes, but the agent can only *act* on it if it maintains the
   target representation for that long. Without Gap C's goal→policy bridge, a
   minutes-long value has nothing to act through. **Extend γ and add goal
   persistence together, or the γ change is inert.**
5. **Slower learning.** Minutes-long episodes yield fewer completed episodes per
   hour. *Mitigate:* HER and imagination (both present) already densify the signal;
   the within-reach curriculum (Gap A) supplies frequent short successes first.
6. **Shaping-weight rescale.** `SF_VALUE_WEIGHT=0.3` was tuned against a
   short-horizon value; if targets are *not* `(1−γ)`-normalized, the effective
   shaping strength shifts and must be re-tuned. Normalizing (mitigation 1) makes
   this a non-issue.
7. **Episode fragmentation.** `GOAL_ABANDON_CYCLES=40` switches the active goal if
   a competing need dominates for ~10 s; frequent switching fragments episodes
   below the horizon. Fine for a genuinely dominant need; worth watching if
   multiple deficits compete.

**Recommended horizon plan (staged, reversible — all env-tunable):**

- Step 1: normalize SF targets by `(1−γ)` in `returns.py`/consolidator (magnitude
  safety) — do this *before* touching γ.
- Step 2: raise `SF_GAMMA` 0.97 → **0.995** (~50 s), lower `SF_LAMBDA` 0.9 → ~0.8.
  Re-run the soak; confirm SF loss stays bounded and `successor_value` telemetry
  is finite and discriminative.
- Step 3: only if approaches routinely exceed ~50 s, push γ → 0.998 (~2 min),
  paired with Gap C. Do **not** set the horizon longer than the behavior it must
  bridge.

---

## 5. The memory→goal bridge (Gap C), paired with the horizon

The minimal architectural addition that turns "approach what I see" into "go to
what I remember," reusing existing hooks:

- **Goal signal already exists** (`GoalState.goal_id` + latched deficit). Surface
  it to the policy: concatenate a goal embedding into `pol_in_t`
  (`neural_stack.py:854`) so the motor head is *conditioned on the active need*,
  not only the current percept.
- **Remembered location already exists** (LTM `Entity.position_json`; the
  `predicts_*_relief` belief identifies *which* entity). Derive an **egocentric
  bearing** to the last-known location of the goal-relevant resource from current
  proprioceptive pose, and feed it as a small goal-conditioning input. This is the
  piece that lets incentive salience pull toward an **out-of-view** target.
- **Value then operates on the goal, not just the cue:** with a goal-conditioned
  policy, `successor_predict(z5, motor_u)` can value "turn toward remembered
  bearing" even with the resource off-screen — and the longer horizon (§4) is what
  makes that value non-trivial over a multi-step search.
- Keep the WS3B anti-hallucination discipline: goal conditioning is an *input*;
  SF weights stay detached so the policy cannot inflate its own value.

This is additive wiring around parts that already exist (the SF hooks, GoalState,
LTM positions), in the spirit of the house's "thin wiring on a large stack" rule —
not a new subsystem.

---

## 6. Recommended sequencing

1. **Gap A now** — add "place within reach"; feed it while genuinely thirsty; this
   alone may produce the first reinforced approach with zero code risk.
2. **Horizon Step 1** — `(1−γ)` target normalization (safety, no behavior change).
3. **Horizon Step 2** — γ→0.995 / λ→~0.8; verify stability on a soak.
4. **Gap D** — advance the locomotion curriculum in parallel (teacher → autonomy).
5. **Gap C** — goal-conditioned policy + egocentric bearing to remembered target;
   pair with **Horizon Step 3** only once approaches exceed ~50 s.
6. **Gap E** — a light gaze/orient bias toward salient percepts, if E proves
   rate-limiting after A–D.

Each step is independently verifiable (SF-loss bounds, `successor_value`
telemetry, a "did it approach a within-reach resource" probe, then a "did it
navigate to a remembered out-of-view resource" probe) and env-gated for clean A/B.

---

## 7. Open questions

- Is the intrinsic drive (curiosity = learning-progress) ever *aligned* with
  resource-seeking, or does need only raise arousal? If misaligned, the
  within-reach curriculum is what supplies the extrinsic gradient the curiosity
  drive won't.
- Does a goal-conditioned policy need explicit path memory, or does the
  recurrent state + egocentric bearing suffice for short searches? (Start simple;
  add spatial memory only if searches fail.)
- How much does credit smearing (§4.3) actually cost approach specificity at
  γ=0.995? Measure with a horizon A/B before committing to minutes.
