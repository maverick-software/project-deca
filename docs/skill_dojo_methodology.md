# Skill Dojo Methodology and WBS

## Purpose

The Skill Dojo is the repeatable training layer around the Decadic cognition loop.
It accelerates proof-of-concept skills without replacing the live cognitive cycle
with a generic RL policy. The agent still observes, thinks, acts, and learns
through the Decadic cycle; the dojo adds structured practice, teacher
demonstrations, replay metadata, consolidation-time imitation loss, autonomous
gates, attempt resets, and checkpoints.

## Core Rules

- Teacher signals are training scaffolds, not cognition. They may enter replay
  and consolidation metadata, but the live neural cycle must not import the
  dojo, add a dojo reward term, or let a teacher choose the live action during
  autonomous evaluation.
- Teacher assistance is assist-as-needed. `teacher_weight` is the phase's
  initial/default compatibility value; the supervisor computes live
  `teacher_assist` from stability and danger metrics, then stores that current
  value as replay `demo_weight`.
- Failed attempts reset the body/scaffold only. They do not wipe neural weights,
  memory, replay, or the agent's learned skill state.
- Embodied skill training keeps survival drive active. Dojo phases should use
  `viability_mode="metabolic"` unless a non-embodied diagnostic explicitly
  documents otherwise.
- Caregiver support is environmental, not reward shaping. When enabled, the dojo
  may request the visible parent NPC to deliver food/water/care objects, but it
  must not directly credit reservoirs or inject semantic labels into cognition.
- Final graduation must be autonomous: adaptive teacher assist must be `0`, and
  gates must pass without teacher override.

## Skill SOP

Use this sequence for every new skill.

1. **Skill Card**
   - Choose `skill_id`, version, name, target behavior, required sensors,
     scenario, teacher policy, phases, gates, attempt policy, and checkpoint
     policy.
   - Define the autonomous graduation gate first; earlier phases only scaffold
     toward that gate.

2. **Scenario Design**
   - Specify initial stance, safe body/world commands, resources or threats,
     perturbations, randomization bounds, reset behavior, and failure conditions.
   - Do not use braces or movement hold as skill scaffolds. Braces are manual
     operator/debug tools; teacher targets are the dojo training scaffold.
   - Keep task labels in the dojo/evaluation layer only.

3. **Attempt Policy**
   - Every phase is practiced as one or more attempts.
   - Use `timeout_s` to close stalled attempts.
   - Use `failure_criteria` for immediate failure markers such as collapsed root
     height, excessive torso tilt, or high fall rate.
   - Use `reset_commands` to restore the phase start stance/world state for the
     next attempt.
   - Use `auto_retry` and `max_attempts` to prevent endless education loops.

4. **Teacher Design**
   - Implement a deterministic `TeacherPolicy.motor_target(...)`.
   - Use teacher targets as replay/consolidation hints.
   - Configure `teacher_adaptation` so assist rises when live metrics show
     danger or stalled progress and fades after stable dwell.
   - Force teacher assist to `0` in final evaluation.

5. **Episode Capture**
   - Record phase, attempt, metrics, teacher target, current `teacher_assist`,
     assist reason, origin (`self`, `demo`, `dagger`), success/failure outcome,
     and timestamp.
   - Convert live transitions into replay-compatible records with optional
     `skill_id`, `expert_motor`, and `demo_weight`.

6. **Training**
   - Keep online Decadic learning self-supervised.
   - Use the consolidator for optional imitation loss when replay transitions
     include an expert motor target.
   - Prefer short, high-salience practice attempts over long ungated runs.

7. **Evaluation**
   - Run final gates with teacher weight `0`.
   - Require enough samples plus minimum dwell.
   - For embodied skills with caregiver enabled, require no acute caregiver
     request pending and no missing-parent condition before graduation.
   - Checkpoint and write a skill report on graduation.

8. **Library Maintenance**
   - Each skill ships with a skill card/spec, teacher policy, scenario commands,
     gates, failure criteria, tests, and a sample report from a successful run.
   - Bump the skill version when gates, teacher behavior, or scenario
     assumptions change.

9. **Skill Upload**
   - Prototype and share new skills as JSON files that match the upload schema
     below.
   - Upload through the dashboard Skill Dojo tab or `POST /dojo/skills/upload`.
   - Uploaded skills are validated, persisted under the configured skills
     directory, listed beside built-ins, and can be deleted without touching
     built-in skills.
   - Built-in skills remain code-owned because they may require new teacher
     policies, new body commands, or new metrics.

## Attempt Lifecycle

The dojo supervisor tracks one active run, one active phase, and one active
attempt.

- **Start**: apply phase config, queue phase `body_commands`, bind teacher replay
  metadata, and start attempt timers.
- **Success**: when the phase success gate passes and `min_dwell_s` is satisfied,
  mark the attempt successful, promote to the next phase, or graduate if terminal.
- **Fail-fast**: if any `failure_criteria` criterion is satisfied after
  `failure_min_samples`, close the attempt as failed.
- **Timeout**: if `timeout_s` elapses before success, close the attempt as timed
  out.
- **Retry**: when `auto_retry` is true and `max_attempts` is not exhausted, clear
  the phase window, queue `reset_commands`, reapply phase metadata, and start the
  next attempt.
- **Failed run**: when retries are exhausted or `auto_retry` is false, set dojo
  state to `failed`, clear teacher metadata, and write the report.

## Adaptive Teacher Lifecycle

Each non-terminal phase may define `teacher_adaptation`. This turns the teacher
from a fixed percent into a closed-loop assist controller.

- `teacher_weight` remains the phase start/default value for compatibility.
- `min_weight` and `max_weight` bound live `teacher_assist`.
- `rise_rate` increases assistance quickly when danger metrics exceed
  thresholds.
- `fade_rate` decreases assistance slowly after stable dwell.
- `danger_thresholds` describe when the student needs help, for example
  `root_height_min`, `torso_tilt_max`, `fall_rate_max`,
  `stance_phase_delta_min`, `forward_model_error_max`, or
  `tactile_pred_error_max`.
- `stability_thresholds` describe when the student is holding control well
  enough to fade help.
- `stable_dwell_s` and `unstable_dwell_s` add hysteresis so assist does not
  oscillate every sample.
- `zero_required_for_graduation` requires autonomous control before graduation.

The supervisor recomputes teacher targets every sample. Replay metadata uses the
current adaptive value:

- `origin = "self"` when assist is zero.
- `origin = "dagger"` when the student is acting with partial correction.
- `origin = "demo"` when teacher assist is high.
- `demo_weight = teacher_assist`.
- `expert_motor` is the current teacher motor target.

V1 keeps this consolidation-only. The teacher does not directly override live
actions. A future live-blending option would need explicit intervention logging
and must not count as autonomous graduation.

## Caregiver Survival Scaffold

Embodied dojo skills may set `caregiver_enabled=true` and
`caregiver_threshold=80.0`. The supervisor then runs an always-on monitor
alongside phase progression:

- It samples `hydration`, `energy`, and `integrity`.
- If any reservoir drops below threshold, the lowest reservoir determines the
  request: hydration -> `parent_request:water`, energy -> `parent_request:food`,
  integrity -> `parent_request:care`.
- The request is queued as a body/environment command only. It does not change
  cognition, reward terms, replay labels, teacher targets, or object semantics.
- The MuJoCo parent must visibly approach, carry, and drop the existing gift
  object. Relief happens only when the agent perceives/contacts/consumes the
  object and the normal homeostasis path fires.
- A refractory prevents request spam.
- If the active environment lacks the parent NPC, status reports
  `caregiver_missing_parent` and embodied graduation is blocked.

Status fields are exposed through `/dojo/status`: `caregiver_enabled`,
`caregiver_status`, `caregiver_need`, `caregiver_trigger_reservoir`,
`caregiver_request_kind`, `caregiver_last_offer_cycle`,
`caregiver_last_offer_item`, `caregiver_missing_parent`,
`caregiver_refractory_s`, and `caregiver_delivery_count`.

## Uploadable Skill JSON

Uploaded skills may use existing teachers and a restricted set of safe scenario
commands/config knobs. New teachers or new body commands should be added in code
first, then referenced from uploaded JSON.

```json
{
  "skill_id": "mini_recover",
  "version": "1.0",
  "name": "Mini Recover",
  "description": "Short stand-and-recover variant.",
  "target_behavior": "Return upright after a small perturbation.",
  "teacher": "stand_teacher",
  "required_sensors": ["proprioception", "contacts"],
  "checkpoint_on_graduate": true,
  "caregiver_enabled": true,
  "caregiver_threshold": 80.0,
  "phases": [
    {
      "index": 0,
      "name": "Assisted Upright",
      "description": "Gather stable samples with teacher guidance and no manual braces.",
      "teacher_weight": 0.75,
      "teacher_adaptation": {
        "enabled": true,
        "min_weight": 0.0,
        "max_weight": 0.75,
        "rise_rate": 0.7,
        "fade_rate": 0.1,
        "stable_dwell_s": 3.0,
        "unstable_dwell_s": 0.5,
        "danger_thresholds": {
          "root_height_min": 1.0,
          "torso_tilt_max": 0.8,
          "fall_rate_max": 0.2
        },
        "stability_thresholds": {
          "root_height_min": 1.1,
          "torso_tilt_max": 0.35,
          "fall_rate_max": 0.05
        },
        "zero_required_for_graduation": false
      },
      "config": { "viability_mode": "metabolic", "motor_babble_sigma": 0.0 },
      "body_commands": ["set_stance:stand", "recenter"],
      "reset_commands": ["set_stance:stand", "recenter"],
      "timeout_s": 90,
      "max_attempts": 5,
      "auto_retry": true,
      "min_dwell_s": 20,
      "gate": {
        "min_samples": 8,
        "criteria": [
          {
            "key": "root_height",
            "comparator": ">=",
            "threshold": 1.0,
            "label": "standing height",
            "unit": "m"
          }
        ]
      },
      "failure_min_samples": 2,
      "failure_criteria": [
        {
          "key": "torso_tilt",
          "comparator": ">=",
          "threshold": 1.2,
          "label": "torso tipped over",
          "unit": "rad"
        }
      ]
    }
  ]
}
```

Allowed comparators are `<=`, `>=`, and `trend>=`. Success gates require all
criteria to pass. Failure criteria are fail-fast: any satisfied failure criterion
closes the attempt.

Allowed uploaded body commands are limited to safe scenario setup commands:
`set_stance:*`, `recenter`, and `perturb:small`. Uploaded skills may not command
`braces_on`, `braces_off`, `reset_braces`, `hold_on`, or `hold_off`. Existing
legacy uploaded skills are migrated by stripping those commands on load.

## REST and Dashboard Surface

API endpoints live under `/dojo/*`.

- `GET /dojo/skills`: list built-in and uploaded specs.
- `GET /dojo/skills/{skill_id}`: fetch one spec.
- `POST /dojo/skills/upload`: validate and persist uploaded JSON.
- `DELETE /dojo/skills/{skill_id}`: delete an uploaded skill.
- `POST /dojo/start`: start a run with `agent_id`, `skill_id`, and optional
  `auto_retry`, `max_attempts`, and `timeout_multiplier` overrides.
- `GET /dojo/status`: current run state, phase, gate, failure marker, attempt
  count, timeout, retry policy, adaptive teacher assist/range/rates/reason,
  teacher origin, caregiver status, reservoir levels, manual scaffold flag,
  report path, and history.
- `POST /dojo/pause`, `/dojo/resume`, `/dojo/stop`: lifecycle controls.
- `POST /dojo/phase`: manual phase jump for experiments.

The dashboard Skill Dojo panel supports built-in selection, JSON upload/delete,
auto-retry controls, max-attempt and timeout overrides, start/pause/resume/stop,
manual phase jumps, live gates, failure reason, attempt countdown, and report-path
visibility. During a run it shows a live teacher-assist meter and the reason help
is rising, fading, or staying idle, plus caregiver need/request/delivery state.

## Stand and Recover V1 Skill Card

- `skill_id`: `stand_and_recover`
- `version`: `1.0`
- Target: remain upright and recover from small disturbances without teacher help.
- Teacher: `stand_teacher`, a neutral stand motor target.
- Required sensors: proprioception and contacts.
- Attempt policy: auto-retry on, five attempts per phase by default, phase
  timeouts between 90 and 180 seconds.
- Fail-fast markers: root height below the standing floor, excessive torso tilt,
  or high fall rate.
- Final success: low fall rate, standing root height, bounded torso tilt,
  adaptive teacher assist `0`, teacher override fraction `0`,
  checkpoint/report written.

Phases:

1. **Teacher Standing Familiarization**
   - Body: `set_stance:stand`, `recenter`.
   - Teacher assist: starts high and fades after stable upright dwell.
   - Gates: low fall rate, low forward-model error, low tactile prediction error,
     manual braces off, movement hold off.

2. **Small Perturbation Recovery**
   - Body: `set_stance:stand`, `recenter`; periodic `perturb:small`.
   - Teacher assist: rises on perturbation failure risk and fades after recovery.
   - Gates: low fall rate, standing root height, bounded torso tilt, manual
     scaffold off.

3. **Reduced Assistance**
   - Body: `set_stance:stand`, `recenter`.
   - Teacher assist: capped low with stronger fade.
   - Gates: rare falls, standing root height, manual scaffold off.

4. **Autonomous Evaluation**
   - Body: `set_stance:stand`, `recenter`; periodic `perturb:small`.
   - Teacher assist: forced to `0.0`.
   - Gates: fall rate <= 0.05, root height >= 1.05, torso tilt <= 0.35,
     teacher override fraction == 0, manual scaffold off.

## Stand Up From Floor and Balance V1 Skill Card

- Packaged JSON: `docs/dojo_skills/stand_up_from_floor_balance.json`
- `skill_id`: `stand_up_from_floor_balance`
- Target: start sitting upright on the knees, push through the feet into a stand,
  and balance without teacher hints.
- Teacher: `stand_teacher`, adaptive assist-as-needed, forced to `0` in final
  evaluation.
- Key metric: `stance_phase` gates completion of the `kneel_to_stand` motion.
- Attempt policy: auto-retry on, five attempts per phase, timeouts from 90 to
  180 seconds.

Phases:

1. **Upright Kneel Familiarization**
   - Body: `set_stance:kneel_upright`, `recenter`.
   - Goal: low, vertical, stable pre-stand posture.

2. **Kneel to Stand Push-Off**
   - Body: `set_stance:kneel_to_stand`, `recenter`.
   - Goal: complete the motion (`stance_phase >= 0.95`) without repeated falls.

3. **Assisted Standing Balance**
   - Body: `set_stance:stand`, `recenter`.
   - Goal: establish standing height and bounded torso tilt with partial teacher
     hint.

4. **Autonomous Balance Dwell**
   - Body: `set_stance:stand`, `perturb:small`.
   - Teacher assist: forced to `0.0`.
   - Goal: sustained autonomous standing balance before graduation.

## Reusable WBS Template

1. Add or update the `SkillSpec` or uploadable JSON.
2. Add or update the `TeacherPolicy`.
3. Add body/world commands needed by the scenario.
4. Add any new eval-only metrics to agent telemetry and dojo sampling.
5. Add phase success gates, fail-fast gates, timeouts, reset commands, and retry
   policy.
6. Add supervisor/API tests for start, pause/resume, phase jump, success,
   failure, timeout, retry exhaustion, and reports.
7. Add replay/consolidation tests if the skill uses teacher metadata.
8. If the skill can use existing teachers/commands, package it as uploadable
   JSON and test dashboard upload/list/delete.
9. Run autonomous graduation and save a sample report.

## V1 Implementation Notes

- The Skill Dojo supervisor is `decadic.training.supervisor.SkillDojoSupervisor`.
- Skill specs and upload parsing live under `decadic/training/`.
- Built-ins are code-owned in `decadic/training/skills.py`; uploaded specs are
  persisted under `DECADIC_SKILLS_DIR` or the default data skills directory.
- The consolidator applies imitation only when replay transitions carry
  `expert_motor` and `demo_weight > 0`.
- `teacher_override_fraction` reflects current adaptive assist and is
  telemetry/evaluation only.
