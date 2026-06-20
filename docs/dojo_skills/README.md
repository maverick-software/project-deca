# Dojo Skill JSON Library

This directory holds uploadable Skill Dojo specs. These are JSON skills that use
existing teachers, body commands, metrics, and validation rules. Upload them from
the dashboard Skill Dojo tab or by calling `POST /dojo/skills/upload`.

## Current Packaged Skill

- `stand_up_from_floor_balance.json`
  - Starts from `kneel_upright`.
  - Runs the `kneel_to_stand` motion.
  - Switches into `stand`.
  - Finishes with autonomous balance dwell and teacher weight `0`.
- `walk_from_stand.json`
  - Starts from `stand`.
  - Uses live teacher support for foot loading and first steps.
  - Adds nearby resource targets so walking has a purpose.
  - Finishes with autonomous walking/foraging and teacher weight `0`.

Each phase includes:

- `body_commands`: commands applied when the phase starts.
- `reset_commands`: commands applied before retrying a failed/timed-out attempt.
- `teacher_adaptation`: optional assist-as-needed controller settings. The
  phase `teacher_weight` is the initial/default value; live `teacher_assist`
  rises on danger/stall metrics and fades after stable dwell.
- `timeout_s`: max seconds for one attempt.
- `max_attempts`: retry cap for the phase.
- `auto_retry`: whether the supervisor should restart failed attempts.
- `gate`: success criteria, all of which must pass.
- `failure_criteria`: fail-fast criteria, any of which closes the attempt.

See `docs/skill_dojo_methodology.md` for the full schema and SOP.
