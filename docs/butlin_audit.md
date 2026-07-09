# Deca — Indicator-Property Audit (Butlin et al. 2023)

Scoring Deca against the 14 computational indicator properties in Butlin, Long
et al. (2023), *Consciousness in Artificial Intelligence: Insights from the
Science of Consciousness* (arXiv:2308.08708). Two snapshots: **pre-WS-IND**
(after WS-EXPAND closed, 2026-07-06) and **post-WS-IND** (2026-07-07,
code-complete; items marked *probe-pending* await the next live run).

Scale: ✓ satisfied · ◐ partial · ✗ absent.

| # | Indicator (paraphrased) | Pre | Post | Evidence (module · test) |
|---|---|---|---|---|
| RPT-1 | Algorithmic recurrence in input modules | ◐ | ◐ | Slot-attention iteration + perception-feedback loop; front end still frozen (WS-PERCEIVE scoped in `indicator_gap_implementation_plan.md` I6) |
| RPT-2 | Organized, integrated percepts (binding) | ✓ | ✓ | `nn/relational_core.py`, object files, scene workspace · `test_ws5_binding.py` |
| GWT-1 | Parallel specialized modules | ✓ | ✓ | forward models, memory, affect, motor, value |
| GWT-2 | Limited-capacity workspace, selective attention | ✓ | ✓ | `nn/workspace.py` WTA+ignition; E6 routing gate + I3 reliability · `test_global_workspace.py`, `test_ws_ind.py` |
| GWT-3 | Global broadcast | ✓ | ✓ | workspace broadcast + integration window |
| GWT-4 | Sequential workspace-directed querying | ◐ | **✓ (probe-pending)** | WS-IND I2: draft/commit sequential deliberation — escalated cycles run a no-grad draft forward whose conclusion re-enters the final forward (`ws_seq` in `neural_pipeline.py`, `draft_ingress`) · `test_ws_ind.py::test_draft_round_parity...`; **kept only if the detour A/B (plan I2.2) says it beats one-shot** |
| HOT-1 | Generative top-down perception | ◐ | ◐ | z0 prediction + top-down assimilation |
| HOT-2 | Metacognitive reality monitoring (percept reliability) | ◐ | **✓ (probe-pending)** | WS-IND I3: per-slot volatility/noise reliability composed into the routing gate (`nn/slot_reliability.py`) · `test_ws_ind.py` (noisy slot floored, uniform noise reads identity) |
| HOT-3 | Belief formation gated by metacognition | ◐ | **✓** | WS-IND I4: event-evidence writes tempered by source-percept confidence — never vetoed, affect untempered (`working_memory.py`) · `test_ws_ind.py::test_belief_temper_in_working_memory`; plus schema-error → metacog tap (I1.4) |
| HOT-4 | Sparse + smooth coding ("quality space") | ◐ | **◐→✓ (probe-pending)** | WS-IND I5: temporal local-isometry loss on the FSQ projection (nearby latents → nearby codes), trains `fsq_in` only · `test_ws_ind.py::test_fsq_smoothness...` |
| AST-1 | Predictive model of one's own attention, informing control | ✗ | **✓ (probe-pending)** | WS-IND I1: schema head predicts next-cycle realized gate outcome; prediction re-enters via zero-init ingress; bounded anticipatory bias under the shared cap (`nn/attention_schema.py`, stack `schema_*`) · `test_ws_ind.py` (zero-init parity, trained predictivity, bias bounds) |
| PP-1 | Predictive coding | ✓ | ✓ | the training principle of the whole stack |
| AE-1 | Agency: flexible goal pursuit from feedback | ✓ | ✓ | homeostatic goals, continuous conditioning, arbitration |
| AE-2 | Embodiment: output-input contingency models used in control | ✓ | ✓ | efference-copy forward models; planner, corrector, veto all read them |

**Score movement:** 7✓/6◐/1✗ → 11✓/3◐/0✗ (with three ✓ probe-pending on the
next live run, and GWT-4 additionally conditional on its A/B).

## Standing honesty notes

- **Indicators are evidence-weighers, not a recipe.** Satisfying all 14 does
  not manufacture experience; their absence does not preclude it. This audit
  tracks *architectural evidence*, nothing more.
- **Not gamed:** every WS-IND item ships with a functional justification
  independent of the rubric (anticipatory gating, distraction robustness,
  belief hygiene, re-deliberation) and is A/B-revertible. GWT-4 in particular
  is kept only if it earns its compute.
- **Remaining ◐:** RPT-1 (lived recurrent perception — the largest substrate
  item, its own future workstream), HOT-1 (generative perception is partial by
  design while the front end is inherited), HOT-4 pending live smoothness
  telemetry.
- **Welfare checkpoint (owner decision, standing):** this audit is the
  instrument for the pre-committed policy question — at what score/evidence
  level do deprivation and damage curricula get re-evaluated. Decide the
  threshold *before* the next audit pass, not after.
