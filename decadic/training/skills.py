"""Built-in Skill Dojo skill library."""

from __future__ import annotations

from decadic.training.gates import Criterion
from decadic.training.types import (
    PeriodicBodyCommand,
    SkillGate,
    SkillPhase,
    SkillSpec,
    TeacherAdaptation,
)


NO_MANUAL_SCAFFOLD = (
    Criterion("braces_enabled", "<=", 0.0, "Manual braces off"),
    Criterion("movement_hold", "<=", 0.0, "Manual movement hold off"),
)


def _assist(
    *,
    max_weight: float,
    min_weight: float = 0.0,
    rise_rate: float = 0.55,
    fade_rate: float = 0.08,
    root_min: float = 1.0,
    tilt_max: float = 0.6,
    fall_max: float = 0.2,
    stable_root: float = 1.05,
    stable_tilt: float = 0.35,
    stable_fall: float = 0.08,
    stable_dwell_s: float = 3.0,
    stance_phase_delta_min: float | None = None,
) -> TeacherAdaptation:
    danger = {
        "root_height_min": root_min,
        "torso_tilt_max": tilt_max,
        "fall_rate_max": fall_max,
        "forward_model_error_max": 5.0,
        "tactile_pred_error_max": 5.0,
    }
    if stance_phase_delta_min is not None:
        danger["stance_phase_delta_min"] = stance_phase_delta_min
    return TeacherAdaptation(
        enabled=max_weight > 0.0,
        min_weight=min_weight,
        max_weight=max_weight,
        rise_rate=rise_rate,
        fade_rate=fade_rate,
        danger_thresholds=danger,
        stability_thresholds={
            "root_height_min": stable_root,
            "torso_tilt_max": stable_tilt,
            "fall_rate_max": stable_fall,
            "forward_model_error_max": 2.0,
            "tactile_pred_error_max": 2.0,
        },
        stable_dwell_s=stable_dwell_s,
        unstable_dwell_s=0.0,
        zero_required_for_graduation=True,
    )


AUTONOMOUS_TEACHER = TeacherAdaptation(
    enabled=False,
    min_weight=0.0,
    max_weight=0.0,
    zero_required_for_graduation=True,
)


STAND_AND_RECOVER = SkillSpec(
    skill_id="stand_and_recover",
    version="1.0",
    name="Stand and Recover",
    description="Learn teacher-guided standing, perturbation recovery, reduced assistance, and autonomous upright evaluation.",
    target_behavior="Remain upright and recover from small disturbances without teacher assistance.",
    teacher="stand_teacher",
    caregiver_enabled=True,
    phases=(
        SkillPhase(
            index=0,
            name="Teacher Standing Familiarization",
            description="Stand pose, neutral teacher target, low babble, and no manual braces.",
            teacher_weight=1.0,
            teacher_adaptation=_assist(
                max_weight=1.0,
                rise_rate=0.65,
                fade_rate=0.08,
                root_min=1.0,
                tilt_max=0.65,
                fall_max=0.2,
                stable_dwell_s=3.0,
            ),
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.1},
            body_commands=("set_stance:stand", "recenter"),
            reset_commands=("set_stance:stand", "recenter"),
            gate=SkillGate(
                (
                    Criterion("fall_rate", "<=", 0.05, "No repeated falls"),
                    Criterion("forward_model_error", "<=", 0.08, "Forward model settling"),
                    Criterion("tactile_pred_error", "<=", 0.08, "Tactile model settling"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=8,
            ),
            failure_gate=SkillGate(
                (
                    Criterion("root_height", "<=", 0.65, "Root dropped below standing floor", "m"),
                    Criterion("torso_tilt", ">=", 1.2, "Torso tipped over", "rad"),
                    Criterion("fall_rate", ">=", 0.5, "Fall rate too high"),
                ),
                min_samples=2,
            ),
            timeout_s=90.0,
            max_attempts=5,
            auto_retry=True,
            min_dwell_s=20.0,
        ),
        SkillPhase(
            index=1,
            name="Small Perturbation Recovery",
            description="Queue small body perturbations while teacher hints remain available.",
            teacher_weight=0.6,
            teacher_adaptation=_assist(
                max_weight=0.8,
                rise_rate=0.75,
                fade_rate=0.1,
                root_min=0.95,
                tilt_max=0.65,
                fall_max=0.2,
                stable_dwell_s=3.0,
            ),
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.15},
            reset_commands=("set_stance:stand", "recenter"),
            periodic_body_commands=(PeriodicBodyCommand("perturb:small", 5.0),),
            gate=SkillGate(
                (
                    Criterion("fall_rate", "<=", 0.1, "Recovers without frequent falls"),
                    Criterion("root_height", ">=", 1.05, "Root remains standing-high"),
                    Criterion("torso_tilt", "<=", 0.35, "Torso remains near upright"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=8,
            ),
            failure_gate=SkillGate(
                (
                    Criterion("root_height", "<=", 0.65, "Root dropped below standing floor", "m"),
                    Criterion("torso_tilt", ">=", 1.2, "Torso tipped over", "rad"),
                    Criterion("fall_rate", ">=", 0.5, "Fall rate too high"),
                ),
                min_samples=2,
            ),
            timeout_s=120.0,
            max_attempts=5,
            auto_retry=True,
            min_dwell_s=25.0,
        ),
        SkillPhase(
            index=2,
            name="Reduced Assistance",
            description="Lower teacher pressure while keeping manual brace scaffold disabled.",
            teacher_weight=0.2,
            teacher_adaptation=_assist(
                max_weight=0.35,
                rise_rate=0.45,
                fade_rate=0.16,
                root_min=0.95,
                tilt_max=0.55,
                fall_max=0.15,
                stable_dwell_s=2.0,
            ),
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.12},
            reset_commands=("set_stance:stand", "recenter"),
            gate=SkillGate(
                (
                    Criterion("fall_rate", "<=", 0.08, "Rare falls"),
                    Criterion("root_height", ">=", 1.05, "Standing height retained"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=10,
            ),
            failure_gate=SkillGate(
                (
                    Criterion("root_height", "<=", 0.65, "Root dropped below standing floor", "m"),
                    Criterion("torso_tilt", ">=", 1.2, "Torso tipped over", "rad"),
                    Criterion("fall_rate", ">=", 0.5, "Fall rate too high"),
                ),
                min_samples=2,
            ),
            timeout_s=150.0,
            max_attempts=5,
            auto_retry=True,
            min_dwell_s=30.0,
        ),
        SkillPhase(
            index=3,
            name="Autonomous Evaluation",
            description="Teacher disabled; randomized small perturbations must be survived by the learned loop.",
            teacher_weight=0.0,
            teacher_adaptation=AUTONOMOUS_TEACHER,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.05},
            reset_commands=("set_stance:stand", "recenter"),
            periodic_body_commands=(PeriodicBodyCommand("perturb:small", 5.0),),
            gate=SkillGate(
                (
                    Criterion("fall_rate", "<=", 0.05, "Autonomous low fall rate"),
                    Criterion("root_height", ">=", 1.05, "Root stays standing-high"),
                    Criterion("torso_tilt", "<=", 0.35, "Torso tilt bounded"),
                    Criterion("teacher_override_fraction", "<=", 0.0, "Teacher disabled"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=12,
            ),
            failure_gate=SkillGate(
                (
                    Criterion("root_height", "<=", 0.65, "Root dropped below standing floor", "m"),
                    Criterion("torso_tilt", ">=", 1.2, "Torso tipped over", "rad"),
                    Criterion("fall_rate", ">=", 0.5, "Fall rate too high"),
                ),
                min_samples=2,
            ),
            timeout_s=180.0,
            max_attempts=5,
            auto_retry=True,
            min_dwell_s=40.0,
            is_terminal=True,
        ),
    ),
)

DEVELOPMENTAL_LOCOMOTION = SkillSpec(
    skill_id="developmental_locomotion",
    version="1.0",
    name="Developmental Locomotion",
    description="Legacy walking curriculum migrated into Skill Dojo phases: self-model, posture, locomotion onset, and sustained forage.",
    target_behavior="Develop from body self-modeling into autonomous resource-seeking locomotion without manual brace scaffold.",
    teacher="stand_teacher",
    caregiver_enabled=True,
    phases=(
        SkillPhase(
            index=0,
            name="Self-modeling",
            description=(
                "Low-drive body self-modeling with the manual brace scaffold disabled."
            ),
            teacher_weight=0.0,
            config={
                "viability_mode": "metabolic",
                "metabolic_compression": 1.0,
                "motor_babble_sigma": 0.15,
            },
            body_commands=("set_stance:stand", "recenter"),
            reset_commands=("set_stance:stand", "recenter"),
            gate=SkillGate(
                (
                    Criterion("forward_model_error", "<=", 0.05, "World-model PE low"),
                    Criterion("tactile_pred_error", "<=", 0.05, "Tactile PE low"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=12,
            ),
            min_dwell_s=30.0,
            demote_on_death=False,
        ),
        SkillPhase(
            index=1,
            name="Postural control",
            description=(
                "Gentle metabolism switches on. With partly-free joints the body must hold itself up "
                "while falls stay rare and ROM keeps widening."
            ),
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 1.0},
            gate=SkillGate(
                (
                    Criterion("fall_rate", "<=", 0.1, "Stays upright"),
                    Criterion("root_height", ">=", 1.0, "Root remains upright"),
                    Criterion("forward_model_error", "<=", 0.06, "World-model PE low"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=15,
            ),
            min_dwell_s=45.0,
            demote_on_death=True,
        ),
        SkillPhase(
            index=2,
            name="Locomotion onset",
            description=(
                "Drive rises and food/water are placed near the body so the agent must perceive, move, "
                "and consume through its learned act-to-relief loop."
            ),
            teacher_weight=0.0,
            config={
                "viability_mode": "metabolic",
                "metabolic_compression": 2.0,
                "drive_priority_gain": 3.0,
                "motor_babble_sigma": 0.25,
            },
            periodic_body_commands=(
                PeriodicBodyCommand("give_food_near", 20.0),
                PeriodicBodyCommand("give_water_near", 20.0),
            ),
            gate=SkillGate(
                (
                    Criterion("consume_events", "trend>=", 1.0, "Reaches and consumes"),
                    Criterion("viability", "trend>=", 0.0, "Net non-negative viability"),
                    Criterion("distance_traveled", "trend>=", 0.5, "Moves toward goal", "m"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=15,
            ),
            min_dwell_s=60.0,
            demote_on_death=True,
        ),
        SkillPhase(
            index=3,
            name="Sustained gait / forage",
            description=(
                "Food/water arrive more slowly, requiring sustained foraging, rising travel, and a "
                "more regular left-right gait."
            ),
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0},
            periodic_body_commands=(
                PeriodicBodyCommand("give_food_near", 60.0),
                PeriodicBodyCommand("give_water_near", 60.0),
            ),
            gate=SkillGate(
                (
                    Criterion("consume_events", "trend>=", 3.0, "Sustained foraging"),
                    Criterion("viability", ">=", 50.0, "Surviving comfortably"),
                    Criterion("distance_traveled", "trend>=", 2.0, "Walks to forage", "m"),
                    Criterion("gait_regularity", ">=", 0.3, "Regular gait"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=20,
            ),
            min_dwell_s=120.0,
            demote_on_death=True,
            is_terminal=True,
        ),
    ),
)

_AFFECTIVE_BASE = tuple(
    SkillPhase(
        index=p.index,
        name=p.name,
        description=p.description,
        teacher_weight=p.teacher_weight,
        config=dict(p.config),
        body_commands=p.body_commands,
        periodic_body_commands=p.periodic_body_commands,
        gate=p.gate,
        failure_gate=p.failure_gate,
        reset_commands=p.reset_commands,
        timeout_s=p.timeout_s,
        max_attempts=p.max_attempts,
        auto_retry=p.auto_retry,
        min_dwell_s=p.min_dwell_s,
        demote_on_death=p.demote_on_death,
        is_terminal=False if p.index == 3 else p.is_terminal,
        teacher_adaptation=p.teacher_adaptation,
    )
    for p in DEVELOPMENTAL_LOCOMOTION.phases
)

AFFECTIVE_LOCOMOTION = SkillSpec(
    skill_id="affective_locomotion",
    version="1.0",
    name="Affective Locomotion",
    description="Developmental locomotion plus the legacy affective stretch phase for urgent movement under stronger drive.",
    target_behavior="Forage and keep moving under a stronger affective/drive regime.",
    teacher="stand_teacher",
    caregiver_enabled=True,
    phases=(
        *_AFFECTIVE_BASE,
        SkillPhase(
            index=4,
            name="Affective gait",
            description=(
                "Higher drive pressure asks for urgent locomotion while the agent still forages and survives."
            ),
            teacher_weight=0.0,
            config={
                "viability_mode": "metabolic",
                "metabolic_compression": 2.0,
                "drive_priority_gain": 4.0,
                "motor_babble_sigma": 0.25,
            },
            periodic_body_commands=(
                PeriodicBodyCommand("give_food_near", 45.0),
                PeriodicBodyCommand("give_water_near", 45.0),
            ),
            gate=SkillGate(
                (
                    Criterion("distance_traveled", "trend>=", 3.0, "Flees / forages", "m"),
                    Criterion("viability", ">=", 40.0, "Survives the pressure"),
                    *NO_MANUAL_SCAFFOLD,
                ),
                min_samples=20,
            ),
            min_dwell_s=120.0,
            demote_on_death=True,
            is_terminal=True,
        ),
    ),
)

PERCEPTION_OBJECT_FILES = SkillSpec(
    skill_id="perception_object_files",
    version="1.0",
    name="Perception Object Files",
    description="Bootstrap and evaluate anonymous visual object-file separation before object-dependent skills.",
    target_behavior="Produce separated, persistent, healthy object files and allow LTM to consolidate them without semantic labels.",
    teacher="none",
    required_sensors=("vision", "proprioception"),
    phases=(
        SkillPhase(
            index=0,
            name="Static Scene Separation",
            description="Hold a multi-object visual scene and require non-collapsed anonymous object files.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.0},
            body_commands=("recenter",),
            gate=SkillGate(
                (
                    Criterion("object_files", ">=", 2.0, "Multiple anonymous object files"),
                    Criterion("centroid_spread", ">=", 0.04, "Separated image centroids"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=8,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=90.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=20.0,
        ),
        SkillPhase(
            index=1,
            name="Enter Exit Reappear",
            description="Let objects move through the camera and require stable tracked object files.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.03},
            periodic_body_commands=(PeriodicBodyCommand("perturb:small", 8.0),),
            gate=SkillGate(
                (
                    Criterion("stable_tracked_objects", ">=", 2.0, "Persistent object files"),
                    Criterion("centroid_spread", ">=", 0.04, "Maintains spatial separation"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=10,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=120.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=25.0,
        ),
        SkillPhase(
            index=2,
            name="Motion And Parallax",
            description="Require local visual motion channels to separate moving regions from global scene flow.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.05},
            periodic_body_commands=(PeriodicBodyCommand("perturb:small", 6.0),),
            gate=SkillGate(
                (
                    Criterion("object_files", ">=", 2.0, "Object files available"),
                    Criterion("flow_confidence", ">=", 0.01, "Local motion signal present"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=10,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=120.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=25.0,
        ),
        SkillPhase(
            index=3,
            name="Looming And Stuff Rejection",
            description="Detect expansion/collision cues while keeping background stuff out of foreground object memory.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.05},
            periodic_body_commands=(PeriodicBodyCommand("perturb:small", 6.0),),
            gate=SkillGate(
                (
                    Criterion("object_files", ">=", 2.0, "Foreground object files"),
                    Criterion("stuff_count", ">=", 0.0, "Stuff channel reported"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=10,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=120.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=25.0,
        ),
        SkillPhase(
            index=4,
            name="Body Candidate Correlation",
            description="Visual motion coupled with motor/touch should create anonymous body-part candidates.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.1},
            gate=SkillGate(
                (
                    Criterion("body_candidate_count", ">=", 0.0, "Body-candidate channel active"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=10,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=120.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=25.0,
        ),
        SkillPhase(
            index=5,
            name="Autonomous Memory Gate",
            description="Teacher disabled; LTM must accept healthy object files and create durable relational memory.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "motor_babble_sigma": 0.05},
            gate=SkillGate(
                (
                    Criterion("object_files", ">=", 2.0, "Object files available"),
                    Criterion("stable_tracked_objects", ">=", 2.0, "Stable objects"),
                    Criterion("ltm_write_accepted", ">=", 1.0, "LTM writes accepted"),
                    Criterion("perception_collapsed", "<=", 0.0, "Perception not collapsed"),
                ),
                min_samples=12,
            ),
            failure_gate=SkillGate(
                (Criterion("perception_collapsed", ">=", 1.0, "Perception collapsed"),),
                min_samples=4,
            ),
            timeout_s=150.0,
            max_attempts=3,
            auto_retry=True,
            min_dwell_s=30.0,
            is_terminal=True,
        ),
    ),
    warnings=(
        "Uses anonymous object-file health gates only; semantic oracle labels are not fed to live cognition.",
    ),
)

RESOURCE_ACQUISITION_ENERGY = SkillSpec(
    skill_id="resource_acquisition_energy",
    version="1.0",
    name="Resource Acquisition Energy",
    description="Learn that self-generated movement toward anonymous objects can restore internal reservoirs despite effort cost.",
    target_behavior="Acquire unlabeled resource affordances with positive net energy return and teacher disabled.",
    teacher="none",
    required_sensors=("vision", "proprioception", "contacts"),
    phases=(
        SkillPhase(
            index=0,
            name="Perception And Relief Binding",
            description="Place a reachable anonymous resource object and bind reservoir relief to the attended object file.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.12},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 18.0),),
            gate=SkillGate(
                (
                    Criterion("object_files", ">=", 1.0, "Anonymous object file present"),
                    Criterion("consume_events", "trend>=", 1.0, "Consumption/relief occurred"),
                    Criterion("drive_reward_last", ">=", 0.0, "Drive relief available"),
                ),
                min_samples=10,
            ),
            min_dwell_s=45.0,
            timeout_s=120.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=1,
            name="Body-Localized Contact",
            description="Require contact to register through the body map while reservoir relief occurs.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.15},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 20.0),),
            gate=SkillGate(
                (
                    Criterion("pain_total", ">=", 0.0, "Body-map pain channel present"),
                    Criterion("effort_total", ">=", 0.0, "Effort channel present"),
                    Criterion("consume_events", "trend>=", 1.0, "Relief follows contact"),
                ),
                min_samples=10,
            ),
            min_dwell_s=45.0,
            timeout_s=120.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=2,
            name="Approach And Contact",
            description="Resource is placed outside contact radius; the agent must move, contact, and improve reservoirs.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.22},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 26.0),),
            gate=SkillGate(
                (
                    Criterion("distance_traveled", "trend>=", 0.4, "Self-generated approach movement"),
                    Criterion("consume_events", "trend>=", 1.0, "Consumes after movement"),
                    Criterion("net_energy_return", ">=", -0.02, "Reservoir change not badly negative"),
                ),
                min_samples=12,
            ),
            min_dwell_s=60.0,
            timeout_s=150.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=3,
            name="Reidentification",
            description="Repeated anonymous affordance opportunities from varied placements should strengthen LTM beliefs.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.18},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 35.0),),
            gate=SkillGate(
                (
                    Criterion("ltm_property_beliefs", ">=", 1.0, "LTM property beliefs exist"),
                    Criterion("consume_events", "trend>=", 2.0, "Repeated relief experiences"),
                    Criterion("object_files", ">=", 1.0, "Anonymous objects remain available"),
                ),
                min_samples=12,
            ),
            min_dwell_s=75.0,
            timeout_s=180.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=4,
            name="Effort-Aware Acquisition",
            description="Effort drain is active; consumption must beat movement cost.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.15},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 45.0),),
            gate=SkillGate(
                (
                    Criterion("net_energy_return", ">=", 0.0, "Positive net energy return"),
                    Criterion("effort_total", "<=", 0.8, "Effort remains bounded"),
                    Criterion("fatigue_total", "<=", 0.8, "Fatigue remains bounded"),
                    Criterion("consume_events", "trend>=", 1.0, "Resource acquired"),
                ),
                min_samples=14,
            ),
            min_dwell_s=90.0,
            timeout_s=210.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=5,
            name="Choice Under Cost",
            description="Mixed objects are present; learned anonymous affordance should guide approach without labels.",
            teacher_weight=0.0,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.12},
            periodic_body_commands=(PeriodicBodyCommand("give_food_near", 55.0),),
            gate=SkillGate(
                (
                    Criterion("resource_relief_events", "trend>=", 1.0, "Relief-producing affordance found"),
                    Criterion("net_energy_return", ">=", -0.01, "Cost stays controlled"),
                    Criterion("teacher_override_fraction", "<=", 0.0, "Teacher disabled"),
                ),
                min_samples=14,
            ),
            min_dwell_s=90.0,
            timeout_s=210.0,
            max_attempts=4,
            auto_retry=True,
        ),
        SkillPhase(
            index=6,
            name="Autonomous Survival Evaluation",
            description="Teacher off; randomized resource acquisition must sustain reservoirs under effort drain.",
            teacher_weight=0.0,
            teacher_adaptation=AUTONOMOUS_TEACHER,
            config={"viability_mode": "metabolic", "metabolic_compression": 2.0, "motor_babble_sigma": 0.08},
            periodic_body_commands=(PeriodicBodyCommand("randomize_resources", 80.0),),
            gate=SkillGate(
                (
                    Criterion("viability", ">=", 35.0, "Survival reservoir floor"),
                    Criterion("resource_relief_events", "trend>=", 1.0, "Autonomous relief events"),
                    Criterion("net_energy_return", ">=", -0.02, "Energy trend not collapsing"),
                    Criterion("teacher_override_fraction", "<=", 0.0, "Teacher disabled"),
                ),
                min_samples=16,
            ),
            min_dwell_s=120.0,
            timeout_s=260.0,
            max_attempts=4,
            auto_retry=True,
            is_terminal=True,
        ),
    ),
    warnings=(
        "External object labels remain forbidden; body-map names are sensor addresses.",
        "Run perception_object_files first when discovered perception is unhealthy.",
    ),
)

SKILLS: dict[str, SkillSpec] = {
    PERCEPTION_OBJECT_FILES.skill_id: PERCEPTION_OBJECT_FILES,
    RESOURCE_ACQUISITION_ENERGY.skill_id: RESOURCE_ACQUISITION_ENERGY,
    STAND_AND_RECOVER.skill_id: STAND_AND_RECOVER,
    DEVELOPMENTAL_LOCOMOTION.skill_id: DEVELOPMENTAL_LOCOMOTION,
    AFFECTIVE_LOCOMOTION.skill_id: AFFECTIVE_LOCOMOTION,
}


def list_skills() -> list[dict]:
    return [s.as_dict() for s in SKILLS.values()]


def get_skill(skill_id: str) -> SkillSpec | None:
    return SKILLS.get(str(skill_id).strip().lower())
