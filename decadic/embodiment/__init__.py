"""Embodiment helpers for the MuJoCo body adapter.

Pure, unit-testable building blocks for the scripted NPC *crowd* (a small
"village" of kinematically-animated demonstrators confined to their own
habitats around the learner). Kept out of ``scripts/mujoco_decadic_adapter.py``
so that file does not keep growing and so the crowd logic can be tested without
spinning up the whole body process.

Modules:
- :mod:`decadic.embodiment.habitats` - habitat descriptors, env-var config
  accessors, and zone-clamp math.
- :mod:`decadic.embodiment.npc_behaviors` - the pure kinematic pose/behavior
  library (walk / stand / sit / sit-stand / communicate poses).
- :mod:`decadic.embodiment.npc_xml` - multi-NPC humanoid XML generation,
  habitat resources, and zone markers spliced into the worldbody.
- :mod:`decadic.embodiment.npc_controller` - the MuJoCo-bound ``CrowdController``
  that animates the crowd, runs per-zone behavior/forage FSMs, and drives the
  need-threshold parental provisioning.
"""

from __future__ import annotations

from decadic.embodiment.habitats import (
    BEHAVIOR_COMMUNICATE,
    BEHAVIOR_FORAGE,
    BEHAVIOR_IDLE,
    BEHAVIOR_SIT,
    BEHAVIOR_SIT_STAND,
    BEHAVIOR_WANDER,
    DEFAULT_HABITATS,
    Habitat,
    active_habitats,
    clamp_to_zone,
    crowd_lod_distance,
    crowd_size,
    habitat_radius,
    parent_fade_per_offer,
    parent_need_threshold,
    parent_refractory_s,
    parent_threshold_floor,
)

__all__ = [
    "BEHAVIOR_COMMUNICATE",
    "BEHAVIOR_FORAGE",
    "BEHAVIOR_IDLE",
    "BEHAVIOR_SIT",
    "BEHAVIOR_SIT_STAND",
    "BEHAVIOR_WANDER",
    "DEFAULT_HABITATS",
    "Habitat",
    "active_habitats",
    "clamp_to_zone",
    "crowd_lod_distance",
    "crowd_size",
    "habitat_radius",
    "parent_fade_per_offer",
    "parent_need_threshold",
    "parent_refractory_s",
    "parent_threshold_floor",
]
