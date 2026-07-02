"""Multi-NPC humanoid XML generation for the scripted crowd.

Generates N prefixed humanoids (cloned from the agent asset) placed at habitat
centers, plus co-located respawning food/water and optional zone markers, ready
to splice into the worldbody. The clones:

- prefix every name (``npc{idx}_``) so the agent's discovery loops skip them and
  there are no name collisions,
- drop cameras and touch sites (no eye, no sensors),
- declare no actuators/sensors, so ``model.nu`` stays at the agent's 21 and the
  brain's motor contract is unchanged (the bodies are driven kinematically),
- stiffen hinges so limbs hold a pose passively, and
- are **collisionless** (``contype=0 conaffinity=0``): 8 extra bodies cost render
  but not contact solving, and they never push the learner's own physical body.

Exactly one clone (the parent) carries movable "gift" props it can drop for
the learner; the gift names contain ``food``/``water``/``medical`` (so the
shared resource machinery credits the *agent* when it uses them) and ``gift``
(so NPC foraging skips them).
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from decadic.embodiment.habitats import Habitat, active_habitats

# Movable parent gifts (distinct from the legacy single-parent "prop_*_gift" so
# the crowd and the legacy "npc" element can never collide on a body name).
GIFT_NAMES: dict[str, str] = {
    "food": "prop_food_gift_c",
    "water": "prop_water_gift_c",
    "medical_kit": "prop_medical_gift_c",
}


def _load_torso(asset_path: Path) -> ET.Element:
    """Parse the agent asset and return a fresh ``<body name='torso'>`` element."""
    # Strip XML comments first: the asset contains "(--scene)" inside a comment,
    # which MuJoCo tolerates but strict ElementTree rejects ("--" in comments).
    raw = re.sub(
        r"<!--.*?-->", "", asset_path.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    torso = ET.fromstring(raw).find("./worldbody/body[@name='torso']")
    if torso is None:  # pragma: no cover - asset ships with the repo
        raise RuntimeError("agent asset is missing <body name='torso'>")
    return torso


def clone_humanoid_xml(
    asset_path: Path,
    prefix: str,
    pos: tuple[float, float, float],
    *,
    collisionless: bool = True,
    tint_rgba: tuple[float, float, float, float] | None = None,
) -> str:
    """One prefixed, actuator-free, optionally-collisionless humanoid clone."""
    torso = _load_torso(asset_path)

    def _rewrite(el: ET.Element) -> None:
        for child in list(el):
            if child.tag in ("camera", "site"):
                el.remove(child)
            else:
                _rewrite(child)
        if "name" in el.attrib:
            el.attrib["name"] = prefix + el.attrib["name"]
        if el.tag == "joint" and el.attrib.get("type", "hinge") != "free":
            el.attrib["stiffness"] = "60"
            el.attrib["damping"] = "6"
        if el.tag == "geom" and collisionless:
            el.attrib["contype"] = "0"
            el.attrib["conaffinity"] = "0"
        if el.tag == "geom" and tint_rgba is not None:
            alpha = tint_rgba[3]
            if "rgba" in el.attrib:
                parts = el.attrib["rgba"].split()
                if len(parts) >= 4:
                    try:
                        alpha = float(parts[3])
                    except ValueError:
                        alpha = tint_rgba[3]
            el.attrib["rgba"] = f"{tint_rgba[0]} {tint_rgba[1]} {tint_rgba[2]} {alpha}"

    _rewrite(torso)
    torso.attrib["pos"] = f"{pos[0]} {pos[1]} {pos[2]}"
    return ET.tostring(torso, encoding="unicode")


def parent_gifts_xml(pos: tuple[float, float]) -> str:
    """The parent's movable (collidable) gift props near ``pos``."""
    gx, gy = pos
    food = (
        f'\n    <body name="{GIFT_NAMES["food"]}" pos="{gx} {gy} 0.12">'
        f'\n      <joint name="{GIFT_NAMES["food"]}_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="{GIFT_NAMES["food"]}_geom" type="sphere" size="0.12"'
        f' rgba="0.95 0.55 0.1 1" condim="3" mass="0.2"/>'
        f"\n    </body>"
    )
    water = (
        f'\n    <body name="{GIFT_NAMES["water"]}" pos="{gx + 0.4} {gy} 0.12">'
        f'\n      <joint name="{GIFT_NAMES["water"]}_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="{GIFT_NAMES["water"]}_geom" type="sphere" size="0.12"'
        f' rgba="0.2 0.5 0.95 1" condim="3" mass="0.2"/>'
        f"\n    </body>"
    )
    medical = (
        f'\n    <body name="{GIFT_NAMES["medical_kit"]}" pos="{gx + 0.8} {gy} 0.13">'
        f'\n      <joint name="{GIFT_NAMES["medical_kit"]}_free" type="free" limited="false"'
        f' armature="0" damping="0.4"/>'
        f'\n      <geom name="{GIFT_NAMES["medical_kit"]}_box" type="box" size="0.18 0.11 0.065"'
        f' rgba="0.96 0.96 0.92 1" condim="3" mass="0.2"/>'
        f'\n      <geom name="{GIFT_NAMES["medical_kit"]}_handle" type="box" pos="0 0 0.12"'
        f' size="0.11 0.018 0.025" rgba="0.08 0.10 0.12 1" contype="0" conaffinity="0"/>'
        f'\n      <geom name="{GIFT_NAMES["medical_kit"]}_cross_a" type="box" pos="0 0 0.07"'
        f' size="0.035 0.088 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>'
        f'\n      <geom name="{GIFT_NAMES["medical_kit"]}_cross_b" type="box" pos="0 0 0.084"'
        f' size="0.09 0.035 0.011" rgba="0.88 0.03 0.07 1" contype="0" conaffinity="0"/>'
        f"\n    </body>"
    )
    return food + water + medical + "\n"


def _ring_offsets(n: int, radius: float, phase: float = 0.0) -> list[tuple[float, float]]:
    """``n`` points evenly spaced on a circle of ``radius`` (small inner ring)."""
    if n <= 0:
        return []
    return [
        (
            radius * math.cos(phase + 2.0 * math.pi * i / n),
            radius * math.sin(phase + 2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def habitat_resources_xml(habitats: list[Habitat]) -> str:
    """Co-located respawning food/water scattered inside each habitat.

    Names contain ``food``/``water`` (registered by the adapter's consumable
    discovery) and a per-habitat suffix, so they are net-additive to the
    learner's own supply and respawn via the shared machinery.
    """
    parts = ["\n    <!-- Crowd habitat resources (net-additive, respawning) -->"]
    for hi, hab in enumerate(habitats):
        cx, cy = hab.center
        r = max(0.4, hab.radius * 0.55)
        for n, (ox, oy) in enumerate(_ring_offsets(hab.food, r, phase=0.3 * hi), 1):
            x, y = round(cx + ox, 2), round(cy + oy, 2)
            parts.append(
                f'\n    <body name="prop_food_h{hi}_{n}" pos="{x} {y} 0.12">'
                f'\n      <geom name="prop_food_h{hi}_{n}_geom" type="sphere" size="0.12"'
                f' rgba="0.95 0.55 0.1 1" condim="3"/>'
                f"\n    </body>"
            )
        for n, (ox, oy) in enumerate(_ring_offsets(hab.water, r * 0.6, phase=1.1 * hi), 1):
            x, y = round(cx + ox, 2), round(cy + oy, 2)
            parts.append(
                f'\n    <body name="prop_water_h{hi}_{n}" pos="{x} {y} 0.13">'
                f'\n      <geom name="prop_water_h{hi}_{n}_geom" type="cylinder"'
                f' size="0.08 0.13" rgba="0.3 0.55 0.95 0.7" condim="3"/>'
                f"\n    </body>"
            )
    return "".join(parts) + "\n"


def zone_markers_xml(habitats: list[Habitat]) -> str:
    """Thin collisionless floor discs marking each habitat (pure decor)."""
    parts = ["\n    <!-- Crowd habitat markers (collisionless decor) -->"]
    palette = [
        "0.85 0.75 0.45 0.35", "0.55 0.75 0.85 0.35", "0.75 0.55 0.85 0.35",
        "0.6 0.85 0.6 0.35", "0.85 0.6 0.6 0.35", "0.85 0.6 0.6 0.35",
        "0.7 0.7 0.7 0.35", "0.8 0.8 0.5 0.35",
    ]
    for hi, hab in enumerate(habitats):
        cx, cy = hab.center
        rgba = palette[hi % len(palette)]
        parts.append(
            f'\n    <geom name="prop_zone_{hi}" type="cylinder" pos="{cx} {cy} 0.02"'
            f' size="{round(hab.radius, 2)} 0.02" rgba="{rgba}"'
            f' contype="0" conaffinity="0"/>'
        )
    return "".join(parts) + "\n"


def crowd_scene_xml(asset_path: Path, habitats: list[Habitat] | None = None) -> str:
    """Full crowd snippet: N humanoids + parent gifts + resources + markers."""
    habs = habitats if habitats is not None else active_habitats()
    parts = ["\n    <!-- Scripted NPC crowd: confined kinematic demonstrators -->"]
    for hi, hab in enumerate(habs):
        cx, cy = hab.center
        parts.append(
            clone_humanoid_xml(
                asset_path,
                f"npc{hi}_",
                (cx, cy, 1.30),
                collisionless=True,
                tint_rgba=(0.9, 0.82, 0.35, 1.0) if hab.is_parent else None,
            )
        )
        if hab.is_parent:
            parts.append(parent_gifts_xml((cx, cy + 0.6)))
    parts.append(habitat_resources_xml(habs))
    parts.append(zone_markers_xml(habs))
    return "\n    ".join(parts) + "\n"
