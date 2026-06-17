"""Bear/food scene helpers in the MuJoCo adapter (pure parts, no MuJoCo needed)."""

import importlib.util
import sys
from pathlib import Path


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter_scenes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter_scenes"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scene_xml_splices_props():
    mod = _load_adapter_module()
    assert "prop_bear" in mod.scene_xml("bear")
    assert "prop_food_3" in mod.scene_xml("food")
    assert "prop_box_red" in mod.scene_xml("default")
    # The bear stays exclusive to its scene; the food *ring* to the food scene
    assert "prop_bear" not in mod.scene_xml("food")
    assert 'prop_food_1"' not in mod.scene_xml("bear")
    # Shared landmarks appear in every scene: the house and spawn-side snacks
    for scene in ("default", "bear", "food"):
        xml = mod.scene_xml(scene)
        assert "prop_house" in xml
        assert "prop_food_s1" in xml
    # Splice point consumed exactly once and worldbody still closed
    assert mod.scene_xml("bear").count("</worldbody>") == 1


def test_scene_xml_unknown_raises():
    mod = _load_adapter_module()
    try:
        mod.scene_xml("volcano")
    except ValueError as e:
        assert "volcano" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown scene")


def test_prop_kind_labels():
    mod = _load_adapter_module()
    assert mod.prop_kind("prop_bear") == "bear"
    assert mod.prop_kind("prop_food_2") == "food"
    assert mod.prop_kind("prop_box_red") == "box"
    assert mod.prop_kind("prop_sphere_green") == "sphere"
    assert mod.prop_kind("prop_pillar_blue") == "pillar"
    assert mod.prop_kind("prop_house") == "house"
    assert mod.prop_kind("npc") == "npc"
    # The parent's movable gifts are shared food/water so the agent is credited.
    assert mod.prop_kind("prop_food_gift") == "food"
    assert mod.prop_kind("prop_water_gift") == "water"


def test_threat_intensity_scaling():
    mod = _load_adapter_module()
    assert mod.threat_intensity(99.0) is None
    near = mod.threat_intensity(0.5)
    far = mod.threat_intensity(2.8)
    assert near is not None and far is not None
    assert near > far
    assert 0.0 < far <= 1.0
    assert mod.threat_intensity(0.0) == 1.0


def test_eaten_now_radius():
    mod = _load_adapter_module()
    foods = {
        "prop_food_1": [0.5, 0.0, 0.12],
        "prop_food_2": [5.0, 5.0, 0.12],
    }
    root = [0.0, 0.0, 1.4]
    assert mod.eaten_now(root, foods) == ["prop_food_1"]
    # Height difference is ignored; eating is an XY-plane test
    assert mod.eaten_now([5.0, 4.6, 1.4], foods) == ["prop_food_2"]
    assert mod.eaten_now([10.0, 10.0, 1.4], foods) == []
