"""Composable scene elements, legacy scene wrapper, and adapter CLI parsing."""

import importlib.util
import sys
from pathlib import Path

from decadic.api.environment import VALID_ELEMENTS


def _load_adapter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_decadic_adapter.py"
    spec = importlib.util.spec_from_file_location("mujoco_decadic_adapter_env", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mujoco_decadic_adapter_env"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_compose_scene_includes_only_selected_elements():
    mod = _load_adapter_module()

    bear_only = mod.compose_scene(["bear"])
    assert "prop_bear" in bear_only
    assert "prop_house" not in bear_only
    assert "prop_water_w1" not in bear_only

    calm = mod.compose_scene(["house", "food", "water"])
    assert "prop_house" in calm
    assert "prop_food_s1" in calm
    assert "prop_water_w1" in calm
    assert "prop_bear" not in calm

    assert "prop_ball" in mod.compose_scene(["ball"])
    assert "prop_box_red" in mod.compose_scene(["obstacles"])

    # Each composition splices the worldbody exactly once.
    for elements in (["bear"], ["house", "food", "water"], ["ball"], []):
        assert mod.compose_scene(elements).count("</worldbody>") == 1


def test_compose_scene_includes_npc():
    mod = _load_adapter_module()
    npc = mod.compose_scene(["npc"])
    # The parent humanoid (prefixed) and its movable food + water gifts splice in.
    assert 'name="npc_torso"' in npc
    assert 'name="npc_root"' in npc
    assert "prop_food_gift" in npc
    assert "prop_water_gift" in npc
    # The parent carries no actuators of its own (those live outside the
    # worldbody splice), so it adds zero motors over the bare model, and the
    # worldbody is still closed exactly once.
    assert npc.count("<motor ") == mod.compose_scene([]).count("<motor ")
    assert npc.count("</worldbody>") == 1
    # A scene without the parent has no npc bodies.
    assert "npc_torso" not in mod.compose_scene(["house", "food", "water"])


def test_compose_scene_dedupes_and_ignores_unknown():
    mod = _load_adapter_module()
    xml = mod.compose_scene(["house", "house", "volcano", "bear"])
    # de-duplicated: the house body appears once
    assert xml.count('name="prop_house"') == 1
    assert "prop_bear" in xml
    # unknown element contributes nothing and does not error
    no_props = mod.compose_scene(["volcano", "atlantis"])
    assert "prop_bear" not in no_props
    assert no_props.count("</worldbody>") == 1


def test_legacy_scene_xml_still_resolves():
    mod = _load_adapter_module()
    assert "prop_bear" in mod.scene_xml("bear")
    assert "prop_food_3" in mod.scene_xml("food")  # ring food, legacy-only
    assert "prop_box_red" in mod.scene_xml("default")
    assert "prop_bear" not in mod.scene_xml("food")

    try:
        mod.scene_xml("volcano")
    except ValueError as e:
        assert "volcano" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown scene")


def test_selectable_elements_match_supervisor():
    mod = _load_adapter_module()
    assert set(mod.SELECTABLE_ELEMENTS) == set(VALID_ELEMENTS)


def test_arg_parser_agent_id_and_scenario():
    mod = _load_adapter_module()
    parser = mod.build_arg_parser()

    args = parser.parse_args(["--agent-id", "abc-123", "--scenario", "house,water,bear"])
    assert args.agent_id == "abc-123"
    assert args.scenario == "house,water,bear"
    assert args.scene == "default"

    defaults = parser.parse_args([])
    assert defaults.agent_id is None
    assert defaults.scenario is None
    # Vision is ON by default for every scenario; --no-vision is the opt-out.
    assert defaults.vision is True
    assert parser.parse_args(["--no-vision"]).vision is False
    assert parser.parse_args(["--vision"]).vision is True
