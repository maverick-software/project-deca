"""VastController._scene_elements: scene resolution for the Deploy/GPU tab.

Locks in the deploy-config unification fix - a Vast deploy must resolve a
scene to the SAME element list decadic.api.presets.store.BUILTIN_PRESETS (and
therefore the local "+ New agent" flow) would give it, instead of a second
hand-maintained bear/food mapping. Also covers the legacy aliases and the raw
comma-separated fallback used by manual/API callers.
"""

from __future__ import annotations

from decadic.api.presets.store import BUILTIN_PRESETS
from decadic.api.vast.controller import VastController
from decadic.api.vast.settings_store import VastSettingsStore


def _controller(tmp_path) -> VastController:
    store = VastSettingsStore(tmp_path / "vast.json")
    return VastController(store)


def _builtin_elements(preset_id: str) -> list[str]:
    for p in BUILTIN_PRESETS:
        if p["id"] == preset_id:
            return list(p["elements"])
    raise KeyError(preset_id)


def test_resolves_every_builtin_preset_id_like_the_local_flow(tmp_path):
    ctrl = _controller(tmp_path)
    for preset in BUILTIN_PRESETS:
        pid = str(preset["id"])
        assert ctrl._scene_elements(pid) == _builtin_elements(pid)


def test_none_and_mind_variants_resolve_to_no_body(tmp_path):
    ctrl = _controller(tmp_path)
    for scene in (None, "", "none", "None", "mind", "mind_only"):
        assert ctrl._scene_elements(scene) == []


def test_legacy_bear_and_food_aliases_still_resolve(tmp_path):
    ctrl = _controller(tmp_path)
    assert ctrl._scene_elements("bear") == _builtin_elements("predator")
    assert ctrl._scene_elements("food") == _builtin_elements("forage")
    # Case-insensitive, like the rest of the method.
    assert ctrl._scene_elements("BEAR") == _builtin_elements("predator")


def test_unknown_value_falls_back_to_raw_element_csv(tmp_path):
    ctrl = _controller(tmp_path)
    assert ctrl._scene_elements("house,water") == ["house", "water"]
    assert ctrl._scene_elements(" house , water ,, ") == ["house", "water"]
