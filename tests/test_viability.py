from decadic.state.viability import damage_from_events, nourishment_from_events


def test_damage_from_collision_events():
    events = [{"type": "collision", "intensity": 0.8}]
    d = damage_from_events(events, threshold=0.35)
    assert d > 0


def test_damage_ignores_low_intensity():
    events = [{"type": "collision", "intensity": 0.1}]
    assert damage_from_events(events, threshold=0.35) == 0.0


def test_damage_region_change_stress():
    events = [{"type": "region_change", "intensity": 0.5}]
    assert damage_from_events(events, threshold=0.35) > 0


def test_damage_generic_types():
    events = [{"type": "damage", "intensity": 0.8}]
    assert damage_from_events(events, threshold=0.35) > 0


def test_threat_near_mild_stress():
    threat = damage_from_events([{"type": "threat_near", "intensity": 0.8}], threshold=0.35)
    hit = damage_from_events([{"type": "collision", "intensity": 0.8}], threshold=0.35)
    assert 0 < threat < hit


def test_nourishment_from_food_events():
    events = [{"type": "food", "intensity": 1.0}]
    assert nourishment_from_events(events, threshold=0.35) == 6.0


def test_nourishment_respects_threshold_and_types():
    assert nourishment_from_events([{"type": "food", "intensity": 0.1}], threshold=0.35) == 0.0
    assert nourishment_from_events([{"type": "collision", "intensity": 1.0}], threshold=0.35) == 0.0
    # And food never counts as damage
    assert damage_from_events([{"type": "food", "intensity": 1.0}], threshold=0.35) == 0.0
