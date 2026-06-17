from decadic.cycle.pipeline import run_cycle
from decadic.cycle.types import CycleContext
from decadic.memory.episodic_store import EpisodicStore
from decadic.state.perceptual_state import PerceptualState
from decadic.state.state_bus import StateBus
from decadic.state.viability import ViabilityState


def test_run_cycle_advances_index_and_emits_action():
    bus = StateBus()
    percept = PerceptualState()
    via = ViabilityState()
    episodic = EpisodicStore(None)
    ctx = CycleContext(
        state_bus=bus,
        perceptual=percept,
        viability=via,
        episodic=episodic,
    )
    out = run_cycle(ctx)
    assert bus.cycle_index == 1
    assert "action" in out
    assert out["action"]["type"] in {"move", "noop"}
    eps = episodic.recent(10)
    assert len(eps) == 1
