"""WS-SYM 2.2/2.3: entity<->symbol binding by co-occurrence.

The cycle's self-derived FSQ code is bound to the *attended* entity in the
semantic graph, accruing evidence cross-situationally. Meaning lives in the
binding (its evidence_count), not the code's geometry -- so it is robust to
fsq_in drift. Integer codes only (label-firewall safe). See
docs/ws_symbol_integration_analysis.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from decadic.memory.semantic_graph import LongTermGraph


def _slot(entity_id: str, *, focused: bool = False):
    return SimpleNamespace(
        entity_id=entity_id,
        confidence=0.9,
        precision=0.9,
        entity_role="compact_entity",
        kind_hint="object",
        provisional=False,
        seen_count=3,
        property_evidence={},
        affective_weight=0.0,
        attention_focused=focused,
    )


def test_symbol_binds_to_focused_entity():
    g = LongTermGraph(None)
    g.record_semantic_evidence(
        [_slot("e1", focused=True), _slot("e2", focused=False)],
        cycle=1,
        symbol_code=1234,
    )
    sym = g._semantic["symbol"]
    rel = g._semantic["relationship"]
    assert "symbol:1234" in sym  # concept node for the code
    # bound to the FOCUSED entity, not the unfocused one (joint attention)
    assert "relationship:e1:symbol:1234" in rel
    assert "relationship:e2:symbol:1234" not in rel
    assert g.semantic_stats()["symbols"] == 1


def test_symbol_binding_accrues_cross_situationally():
    g = LongTermGraph(None)
    for c in range(1, 6):
        g.record_semantic_evidence([_slot("e1", focused=True)], cycle=c, symbol_code=7)
    rec = g._semantic["relationship"]["relationship:e1:symbol:7"]
    assert rec["evidence_count"] > 1.0  # accumulated over 5 co-occurrences
    # one concept node, one binding -- not a fresh record each cycle
    assert len(g._semantic["symbol"]) == 1


def test_no_symbol_code_is_parity():
    g = LongTermGraph(None)
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=1)  # no symbol_code
    assert len(g._semantic["symbol"]) == 0
    assert "symbols" in g.semantic_stats() and g.semantic_stats()["symbols"] == 0


def test_recall_entity_symbol_ranks_by_evidence():
    """WS-SYM 3.1: recall returns the strongest-bound code for an entity, and
    the reverse lookup returns the entity for a code. Pure read."""
    g = LongTermGraph(None)
    # e1 co-occurs with code 7 five times, code 9 twice -> 7 should rank first.
    for c in range(1, 6):
        g.record_semantic_evidence([_slot("e1", focused=True)], cycle=c, symbol_code=7)
    for c in range(6, 8):
        g.record_semantic_evidence([_slot("e1", focused=True)], cycle=c, symbol_code=9)
    ranked = g.recall_entity_symbol("e1", top_k=3)
    assert [code for code, _ev in ranked] == [7, 9]
    assert ranked[0][1] > ranked[1][1]  # code 7 has more evidence
    # reverse: code 7 -> e1
    assert g.entities_for_symbol(7)[0][0] == "e1"


def test_recall_hit_telemetry_counts_prior_bindings():
    """WS-SYM 3.2: recall-hit telemetry -- first time an entity is bound is a
    miss (no prior code), subsequent bindings are hits. A climbing hit-rate is
    the live proof recall works."""
    g = LongTermGraph(None)
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=1, symbol_code=7)  # miss
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=2, symbol_code=7)  # hit
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=3, symbol_code=9)  # hit
    assert g._symbol_recall_queries == 3
    assert g._symbol_recall_hits == 2  # cycles 2 and 3 had a prior binding


def test_fsq_code_to_vector_roundtrips():
    """WS-SYM 3.3: decode(index) == the quantized vector encode produced, so a
    recalled code feeds the trunk as the exact grid point."""
    torch = pytest.importorskip("torch")
    from decadic.nn.symbol import FSQ_DIMS, fsq_code_to_vector, fsq_quantize

    x = torch.randn(4, FSQ_DIMS, generator=torch.Generator().manual_seed(3))
    q, idx = fsq_quantize(x)
    for b in range(4):
        vec = fsq_code_to_vector(int(idx[b].item()))
        assert len(vec) == FSQ_DIMS
        for d in range(FSQ_DIMS):
            assert abs(vec[d] - float(q[b, d].item())) < 1e-5


def test_binding_churn_and_recalled_vec(monkeypatch):
    """WS-SYM 3.3/5.0: the focused entity's TOP code is stashed as a vector
    (recall-conditioned feedback) and top-code flips are counted (drift proxy)."""
    from decadic.nn.symbol import FSQ_DIMS

    g = LongTermGraph(None)
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=1, symbol_code=7)
    assert g._last_recalled_symbol_code == 7
    assert g._last_recalled_symbol_vec is not None and len(g._last_recalled_symbol_vec) == FSQ_DIMS
    assert g._symbol_binding_updates == 1 and g._symbol_binding_flips == 0
    # code 9 ties then exceeds 7's evidence -> top flips to 9 (churn).
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=2, symbol_code=9)  # 9=1, 7=1 -> top stays 7
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=3, symbol_code=9)  # 9=2 > 7=1 -> flip
    assert g._last_recalled_symbol_code == 9
    assert g._symbol_binding_flips == 1
    assert g._symbol_binding_updates == 3


def test_recall_empty_for_unbound_entity():
    g = LongTermGraph(None)
    g.record_semantic_evidence([_slot("e1", focused=True)], cycle=1, symbol_code=7)
    assert g.recall_entity_symbol("never_seen") == []
    assert g.entities_for_symbol(9999) == []


def test_symbol_falls_back_to_copresent_when_no_focus():
    """With no attention-focused slot, bind to the co-present set (capped) so
    cross-situational aggregation can still disambiguate over time."""
    g = LongTermGraph(None)
    g.record_semantic_evidence(
        [_slot("a"), _slot("b"), _slot("c"), _slot("d")],  # none focused
        cycle=1,
        symbol_code=42,
    )
    bound = [k for k in g._semantic["relationship"] if k.endswith(":symbol:42")]
    assert 1 <= len(bound) <= 3  # capped at 3 targets
