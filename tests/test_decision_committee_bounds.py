from __future__ import annotations

from product import decision_committee as dc


def _record(symbol: str) -> dc.CommitteeRecord:
    return dc.CommitteeRecord(
        symbol=symbol,
        decision="WAIT",
        candidate_state="WAIT",
        entry_state="WAIT_FOR_ENTRY",
        execution_state="NOT_APPLICABLE",
        reason_code="TEST",
        reason="test",
    )


def test_evaluate_many_is_bounded_by_default(monkeypatch):
    seen: list[str] = []

    def fake_evaluate(card, **_kwargs):
        symbol = str(card["symbol"])
        seen.append(symbol)
        return _record(symbol)

    monkeypatch.setattr(dc, "evaluate_committee", fake_evaluate)
    cards = [{"symbol": f"S{i:02d}"} for i in range(dc.MAX_COMMITTEE_CANDIDATES + 9)]

    records = dc.evaluate_many(cards)

    assert len(records) == dc.MAX_COMMITTEE_CANDIDATES
    assert seen == [card["symbol"] for card in cards[: dc.MAX_COMMITTEE_CANDIDATES]]


def test_evaluate_many_can_be_unbounded_only_when_explicit(monkeypatch):
    monkeypatch.setattr(dc, "evaluate_committee", lambda card, **_kwargs: _record(str(card["symbol"])))
    cards = [{"symbol": f"S{i:02d}"} for i in range(dc.MAX_COMMITTEE_CANDIDATES + 3)]

    records = dc.evaluate_many(cards, max_records=None)

    assert len(records) == len(cards)


def test_current_research_snapshot_does_not_build_research_without_durable_facts(monkeypatch):
    monkeypatch.setattr(
        "product.due_diligence.acquire.load_autonomy_facts",
        lambda _symbol: {},
    )

    class BombResearchEngine:
        def __init__(self):
            raise AssertionError("committee must not build research before durable facts exist")

    monkeypatch.setattr(
        "product.due_diligence.research_engine.StockResearchEngine",
        BombResearchEngine,
    )

    result = dc._research_snapshot("TCS", {})

    assert result["available"] is False
    assert result["cached_only"] is True
    assert result["reason"] == "autonomy_facts_missing"


def test_committee_record_declares_deterministic_no_llm_contract():
    payload = _record("TCS").as_dict()

    assert payload["decision_schema_version"] == 1
    assert payload["decision_engine"] == "deterministic_committee"
    assert payload["deterministic"] is True
    assert payload["llm_required"] is False
