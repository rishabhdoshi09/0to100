"""The production paper path carries the decision that opened the trade.

Without this the learning loop works only in its own tests: the book records a
position, the position closes, a statistic moves, and nothing in the chain can
say which decision it came from. A statistic whose trades cannot be listed is
not checkable, and an unattributable outcome must not become evidence at all.
"""
from __future__ import annotations

import pytest

from product.conditional_evidence import read as read_cell
from product.paper_autopilot import AutopilotDecision, _intent_for
from product.paper_learning_loop import ingest_closed_trade, record_conditional_evidence
from research.auto_research.paper_book import ClosedTrade, PaperBook, PaperPosition


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))
    monkeypatch.setenv("QT_TAKEN_EVIDENCE", str(tmp_path / "taken.jsonl"))
    monkeypatch.setenv("QT_LEARNING_INGESTED", str(tmp_path / "ingested.json"))


def _card(**overrides):
    card = {
        "symbol": "INFY",
        "reco_tier": "high_conviction",
        "primary_thesis": "VCP",
        "entry": 100.0,
        "stop": 95.0,
        "target": 115.0,
        "allows_recommend": True,
        "approved_quantity": 10,
        "atr_pct": 2.5,
        "pct_from_pivot": 1.0,
        "market_state": "HEALTHY",
        "sector_state": "LEADING",
        "scan_scanned_at": "2026-09-12T03:00:00+00:00",
    }
    card.update(overrides)
    return card


# ── the intent ─────────────────────────────────────────────────────────────
def test_the_live_intent_names_its_decision_and_its_cell():
    intent = _intent_for(
        AutopilotDecision("INFY", "ENTER_NOW", "ELIGIBLE", "ok", _card()),
        as_of="2026-09-12", snapshot_id="snap1",
    )
    assert intent.decision_id.startswith("dec_")
    assert "setup=VCP" in intent.context_key
    assert "regime=HEALTHY" in intent.context_key
    assert intent.record_id, "the intent keeps its own deterministic identity"


def test_a_card_that_cannot_be_canonicalised_never_stops_the_paper_cycle():
    """No linkage is a missing attribution, not a crash mid-cycle."""
    intent = _intent_for(
        AutopilotDecision("", "ENTER_NOW", "ELIGIBLE", "ok", {"symbol": ""}),
        as_of="2026-09-12", snapshot_id="snap1",
    )
    assert intent.decision_id == ""
    assert intent.context_key == ""


# ── the book ───────────────────────────────────────────────────────────────
def test_the_book_carries_the_chain_from_open_to_close():
    book = PaperBook(capital=100_000.0)
    position = book.open_position(
        "s1", "INFY", 100.0, 95.0, 115.0, "2026-09-12", 10,
        decision_id="dec_x", paper_intent_id="int_x", context_key="setup=VCP",
    )
    assert position is not None
    assert position.decision_id == "dec_x"

    closed = book.mark({"INFY": (100.0, 120.0, 90.0, 118.0)}, "2026-09-15")
    assert closed, "the position should have resolved"
    assert closed[0].decision_id == "dec_x"
    assert closed[0].paper_intent_id == "int_x"
    assert closed[0].context_key == "setup=VCP"


def test_the_chain_survives_a_snapshot_and_restore():
    """A restart must not drop the linkage: the book is reloaded from disk."""
    book = PaperBook(capital=100_000.0)
    book.open_position("s1", "INFY", 100.0, 95.0, 115.0, "2026-09-12", 10,
                       decision_id="dec_x", paper_intent_id="int_x",
                       context_key="setup=VCP")
    restored = PaperBook(capital=100_000.0)
    restored.restore(book.snapshot())
    position = next(iter(restored.open.values()))
    assert position.decision_id == "dec_x"
    assert position.context_key == "setup=VCP"


def test_a_book_saved_before_the_linkage_existed_still_restores():
    legacy = {
        "capital": 100_000.0, "realized_pnl": 0.0, "equity_curve": [100_000.0],
        "closed": [], "open": [{
            "strategy_id": "s1", "symbol": "AAA", "entry_price": 10.0,
            "stop_price": 9.0, "target_price": 12.0, "qty": 1,
            "entry_date": "2026-01-01", "max_holding_days": 5, "risk_amount": 1.0,
        }],
    }
    book = PaperBook()
    book.restore(legacy)
    position = next(iter(book.open.values()))
    assert position.decision_id == ""
    assert position.context_key == ""


def test_research_simulator_positions_carry_no_linkage():
    """They have no canonical decision, so they must claim none."""
    book = PaperBook(capital=100_000.0)
    position = book.open_position("sim", "AAA", 10.0, 9.0, 12.0, "2026-09-12", 5)
    assert position.decision_id == ""
    assert position.context_key == ""


# ── settlement ─────────────────────────────────────────────────────────────
def _closed_row(**overrides):
    row = {
        "symbol": "INFY", "realized_R": -1.0, "exit_reason": "STOP",
        "entry_date": "2026-09-12", "exit_date": "2026-09-20",
        "entry_price": 100.0, "exit_price": 95.0,
        "decision_id": "dec_abc", "paper_intent_id": "int_abc",
        "context_key": "setup=VCP|regime=HEALTHY",
    }
    row.update(overrides)
    return row


def test_a_settled_linked_trade_updates_its_cell():
    update = record_conditional_evidence(_closed_row())
    assert update["changed"] is True
    assert update["before"]["count"] == 0
    assert update["after"]["count"] == 1
    assert read_cell("setup=VCP|regime=HEALTHY")["count"] == 1


@pytest.mark.parametrize("missing", ["decision_id", "paper_intent_id", "context_key"])
def test_an_unattributable_outcome_is_not_recorded(missing):
    assert record_conditional_evidence(_closed_row(**{missing: ""})) is None
    assert read_cell("setup=VCP|regime=HEALTHY")["count"] == 0


def test_an_unresolved_trade_is_not_recorded():
    assert record_conditional_evidence(_closed_row(realized_R=None)) is None


def test_settling_the_same_trade_twice_counts_it_once():
    row = _closed_row()
    record_conditional_evidence(row)
    record_conditional_evidence(row)
    assert read_cell("setup=VCP|regime=HEALTHY")["count"] == 1


def test_the_policy_row_can_name_the_decision_behind_it():
    result = ingest_closed_trade(_closed_row())
    assert result is not None
    chain = result.get("conditional_evidence") or {}
    assert chain.get("decision_id") == "dec_abc"
    assert chain.get("evidence_update_id", "").startswith("evu_")


def test_an_unlinked_trade_still_updates_the_old_policy_ladder():
    """The linkage gates conditional evidence, never the existing learning."""
    result = ingest_closed_trade(_closed_row(decision_id="", context_key=""))
    assert result is not None
    assert "conditional_evidence" not in result
