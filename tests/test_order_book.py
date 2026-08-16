"""Order-book cascade — Kite, then NSE, then Groww public tape. Never invents levels."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from data.order_book import (
    book_from_levels,
    empty_book,
    fetch_groww_tape,
    fetch_order_book,
)


def test_empty_book_has_no_invented_levels():
    book = empty_book(note="blocked", source="nse")
    assert book["available"] is False
    assert book["bids"] == []
    assert book["asks"] == []


def test_book_from_levels_marks_bid_heavy():
    book = book_from_levels(
        [{"price": 100.0, "quantity": 500}, {"price": 99.5, "quantity": 200}],
        [{"price": 100.2, "quantity": 50}],
        source="kite",
        last_price=100.1,
    )
    assert book["available"] is True
    assert book["status"] == "bid_heavy"
    assert book["source"] == "kite"
    assert len(book["bids"]) == 2
    assert book["last_price"] == 100.1


def test_groww_tape_uses_aggregate_qty_not_fake_prices():
    payload = {
        "ltp": 3447.0,
        "totalBuyQty": 165,
        "totalSellQty": 0,
        "lastTradeTime": 1786703369,
        "type": "LIVE_PRICE",
    }
    resp = MagicMock(status_code=200, content=b"{")
    resp.json.return_value = payload
    with patch("data.order_book.requests.get", return_value=resp):
        import data.order_book as ob
        ob._groww_cache.clear()
        book = fetch_groww_tape("BSE")
    assert book["source"] == "groww"
    assert book["available"] is True
    assert book["last_price"] == 3447.0
    assert book["bid_qty"] == 165
    assert book["ask_qty"] == 0
    assert book["bids"] == []
    assert book["asks"] == []
    assert "5-level" in book["note"].lower() or "not nse" in book["note"].lower()


def test_groww_empty_qty_keeps_last_print_and_does_not_claim_a_book():
    payload = {"ltp": 2005.5, "totalBuyQty": 0, "totalSellQty": 0, "lastTradeTime": 1786701597}
    resp = MagicMock(status_code=200, content=b"{")
    resp.json.return_value = payload
    with patch("data.order_book.requests.get", return_value=resp):
        # bypass module cache from the previous test
        import data.order_book as ob
        ob._groww_cache.clear()
        book = fetch_groww_tape("EIMCOELECO")
    assert book["available"] is False
    assert book["last_price"] == 2005.5
    assert book["bids"] == []
    assert "₹2005.5" in book["note"] or "2005.5" in book["note"]


def test_fetch_order_book_prefers_kite_then_falls_to_groww():
    kite_book = book_from_levels(
        [{"price": 10.0, "quantity": 1}],
        [{"price": 10.2, "quantity": 1}],
        source="kite",
        last_price=10.1,
    )
    nse_blocked = empty_book(note="NSE quote HTTP 403", source="nse")
    groww_tape = {
        "available": True, "status": "bid_heavy", "note": "Groww public tape",
        "source": "groww", "bids": [], "asks": [], "bid_qty": 50, "ask_qty": 0,
        "last_price": 10.0, "as_of": "",
    }
    with patch("data.order_book.fetch_kite_depth", return_value=kite_book):
        assert fetch_order_book("BSE")["source"] == "kite"
    with patch("data.order_book.fetch_kite_depth", return_value=empty_book(note="no kite", source="kite")), \
         patch("data.order_book.fetch_nse_depth", return_value=nse_blocked), \
         patch("data.order_book.fetch_groww_tape", return_value=groww_tape):
        out = fetch_order_book("BSE")
    assert out["source"] == "groww"
    assert out["last_price"] == 10.0
    assert out["bids"] == []
