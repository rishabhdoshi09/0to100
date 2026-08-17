"""Company order book = backlog of customer orders, not exchange depth."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

from product.company_order_book import (
    build_company_order_book,
    extract_open_orders,
    parse_as_of_label,
)


EIMCO_PPT_SNIPPET = (
    "Quarterly Performance7068-3.1%151221.6%18.2%-18.3%151421.2%21.4%-2.0%"
    "Revenue (INR Cr) EBITDA (INR Cr) and Margin (%) PAT (INR Cr) and Margin (%)"
    "Q1FY25 Q1FY26 Q1FY25 Q1FY26 Q1FY25 Q1FY26"
    "12539-68.8%Open Order as at 30th June 2025 (INR Cr)Q1FY25 Q1FY26"
)


def test_extracts_open_order_from_jammed_presentation_slide():
    hits = extract_open_orders(EIMCO_PPT_SNIPPET)
    assert hits
    hit = hits[0]
    assert hit["value_cr"] == 39
    assert hit["prior_cr"] == 125
    assert hit["change_pct"] == -68.8
    assert parse_as_of_label(hit["as_of_label"]) == date(2025, 6, 30)


def test_extracts_plain_stood_at_sentence():
    hits = extract_open_orders(
        "The open order book stood at Rs 42 crore as of 30 June 2026."
    )
    assert hits[0]["value_cr"] == 42
    assert parse_as_of_label(hits[0]["as_of_label"]) == date(2026, 6, 30)


def test_does_not_invent_when_text_has_no_backlog():
    assert extract_open_orders("Revenue from operations stood at 77.52 crore.") == []


def test_structured_upload_beats_missing_presentation():
    row = {
        "as_of_date": "2026-06-30",
        "metric": "order_book",
        "value": "55",
        "unit": "INR_cr",
        "management_wording": "Order book stood at about ₹55 crore.",
        "source_url": "https://example.com/filing",
    }
    with patch("product.company_order_book._from_structured", return_value={
        "value_cr": 55.0,
        "prior_cr": None,
        "change_pct": None,
        "as_of_label": "2026-06-30",
        "as_of": "2026-06-30",
        "source": "user_structured_upload",
        "source_url": row["source_url"],
        "wording": row["management_wording"],
    }), patch("product.company_order_book._from_presentations", return_value=None), \
         patch("product.company_order_book._fetch_documents", return_value=[]):
        book = build_company_order_book("EIMCOELECO", as_of=date(2026, 8, 17))
    assert book["available"] is True
    assert book["kind"] == "company_backlog"
    assert book["value_cr"] == 55
    assert book["stale"] is False
    assert "Latest disclosed" in " ".join(book["bullets"])


def test_june_2025_backlog_is_stale_in_aug_2026():
    with patch("product.company_order_book._from_structured", return_value=None), \
         patch("product.company_order_book._fetch_documents", return_value=[]), \
         patch("product.company_order_book._from_presentations", return_value={
             "value_cr": 39.0,
             "prior_cr": 125.0,
             "change_pct": -68.8,
             "as_of_label": "30 June 2025",
             "as_of": "2025-06-30",
             "source": "company_presentation",
             "source_url": "https://www.bseindia.com/example.pdf",
         }):
        book = build_company_order_book(
            "EIMCOELECO",
            ttm_sales_cr=280.0,
            as_of=date(2026, 8, 17),
        )
    assert book["available"] is True
    assert book["stale"] is True
    assert book["value_cr"] == 39
    joined = " ".join(book["bullets"])
    assert "Latest" not in joined
    assert "stale" in joined.lower()
    assert book["coverage_months"] == round(39 / (280 / 12), 1)


def test_missing_backlog_does_not_fall_back_to_exchange_depth():
    with patch("product.company_order_book._from_structured", return_value=None), \
         patch("product.company_order_book._from_presentations", return_value=None), \
         patch("product.company_order_book._fetch_documents", return_value=[]):
        book = build_company_order_book("BSE")
    assert book["available"] is False
    assert book["value_cr"] is None
    assert "bid/ask" in book["note"].lower() or "unexecuted" in book["note"].lower()
    assert "bids" not in book
