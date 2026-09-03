"""Official corporate-event classification from dated warehouse rows.

Does not synthesize media coverage. Headlines without a proven
publication date never become events. Classification is keyword-based
and conservative — unknown stays OTHER.
"""
from __future__ import annotations

from typing import Any

from product.pit_warehouse import (
    DOC_CORPORATE_ANNOUNCEMENT,
    DOC_CREDIT_RATING,
    DOC_EXCHANGE_FILING,
    DOC_QUARTERLY_RESULT,
    DOC_SHAREHOLDING_PATTERN,
    get_evidence,
)

RESULTS = "RESULTS"
CORPORATE_ACTION = "CORPORATE_ACTION"
SHAREHOLDING = "SHAREHOLDING"
FUND_RAISING = "FUND_RAISING"
LARGE_ORDER = "LARGE_ORDER"
MANAGEMENT = "MANAGEMENT"
RATING = "RATING"
MERGER = "MERGER"
SPLIT_BONUS = "SPLIT_BONUS"
PLEDGE = "PLEDGE"
MATERIAL = "MATERIAL"
OTHER = "OTHER"

_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (RESULTS, ("financial result", "audited result", "unaudited result", "quarterly result", "annual result")),
    (SPLIT_BONUS, ("bonus", "split", "sub-division", "subdivision", "stock split")),
    (MERGER, ("merger", "demerger", "amalgamation", "scheme of arrangement")),
    (FUND_RAISING, ("qip", "preferential", "rights issue", "fund raise", "qualified institutional")),
    (PLEDGE, ("pledge", "invoke", "revocation of pledge")),
    (SHAREHOLDING, ("shareholding pattern", "promoter holding")),
    (RATING, ("credit rating", "rating action", "outlook revised", "care ratings", "crisil", "icra")),
    (MANAGEMENT, ("appointment of", "resignation of", "cessation of", "managing director", "cfo ", "ceo ")),
    (LARGE_ORDER, ("order win", "letter of award", "contract worth", "purchase order")),
    (CORPORATE_ACTION, ("dividend", "buyback", "record date", "book closure")),
    (MATERIAL, ("regulation 30", "material event", "disclosure under")),
)


def classify_headline(text: str) -> str:
    blob = str(text or "").lower()
    if not blob.strip():
        return OTHER
    for label, needles in _RULES:
        if any(n in blob for n in needles):
            return label
    return OTHER


def get_events(symbol: str, *, as_of: str, path=None, limit: int = 40) -> list[dict[str, Any]]:
    """Events whose available_from <= T. Future announcements are invisible."""
    rows = get_evidence(
        symbol,
        as_of=as_of,
        evidence_types=(
            DOC_CORPORATE_ANNOUNCEMENT,
            DOC_EXCHANGE_FILING,
            DOC_QUARTERLY_RESULT,
            DOC_CREDIT_RATING,
            DOC_SHAREHOLDING_PATTERN,
        ),
        path=path,
    )
    out = []
    for row in rows[: max(1, int(limit))]:
        extracted = dict(row.get("extracted") or {})
        headline = str(extracted.get("headline") or row.get("evidence_type") or "")
        kind = classify_headline(headline)
        if row.get("evidence_type") == DOC_QUARTERLY_RESULT:
            kind = RESULTS
        elif row.get("evidence_type") == DOC_CREDIT_RATING:
            kind = RATING
        elif row.get("evidence_type") == DOC_SHAREHOLDING_PATTERN:
            kind = SHAREHOLDING
        out.append({
            "symbol": str(symbol).upper(),
            "as_of": str(as_of)[:10],
            "event_class": kind,
            "headline": headline,
            "available_from": row.get("available_from"),
            "source": row.get("source"),
            "source_url": row.get("source_url"),
            "evidence_id": row.get("evidence_id"),
            "not_media_coverage": True,
        })
    return out
