"""Derive Recommendations and Market Reports from one saved market scan.

Called after the whole-market scan (and long-term overlay) persist. Must stay
fast: no pulse crawl, no StockResearchEngine. Failure must never fail the scan.
GET endpoints read these files cache-only.
"""
from __future__ import annotations

from typing import Any, Mapping


def persist_desks_from_market_scan(scan_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    scan = dict(scan_payload or {})
    lt: dict[str, Any] = {}
    try:
        from product.long_term_store import load_long_term_scan
        lt = load_long_term_scan() or {}
    except Exception:
        lt = {}
    reco_status = "skipped"
    reco_cards = 0
    try:
        from product.recommendations_store import save_recommendations
        from product.recommendations_workspace import (
            build_recommendations_workspace,
            slim_workspace_for_desk,
        )
        reco = build_recommendations_workspace(
            scan_payload=scan,
            long_term_payload=lt,
            refresh_technicals=False,
            settle_cases=False,
            deep_confirm=False,
            persist_ledger=True,
        )
        slim = slim_workspace_for_desk(reco)
        save_recommendations(slim)
        reco_status = "saved"
        reco_cards = int((slim.get("scan_meta") or {}).get("assigned_count") or 0)
    except Exception as exc:
        reco_status = type(exc).__name__
    reports_status = "skipped"
    try:
        from product.recommendations_workspace import build_market_reports_workspace
        news: dict[str, Any] = {}
        build_market_reports_workspace(
            persist_today=True,
            news_payload=news,
            scan_payload=scan,
            rebuild=False,
        )
        reports_status = "saved"
    except Exception as exc:
        reports_status = type(exc).__name__
    return {
        "recommendations": reco_status,
        "recommendation_cards": reco_cards,
        "market_reports": reports_status,
    }
