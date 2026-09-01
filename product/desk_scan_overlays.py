"""Derive Recommendations and Market Reports from one saved market scan.

Called after the whole-market scan (and long-term overlay) persist. Must stay
fast: no pulse crawl, no StockResearchEngine. Failure must never fail the scan.
GET endpoints read these files cache-only.

The same durable post-scan boundary also captures and settles production
recommendation evidence. Settlement is store-local: official bhavcopy only, no
live quotes and no network requirement. A settlement failure is reported but can
never invalidate a successful market scan.
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
    production_evidence: dict[str, Any] = {
        "status": "skipped",
        "sample_size": 0,
        "evidence_ready": False,
    }
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

        # The ledger call above has frozen this scan's point-in-time candidates.
        # Now settle older same-hash candidates from official bhavcopy. The just-
        # captured scan will normally remain PENDING until enough sessions exist.
        try:
            from product.production_signal_evidence import refresh_production_signal_evidence
            evidence = refresh_production_signal_evidence()
            metrics = dict(evidence.get("metrics") or {})
            production_evidence = {
                "status": "ready" if evidence.get("evidence_ready") else "collecting",
                "sample_size": int(metrics.get("sample_size") or 0),
                "distinct_scan_dates": int(metrics.get("distinct_scan_dates") or 0),
                "pending": int(metrics.get("pending") or 0),
                "no_fill": int(metrics.get("no_fill") or 0),
                "evidence_ready": bool(evidence.get("evidence_ready")),
                "point_in_time_verified": bool(evidence.get("point_in_time_verified")),
                "detail": str(evidence.get("detail") or ""),
            }
        except Exception as exc:
            production_evidence = {
                "status": "error",
                "sample_size": 0,
                "evidence_ready": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
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
        "production_signal_evidence": production_evidence,
        "market_reports": reports_status,
    }
