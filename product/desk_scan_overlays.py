"""Derive durable Recommendations and Market Reports from one saved market scan.

A successful whole-market scan is the production boundary for the retail desk.
The existing ensemble first nominates candidates. Selection Authority then uses
already-measured learning plus cache/file-only Due Diligence on the small finalist
set. Only that post-gate state is persisted and journaled.

No pulse crawl and no new scanner are introduced here. A DD/evidence failure may
block a recommendation but must never invalidate the successful market scan.
"""
from __future__ import annotations

from typing import Any, Mapping


def _workspace_cards(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for category in payload.get("categories") or []:
        if not isinstance(category, Mapping):
            continue
        cards.extend(dict(card) for card in (category.get("cards") or []) if isinstance(card, Mapping))
    return cards


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
    selection_summary: dict[str, Any] = {"applied": False}
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
        from product.selection_authority import apply_workspace_selection_authority

        # Build first without journaling. The ledger must record the FINAL decision
        # after learning + DD, not the pre-DD nomination.
        reco = build_recommendations_workspace(
            scan_payload=scan,
            long_term_payload=lt,
            refresh_technicals=False,
            settle_cases=False,
            deep_confirm=False,
            persist_ledger=False,
        )
        reco = apply_workspace_selection_authority(reco, max_due_diligence=8)
        selection_summary = dict(reco.get("selection_authority") or {})

        final_cards = _workspace_cards(reco)
        try:
            from product.reco_ledger import append_recommendations
            append_recommendations(final_cards, scan_scanned_at=str(scan.get("scanned_at") or ""))
        except Exception:
            # Ledger failure must be visible through evidence status later, but must
            # never erase the successfully built desk.
            pass

        slim = slim_workspace_for_desk(reco)
        save_recommendations(slim)
        reco_status = "saved"
        reco_cards = len(final_cards)

        # Settle older same-hash final recommendations from official bhavcopy.
        try:
            from product.production_signal_evidence import (
                build_production_signal_evidence,
                save_production_signal_evidence,
            )
            evidence = build_production_signal_evidence()
            evidence["scope"] = (
                "PRODUCTION_SIGNAL_OUTCOMES"
                if evidence.get("evidence_ready")
                else "COLLECTING_PRODUCTION_SIGNAL_OUTCOMES"
            )
            save_production_signal_evidence(evidence)
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
        reco_status = f"{type(exc).__name__}: {exc}"

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
        "selection_authority": selection_summary,
        "production_signal_evidence": production_evidence,
        "market_reports": reports_status,
    }
