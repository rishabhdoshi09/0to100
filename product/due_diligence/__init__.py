"""Second-stage due diligence — one durable public research entry point.

This package never scans the market. It reads persisted fundamentals, news and
the current scan/long-term rows, then returns an evidence-backed research view.

A successful report is atomically saved as the ticker's last-good snapshot. If a
later rebuild fails because a backend lane/process is temporarily unavailable,
callers receive that last-good report with an explicit ``STALE_LAST_GOOD`` state.
Interactive research can additionally enrich the report with internet-backed
corporate actions; failure of that enrichment never invalidates due diligence.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.due_diligence.engine import build_due_diligence as _build_due_diligence
from product.due_diligence.research_engine import StockResearchEngine, investigate_stock
from product.due_diligence.store import fresh_delivery, load_report, save_report, stale_delivery
from product.due_diligence.suggest import suggest_tickers


def _attach_corporate_actions(report: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(report)
    symbol = str(out.get("symbol") or "").strip().upper()
    if not symbol:
        return out
    try:
        from product.corporate_actions import get_corporate_actions
        corporate = get_corporate_actions(symbol)
    except Exception as exc:
        out["corporate_actions"] = {
            "available": False,
            "delivery_state": "UNAVAILABLE",
            "actions": [],
            "count": 0,
            "note": f"Corporate-actions enrichment failed: {type(exc).__name__}: {exc}"[:400],
        }
        return out

    out["corporate_actions"] = corporate
    events = [dict(x) for x in (out.get("events") or []) if isinstance(x, Mapping)]
    seen = {str(x.get("headline") or "").strip().lower() for x in events if x.get("headline")}
    for action in list(corporate.get("actions") or [])[:20]:
        if not isinstance(action, Mapping):
            continue
        headline = str(action.get("subject") or action.get("action_type") or "Corporate action").strip()
        if not headline or headline.lower() in seen:
            continue
        seen.add(headline.lower())
        events.append({
            "headline": headline,
            "category": "corporate_action",
            "event_type": str(action.get("action_type") or "OTHER"),
            "published_at": action.get("announcement_date") or action.get("ex_date") or action.get("record_date") or "",
            "source": action.get("source") or corporate.get("source") or "",
            "url": action.get("source_url") or "",
            "official": action.get("source_tier") == "official_exchange",
            "verified": bool(action.get("source_tier") in {"official_exchange", "reputable_secondary"}),
            "materiality": "context",
            "ex_date": action.get("ex_date"),
            "record_date": action.get("record_date"),
            "source_tier": action.get("source_tier"),
        })
    out["events"] = events
    return out


def build_due_diligence(symbol: str, **kwargs: Any) -> dict[str, Any]:
    """Build fresh research when possible; otherwise serve the last-good snapshot.

    ``include_corporate_actions`` is a wrapper-only option. Interactive callers
    default to internet-backed enrichment; production selection can disable it so
    a whole-market recommendation build never waits on per-symbol web requests.
    """
    clean = str(symbol or "").strip().upper()
    include_corporate_actions = bool(kwargs.pop("include_corporate_actions", True))
    try:
        report = _build_due_diligence(clean, **kwargs)
        save_report(report)
        delivered = fresh_delivery(report)
        saved = load_report(clean)
        if saved and saved.get("snapshot_saved_at"):
            delivered["snapshot_saved_at"] = saved.get("snapshot_saved_at")
    except ValueError:
        raise
    except Exception as exc:
        saved = load_report(clean)
        if saved:
            delivered = stale_delivery(saved, error=f"{type(exc).__name__}: {exc}")
        else:
            raise
    if include_corporate_actions:
        return _attach_corporate_actions(delivered)
    return delivered


__all__ = [
    "build_due_diligence",
    "investigate_stock",
    "StockResearchEngine",
    "suggest_tickers",
]
