"""Second-stage due diligence — one durable public research entry point.

This package never scans the market. It reads persisted fundamentals, news and
the current scan/long-term rows, then returns an evidence-backed research view.

A successful report is atomically saved as the ticker's last-good snapshot. If a
later cache-only rebuild fails because a backend lane/process is temporarily
unavailable, callers receive that last-good report with an explicit
``STALE_LAST_GOOD`` delivery state instead of watching Investigate disappear.
"""
from __future__ import annotations

from typing import Any

from product.due_diligence.engine import build_due_diligence as _build_due_diligence
from product.due_diligence.research_engine import StockResearchEngine, investigate_stock
from product.due_diligence.store import fresh_delivery, load_report, save_report, stale_delivery
from product.due_diligence.suggest import suggest_tickers


def build_due_diligence(symbol: str, **kwargs: Any) -> dict[str, Any]:
    """Build fresh research when possible; otherwise serve the last-good snapshot.

    The fallback is intentionally narrow: invalid symbols still fail, and a ticker
    with no prior successful report still raises the real build error. Missing
    evidence inside a valid report remains missing — this wrapper never fills or
    estimates research fields.
    """
    clean = str(symbol or "").strip().upper()
    try:
        report = _build_due_diligence(clean, **kwargs)
        save_report(report)
        delivered = fresh_delivery(report)
        # Preserve the store timestamp that was written by save_report when it can
        # be read back; failure to read metadata must never invalidate fresh work.
        saved = load_report(clean)
        if saved and saved.get("snapshot_saved_at"):
            delivered["snapshot_saved_at"] = saved.get("snapshot_saved_at")
        return delivered
    except ValueError:
        raise
    except Exception as exc:
        saved = load_report(clean)
        if saved:
            return stale_delivery(saved, error=f"{type(exc).__name__}: {exc}")
        raise


__all__ = [
    "build_due_diligence",
    "investigate_stock",
    "StockResearchEngine",
    "suggest_tickers",
]
