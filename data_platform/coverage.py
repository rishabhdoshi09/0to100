"""Per-symbol and universe data coverage with remediation queue."""
from __future__ import annotations

from typing import Any, Sequence

from data_platform.contracts import DataCoverage, QualityStatus, utc_now_iso


def _status_fresh(available: bool, age_days: int | None, max_age: int) -> QualityStatus:
    if not available:
        return QualityStatus.MISSING
    if age_days is None:
        return QualityStatus.PARTIAL
    if age_days > max_age:
        return QualityStatus.STALE
    return QualityStatus.FRESH


def audit_symbol(symbol: str) -> DataCoverage:
    sym = str(symbol or "").strip().upper()
    cov = DataCoverage(symbol=sym, reasons={})
    if not sym:
        cov.identity = QualityStatus.ERROR
        cov.reasons["identity"] = "empty symbol"
        return cov

    try:
        from data_platform.security_master import profile_for_symbol
        profile = profile_for_symbol(sym)
        cov.identity = QualityStatus.FRESH if profile.sector else QualityStatus.PARTIAL
        if not profile.sector:
            cov.reasons["identity"] = "sector not mapped"
    except Exception as exc:
        cov.identity = QualityStatus.ERROR
        cov.reasons["identity"] = str(exc)

    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(sym)
        if df is not None and not df.empty and len(df) >= 60:
            cov.price_history = QualityStatus.FRESH
            cov.scan_eligible = QualityStatus.FRESH
        elif df is not None and not df.empty:
            cov.price_history = QualityStatus.PARTIAL
            cov.reasons["price_history"] = f"only {len(df)} sessions"
        else:
            cov.reasons["price_history"] = "no OHLCV in bhav store"
    except Exception as exc:
        cov.reasons["price_history"] = str(exc)

    try:
        from fundamentals.cache import FundamentalsCache
        raw = FundamentalsCache().get(sym) or {}
        if raw:
            cov.fundamentals = QualityStatus.PARTIAL
            cov.ratios = QualityStatus.PARTIAL
            if float(raw.get("coverage_pct") or 0) >= 40:
                cov.fundamentals = QualityStatus.FRESH
                cov.long_term_eligible = QualityStatus.FRESH
        else:
            cov.reasons["fundamentals"] = "not in fundamentals cache"
    except Exception as exc:
        cov.reasons["fundamentals"] = str(exc)

    try:
        from data.corporate_actions import load_events
        events = load_events()
        if events:
            cov.corporate_actions = QualityStatus.FRESH
        else:
            cov.corporate_actions = QualityStatus.MISSING
            cov.reasons["corporate_actions"] = "logs/ca_events.json empty or absent"
    except Exception as exc:
        cov.reasons["corporate_actions"] = str(exc)

    cov.latest_market = cov.price_history
    return cov


def remediation_for(coverage: DataCoverage) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if coverage.price_history in (QualityStatus.MISSING, QualityStatus.STALE):
        actions.append({
            "action": "schedule_price_backfill",
            "symbol": coverage.symbol,
            "reason": coverage.reasons.get("price_history", "price history incomplete"),
        })
    if coverage.fundamentals in (QualityStatus.MISSING, QualityStatus.STALE, QualityStatus.PARTIAL):
        actions.append({
            "action": "schedule_fundamentals_refresh",
            "symbol": coverage.symbol,
            "reason": coverage.reasons.get("fundamentals", "fundamentals incomplete"),
        })
    if coverage.corporate_actions == QualityStatus.MISSING:
        actions.append({
            "action": "schedule_corporate_action_refresh",
            "symbol": coverage.symbol,
            "reason": coverage.reasons.get("corporate_actions", "CA table missing"),
        })
    return actions


def audit_universe(symbols: Sequence[str], limit: int = 120) -> dict[str, Any]:
    rows = [audit_symbol(s) for s in list(symbols)[:limit]]
    remediation: list[dict[str, str]] = []
    for row in rows:
        remediation.extend(remediation_for(row))
    status_counts: dict[str, int] = {}
    for row in rows:
        for field in ("identity", "price_history", "fundamentals", "ratios", "long_term_eligible"):
            val = getattr(row, field)
            key = val.value if hasattr(val, "value") else str(val)
            status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "generated_at": utc_now_iso(),
        "audited": len(rows),
        "status_counts": status_counts,
        "symbols": [
            {
                "symbol": r.symbol,
                "identity": r.identity.value,
                "price_history": r.price_history.value,
                "fundamentals": r.fundamentals.value,
                "long_term_eligible": r.long_term_eligible.value,
                "reasons": r.reasons,
            }
            for r in rows
        ],
        "remediation_queue": remediation[:50],
    }
