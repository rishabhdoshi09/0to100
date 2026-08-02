"""User-facing readiness projection for QuantTerm.

The module converts authoritative persisted state into a small product contract:
what works, what is stale, what is missing, why the lane exists, and which owner
operation repairs it. It performs no network access and starts no workers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text[:11], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _age_seconds(value: Any, *, now: datetime) -> float | None:
    stamp = _parse_time(value)
    if stamp is None:
        return None
    return max(0.0, (now - stamp.astimezone(timezone.utc)).total_seconds())


def _lane(
    *,
    key: str,
    label: str,
    meaning: str,
    available: bool,
    as_of: Any,
    max_age_seconds: float,
    weight: int,
    action: str,
    details: str,
    now: datetime,
) -> dict[str, Any]:
    age = _age_seconds(as_of, now=now)
    if not available:
        status = "MISSING"
        factor = 0.0
    elif age is None:
        status = "UNKNOWN_DATE"
        factor = 0.45
    elif age > max_age_seconds:
        status = "STALE"
        factor = 0.45
    else:
        status = "FRESH"
        factor = 1.0
    return {
        "key": key,
        "label": label,
        "meaning": meaning,
        "status": status,
        "available": bool(available),
        "as_of": str(as_of or ""),
        "age_seconds": round(age, 1) if age is not None else None,
        "max_age_seconds": max_age_seconds,
        "weight": weight,
        "earned_weight": round(weight * factor, 2),
        "action": action,
        "details": details,
    }


def build_product_readiness(
    *,
    market: Mapping[str, Any] | None,
    scan: Mapping[str, Any] | None,
    long_term: Mapping[str, Any] | None,
    news: Mapping[str, Any] | None,
    fno: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    operations: Mapping[str, Any] | None,
    now: datetime | None = None,
    ca: Mapping[str, Any] | None = None,
    universe: Mapping[str, Any] | None = None,
    pit_valuations: Mapping[str, Any] | None = None,
    live_edge: Mapping[str, Any] | None = None,
    book_correlation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the readiness score shown in the product shell."""
    now = now or datetime.now(timezone.utc)
    market = dict(market or {})
    scan = dict(scan or {})
    long_term = dict(long_term or {})
    news = dict(news or {})
    fno = dict(fno or {})
    data = dict(data or {})
    operations = dict(operations or {})
    history = dict(data.get("bhavcopy", {}) or {})
    snapshot = dict(data.get("snapshot", {}) or {})
    options_eod = dict(data.get("options_eod", {}) or {})
    news_articles = list(news.get("articles", []) or [])
    latest_news = ""
    if news_articles:
        latest_news = str(news_articles[0].get("published_at") or news_articles[0].get("fetched_at") or "")
    if not latest_news:
        latest_news = str((news.get("latest_refresh", {}) or {}).get("finished_at") or "")

    lanes = [
        _lane(
            key="operations",
            label="Market operations",
            meaning="Runs scans, data preparation, news and F&O refreshes without blocking the interface.",
            available=bool(operations.get("running")),
            as_of=operations.get("heartbeat"),
            max_age_seconds=20,
            weight=10,
            action="BOOTSTRAP_PRODUCT_NOW",
            details=f"{len(operations.get('active', []) or [])} operation(s) active",
            now=now,
        ),
        _lane(
            key="history",
            label="Official price history",
            meaning="Daily NSE OHLCV used for charts, momentum, breakouts, risk and long-term technicals.",
            available=bool(history.get("ready")) and int(history.get("sessions", 0) or 0) >= 60,
            as_of=history.get("latest_date") or history.get("csv_latest_date"),
            max_age_seconds=4 * 24 * 60 * 60,
            weight=20,
            action="REFRESH_DATA_NOW",
            details=f"{int(history.get('sessions', 0) or 0)} sessions · {int(history.get('symbols', 0) or 0):,} symbols",
            now=now,
        ),
        _lane(
            key="snapshot",
            label="Verified market snapshot",
            meaning="Immutable, content-addressed bars pinned for PAPER autonomy and reproducible research.",
            available=bool(snapshot.get("ready")) and bool(snapshot.get("snapshot_id")),
            as_of=snapshot.get("latest_date"),
            max_age_seconds=4 * 24 * 60 * 60,
            weight=10,
            action="CERTIFY_SNAPSHOT_NOW",
            details=(
                f"{snapshot.get('snapshot_id') or 'none'} · source={snapshot.get('source') or 'unknown'}"
            ),
            now=now,
        ),
        _lane(
            key="scanner",
            label="Whole-market scanner",
            meaning="Ranks current technical setups across the cash-market universe; it is not a guaranteed buy list.",
            available=bool(scan.get("available")) and bool(scan.get("records")),
            as_of=scan.get("scanned_at"),
            max_age_seconds=8 * 60 * 60,
            weight=18,
            action="RUN_SCAN_NOW",
            details=f"{len(scan.get('records', []) or [])} saved setups from {int(scan.get('universe_size', 0) or 0):,} evaluated stocks",
            now=now,
        ),
        _lane(
            key="long_term",
            label="Long-term research",
            meaning="Combines current business-quality metrics with official-history technical timing.",
            available=bool(long_term.get("available")) and bool(long_term.get("records")),
            as_of=long_term.get("scanned_at"),
            max_age_seconds=4 * 24 * 60 * 60,
            weight=18,
            action="RUN_LONG_TERM_SCAN_NOW",
            details=f"{len(long_term.get('records', []) or [])} researched candidates · {float((long_term.get('summary', {}) or {}).get('coverage_pct', 0) or 0):.0f}% reported coverage",
            now=now,
        ),
        _lane(
            key="news",
            label="News and filings",
            meaning="Adds dated context and source health; news never becomes an order signal by itself.",
            available=bool(news.get("available")) and bool(news_articles),
            as_of=latest_news,
            max_age_seconds=6 * 60 * 60,
            weight=8,
            action="REFRESH_NEWS_NOW",
            details=f"{int((news.get('stats', {}) or {}).get('total', 0) or 0)} articles in the latest 24-hour statistics",
            now=now,
        ),
        _lane(
            key="fno",
            label="F&O coverage",
            meaning="Shows which cash stocks currently map to derivatives contracts, expiry and lot size. It is not an options strategy engine.",
            available=bool(fno.get("available")) and int(fno.get("mapped_underlyings", 0) or 0) > 0,
            as_of=fno.get("generated_at") or fno.get("cache_mtime"),
            max_age_seconds=30 * 60 * 60,
            weight=5,
            action="REFRESH_FNO_NOW",
            details=f"{int(fno.get('mapped_underlyings', 0) or 0)} mapped stock underlyings",
            now=now,
        ),
        _lane(
            key="options_eod",
            label="Options EOD history",
            meaning="Persisted daily index option chains (OI/IV/PCR) so multi-day options research is possible.",
            available=bool(options_eod.get("available")) and int(options_eod.get("snapshots", 0) or 0) > 0,
            as_of=options_eod.get("latest_as_of"),
            max_age_seconds=4 * 24 * 60 * 60,
            weight=5,
            action="CAPTURE_OPTIONS_EOD_NOW",
            details=(
                f"{int(options_eod.get('snapshots', 0) or 0)} snapshots · "
                f"{int(options_eod.get('symbols', 0) or 0)} underlyings"
            ),
            now=now,
        ),
        _lane(
            key="market",
            label="Market regime and breadth",
            meaning="Provides the environment in which stock signals are interpreted: trend, breadth, leadership and volatility.",
            available=bool(market.get("available")),
            as_of=history.get("latest_date") or history.get("csv_latest_date"),
            max_age_seconds=4 * 24 * 60 * 60,
            weight=6,
            action="REFRESH_DATA_NOW",
            details=str(market.get("summary") or "No market-regime summary"),
            now=now,
        ),
    ]
    score = round(sum(float(item["earned_weight"]) for item in lanes))
    if score >= 90:
        state = "READY"
        summary = "QuantTerm has enough fresh data for normal research use. Review per-stock gaps before acting."
    elif score >= 55:
        state = "PARTIAL"
        summary = "Core research works, but one or more important data lanes are stale or incomplete."
    elif score > 0:
        state = "INCOMPLETE"
        summary = "QuantTerm can show persisted facts, but the product is not research-ready yet."
    else:
        state = "EMPTY"
        summary = "No usable product state is loaded. Start the preparation workflow."
    blockers = [
        f"{item['label']}: {item['status']}"
        for item in lanes
        if item["status"] != "FRESH" and item["weight"] >= 10
    ]
    from product.retail_research_checklist import build_retail_research_checklist

    checklist = build_retail_research_checklist(
        data=data,
        ca=ca,
        universe=universe,
        pit_valuations=pit_valuations,
        live_edge=live_edge,
        book_correlation=book_correlation,
        options_eod=options_eod,
    )
    return {
        "schema_version": 2,
        "generated_at": now.isoformat(),
        "score": score,
        "state": state,
        "summary": summary,
        "lanes": lanes,
        "blockers": blockers,
        "recommended_action": "BOOTSTRAP_PRODUCT_NOW" if score < 90 else "RUN_SCAN_NOW",
        "retail_research_checklist": checklist,
    }
