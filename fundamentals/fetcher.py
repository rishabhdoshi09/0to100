"""
Unified fundamentals entry point.

Authoritative current financial tables come from QuantTerm's dated official
XBRL warehouse when available. Secondary public fundamentals are used to fill
non-official gaps (company description, peers, valuation, etc.) and as a
fallback when the warehouse has no usable current filing.

The important rule is source precedence, not source popularity:

    official warehouse > fresh secondary cache > live secondary scrape > stale last-good

Official tables never get overwritten by a secondary scrape.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Mapping

from fundamentals.cache import FundamentalsCache
from fundamentals.screener_deep import ScreenerDeepFetcher
from logger import get_logger

log = get_logger(__name__)

_cache = FundamentalsCache()
_scraper = ScreenerDeepFetcher()

_OFFICIAL_TABLE_KEYS = (
    "quarterly_results",
    "profit_loss",
    "balance_sheet",
    "cash_flow",
)


def _official_warehouse_snapshot(symbol: str) -> Dict[str, Any] | None:
    """Return latest dated official financial tables for live research.

    This is a *current* query using today's cutoff. Historical replay continues
    to use its explicit as_of=T adapter; this helper never changes PIT replay.
    """
    try:
        from product.pit_query import get_financial_snapshot

        snap = get_financial_snapshot(symbol, as_of=date.today().isoformat())
    except Exception as exc:
        log.info("official_fundamentals_unavailable", symbol=symbol, error=str(exc)[:160])
        return None

    if not snap.get("numbers_parsed") or snap.get("stale_for_production"):
        return None
    tables = dict(snap.get("tables") or {})
    if not any(tables.get(key) for key in _OFFICIAL_TABLE_KEYS):
        return None

    facts = dict(snap.get("facts") or {})
    derived = dict(snap.get("derived") or {})
    data: Dict[str, Any] = dict(tables)

    # Preserve directly usable quality fields without inventing unavailable
    # ratios. These are derived from official facts already eligible today.
    mapping = {
        "debt_to_equity": "debt_to_equity",
        "roe": "roe_approx_pct",
        "roce": "roce_approx_pct",
        "cfo_to_pat": "cash_conversion",
    }
    for target, source in mapping.items():
        if derived.get(source) is not None:
            data[target] = derived[source]

    data.update({
        "source_label": "NSE official XBRL warehouse",
        "source_tier": "official",
        "official": True,
        "retrieved_at": snap.get("latest_publication") or datetime.now(timezone.utc).isoformat(),
        "latest_publication": snap.get("latest_publication"),
        "latest_period_end": snap.get("latest_period_end"),
        "latest_source_url": snap.get("latest_source_url"),
        "latest_evidence_id": snap.get("latest_evidence_id"),
        "official_facts": facts,
        "official_derived": derived,
        "section_as_of": {
            "financial_history": snap.get("latest_publication") or "",
        },
    })
    return data


def _merge_official(base: Mapping[str, Any] | None, official: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Keep useful secondary fields, but official financials always win."""
    merged: Dict[str, Any] = dict(base or {})
    if not official:
        return merged

    for key, value in official.items():
        if key in _OFFICIAL_TABLE_KEYS:
            if value:
                merged[key] = value
            continue
        # Provenance and official derived fields must not be masked by cached
        # secondary values. Other official fields only fill/replace when set.
        if key in {
            "source_label", "source_tier", "official", "retrieved_at",
            "latest_publication", "latest_period_end", "latest_source_url",
            "latest_evidence_id", "official_facts", "official_derived",
            "section_as_of", "debt_to_equity", "roe", "roce", "cfo_to_pat",
        }:
            if value not in (None, "", {}, []):
                merged[key] = value
        elif merged.get(key) in (None, "", {}, []):
            merged[key] = value

    merged["official"] = True
    merged["source_tier"] = "official"
    merged["source_label"] = "NSE official XBRL warehouse"
    return merged


def get_deep_fundamentals(
    symbol: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Return the best current fundamentals pack for *symbol*.

    Normal operation does not scrape when QuantTerm already has usable official
    XBRL financials plus a cached secondary enrichment pack. A forced refresh
    may update the secondary enrichment, but official tables still win.
    """
    symbol = symbol.upper().strip()
    last_good = _cache.get(symbol, allow_stale=True)
    official = _official_warehouse_snapshot(symbol)

    if not force_refresh:
        cached = _cache.get(symbol, allow_stale=False)
        if cached is not None:
            merged = _merge_official(cached, official)
            if official:
                _cache.set(symbol, merged)
                log.info("fundamentals_served_official_plus_cache", symbol=symbol)
            else:
                merged.setdefault("source_label", "cache")
                merged.setdefault("source_tier", "cache")
                log.info("fundamentals_served_from_cache", symbol=symbol)
            return merged

        # Official financials are already sufficient to avoid a network scrape
        # for core company numbers. Keep any stale enrichment fields if present.
        if official:
            merged = _merge_official(last_good, official)
            _cache.set(symbol, merged)
            log.info("fundamentals_served_from_official_warehouse", symbol=symbol)
            return merged

    log.info("fundamentals_scraping", symbol=symbol, force=force_refresh)
    try:
        secondary = _scraper.fetch_all(symbol)
    except Exception as exc:
        if official:
            merged = _merge_official(last_good, official)
            merged["secondary_refresh_error"] = str(exc)[:240]
            _cache.set(symbol, merged)
            log.info("fundamentals_served_official_after_secondary_failure", symbol=symbol)
            return merged
        if last_good:
            last_good["stale"] = True
            last_good["source_label"] = "last_good_snapshot"
            last_good["source_tier"] = "last_good"
            last_good["official"] = False
            log.info("fundamentals_served_last_good", symbol=symbol)
            return last_good
        raise

    data = dict(secondary or {}) if isinstance(secondary, dict) else {}
    data.setdefault("source_label", "secondary_public")
    data.setdefault("source_tier", "secondary")
    data["official"] = False
    data["retrieved_at"] = data.get("retrieved_at") or datetime.now(timezone.utc).isoformat()

    merged = _merge_official(data, official)
    _cache.set(symbol, merged)
    return merged
