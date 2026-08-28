"""BSE official corporate-action adapter for the resilient CA runtime.

BSE's DefaultData/w payload uses wire formats that differ from NSE, notably
``Ex_date='25 Oct 2023'``, ``RD_Date`` and ``long_name``. Keep that provider-
specific normalization here and inject it into the canonical resilient refresh
without weakening the conservative symbol/factor rules.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from data import corporate_actions_resilient as CAR

_BASE_REFRESH = CAR.refresh_events_resilient


def parse_bse_date(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y%m%d", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return CAR._parse_date(raw)


def map_bse_symbol(raw: dict[str, Any], maps=None) -> str | None:
    symbols, by_isin, by_name = maps or CAR._instrument_maps()
    candidate = str(
        raw.get("symbol") or raw.get("short_name") or raw.get("Scrip_ID") or raw.get("scrip_id") or ""
    ).strip().upper()
    if candidate in symbols:
        return candidate
    isin = str(raw.get("ISIN") or raw.get("isin") or raw.get("ISIN_NUMBER") or "").strip().upper()
    if isin and isin in by_isin:
        return by_isin[isin]
    for key in (
        "long_name", "sLongName", "LongName", "company_name", "scripname", "Scrip_Name", "short_name"
    ):
        name = CAR._norm_text(raw.get(key))
        if name and name in by_name:
            return by_name[name]
    return None


def fetch_bse_window(start: date, end: date, *, session=None, maps=None) -> list[dict[str, Any]]:
    import requests

    session = session or requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Origin": "https://www.bseindia.com",
        "Referer": CAR.BSE_PAGE,
    })
    params = {
        "ddlcategorys": "E",
        "ddlindustrys": "",
        "segment": "0",
        "strSearch": "D",
        "Fdate": start.strftime("%Y%m%d"),
        "TDate": end.strftime("%Y%m%d"),
    }
    response = session.get(CAR.BSE_API, params=params, timeout=12)
    if response.status_code != 200:
        raise RuntimeError(f"BSE corporate-actions HTTP {response.status_code}")
    payload = response.json()
    rows = payload if isinstance(payload, list) else list((payload or {}).get("Table") or [])
    if not isinstance(rows, list):
        raise RuntimeError("BSE corporate-actions JSON shape invalid")

    maps = maps or CAR._instrument_maps()
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        symbol = map_bse_symbol(raw, maps)
        if not symbol:
            continue
        subject = raw.get("Purpose") or raw.get("purpose") or raw.get("purpose_name") or raw.get("Details")
        parsed = CAR.parse_share_count_action(str(subject or ""))
        ex_date = parse_bse_date(raw.get("Ex_date") or raw.get("ExDate") or raw.get("ex_date") or raw.get("exdate"))
        if parsed is None or not ex_date:
            continue
        kind, factor = parsed
        record_date = parse_bse_date(raw.get("RD_Date") or raw.get("Record_date") or raw.get("record_date")) or ""
        now = CAR._now_iso()
        event = {
            "symbol": symbol,
            "ex_date": ex_date,
            "record_date": record_date,
            "factor": round(float(factor), 8),
            "type": str(kind),
            "subject": str(subject or "").strip(),
            "source": "bse_api",
            "source_url": CAR.BSE_API,
            "source_tier": 2,
            "fetched_at": now,
            "verification": "official_single_source",
            "provenance": [{
                "source": "bse_api",
                "source_url": CAR.BSE_API,
                "fetched_at": now,
                "source_tier": 2,
            }],
            "bse_scrip_code": str(raw.get("scrip_code") or raw.get("scripcode") or ""),
        }
        if event["factor"] > 0 and abs(event["factor"] - 1.0) > 1e-12:
            out.append(event)
    return out


def refresh_events_resilient(*args, **kwargs):
    kwargs.setdefault("bse_fetcher", fetch_bse_window)
    return _BASE_REFRESH(*args, **kwargs)


def install() -> None:
    """Make every runtime caller of CAR.refresh_events_resilient use the hardened BSE adapter."""
    CAR.refresh_events_resilient = refresh_events_resilient
    CAR.refresh_events = refresh_events_resilient
