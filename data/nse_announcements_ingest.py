"""NSE corporate-announcements ingest → PIT events ledger.

Uses official ``an_dt`` / ``exchdisstime`` / ``sort_date`` as AVAILABLE_AT.
Never uses scrape fetch time as availability.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from data.pit_events import merge_events

_API = "https://www.nseindia.com/api/corporate-announcements"
_HOME = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": _HOME,
    "Accept": "application/json",
}

_RESULTS_DESC_KEYS = (
    "financial result",
    "result update",
    "quarterly",
    "audited result",
    "unaudited result",
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_HOME, timeout=20)
    except Exception:
        pass
    return s


def fetch_announcements_range(
    from_date: str,
    to_date: str,
    *,
    session: requests.Session | None = None,
) -> tuple[list[dict], dict]:
    """Dates as DD-MM-YYYY."""
    sess = session or _session()
    url = f"{_API}?index=equities&from_date={from_date}&to_date={to_date}"
    meta = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "from_date": from_date,
        "to_date": to_date,
    }
    try:
        resp = sess.get(url, timeout=180)
        meta["http_status"] = resp.status_code
        if resp.status_code != 200:
            return [], meta
        data = resp.json()
        rows = data if isinstance(data, list) else []
        meta["n"] = len(rows)
        return rows, meta
    except Exception as exc:
        meta["error"] = str(exc)
        return [], meta


def _event_type_for_desc(desc: str) -> str:
    d = (desc or "").lower()
    if any(k in d for k in _RESULTS_DESC_KEYS):
        return "FINANCIAL_RESULT_UPDATE"
    return "CORPORATE_ANNOUNCEMENT"


def announcements_to_event_rows(
    payload: list[dict],
    *,
    symbols: set[str] | None = None,
    results_only: bool = False,
    source: str = "nse_corporate_announcements",
) -> list[dict]:
    out = []
    for raw in payload or []:
        sym = str(raw.get("symbol") or "").strip().upper()
        if symbols is not None and sym not in symbols:
            continue
        desc = str(raw.get("desc") or "")
        et = _event_type_for_desc(desc)
        if results_only and et != "FINANCIAL_RESULT_UPDATE":
            continue
        avail = raw.get("an_dt") or raw.get("exchdisstime") or raw.get("sort_date")
        if not sym or not avail:
            continue
        out.append({
            "symbol": sym,
            "isin": raw.get("sm_isin"),
            "available_at": avail,
            "available_at_ts": avail,
            "event_type": et,
            "headline": str(raw.get("attchmntText") or desc)[:500],
            "desc": desc,
            "seq_id": str(raw.get("seq_id") or ""),
            "source": source,
            "source_url": raw.get("attchmntFile") or "",
        })
    return out


def materialize_announcement_events(
    from_date: str,
    to_date: str,
    *,
    symbols: set[str] | None = None,
    results_only: bool = True,
    dest=None,
) -> dict:
    rows, meta = fetch_announcements_range(from_date, to_date)
    events = announcements_to_event_rows(
        rows, symbols=symbols, results_only=results_only,
    )
    status = merge_events(
        events,
        path=dest,
        source=f"nse_corporate_announcements:{from_date}:{to_date}",
    )
    status["fetch_meta"] = meta
    status["raw_announcements"] = meta.get("n", 0)
    status["kept_events"] = len(events)
    return status
