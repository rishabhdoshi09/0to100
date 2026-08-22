"""NSE financial-results + XBRL ingest → PIT events and fundamentals ledgers.

Official sources only:
  - https://www.nseindia.com/api/corporates-financial-results
  - linked XBRL instance documents on nsearchives.nseindia.com

``available_at`` = exchange broadcast / disseminate time from the results API.
Never uses local fetch time as availability.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import requests

from data.pit_events import merge_events, write_events
from data.pit_fundamentals import merge_fundamentals, write_fundamentals

_RESULTS_API = "https://www.nseindia.com/api/corporates-financial-results"
_HOME = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": _HOME,
    "Accept": "application/json",
}

XBRL_CACHE = Path(__file__).resolve().parent.parent / "logs" / "xbrl_cache"

# Local-name → fundamentals field
_XBRL_MAP = {
    "RevenueFromOperations": "revenue_from_operations",
    "OtherIncome": "other_income",
    "ProfitFromOperationsBeforeOtherIncomeFinanceCostsAndExceptionalItems": "operating_profit",
    "ProfitFromOrdinaryActivitiesBeforeFinanceCostsAndExceptionalItems": "operating_profit",
    "ProfitBeforeTax": "profit_before_tax",
    "ProfitLossForPeriod": "profit_after_tax",
    "ProfitLossForPeriodFromContinuingOperations": "profit_after_tax",
    "ComprehensiveIncomeForThePeriod": "comprehensive_income",
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "basic_eps",
    "BasicEarningsLossPerShareFromContinuingOperations": "basic_eps",
    "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "diluted_eps",
    "DilutedEarningsLossPerShareFromContinuingOperations": "diluted_eps",
    "FaceValueOfEquityShareCapital": "face_value",
    "PaidUpValueOfEquityShareCapital": "paid_up_equity_capital",
    "DebtEquityRatio": "debt_equity_ratio",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_HOME, timeout=20)
    except Exception:
        pass
    return s


def _parse_nse_dt(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return None
    for fmt in (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw[: len(fmt) + 5], fmt).isoformat()
        except ValueError:
            continue
    # Fallback: first 10-ish tokens via pandas in ledger validate
    return raw


def _iso_date_from_nse(raw: str | None) -> str | None:
    ts = _parse_nse_dt(raw)
    if not ts:
        return None
    try:
        return str(datetime.fromisoformat(ts).date())
    except Exception:
        if len(raw or "") >= 10:
            return None
        return None


def fetch_results_range(
    from_date: str,
    to_date: str,
    *,
    period: str = "Quarterly",
    session: requests.Session | None = None,
) -> tuple[list[dict], dict]:
    """Fetch NSE financial-results list for a calendar window.

    Dates must be DD-MM-YYYY as required by the NSE API.
    """
    sess = session or _session()
    url = (
        f"{_RESULTS_API}?index=equities&period={period}"
        f"&from_date={from_date}&to_date={to_date}"
    )
    meta = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
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


def results_to_event_rows(payload: list[dict], *, source: str = "nse_financial_results") -> list[dict]:
    out = []
    for raw in payload or []:
        sym = str(raw.get("symbol") or "").strip().upper()
        avail_raw = (
            raw.get("broadCastDate")
            or raw.get("exchdisstime")
            or raw.get("filingDate")
        )
        avail_ts = _parse_nse_dt(str(avail_raw) if avail_raw else None)
        if not sym or not avail_ts:
            continue
        out.append({
            "symbol": sym,
            "isin": raw.get("isin"),
            "available_at": avail_ts,
            "available_at_ts": avail_ts,
            "event_type": "EARNINGS_RESULT",
            "period": raw.get("period"),
            "period_start": _iso_date_from_nse(raw.get("fromDate")) or raw.get("fromDate"),
            "period_end": _iso_date_from_nse(raw.get("toDate")) or raw.get("toDate"),
            "relating_to": raw.get("relatingTo"),
            "financial_year": raw.get("financialYear"),
            "consolidated": raw.get("consolidated"),
            "audited": raw.get("audited"),
            "seq_id": str(raw.get("seqNumber") or ""),
            "source": source,
            "source_url": raw.get("xbrl") or "",
            "headline": (
                f"{sym} {raw.get('period') or ''} results "
                f"{raw.get('relatingTo') or ''} ({raw.get('consolidated') or ''})"
            ).strip(),
            "desc": raw.get("resultDescription") or raw.get("companyName") or "",
        })
    return out


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _context_rank(context_ref: str | None) -> int:
    """Prefer the current reporting duration (NSE OneD) over YTD/comparatives.

    NSE result instances typically use OneD = this quarter/year and FourD =
    a same-dated cumulative/YTD bucket. Document order is not a contract.
    """
    ref = str(context_ref or "")
    if ref == "OneD":
        return 0
    if ref.startswith("One"):
        return 1
    if ref == "FourD" or ref.startswith("Four"):
        return 50
    return 10


def parse_xbrl_metrics(xml_bytes: bytes) -> dict[str, float]:
    """Extract a small set of Ind-AS metrics from an NSE XBRL instance.

    When the same tag appears in several contexts, prefer NSE ``OneD``
    (current period) over ``FourD`` (often YTD/cumulative with reused dates).
    """
    root = ET.fromstring(xml_bytes)
    scored: dict[str, tuple[int, float]] = {}
    for el in root.iter():
        name = _local_name(el.tag)
        field = _XBRL_MAP.get(name)
        if not field:
            continue
        text = (el.text or "").strip()
        if not text:
            continue
        try:
            val = float(text)
        except ValueError:
            continue
        rank = _context_rank(el.get("contextRef"))
        prev = scored.get(field)
        if prev is None or rank < prev[0]:
            scored[field] = (rank, val)
    return {k: v[1] for k, v in scored.items()}


def _xbrl_cache_path(url: str) -> Path:
    h = re.sub(r"[^a-zA-Z0-9]+", "_", url)[-120:]
    return XBRL_CACHE / f"{h}.xml"


def fetch_xbrl(url: str, *, session: requests.Session | None = None, retries: int = 2) -> bytes | None:
    if not url:
        return None
    cache = _xbrl_cache_path(url)
    if cache.exists() and cache.stat().st_size > 500:
        return cache.read_bytes()
    sess = session or _session()
    for attempt in range(retries + 1):
        try:
            resp = sess.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 500:
                XBRL_CACHE.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(resp.content)
                return resp.content
        except Exception:
            if attempt >= retries:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def result_row_to_fundamentals(raw: dict, metrics: dict[str, float]) -> dict | None:
    if not metrics:
        return None
    avail_ts = _parse_nse_dt(
        str(raw.get("broadCastDate") or raw.get("exchdisstime") or raw.get("filingDate") or "")
    )
    sym = str(raw.get("symbol") or "").strip().upper()
    if not sym or not avail_ts:
        return None
    row = {
        "symbol": sym,
        "isin": raw.get("isin"),
        "available_at": avail_ts,
        "period": raw.get("period"),
        "period_start": _iso_date_from_nse(raw.get("fromDate")) or raw.get("fromDate"),
        "period_end": _iso_date_from_nse(raw.get("toDate")) or raw.get("toDate"),
        "relating_to": raw.get("relatingTo"),
        "financial_year": raw.get("financialYear"),
        "consolidated": raw.get("consolidated"),
        "audited": raw.get("audited"),
        "seq_id": str(raw.get("seqNumber") or ""),
        "source": "nse_xbrl",
        "xbrl_url": raw.get("xbrl") or "",
        "unit": "INR",
        "currency": "INR",
        **metrics,
    }
    return row


def select_xbrl_candidates(
    raw_rows: list[dict],
    *,
    prefer_consolidated: bool = True,
    min_period_end_year: int = 2022,
) -> list[dict]:
    """Deduplicate to one XBRL candidate per symbol+period_end (+consol preference)."""
    scored: dict[tuple, tuple[int, dict]] = {}
    for raw in raw_rows:
        if not raw.get("xbrl"):
            continue
        pe = _iso_date_from_nse(raw.get("toDate")) or ""
        try:
            year = int(pe[:4]) if pe else 0
        except ValueError:
            year = 0
        if year and year < min_period_end_year:
            continue
        sym = str(raw.get("symbol") or "").upper()
        consol = str(raw.get("consolidated") or "")
        period = str(raw.get("period") or "Quarterly")
        key = (sym, pe, period)
        score = 0
        if prefer_consolidated and consol.lower().startswith("consolid"):
            score += 10
        if period == "Quarterly":
            score += 2
        if period == "Annual":
            score += 1
        prev = scored.get(key)
        if prev is None or score > prev[0]:
            scored[key] = (score, raw)
    return [v[1] for v in scored.values()]


def materialize_events_from_results(
    raw_rows: list[dict],
    *,
    dest=None,
    source: str = "nse_financial_results",
    replace: bool = True,
) -> dict:
    events = results_to_event_rows(raw_rows, source=source)
    if replace:
        return write_events(events, path=dest, source=source)
    return merge_events(events, path=dest, source=source)


def materialize_fundamentals_from_xbrl(
    raw_rows: list[dict],
    *,
    max_files: int | None = None,
    workers: int = 10,
    dest=None,
    progress_every: int = 100,
) -> dict:
    candidates = select_xbrl_candidates(raw_rows)
    if max_files is not None:
        candidates = candidates[: max_files]
    print(f"xbrl_candidates {len(candidates)}", flush=True)
    sess = _session()
    rows: list[dict] = []
    errors = 0

    def _one(raw: dict) -> dict | None:
        blob = fetch_xbrl(str(raw.get("xbrl") or ""), session=sess)
        if not blob:
            return None
        try:
            metrics = parse_xbrl_metrics(blob)
        except Exception:
            return None
        return result_row_to_fundamentals(raw, metrics)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, raw): i for i, raw in enumerate(candidates)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                row = fut.result()
                if row:
                    rows.append(row)
                else:
                    errors += 1
            except Exception:
                errors += 1
            if progress_every and done % progress_every == 0:
                print(f"xbrl_progress {done}/{len(candidates)} ok={len(rows)} err={errors}", flush=True)

    status = write_fundamentals(
        rows, path=dest, source="nse_xbrl_financial_results"
    )
    status["xbrl_attempted"] = len(candidates)
    status["xbrl_parsed"] = len(rows)
    status["xbrl_errors"] = errors
    return status
