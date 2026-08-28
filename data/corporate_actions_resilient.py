"""Resilient, auditable corporate-action acquisition for QuantTerm.

This module fills the canonical ``logs/ca_events.json`` used by
``data.corporate_actions.load_events`` without inventing adjustments.

Source order:
1. NSE official JSON API (date-windowed, bounded retries)
2. NSE official CSV download for the same corporate-actions table
3. BSE official corporate-actions API as a corroborating/fill source

Only split/sub-division, consolidation/reverse-split and bonus events become
share-count adjustment factors. Dividends, rights, buybacks and ambiguous text
are ignored by the price adjuster. BSE rows are accepted only when they map
conservatively to a canonical NSE symbol (exact symbol/ISIN/company-name map).

Coverage is persisted separately from the event ledger. A non-empty ledger is
not treated as complete coverage. One failed window never aborts the remaining
historical walk.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from data import corporate_actions as CA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_PATH = ROOT / "logs" / "ca_events.json"
DEFAULT_COVERAGE_PATH = ROOT / "logs" / "ca_coverage.json"
NSE_PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
NSE_API = "https://www.nseindia.com/api/corporates-corporateActions"
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
BSE_PAGE = "https://www.bseindia.com/corporates/corporates_act.html"

_REFRESH_DUE_S = 20 * 60 * 60
_WINDOW_MONTHS = 6
_MAX_ERROR = 220

_BONUS_RE = re.compile(r"\bbonus(?:\s+issue)?[^0-9]{0,30}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
_FACE_FROM_TO_RE = re.compile(
    r"from\s+(?:rs\.?|re\.?|inr)?\s*([\d.]+)(?:\s*/-)?(?:\s*per\s*share)?"
    r".{0,90}?to\s+(?:rs\.?|re\.?|inr)?\s*([\d.]+)",
    re.I,
)


@dataclass(frozen=True)
class Window:
    window_id: str
    start: date
    end: date


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _norm_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _parse_date(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw in {"-", "--", "None", "nan"}:
        return None
    for fmt in (
        "%d-%b-%Y", "%d-%b-%y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def parse_share_count_action(subject: str) -> tuple[str, float] | None:
    """Return (type, share-count factor) only when official text is unambiguous."""
    text = " ".join(str(subject or "").replace("\xa0", " ").split())
    if not text:
        return None
    low = text.lower()
    if any(token in low for token in ("right issue", "rights issue", "dividend", "buy back", "buyback")):
        return None

    # Preserve compatibility with the canonical parser first.
    try:
        parsed = CA.parse_action_subject(text)
    except Exception:
        parsed = None
    if parsed is not None:
        kind, factor = parsed
        try:
            value = float(factor)
        except Exception:
            return None
        if value > 0 and abs(value - 1.0) > 1e-12:
            return str(kind), value

    bonus = _BONUS_RE.search(text)
    if bonus and "bonus" in low:
        issued, held = float(bonus.group(1)), float(bonus.group(2))
        if issued > 0 and held > 0:
            return "bonus", 1.0 + issued / held

    if any(token in low for token in ("split", "sub-division", "sub division", "consolidat", "reverse split")):
        match = _FACE_FROM_TO_RE.search(text)
        if match:
            old_face, new_face = float(match.group(1)), float(match.group(2))
            if old_face > 0 and new_face > 0 and abs(old_face - new_face) > 1e-12:
                kind = "consolidation" if new_face > old_face else "split"
                return kind, old_face / new_face
    return None


def build_windows(*, today: date | None = None, years: int = 5) -> list[Window]:
    """Stable Jan-Jun / Jul-Dec windows so yesterday's completed coverage stays valid."""
    today = today or date.today()
    years = max(1, int(years))
    horizon = date(today.year - years, today.month, min(today.day, 28))
    out: list[Window] = []
    for year in range(horizon.year, today.year + 1):
        halves = (
            ("H1", date(year, 1, 1), date(year, 6, 30)),
            ("H2", date(year, 7, 1), date(year, 12, 31)),
        )
        for half, start, end in halves:
            if end < horizon or start > today:
                continue
            start = max(start, horizon)
            end = min(end, today)
            out.append(Window(f"{year}-{half}", start, end))
    return out


def _nse_session():
    import requests

    session = requests.Session()
    session.headers.update({
        **getattr(CA, "_NSE_HEADERS", {}),
        "Accept": "application/json,text/csv,text/plain,*/*",
        "Referer": NSE_PAGE,
    })
    try:
        session.get("https://www.nseindia.com/", timeout=8)
    except Exception:
        pass
    return session


def _nse_json_fetch(start: date, end: date, *, session=None) -> list[dict[str, Any]]:
    session = session or _nse_session()
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    }
    response = session.get(NSE_API, params=params, timeout=10)
    if response.status_code in {401, 403}:
        try:
            session.get("https://www.nseindia.com/", timeout=8)
        except Exception:
            pass
        response = session.get(NSE_API, params=params, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"NSE corporate-actions JSON HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, (list, dict)):
        raise RuntimeError("NSE corporate-actions JSON shape invalid")
    return _rows_from_nse_payload(payload, source="nse_api")


def _csv_field(row: dict[str, Any], *names: str) -> Any:
    normalised = {_norm_text(k): v for k, v in row.items()}
    for name in names:
        key = _norm_text(name)
        if key in normalised:
            return normalised[key]
    return None


def _nse_csv_fetch(start: date, end: date, *, session=None) -> list[dict[str, Any]]:
    """Use the official Download (.csv) path exposed by the NSE corporate-actions page."""
    session = session or _nse_session()
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
        "csv": "true",
    }
    response = session.get(NSE_API, params=params, timeout=12)
    if response.status_code in {401, 403}:
        try:
            session.get(NSE_PAGE, timeout=8)
        except Exception:
            pass
        response = session.get(NSE_API, params=params, timeout=12)
    if response.status_code != 200:
        raise RuntimeError(f"NSE corporate-actions CSV HTTP {response.status_code}")
    text = response.content.decode("utf-8-sig", "replace")
    reader = csv.DictReader(io.StringIO(text))
    fields = [_norm_text(x) for x in (reader.fieldnames or [])]
    if not fields or not any("SYMBOL" == f for f in fields):
        raise RuntimeError("NSE corporate-actions CSV header invalid")
    out: list[dict[str, Any]] = []
    for raw in reader:
        event = _normalise_event(
            symbol=_csv_field(raw, "SYMBOL"),
            subject=_csv_field(raw, "PURPOSE", "SUBJECT"),
            ex_date=_csv_field(raw, "EX-DATE", "EX DATE", "EXDATE"),
            record_date=_csv_field(raw, "RECORD DATE", "RECORDDATE"),
            series=_csv_field(raw, "SERIES"),
            source="nse_csv",
            source_url=response.url,
            source_tier=1,
            raw=raw,
        )
        if event:
            out.append(event)
    return out


def _rows_from_nse_payload(payload: Any, *, source: str) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else list(
        (payload or {}).get("data") or (payload or {}).get("corporateActions") or []
    )
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        event = _normalise_event(
            symbol=raw.get("symbol"),
            subject=raw.get("subject") or raw.get("purpose"),
            ex_date=raw.get("exDate") or raw.get("ex_date"),
            record_date=raw.get("recDate") or raw.get("recordDate") or raw.get("record_date"),
            series=raw.get("series"),
            source=source,
            source_url=NSE_API,
            source_tier=1,
            raw=raw,
        )
        if event:
            out.append(event)
    return out


def _instrument_maps(path: Path | None = None) -> tuple[set[str], dict[str, str], dict[str, str]]:
    path = path or (ROOT / "logs" / "instruments_cache.csv")
    symbols: set[str] = set()
    by_isin: dict[str, str] = {}
    names: dict[str, set[str]] = {}
    if not path.exists():
        return symbols, by_isin, {}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                exchange = str(row.get("exchange") or "NSE").upper()
                segment = str(row.get("segment") or "").upper()
                instrument_type = str(row.get("instrument_type") or "").upper()
                if exchange not in {"NSE", ""} or "NFO" in segment or instrument_type in {"FUT", "CE", "PE"}:
                    continue
                symbol = str(row.get("tradingsymbol") or row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                symbols.add(symbol)
                isin = str(row.get("isin") or "").strip().upper()
                if isin:
                    by_isin.setdefault(isin, symbol)
                name = _norm_text(row.get("name") or row.get("company_name") or "")
                if name:
                    names.setdefault(name, set()).add(symbol)
    except Exception:
        return set(), {}, {}
    unique_names = {name: next(iter(vals)) for name, vals in names.items() if len(vals) == 1}
    return symbols, by_isin, unique_names


def _map_bse_symbol(raw: dict[str, Any], maps=None) -> str | None:
    symbols, by_isin, by_name = maps or _instrument_maps()
    candidate = str(
        raw.get("symbol") or raw.get("short_name") or raw.get("Scrip_ID") or raw.get("scrip_id") or ""
    ).strip().upper()
    if candidate in symbols:
        return candidate
    isin = str(raw.get("ISIN") or raw.get("isin") or raw.get("ISIN_NUMBER") or "").strip().upper()
    if isin and isin in by_isin:
        return by_isin[isin]
    for key in ("sLongName", "LongName", "company_name", "scripname", "Scrip_Name", "short_name"):
        name = _norm_text(raw.get(key))
        if name and name in by_name:
            return by_name[name]
    return None


def _bse_fetch(start: date, end: date, *, session=None, maps=None) -> list[dict[str, Any]]:
    import requests

    session = session or requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": BSE_PAGE,
    })
    params = {
        "ddlcategorys": "E",
        "ddlindustrys": "",
        "segment": "0",
        "strSearch": "D",
        "Fdate": start.strftime("%Y%m%d"),
        "TDate": end.strftime("%Y%m%d"),
    }
    response = session.get(BSE_API, params=params, timeout=12)
    if response.status_code != 200:
        raise RuntimeError(f"BSE corporate-actions HTTP {response.status_code}")
    payload = response.json()
    rows = payload if isinstance(payload, list) else list((payload or {}).get("Table") or [])
    if not isinstance(rows, list):
        raise RuntimeError("BSE corporate-actions JSON shape invalid")
    maps = maps or _instrument_maps()
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        symbol = _map_bse_symbol(raw, maps)
        if not symbol:
            continue
        event = _normalise_event(
            symbol=symbol,
            subject=raw.get("Purpose") or raw.get("purpose") or raw.get("purpose_name") or raw.get("Details"),
            ex_date=raw.get("Ex_date") or raw.get("ExDate") or raw.get("ex_date"),
            record_date=raw.get("RD_Date") or raw.get("Record_date") or raw.get("record_date"),
            series="EQ",
            source="bse_api",
            source_url=BSE_API,
            source_tier=2,
            raw=raw,
        )
        if event:
            event["bse_scrip_code"] = str(raw.get("scrip_code") or raw.get("scripcode") or "")
            out.append(event)
    return out


def _normalise_event(*, symbol: Any, subject: Any, ex_date: Any, record_date: Any,
                     series: Any, source: str, source_url: str, source_tier: int,
                     raw: dict[str, Any] | None = None) -> dict[str, Any] | None:
    symbol = str(symbol or "").strip().upper()
    series = str(series or "").strip().upper()
    if not symbol or (series and series not in {"EQ", "BE", "SM"}):
        return None
    parsed = parse_share_count_action(str(subject or ""))
    if parsed is None:
        return None
    kind, factor = parsed
    ex = _parse_date(ex_date)
    if not ex or factor <= 0 or abs(float(factor) - 1.0) <= 1e-12:
        return None
    record = _parse_date(record_date)
    return {
        "symbol": symbol,
        "ex_date": ex,
        "record_date": record or "",
        "factor": round(float(factor), 8),
        "type": str(kind),
        "subject": str(subject or "").strip(),
        "source": source,
        "source_url": source_url,
        "source_tier": int(source_tier),
        "fetched_at": _now_iso(),
        "verification": "official_single_source",
        "provenance": [{
            "source": source,
            "source_url": source_url,
            "fetched_at": _now_iso(),
            "source_tier": int(source_tier),
        }],
    }


def _event_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol") or "").upper(),
        str(row.get("ex_date") or "")[:10],
        str(row.get("type") or "").lower(),
    )


def _event_exact(row: dict[str, Any]) -> tuple[str, str, str, float]:
    ident = _event_identity(row)
    try:
        factor = round(float(row.get("factor") or 0.0), 8)
    except Exception:
        factor = 0.0
    return (*ident, factor)


def merge_verified_events(existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge exact events and preserve provenance; conflicting factors fail closed."""
    rows: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    factors: dict[tuple[str, str, str], set[float]] = {}
    conflicts: list[dict[str, Any]] = []

    for source_row in list(existing) + list(incoming):
        if not isinstance(source_row, dict):
            continue
        row = dict(source_row)
        exact = _event_exact(row)
        ident, factor = exact[:3], exact[3]
        if not ident[0] or not ident[1] or factor <= 0 or abs(factor - 1.0) <= 1e-12:
            continue
        known = factors.setdefault(ident, set())
        if known and factor not in known:
            conflicts.append({
                "symbol": ident[0], "ex_date": ident[1], "type": ident[2],
                "factors": sorted(known | {factor}),
                "incoming_source": row.get("source", ""),
            })
            # Remove any already-staged event for this identity: disagreement means no adjustment.
            for key in [k for k in rows if k[:3] == ident]:
                rows.pop(key, None)
            known.add(factor)
            continue
        if len(known) > 1:
            continue
        known.add(factor)
        current = rows.get(exact)
        if current is None:
            row.setdefault("provenance", [])
            if not row["provenance"] and row.get("source"):
                row["provenance"] = [{
                    "source": row.get("source"),
                    "source_url": row.get("source_url", ""),
                    "fetched_at": row.get("fetched_at", ""),
                    "source_tier": row.get("source_tier"),
                }]
            rows[exact] = row
            continue
        provenance = list(current.get("provenance") or []) + list(row.get("provenance") or [])
        deduped = []
        seen = set()
        for item in provenance:
            if not isinstance(item, dict):
                continue
            key = (item.get("source"), item.get("source_url"), item.get("fetched_at"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        current["provenance"] = deduped
        sources = {str(p.get("source") or "") for p in deduped if p.get("source")}
        if len(sources) >= 2:
            current["verification"] = "official_cross_verified"
    ordered = sorted(rows.values(), key=lambda r: (str(r.get("ex_date") or ""), str(r.get("symbol") or "")))
    return ordered, conflicts


def _coverage_file(path: Path, *, years: int) -> dict[str, Any]:
    raw = _safe_json(path, {})
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("version", 1)
    raw.setdefault("years", int(years))
    raw.setdefault("windows", {})
    raw.setdefault("conflicts", [])
    raw.setdefault("last_refresh_at", "")
    return raw


def _window_covered(record: dict[str, Any], window: Window) -> bool:
    if not record.get("success"):
        return False
    return str(record.get("covered_from") or "") <= window.start.isoformat() and str(record.get("covered_to") or "") >= window.end.isoformat()


def coverage_status(*, years: int = 5, events_path: Path | None = None,
                    coverage_path: Path | None = None, today: date | None = None) -> dict[str, Any]:
    events_path = Path(events_path or DEFAULT_EVENTS_PATH)
    coverage_path = Path(coverage_path or DEFAULT_COVERAGE_PATH)
    today = today or date.today()
    coverage = _coverage_file(coverage_path, years=years)
    windows = build_windows(today=today, years=years)
    records = dict(coverage.get("windows") or {})
    missing = [w.window_id for w in windows if not _window_covered(dict(records.get(w.window_id) or {}), w)]
    events = CA.load_events(events_path)
    n_events = sum(len(v) for v in events.values())
    last_refresh = str(coverage.get("last_refresh_at") or "")
    refresh_due = True
    if last_refresh:
        try:
            stamped = datetime.fromisoformat(last_refresh)
            if stamped.tzinfo is None:
                stamped = stamped.astimezone()
            refresh_due = (datetime.now().astimezone() - stamped).total_seconds() >= _REFRESH_DUE_S
        except Exception:
            refresh_due = True
    return {
        "available": bool(n_events),
        "coverage_complete": not missing,
        "refresh_due": bool(refresh_due),
        "requested_years": int(years),
        "requested_from": windows[0].start.isoformat() if windows else "",
        "requested_to": windows[-1].end.isoformat() if windows else "",
        "missing_windows": missing,
        "windows_total": len(windows),
        "windows_complete": len(windows) - len(missing),
        "n_symbols": len(events),
        "n_events": n_events,
        "conflicts": list(coverage.get("conflicts") or []),
        "unresolved_conflicts": len(list(coverage.get("conflicts") or [])),
        "last_refresh_at": last_refresh,
        "events_path": str(events_path),
        "coverage_path": str(coverage_path),
        "source_mix": dict(coverage.get("source_mix") or {}),
    }


def _retry_fetch(fetcher: Callable[[date, date], list[dict[str, Any]]], start: date, end: date,
                 *, attempts: int, sleep_fn: Callable[[float], None]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    for attempt in range(max(1, int(attempts))):
        try:
            return list(fetcher(start, end) or []), errors
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:_MAX_ERROR])
            if attempt + 1 < attempts:
                sleep_fn(min(2.5, 0.25 * (2 ** attempt)) + random.random() * 0.15)
    raise RuntimeError(errors[-1] if errors else "fetch failed")


def refresh_events_resilient(*, force: bool = False, years: int = 5, budget_s: float = 60.0,
                             events_path: Path | None = None, coverage_path: Path | None = None,
                             today: date | None = None,
                             nse_json_fetcher: Callable[[date, date], list[dict[str, Any]]] | None = None,
                             nse_csv_fetcher: Callable[[date, date], list[dict[str, Any]]] | None = None,
                             bse_fetcher: Callable[[date, date], list[dict[str, Any]]] | None = None,
                             sleep_fn: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Fill the verified ledger while preserving coverage and continuing after window failures."""
    events_path = Path(events_path or DEFAULT_EVENTS_PATH)
    coverage_path = Path(coverage_path or DEFAULT_COVERAGE_PATH)
    today = today or date.today()
    windows = build_windows(today=today, years=years)
    coverage = _coverage_file(coverage_path, years=years)
    existing_raw = _safe_json(events_path, [])
    existing = [dict(x) for x in existing_raw] if isinstance(existing_raw, list) else []
    incoming: list[dict[str, Any]] = []
    started = time.monotonic()
    attempted = 0
    successful = 0

    nse_json_fetcher = nse_json_fetcher or (lambda s, e: _nse_json_fetch(s, e))
    nse_csv_fetcher = nse_csv_fetcher or (lambda s, e: _nse_csv_fetch(s, e))
    bse_fetcher = bse_fetcher or (lambda s, e: _bse_fetch(s, e))

    coverage_windows = dict(coverage.get("windows") or {})
    for window in windows:
        previous = dict(coverage_windows.get(window.window_id) or {})
        current_window = window.end == today
        if not force and _window_covered(previous, window) and not current_window:
            continue
        if (time.monotonic() - started) >= max(5.0, float(budget_s)):
            break
        attempted += 1
        window_errors: list[str] = []
        nse_success = False
        source = ""
        rows: list[dict[str, Any]] = []

        try:
            rows, errs = _retry_fetch(nse_json_fetcher, window.start, window.end, attempts=3, sleep_fn=sleep_fn)
            window_errors.extend(errs)
            nse_success = True
            source = "nse_api"
        except Exception as exc:
            window_errors.append(str(exc)[:_MAX_ERROR])
            try:
                rows, errs = _retry_fetch(nse_csv_fetcher, window.start, window.end, attempts=2, sleep_fn=sleep_fn)
                window_errors.extend(errs)
                nse_success = True
                source = "nse_csv"
            except Exception as csv_exc:
                window_errors.append(str(csv_exc)[:_MAX_ERROR])

        bse_rows: list[dict[str, Any]] = []
        if not nse_success:
            try:
                bse_rows, errs = _retry_fetch(bse_fetcher, window.start, window.end, attempts=2, sleep_fn=sleep_fn)
                window_errors.extend(errs)
            except Exception as bse_exc:
                window_errors.append(str(bse_exc)[:_MAX_ERROR])
        incoming.extend(rows)
        incoming.extend(bse_rows)

        if nse_success:
            successful += 1
        coverage_windows[window.window_id] = {
            "window_id": window.window_id,
            "covered_from": window.start.isoformat(),
            "covered_to": window.end.isoformat(),
            "success": bool(nse_success),
            "source": source or ("bse_api" if bse_rows else "unavailable"),
            "nse_rows": len(rows),
            "bse_rows": len(bse_rows),
            "attempted_at": _now_iso(),
            "errors": window_errors[-6:],
        }
        # Persist after every window so a crash resumes instead of restarting the walk.
        coverage["windows"] = coverage_windows
        _atomic_json(coverage_path, coverage)

    merged, conflicts = merge_verified_events(existing, incoming)
    if merged or existing:
        _atomic_json(events_path, merged)
        try:
            from data.bhavcopy_store import reload_corporate_actions
            reload_corporate_actions()
        except Exception:
            pass

    source_mix: dict[str, int] = {}
    for row in merged:
        for item in list(row.get("provenance") or []):
            source = str((item or {}).get("source") or row.get("source") or "unknown")
            source_mix[source] = source_mix.get(source, 0) + 1
    coverage.update({
        "version": 1,
        "years": int(years),
        "windows": coverage_windows,
        "conflicts": conflicts,
        "source_mix": source_mix,
        "last_refresh_at": _now_iso(),
        "last_attempted_windows": attempted,
        "last_successful_windows": successful,
    })
    _atomic_json(coverage_path, coverage)
    status = coverage_status(years=years, events_path=events_path, coverage_path=coverage_path, today=today)
    status.update({
        "attempted_windows": attempted,
        "successful_windows": successful,
        "elapsed_s": round(time.monotonic() - started, 3),
    })
    print(
        "[CA] resilient refresh · "
        f"events={status['n_events']} · symbols={status['n_symbols']} · "
        f"coverage={status['windows_complete']}/{status['windows_total']} · "
        f"complete={status['coverage_complete']} · conflicts={status['unresolved_conflicts']}",
        flush=True,
    )
    return status


# Backwards-friendly aliases for callers/tests.
refresh_events = refresh_events_resilient
ledger_status = coverage_status
