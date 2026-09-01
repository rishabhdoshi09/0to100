"""Internet-backed corporate actions with explicit source hierarchy.

Policy:
1. NSE public corporate-actions data first (official).
2. Screener.in public company announcements as a reputable secondary fallback.
3. Persisted last-good snapshot if the internet is temporarily unavailable.

A lower-tier source is never relabelled as official. Missing dates stay missing.
A successful official empty result is authoritative and does not trigger a fake
secondary action.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "logs" / "product" / "corporate_actions"
CACHE_TTL_S = 6 * 60 * 60
REQUEST_TIMEOUT_S = 8

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}
_ACTION_WORDS = (
    "dividend", "bonus", "split", "sub-division", "subdivision", "rights", "right issue",
    "buyback", "buy back", "record date", "corporate action", "demerger", "merger",
    "amalgamation", "capital reduction", "consolidation of shares", "open offer",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym or not re.fullmatch(r"[A-Z0-9&._-]{1,40}", sym):
        raise ValueError("valid NSE symbol required")
    return sym


def _cache_path(symbol: str) -> Path:
    return CACHE_ROOT / f"{_clean_symbol(symbol)}.json"


def _load_cache(symbol: str) -> dict[str, Any] | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _save_cache(symbol: str, payload: Mapping[str, Any]) -> None:
    path = _cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _cache_age_s(payload: Mapping[str, Any]) -> float | None:
    text = str(payload.get("retrieved_at") or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
    except Exception:
        return None


def _action_type(subject: str) -> str:
    text = str(subject or "").lower()
    if "dividend" in text:
        return "DIVIDEND"
    if "bonus" in text:
        return "BONUS"
    if "split" in text or "sub-division" in text or "subdivision" in text:
        return "SPLIT"
    if "right" in text:
        return "RIGHTS"
    if "buyback" in text or "buy back" in text:
        return "BUYBACK"
    if "demerger" in text:
        return "DEMERGER"
    if "merger" in text or "amalgamation" in text:
        return "MERGER"
    if "open offer" in text:
        return "OPEN_OFFER"
    return "OTHER"


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "-"):
            return value
    return None


def _normalise_nse(symbol: str, row: Mapping[str, Any]) -> dict[str, Any]:
    subject = str(_first(row, "subject", "purpose", "PURPOSE", "desc") or "Corporate action")
    return {
        "symbol": symbol,
        "company": _first(row, "comp", "companyName", "company", "COMPANY NAME"),
        "action_type": _action_type(subject),
        "subject": subject,
        "ex_date": _first(row, "exDate", "ex_date", "EX-DATE"),
        "record_date": _first(row, "recordDate", "record_date", "RECORD DATE"),
        "announcement_date": _first(row, "an_dt", "announcementDate", "announcement_date"),
        "face_value": _first(row, "faceVal", "faceValue", "FACE VALUE"),
        "series": _first(row, "series", "SERIES"),
        "source": "NSE India",
        "source_tier": "official_exchange",
        "source_url": f"https://www.nseindia.com/companies-listing/corporate-filings-actions?symbol={symbol}&tabIndex=equity",
        "confidence": "high",
    }


def _fetch_nse(symbol: str) -> list[dict[str, Any]]:
    session = requests.Session()
    landing = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
    # NSE commonly requires the browser cookies set by a landing-page request.
    session.get(landing, headers=_HEADERS, timeout=REQUEST_TIMEOUT_S)
    response = session.get(
        "https://www.nseindia.com/api/corporates-corporateActions",
        params={"index": "equities", "symbol": symbol},
        headers={**_HEADERS, "Accept": "application/json,text/plain,*/*"},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    raw = response.json()
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, Mapping):
        rows = raw.get("data") or raw.get("records") or raw.get("rows") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise RuntimeError("NSE corporate-actions response shape changed")
    return [_normalise_nse(symbol, row) for row in rows if isinstance(row, Mapping)]


def _extract_announcement_date(text: str) -> str | None:
    # Screener often renders announcement dates as '28 Aug' without a year.
    match = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]{2}(?:\s+\d{4})?)\b", text)
    return match.group(1) if match else None


def _fetch_screener(symbol: str) -> list[dict[str, Any]]:
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    response = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT_S)
    if response.status_code == 404:
        # Some companies do not expose consolidated statements.
        url = f"https://www.screener.in/company/{symbol}/"
        response = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Restrict to the Documents/Announcements area where possible, then filter
    # by action words. This deliberately does not infer ex/record dates.
    containers = []
    documents = soup.find(id="documents")
    if documents is not None:
        containers.append(documents)
    if not containers:
        for heading in soup.find_all(["h2", "h3"]):
            if "announcement" in heading.get_text(" ", strip=True).lower():
                containers.append(heading.parent)
    if not containers:
        containers = [soup]

    for container in containers:
        for node in container.find_all(["li", "article"]):
            text = " ".join(node.get_text(" ", strip=True).split())
            lower = text.lower()
            if not text or not any(word in lower for word in _ACTION_WORDS):
                continue
            key = lower[:240]
            if key in seen:
                continue
            seen.add(key)
            link = node.find("a", href=True)
            source_url = urljoin(url, link.get("href")) if link else url
            actions.append({
                "symbol": symbol,
                "company": None,
                "action_type": _action_type(text),
                "subject": text[:800],
                "ex_date": None,
                "record_date": None,
                "announcement_date": _extract_announcement_date(text),
                "face_value": None,
                "series": None,
                "source": "Screener.in",
                "source_tier": "reputable_secondary",
                "source_url": source_url,
                "confidence": "medium",
            })
            if len(actions) >= 30:
                break
        if len(actions) >= 30:
            break
    return actions


def _payload(
    symbol: str,
    actions: list[dict[str, Any]],
    *,
    delivery_state: str,
    source: str,
    source_tier: str,
    attempts: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "symbol": symbol,
        "available": True,
        "delivery_state": delivery_state,
        "source": source,
        "source_tier": source_tier,
        "retrieved_at": _now(),
        "actions": actions,
        "count": len(actions),
        "attempts": attempts,
        "note": note,
        "source_policy": "NSE official -> Screener.in public fallback -> persisted last-good",
        "missing_is_zero": False,
    }


def get_corporate_actions(symbol: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Return corporate actions without turning a transient source failure into a dead feature."""
    sym = _clean_symbol(symbol)
    cached = _load_cache(sym)
    age = _cache_age_s(cached or {})
    if not force_refresh and cached and age is not None and age <= CACHE_TTL_S:
        out = dict(cached)
        out["delivery_state"] = "CACHED_LAST_GOOD"
        out["cache_age_seconds"] = round(age, 1)
        return out

    attempts: list[dict[str, Any]] = []
    try:
        rows = _fetch_nse(sym)
        attempts.append({"source": "NSE India", "source_tier": "official_exchange", "ok": True, "count": len(rows)})
        out = _payload(
            sym, rows, delivery_state="FRESH_OFFICIAL", source="NSE India",
            source_tier="official_exchange", attempts=attempts,
            note="Official NSE result. An empty list is a valid official result.",
        )
        _save_cache(sym, out)
        return out
    except Exception as exc:
        attempts.append({"source": "NSE India", "source_tier": "official_exchange", "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"[:300]})

    try:
        rows = _fetch_screener(sym)
        attempts.append({"source": "Screener.in", "source_tier": "reputable_secondary", "ok": True, "count": len(rows)})
        out = _payload(
            sym, rows, delivery_state="FALLBACK_SECONDARY", source="Screener.in",
            source_tier="reputable_secondary", attempts=attempts,
            note=(
                "NSE was unavailable. Public Screener.in announcements are shown as a secondary fallback; "
                "unknown ex/record dates remain empty."
            ),
        )
        _save_cache(sym, out)
        return out
    except Exception as exc:
        attempts.append({"source": "Screener.in", "source_tier": "reputable_secondary", "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"[:300]})

    if cached:
        out = dict(cached)
        out.update({
            "available": True,
            "delivery_state": "STALE_LAST_GOOD",
            "attempts": attempts,
            "note": "Live sources failed; showing the persisted last-good corporate-actions snapshot.",
            "cache_age_seconds": round(age, 1) if age is not None else None,
        })
        return out

    return {
        "schema_version": 1,
        "symbol": sym,
        "available": False,
        "delivery_state": "UNAVAILABLE",
        "source": None,
        "source_tier": None,
        "retrieved_at": _now(),
        "actions": [],
        "count": 0,
        "attempts": attempts,
        "note": "NSE and reputable secondary acquisition both failed and no last-good snapshot exists.",
        "source_policy": "NSE official -> Screener.in public fallback -> persisted last-good",
        "missing_is_zero": False,
    }
