"""Company order book = unexecuted customer orders, not exchange bid/ask.

Sources, in order, never invented:
  1. User structured Research Data (`order_book_guidance`)
  2. Company presentation / concall PDFs linked from this symbol's Screener
     documents (BSE AnnPdfOpen / IR PPT)

News is not used — aggregators mix Eimco Elecon with Elecon Engineering.
Screener Insights "Open Order / Order Book" is paywalled (xxx) and is ignored.
Missing stays missing.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _ROOT / "data" / "order_book_extracts"
_TIMEOUT = 25
_MAX_PDF_BYTES = 8_000_000
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.screener.in/",
}

_PRESENTATION_NEEDLES = (
    "ppt", "presentation", "concall", "transcript", "investor present",
    "earnings presentation", "q1fy", "q2fy", "q3fy", "q4fy",
)

_METRIC_NEEDLES = (
    "order_book", "open_order", "open_order_book", "order_backlog",
    "backlog", "orders_on_hand", "open order", "order book",
)

_JAMMED = re.compile(
    r"(?P<digits>\d{4,8})(?P<pct>[+-]\d+(?:\.\d+)?)%\s*"
    r"(?:open order|order book)",
    re.I,
)
_STOOD_AT = re.compile(
    r"(?:open order|order book|order backlog|orders on hand)"
    r".{0,60}(?:stood at|was|of|:)\s*"
    r"(?:rs\.?|inr|₹)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>crore|cr|lakh|lakhs)\b",
    re.I | re.S,
)
_DATE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(20\d{2})",
    re.I,
)


def empty_company_book(*, note: str, source: str = "") -> dict[str, Any]:
    return {
        "kind": "company_backlog",
        "available": False,
        "status": "unavailable",
        "value_cr": None,
        "prior_cr": None,
        "change_pct": None,
        "as_of": "",
        "as_of_label": "",
        "coverage_months": None,
        "stale": False,
        "source": source,
        "source_url": "",
        "note": note,
        "bullets": [],
    }


def parse_as_of_label(text: str) -> date | None:
    match = _DATE.search(str(text or ""))
    if not match:
        return None
    day, month, year = match.group(1), match.group(2), match.group(3)
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{int(day)} {month} {year}", fmt).date()
        except ValueError:
            continue
    return None


def _split_jammed(digits: str, pct: float) -> tuple[float, float] | None:
    # Leading junk (e.g. FY26 glued onto 12539) is tried as suffixes.
    for start in range(0, max(1, len(digits) - 3)):
        chunk = digits[start:]
        for i in range(2, len(chunk) - 1):
            try:
                prior = float(chunk[:i])
                current = float(chunk[i:])
            except ValueError:
                continue
            if prior <= 0 or current < 0:
                continue
            expected = prior * (1.0 + pct / 100.0)
            if abs(expected - current) <= max(0.51, 0.03 * max(current, 1.0)):
                return prior, current
    return None


def extract_open_orders(text: str) -> list[dict[str, Any]]:
    """Pull disclosed open-order figures from presentation/filing text."""
    blob = str(text or "")
    if not blob.strip():
        return []
    hits: list[dict[str, Any]] = []

    for match in _JAMMED.finditer(blob):
        split = _split_jammed(match.group("digits"), float(match.group("pct")))
        if not split:
            continue
        prior, current = split
        ahead = blob[match.end(): match.end() + 160]
        dated = _DATE.search(ahead) or _DATE.search(blob[max(0, match.start() - 80): match.end() + 160])
        as_of_label = f"{dated.group(1)} {dated.group(2)} {dated.group(3)}" if dated else ""
        stamp = parse_as_of_label(as_of_label)
        hits.append({
            "value_cr": current,
            "prior_cr": prior,
            "change_pct": float(match.group("pct")),
            "as_of_label": as_of_label,
            "as_of": stamp.isoformat() if stamp else "",
            "source": "company_presentation",
        })

    for match in _STOOD_AT.finditer(blob):
        raw = float(str(match.group("value")).replace(",", ""))
        unit = match.group("unit").lower()
        value_cr = raw / 100.0 if unit.startswith("lakh") else raw
        around = blob[max(0, match.start() - 80): match.end() + 80]
        dated = _DATE.search(around)
        as_of_label = f"{dated.group(1)} {dated.group(2)} {dated.group(3)}" if dated else ""
        stamp = parse_as_of_label(as_of_label)
        hits.append({
            "value_cr": value_cr,
            "prior_cr": None,
            "change_pct": None,
            "as_of_label": as_of_label,
            "as_of": stamp.isoformat() if stamp else "",
            "source": "company_presentation",
        })

    uniq: dict[tuple[str, float], dict[str, Any]] = {}
    for hit in hits:
        key = (str(hit.get("as_of") or hit.get("as_of_label") or ""), float(hit["value_cr"]))
        prev = uniq.get(key)
        if prev is None or (hit.get("prior_cr") and not prev.get("prior_cr")):
            uniq[key] = hit
    return list(uniq.values())


def _to_cr(value: Any, unit: str | None) -> float | None:
    try:
        n = float(str(value).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return None
    if n != n:
        return None
    u = str(unit or "").lower()
    if "lakh" in u:
        return n / 100.0
    return n


def _from_structured(symbol: str) -> dict[str, Any] | None:
    try:
        from reporting.evidence_intake import structured_rows
        rows = structured_rows(symbol, "order_book_guidance")
    except Exception:
        return None
    best: dict[str, Any] | None = None
    best_date = ""
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        metric = str(row.get("metric") or "").strip().lower().replace(" ", "_")
        label = metric or str(row.get("management_wording") or "").lower()
        if "order" not in label and "backlog" not in label:
            continue
        value = _to_cr(row.get("value"), str(row.get("unit") or "INR_cr"))
        if value is None:
            continue
        as_of = str(row.get("as_of_date") or row.get("period") or "")
        if best is None or as_of > best_date:
            best_date = as_of
            best = {
                "value_cr": value,
                "prior_cr": None,
                "change_pct": None,
                "as_of_label": as_of,
                "as_of": as_of[:10],
                "source": "user_structured_upload",
                "source_url": str(row.get("source_url") or ""),
                "wording": str(row.get("management_wording") or ""),
            }
    return best


def _is_presentation(title: str, url: str) -> bool:
    blob = f"{title} {url}".lower()
    if any(skip in blob for skip in ("newspaper", "credit rating", "annual report", "governance")):
        return False
    return any(needle in blob for needle in _PRESENTATION_NEEDLES)


def _pdf_text(url: str) -> str:
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(payload.get("text") or "")
        except Exception:
            pass
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except Exception as exc:
        log.debug("order_book_pdf_fetch_failed", url=url, error=str(exc))
        return ""
    if resp.status_code != 200 or len(resp.content) < 100:
        return ""
    if len(resp.content) > _MAX_PDF_BYTES:
        log.debug("order_book_pdf_too_large", url=url, bytes=len(resp.content))
        return ""
    if not resp.content[:8].startswith(b"%PDF"):
        return ""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(resp.content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])
    except Exception as exc:
        log.debug("order_book_pdf_parse_failed", url=url, error=str(exc))
        return ""
    try:
        cache_path.write_text(
            json.dumps({"url": url, "text": text[:80000]}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    return text


def _fetch_documents(symbol: str) -> list[dict[str, str]]:
    try:
        from fundamentals.screener_deep import ScreenerDeepFetcher
        fetcher = ScreenerDeepFetcher()
        _url, soup = fetcher._fetch_page(symbol, consolidated=False)
        return fetcher._parse_documents(soup)
    except Exception as exc:
        log.debug("order_book_documents_fetch_failed", symbol=symbol, error=str(exc))
        return []


def _from_presentations(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for doc in documents or []:
        if not isinstance(doc, Mapping):
            continue
        title = str(doc.get("title") or "")
        url = str(doc.get("url") or "")
        if not url or not _is_presentation(title, url):
            continue
        text = _pdf_text(url)
        for hit in extract_open_orders(text):
            hit = dict(hit)
            hit["source_url"] = url
            hit["source"] = "company_presentation"
            hit["title"] = title
            if best is None or str(hit.get("as_of") or "") > str(best.get("as_of") or ""):
                best = hit
    return best


def _coverage_months(value_cr: float | None, ttm_sales_cr: float | None) -> float | None:
    if value_cr is None or not ttm_sales_cr or ttm_sales_cr <= 0:
        return None
    return round(value_cr / (ttm_sales_cr / 12.0), 1)


def build_company_order_book(
    symbol: str,
    *,
    raw_data: Mapping[str, Any] | None = None,
    ttm_sales_cr: float | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    symbol = str(symbol or "").upper().strip()
    note_missing = (
        "Company order book is unexecuted customer orders already won — "
        "not the stock’s bid/ask tape. No rupee backlog in the latest results filing."
    )
    if not symbol:
        return empty_company_book(note=note_missing)

    structured = _from_structured(symbol)
    docs = list((raw_data or {}).get("documents") or [])
    if not docs and structured is None:
        docs = _fetch_documents(symbol)
    presented = _from_presentations(docs)
    hit = structured or presented
    if hit is None:
        return empty_company_book(note=note_missing)

    from fundamentals.period_freshness import quarters_behind

    as_of_label = str(hit.get("as_of_label") or "")
    as_of_iso = str(hit.get("as_of") or "")
    stamp = parse_as_of_label(as_of_label)
    if stamp is None and as_of_iso:
        try:
            stamp = date.fromisoformat(as_of_iso[:10])
        except ValueError:
            stamp = None
    behind = quarters_behind(stamp, as_of) if stamp else None
    stale = behind is not None and behind >= 2
    value = hit.get("value_cr")
    coverage = _coverage_months(value if isinstance(value, (int, float)) else None, ttm_sales_cr)
    when = as_of_label or as_of_iso or "an undisclosed date"
    prefix = "Latest disclosed" if not stale else f"As of {when} (stale)"
    bullets = [
        f"{prefix} company order book ₹{value:,.0f} cr"
        if not stale else
        f"As of {when} (stale) company order book ₹{value:,.0f} cr"
    ]
    if hit.get("change_pct") is not None and hit.get("prior_cr") is not None:
        bullets.append(
            f"{hit['change_pct']:+.1f}% vs prior ₹{hit['prior_cr']:,.0f} cr"
        )
    if coverage is not None:
        bullets.append(f"About {coverage:.1f} months of TTM sales")
    if stale:
        bullets.append(
            "Newer quarterly results are out and did not restate a rupee backlog in the results PDF."
        )
    wording = str(hit.get("wording") or "").strip()
    if wording:
        bullets.append(wording[:180])
    note = (
        "Unexecuted customer orders already won, from the company’s own presentation or uploaded filing. "
        "Not exchange bid/ask."
    )
    if stale:
        note = (
            f"Last company-disclosed open order is {when}. "
            "Newer results PDFs did not include this number. Not the stock’s bid/ask tape."
        )
    return {
        "kind": "company_backlog",
        "available": True,
        "status": "disclosed",
        "value_cr": value,
        "prior_cr": hit.get("prior_cr"),
        "change_pct": hit.get("change_pct"),
        "as_of": as_of_iso or (stamp.isoformat() if stamp else ""),
        "as_of_label": when,
        "coverage_months": coverage,
        "stale": stale,
        "source": str(hit.get("source") or ""),
        "source_url": str(hit.get("source_url") or ""),
        "note": note,
        "bullets": bullets,
    }
