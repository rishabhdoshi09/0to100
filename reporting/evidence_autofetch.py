"""Auto-download and attach research evidence from known source links.

Honesty rules:
  • Never invents guidance quotes, FY targets, or financial figures.
  • Screener-derived CSVs are labelled as SCREENED_CACHE exports of already-fetched tables.
  • HTML listing pages become SOURCE_ATTACHED_UNPARSED snapshots (not parsed guidance).
  • Google search links are skipped (not downloadable filings).
  • Failures stay failures — reported in the result pack.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlparse

import requests

from reporting.evidence_intake import (
    RESOURCE_SPECS,
    clean_symbol,
    evidence_requirements,
    load_raw_fundamentals,
    resource_links,
    save_upload,
)


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TIMEOUT = 25
_MAX_BYTES = 20 * 1024 * 1024
_SKIP_HOST_FRAGMENTS = ("google.", "bing.", "duckduckgo.", "yahoo.")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        text = str(value).replace(",", "").replace("%", "").replace("₹", "").strip()
        if text in {"", "-", "--", "N/A", "—"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _label(row: Mapping[str, Any]) -> str:
    for key in ("row_label", "label", "", "Particulars", "particulars"):
        if key in row and str(row.get(key) or "").strip():
            return str(row.get(key)).strip().lower()
    for value in row.values():
        if isinstance(value, str) and value.strip() and not re.fullmatch(r"[-+]?\d[\d,.]*", value.strip()):
            return value.strip().lower()
    return ""


def _period_columns(row: Mapping[str, Any]) -> list[str]:
    skip = {"", "row_label", "label", "particulars", "particular"}
    cols: list[str] = []
    for key in row.keys():
        text = str(key or "").strip()
        if not text or text.lower() in skip:
            continue
        cols.append(text)
    return cols


def _find_row(table: Sequence[Mapping[str, Any]], *needles: str) -> Mapping[str, Any] | None:
    for row in table or []:
        if not isinstance(row, Mapping):
            continue
        label = _label(row)
        if any(n in label for n in needles):
            return row
    return None


def _parse_period_end(text: str) -> str | None:
    text = str(text or "").strip()
    if not text:
        return None
    for fmt in ("%b %Y", "%B %Y", "%Y-%m-%d", "%d-%m-%Y", "%Y"):
        try:
            dt = datetime.strptime(text[:12].strip(), fmt)
            if fmt == "%Y":
                return f"{dt.year}-03-31"
            if fmt in {"%b %Y", "%B %Y"}:
                return f"{dt.year:04d}-{dt.month:02d}-28"
            return dt.date().isoformat()
        except ValueError:
            continue
    return None


def _csv_bytes(rows: list[dict[str, Any]], columns: Sequence[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in columns})
    return buf.getvalue().encode("utf-8")


def _screener_source_url(symbol: str) -> str:
    return f"https://www.screener.in/company/{symbol}/consolidated/"


def _build_financial_history_from_screener(symbol: str, raw: Mapping[str, Any]) -> tuple[bytes, str, str] | None:
    pl = list(raw.get("profit_loss") or [])
    sales = _find_row(pl, "sales", "revenue")
    profit = _find_row(pl, "net profit", "profit after tax", "pat")
    op_profit = _find_row(pl, "operating profit", "ebit")
    if sales is None and profit is None:
        return None
    cols = _period_columns(sales or profit or {})
    if not cols:
        return None
    source_url = _screener_source_url(symbol)
    rows: list[dict[str, Any]] = []
    as_of = _today()
    for col in cols:
        period_end = _parse_period_end(col) or as_of
        rev = _f((sales or {}).get(col))
        pat = _f((profit or {}).get(col))
        ebitda = _f((op_profit or {}).get(col))
        if rev is None and pat is None:
            continue
        ebitda_margin = round(ebitda / rev * 100.0, 2) if ebitda is not None and rev not in (None, 0) else ""
        rows.append(
            {
                "period_end": period_end,
                "period_type": "annual",
                "revenue_cr": rev if rev is not None else "",
                "ebitda_cr": ebitda if ebitda is not None else "",
                "ebitda_margin_pct": ebitda_margin,
                "pat_cr": pat if pat is not None else "",
                "cfo_cr": "",
                "debt_cr": "",
                "source_url": source_url,
                "as_of_date": period_end,
            }
        )
        as_of = period_end
    if not rows:
        return None
    content = _csv_bytes(
        rows,
        (
            "period_end",
            "period_type",
            "revenue_cr",
            "ebitda_cr",
            "ebitda_margin_pct",
            "pat_cr",
            "cfo_cr",
            "debt_cr",
            "source_url",
            "as_of_date",
        ),
    )
    return content, as_of, source_url


def _build_business_profile_from_screener(symbol: str, raw: Mapping[str, Any]) -> tuple[bytes, str, str] | None:
    about = str(raw.get("about") or "").strip()
    if not about:
        return None
    source_url = _screener_source_url(symbol)
    as_of = _today()
    content = _csv_bytes(
        [
            {
                "as_of_date": as_of,
                "business_summary": about[:2000],
                "customers": "",
                "demand_drivers": "",
                "source_url": source_url,
            }
        ],
        ("as_of_date", "business_summary", "customers", "demand_drivers", "source_url"),
    )
    return content, as_of, source_url


def _build_shareholding_from_screener(symbol: str, raw: Mapping[str, Any]) -> tuple[bytes, str, str] | None:
    table = list(raw.get("shareholding") or [])
    if not table:
        return None

    def _series(needles: tuple[str, ...]) -> Mapping[str, Any] | None:
        return _find_row(table, *needles)

    promoter = _series(("promoter",))
    fii = _series(("fii", "foreign"))
    dii = _series(("dii", "domestic"))
    public = _series(("public",))
    pledge = _series(("pledge",))
    # Intake requires promoter + FII + DII for structured shareholding rows.
    if promoter is None or fii is None or dii is None:
        return None
    cols = _period_columns(promoter)
    if not cols:
        return None
    source_url = _screener_source_url(symbol)
    rows: list[dict[str, Any]] = []
    as_of = _today()
    for col in cols:
        period = _parse_period_end(col) or as_of
        promoter_pct = _f(promoter.get(col))
        fii_pct = _f(fii.get(col))
        dii_pct = _f(dii.get(col))
        if promoter_pct is None or fii_pct is None or dii_pct is None:
            continue
        rows.append(
            {
                "quarter_end": period,
                "promoter_pct": promoter_pct,
                "fii_pct": fii_pct,
                "dii_pct": dii_pct,
                "public_pct": _f((public or {}).get(col)) if public else "",
                "promoter_pledge_pct": _f((pledge or {}).get(col)) if pledge else "",
                "source_url": source_url,
                "as_of_date": period,
            }
        )
        as_of = period
    if not rows:
        return None
    content = _csv_bytes(
        rows,
        (
            "quarter_end",
            "promoter_pct",
            "fii_pct",
            "dii_pct",
            "public_pct",
            "promoter_pledge_pct",
            "source_url",
            "as_of_date",
        ),
    )
    return content, as_of, source_url


def _should_skip_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(frag in host for frag in _SKIP_HOST_FRAGMENTS)


def _extension_for(content_type: str, url: str, kind: str) -> str | None:
    path = Path(urlparse(url).path)
    suffix = path.suffix.lower()
    accepted = set(RESOURCE_SPECS[kind].accepted_extensions)
    if suffix in accepted:
        return suffix
    ct = (content_type or "").lower()
    if "pdf" in ct and ".pdf" in accepted:
        return ".pdf"
    if ("csv" in ct or "text/csv" in ct) and ".csv" in accepted:
        return ".csv"
    if ("json" in ct) and ".json" in accepted:
        return ".json"
    if ("text/plain" in ct or "text/html" in ct) and ".txt" in accepted:
        return ".txt"
    if "html" in ct and ".txt" in accepted:
        return ".txt"
    return None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _UA, "Accept": "*/*"})
    return session


def _download(session: requests.Session, url: str) -> tuple[bytes, str, str]:
    response = session.get(url, timeout=_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    content = response.content or b""
    if len(content) > _MAX_BYTES:
        raise ValueError(f"download exceeds {_MAX_BYTES} bytes")
    final_url = str(response.url or url)
    content_type = str(response.headers.get("Content-Type") or "")
    return content, content_type, final_url


def _html_to_txt(content: bytes, url: str, symbol: str) -> bytes:
    text = content.decode("utf-8", errors="ignore")
    # Strip tags lightly without requiring BeautifulSoup.
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    header = (
        f"QuantTerm auto-fetched source snapshot for {symbol}\n"
        f"Source URL: {url}\n"
        f"Fetched at: {datetime.now(timezone.utc).isoformat()}\n"
        "Status: SOURCE_ATTACHED_UNPARSED — not parsed into guidance quotes.\n\n"
    )
    body = text[:120_000]
    return (header + body).encode("utf-8")


def _find_pdf_links(html: bytes, base_url: str, limit: int = 3) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        lower = absolute.lower()
        if ".pdf" in lower.split("?")[0]:
            found.append(absolute)
        if len(found) >= limit:
            break
    # Dedupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for url in found:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _attach_bytes(
    symbol: str,
    kind: str,
    content: bytes,
    *,
    filename: str,
    as_of: str,
    source_url: str,
) -> dict[str, Any]:
    return save_upload(
        symbol,
        kind,
        content,
        filename=filename,
        as_of=as_of,
        source_url=source_url,
    )


def _autofetch_from_screener(symbol: str, kind: str, raw: Mapping[str, Any]) -> dict[str, Any] | None:
    builders = {
        "financial_history": _build_financial_history_from_screener,
        "business_profile": _build_business_profile_from_screener,
        "shareholding_history": _build_shareholding_from_screener,
    }
    builder = builders.get(kind)
    if builder is None:
        return None
    built = builder(symbol, raw)
    if built is None:
        return {
            "kind": kind,
            "ok": False,
            "method": "screener_cache_export",
            "error": "Screener cache missing the tables needed for this export. Refresh fundamentals first.",
        }
    content, as_of, source_url = built
    item = _attach_bytes(
        symbol,
        kind,
        content,
        filename=f"{symbol}_{kind}_screener_export.csv",
        as_of=as_of,
        source_url=source_url,
    )
    return {
        "kind": kind,
        "ok": True,
        "method": "screener_cache_export",
        "evidence_id": item.get("evidence_id"),
        "filename": item.get("filename"),
        "extraction_status": item.get("extraction_status"),
        "source_url": source_url,
        "as_of": as_of,
        "note": "Exported from Screener cache already on disk — not invented numbers.",
    }


def _autofetch_from_links(
    symbol: str,
    kind: str,
    links: Sequence[Mapping[str, Any]],
    *,
    max_downloads: int = 3,
) -> dict[str, Any]:
    session = _session()
    attempts: list[dict[str, Any]] = []
    downloads_left = max(0, int(max_downloads))
    for link in links:
        if downloads_left <= 0:
            attempts.append(
                {
                    "url": "",
                    "label": kind,
                    "ok": False,
                    "error": "download budget exhausted for this kind (load protection)",
                }
            )
            break
        url = str(link.get("url") or "").strip()
        label = str(link.get("label") or url)
        if not url or _should_skip_url(url):
            attempts.append({"url": url, "label": label, "ok": False, "error": "skipped search/non-filing link"})
            continue
        try:
            content, content_type, final_url = _download(session, url)
            downloads_left -= 1
        except Exception as exc:
            downloads_left -= 1
            attempts.append({"url": url, "label": label, "ok": False, "error": str(exc)})
            continue

        # Prefer nested PDFs from HTML listing pages for annual/guidance packs.
        if "html" in content_type.lower() and kind in {"annual_report", "management_commentary", "order_book_guidance"}:
            pdfs = _find_pdf_links(content, final_url, limit=min(2, downloads_left or 1))
            for pdf_url in pdfs:
                if downloads_left <= 0:
                    break
                try:
                    pdf_bytes, pdf_ct, pdf_final = _download(session, pdf_url)
                    downloads_left -= 1
                    ext = _extension_for(pdf_ct, pdf_final, kind) or (
                        ".pdf" if ".pdf" in RESOURCE_SPECS[kind].accepted_extensions else None
                    )
                    if ext != ".pdf":
                        continue
                    item = _attach_bytes(
                        symbol,
                        kind,
                        pdf_bytes,
                        filename=Path(urlparse(pdf_final).path).name or f"{symbol}_{kind}.pdf",
                        as_of=_today(),
                        source_url=pdf_final,
                    )
                    return {
                        "kind": kind,
                        "ok": True,
                        "method": "official_pdf_from_page",
                        "evidence_id": item.get("evidence_id"),
                        "filename": item.get("filename"),
                        "extraction_status": item.get("extraction_status"),
                        "source_url": pdf_final,
                        "as_of": _today(),
                        "note": f"Downloaded PDF discovered from {label}. Unparsed unless CSV/JSON.",
                        "attempts": attempts,
                    }
                except Exception as exc:
                    attempts.append({"url": pdf_url, "label": f"pdf via {label}", "ok": False, "error": str(exc)})

        ext = _extension_for(content_type, final_url, kind)
        payload = content
        if ext is None and "html" in content_type.lower() and ".txt" in RESOURCE_SPECS[kind].accepted_extensions:
            ext = ".txt"
            payload = _html_to_txt(content, final_url, symbol)
        if ext is None:
            attempts.append(
                {
                    "url": final_url,
                    "label": label,
                    "ok": False,
                    "error": f"content-type {content_type or 'unknown'} not accepted for {kind}",
                }
            )
            continue
        try:
            item = _attach_bytes(
                symbol,
                kind,
                payload,
                filename=Path(urlparse(final_url).path).name or f"{symbol}_{kind}{ext}",
                as_of=_today(),
                source_url=final_url,
            )
            return {
                "kind": kind,
                "ok": True,
                "method": "direct_download",
                "evidence_id": item.get("evidence_id"),
                "filename": item.get("filename"),
                "extraction_status": item.get("extraction_status"),
                "source_url": final_url,
                "as_of": _today(),
                "note": (
                    "Attached source document. Guidance quotes stay missing until structured CSV/JSON is uploaded."
                    if item.get("extraction_status") == "SOURCE_ATTACHED_UNPARSED"
                    else "Structured evidence attached from download/export."
                ),
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append({"url": final_url, "label": label, "ok": False, "error": str(exc)})
            continue

    return {
        "kind": kind,
        "ok": False,
        "method": "official_link_download",
        "error": "No downloadable filing could be attached from the listed sources.",
        "attempts": attempts,
    }


DEFAULT_KINDS = (
    "financial_history",
    "business_profile",
    "shareholding_history",
    "management_commentary",
    "order_book_guidance",
    "annual_report",
)


def _kind_already_covered(symbol: str, kind: str) -> bool:
    status = evidence_requirements(symbol)
    for item in status.get("requirements") or []:
        if str(item.get("key")) != kind:
            continue
        if item.get("available"):
            return True
        if kind == "annual_report" and item.get("source_attached"):
            return True
        return False
    return False


def autofetch_evidence(
    symbol: str,
    *,
    kinds: Iterable[str] | None = None,
    refresh_screener: bool = True,
    only_missing: bool = True,
    max_link_downloads: int = 3,
) -> dict[str, Any]:
    """Download/export evidence for a symbol and attach into the evidence store."""
    symbol = clean_symbol(symbol)
    wanted = [str(k).strip() for k in (kinds or DEFAULT_KINDS) if str(k).strip()]
    wanted = [k for k in wanted if k in RESOURCE_SPECS]
    if not wanted:
        raise ValueError("no valid evidence kinds requested")

    if only_missing:
        wanted = [k for k in wanted if not _kind_already_covered(symbol, k)]
        if not wanted:
            return {
                "accepted": True,
                "symbol": symbol,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "screener_note": "Skipped — all requested kinds already present.",
                "attached_count": 0,
                "failed_count": 0,
                "results": [],
                "skipped": True,
                "status": evidence_requirements(symbol),
                "places_orders": False,
                "honesty": (
                    "Auto-fetch attaches real downloads or Screener-cache exports only. "
                    "Nothing was missing for the requested kinds."
                ),
            }

    screener_note = ""
    needs_screener = any(
        k in {"financial_history", "business_profile", "shareholding_history"} for k in wanted
    )
    if refresh_screener and needs_screener:
        try:
            from fundamentals.lazy import ensure_deep_fundamentals

            ensure_deep_fundamentals(symbol, force_refresh=False)
            screener_note = "Screener cache checked/refreshed before export."
        except Exception as exc:
            screener_note = f"Screener refresh skipped/failed: {exc}"

    raw_record = load_raw_fundamentals(symbol, auto_fetch=False)
    raw = dict(raw_record.get("data") or {})
    links = resource_links(symbol)
    download_budget = max(0, int(max_link_downloads))

    results: list[dict[str, Any]] = []
    for kind in wanted:
        if kind in {"financial_history", "business_profile", "shareholding_history"}:
            result = _autofetch_from_screener(symbol, kind, raw)
            if result is not None:
                results.append(result)
                # If Screener export failed, still try link download as fallback for PDFs.
                if result.get("ok"):
                    continue
        if download_budget <= 0:
            results.append(
                {
                    "kind": kind,
                    "ok": False,
                    "method": "official_link_download",
                    "error": "Global download budget exhausted (load protection).",
                }
            )
            continue
        link_rows = list(links.get(kind) or [])
        # Prefer official links first.
        link_rows.sort(key=lambda row: 0 if str(row.get("official")) == "true" else 1)
        per_kind = min(2, download_budget)
        link_result = _autofetch_from_links(symbol, kind, link_rows, max_downloads=per_kind)
        # Account for attempted downloads roughly from attempts + success.
        used = 1 if link_result.get("ok") else 0
        used += sum(1 for a in (link_result.get("attempts") or []) if a.get("url"))
        download_budget = max(0, download_budget - max(used, 1))
        results.append(link_result)

    attached = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    return {
        "accepted": True,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screener_note": screener_note,
        "attached_count": len(attached),
        "failed_count": len(failed),
        "results": results,
        "status": evidence_requirements(symbol),
        "places_orders": False,
        "honesty": (
            "Auto-fetch attaches real downloads or Screener-cache exports only. "
            "HTML announcement pages become unparsed source snapshots. "
            "Management guidance quotes are never invented — upload a transcript CSV "
            "or wait until a structured extract exists."
        ),
    }
