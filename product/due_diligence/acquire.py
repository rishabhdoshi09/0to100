"""Find, download and persist Investigate inputs. GET never calls this."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from product.due_diligence.extract import (
    extract_from_uploads,
    extract_guidance,
    extract_kpis_from_raw,
    extract_research_pack,
    html_to_text,
    merge_kpi_maps,
)
from product.due_diligence.option_chain import summarize_option_chain

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "logs" / "research_evidence"
ACQUIRE_CAP = 6
FACTS_NAME = "autonomy_facts.json"
FRESH_S = 24 * 60 * 60
MAX_ATTACHMENT_BYTES = 16_000_000
_ALLOWED_HOSTS = {
    "www.screener.in",
    "screener.in",
    "www.nseindia.com",
    "nseindia.com",
    "nsearchives.nseindia.com",
    "archives.nseindia.com",
}
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


def _symbol_dir(symbol: str) -> Path:
    path = EVIDENCE_ROOT / symbol.upper() / "autonomy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def facts_path(symbol: str) -> Path:
    return _symbol_dir(symbol) / FACTS_NAME


def load_autonomy_facts(symbol: str) -> dict[str, Any]:
    path = facts_path(symbol)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_autonomy_facts(symbol: str, payload: Mapping[str, Any]) -> Path:
    path = facts_path(symbol)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def shortlist_symbols(limit: int = ACQUIRE_CAP, scan_payload: Mapping[str, Any] | None = None) -> list[str]:
    """Names the desk already shortlisted. Does not scan the market."""
    if scan_payload is None:
        try:
            from product.scan_store import load_scan
            scan_payload = load_scan() or {}
        except Exception:
            scan_payload = {}
    ranked: list[tuple[int, float, float, str]] = []
    for rec in list((scan_payload or {}).get("records") or []):
        if not isinstance(rec, Mapping):
            continue
        symbol = str(rec.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            sepa = float(rec["sepa_score"]) if rec.get("sepa_score") is not None else None
        except (TypeError, ValueError):
            sepa = None
        try:
            score = float(rec.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        status = str(rec.get("status") or "")
        if sepa is not None and sepa >= 40:
            ranked.append((0, -sepa, -score, symbol))
        elif status == "Ready to trade":
            ranked.append((1, -score, 0.0, symbol))
    ranked.sort()
    out: list[str] = []
    for *_, symbol in ranked:
        if symbol not in out:
            out.append(symbol)
        if len(out) >= limit:
            break
    return out


def acquire_is_fresh(*, now: float | None = None, scan_payload: Mapping[str, Any] | None = None) -> bool:
    symbols = shortlist_symbols(scan_payload=scan_payload)
    if not symbols:
        return True
    now = time.time() if now is None else now
    for symbol in symbols:
        payload = load_autonomy_facts(symbol)
        stamp = str(payload.get("acquired_at") or "")
        if not stamp:
            return False
        try:
            acquired = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return False
        if now - acquired > FRESH_S:
            return False
    return True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _ALLOWED_HOSTS


def _attachment_rank(text: str) -> int:
    low = (text or "").lower()
    if any(tok in low for tok in ("advertisement", "newspaper", "reg 47", "reg. 47", "regulation 47")):
        return 99
    if any(tok in low for tok in ("financial result", "results for the period", "un-audited financial", "audited financial")):
        return 0
    if "transcript" in low or "concall" in low or "conference call" in low:
        return 1
    if "investor presentation" in low:
        return 2
    if "press release" in low and any(tok in low for tok in ("result", "profit", "npa")):
        return 3
    if "annual report" in low or "annual-report" in low:
        return 4
    return 99


def pdf_to_text(content: bytes, *, max_pages: int = 12) -> str:
    if not content:
        return ""
    try:
        from io import BytesIO
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        pages = []
        for page in list(reader.pages)[:max_pages]:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""


def bytes_to_text(content: bytes, ext: str, *, max_pages: int = 12) -> str:
    ext = (ext or "").lower()
    if ext == ".pdf":
        return pdf_to_text(content, max_pages=max_pages)
    if ext in {".html", ".htm", ".txt", ".xml", ".csv", ".json", ".vtt", ".srt"}:
        return content.decode("utf-8", errors="ignore")
    return ""


def _extend_unique(dst: list[dict[str, Any]], src: list[dict[str, Any]], key, cap: int) -> list[dict[str, Any]]:
    seen = {key(item) for item in dst}
    for item in src:
        token = key(item)
        if not token or token in seen:
            continue
        seen.add(token)
        dst.append(item)
        if len(dst) >= cap:
            break
    return dst


def _empty_nse() -> dict[str, Any]:
    return {"step": {"id": "nse_filings", "ok": False}, "downloads": [], "texts": [], "headlines": []}


def _nse_session():
    import requests

    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    session.headers["Referer"] = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
    try:
        session.get("https://www.nseindia.com/", timeout=12)
    except Exception:
        pass
    return session


def _save_bytes(symbol: str, name: str, content: bytes) -> Path:
    path = _symbol_dir(symbol) / name
    path.write_bytes(content)
    return path


def _download(session, url: str, *, symbol: str, name: str, max_bytes: int | None = None) -> dict[str, Any]:
    if not _allowed(url):
        return {"ok": False, "url": url, "error": "host not on the official allow-list"}
    try:
        response = session.get(url, timeout=18, allow_redirects=True)
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)[:240]}
    if response.status_code != 200:
        return {"ok": False, "url": url, "error": f"HTTP {response.status_code}"}
    content = response.content or b""
    limit = MAX_ATTACHMENT_BYTES if max_bytes is None else max_bytes
    if len(content) > limit:
        return {"ok": False, "url": url, "error": f"file larger than {limit // 1_000_000} MB — skipped"}
    path = _save_bytes(symbol, name, content)
    return {
        "ok": True,
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "bytes": len(content),
        "content_type": str(response.headers.get("Content-Type") or ""),
    }


def _quarterly_has_values(data: Mapping[str, Any]) -> bool:
    from product.due_diligence.series import dated_series

    for row in list(data.get("quarterly_results") or []):
        if dated_series(row):
            return True
    return False


def _fetch_screener(symbol: str, *, force: bool) -> dict[str, Any]:
    from fundamentals.fetcher import get_deep_fundamentals
    from reporting.evidence_intake import load_raw_fundamentals

    cached = load_raw_fundamentals(symbol)
    data = dict((cached or {}).get("data") or {})
    need = force or not data or not _quarterly_has_values(data)
    step = {"id": "screener", "ok": False, "forced": bool(need)}
    try:
        payload = get_deep_fundamentals(symbol, force_refresh=need)
        data = dict(payload or data)
        _atomic_json(_symbol_dir(symbol) / "screener.json", data)
        step.update({
            "ok": True,
            "quarterly_rows": len(data.get("quarterly_results") or []),
            "has_values": _quarterly_has_values(data),
        })
        return {"step": step, "data": data}
    except Exception as exc:
        step["error"] = str(exc)[:240]
        if data:
            step["ok"] = True
            step["used_cache"] = True
            step["error_on_refresh"] = step.pop("error")
        return {"step": step, "data": data}


def _fetch_nse(symbol: str, session) -> dict[str, Any]:
    urls = (
        f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}",
        f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}",
    )
    downloads: list[dict[str, Any]] = []
    bodies: list[tuple[str, str]] = []
    headlines: list[tuple[str, str]] = []
    rows_kept = 0
    candidates: list[tuple[int, int, str, str]] = []
    for index, url in enumerate(urls):
        item = _download(session, url, symbol=symbol, name=f"nse_{index}.json")
        downloads.append(item)
        if not item.get("ok"):
            continue
        path = ROOT / str(item["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            bodies.append((html_to_text(path.read_text(encoding="utf-8", errors="ignore")), url))
            continue
        rows = payload if isinstance(payload, list) else payload.get("data") or payload.get("financialResults") or []
        if isinstance(payload, dict) and not rows:
            rows = [payload]
        kept_rows = []
        for row in list(rows)[:400]:
            if not isinstance(row, Mapping):
                continue
            row_symbol = str(row.get("symbol") or row.get("sm_symbol") or "").upper()
            if row_symbol and row_symbol != symbol:
                continue
            kept_rows.append(row)
            desc = " ".join(
                str(row.get(k) or "")
                for k in ("attchmntText", "desc", "subject", "details", "remarks", "resultDescription")
            )
            headlines.append((desc, url))
            rows_kept += 1
            attachment = str(
                row.get("attchmntFile")
                or row.get("attachment")
                or row.get("resultDetailedDataLink")
                or row.get("fileName")
                or ""
            )
            rank = _attachment_rank(desc)
            if not attachment or rank >= 99:
                continue
            if not attachment.startswith("http"):
                attachment = urljoin("https://nsearchives.nseindia.com/corporate/", attachment)
            candidates.append((rank, len(candidates), attachment, desc[:80]))
        if kept_rows and kept_rows != rows:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(kept_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    seen = set()
    picked: list[tuple[int, str, str, str]] = []
    for item in sorted(candidates, key=lambda row: (row[0], row[1])):
        if item[2] in seen:
            continue
        seen.add(item[2])
        picked.append(item)
        if len(picked) >= 6:
            break
    for index, (_rank, _when, attachment, _desc) in enumerate(picked, start=1):
        ext = Path(urlparse(attachment).path).suffix.lower() or ".bin"
        downloaded = _download(session, attachment, symbol=symbol, name=f"nse_att_{index}{ext}")
        downloads.append(downloaded)
        if not downloaded.get("ok"):
            continue
        file_path = ROOT / str(downloaded["path"])
        try:
            blob = file_path.read_bytes()
        except OSError:
            continue
        extracted = bytes_to_text(blob, ext, max_pages=16)
        if extracted.strip():
            bodies.append((extracted, attachment))
    return {
        "step": {
            "id": "nse_filings",
            "ok": any(d.get("ok") for d in downloads),
            "rows": rows_kept,
            "files": len(downloads),
            "picked": [row[2] for row in picked],
        },
        "downloads": downloads,
        "texts": bodies,
        "headlines": headlines,
    }


def _fetch_annual_reports(symbol: str, session) -> dict[str, Any]:
    url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
    downloads: list[dict[str, Any]] = []
    bodies: list[tuple[str, str]] = []
    item = _download(session, url, symbol=symbol, name="nse_annual_reports.json")
    downloads.append(item)
    rows: list[Mapping[str, Any]] = []
    if item.get("ok"):
        path = ROOT / str(item["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, Mapping)]
        elif isinstance(payload, dict):
            raw_rows = payload.get("data") or payload.get("annualReports") or payload.get("reports") or []
            rows = [row for row in list(raw_rows) if isinstance(row, Mapping)]
            if not rows and payload.get("fileName"):
                rows = [payload]
    picked: list[str] = []
    seen: set[str] = set()
    for row in rows[:8]:
        attachment = str(row.get("fileName") or row.get("attchmntFile") or row.get("file") or "")
        if not attachment:
            continue
        if not attachment.startswith("http"):
            attachment = urljoin("https://nsearchives.nseindia.com/annual_reports/", attachment)
        if not _allowed(attachment) or attachment in seen:
            continue
        seen.add(attachment)
        picked.append(attachment)
        if len(picked) >= 2:
            break
    for index, attachment in enumerate(picked, start=1):
        ext = Path(urlparse(attachment).path).suffix.lower() or ".pdf"
        downloaded = _download(session, attachment, symbol=symbol, name=f"nse_ar_{index}{ext}", max_bytes=32_000_000)
        downloads.append(downloaded)
        if not downloaded.get("ok"):
            continue
        file_path = ROOT / str(downloaded["path"])
        try:
            blob = file_path.read_bytes()
        except OSError:
            continue
        extracted = bytes_to_text(blob, ext, max_pages=40)
        if extracted.strip():
            bodies.append((extracted, attachment))
    return {
        "step": {
            "id": "nse_annual_reports",
            "ok": any(d.get("ok") for d in downloads),
            "rows": len(rows),
            "picked": picked,
        },
        "downloads": downloads,
        "texts": bodies,
    }


def _fetch_option_chain(symbol: str, session) -> dict[str, Any]:
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    snapshot: dict[str, Any] = {
        "available": False,
        "source": "NSE option-chain-equities",
        "source_url": url,
        "not_a_signal": True,
        "places_orders": False,
    }
    downloaded: dict[str, Any] = {"ok": False, "url": url, "error": "not attempted"}
    last_error = ""
    try:
        session.get("https://www.nseindia.com/option-chain", timeout=12)
        session.get(f"https://www.nseindia.com/option-chain?symbol={symbol}", timeout=12)
    except Exception:
        pass
    for attempt in range(3):
        if attempt:
            time.sleep(0.6 * attempt)
        downloaded = _download(session, url, symbol=symbol, name="nse_option_chain.json")
        if not downloaded.get("ok"):
            last_error = str(downloaded.get("error") or "download failed")
            if "HTTP 401" in last_error or "HTTP 403" in last_error:
                try:
                    session.get("https://www.nseindia.com/option-chain", timeout=12)
                except Exception:
                    pass
            continue
        path = ROOT / str(downloaded["path"])
        try:
            blob = path.read_text(encoding="utf-8", errors="ignore")
            payload = json.loads(blob)
        except (OSError, json.JSONDecodeError):
            last_error = "Option-chain response was not JSON."
            continue
        if not isinstance(payload, dict) or payload == {}:
            last_error = "NSE returned an empty option-chain payload (no contracts, or the API stubbed this session)."
            continue
        snapshot = summarize_option_chain(payload, source_url=url)
        if snapshot.get("available"):
            last_error = ""
            break
        last_error = str(snapshot.get("reason") or "NSE returned no option-chain rows for this symbol.")
    if last_error and not snapshot.get("available"):
        snapshot["reason"] = last_error
    snapshot["acquired"] = True
    return {
        "step": {
            "id": "option_chain",
            "ok": bool(snapshot.get("available")),
            "expiry": snapshot.get("expiry"),
            "error": None if snapshot.get("available") else snapshot.get("reason"),
        },
        "download": downloaded,
        "snapshot": snapshot,
    }


def _news_snippets(symbol: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            for item in store.recent(hours=24 * 90, limit=20, symbol=symbol):
                payload = item.as_dict() if hasattr(item, "as_dict") else dict(item)
                text = " ".join(
                    str(payload.get(k) or "")
                    for k in ("headline", "summary", "why_it_matters")
                )
                out.append((text, str(payload.get("source") or "curator"), str(payload.get("url") or "")))
        finally:
            store.close()
    except Exception:
        return out
    return out


def acquire_symbol(symbol: str, *, force: bool = True, now: datetime | None = None) -> dict[str, Any]:
    """Download official sources for one symbol, extract facts, persist, never invent."""
    from product.stock_workspace import clean_symbol

    symbol = clean_symbol(symbol)
    now = now or datetime.now(timezone.utc)
    steps: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []

    screener = _fetch_screener(symbol, force=force)
    steps.append(screener["step"])
    raw = dict(screener.get("data") or {})

    session = None
    try:
        session = _nse_session()
    except Exception as exc:
        session = None
        nse_error = str(exc)[:240]
    else:
        nse_error = ""
    if session is None:
        nse = _empty_nse()
        if nse_error:
            nse["step"]["error"] = nse_error
        annual = {"step": {"id": "nse_annual_reports", "ok": False, "error": nse_error or "no NSE session"}, "downloads": [], "texts": []}
        chain = {
            "step": {"id": "option_chain", "ok": False, "error": nse_error or "no NSE session"},
            "download": {},
            "snapshot": {"available": False, "acquired": False, "reason": nse_error or "no NSE session", "not_a_signal": True, "places_orders": False},
        }
    else:
        try:
            nse = _fetch_nse(symbol, session)
        except Exception as exc:
            nse = _empty_nse()
            nse["step"]["error"] = str(exc)[:240]
        try:
            annual = _fetch_annual_reports(symbol, session)
        except Exception as exc:
            annual = {"step": {"id": "nse_annual_reports", "ok": False, "error": str(exc)[:240]}, "downloads": [], "texts": []}
        try:
            chain = _fetch_option_chain(symbol, session)
        except Exception as exc:
            chain = {
                "step": {"id": "option_chain", "ok": False, "error": str(exc)[:240]},
                "download": {},
                "snapshot": {"available": False, "acquired": True, "reason": str(exc)[:240], "not_a_signal": True, "places_orders": False},
            }
    steps.append(nse["step"])
    steps.append(annual["step"])
    steps.append(chain["step"])
    downloads.extend(nse.get("downloads") or [])
    downloads.extend(annual.get("downloads") or [])
    if chain.get("download"):
        downloads.append(chain["download"])

    text_kpis: dict[str, dict[str, Any]] = {}
    guidance: list[dict[str, Any]] = []
    commentary: list[dict[str, Any]] = []
    order_book: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    def _ingest(text: str, url: str, source: str) -> None:
        nonlocal text_kpis, guidance, commentary, order_book, segments
        parsed = extract_research_pack(text, source=source, source_url=url)
        text_kpis = merge_kpi_maps(text_kpis, parsed.get("kpis") or {})
        guidance.extend(parsed.get("guidance") or [])
        commentary = _extend_unique(commentary, list(parsed.get("commentary") or []), lambda row: (row.get("commentary") or "")[:80], 6)
        order_book = _extend_unique(order_book, list(parsed.get("order_book") or []), lambda row: (row.get("metric"), row.get("value")), 6)
        segments = _extend_unique(segments, list(parsed.get("segments") or []), lambda row: str(row.get("segment") or "").lower(), 6)

    for text, url in nse.get("texts") or []:
        _ingest(text, url, "NSE filing")
    for text, url in annual.get("texts") or []:
        _ingest(text, url, "NSE annual report")
    for text, url in nse.get("headlines") or []:
        guidance.extend(extract_guidance(text, source="NSE announcement", source_url=url))

    uploads = extract_from_uploads(symbol)
    for item in uploads.get("guidance") or []:
        guidance.append(item)
    commentary = _extend_unique(list(uploads.get("commentary") or []), commentary, lambda row: (row.get("commentary") or "")[:80], 6)
    order_book = _extend_unique(list(uploads.get("order_book") or []), order_book, lambda row: (row.get("metric"), row.get("value")), 6)
    segments = _extend_unique(list(uploads.get("segments") or []), segments, lambda row: str(row.get("segment") or "").lower(), 6)

    news_guidance: list[dict[str, Any]] = []
    for text, source, url in _news_snippets(symbol):
        news_guidance.extend(extract_guidance(text, source=f"Curated news ({source})", source_url=url))

    kpis = merge_kpi_maps(
        extract_kpis_from_raw(raw),
        uploads.get("kpis") or {},
        text_kpis,
    )
    gnpa = (kpis.get("gnpa") or {}).get("current")
    nnpa = (kpis.get("nnpa") or {}).get("current")
    if gnpa is not None and nnpa is not None and nnpa > gnpa:
        kpis.pop("nnpa", None)
    seen = set()
    unique_guidance = []
    for item in guidance + news_guidance:
        key = (item.get("excerpt"), item.get("source"))
        if key in seen:
            continue
        seen.add(key)
        unique_guidance.append(item)
        if len(unique_guidance) >= 8:
            break

    option_chain = dict(chain.get("snapshot") or {})
    missing = [key for key in ("gnpa", "nnpa", "pledge") if key not in kpis]
    if not commentary:
        missing.append("commentary")
    if not order_book:
        missing.append("order_book")
    if not segments:
        missing.append("segments")
    if not option_chain.get("available"):
        missing.append("option_chain")
    payload = {
        "schema_version": 2,
        "symbol": symbol,
        "acquired_at": now.isoformat(),
        "method": "autonomy_download",
        "not_an_llm": True,
        "kpis": kpis,
        "guidance": unique_guidance,
        "commentary": commentary,
        "order_book": order_book,
        "segments": segments,
        "option_chain": option_chain,
        "downloads": downloads,
        "steps": steps,
        "still_missing": missing,
        "files_on_disk": [d.get("path") for d in downloads if d.get("ok")],
        "places_orders": False,
    }
    save_autonomy_facts(symbol, payload)
    return payload


def acquire_shortlist(
    *,
    limit: int = ACQUIRE_CAP,
    force: bool = False,
    scan_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    symbols = shortlist_symbols(limit=limit, scan_payload=scan_payload)
    results = []
    errors = []
    for symbol in symbols:
        try:
            results.append(acquire_symbol(symbol, force=force))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:240]})
        time.sleep(0.2)
    return {
        "accepted": True,
        "symbols": symbols,
        "acquired": [row.get("symbol") for row in results],
        "errors": errors,
        "n_ok": len(results),
        "n_failed": len(errors),
        "places_orders": False,
        "gates_scanner": False,
    }
