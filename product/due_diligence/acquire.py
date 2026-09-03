"""Find, download and persist Investigate inputs. GET never calls this."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
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
    "www.moneycontrol.com",
    "moneycontrol.com",
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


def _recommendation_shortlist(limit: int = ACQUIRE_CAP) -> list[str]:
    """Prefer recommendation-worthy names over raw scan rank. Cheap first."""
    try:
        from product.recommendations_store import load_recommendations
        from product.autopilot_journal import flatten_cards

        reco = load_recommendations() or {}
        cards = [c for c in flatten_cards(reco) if isinstance(c, dict)]
    except Exception:
        return []
    ranked = [
        c for c in cards
        if str(c.get("reco_tier") or "") in {"high_conviction", "good_setup"}
    ]
    ranked.sort(key=lambda c: (str(c.get("reco_tier")) != "high_conviction", str(c.get("symbol") or "")))
    out: list[str] = []
    for card in ranked:
        symbol = str(card.get("symbol") or "").upper()
        if symbol and symbol not in out:
            out.append(symbol)
        if len(out) >= int(limit):
            break
    return out


def shortlist_symbols(limit: int = ACQUIRE_CAP, scan_payload: Mapping[str, Any] | None = None) -> list[str]:
    """Names the desk already shortlisted. Does not scan the market."""
    if scan_payload is None:
        reco_names = _recommendation_shortlist(limit)
        if reco_names:
            return reco_names
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
    if any(tok in low for tok in ("financial result", "results for the period", "un-audited financial", "audited financial", "quarterly result")):
        return 0
    if "investor presentation" in low or "earnings presentation" in low or "investor deck" in low:
        return 1
    if "transcript" in low or "concall" in low or "conference call" in low:
        return 2
    if "press release" in low and any(tok in low for tok in ("result", "profit", "npa")):
        return 3
    if "basel" in low or "pillar 3" in low or "pillar-3" in low:
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


def _document_type(source: str, url: str = "") -> str:
    blob = f"{source} {url}".lower()
    if "investor presentation" in blob or "earnings presentation" in blob:
        return "investor_presentation"
    if "annual report" in blob:
        return "annual_report"
    if "nse" in blob or "filing" in blob or "result" in blob:
        return "exchange_filing"
    return "unknown"


def _archive_texts(symbol: str) -> list[tuple[str, str, str]]:
    """Re-read already-downloaded filings. Does not hit the network."""
    folder = _symbol_dir(symbol)
    if not folder.exists():
        return []
    out: list[tuple[str, str, str]] = []
    files = sorted(folder.glob("nse_att_*")) + sorted(folder.glob("nse_ar_*"))
    for path in files:
        ext = path.suffix.lower()
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        max_pages = 40 if "ar_" in path.name else 28
        text = bytes_to_text(blob, ext, max_pages=max_pages)
        if not text.strip():
            continue
        source = "NSE annual report" if "ar_" in path.name else "NSE filing"
        out.append((text, str(path), source))
    return out


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
        extracted = bytes_to_text(blob, ext, max_pages=28)
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


def _news_items(symbol: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            for item in store.recent(hours=24 * 90, limit=40, symbol=symbol):
                payload = item.as_dict() if hasattr(item, "as_dict") else dict(item)
                if isinstance(payload, dict):
                    items.append(payload)
        finally:
            store.close()
    except Exception:
        return items
    return items


def _framework_id_for(symbol: str, raw: Mapping[str, Any]) -> str:
    from product.due_diligence.classify import classify_company

    profile = classify_company(
        symbol,
        sector=str(raw.get("sector") or ""),
        about=str(raw.get("about") or ""),
        quarterly_rows=list(raw.get("quarterly_results") or []),
    )
    return str(profile.get("framework_id") or "generic")


def _framework_for(symbol: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    from product.due_diligence.frameworks import get_framework
    return get_framework(_framework_id_for(symbol, raw))


_ACQUIRE_LANE_DATASETS = {
    "financials": ("quarterly_results", "annual_financials", "shareholding", "promoter_pledge"),
    "ir": ("exchange_filings", "sector_kpis", "valuation", "peer_data"),
    "filings": ("exchange_filings", "corporate_announcements", "credit_ratings"),
    "news": ("recent_news",),
}


def _order_to_fetch(to_fetch: list[str], acquire_priority: Sequence[str] | None) -> list[str]:
    if not to_fetch:
        return []
    order: list[str] = []
    seen: set[str] = set()
    for lane in acquire_priority or ():
        for ds_id in _ACQUIRE_LANE_DATASETS.get(str(lane), ()):
            if ds_id in to_fetch and ds_id not in seen:
                order.append(ds_id)
                seen.add(ds_id)
    for ds_id in to_fetch:
        if ds_id not in seen:
            order.append(ds_id)
    return order


def inspect_symbol_coverage(symbol: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Cache-only dataset inventory used by the acquire planner."""
    from product.due_diligence.coverage import inspect_research_coverage
    from reporting.evidence_intake import load_raw_fundamentals

    raw_record = load_raw_fundamentals(symbol) or {}
    raw = dict(raw_record.get("data") or {})
    facts = load_autonomy_facts(symbol)
    measured = merge_kpi_maps(extract_kpis_from_raw(raw), dict(facts.get("kpis") or {}))
    findings = [
        {"id": key, "available": True, "latest": snap.get("current")}
        for key, snap in measured.items()
        if isinstance(snap, Mapping)
    ]
    return inspect_research_coverage(
        symbol=symbol,
        raw=raw,
        autonomy=facts,
        news=_news_items(symbol),
        framework_id=_framework_id_for(symbol, raw),
        findings=findings,
        events=list(facts.get("announcements") or []),
        fetched_at=str(raw_record.get("fetched_at") or ""),
        now=now,
    )


def plan_acquire(
    symbol: str,
    *,
    force: bool = False,
    datasets: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decide which provider lanes to run. Does not hit the internet."""
    from product.due_diligence.coverage import DATASET_IDS, REQUIRED_FOR_COVERAGE, provider_lanes

    coverage = inspect_symbol_coverage(symbol, now=now)
    if datasets:
        to_fetch = [item for item in datasets if item in DATASET_IDS or item == "option_chain"]
    elif force:
        to_fetch = list(REQUIRED_FOR_COVERAGE)
    else:
        to_fetch = list(coverage.get("to_fetch") or [])
    try:
        from reporting.evidence_intake import load_raw_fundamentals
        raw_record = load_raw_fundamentals(symbol) or {}
        raw = dict(raw_record.get("data") or {})
        to_fetch = _order_to_fetch(to_fetch, _framework_for(symbol, raw).get("acquire_priority"))
    except Exception:
        pass
    lanes = provider_lanes(to_fetch)
    lanes["option_chain"] = bool(force) or "option_chain" in (datasets or [])
    return {
        "to_fetch": to_fetch,
        "lanes": lanes,
        "coverage": coverage,
        "force": bool(force),
    }


def _stamp_meta(
    meta: dict[str, Any],
    dataset_id: str,
    *,
    now: datetime,
    status: str,
    provider: str = "",
    error: str = "",
    fetched: bool = False,
) -> None:
    prev = dict(meta.get(dataset_id) or {})
    row = {
        **prev,
        "checked_at": now.isoformat(),
        "status": status,
        "provider": provider or prev.get("provider") or "",
    }
    if fetched:
        row["fetched_at"] = now.isoformat()
    if error:
        row["error"] = error[:240]
    elif status in {"current", "stale"}:
        row.pop("error", None)
    meta[dataset_id] = row


def _announcements_from_headlines(headlines: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for text, url in headlines:
        headline = re.sub(r"\s+", " ", str(text or "")).strip()
        if not headline:
            continue
        out.append({
            "headline": headline[:280],
            "url": url,
            "source": "NSE announcements",
            "source_kind": "exchange",
        })
        if len(out) >= 40:
            break
    return out


def _ratings_from_announcements(announcements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in announcements:
        blob = str(item.get("headline") or "").lower()
        if "rating" in blob and any(
            tok in blob for tok in ("credit", "outlook", "upgrade", "downgrade", "crisil", "icra", "care", "fitch")
        ):
            out.append({**item, "kind": "credit_rating"})
    return out


def acquire_symbol(
    symbol: str,
    *,
    force: bool = False,
    datasets: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Download only the datasets that are missing or stale. Never invent numbers."""
    from product.stock_workspace import clean_symbol

    symbol = clean_symbol(symbol)
    now = now or datetime.now(timezone.utc)
    previous = load_autonomy_facts(symbol)
    try:
        plan = plan_acquire(symbol, force=force, datasets=datasets, now=now)
    except Exception as exc:
        from product.due_diligence.coverage import REQUIRED_FOR_COVERAGE, provider_lanes

        to_fetch = list(datasets or REQUIRED_FOR_COVERAGE)
        plan = {
            "to_fetch": to_fetch,
            "lanes": {**provider_lanes(to_fetch), "option_chain": bool(force)},
            "coverage": {},
            "force": bool(force),
            "plan_error": str(exc)[:240],
        }
    lanes = dict(plan["lanes"])
    to_fetch = list(plan["to_fetch"])
    steps: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = list(previous.get("downloads") or []) if not force else []
    dataset_meta: dict[str, Any] = dict(previous.get("dataset_meta") or {})
    skipped: list[str] = []

    raw: dict[str, Any] = {}
    if lanes.get("screener"):
        screener = _fetch_screener(symbol, force=force or "quarterly_results" in to_fetch)
        steps.append(screener["step"])
        raw = dict(screener.get("data") or {})
        screener_ok = bool(screener["step"].get("ok"))
        screener_err = str(screener["step"].get("error") or screener["step"].get("error_on_refresh") or "")
        status = "current" if screener_ok else ("acquisition_failed" if screener_err else "source_unavailable")
        for ds_id in (
            "company_master", "quarterly_results", "annual_financials",
            "shareholding", "promoter_pledge", "valuation", "peer_data",
        ):
            if ds_id in to_fetch or force:
                _stamp_meta(
                    dataset_meta, ds_id, now=now, status=status,
                    provider="screener.in", error=screener_err, fetched=True,
                )
    else:
        from reporting.evidence_intake import load_raw_fundamentals

        raw = dict((load_raw_fundamentals(symbol) or {}).get("data") or {})
        steps.append({"id": "screener", "ok": True, "skipped": True, "reason": "datasets current"})
        skipped.append("screener")
        for ds_id in (
            "company_master", "quarterly_results", "annual_financials",
            "shareholding", "promoter_pledge", "valuation", "peer_data",
        ):
            if ds_id not in to_fetch:
                _stamp_meta(dataset_meta, ds_id, now=now, status="current", provider="cache")

    nse = dict(previous.get("_nse_cache") or {}) or _empty_nse()
    annual = {"step": {"id": "nse_annual_reports", "ok": False, "skipped": True}, "downloads": [], "texts": []}
    chain = {
        "step": {"id": "option_chain", "ok": False, "skipped": True},
        "download": {},
        "snapshot": dict(previous.get("option_chain") or {})
        or {"available": False, "acquired": False, "not_a_signal": True, "places_orders": False},
    }
    nse_error = ""
    session = None
    need_nse = lanes.get("nse_filings") or lanes.get("nse_annual") or lanes.get("option_chain")
    if need_nse:
        try:
            session = _nse_session()
        except Exception as exc:
            session = None
            nse_error = str(exc)[:240]
        if session is None:
            if lanes.get("nse_filings"):
                nse = _empty_nse()
                nse["step"]["error"] = nse_error or "no NSE session"
                nse["step"]["status"] = "source_unavailable"
            if lanes.get("nse_annual"):
                annual = {
                    "step": {
                        "id": "nse_annual_reports",
                        "ok": False,
                        "error": nse_error or "no NSE session",
                        "status": "source_unavailable",
                    },
                    "downloads": [],
                    "texts": [],
                }
            if lanes.get("option_chain"):
                chain = {
                    "step": {"id": "option_chain", "ok": False, "error": nse_error or "no NSE session"},
                    "download": {},
                    "snapshot": {
                        "available": False,
                        "acquired": False,
                        "reason": nse_error or "no NSE session",
                        "not_a_signal": True,
                        "places_orders": False,
                    },
                }
        else:
            if lanes.get("nse_filings"):
                try:
                    nse = _fetch_nse(symbol, session)
                except Exception as exc:
                    nse = _empty_nse()
                    nse["step"]["error"] = str(exc)[:240]
            else:
                nse = _empty_nse()
                nse["step"]["skipped"] = True
                skipped.append("nse_filings")
            if lanes.get("nse_annual"):
                try:
                    annual = _fetch_annual_reports(symbol, session)
                except Exception as exc:
                    annual = {"step": {"id": "nse_annual_reports", "ok": False, "error": str(exc)[:240]}, "downloads": [], "texts": []}
            else:
                skipped.append("nse_annual")
            if lanes.get("option_chain"):
                try:
                    chain = _fetch_option_chain(symbol, session)
                except Exception as exc:
                    chain = {
                        "step": {"id": "option_chain", "ok": False, "error": str(exc)[:240]},
                        "download": {},
                        "snapshot": {
                            "available": False,
                            "acquired": True,
                            "reason": str(exc)[:240],
                            "not_a_signal": True,
                            "places_orders": False,
                        },
                    }
            else:
                skipped.append("option_chain")
    else:
        skipped.extend(["nse_filings", "nse_annual", "option_chain"])
        steps.append({"id": "nse_filings", "ok": True, "skipped": True, "reason": "datasets current"})
        steps.append({"id": "nse_annual_reports", "ok": True, "skipped": True, "reason": "datasets current"})
        steps.append({"id": "option_chain", "ok": True, "skipped": True, "reason": "not requested"})

    if need_nse:
        steps.append(nse["step"])
        steps.append(annual["step"])
        steps.append(chain["step"])
        if nse.get("downloads"):
            downloads.extend(nse.get("downloads") or [])
        if annual.get("downloads"):
            downloads.extend(annual.get("downloads") or [])
        if chain.get("download"):
            downloads.append(chain["download"])

    filing_status = "current" if nse.get("step", {}).get("ok") else (
        "source_unavailable" if (nse.get("step") or {}).get("status") == "source_unavailable" or nse_error
        else "acquisition_failed" if (nse.get("step") or {}).get("error") else "current"
    )
    if lanes.get("nse_filings"):
        err = str((nse.get("step") or {}).get("error") or nse_error or "")
        if not nse.get("step", {}).get("ok") and not err:
            err = "NSE filings were not downloaded"
        for ds_id in ("exchange_filings", "corporate_announcements"):
            _stamp_meta(
                dataset_meta, ds_id, now=now,
                status="current" if nse.get("step", {}).get("ok") else (filing_status if err else "not_yet_acquired"),
                provider="nseindia.com", error=err, fetched=True,
            )
    if lanes.get("nse_annual"):
        err = str((annual.get("step") or {}).get("error") or "")
        _stamp_meta(
            dataset_meta, "annual_financials", now=now,
            status="current" if annual.get("step", {}).get("ok") or _quarterly_has_values({"quarterly_results": raw.get("profit_loss") or []}) else (
                "acquisition_failed" if err else "source_unavailable"
            ),
            provider="nseindia.com", error=err, fetched=True,
        )

    text_kpis: dict[str, dict[str, Any]] = dict(previous.get("kpis") or {}) if not force else {}
    if lanes.get("nse_filings") or lanes.get("nse_annual"):
        text_kpis = {
            key: value
            for key, value in text_kpis.items()
            if "nse" not in str(value.get("source") or "").lower()
            and "filing" not in str(value.get("source") or "").lower()
            and "annual report" not in str(value.get("source") or "").lower()
        }
    guidance: list[dict[str, Any]] = list(previous.get("guidance") or []) if not force else []
    commentary: list[dict[str, Any]] = list(previous.get("commentary") or []) if not force else []
    order_book: list[dict[str, Any]] = list(previous.get("order_book") or []) if not force else []
    segments: list[dict[str, Any]] = list(previous.get("segments") or []) if not force else []

    def _ingest(text: str, url: str, source: str) -> None:
        nonlocal text_kpis, guidance, commentary, order_book, segments
        try:
            parsed = extract_research_pack(
                text,
                source=source,
                source_url=url,
                document_type=_document_type(source, url),
            )
        except Exception as exc:
            steps.append({"id": "parse", "ok": False, "source": source, "error": str(exc)[:240]})
            return
        text_kpis = merge_kpi_maps(text_kpis, parsed.get("kpis") or {})
        if "annual report" not in source.lower():
            guidance.extend(parsed.get("guidance") or [])
        commentary = _extend_unique(commentary, list(parsed.get("commentary") or []), lambda row: (row.get("commentary") or "")[:80], 6)
        order_book = _extend_unique(order_book, list(parsed.get("order_book") or []), lambda row: (row.get("metric"), row.get("value")), 6)
        segments = _extend_unique(segments, list(parsed.get("segments") or []), lambda row: str(row.get("segment") or "").lower(), 6)

    for text, url in nse.get("texts") or []:
        _ingest(text, url, "NSE filing")
    for text, url in annual.get("texts") or []:
        _ingest(text, url, "NSE annual report")
    for text, path, source in _archive_texts(symbol):
        _ingest(text, path, source)
    nse_headlines = list(nse.get("headlines") or [])
    for text, url in nse_headlines:
        guidance.extend(extract_guidance(text, source="NSE announcement", source_url=url))

    announcements = list(previous.get("announcements") or [])
    if nse_headlines:
        announcements = _announcements_from_headlines(nse_headlines)
    credit_ratings = _ratings_from_announcements(announcements)
    if lanes.get("nse_filings"):
        _stamp_meta(
            dataset_meta, "credit_ratings", now=now,
            status="current" if credit_ratings else "not_yet_acquired",
            provider="nseindia.com", fetched=True,
        )

    uploads = extract_from_uploads(symbol)
    for item in uploads.get("guidance") or []:
        guidance.append(item)
    commentary = _extend_unique(list(uploads.get("commentary") or []), commentary, lambda row: (row.get("commentary") or "")[:80], 6)
    order_book = _extend_unique(list(uploads.get("order_book") or []), order_book, lambda row: (row.get("metric"), row.get("value")), 6)
    segments = _extend_unique(list(uploads.get("segments") or []), segments, lambda row: str(row.get("segment") or "").lower(), 6)

    news_guidance: list[dict[str, Any]] = []
    news_items = _news_snippets(symbol)
    if lanes.get("news") or True:
        for text, source, url in news_items:
            news_guidance.extend(extract_guidance(text, source=f"Curated news ({source})", source_url=url))
        _stamp_meta(
            dataset_meta, "recent_news", now=now,
            status="current" if news_items else "not_yet_acquired",
            provider="news_curator", fetched=False,
        )

    kpis = merge_kpi_maps(
        extract_kpis_from_raw(raw),
        uploads.get("kpis") or {},
        text_kpis,
    )

    framework = _framework_for(symbol, raw)
    if (
        lanes.get("sector_fallback")
        and framework.get("lending")
        and not any(kpis.get(key) for key in ("casa", "cet1", "gnpa", "nim", "pcr", "crar"))
    ):
        try:
            from product.due_diligence.providers.moneycontrol import fetch_moneycontrol_kpis

            mc = fetch_moneycontrol_kpis(symbol)
            steps.append({
                "id": "moneycontrol",
                "ok": bool(mc.get("ok")),
                "error": mc.get("error") or None,
                "status": mc.get("status"),
            })
            if mc.get("kpis"):
                kpis = merge_kpi_maps(kpis, mc.get("kpis") or {})
            _stamp_meta(
                dataset_meta, "sector_kpis", now=now,
                status=str(mc.get("status") or "acquisition_failed"),
                provider="moneycontrol.com",
                error=str(mc.get("error") or ""),
                fetched=True,
            )
        except Exception as exc:
            steps.append({"id": "moneycontrol", "ok": False, "error": str(exc)[:240]})
            _stamp_meta(
                dataset_meta, "sector_kpis", now=now,
                status="acquisition_failed",
                provider="moneycontrol.com",
                error=str(exc)[:240],
                fetched=True,
            )
    elif "sector_kpis" in to_fetch or force:
        sector_ids = [
            spec.id for spec in framework["kpis"]
            if spec.importance in {"critical", "important"} and spec.id not in {
                "pat", "sales", "opm", "eps", "promoter", "pledge", "fii", "dii", "public", "cfo", "roe", "roce", "borrowings",
            }
        ] or [spec.id for spec in framework["kpis"] if spec.importance in {"critical", "important"}]
        sector_ok = any(kpis.get(key) for key in sector_ids)
        _stamp_meta(
            dataset_meta, "sector_kpis", now=now,
            status="current" if sector_ok else "not_yet_acquired",
            provider="nseindia.com / screener.in",
            fetched=bool(lanes.get("nse_filings") or lanes.get("screener")),
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

    option_chain = dict(chain.get("snapshot") or previous.get("option_chain") or {})
    missing = [
        spec.id for spec in framework["kpis"]
        if spec.importance in {"critical", "important"} and spec.id not in kpis
    ][:12]
    if not commentary:
        missing.append("commentary")
    if not order_book:
        missing.append("order_book")
    if not segments:
        missing.append("segments")
    if not option_chain.get("available"):
        missing.append("option_chain")
    acquired_at = now.isoformat() if any(not s.get("skipped") for s in steps if isinstance(s, dict)) else (
        str(previous.get("acquired_at") or now.isoformat())
    )
    payload = {
        "schema_version": 3,
        "symbol": symbol,
        "acquired_at": acquired_at,
        "inspected_at": now.isoformat(),
        "method": "autonomy_download",
        "mode": "all" if force else "missing_or_stale",
        "to_fetch": to_fetch,
        "skipped": skipped,
        "not_an_llm": True,
        "kpis": kpis,
        "guidance": unique_guidance,
        "commentary": commentary,
        "order_book": order_book,
        "segments": segments,
        "option_chain": option_chain,
        "announcements": announcements,
        "filings": [
            {
                "title": d.get("url") or d.get("path"),
                "url": d.get("url"),
                "path": d.get("path"),
                "ok": d.get("ok"),
            }
            for d in downloads
            if isinstance(d, dict)
        ],
        "credit_ratings": credit_ratings,
        "downloads": downloads,
        "steps": steps,
        "dataset_meta": dataset_meta,
        "still_missing": missing,
        "files_on_disk": [d.get("path") for d in downloads if d.get("ok")],
        "places_orders": False,
        "sector": _framework_id_for(symbol, raw),
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
