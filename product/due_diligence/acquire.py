"""Find, download and persist Investigate inputs. GET never calls this."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from product.due_diligence.extract import (
    extract_from_html,
    extract_from_uploads,
    extract_guidance,
    extract_kpis_from_raw,
    extract_rates_from_text,
    html_to_text,
    merge_kpi_maps,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "logs" / "research_evidence"
ACQUIRE_CAP = 6
FACTS_NAME = "autonomy_facts.json"
FRESH_S = 24 * 60 * 60
MAX_ATTACHMENT_BYTES = 4_000_000
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


def _download(session, url: str, *, symbol: str, name: str) -> dict[str, Any]:
    if not _allowed(url):
        return {"ok": False, "url": url, "error": "host not on the official allow-list"}
    try:
        response = session.get(url, timeout=18, allow_redirects=True)
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)[:240]}
    if response.status_code != 200:
        return {"ok": False, "url": url, "error": f"HTTP {response.status_code}"}
    content = response.content or b""
    if len(content) > MAX_ATTACHMENT_BYTES:
        return {"ok": False, "url": url, "error": "file larger than 4 MB — skipped"}
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
    texts: list[tuple[str, str]] = []
    rows_kept = 0
    for index, url in enumerate(urls):
        item = _download(session, url, symbol=symbol, name=f"nse_{index}.json")
        downloads.append(item)
        if not item.get("ok"):
            continue
        path = ROOT / str(item["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            texts.append((html_to_text(path.read_text(encoding="utf-8", errors="ignore")), url))
            continue
        rows = payload if isinstance(payload, list) else payload.get("data") or payload.get("financialResults") or []
        if isinstance(payload, dict) and not rows:
            rows = [payload]
        interesting = 0
        for row in list(rows)[:80]:
            if not isinstance(row, Mapping):
                continue
            row_symbol = str(row.get("symbol") or row.get("sm_symbol") or "").upper()
            if row_symbol and row_symbol != symbol:
                continue
            desc = " ".join(
                str(row.get(k) or "")
                for k in ("attchmntText", "desc", "subject", "details", "remarks", "sm_name")
            )
            texts.append((desc, url))
            rows_kept += 1
            attachment = str(row.get("attchmntFile") or row.get("attachment") or row.get("fileName") or "")
            low = desc.lower()
            want = any(tok in low for tok in ("result", "npa", "transcript", "concall", "investor", "financial"))
            if not attachment or not want or interesting >= 3:
                continue
            if not attachment.startswith("http"):
                attachment = urljoin("https://nsearchives.nseindia.com/corporate/", attachment)
            interesting += 1
            ext = Path(urlparse(attachment).path).suffix.lower() or ".bin"
            downloaded = _download(session, attachment, symbol=symbol, name=f"nse_att_{interesting}{ext}")
            downloads.append(downloaded)
            if downloaded.get("ok"):
                file_path = ROOT / str(downloaded["path"])
                try:
                    blob = file_path.read_bytes()
                except OSError:
                    continue
                if ext in {".html", ".htm", ".txt", ".xml", ".csv", ".json"}:
                    texts.append((blob.decode("utf-8", errors="ignore"), attachment))
    return {
        "step": {"id": "nse_filings", "ok": any(d.get("ok") for d in downloads), "rows": rows_kept, "files": len(downloads)},
        "downloads": downloads,
        "texts": texts,
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

    try:
        session = _nse_session()
        nse = _fetch_nse(symbol, session)
    except Exception as exc:
        nse = {"step": {"id": "nse_filings", "ok": False, "error": str(exc)[:240]}, "downloads": [], "texts": []}
    steps.append(nse["step"])
    downloads.extend(nse.get("downloads") or [])

    text_kpis: dict[str, dict[str, Any]] = {}
    guidance: list[dict[str, Any]] = []
    for text, url in nse.get("texts") or []:
        parsed = extract_from_html(text, source="NSE filing / announcement", source_url=url)
        text_kpis = merge_kpi_maps(text_kpis, parsed.get("kpis") or {})
        guidance.extend(parsed.get("guidance") or [])
        extra = extract_rates_from_text(text, source="NSE filing / announcement", source_url=url)
        text_kpis = merge_kpi_maps(text_kpis, extra)

    uploads = extract_from_uploads(symbol)
    for item in uploads.get("guidance") or []:
        guidance.append(item)

    news_guidance: list[dict[str, Any]] = []
    for text, source, url in _news_snippets(symbol):
        news_guidance.extend(extract_guidance(text, source=f"Curated news ({source})", source_url=url))

    kpis = merge_kpi_maps(
        extract_kpis_from_raw(raw),
        uploads.get("kpis") or {},
        text_kpis,
    )
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

    missing = [key for key in ("gnpa", "nnpa", "pledge") if key not in kpis]
    payload = {
        "schema_version": 1,
        "symbol": symbol,
        "acquired_at": now.isoformat(),
        "method": "autonomy_download",
        "not_an_llm": True,
        "kpis": kpis,
        "guidance": unique_guidance,
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
