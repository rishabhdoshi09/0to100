"""Research evidence intake, freshness and user guidance for QuantTerm.

The intake layer distinguishes three states rigorously:
- usable structured evidence,
- a source document attached but not extracted,
- evidence missing.

An uploaded PDF is preserved and checksum-traced, but it cannot satisfy an analytical
section until structured extraction exists. CSV/JSON uploads are schema validated and
must carry a source URL and an explicit as-of date.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "logs" / "research_evidence"
FUNDAMENTALS_DB = ROOT / "data" / "fundamentals_cache.db"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class ResourceSpec:
    key: str
    label: str
    why: str
    instructions: str
    accepted_extensions: tuple[str, ...]
    max_age_days: int
    template_columns: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()


RESOURCE_SPECS: dict[str, ResourceSpec] = {
    "business_profile": ResourceSpec(
        "business_profile", "Business description",
        "Explains what the company sells, who pays, and which demand drivers matter.",
        "Download the latest annual report or investor presentation. Upload the original PDF as a source, or fill the structured business-profile CSV.",
        (".pdf", ".csv", ".json", ".txt"), 450,
        ("as_of_date", "business_summary", "customers", "demand_drivers", "source_url"),
        ("as_of_date", "business_summary", "source_url"),
    ),
    "financial_history": ResourceSpec(
        "financial_history", "Financial history",
        "Supports quarterly and annual revenue, margin, profit, cash-flow and balance-sheet analysis.",
        "Use NSE/BSE financial results, the annual report, or Screener. Prefer consolidated figures and preserve every period-end date.",
        (".csv", ".xlsx", ".xls", ".pdf", ".json"), 120,
        ("period_end", "period_type", "revenue_cr", "ebitda_cr", "ebitda_margin_pct", "pat_cr", "cfo_cr", "debt_cr", "source_url", "as_of_date"),
        ("period_end", "period_type", "revenue_cr", "pat_cr", "source_url", "as_of_date"),
    ),
    "shareholding_history": ResourceSpec(
        "shareholding_history", "Shareholding history",
        "Allows promoter, FII, DII and public ownership changes to be measured quarter by quarter.",
        "Download exchange shareholding-pattern CSV/XBRL for at least four quarters, or fill the QuantTerm CSV template.",
        (".csv", ".xlsx", ".xls", ".xml", ".pdf", ".json"), 120,
        ("quarter_end", "promoter_pct", "fii_pct", "dii_pct", "public_pct", "promoter_pledge_pct", "source_url", "as_of_date"),
        ("quarter_end", "promoter_pct", "fii_pct", "dii_pct", "source_url", "as_of_date"),
    ),
    "business_segments": ResourceSpec(
        "business_segments", "Business and segment mix",
        "Shows where revenue and profit come from instead of treating the company as one undifferentiated number.",
        "Use the segment note in the annual report, investor presentation or results filing. Enter one segment per row.",
        (".csv", ".xlsx", ".xls", ".pdf", ".json"), 450,
        ("period_end", "segment", "revenue_cr", "revenue_mix_pct", "growth_pct", "margin_pct", "driver", "source_url", "as_of_date"),
        ("period_end", "segment", "revenue_cr", "source_url", "as_of_date"),
    ),
    "management_commentary": ResourceSpec(
        "management_commentary", "Management commentary and guidance",
        "Separates management guidance, risks and operating explanations from editorial summaries.",
        "Download an earnings-call transcript, exchange-filed presentation or official call summary. Structured rows must identify speaker, date and source.",
        (".pdf", ".txt", ".csv", ".json", ".vtt", ".srt"), 120,
        ("event_date", "speaker", "topic", "commentary", "guidance_metric", "guidance_value", "guidance_period", "source_url", "as_of_date"),
        ("event_date", "speaker", "commentary", "source_url", "as_of_date"),
    ),
    "order_book_guidance": ResourceSpec(
        "order_book_guidance", "Order book and forward guidance",
        "Supports forward-visibility claims with dated company disclosures rather than narrative inference.",
        "Use an exchange announcement, results presentation or earnings-call transcript. Upload the source or fill the guidance template.",
        (".csv", ".xlsx", ".xls", ".pdf", ".json"), 180,
        ("as_of_date", "metric", "value", "unit", "period", "management_wording", "source_url"),
        ("as_of_date", "metric", "value", "source_url"),
    ),
    "annual_report": ResourceSpec(
        "annual_report", "Annual report",
        "Primary source for the business model, segment disclosures, risks, governance and audited financials.",
        "Download the latest annual report from NSE/BSE or the company investor-relations page and upload the original PDF.",
        (".pdf",), 450,
    ),
}


def clean_symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Z0-9&.-]", "", str(value or "").strip().upper())
    if not symbol or len(symbol) > 32:
        raise ValueError("invalid NSE symbol")
    return symbol


def _valid_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resource_links(symbol: str) -> dict[str, list[dict[str, str]]]:
    symbol = clean_symbol(symbol)
    nse_announcements = "https://www.nseindia.com/companies-listing/corporate-filings-application?id=allAnnouncements"
    nse_financials = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
    nse_shareholding = f"https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern?symbol={symbol}"
    nse_annual = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
    bse_announcements = "https://www.bseindia.com/corporates/ann.html"
    bse_financials = "https://www.bseindia.com/corporates/comp_results.aspx"
    bse_shareholding = "https://www.bseindia.com/corporates/Sharehold_Searchnew.aspx"
    screener = f"https://www.screener.in/company/{symbol}/consolidated/"
    company_search = f"https://www.google.com/search?q={symbol}+investor+relations+annual+report+earnings+call+transcript"
    official_nse_annual = {"label": "NSE annual reports", "url": nse_annual, "official": "true"}
    ir_search = {"label": "Company investor-relations search", "url": company_search, "official": "false"}
    return {
        "business_profile": [official_nse_annual, ir_search, {"label": "Screener company page", "url": screener, "official": "false"}],
        "financial_history": [
            {"label": "NSE financial results", "url": nse_financials, "official": "true"},
            {"label": "BSE financial results", "url": bse_financials, "official": "true"},
            official_nse_annual,
            {"label": "Screener company page", "url": screener, "official": "false"},
        ],
        "shareholding_history": [
            {"label": "NSE shareholding patterns", "url": nse_shareholding, "official": "true"},
            {"label": "BSE shareholding search", "url": bse_shareholding, "official": "true"},
        ],
        "business_segments": [official_nse_annual, ir_search],
        "management_commentary": [
            {"label": "NSE announcements and call filings", "url": nse_announcements, "official": "true"},
            {"label": "BSE announcements", "url": bse_announcements, "official": "true"},
            ir_search,
        ],
        "order_book_guidance": [
            {"label": "NSE corporate announcements", "url": nse_announcements, "official": "true"},
            {"label": "BSE announcements", "url": bse_announcements, "official": "true"},
            {"label": "NSE financial results", "url": nse_financials, "official": "true"},
            ir_search,
        ],
        "annual_report": [official_nse_annual, ir_search],
    }


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:11], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def age_days(as_of: Any, *, now: datetime | None = None) -> int | None:
    stamp = _parse_datetime(as_of)
    if stamp is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0, int((current - stamp.astimezone(timezone.utc)).total_seconds() // 86400))


def freshness(as_of: Any, max_age_days: int) -> tuple[str, int | None]:
    age = age_days(as_of)
    if age is None:
        return "UNKNOWN_DATE", None
    return ("FRESH" if age <= max_age_days else "STALE"), age


def load_raw_fundamentals(symbol: str) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    if not FUNDAMENTALS_DB.exists():
        return {"available": False, "data": {}, "fetched_at": "", "age_days": None, "freshness": "MISSING"}
    try:
        connection = sqlite3.connect(str(FUNDAMENTALS_DB), timeout=2.0)
        try:
            row = connection.execute("SELECT data_json, fetched_at FROM fundamentals_cache WHERE symbol = ?", (symbol,)).fetchone()
        finally:
            connection.close()
        if row is None:
            return {"available": False, "data": {}, "fetched_at": "", "age_days": None, "freshness": "MISSING"}
        data = json.loads(row[0])
        fetched = datetime.fromtimestamp(float(row[1]), tz=timezone.utc).isoformat()
        state, age = freshness(fetched, 2)
        return {"available": True, "data": data, "fetched_at": fetched, "age_days": age, "freshness": state}
    except Exception as exc:
        return {"available": False, "data": {}, "fetched_at": "", "age_days": None, "freshness": "ERROR", "error": str(exc)}


def _symbol_root(symbol: str) -> Path:
    return EVIDENCE_ROOT / clean_symbol(symbol)


def _manifest_path(symbol: str) -> Path:
    return _symbol_root(symbol) / "manifest.json"


def _load_manifest(symbol: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_manifest_path(symbol).read_text(encoding="utf-8"))
        return [dict(item) for item in payload.get("items", []) if isinstance(item, Mapping)]
    except Exception:
        return []


def _save_manifest(symbol: str, items: Iterable[Mapping[str, Any]]) -> None:
    path = _manifest_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "symbol": clean_symbol(symbol), "items": [dict(item) for item in items]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def list_uploads(symbol: str) -> list[dict[str, Any]]:
    items = _load_manifest(symbol)
    items.sort(key=lambda item: str(item.get("uploaded_at", "")), reverse=True)
    return items


def latest_upload(symbol: str, kind: str, *, usable_only: bool = False) -> dict[str, Any] | None:
    for item in list_uploads(symbol):
        if item.get("kind") == kind and (not usable_only or item.get("extracted")):
            return item
    return None


def _structured_records(content: bytes, extension: str) -> list[dict[str, Any]]:
    if extension == ".csv":
        text = content.decode("utf-8-sig")
        return [dict(row) for row in csv.DictReader(StringIO(text))]
    if extension == ".json":
        payload = json.loads(content.decode("utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _validate_structured(kind: str, content: bytes, extension: str, source_url: str) -> list[dict[str, Any]]:
    spec = RESOURCE_SPECS[kind]
    try:
        rows = _structured_records(content, extension)
    except Exception as exc:
        raise ValueError(f"structured upload could not be parsed: {exc}") from exc
    if not rows:
        raise ValueError("structured upload must contain at least one data row")
    columns = {str(key).strip() for row in rows for key in row.keys()}
    missing = [column for column in spec.required_columns if column not in columns]
    if missing:
        raise ValueError("structured upload is missing required columns: " + ", ".join(missing))
    for index, row in enumerate(rows, start=1):
        for column in spec.required_columns:
            if column == "source_url":
                continue
            if str(row.get(column, "")).strip() == "":
                raise ValueError(f"row {index} is missing required value: {column}")
        row_url = str(row.get("source_url") or source_url).strip()
        if not _valid_url(row_url):
            raise ValueError(f"row {index} has no valid http(s) source_url")
        row["source_url"] = row_url
    return rows


def save_upload(
    symbol: str,
    kind: str,
    content: bytes,
    *,
    filename: str,
    as_of: str,
    source_url: str = "",
) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    if kind not in RESOURCE_SPECS:
        raise ValueError("unknown evidence kind")
    if not content:
        raise ValueError("empty upload")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("upload exceeds 50 MB")
    source_url = str(source_url or "").strip()
    if not _valid_url(source_url):
        raise ValueError("a valid http(s) source_url is required")
    parsed_as_of = _parse_datetime(as_of)
    if parsed_as_of is None:
        raise ValueError("as_of must be a valid date such as 2026-06-30")
    safe_name = Path(filename or "evidence.bin").name
    extension = Path(safe_name).suffix.lower()
    spec = RESOURCE_SPECS[kind]
    if extension not in spec.accepted_extensions:
        raise ValueError(f"{extension or 'file type'} is not accepted for {kind}")
    structured = extension in {".csv", ".json"}
    rows = _validate_structured(kind, content, extension, source_url) if structured else []
    digest = hashlib.sha256(content).hexdigest()
    uploaded_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = _symbol_root(symbol) / kind / f"{stamp}_{digest[:10]}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    item = {
        "evidence_id": digest[:20], "symbol": symbol, "kind": kind,
        "filename": safe_name, "path": str(destination.relative_to(ROOT)),
        "sha256": digest, "bytes": len(content),
        "as_of": parsed_as_of.date().isoformat(), "uploaded_at": uploaded_at,
        "source_url": source_url, "structured": structured,
        "extracted": structured, "validated_rows": len(rows),
        "extraction_status": "STRUCTURED_VALIDATED" if structured else "SOURCE_ATTACHED_UNPARSED",
    }
    existing = [entry for entry in _load_manifest(symbol) if entry.get("sha256") != digest]
    _save_manifest(symbol, [item, *existing])
    return item


def upload_path(symbol: str, evidence_id: str) -> Path | None:
    for item in _load_manifest(symbol):
        if str(item.get("evidence_id")) == str(evidence_id):
            path = ROOT / str(item.get("path", ""))
            return path if path.exists() else None
    return None


def template_csv(kind: str) -> bytes:
    spec = RESOURCE_SPECS.get(kind)
    if spec is None or not spec.template_columns:
        raise ValueError("no CSV template is available for this evidence kind")
    output = StringIO()
    csv.writer(output).writerow(spec.template_columns)
    return output.getvalue().encode("utf-8")


def structured_rows(symbol: str, kind: str) -> list[dict[str, Any]]:
    item = latest_upload(symbol, kind, usable_only=True)
    if not item:
        return []
    path = ROOT / str(item.get("path", ""))
    if not path.exists():
        return []
    try:
        rows = _structured_records(path.read_bytes(), path.suffix.lower())
        for row in rows:
            row.setdefault("source_url", item.get("source_url", ""))
            row.setdefault("as_of_date", item.get("as_of", ""))
        return rows
    except Exception:
        return []


def _raw_section_state(raw: Mapping[str, Any], section: str, minimum_rows: int = 1) -> bool:
    value = raw.get(section)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) >= minimum_rows
    if isinstance(value, Mapping):
        return bool(value)
    return False


def evidence_requirements(
    symbol: str,
    *,
    price_as_of: str = "", scan_as_of: str = "", long_term_as_of: str = "",
    news_as_of: str = "", fno_as_of: str = "",
) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    raw_record = load_raw_fundamentals(symbol)
    raw = dict(raw_record.get("data", {}) or {})
    links = resource_links(symbol)
    uploads = list_uploads(symbol)
    auto_presence = {
        "business_profile": _raw_section_state(raw, "about"),
        "financial_history": _raw_section_state(raw, "profit_loss", 3) or _raw_section_state(raw, "quarterly_results", 4),
        "shareholding_history": _raw_section_state(raw, "shareholding", 2),
        "business_segments": False, "management_commentary": False,
        "order_book_guidance": False, "annual_report": False,
    }
    auto_as_of = {key: (raw_record.get("fetched_at", "") if auto_presence.get(key) else "") for key in RESOURCE_SPECS}
    items: list[dict[str, Any]] = []
    for key, spec in RESOURCE_SPECS.items():
        attached = latest_upload(symbol, key)
        usable = latest_upload(symbol, key, usable_only=True)
        if key == "annual_report" and attached:
            available = True
            source = "USER_SOURCE_DOCUMENT"
            as_of = str(attached.get("as_of") or "")
            state, age = freshness(as_of, spec.max_age_days)
        elif auto_presence.get(key):
            available = True
            source = "SCREENER_DEEP_CACHE"
            as_of = str(auto_as_of.get(key) or "")
            state, age = freshness(as_of, spec.max_age_days)
        elif usable:
            available = True
            source = "USER_STRUCTURED_UPLOAD"
            as_of = str(usable.get("as_of") or "")
            state, age = freshness(as_of, spec.max_age_days)
        elif attached:
            available = False
            source = "USER_SOURCE_DOCUMENT"
            as_of = str(attached.get("as_of") or "")
            state, age = "SOURCE_ATTACHED_UNPARSED", age_days(as_of)
        else:
            available, source, as_of, state, age = False, "", "", "MISSING", None
        items.append({
            "key": key, "label": spec.label, "status": state,
            "available": available, "source_attached": bool(attached),
            "source": source, "as_of": as_of, "age_days": age,
            "max_age_days": spec.max_age_days, "why": spec.why,
            "instructions": spec.instructions,
            "accepted_extensions": list(spec.accepted_extensions),
            "template_available": bool(spec.template_columns),
            "template_url": f"/evidence/templates/{key}.csv" if spec.template_columns else "",
            "links": links.get(key, []), "latest_upload": attached or {},
        })
    runtime_specs = [
        ("price_history", "Official price history", price_as_of, 3),
        ("scanner", "Whole-market scan", scan_as_of, 2),
        ("long_term", "Long-Term shortlist", long_term_as_of, 7),
        ("news", "Curated news and filings", news_as_of, 2),
        ("fno", "Current F&O instrument master", fno_as_of, 2),
        ("fundamentals_cache", "Deep fundamentals cache", raw_record.get("fetched_at", ""), 2),
    ]
    runtime = []
    for key, label, as_of, max_days in runtime_specs:
        state, age = freshness(as_of, max_days) if as_of else ("MISSING", None)
        runtime.append({
            "key": key, "label": label, "status": state,
            "available": bool(as_of), "as_of": as_of,
            "age_days": age, "max_age_days": max_days,
        })
    fresh = sum(1 for item in items if item["available"] and item["status"] == "FRESH")
    return {
        "schema_version": 2, "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_pct": round(fresh / len(items) * 100) if items else 0,
        "requirements": items, "runtime_sources": runtime, "uploads": uploads,
        "raw_fundamentals": {
            "available": bool(raw_record.get("available")),
            "fetched_at": raw_record.get("fetched_at", ""),
            "age_days": raw_record.get("age_days"),
            "freshness": raw_record.get("freshness", "MISSING"),
            "sections": {
                "about": _raw_section_state(raw, "about"),
                "quarterly_results": len(raw.get("quarterly_results", []) or []),
                "profit_loss": len(raw.get("profit_loss", []) or []),
                "balance_sheet": len(raw.get("balance_sheet", []) or []),
                "cash_flow": len(raw.get("cash_flow", []) or []),
                "shareholding": len(raw.get("shareholding", []) or []),
                "peer_comparison": len(raw.get("peer_comparison", []) or []),
            },
        },
    }
