"""Research evidence intake, freshness and user guidance for QuantTerm.

This module answers four questions for every research section:
1. What data does QuantTerm currently have?
2. What is the strict as-of date and freshness state?
3. Where can missing evidence be obtained?
4. How can a user upload a source or a structured CSV template?

Uploaded PDFs are preserved as source documents but are never converted into claims
without a structured extraction step. Structured CSV uploads can be consumed directly
by the dossier assembler. No missing field is replaced with model memory.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping

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


RESOURCE_SPECS: dict[str, ResourceSpec] = {
    "business_profile": ResourceSpec(
        key="business_profile",
        label="Business description",
        why="Explains what the company sells, who pays, and which demand drivers matter.",
        instructions="Download the latest annual report or investor presentation. Upload the PDF, or upload the structured business profile CSV.",
        accepted_extensions=(".pdf", ".csv", ".json", ".txt"),
        max_age_days=450,
        template_columns=("as_of_date", "business_summary", "customers", "demand_drivers", "source_url"),
    ),
    "financial_history": ResourceSpec(
        key="financial_history",
        label="Financial history",
        why="Supports quarterly and annual revenue, margin, profit, cash-flow and balance-sheet analysis.",
        instructions="Use NSE Financial Results, the company annual report, or Screener. Prefer consolidated figures and include the period-end date in every row.",
        accepted_extensions=(".csv", ".xlsx", ".xls", ".pdf", ".json"),
        max_age_days=120,
        template_columns=(
            "period_end", "period_type", "revenue_cr", "ebitda_cr", "ebitda_margin_pct",
            "pat_cr", "cfo_cr", "debt_cr", "source_url", "as_of_date",
        ),
    ),
    "shareholding_history": ResourceSpec(
        key="shareholding_history",
        label="Shareholding history",
        why="Allows promoter, FII, DII and public ownership changes to be measured quarter by quarter.",
        instructions="Download the exchange shareholding-pattern CSV/XBRL for at least four quarters. Upload the exchange file or fill the QuantTerm CSV template.",
        accepted_extensions=(".csv", ".xlsx", ".xls", ".xml", ".pdf", ".json"),
        max_age_days=120,
        template_columns=(
            "quarter_end", "promoter_pct", "fii_pct", "dii_pct", "public_pct",
            "promoter_pledge_pct", "source_url", "as_of_date",
        ),
    ),
    "business_segments": ResourceSpec(
        key="business_segments",
        label="Business and segment mix",
        why="Shows where revenue and profit come from instead of treating the company as one undifferentiated number.",
        instructions="Use the segment note in the annual report, investor presentation or results filing. Enter each segment as a separate row.",
        accepted_extensions=(".csv", ".xlsx", ".xls", ".pdf", ".json"),
        max_age_days=450,
        template_columns=(
            "period_end", "segment", "revenue_cr", "revenue_mix_pct", "growth_pct",
            "margin_pct", "driver", "source_url", "as_of_date",
        ),
    ),
    "management_commentary": ResourceSpec(
        key="management_commentary",
        label="Management commentary and guidance",
        why="Separates management guidance, risks and operating explanations from editorial summaries.",
        instructions="Download the earnings-call transcript, exchange-filed investor presentation, or official call summary. Include speaker, date and exact source URL.",
        accepted_extensions=(".pdf", ".txt", ".csv", ".json", ".vtt", ".srt"),
        max_age_days=120,
        template_columns=(
            "event_date", "speaker", "topic", "commentary", "guidance_metric",
            "guidance_value", "guidance_period", "source_url", "as_of_date",
        ),
    ),
    "order_book_guidance": ResourceSpec(
        key="order_book_guidance",
        label="Order book and forward guidance",
        why="Supports forward visibility claims with dated company disclosures rather than narrative inference.",
        instructions="Use an exchange announcement, results presentation or earnings-call transcript. Upload the source or fill the guidance template.",
        accepted_extensions=(".csv", ".xlsx", ".xls", ".pdf", ".json"),
        max_age_days=180,
        template_columns=(
            "as_of_date", "metric", "value", "unit", "period", "management_wording",
            "source_url",
        ),
    ),
    "annual_report": ResourceSpec(
        key="annual_report",
        label="Annual report",
        why="Primary source for the business model, segment disclosures, risks, governance and audited financials.",
        instructions="Download the latest annual report from NSE or the company investor-relations page and upload the original PDF.",
        accepted_extensions=(".pdf",),
        max_age_days=450,
    ),
}


def clean_symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Z0-9&.-]", "", str(value or "").strip().upper())
    if not symbol or len(symbol) > 32:
        raise ValueError("invalid NSE symbol")
    return symbol


def resource_links(symbol: str) -> dict[str, list[dict[str, str]]]:
    symbol = clean_symbol(symbol)
    nse_announcements = "https://www.nseindia.com/companies-listing/corporate-filings-application?id=allAnnouncements"
    nse_financials = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
    nse_shareholding = f"https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern?symbol={symbol}"
    nse_annual = "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
    screener = f"https://www.screener.in/company/{symbol}/consolidated/"
    company_search = f"https://www.google.com/search?q={symbol}+investor+relations+annual+report+earnings+call+transcript"
    return {
        "business_profile": [
            {"label": "NSE annual reports", "url": nse_annual, "official": "true"},
            {"label": "Company investor-relations search", "url": company_search, "official": "false"},
            {"label": "Screener company page", "url": screener, "official": "false"},
        ],
        "financial_history": [
            {"label": "NSE financial results", "url": nse_financials, "official": "true"},
            {"label": "NSE annual reports", "url": nse_annual, "official": "true"},
            {"label": "Screener company page", "url": screener, "official": "false"},
        ],
        "shareholding_history": [
            {"label": "NSE shareholding patterns", "url": nse_shareholding, "official": "true"},
            {"label": "NSE corporate filings", "url": nse_announcements, "official": "true"},
        ],
        "business_segments": [
            {"label": "NSE annual reports", "url": nse_annual, "official": "true"},
            {"label": "Company investor-relations search", "url": company_search, "official": "false"},
        ],
        "management_commentary": [
            {"label": "NSE announcements and call filings", "url": nse_announcements, "official": "true"},
            {"label": "Company investor-relations search", "url": company_search, "official": "false"},
        ],
        "order_book_guidance": [
            {"label": "NSE corporate announcements", "url": nse_announcements, "official": "true"},
            {"label": "NSE financial results", "url": nse_financials, "official": "true"},
            {"label": "Company investor-relations search", "url": company_search, "official": "false"},
        ],
        "annual_report": [
            {"label": "NSE annual reports", "url": nse_annual, "official": "true"},
            {"label": "Company investor-relations search", "url": company_search, "official": "false"},
        ],
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
            row = connection.execute(
                "SELECT data_json, fetched_at FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
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
    payload = {"schema_version": 1, "symbol": clean_symbol(symbol), "items": [dict(item) for item in items]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def list_uploads(symbol: str) -> list[dict[str, Any]]:
    items = _load_manifest(symbol)
    items.sort(key=lambda item: str(item.get("uploaded_at", "")), reverse=True)
    return items


def latest_upload(symbol: str, kind: str) -> dict[str, Any] | None:
    for item in list_uploads(symbol):
        if item.get("kind") == kind:
            return item
    return None


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
    safe_name = Path(filename or "evidence.bin").name
    extension = Path(safe_name).suffix.lower()
    spec = RESOURCE_SPECS[kind]
    if extension not in spec.accepted_extensions:
        raise ValueError(f"{extension or 'file type'} is not accepted for {kind}")
    parsed_as_of = _parse_datetime(as_of)
    if parsed_as_of is None:
        raise ValueError("as_of must be a valid date such as 2026-06-30")
    digest = hashlib.sha256(content).hexdigest()
    uploaded_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = _symbol_root(symbol) / kind / f"{stamp}_{digest[:10]}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    item = {
        "evidence_id": digest[:20],
        "symbol": symbol,
        "kind": kind,
        "filename": safe_name,
        "path": str(destination.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(content),
        "as_of": parsed_as_of.date().isoformat(),
        "uploaded_at": uploaded_at,
        "source_url": str(source_url or "").strip(),
        "structured": extension in {".csv", ".json"},
        "extracted": extension in {".csv", ".json"},
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
    writer = csv.writer(output)
    writer.writerow(spec.template_columns)
    return output.getvalue().encode("utf-8")


def structured_rows(symbol: str, kind: str) -> list[dict[str, Any]]:
    item = latest_upload(symbol, kind)
    if not item or not item.get("structured"):
        return []
    path = ROOT / str(item.get("path", ""))
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("records", [])
            return [dict(row) for row in rows if isinstance(row, Mapping)]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
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
    price_as_of: str = "",
    scan_as_of: str = "",
    long_term_as_of: str = "",
    news_as_of: str = "",
    fno_as_of: str = "",
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
        "business_segments": False,
        "management_commentary": False,
        "order_book_guidance": False,
        "annual_report": False,
    }
    auto_as_of = {
        "business_profile": raw_record.get("fetched_at", ""),
        "financial_history": raw_record.get("fetched_at", ""),
        "shareholding_history": raw_record.get("fetched_at", ""),
        "business_segments": "",
        "management_commentary": "",
        "order_book_guidance": "",
        "annual_report": "",
    }

    items: list[dict[str, Any]] = []
    for key, spec in RESOURCE_SPECS.items():
        upload = latest_upload(symbol, key)
        present = bool(auto_presence.get(key) or upload)
        as_of = str((upload or {}).get("as_of") or auto_as_of.get(key) or "")
        state, age = freshness(as_of, spec.max_age_days) if present else ("MISSING", None)
        source = "USER_UPLOAD" if upload else ("SCREENER_DEEP_CACHE" if auto_presence.get(key) else "")
        items.append({
            "key": key,
            "label": spec.label,
            "status": state,
            "available": present,
            "source": source,
            "as_of": as_of,
            "age_days": age,
            "max_age_days": spec.max_age_days,
            "why": spec.why,
            "instructions": spec.instructions,
            "accepted_extensions": list(spec.accepted_extensions),
            "template_available": bool(spec.template_columns),
            "template_url": f"/evidence/templates/{key}.csv" if spec.template_columns else "",
            "links": links.get(key, []),
            "latest_upload": upload or {},
        })

    runtime_sources = [
        ("price_history", "Official price history", price_as_of, 3),
        ("scanner", "Whole-market scan", scan_as_of, 2),
        ("long_term", "Long-Term shortlist", long_term_as_of, 7),
        ("news", "Curated news and filings", news_as_of, 2),
        ("fno", "Current F&O instrument master", fno_as_of, 2),
        ("fundamentals_cache", "Deep fundamentals cache", raw_record.get("fetched_at", ""), 2),
    ]
    runtime = []
    for key, label, as_of, max_days in runtime_sources:
        state, age = freshness(as_of, max_days) if as_of else ("MISSING", None)
        runtime.append({
            "key": key,
            "label": label,
            "status": state,
            "available": bool(as_of),
            "as_of": as_of,
            "age_days": age,
            "max_age_days": max_days,
        })

    available_weight = sum(1 for item in items if item["available"] and item["status"] == "FRESH")
    total_weight = len(items)
    coverage_pct = round(available_weight / total_weight * 100) if total_weight else 0
    return {
        "schema_version": 1,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_pct": coverage_pct,
        "requirements": items,
        "runtime_sources": runtime,
        "uploads": uploads,
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
