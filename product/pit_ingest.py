"""Harvest already-downloaded NSE archives into the PIT warehouse.

Does not invent publication dates. filingDate='-' or missing sort_date
becomes PIT_UNVERIFIED and is not eligible for historical decisions.
Does not read today's Screener cache into historical rows.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from product.pit_availability import PIT_UNVERIFIED
from product.pit_warehouse import (
    DOC_ANNUAL_REPORT,
    DOC_QUARTERLY_RESULT,
    classify_document,
    persist,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "logs" / "research_evidence"
PARSER_VERSION = "pit_ingest.v1"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_nse_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"-", "None", "null"}:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    # 12-Oct-2012 09:25  or  01-Sep-2026 15:34:10
    parts = text.replace(",", " ").split()
    if not parts:
        return ""
    token = parts[0]
    bits = token.split("-")
    if len(bits) == 3 and bits[1].isalpha():
        day, mon, year = bits
        month = _MONTHS.get(mon[:3].lower())
        if month and year.isdigit() and day.isdigit():
            return f"{int(year):04d}-{month:02d}-{int(day):02d}"
    return ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ingest_announcement_row(symbol: str, row: Mapping[str, Any], *, raw_path: str = "", warehouse_path=None) -> dict[str, Any]:
    pub = _parse_nse_date(row.get("sort_date") or row.get("an_dt") or row.get("exchdisstime"))
    desc = " ".join(
        str(row.get(k) or "")
        for k in ("attchmntText", "desc", "subject", "details")
    )
    url = str(row.get("attchmntFile") or row.get("attachment") or "")
    seq = str(row.get("seq_id") or row.get("seq_id") or row.get("dt") or url or desc[:40])
    doc = classify_document(desc)
    return persist({
        "symbol": symbol,
        "evidence_type": doc,
        "document_type": doc,
        "publication_date": pub,
        "filing_date": pub,
        "exchange_timestamp": str(row.get("exchdisstime") or row.get("sort_date") or ""),
        "available_from": pub,
        "period_end": "",
        "source": "NSE corporate announcements",
        "source_url": url,
        "source_identity": f"nse_ann:{seq}",
        "raw_artifact_id": raw_path or _sha(seq),
        "parser_version": PARSER_VERSION,
        "extracted": {
            "headline": desc[:240],
            "industry": row.get("smIndustry"),
            "has_xbrl": row.get("hasXbrl"),
        },
        "pit_status": "INDEXED" if pub else PIT_UNVERIFIED,
        "reason_code": "" if pub else "PUBLICATION_DATE_UNKNOWN",
    }, path=warehouse_path)


def ingest_result_row(symbol: str, row: Mapping[str, Any], *, raw_path: str = "", warehouse_path=None) -> dict[str, Any]:
    pub = _parse_nse_date(row.get("filingDate") or row.get("broadCastDate") or row.get("exchdisstime"))
    period_start = _parse_nse_date(row.get("fromDate"))
    period_end = _parse_nse_date(row.get("toDate"))
    url = str(row.get("resultDetailedDataLink") or row.get("xbrl") or "")
    seq = str(row.get("seqNumber") or row.get("params") or url)
    return persist({
        "symbol": symbol,
        "evidence_type": DOC_QUARTERLY_RESULT,
        "document_type": DOC_QUARTERLY_RESULT,
        "period_start": period_start,
        "period_end": period_end,
        "publication_date": pub,
        "filing_date": pub,
        "exchange_timestamp": str(row.get("broadCastDate") or ""),
        "available_from": pub,
        "source": "NSE financial results",
        "source_url": url,
        "source_identity": f"nse_result:{seq}",
        "raw_artifact_id": raw_path or _sha(seq),
        "parser_version": PARSER_VERSION,
        "extracted": {
            "period": row.get("period"),
            "relating_to": row.get("relatingTo"),
            "audited": row.get("audited"),
            "consolidated": row.get("consolidated"),
            "financial_year": row.get("financialYear"),
            "numbers_parsed": False,
        },
        "pit_status": "INDEXED" if pub else PIT_UNVERIFIED,
        "reason_code": "" if pub else "PUBLICATION_DATE_UNKNOWN",
    }, path=warehouse_path)


def ingest_annual_report_index(symbol: str, rows: list, *, raw_path: str = "", warehouse_path=None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("fileName") or row.get("name") or row.get("url") or "")
        # Filenames are not publication dates. Do not guess.
        out.append(persist({
            "symbol": symbol,
            "evidence_type": DOC_ANNUAL_REPORT,
            "document_type": DOC_ANNUAL_REPORT,
            "publication_date": "",
            "available_from": "",
            "source": "NSE annual reports",
            "source_url": name,
            "source_identity": f"nse_ar:{name}",
            "raw_artifact_id": raw_path,
            "parser_version": PARSER_VERSION,
            "extracted": {"file": name},
            "pit_status": PIT_UNVERIFIED,
            "reason_code": "PUBLICATION_DATE_UNKNOWN",
        }, path=warehouse_path))
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def harvest_symbol(symbol: str, *, folder: Path | None = None, warehouse_path=None) -> dict[str, Any]:
    """Index one already-acquired autonomy folder. No network."""
    target = folder or (EVIDENCE_ROOT / symbol.upper() / "autonomy")
    report = {
        "symbol": symbol.upper(),
        "attempted": 0,
        "acquired": 0,
        "parsed": 0,
        "failed": 0,
        "unavailable": 0,
        "unverified": 0,
        "deduped": 0,
        "reasons": {},
    }
    if not target.exists():
        report["unavailable"] = 1
        report["reasons"]["NOT_FOUND"] = 1
        return report

    nse0 = target / "nse_0.json"
    payload = _read_json(nse0)
    rows = payload if isinstance(payload, list) else (payload or {}).get("data") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            report["attempted"] += 1
            item = ingest_announcement_row(symbol, row, raw_path=str(nse0), warehouse_path=warehouse_path)
            report["acquired"] += 1
            if item.get("deduped"):
                report["deduped"] += 1
            elif item.get("pit_status") == PIT_UNVERIFIED:
                report["unverified"] += 1
            else:
                report["parsed"] += 1

    nse1 = target / "nse_1.json"
    payload = _read_json(nse1)
    rows = payload if isinstance(payload, list) else (payload or {}).get("data") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            report["attempted"] += 1
            item = ingest_result_row(symbol, row, raw_path=str(nse1), warehouse_path=warehouse_path)
            report["acquired"] += 1
            if item.get("deduped"):
                report["deduped"] += 1
            elif item.get("pit_status") == PIT_UNVERIFIED:
                report["unverified"] += 1
            else:
                report["parsed"] += 1

    ar = target / "nse_annual_reports.json"
    payload = _read_json(ar)
    if payload:
        rows = payload if isinstance(payload, list) else (payload or {}).get("data") or []
        if isinstance(rows, list):
            for item in ingest_annual_report_index(symbol, rows, raw_path=str(ar), warehouse_path=warehouse_path):
                report["attempted"] += 1
                report["acquired"] += 1
                if item.get("pit_status") == PIT_UNVERIFIED:
                    report["unverified"] += 1
    return report


def harvest_existing(*, limit: int | None = None, warehouse_path=None) -> dict[str, Any]:
    if not EVIDENCE_ROOT.exists():
        return {"symbols": [], "n": 0, "note": "no research_evidence archive"}
    names = sorted(p.name for p in EVIDENCE_ROOT.iterdir() if p.is_dir())
    if limit:
        names = names[: int(limit)]
    rows = [harvest_symbol(name, warehouse_path=warehouse_path) for name in names]
    return {
        "n": len(rows),
        "symbols": names,
        "attempted": sum(r["attempted"] for r in rows),
        "parsed": sum(r["parsed"] for r in rows),
        "unverified": sum(r["unverified"] for r in rows),
        "deduped": sum(r["deduped"] for r in rows),
        "details": rows,
        "note": "Harvested existing NSE archives. Screener cache was not ingested.",
    }
