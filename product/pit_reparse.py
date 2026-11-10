"""Re-parse stored official XBRL filings with the current parser.

``pit_xbrl.v1`` stored "Lakhs"-rounded filings 100x too large and dropped
balance-sheet borrowings (so ROCE and debt/equity were never measurable).
The raw exchange documents are already on disk under ``research_evidence``;
this re-reads them -- no network -- and writes each corrected parse as a NEW
revision whose ``supersedes`` names the row it replaces. The original row is
kept for audit (the warehouse never mutates facts in place) and
``product.pit_financials.parsed_results`` stops reading it.

Availability is unchanged: the filing was public on its original publication
date, only QuantTerm's reading of it is corrected, so no look-ahead is added.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from logger import get_logger

log = get_logger(__name__)

_IDENTITY = re.compile(r"^nse_xbrl:integrated:(\d+)$")


def _artifact_index(evidence_root: Path) -> dict[tuple[str, str], Path]:
    """(SYMBOL, filing id) -> stored XML path."""
    index: dict[tuple[str, str], Path] = {}
    if not evidence_root.is_dir():
        return index
    for xml in evidence_root.glob("*/**/nse_xbrl_integrated_*.xml"):
        try:
            symbol = xml.relative_to(evidence_root).parts[0].upper()
        except ValueError:
            continue
        filing_id = xml.stem.rsplit("_", 1)[-1]
        index.setdefault((symbol, filing_id), xml)
    return index


def _candidates(rows: Iterable[Mapping[str, Any]], current: str) -> list[Mapping[str, Any]]:
    rows = list(rows)
    superseded = {str(r.get("supersedes")) for r in rows if r.get("supersedes")}
    out = []
    for row in rows:
        if str(row.get("evidence_id") or "") in superseded:
            continue  # already replaced
        if str(row.get("parser_version") or "") == current:
            continue
        if not _IDENTITY.match(str(row.get("source_identity") or "")):
            continue
        out.append(row)
    return out


def reparse_legacy_xbrl(
    *,
    warehouse_path: Path | None = None,
    evidence_root: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Re-parse every stored XBRL filing whose live row predates the current parser.

    Idempotent: a row that already has a superseding revision is skipped.
    """
    from product import pit_warehouse as W
    from product.pit_xbrl import PARSER_VERSION, parse_xbrl

    if evidence_root is None:
        from core.runtime_paths import logs_dir
        evidence_root = Path(logs_dir()) / "research_evidence"
    evidence_root = Path(evidence_root)
    index = _artifact_index(evidence_root)

    con = W._connect(warehouse_path)
    try:
        con.row_factory = sqlite3.Row
        rows = [W._row(r) for r in con.execute(
            "SELECT * FROM evidence WHERE evidence_type=?", (W.DOC_QUARTERLY_RESULT,))]
    finally:
        con.close()

    report: dict[str, Any] = {
        "parser_version": PARSER_VERSION,
        "dry_run": dry_run,
        "artifacts_indexed": len(index),
        "candidates": 0,
        "reparsed": 0,
        "missing_artifact": 0,
        "parse_failed": 0,
        "borrowings_recovered": 0,
    }
    todo = _candidates(rows, PARSER_VERSION)
    report["candidates"] = len(todo)
    for row in todo[: limit if limit else None]:
        symbol = str(row.get("symbol") or "").upper()
        filing_id = _IDENTITY.match(str(row.get("source_identity"))).group(1)
        xml_path = index.get((symbol, filing_id))
        if xml_path is None:
            report["missing_artifact"] += 1
            continue
        parsed = parse_xbrl(xml_path.read_text(encoding="utf-8", errors="ignore"))
        if not parsed.get("numbers_parsed"):
            report["parse_failed"] += 1
            continue
        old = dict(row.get("extracted") or {})
        if "bs_total_debt" in (parsed.get("facts") or {}) and "bs_total_debt" not in (old.get("facts") or {}):
            report["borrowings_recovered"] += 1
        report["reparsed"] += 1
        if dry_run:
            continue
        W.persist({
            **{k: row.get(k) for k in (
                "symbol", "evidence_type", "document_type", "period_start", "period_end",
                "publication_date", "filing_date", "exchange_timestamp", "available_from",
                "source", "source_url", "source_identity", "source_trust", "raw_artifact_id",
                "pit_status", "reason_code",
            )},
            "parser_version": PARSER_VERSION,
            "revision": int(row.get("revision") or 1) + 1,
            "supersedes": row.get("evidence_id"),
            "extracted": {
                **parsed,
                "numbers_parsed": True,
                # Filing-level metadata the parser does not see.
                "consolidated": old.get("consolidated"),
                "audited": old.get("audited") or parsed.get("audited"),
                "reparsed_from": row.get("evidence_id"),
            },
        }, path=warehouse_path)
    log.info("pit_reparse_complete", **{k: v for k, v in report.items() if k != "parser_version"})
    return report
