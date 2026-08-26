"""Coordinate Investigate with stores QuantTerm already has.

Does not scrape, score long-term quality twice, or invent commentary.
Empty uploads stay empty and point at Research Data.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from product.due_diligence.series import _f

_SNAPSHOT_FIELDS = (
    ("roe", "Return on equity", "%", "profitability"),
    ("roce", "Return on capital employed", "%", "profitability"),
    ("sales_growth_3y", "Sales CAGR (3Y)", "%", "growth"),
    ("profit_growth_3y", "Profit CAGR (3Y)", "%", "growth"),
    ("cfo_to_pat", "Cash flow / PAT", "x", "cash"),
    ("promoter_pledge", "Promoter pledge", "%", "governance"),
    ("pe", "Price / earnings", "x", "valuation"),
)


def _rows(symbol: str, kind: str, loader: Callable[[str, str], list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    if loader is not None:
        try:
            return list(loader(symbol, kind) or [])
        except Exception:
            return []
    try:
        from reporting.evidence_intake import structured_rows
        return list(structured_rows(symbol, kind) or [])
    except Exception:
        return []


def _extract(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from screener.engine import _extract_fundamentals
        return dict(_extract_fundamentals(dict(raw) or {}) or {})
    except Exception:
        return {}


def _snapshot_metrics(extracted: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, label, unit, pillar in _SNAPSHOT_FIELDS:
        value = _f(extracted.get(key))
        available = value is not None
        fact = f"{label}: {value} {unit}" if available else "Data unavailable"
        interpretation = "Data unavailable"
        if available and key == "promoter_pledge":
            interpretation = (
                "No reported pledge in the current snapshot."
                if value == 0 else
                "Elevated pledge in the current snapshot." if value > 10 else
                "Limited reported pledge in the current snapshot."
            )
        elif available:
            interpretation = f"{label} is a current snapshot print, not a quarterly trend."
        out.append({
            "id": key,
            "label": label,
            "pillar": pillar,
            "unit": unit,
            "available": available,
            "value": value,
            "fact": fact,
            "interpretation": interpretation,
            "implication": (
                "Not used in the Investigate score — shown because Stock Intelligence already extracts it."
                if available else
                "No implication without a measured value."
            ),
            "source": "Current snapshot extractor / key-ratios cache — not a quarterly series",
            "used_in_score": False,
        })
    return out


def _peer_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in list(raw.get("peer_comparison") or []):
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("Name") or row.get("name") or row.get("row_label") or "").strip()
        if not name or name.lower() in {"raw pdf", "s.no."}:
            continue
        cells = {
            str(key): value
            for key, value in row.items()
            if str(key) not in {"Name", "name", "row_label", ""} and value not in (None, "")
        }
        if not cells:
            continue
        rows.append({"name": name, "cells": cells, "fact": f"{name}: " + ", ".join(f"{k} {v}" for k, v in list(cells.items())[:6])})
        if len(rows) >= 8:
            break
    return rows


def _commentary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("commentary") or row.get("management_wording") or "").strip()
        if not text:
            continue
        out.append({
            "speaker": str(row.get("speaker") or "Management"),
            "topic": str(row.get("topic") or ""),
            "commentary": text,
            "event_date": str(row.get("event_date") or row.get("as_of_date") or ""),
            "source_url": str(row.get("source_url") or ""),
            "guidance_metric": str(row.get("guidance_metric") or ""),
            "guidance_value": str(row.get("guidance_value") or ""),
        })
        if len(out) >= 6:
            break
    return out


def _segments_line(rows: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        name = str(row.get("segment") or "").strip()
        if not name:
            continue
        revenue = row.get("revenue_cr")
        mix = row.get("revenue_mix_pct")
        bit = name
        if revenue not in (None, ""):
            bit += f" {revenue} cr"
        if mix not in (None, ""):
            bit += f" ({mix}%)"
        parts.append(bit)
        if len(parts) >= 6:
            break
    return "; ".join(parts) if parts else "Data unavailable — no segment table on file."


def _order_book(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metric = str(row.get("metric") or "").strip()
        value = row.get("value")
        if not metric or value in (None, ""):
            continue
        out.append({
            "metric": metric,
            "value": value,
            "unit": str(row.get("unit") or ""),
            "period": str(row.get("period") or ""),
            "wording": str(row.get("management_wording") or ""),
            "as_of": str(row.get("as_of_date") or ""),
            "source_url": str(row.get("source_url") or ""),
            "fact": f"{metric}: {value} {row.get('unit') or ''} ({row.get('period') or row.get('as_of_date') or 'date unavailable'})".strip(),
        })
        if len(out) >= 8:
            break
    return out


def _gaps(requirements: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(requirements.get("requirements") or []):
        if item.get("available"):
            continue
        links = list(item.get("links") or [])
        official = next((link for link in links if str(link.get("official")) == "true"), links[0] if links else {})
        out.append({
            "key": item.get("key"),
            "label": item.get("label"),
            "status": item.get("status") or "MISSING",
            "why": item.get("why") or "",
            "instructions": item.get("instructions") or "",
            "source_attached": bool(item.get("source_attached")),
            "link_label": official.get("label") or "",
            "link_url": official.get("url") or "",
        })
    return out


def load_evidence_pack(
    symbol: str,
    *,
    raw: Mapping[str, Any] | None = None,
    scan_as_of: str = "",
    long_term_as_of: str = "",
    news_as_of: str = "",
    long_row: Mapping[str, Any] | None = None,
    row_loader: Callable[[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Assemble existing intake + extractor + long-term overlay. Never scrapes."""
    raw = dict(raw or {})
    long_row = dict(long_row or {})
    try:
        from reporting.evidence_intake import evidence_requirements
        requirements = evidence_requirements(
            symbol,
            scan_as_of=scan_as_of,
            long_term_as_of=long_term_as_of,
            news_as_of=news_as_of,
        )
    except Exception:
        requirements = {"requirements": [], "coverage_pct": 0}

    commentary = _commentary(_rows(symbol, "management_commentary", row_loader))
    segments = _rows(symbol, "business_segments", row_loader)
    order_book = _order_book(_rows(symbol, "order_book_guidance", row_loader))
    profile_rows = _rows(symbol, "business_profile", row_loader)
    extracted = _extract(raw)
    snapshot = _snapshot_metrics(extracted)
    peers = _peer_rows(raw)
    gaps = _gaps(requirements)

    flags: list[dict[str, Any]] = []
    pledge = next((item for item in snapshot if item["id"] == "promoter_pledge"), None)
    if pledge and pledge.get("available") and (pledge.get("value") or 0) > 20:
        flags.append({
            "id": "flag-snapshot-pledge",
            "title": f"Promoter pledge is {pledge['value']}% in the current snapshot",
            "kind": "governance",
            "fact": pledge["fact"],
            "source": pledge["source"],
            "source_date": "current snapshot",
        })

    next_actions = [
        {
            "id": "acquire",
            "label": "Acquire missing data from the internet",
            "control": "ACQUIRE_DUE_DILIGENCE",
            "detail": "Downloads Screener tables and official NSE filings for this symbol, extracts facts with rules, then Investigate re-reads the files. GET still does not scrape.",
        },
        {
            "id": "research-data",
            "label": "Complete missing research data",
            "page": "Research Data",
            "detail": "Upload commentary, segments, order-book or annual report. Investigate will read them on the next open.",
        },
        {
            "id": "refresh-fundamentals",
            "label": "Refresh this stock's fundamentals cache",
            "control": "REFRESH_STOCK_FUNDAMENTALS",
            "detail": "Uses the existing Stock Intelligence refresh — Investigate does not scrape on GET.",
        },
    ]

    about = ""
    if profile_rows:
        about = str(profile_rows[0].get("business_summary") or "").strip()

    return {
        "coverage_pct": int(requirements.get("coverage_pct") or 0),
        "gaps": gaps,
        "management_commentary": commentary,
        "order_book": order_book,
        "peers": peers,
        "snapshot_metrics": snapshot,
        "revenue_drivers": _segments_line(segments),
        "business_model": about or "Data unavailable",
        "long_term_overlay": {
            "classification": str(long_row.get("classification") or "") or None,
            "quality_factors": list(long_row.get("quality_factors") or [])[:5],
            "risk_flags": list(long_row.get("risk_flags") or [])[:5],
            "note": "From the long-term scan overlay. Not the Investigate score.",
        },
        "flags": flags,
        "next_actions": next_actions,
        "empty_note": (
            "Management commentary, segments, order-book, peers and option-chain stay Data unavailable until a structured upload, an Acquire download, or a cache row exists."
        ),
    }


def apply_autonomy_pack(pack: Mapping[str, Any] | None, autonomy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill Investigate holes from Acquire files. Uploads and cache rows win."""
    out = dict(pack or {})
    facts = dict(autonomy or {})
    if not out.get("management_commentary"):
        out["management_commentary"] = _commentary(list(facts.get("commentary") or []))
    if not out.get("order_book"):
        out["order_book"] = _order_book(list(facts.get("order_book") or []))
    drivers = str(out.get("revenue_drivers") or "")
    if (not drivers or drivers.startswith("Data unavailable")) and facts.get("segments"):
        line = _segments_line(list(facts.get("segments") or []))
        if line and not line.startswith("Data unavailable"):
            out["revenue_drivers"] = line
    chain = facts.get("option_chain")
    if isinstance(chain, Mapping) and chain:
        out["option_chain"] = dict(chain)
    elif "option_chain" not in out:
        out["option_chain"] = {}
    return out
