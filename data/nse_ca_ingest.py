"""Ingest corporate actions from the official NSE corporates API.

Only events with an unambiguous share-count factor are written into the
adjustment ledger. Cash dividends may be stored as non-adjusting records for
provenance; ``data.corporate_actions.adjust_frame`` continues to ignore them
unless a research target opts into a versioned dividend policy.

Never invents factors from price gaps.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from data.corporate_actions import events_path, merge_events, write_events

_API = "https://www.nseindia.com/api/corporates-corporateActions"
_HOME = "https://www.nseindia.com/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json",
}

# Versioned adjustment policy referenced by research snapshots.
ADJUSTMENT_POLICY_VERSION = "ca_sharecount_v1"
ADJUSTMENT_POLICY = {
    "version": ADJUSTMENT_POLICY_VERSION,
    "share_count_types": ["split", "bonus", "consolidation"],
    "dividend_adjustment": "NONE_BY_DEFAULT",
    "dividend_note": (
        "Dividends are stored for provenance when sourced, but prices are NOT "
        "cash-adjusted unless a research target declares an explicit policy."
    ),
    "raw_prices_immutable": True,
}


_BONUS_RE = re.compile(r"\bbonus\s+(\d+)\s*:\s*(\d+)\b", re.I)
_SPLIT_RE = re.compile(
    r"face\s*value\s*split.*?from\s*r[se]\.?\s*([\d.]+)\s*to\s*r[se]\.?\s*([\d.]+)",
    re.I,
)
_SPLIT_RE2 = re.compile(
    r"sub[- ]?division.*?from\s*r[se]\.?\s*([\d.]+)\s*to\s*r[se]\.?\s*([\d.]+)",
    re.I,
)
_CONSOL_RE = re.compile(
    r"consolidat.*?from\s*r[se]\.?\s*([\d.]+)\s*to\s*r[se]\.?\s*([\d.]+)",
    re.I,
)


def _parse_ex_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_share_factor(subject: str) -> tuple[str, float] | None:
    """Return (type, factor) when the subject string is unambiguous; else None."""
    s = str(subject or "")
    m = _BONUS_RE.search(s)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        if d <= 0:
            return None
        # Bonus N:D → N new shares per D held → share multiple (N+D)/D
        factor = (n + d) / d
        if factor <= 1.0:
            return None
        return "bonus", float(factor)
    m = _SPLIT_RE.search(s) or _SPLIT_RE2.search(s)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if new_fv <= 0 or old_fv <= 0:
            return None
        factor = old_fv / new_fv
        if abs(factor - 1.0) < 1e-9:
            return None
        if factor > 1.0:
            return "split", float(factor)
        # face value increased = consolidation
        return "consolidation", float(1.0 / factor)
    m = _CONSOL_RE.search(s)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv <= 0 or new_fv <= 0 or new_fv <= old_fv:
            return None
        return "consolidation", float(new_fv / old_fv)
    return None


def _is_dividend_subject(subject: str) -> bool:
    s = subject.lower()
    return "dividend" in s and "bonus" not in s


def fetch_ca_range(from_date: str, to_date: str, *, session: requests.Session | None = None) -> tuple[list[dict], dict]:
    """from_date/to_date as DD-MM-YYYY (NSE API convention)."""
    sess = session or requests.Session()
    sess.headers.update(_HEADERS)
    # Warm cookies — NSE API often 403 without a homepage hit.
    sess.get(_HOME, timeout=30)
    params = {"index": "equities", "from_date": from_date, "to_date": to_date}
    resp = sess.get(_API, params=params, timeout=90)
    resp.raise_for_status()
    raw = resp.content
    data = resp.json()
    if not isinstance(data, list):
        data = []
    meta = {
        "source_url": _API,
        "params": params,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "n_raw": len(data),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return data, meta


def rows_from_nse_payload(payload: list[dict], *, source_sha256: str = "") -> dict[str, list]:
    """Split into adjusting events vs provenance-only (dividends / unparsed)."""
    adjusting: list[dict] = []
    provenance: list[dict] = []
    skipped = 0
    for row in payload:
        if not isinstance(row, dict):
            skipped += 1
            continue
        series = str(row.get("series") or "").strip().upper()
        if series and series != "EQ":
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        subject = str(row.get("subject") or "")
        ex = _parse_ex_date(str(row.get("exDate") or ""))
        isin = str(row.get("isin") or "").strip().upper() or None
        if not sym or not ex:
            skipped += 1
            continue
        base = {
            "symbol": sym,
            "ex_date": ex,
            "isin": isin,
            "subject": subject,
            "source": "nse_corporates_api",
            "source_sha256": source_sha256,
            "ca_broadcast_date": row.get("caBroadcastDate"),
            "face_val": row.get("faceVal"),
        }
        parsed = parse_share_factor(subject)
        if parsed:
            typ, factor = parsed
            adjusting.append({**base, "type": typ, "factor": factor})
        elif _is_dividend_subject(subject):
            provenance.append({**base, "type": "dividend", "factor": None})
        else:
            # AGM, buyback, meetings, etc. — keep lightly for audit, not in adjust ledger
            skipped += 1
    return {"adjusting": adjusting, "dividends": provenance, "skipped": skipped}


def materialize_ca_ledger(
    year_ranges: list[tuple[str, str]],
    *,
    path: str | Path | None = None,
    include_dividend_provenance_file: bool = True,
) -> dict[str, Any]:
    """Fetch NSE CA for each (from,to) DD-MM-YYYY range and write adjust ledger."""
    sess = requests.Session()
    all_adjusting: list[dict] = []
    all_divs: list[dict] = []
    metas: list[dict] = []
    for fr, to in year_ranges:
        payload, meta = fetch_ca_range(fr, to, session=sess)
        metas.append(meta)
        packed = rows_from_nse_payload(payload, source_sha256=meta["source_sha256"])
        all_adjusting.extend(packed["adjusting"])
        all_divs.extend(packed["dividends"])

    out_path = Path(path) if path else events_path()
    # write_events expects share-count rows only
    write_events(
        [{"symbol": e["symbol"], "ex_date": e["ex_date"], "factor": e["factor"], "type": e["type"]}
         for e in all_adjusting],
        path=out_path,
        source="nse_corporates_api",
    )
    # Enrich file with provenance block (non-breaking for load_events)
    try:
        raw = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw["adjustment_policy"] = ADJUSTMENT_POLICY
            raw["source_meta"] = metas
            raw["events_full"] = all_adjusting
            out_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

    div_path = None
    if include_dividend_provenance_file:
        div_path = out_path.with_name("ca_dividends_provenance.json")
        div_path.write_text(
            json.dumps({
                "schema_version": 1,
                "adjustment_policy": ADJUSTMENT_POLICY,
                "note": "Provenance only — not applied by adjust_frame",
                "events": all_divs,
                "source_meta": metas,
            }, indent=2, default=str),
            encoding="utf-8",
        )

    return {
        "path": str(out_path),
        "n_adjusting": len(all_adjusting),
        "n_dividends_provenance": len(all_divs),
        "dividend_provenance_path": str(div_path) if div_path else None,
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "source_meta": metas,
    }
