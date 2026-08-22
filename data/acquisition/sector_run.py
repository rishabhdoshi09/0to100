"""Official current industry classification — STATIC_BACKFILL only.

NSE index constituent CSVs carry Industry for listed names. That raises
coverage. It is still today's classification projected backward — not PIT.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from data.acquisition.cache import write_manifest, write_raw
from data.acquisition.http import HEADERS, get_bytes
from data.sector_map import STATIC_BACKFILL, build_static_map, freeze_snapshot

# Official current constituent lists (Industry column). Not a history file.
_LISTS = (
    ("nifty50", "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"),
    ("nifty100", "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"),
    ("nifty200", "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"),
    ("nifty500", "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"),
    ("nifty_total_market", "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"),
    ("nifty_midcap150", "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv"),
    ("nifty_smallcap250", "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"),
)


def _parse_industry_csv(blob: bytes) -> list[dict[str, str]]:
    text = blob.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        r = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        sym = str(r.get("Symbol") or r.get("SYMBOL") or "").strip().upper()
        industry = str(r.get("Industry") or r.get("industry") or "").strip()
        if not sym or not industry:
            continue
        rows.append({
            "symbol": sym,
            "industry": industry,
            "company": str(r.get("Company Name") or r.get("Company") or ""),
            "isin": str(r.get("ISIN Code") or r.get("ISIN") or ""),
        })
    return rows


def ingest_official_industry(*, freeze: bool = True) -> dict[str, Any]:
    import requests
    sess = requests.Session()
    sess.headers.update(HEADERS)
    by_sym: dict[str, dict] = {}
    fetched = {}
    for name, url in _LISTS:
        blob, meta = get_bytes(url, session=sess, timeout=40, retries=2)
        if not blob:
            fetched[name] = {"ok": False, **meta}
            continue
        rec = write_raw(f"sector/{name}.csv", blob, meta={"url": url})
        parsed = _parse_industry_csv(blob)
        fetched[name] = {"ok": True, "n": len(parsed), "sha256": rec.get("sha256")}
        for row in parsed:
            # First list wins only if empty; later broader lists fill gaps.
            prev = by_sym.get(row["symbol"])
            if prev is None:
                by_sym[row["symbol"]] = {**row, "source_list": name}

    existing = build_static_map()
    merged = dict(existing.get("map") or {})
    official_industry = {s: r["industry"] for s, r in by_sym.items()}
    # Official industry is finer; keep SEPA macro sector when present.
    for sym, industry in official_industry.items():
        if sym not in merged:
            merged[sym] = industry

    rows = []
    for sym, sec in sorted(merged.items()):
        official = official_industry.get(sym)
        rows.append({
            "symbol": sym,
            "sector": sec,
            "industry": official or sec,
            "macro_sector": sec,
            "classification_source": (
                "nse_index_constituent_industry+sepa003_overlay"
                if official else (existing.get("source") or "sepa003")
            ),
            "valid_from": None,
            "valid_to": None,
            "source_timestamp": None,
            "mapping_version": "sector_map.v3_static_official_industry",
            "pit_status": STATIC_BACKFILL,
        })
    import hashlib
    import json
    blob = json.dumps({"version": "sector_map.v3_static_official_industry", "rows": rows}, sort_keys=True).encode()
    payload = {
        "version": "sector_map.v3_static_official_industry",
        "pit_class": STATIC_BACKFILL,
        "sector_identity_pit": False,
        "n_mapped": len(rows),
        "n_official_industry": len(official_industry),
        "n_unknown_policy": "unmapped stays UNKNOWN",
        "source": "nse_index_constituent_lists+sepa003_overlay",
        "never_projects_silently": True,
        "content_hash": hashlib.sha256(blob).hexdigest(),
        "rows": rows,
        "map": merged,
    }
    dest = None
    if freeze:
        dest = freeze_snapshot(as_of=datetime.now(timezone.utc).date().isoformat(), mapping=payload)
    man = {
        "source": "nse_index_constituent_industry",
        "fetched": fetched,
        "n_official_industry": len(official_industry),
        "n_mapped_total": len(rows),
        "pit_class": STATIC_BACKFILL,
        "coverage_rose_pit_did_not": True,
        "frozen_path": str(dest) if dest else None,
    }
    write_manifest("sector_official_industry", man)
    return {**man, "mapping": payload}
