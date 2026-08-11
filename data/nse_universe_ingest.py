"""Materialize PIT universe membership from official NSE sources.

Sources (evidence only):
  • EQUITY_L — current EQ master listing dates + ISIN
  • delisted.csv — official delisting dates

Delisted rows without a known listing date are included ONLY when local official
bhav coverage supplies a first-session date (documented as window-lower-bound,
not an IPO claim). Symbols that cannot be dated are omitted — never invented.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from data.security_identity import fetch_delisted, fetch_equity_l
from data.universe_history import history_path, write_universe_history, ledger_status


def materialize_universe_from_nse(
    *,
    path: str | Path | None = None,
    session: requests.Session | None = None,
    use_bhav_first_seen_for_delisted: bool = True,
) -> dict[str, Any]:
    sess = session or requests.Session()
    equity_rows, eq_meta = fetch_equity_l(session=sess)
    delisted_rows, de_meta = fetch_delisted(session=sess)

    by_sym: dict[str, dict] = {}
    for r in equity_rows:
        if r.get("symbol") and r.get("listing_date"):
            by_sym[r["symbol"]] = {
                "symbol": r["symbol"],
                "listed": r["listing_date"],
                "listing_provenance": "nse_equity_l",
            }

    spans: dict[str, dict] = {}
    if use_bhav_first_seen_for_delisted:
        try:
            from data.bhavcopy_runtime import ensure_loaded
            from data import bhavcopy_store as BS
            ensure_loaded(rebuild_from_local=False)
            spans = BS.symbol_date_spans() or {}
        except Exception:
            spans = {}

    omitted_no_listed = 0
    delisted_added = 0
    for d in delisted_rows:
        sym = d["symbol"]
        if sym in by_sym:
            by_sym[sym]["delisted"] = d["delisted"]
            by_sym[sym]["delist_provenance"] = "nse_delisted"
            delisted_added += 1
            continue
        listed = None
        list_prov = None
        span = spans.get(sym)
        if span and span.get("first"):
            listed = str(span["first"])[:10]
            list_prov = "nse_bhav_first_seen_window_bound"
        if not listed:
            omitted_no_listed += 1
            continue
        if listed >= d["delisted"]:
            omitted_no_listed += 1
            continue
        by_sym[sym] = {
            "symbol": sym,
            "listed": listed,
            "delisted": d["delisted"],
            "listing_provenance": list_prov,
            "delist_provenance": "nse_delisted",
        }
        delisted_added += 1

    membership = [
        {k: v for k, v in row.items() if k in {"symbol", "listed", "delisted"}}
        for row in by_sym.values()
    ]
    note = (
        "Listings from NSE EQUITY_L; delistings from NSE delisted.csv. "
        "Delisted names without EQUITY_L listing use bhav first-seen as a "
        "window lower bound when available; undated names are omitted."
    )
    write_universe_history(
        membership,
        path=path,
        source="nse_equity_l+nse_delisted",
        note=note,
    )
    has_delist_rows = any(r.get("delisted") for r in membership)
    bhav_delisted_undated = 0
    for d in delisted_rows:
        if d["symbol"] in spans and d["symbol"] not in by_sym:
            bhav_delisted_undated += 1
    surv = bool(has_delist_rows) and bhav_delisted_undated == 0

    completeness = {
        "has_official_listings": True,
        "has_official_delistings": True,
        "survivorship_complete": surv,
        "reconstructed_from_survivors_only": False,
        "delisted_membership_rows": delisted_added,
        "delisted_omitted_no_listed_date": omitted_no_listed,
        "bhav_delisted_undated": bhav_delisted_undated,
    }
    p = history_path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw["source_meta"] = {"equity_l": eq_meta, "delisted": de_meta}
            raw["completeness"] = completeness
            raw["generated_at"] = datetime.now(timezone.utc).isoformat()
            p.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    except Exception:
        pass
    st = ledger_status(p)
    st["completeness"] = completeness
    st["source_meta"] = {"equity_l": eq_meta, "delisted": de_meta}
    return st


def materialize_universe_from_equity_l(**kwargs):
    return materialize_universe_from_nse(**kwargs)
