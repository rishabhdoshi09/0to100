"""Materialize PIT universe membership from official NSE EQUITY_L listing dates.

EQUITY_L is the current equity master: listing dates are official for names that
are still listed. Official delisting dates are NOT in EQUITY_L — those remain
unknown. Therefore ``survivorship_complete`` stays False until a delisting
archive is supplied. Today's survivors are never back-filled as if they were
the historical universe.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from data.security_identity import fetch_equity_l
from data.universe_history import history_path, write_universe_history, ledger_status


def materialize_universe_from_equity_l(
    *,
    path: str | Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    rows, meta = fetch_equity_l(session=session)
    membership = [
        {"symbol": r["symbol"], "listed": r["listing_date"]}
        for r in rows
        if r.get("symbol") and r.get("listing_date")
    ]
    note = (
        "Listing dates from NSE EQUITY_L (current EQ master). "
        "Delisting dates unknown — survivorship_complete remains False until an "
        "official delisting archive is ingested. Do not treat as full historical "
        "universe reconstruction."
    )
    status = write_universe_history(
        membership,
        path=path,
        source="nse_equity_l",
        note=note,
    )
    # Stamp honest completeness flags into the file (beyond source label).
    p = history_path(path)
    try:
        import json
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw["source_meta"] = meta
            raw["completeness"] = {
                "has_official_listings": True,
                "has_official_delistings": False,
                "survivorship_complete": False,
                "reconstructed_from_survivors_only": True,
            }
            raw["generated_at"] = datetime.now(timezone.utc).isoformat()
            p.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    except Exception:
        pass
    st = ledger_status(p)
    st["completeness"] = {
        "has_official_listings": True,
        "has_official_delistings": False,
        "survivorship_complete": False,
        "reconstructed_from_survivors_only": True,
    }
    st["source_meta"] = meta
    # Force research_grade False at materialization time — earned only by gate.
    st["research_grade"] = False
    st["research_grade_note"] = (
        "Source label alone does not earn RESEARCH_GRADE. Delistings missing → "
        "survivorship incomplete."
    )
    return st
