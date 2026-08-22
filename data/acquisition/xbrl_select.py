"""Bounded XBRL candidate lists — prefer recent consolidated quarters."""
from __future__ import annotations

from collections import defaultdict

from data.nse_results_ingest import _iso_date_from_nse, select_xbrl_candidates
from data.period_alignment import consol_label


def select_bounded(
    raw_rows: list[dict],
    *,
    min_year: int = 2019,
    max_per_symbol: int = 12,
    prefer_consolidated: bool = True,
) -> list[dict]:
    cands = select_xbrl_candidates(
        raw_rows, prefer_consolidated=prefer_consolidated, min_period_end_year=min_year,
    )
    if prefer_consolidated:
        consol = [r for r in cands if consol_label(r.get("consolidated")) == "CONSOLIDATED"]
        if len(consol) >= 100:
            cands = consol
    by: dict[str, list] = defaultdict(list)
    def pe(r):
        return _iso_date_from_nse(r.get("toDate")) or ""
    for r in sorted(cands, key=pe, reverse=True):
        sym = str(r.get("symbol") or "").upper()
        if not sym:
            continue
        if len(by[sym]) >= max_per_symbol:
            continue
        by[sym].append(r)
    out = [r for rows in by.values() for r in rows]
    return out
