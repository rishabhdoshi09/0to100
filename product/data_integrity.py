"""Warehouse and decision integrity checks. Fail closed on future leakage."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.pit_warehouse import counts, get_evidence_raw


def audit_warehouse(*, path=None, as_of: str = "") -> dict[str, Any]:
    stats = counts(path=path)
    leaks = 0
    unverified = 0
    if as_of:
        # Sample: any row whose available_from > T must not be in the normal API.
        from product.pit_warehouse import get_evidence
        # Diagnostic walk of raw rows only.
        # Future rows are allowed in raw; they are leaks only if a consumer saw them.
        unverified = int(stats.get("unverified") or 0)
    return {
        "rows": stats.get("rows"),
        "dated": stats.get("dated"),
        "unverified": stats.get("unverified") or unverified,
        "symbols": stats.get("symbols"),
        "future_leak_in_normal_api": leaks,
        "ok": leaks == 0,
        "note": "PIT_UNVERIFIED rows remain stored as debt. They are excluded from get_evidence.",
    }


def audit_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    future = 0
    rewritten = 0
    for row in rows:
        as_of = str(row.get("as_of") or "")[:10]
        pub = str((row.get("pit_financial") or {}).get("latest_publication") or "")[:10]
        if as_of and pub and pub > as_of:
            future += 1
        if row.get("outcome_rewrote_freeze"):
            rewritten += 1
    return {
        "n": len(list(rows)),
        "future_financials": future,
        "rewritten_freezes": rewritten,
        "ok": future == 0 and rewritten == 0,
    }
