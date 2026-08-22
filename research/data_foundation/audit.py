"""Automated self-audit for a frozen dataset. Download ≠ RESEARCH_READY."""
from __future__ import annotations

from typing import Any


def run(
    rows: list[dict],
    *,
    key_fields: tuple[str, ...] = ("symbol", "available_at"),
    as_of: str | None = None,
    sample: int = 3,
) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(rows, list):
        return {"ok": False, "issues": ["rows_not_a_list"], "status": "UNUSABLE"}

    # schema
    missing_keys = 0
    for r in rows:
        if not isinstance(r, dict):
            missing_keys += 1
            continue
        for k in key_fields:
            if r.get(k) in (None, ""):
                missing_keys += 1
                break
    if missing_keys:
        issues.append(f"schema_missing_keys:{missing_keys}")

    # duplicates
    seen: set[str] = set()
    dups = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        ident = "|".join(str(r.get(k) or "") for k in key_fields)
        if ident in seen:
            dups += 1
        seen.add(ident)
    if dups:
        issues.append(f"duplicates:{dups}")

    # time order
    dates = [str(r.get("available_at") or r.get("date") or "")[:10] for r in rows if isinstance(r, dict)]
    ordered = dates == sorted(dates)
    if dates and not ordered:
        issues.append("time_order_not_sorted")

    # as-of invariance / future leakage
    leaked = 0
    if as_of:
        for r in rows:
            if not isinstance(r, dict):
                continue
            avail = str(r.get("available_at") or r.get("date") or "")[:10]
            if avail and avail > as_of:
                leaked += 1
    if leaked:
        issues.append(f"future_leakage:{leaked}")

    n = len(rows)
    missing_rate = (missing_keys / n) if n else 1.0
    stale_rate = None  # caller supplies field-specific staleness

    samples = [r for r in rows[:sample] if isinstance(r, dict)]
    provenance_ok = all(
        (r.get("source") or r.get("source_hash") or r.get("row_id")) for r in samples
    ) if samples else False
    if samples and not provenance_ok:
        issues.append("provenance_missing_on_sample")

    return {
        "ok": not any(x.startswith("future_leakage") or x.startswith("schema") for x in issues),
        "n": n,
        "duplicates": dups,
        "time_order_sorted": ordered or not dates,
        "future_leakage": leaked,
        "missing_rate": missing_rate,
        "stale_rate": stale_rate,
        "sample_records": samples,
        "provenance_ok": provenance_ok,
        "issues": issues,
        "as_of_invariance": leaked == 0 if as_of else None,
    }
