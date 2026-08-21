"""Old vs new VCP timing diagnostics (research only)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from research.sepa.config import DEFAULT_CONFIG, LEGACY_CONFIG, SepaConfig
from research.sepa.frames import iso_date
from research.sepa.vcp import detect_vcp, detect_vcp_legacy


def _px(frame, i: int) -> float:
    return float(frame["close"].iloc[i])


def diagnose_symbol(
    symbol: str,
    frame: pd.DataFrame,
    *,
    config: SepaConfig | None = None,
    start: int | None = None,
) -> dict[str, Any]:
    """Walk daily and record first-knowable dates for legacy vs causal VCP."""
    cfg = config or DEFAULT_CONFIG
    n = len(frame)
    if n < 80:
        return {"symbol": symbol, "note": "short history"}
    t0 = start if start is not None else max(40, n - 400)
    first_new = first_legacy = first_entry = first_break = None
    new_dist = legacy_dist = None
    pivot_new = pivot_legacy = None
    rows = []
    for t in range(t0, n):
        hist = frame.iloc[: t + 1]
        as_of = iso_date(hist.index[-1])
        price = _px(hist, -1)
        neu = detect_vcp(hist, cfg)
        old = detect_vcp_legacy(hist, LEGACY_CONFIG)
        if neu.get("detected") and first_new is None:
            first_new = {
                "date": as_of,
                "pivot": neu.get("pivot"),
                "price": price,
                "distance_pct": None if not neu.get("pivot") else round((price / neu["pivot"] - 1) * 100.0, 3),
                "state": neu.get("state"),
                "pivot_knowable_date": neu.get("pivot_knowable_date"),
                "vcp_knowable_date": neu.get("vcp_knowable_date"),
                "base_start_date": neu.get("base_start_date"),
            }
            new_dist = first_new["distance_pct"]
            pivot_new = neu.get("pivot")
        if old.get("detected") and first_legacy is None:
            first_legacy = {
                "date": as_of,
                "pivot": old.get("pivot"),
                "price": price,
                "distance_pct": None if not old.get("pivot") else round((price / old["pivot"] - 1) * 100.0, 3),
                "state": old.get("state"),
            }
            legacy_dist = first_legacy["distance_pct"]
            pivot_legacy = old.get("pivot")
        if neu.get("detected") and neu.get("state") == "ENTRY_READY" and first_entry is None:
            first_entry = as_of
        if neu.get("broken_out") and first_break is None:
            first_break = as_of
        if neu.get("detected") or old.get("detected"):
            rows.append({
                "as_of": as_of,
                "new_detected": bool(neu.get("detected")),
                "legacy_detected": bool(old.get("detected")),
                "new_state": neu.get("state"),
                "new_dist": None if not neu.get("pivot") else round((price / neu["pivot"] - 1) * 100.0, 3),
                "legacy_dist": None if not old.get("pivot") else round((price / old["pivot"] - 1) * 100.0, 3),
            })
    latency = None
    if first_new and first_legacy:
        try:
            latency = int((pd.Timestamp(first_legacy["date"]) - pd.Timestamp(first_new["date"])).days)
        except Exception:
            latency = None
    in_zone = False
    if first_new and first_new.get("distance_pct") is not None:
        in_zone = -0.25 <= float(first_new["distance_pct"]) <= 1.5
    return {
        "symbol": symbol,
        "setup_start": (first_new or {}).get("base_start_date"),
        "first_knowable_vcp": (first_new or {}).get("vcp_knowable_date") or (first_new or {}).get("date"),
        "pivot_knowable": (first_new or {}).get("pivot_knowable_date"),
        "pivot_price": pivot_new,
        "breakout": first_break,
        "first_valid_entry": first_entry,
        "old_detection": (first_legacy or {}).get("date"),
        "old_dist_to_pivot": legacy_dist,
        "new_detection": (first_new or {}).get("date"),
        "new_dist_to_pivot": new_dist,
        "new_in_buy_zone_at_detection": in_zone,
        "legacy_later_by_calendar_days": latency,
        "legacy_pivot": pivot_legacy,
        "trace_n": len(rows),
    }


def format_timing_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SEPA-001R VCP timing diagnostics",
        "",
        "Old detector = frozen SEPA-001 (pattern-high pivot + 92% near-pivot VCP fail).",
        "New detector = causal last-contraction pivot; distance-to-pivot is entry, not a VCP fail.",
        "",
        "| Symbol | Setup Start | First Knowable VCP | Pivot Knowable | Breakout | Old Detection | Old Dist. to Pivot | New Detection | New Dist. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {symbol} | {setup_start} | {first_knowable_vcp} | {pivot_knowable} | {breakout} | {old_detection} | {old_dist_to_pivot} | {new_detection} | {new_dist_to_pivot} |".format(
                **{k: ("" if r.get(k) is None else r.get(k)) for k in (
                    "symbol", "setup_start", "first_knowable_vcp", "pivot_knowable",
                    "breakout", "old_detection", "old_dist_to_pivot", "new_detection",
                    "new_dist_to_pivot",
                )}
            )
        )
    return "\n".join(lines) + "\n"
