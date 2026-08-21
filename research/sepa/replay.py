"""Historical candidate replay — print full SepaEligibility objects."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from research.sepa.config import DEFAULT_CONFIG
from research.sepa.engine import evaluate_sepa_eligibility


def replay_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for case in cases:
        result = evaluate_sepa_eligibility(
            case["symbol"],
            case["as_of"],
            frame=case.get("frame"),
            rs_percentile=case.get("rs_percentile"),
            rs_table=case.get("rs_table"),
            config=case.get("config") or DEFAULT_CONFIG,
            pit_meta=case.get("pit_meta") or {"universe_complete": False, "ca_complete": False},
            buy_zone_above_pct=case.get("buy_zone_above_pct"),
        )
        out.append({
            "label": case.get("label") or case["symbol"],
            "want": case.get("want") or "",
            "eligibility": result.to_dict(),
        })
    return out


def format_replay(rows: list[dict[str, Any]]) -> str:
    lines = ["# SEPA-001 historical candidate replay", ""]
    for row in rows:
        el = row["eligibility"]
        lines.append(f"## {row['label']}")
        if row.get("want"):
            lines.append(f"Intent: {row['want']}")
        lines.append(f"- eligible: `{el.get('eligible')}`")
        lines.append(f"- headline: {el.get('headline')}")
        lines.append(f"- good_stock / setup / entry: {el.get('good_stock')} / {el.get('good_setup')} / {el.get('good_entry')}")
        lines.append(f"- rejection: {el.get('rejection_codes')}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(el, indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def try_live_examples(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    """Best-effort NSE examples from the local bhav store. Empty if store cold."""
    from data.bhavcopy_runtime import ensure_loaded, get_ohlcv
    from research.sepa.rs import build_rs_table

    ensure_loaded(rebuild_from_local=False)
    names = [s.upper() for s in (symbols or ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"])]
    frames = {}
    for sym in names:
        df = get_ohlcv(sym)
        if df is not None and len(df) > 260:
            frames[sym] = df
    if not frames:
        return []
    as_of = min(df.index[-1] for df in frames.values())
    table = build_rs_table(frames, as_of, DEFAULT_CONFIG, universe=list(frames))
    rows = []
    for sym, df in frames.items():
        el = evaluate_sepa_eligibility(sym, as_of, frame=df, rs_table=table)
        rows.append({"label": f"NSE {sym} @ {as_of.date() if hasattr(as_of,'date') else as_of}",
                     "want": "live-store snapshot", "eligibility": el.to_dict()})
    return rows
