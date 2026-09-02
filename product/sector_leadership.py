"""Sector leadership / participation score from existing scan + regime.

This is a ranking proxy, not a hard gate and not literal institutional cash-flow.
Missing inputs stay missing. A weak company or chase setup cannot be rescued.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _sector_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _boolish(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    if value in {1, "1", "true", "True", "yes"}:
        return True
    if value in {0, "0", "false", "False", "no"}:
        return False
    return None


def board_from_rows(
    scan_rows: Sequence[Mapping[str, Any]],
    leaders: Sequence[str] | None = None,
    laggards: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Project one leadership card per sector from the saved scan. No second scanner."""
    lead = {_sector_key(x) for x in (leaders or ()) if _sector_key(x)}
    lag = {_sector_key(x) for x in (laggards or ()) if _sector_key(x)}
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in scan_rows:
        key = _sector_key(row.get("sector"))
        if not key:
            continue
        buckets.setdefault(key, []).append(row)

    board: dict[str, dict[str, Any]] = {}
    for key, rows in buckets.items():
        n = len(rows)
        above = [r for r in rows if _boolish(r.get("above_sma50")) is True]
        known_sma = [r for r in rows if _boolish(r.get("above_sma50")) is not None]
        rs_vals = [_f(r.get("rs_percentile") or r.get("rs_score") or r.get("relative_strength")) for r in rows]
        rs_known = [v for v in rs_vals if v is not None]
        vol_vals = [_f(r.get("volume_ratio")) for r in rows]
        vol_known = [v for v in vol_vals if v is not None and v > 0]
        setups = [
            r for r in rows
            if str(r.get("status") or "") == "Ready to trade"
            or str(r.get("verdict") or "").upper() == "BUY"
            or str(r.get("decision") or "").upper() == "ENTER"
        ]
        pct_above = (100.0 * len(above) / len(known_sma)) if known_sma else None
        avg_rs = (sum(rs_known) / len(rs_known)) if rs_known else None
        avg_vol = (sum(vol_known) / len(vol_known)) if vol_known else None
        setup_pct = 100.0 * len(setups) / n if n else 0.0

        score = 50.0
        if key in lead:
            score += 18
        if key in lag:
            score -= 18
        if pct_above is not None:
            score += (pct_above - 50.0) * 0.35
        if avg_rs is not None:
            score += (avg_rs - 50.0) * 0.25
        if setup_pct:
            score += min(12.0, setup_pct * 0.25)
        if avg_vol is not None and avg_vol >= 1.2:
            score += 8
        score = max(0.0, min(100.0, score))

        if key in lead and (pct_above or 0) >= 55:
            kind = "Sector Leadership"
        elif avg_vol is not None and avg_vol >= 1.3:
            kind = "Sector Money-Flow Proxy"
        else:
            kind = "Sector Participation"

        breadth = "Strong" if (pct_above or 0) >= 60 else ("Weak" if pct_above is not None and pct_above < 40 else ("Mixed" if pct_above is not None else "Not available"))
        momentum = "Improving" if (avg_rs or 0) >= 60 else ("Fading" if avg_rs is not None and avg_rs < 40 else ("Steady" if avg_rs is not None else "Not available"))
        volume = "High" if avg_vol is not None and avg_vol >= 1.3 else ("Low" if avg_vol is not None and avg_vol < 0.8 else ("Normal" if avg_vol is not None else "Not available"))
        board[key] = {
            "sector": rows[0].get("sector") or key,
            "score": round(score, 1),
            "label": kind,
            "n": n,
            "breadth": breadth,
            "momentum": momentum,
            "volume_participation": volume,
            "pct_above_sma50": None if pct_above is None else round(pct_above, 1),
            "avg_rs": None if avg_rs is None else round(avg_rs, 1),
            "avg_volume_ratio": None if avg_vol is None else round(avg_vol, 2),
            "strong_setups": len(setups),
            "leader": key in lead,
            "laggard": key in lag,
            "not_institutional_cashflow": True,
        }
    return board


def attach_to_row(row: Mapping[str, Any], board: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    key = _sector_key(row.get("sector"))
    card = dict(board.get(key) or {})
    if not card:
        return {
            "sector_leadership_score": None,
            "sector_leadership_label": "",
            "sector_breadth": "",
            "sector_momentum": "",
            "sector_volume_participation": "",
        }
    return {
        "sector_leadership_score": card.get("score"),
        "sector_leadership_label": card.get("label") or "",
        "sector_breadth": card.get("breadth") or "",
        "sector_momentum": card.get("momentum") or "",
        "sector_volume_participation": card.get("volume_participation") or "",
        "sector_leader": bool(card.get("leader") or row.get("sector_leader")),
        "sector_laggard": bool(card.get("laggard") or row.get("sector_laggard")),
    }


def ranking_boost(row: Mapping[str, Any]) -> float:
    """Tie-break only. Never a permission to bypass hard gates."""
    score = _f(row.get("sector_leadership_score"))
    return 0.0 if score is None else score
