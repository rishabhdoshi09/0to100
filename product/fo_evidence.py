"""Evidence/calibration primitives for the NSE F&O paper lane.

Only settled FORWARD_PAPER outcomes may create a production probability/EV
claim. Historical replay and model-only scenarios remain useful research
evidence but cannot masquerade as forward validation.
"""
from __future__ import annotations

from math import sqrt
from typing import Any, Iterable, Mapping


FORWARD_PAPER = "FORWARD_PAPER"
HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
MODEL_ONLY = "MODEL_ONLY"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def wilson_lower_bound(p: float, n: int, z: float = 1.28) -> float:
    if n <= 0:
        return 0.0
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    spread = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - spread) / denom)


def _bucket(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def fo_context_key(
    *,
    direction: str,
    futures_oi_state: str,
    rvol: float,
    adx: float,
    delta: float,
    dte: int,
    iv_percentile: float | None,
) -> str:
    """Stable coarse context key; avoids overfitting to exact continuous values."""
    rvol_bucket = _bucket(_f(rvol), (1.5, 2.0, 2.5), ("RVOL_LT1.5", "RVOL_1.5_2", "RVOL_2_2.5", "RVOL_GE2.5"))
    adx_bucket = _bucket(_f(adx), (20.0, 25.0, 30.0), ("ADX_LT20", "ADX_20_25", "ADX_25_30", "ADX_GE30"))
    delta_bucket = _bucket(abs(_f(delta)), (0.45, 0.55, 0.65, 0.75), ("D_35_45", "D_45_55", "D_55_65", "D_65_75", "D_GE75"))
    dte_bucket = _bucket(float(dte), (5.0, 8.0, 15.0, 31.0), ("DTE_2_4", "DTE_5_7", "DTE_8_14", "DTE_15_30", "DTE_31P"))
    if iv_percentile is None:
        iv_bucket = "IV_UNKNOWN"
    else:
        iv_bucket = _bucket(_f(iv_percentile), (30.0, 60.0, 80.0), ("IV_LOW", "IV_NORMAL", "IV_HIGH", "IV_EXTREME"))
    return "|".join((
        str(direction or "").upper(),
        str(futures_oi_state or "NEUTRAL").upper(),
        rvol_bucket,
        adx_bucket,
        delta_bucket,
        dte_bucket,
        iv_bucket,
    ))


def _max_drawdown(returns_pct: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for ret in returns_pct:
        equity *= max(0.0, 1.0 + ret / 100.0)
        peak = max(peak, equity)
        drawdown = (equity / peak - 1.0) * 100.0 if peak > 0 else -100.0
        worst = min(worst, drawdown)
    return abs(worst)


def summarize_fo_outcomes(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    context_key: str | None = None,
    evidence_lane: str = FORWARD_PAPER,
    min_n: int = 30,
) -> dict[str, Any]:
    """Summarize settled outcomes without inventing a probability claim."""
    rows = []
    for raw in outcomes:
        row = dict(raw)
        if not bool(row.get("settled", False)):
            continue
        if str(row.get("evidence_lane") or "").upper() != evidence_lane:
            continue
        if context_key is not None and str(row.get("context_key") or "") != context_key:
            continue
        if row.get("net_option_return_pct") is None:
            continue
        rows.append(row)

    rows.sort(key=lambda row: str(row.get("settled_at") or row.get("opened_at") or ""))
    returns = [_f(row.get("net_option_return_pct")) for row in rows]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    n = len(returns)
    p = len(wins) / n if n else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy = p * avg_win - (1.0 - p) * avg_loss if n else 0.0
    p_lb = wilson_lower_bound(p, n)
    conservative_ev = p_lb * avg_win - (1.0 - p_lb) * avg_loss if n else 0.0

    mfes = [_f(row.get("mfe_pct")) for row in rows if row.get("mfe_pct") is not None]
    maes = [_f(row.get("mae_pct")) for row in rows if row.get("mae_pct") is not None]
    false_breakouts = sum(1 for row in rows if bool(row.get("false_breakout", False)))
    claim = evidence_lane == FORWARD_PAPER and n >= max(1, int(min_n))

    return {
        "context_key": context_key,
        "evidence_lane": evidence_lane,
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "probability_claim_available": claim,
        "win_probability_pct": round(p * 100.0, 2) if claim else None,
        "win_probability_wilson_lb_pct": round(p_lb * 100.0, 2) if claim else None,
        "expectancy_pct": round(expectancy, 3) if claim else None,
        "conservative_ev_pct": round(conservative_ev, 3) if claim else None,
        "avg_win_pct": round(avg_win, 3) if wins else None,
        "avg_loss_pct": round(avg_loss, 3) if losses else None,
        "avg_mfe_pct": round(sum(mfes) / len(mfes), 3) if mfes else None,
        "avg_mae_pct": round(sum(maes) / len(maes), 3) if maes else None,
        "false_breakout_rate_pct": round(false_breakouts / n * 100.0, 2) if n else None,
        "max_drawdown_pct": round(_max_drawdown(returns), 3) if returns else None,
        "insufficient_evidence": not claim,
        "minimum_required_n": max(1, int(min_n)),
        "production_influence_allowed": claim,
    }
