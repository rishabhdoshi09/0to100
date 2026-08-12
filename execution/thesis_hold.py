"""Thesis hold — keep a winner while technicals + fundamentals look good.

Not a scalp. Fixed ₹ / % profit-booking is optional and separate; this module
only answers: is the reason we entered still intact?
"""
from __future__ import annotations

from typing import Any, Mapping

RSI_BLOWOFF = 70.0
MIN_FUND_COVERAGE = 0.50
FUND_COLLAPSE = 30.0


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def evaluate_thesis(
    *,
    entry: float,
    stop: float,
    live_px: float,
    scan_row: Mapping[str, Any] | None = None,
    fund_row: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (still_good, reason). reason is empty when the thesis holds.

    Demote-only exits — never invents scan/fund data. Missing context means
    we do not force an exit from this lens (stop/GTT still protect).
    """
    if live_px > 0 and stop > 0 and live_px <= stop:
        return False, "price at/below stop — structure broken"

    if scan_row:
        rsi = _f(scan_row.get("rsi"))
        if rsi > RSI_BLOWOFF:
            return False, (
                f"RSI {rsi:.0f} blow-off — technicals no longer healthy"
            )
        status = str(scan_row.get("status") or "")
        verdict = str(scan_row.get("verdict") or "").upper()
        mom5 = _f(scan_row.get("momentum_5d"))
        if verdict == "AVOID":
            return False, "scanner verdict AVOID"
        if status == "Wait for pullback":
            return False, "setup broken — wait for pullback"
        if bool(scan_row.get("chase_risk")) and mom5 < 0:
            return False, "chase-risk + fading 5d momentum"

    if fund_row:
        cov = _f(fund_row.get("fundamental_coverage"))
        cls = str(fund_row.get("classification") or "")
        if cov >= MIN_FUND_COVERAGE and cls == "AVOID_REVIEW":
            return False, "fundamentals AVOID_REVIEW"
        fs = fund_row.get("fundamental_score")
        if cov >= MIN_FUND_COVERAGE and fs is not None and _f(fs) < FUND_COLLAPSE:
            return False, f"fundamentals collapsed ({_f(fs):.0f})"

    return True, ""


def runner_target(entry: float, signal_target: float, runner_pct: float) -> float:
    """Wide GTT ceiling so a healthy runner is not cut at a tiny scalp %."""
    entry = float(entry or 0)
    if entry <= 0:
        return 0.0
    wide = round(entry * (1 + max(1.0, float(runner_pct or 10)) / 100.0), 1)
    sig = float(signal_target or 0)
    if sig > entry:
        return max(wide, round(sig, 1))
    return wide
