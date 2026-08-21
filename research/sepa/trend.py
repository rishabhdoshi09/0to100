"""Strict 8-rule Minervini Trend Template — AND gate, fail-closed."""
from __future__ import annotations

from typing import Any

from research.sepa.config import SepaConfig
from research.sepa.frames import close_series, sma
from research.sepa.types import RuleResult


def evaluate_trend(frame, config: SepaConfig, *, rs_percentile: float | None) -> dict[str, Any]:
    close = close_series(frame)
    rules: list[RuleResult] = []
    levels: dict[str, Any] = {}
    if close is None or frame is None or len(frame) == 0:
        for spec in _SPECS:
            rules.append(RuleResult(spec[0], None, "No official history."))
        return _pack(rules, levels)

    price = float(close.iloc[-1])
    high_col = frame["high"] if "high" in frame.columns else close
    low_col = frame["low"] if "low" in frame.columns else close
    win = min(int(config.high_low_lookback), len(frame))
    high_52w = float(high_col.astype(float).tail(win).max())
    low_52w = float(low_col.astype(float).tail(win).min())
    s50 = sma(close, config.sma50)
    s150 = sma(close, config.sma150)
    s200 = sma(close, config.sma200)
    s200_prev = None
    need_slope = config.sma200 + config.sma200_slope_lookback
    if len(close) >= need_slope:
        s200_prev = sma(close.iloc[: -config.sma200_slope_lookback], config.sma200)

    below_high = ((1.0 - price / high_52w) * 100.0) if high_52w > 0 else None
    above_low = ((price / low_52w - 1.0) * 100.0) if low_52w > 0 else None
    levels = {
        "price": round(price, 4),
        "sma50": None if s50 is None else round(s50, 4),
        "sma150": None if s150 is None else round(s150, 4),
        "sma200": None if s200 is None else round(s200, 4),
        "sma200_prev": None if s200_prev is None else round(s200_prev, 4),
        "high_52w": round(high_52w, 4),
        "low_52w": round(low_52w, 4),
        "below_high_pct": None if below_high is None else round(below_high, 2),
        "above_low_pct": None if above_low is None else round(above_low, 2),
    }

    def add(rid: str, passed: bool | None, detail: str, **values):
        rules.append(RuleResult(rid, passed, detail, values))

    if s150 is None or s200 is None:
        add("price_gt_150_200", None, "Need 200 sessions for SMA150/SMA200.",
            price=price, sma150=s150, sma200=s200)
    else:
        ok = price > s150 and price > s200
        add("price_gt_150_200", ok,
            f"Close {price:.2f} vs SMA150 {s150:.2f} / SMA200 {s200:.2f}",
            price=price, sma150=s150, sma200=s200)

    if s150 is None or s200 is None:
        add("sma150_gt_200", None, "Need 200 sessions to compare SMA150 vs SMA200.")
    else:
        ok = s150 > s200
        add("sma150_gt_200", ok, f"SMA150 {s150:.2f} vs SMA200 {s200:.2f}",
            sma150=s150, sma200=s200)

    if s200 is None or s200_prev is None:
        add("sma200_rising", None,
            f"Need {need_slope} sessions to test whether SMA200 is rising.")
    else:
        ok = s200 > s200_prev
        add("sma200_rising", ok,
            f"SMA200 {s200:.2f} vs {s200_prev:.2f} ({config.sma200_slope_lookback} sessions earlier)",
            sma200=s200, sma200_prev=s200_prev)

    if s50 is None or s150 is None or s200 is None:
        add("sma50_leads", None, "Need 200 sessions for the 50/150/200 stack.")
    else:
        ok = s50 > s150 and s50 > s200
        add("sma50_leads", ok,
            f"SMA50 {s50:.2f} vs SMA150 {s150:.2f} / SMA200 {s200:.2f}",
            sma50=s50, sma150=s150, sma200=s200)

    if s50 is None:
        add("price_gt_sma50", None, "Need 50 sessions for SMA50.")
    else:
        ok = price > s50
        add("price_gt_sma50", ok, f"Close {price:.2f} vs SMA50 {s50:.2f}",
            price=price, sma50=s50)

    if above_low is None:
        add("off_52w_low", None, "52-week low unavailable.")
    else:
        ok = above_low >= config.off_52w_low_pct
        add("off_52w_low", ok,
            f"{above_low:.1f}% above 52-week low (need ≥{config.off_52w_low_pct:.0f}%)",
            above_low_pct=above_low, threshold=config.off_52w_low_pct, low_52w=low_52w)

    if below_high is None:
        add("near_52w_high", None, "52-week high unavailable.")
    else:
        ok = below_high <= config.near_52w_high_pct
        add("near_52w_high", ok,
            f"{below_high:.1f}% below 52-week high (need ≤{config.near_52w_high_pct:.0f}%)",
            below_high_pct=below_high, threshold=config.near_52w_high_pct, high_52w=high_52w)

    if rs_percentile is None:
        add("rs_percentile", None, "Cross-sectional RS unavailable (fail-closed).")
    else:
        ok = float(rs_percentile) >= config.rs_threshold
        add("rs_percentile", ok,
            f"RS percentile {float(rs_percentile):.1f} (need ≥{config.rs_threshold:.0f})",
            rs_percentile=float(rs_percentile), threshold=config.rs_threshold)

    return _pack(rules, levels)


_SPECS = (
    ("price_gt_150_200",),
    ("sma150_gt_200",),
    ("sma200_rising",),
    ("sma50_leads",),
    ("price_gt_sma50",),
    ("off_52w_low",),
    ("near_52w_high",),
    ("rs_percentile",),
)


def _pack(rules: list[RuleResult], levels: dict[str, Any]) -> dict[str, Any]:
    passed = sum(1 for r in rules if r.passed is True)
    unknown = sum(1 for r in rules if r.passed is None)
    fail = sum(1 for r in rules if r.passed is False)
    strict = passed == 8 and unknown == 0 and fail == 0
    near = (passed == 7 and unknown == 0 and fail == 1)
    return {
        "rules": rules,
        "passed": passed,
        "total": 8,
        "unknown": unknown,
        "trend_template_pass": strict,
        "near_sepa": near and not strict,
        "levels": levels,
    }
