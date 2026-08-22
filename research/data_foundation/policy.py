"""Missing-data policy. Never silent 0-fill, never silent PIT upgrade."""
from __future__ import annotations

FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
CARRY_FORWARD = "CARRY_FORWARD"

POLICY = {
    "ohlcv_close": {
        "missing": FAIL,
        "zero_fill": False,
        "max_staleness_sessions": 5,
        "note": "Stale prices cannot be used indefinitely.",
    },
    "fundamentals_metric": {
        "missing": UNKNOWN,
        "zero_fill": False,
        "forward_fill_across_unknown_filing": False,
        "usable_only_after": "available_at",
        "note": "A quarter is invisible until its filing timestamp.",
    },
    "derived_ratio": {
        "missing": UNKNOWN,
        "zero_fill": False,
        "requires": "pit_inputs_and_calc_version",
    },
    "earnings_surprise": {
        "missing": FAIL,
        "zero_fill": False,
        "note": "Do not compute without a genuine historical consensus series.",
    },
    "sector": {
        "missing": UNKNOWN,
        "copy_today_backward_as_pit": False,
        "static_backfill_label": "STATIC_BACKFILL",
    },
    "benchmark_return": {
        "missing": UNKNOWN,
        "zero_fill": False,
        "note": "Missing benchmark return is not 0.",
    },
    "universe_membership": {
        "missing": FAIL,
        "invent_listing": False,
        "delisted_remains_investable": False,
    },
    "ca_factor": {
        "missing": FAIL,
        "infer_from_price_gap": False,
    },
}


def decide(field: str, value, *, stale_sessions: int | None = None) -> str:
    rule = POLICY.get(field) or {"missing": UNKNOWN, "zero_fill": False}
    if value is None or value == "":
        return str(rule.get("missing") or UNKNOWN)
    if field == "ohlcv_close" and stale_sessions is not None:
        max_s = int(rule.get("max_staleness_sessions") or 5)
        if stale_sessions > max_s:
            return FAIL
    return "OK"
