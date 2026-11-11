"""Pullback Entries: leaders resting at support, not extended charts (owner 2026-11-11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scan.pullback_scanner import evaluate_pullback, find_pullbacks


def _series(n=300, seed=5, pullback=True):
    rng = np.random.default_rng(seed)
    drift = np.r_[np.full(n - 6, 0.0025), np.full(6, -0.006 if pullback else 0.01)]
    close = 500 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    high = close * 1.006
    low = close * 0.994
    vol = np.r_[np.full(n - 3, 1_000_000.0), np.full(3, 500_000.0)]
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol})


def test_quiet_pullback_in_uptrend_is_found_with_a_plan():
    row = evaluate_pullback("LEADER", _series())
    assert row is not None, "quiet dip to support in a strong trend should qualify"
    assert row["entry_state"] == "ready" and not row["chase_risk"]
    assert row["stop"] < row["entry"] < row["target"]
    assert row["reward_risk"] >= 1.5 and 35 <= row["rsi"] <= 55
    assert "PULLBACK_TO_SUPPORT" in row["signals"] and row["stop_plan"]["recommended_stop"] == row["stop"]


def test_extended_or_downtrend_names_are_not_pullbacks():
    assert evaluate_pullback("RUNNER", _series(pullback=False)) is None        # still rising
    falling = _series()
    falling["close"] = falling["close"][::-1].values
    falling["high"], falling["low"] = falling["close"] * 1.006, falling["close"] * 0.994
    assert evaluate_pullback("FALLER", falling) is None                       # no uptrend
    assert evaluate_pullback("SHORT", _series().tail(100)) is None            # too little history


def test_find_pullbacks_ranks_and_never_raises():
    frames = {"A": _series(seed=5), "B": _series(seed=9), "BAD": None}
    rows = find_pullbacks(["A", "B", "BAD", "MISSING"], frames.get, limit=5)
    assert all(r["symbol"] in {"A", "B"} for r in rows)
    assert rows == sorted(rows, key=lambda r: (-r["score"], r["symbol"]))


def test_workspace_shows_pullbacks_in_their_own_category():
    from product.recommendations_workspace import build_recommendations_workspace

    row = evaluate_pullback("LEADER", _series())
    out = build_recommendations_workspace(scan_payload={
        "scanned_at": "2026-11-11T05:00:00+00:00", "records": [], "pullback_candidates": [row]})
    cat = [c for c in out["categories"] if c["id"] == "pullback_entries"][0]
    assert cat["count"] == 1 and cat["cards"][0]["symbol"] == "LEADER"
    empty = build_recommendations_workspace(scan_payload={"scanned_at": "x", "records": []})
    detail = [c for c in empty["categories"] if c["id"] == "pullback_entries"][0]["empty_detail"]
    assert "resting at support" in detail
