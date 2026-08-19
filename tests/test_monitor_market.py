"""Market-monitor breadth + SEPA modules — no invented internals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from product.monitor_context import breakout_readiness, sepa_modules, volume_pattern
from product.monitor_market import breadth_from_closes, news_tape
from product.sepa_setup import score_sepa


def _closes(n: int, start: float, step: float) -> np.ndarray:
    return np.array([start + i * step for i in range(n)], dtype=float)


def test_breadth_adv_ratio_and_pct_above_20():
    up = _closes(60, 100, 0.4)
    down = _closes(60, 200, -0.5)
    nifty = _closes(60, 24000, -2)
    idx = pd.date_range("2026-05-01", periods=60, freq="B")
    payload = breadth_from_closes(
        [(up, idx), (up.copy(), idx), (down, idx)],
        nifty_close=nifty,
        nifty_index=idx,
        sessions=3,
        min_n=3,
    )
    assert payload["available"] is True
    assert payload["advancers"] == 2
    assert payload["decliners"] == 1
    assert payload["adv_ratio"] == round(2 / 1, 2)
    assert payload["pct_above_20"] is not None
    assert payload["history"]
    assert payload["history"][0]["nifty_close"] == round(float(nifty[-1]), 2)


def test_thin_sample_does_not_invent_breadth():
    up = _closes(30, 100, 0.2)
    idx = pd.date_range("2026-06-01", periods=30, freq="B")
    payload = breadth_from_closes([(up, idx)], sessions=2, min_n=300)
    assert payload["available"] is False
    assert payload["history"] == []


def test_volume_accumulation_when_dryup_and_up_days_lead():
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    px = 100.0
    closes = []
    vols = []
    for i in range(80):
        if i % 4 == 0:
            px -= 0.2
            vol = 18000 if i >= 60 else 90000
        else:
            px += 0.55
            vol = 50000 if i >= 60 else 220000
        closes.append(px)
        vols.append(vol)
    close = pd.Series(closes, index=index)
    frame = pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.4, "low": close - 0.3, "close": close, "volume": vols},
        index=index,
    )
    out = volume_pattern(frame)
    assert out["available"] is True
    assert out["dryup"] is True
    assert out["up_down"] is not None and out["up_down"] >= 1.4
    assert out["label"] == "ACCUMULATION"


def test_breakout_readiness_scores_a_tight_leader():
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series([80 + i * 0.5 for i in range(80)], index=index)
    high = close + 0.4
    high.iloc[-1] = close.iloc[-1] + 0.05
    low = close - 0.4
    low.iloc[-1] = close.iloc[-1] - 0.04
    vol = [150000] * 60 + [40000] * 20
    frame = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=index,
    )
    out = breakout_readiness(frame, {"levels": {"sma50": float(close.iloc[-50:].mean())}})
    assert out["available"] is True
    assert out["score"] >= 45
    assert out["label"] in {"COILING", "READY"}


def test_score_sepa_exposes_six_analyser_modules():
    index = pd.date_range("2024-01-01", periods=280, freq="B")
    close = pd.Series([70 + i * 0.55 for i in range(280)], index=index)
    frame = pd.DataFrame(
        {"open": close - 0.4, "high": close + 1.0, "low": close - 0.9, "close": close, "volume": [100000] * 280},
        index=index,
    )
    bench = frame.copy()
    bench["close"] = pd.Series([20000 + i * 4.0 for i in range(280)], index=index)
    sepa = score_sepa(frame, bench_frame=bench)
    mods = sepa.get("modules") or sepa_modules(sepa)
    assert len(mods) == 6
    assert [m["id"] for m in mods] == [
        "trend_template", "near_52w", "rs_nifty", "volume", "stage", "breakout",
    ]
    assert sepa["volume"]["available"] is True
    assert sepa["breakout"]["available"] is True


def test_news_tape_never_invents_headlines():
    out = news_tape()
    assert "items" in out
    assert out["available"] is bool(out["items"])
    assert all(str(item.get("headline") or "").strip() for item in out["items"])
