"""Point-in-time technical features for the learning record."""

from __future__ import annotations

import pandas as pd
import pytest

from product import pit_features as PF


def _frame(n=80, start="2026-06-01", close=100.0, vol=1_000_000.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": vol},
        index=idx,
    )


@pytest.mark.parametrize(
    "decided_at,expected",
    [
        ("2026-09-04T10:00:00+05:30", "2026-09-03"),  # intraday: today's bar not complete
        ("2026-09-04T15:30:00+05:30", "2026-09-04"),  # at the close
        ("2026-09-04T09:30:00+00:00", "2026-09-03"),  # 15:00 IST
        ("2026-09-04T10:30:00Z", "2026-09-04"),       # 16:00 IST
        ("2026-09-04T22:45:00", "2026-09-04"),        # naive = IST, after close
        ("", None),
        ("not a date", None),
    ],
)
def test_cutoff_session_never_includes_an_incomplete_bar(decided_at, expected):
    got = PF.cutoff_session(decided_at)
    assert (got.isoformat() if got else None) == expected


def test_known_values_on_a_flat_series():
    f = PF.technical_features("X", "2026-12-31T16:00:00+05:30", frame=_frame())
    assert f["atr_pct"] == pytest.approx(2.0)
    assert f["adr_pct"] == pytest.approx(2.0)
    assert f["clv"] == pytest.approx(0.0)
    assert f["dist_from_high_pct"] == pytest.approx(100 * 1 / 101, rel=1e-6)
    assert f["dist_20ema_pct"] == pytest.approx(0.0, abs=1e-9)
    assert f["dist_50dma_pct"] == pytest.approx(0.0, abs=1e-9)
    assert f["liquidity_cr"] == pytest.approx(100.0 * 1_000_000 / 1e7)
    assert "volume_z" not in f  # constant volume: z undefined, never invented


def test_future_bars_cannot_leak_into_a_decision():
    frame = _frame(n=80)
    decided = f"{frame.index[60].date().isoformat()}T16:00:00+05:30"
    before = PF.technical_features("X", decided, frame=frame)
    spiked = frame.copy()
    spiked.iloc[61:, spiked.columns.get_loc("close")] = 500.0
    spiked.iloc[61:, spiked.columns.get_loc("high")] = 600.0
    spiked.iloc[61:, spiked.columns.get_loc("volume")] = 9e9
    assert PF.technical_features("X", decided, frame=spiked) == before


def test_intraday_decision_ignores_that_days_bar():
    frame = _frame(n=80)
    day = frame.index[60].date().isoformat()
    moved = frame.copy()
    moved.iloc[60, moved.columns.get_loc("close")] = 130.0
    moved.iloc[60, moved.columns.get_loc("high")] = 131.0
    intraday = PF.technical_features("X", f"{day}T11:00:00+05:30", frame=moved)
    after_close = PF.technical_features("X", f"{day}T16:00:00+05:30", frame=moved)
    assert intraday["dist_20ema_pct"] == pytest.approx(0.0, abs=1e-9)
    assert after_close["dist_20ema_pct"] > 0


def test_short_or_missing_history_yields_nothing():
    assert PF.technical_features("X", "2026-12-31T16:00:00+05:30", frame=_frame(n=10)) == {}
    assert PF.technical_features("X", "2026-12-31T16:00:00+05:30", frame=pd.DataFrame()) == {}
    assert PF.technical_features("X", "", frame=_frame()) == {}


def test_fill_missing_never_overrides_a_value_the_decision_carried():
    merged, prov = PF.fill_missing({"atr_pct": 7.5, "clv": None}, "X", "2026-12-31T16:00:00+05:30", frame=_frame())
    assert merged["atr_pct"] == 7.5
    assert merged["clv"] == pytest.approx(0.0)
    assert "atr_pct" not in prov["filled"] and "clv" in prov["filled"]
    assert prov["source"] == "official_bhavcopy_pit"


def test_frozen_learning_record_gets_features_but_live_analog_query_does_not(monkeypatch):
    from product import evidence_intelligence as EI
    from product.decision import Decision

    frame = _frame(n=80)
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: frame)
    decision = Decision(symbol="TCS", state="WATCH", setup="VCP",
                        generated_at="2026-09-20T16:00:00+05:30",
                        technical_evidence={"rsi": 55.0, "volume_ratio": 1.4})
    live = EI.decision_features(decision)
    learning, prov = EI.learning_features(decision)
    assert live["atr_pct"] is None              # ranking inputs unchanged
    assert learning["atr_pct"] == pytest.approx(2.0)
    assert set(prov["filled"]) >= {"atr_pct", "adr_pct", "clv", "dist_20ema_pct"}
    assert learning["rsi"] == live["rsi"] == 55.0
