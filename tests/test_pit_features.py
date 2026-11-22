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


# ── market context (delivery, VIX, index trend, breadth) ─────────────────────

from datetime import date as _date, timedelta  # noqa: E402


def _index(n=260, start="2025-09-01", close=100.0, step=0.0):
    idx = pd.bdate_range(start, periods=n)
    closes = [close + step * i for i in range(n)]
    return pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
                         "Close": closes, "Volume": 0.0}, index=idx)


def test_delivery_pct_comes_from_the_cutoff_bar_and_never_from_a_later_one():
    frame = _frame(n=80)
    frame["deliv_per"] = 40.0
    day = frame.index[60]
    frame.loc[day, "deliv_per"] = 63.5
    frame.iloc[61:, frame.columns.get_loc("deliv_per")] = 99.0      # future
    after = PF.technical_features("X", f"{day.date().isoformat()}T16:00:00+05:30", frame=frame)
    intraday = PF.technical_features("X", f"{day.date().isoformat()}T11:00:00+05:30", frame=frame)
    assert after["delivery_pct"] == pytest.approx(63.5)
    assert intraday["delivery_pct"] == pytest.approx(40.0)        # the previous session's bar


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 100.5])
def test_delivery_pct_outside_its_physical_range_is_missing_not_clipped(bad):
    frame = _frame(n=80)
    frame["deliv_per"] = bad
    assert "delivery_pct" not in PF.technical_features("X", "2026-12-31T16:00:00+05:30", frame=frame)


def test_vix_is_the_official_close_at_the_cutoff_and_ignores_rows_a_loader_failed_to_slice():
    vix = _index(n=40, start="2026-08-03", close=14.0)
    vix.loc[vix.index > "2026-09-04", "Close"] = 99.0                  # rows after the cutoff
    calls = []

    def loader(ticker, as_of):
        calls.append((ticker, as_of))
        return vix if ticker == "^INDIAVIX" else None                   # deliberately unsliced
    got = PF.market_context(_date(2026, 9, 4), index_fn=loader)
    assert got["vix"] == pytest.approx(14.0)
    assert ("^INDIAVIX", "2026-09-04") in calls
    assert "index_trend" not in got                                    # no Nifty: no trend


def test_stale_vix_is_missing():
    vix = _index(n=40, start="2026-06-01", close=14.0)                 # ends late July
    got = PF.market_context(_date(2026, 9, 4), index_fn=lambda t, d: vix if t == "^INDIAVIX" else None)
    assert "vix" not in got


@pytest.mark.parametrize("step,expected", [(1.0, "UP"), (-1.0, "DOWN"), (0.0, "FLAT")])
def test_index_trend_uses_the_point_in_time_session_classifier(step, expected):
    nifty = _index(n=260, start="2025-09-01", close=1000.0, step=step)
    cut = nifty.index[-1].date()
    got = PF.market_context(cut, index_fn=lambda t, d: nifty if t == "^NSEI" else None)
    assert got["index_trend"] == expected


def test_index_trend_needs_200_rows_and_a_current_row():
    short = _index(n=150, start="2026-01-01", close=1000.0, step=1.0)
    assert "index_trend" not in PF.market_context(short.index[-1].date(),
                                                  index_fn=lambda t, d: short if t == "^NSEI" else None)
    nifty = _index(n=260, start="2025-01-01", close=1000.0, step=1.0)
    late = nifty.index[-1].date() + timedelta(days=10)
    assert "index_trend" not in PF.market_context(late, index_fn=lambda t, d: nifty if t == "^NSEI" else None)


def _universe(n_up, n_down, *, n_bars=80, start="2026-06-01", extra=None):
    """n_up names end above their 50-DMA, n_down below; plus optional extras."""
    idx = pd.bdate_range(start, periods=n_bars)
    frames = {}
    for i in range(n_up):
        frames[f"UP{i:03d}"] = pd.DataFrame({"close": [100.0] * (n_bars - 1) + [110.0]}, index=idx)
    for i in range(n_down):
        frames[f"DN{i:03d}"] = pd.DataFrame({"close": [100.0] * (n_bars - 1) + [90.0]}, index=idx)
    frames.update(extra or {})
    return idx, frames


def _breadth(cut, frames, official):
    PF.reset_breadth_cache()
    return PF.breadth_pct_above_50dma(
        cut, symbols_fn=lambda: list(frames), ohlcv_fn=frames.get,
        official_fn=lambda: [d.date().isoformat() for d in official])


def test_breadth_matches_scan_breadth_on_the_cutoff_session():
    from scan.breadth import breadth_from_results

    idx, frames = _universe(240, 160)
    rows = []
    for df in frames.values():
        c = df["close"].to_numpy()
        rows.append({"change_pct": (c[-1] / c[-2] - 1) * 100, "above_sma50": bool(c[-1] > c[-50:].mean())})
    expected = breadth_from_results(rows)["pct_above_50"]
    got = _breadth(idx[-1].date(), frames, idx)
    assert got == pytest.approx(60.0)
    assert round(got, 1) == expected


def test_breadth_is_point_in_time_and_official_only():
    idx, frames = _universe(240, 160)
    day = idx[-2].date()                                  # everyone flat at 100 on that session
    assert _breadth(day, frames, idx) == pytest.approx(0.0)
    # The last session is not yet a verified official archive (e.g. a live
    # intraday overlay): it is never measured, and a cutoff on it falls back
    # to the previous official session.
    assert _breadth(idx[-1].date(), frames, idx[:-1]) == pytest.approx(0.0)
    # A cutoff more than three days after the last official session is stale.
    assert _breadth(idx[-1].date() + timedelta(days=5), frames, idx) is None


def test_breadth_needs_300_stocks_and_excludes_funds():
    idx, frames = _universe(150, 140)                     # 290 names
    assert _breadth(idx[-1].date(), frames, idx) is None
    funds = {name: pd.DataFrame({"close": [100.0] * 79 + [110.0]}, index=idx)
             for name in [f"GOLD{i}BEES" for i in range(10)] + [f"LIQUID{i}" for i in range(5)]
             + [f"NIFTY{i}ETF" for i in range(5)]}
    idx, frames = _universe(150, 140, extra=funds)        # 310 symbols, 290 stocks
    assert _breadth(idx[-1].date(), frames, idx) is None
    idx, frames = _universe(160, 140, extra=funds)        # 300 stocks
    assert _breadth(idx[-1].date(), frames, idx) == pytest.approx(160 / 300 * 100)


def test_breadth_needs_51_bars_per_name():
    idx, frames = _universe(300, 0, n_bars=51)
    assert _breadth(idx[-1].date(), frames, idx) == pytest.approx(100.0)
    assert _breadth(idx[-2].date(), frames, idx) is None   # only 50 bars there: nobody measured


def test_breadth_cache_follows_the_official_archive():
    idx, frames = _universe(240, 160)
    calls = []

    def ohlcv(sym):
        calls.append(sym)
        return frames.get(sym)
    official = [d.date().isoformat() for d in idx[:-1]]
    PF.reset_breadth_cache()
    kw = dict(symbols_fn=lambda: list(frames), ohlcv_fn=ohlcv, official_fn=lambda: official)
    assert PF.breadth_pct_above_50dma(idx[-1].date(), **kw) == pytest.approx(0.0)
    first = len(calls)
    assert PF.breadth_pct_above_50dma(idx[-2].date(), **kw) == pytest.approx(0.0)
    assert len(calls) == first                             # one pass per archive state
    official.append(idx[-1].date().isoformat())            # the session's archive lands
    assert PF.breadth_pct_above_50dma(idx[-1].date(), **kw) == pytest.approx(60.0)


def test_fill_missing_treats_unknown_trend_as_missing_and_never_touches_regime():
    ctx = {"vix": 13.2, "index_trend": "UP", "breadth_pct_above_50dma": 55.0}
    merged, prov = PF.fill_missing(
        {"index_trend": "UNKNOWN", "regime": "UNKNOWN", "vix": None, "breadth_pct_above_50dma": 71.0},
        "X", "2026-12-31T16:00:00+05:30", frame=_frame(), context=ctx)
    assert merged["index_trend"] == "UP" and merged["vix"] == 13.2
    assert merged["breadth_pct_above_50dma"] == 71.0       # carried value wins
    assert merged["regime"] == "UNKNOWN"                   # live analogs weight stored regime
    assert set(prov["filled"]) >= {"index_trend", "vix"}
    assert "breadth_pct_above_50dma" not in prov["filled"] and "regime" not in prov["filled"]
    assert prov["context_source"]["vix"] == "official_index_pit"


def test_learning_record_gets_market_context_but_the_live_query_does_not(monkeypatch):
    from product import evidence_intelligence as EI
    from product.decision import Decision

    frame = _frame(n=80)
    frame["deliv_per"] = 47.0
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: frame)
    monkeypatch.setattr(PF, "context_features",
                        lambda decided_at: {"vix": 12.5, "index_trend": "DOWN", "breadth_pct_above_50dma": 38.0})
    decision = Decision(symbol="TCS", state="WATCH", setup="VCP", generated_at="2026-09-20T16:00:00+05:30",
                        technical_evidence={"rsi": 55.0})
    live = EI.decision_features(decision)
    learning, prov = EI.learning_features(decision)
    for name in ("vix", "breadth_pct_above_50dma", "delivery_pct"):
        assert live[name] is None
    assert live["index_trend"] == "UNKNOWN"
    assert learning["vix"] == 12.5 and learning["index_trend"] == "DOWN"
    assert learning["breadth_pct_above_50dma"] == 38.0 and learning["delivery_pct"] == 47.0
    assert learning["regime"] == live["regime"]


def test_breadth_equals_the_scan_definition_on_every_session_of_a_random_market():
    import numpy as np

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-01", periods=140)
    frames = {}
    for i in range(330):
        start = int(rng.integers(0, 60))
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx) - start)))
        if i % 37 == 0:
            closes[int(rng.integers(0, len(closes)))] = np.nan     # a bad print
        frames[f"S{i:03d}"] = pd.DataFrame({"close": closes}, index=idx[start:])
    official = idx[::1]
    for k in (50, 75, 100, 139):
        cut = idx[k]
        rows = []
        for df in frames.values():
            hist = df.loc[:cut, "close"].astype(float)
            if len(hist) < 51 or hist.index[-1] != cut:
                continue
            sma = hist.rolling(50, min_periods=50).mean()
            if not (np.isfinite(hist.iloc[-1]) and np.isfinite(sma.iloc[-1])):
                continue
            rows.append(bool(hist.iloc[-1] > sma.iloc[-1]))
        got = _breadth(cut.date(), frames, official)
        if len(rows) < 300:
            assert got is None
        else:
            assert got == pytest.approx(sum(rows) / len(rows) * 100)


def test_context_is_read_once_per_cutoff_for_a_whole_board(monkeypatch):
    PF.reset_breadth_cache()
    calls = []
    monkeypatch.setattr(PF, "market_context", lambda cut, index_fn=None: calls.append(cut) or {"vix": 11.0})
    monkeypatch.setattr(PF, "breadth_pct_above_50dma", lambda cut, **kw: 52.0)
    for _ in range(50):
        got = PF.context_features("2026-09-04T16:00:00+05:30")
    assert got == {"vix": 11.0, "breadth_pct_above_50dma": 52.0}
    assert len(calls) == 1
    PF.context_features("2026-09-07T16:00:00+05:30")
    assert len(calls) == 2
    PF.reset_breadth_cache()
