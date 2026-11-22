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

import numpy as np  # noqa: E402


def _index(n=260, start="2025-09-01", close=100.0, step=0.0):
    idx = pd.bdate_range(start, periods=n)
    closes = [close + step * i for i in range(n)]
    return pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
                         "Close": closes, "Volume": 0.0}, index=idx)


@pytest.mark.parametrize("decided_at,expected", [
    ("2026-09-04T11:00:00+05:30", "2026-09-03"),   # intraday: yesterday's archive
    ("2026-09-04T15:31:00+05:30", "2026-09-03"),   # closed, not yet published
    ("2026-09-04T17:59:00+05:30", "2026-09-03"),
    ("2026-09-04T18:00:00+05:30", "2026-09-04"),   # the publication convention
    ("2026-09-04T23:00:00+05:30", "2026-09-04"),
    ("2026-09-07T10:00:00+05:30", "2026-09-04"),   # Monday morning: Friday
    ("2026-09-08T10:00:00+05:30", "2026-09-04"),   # Monday's archive never landed: Friday, not a guess
    ("", None),
])
def test_the_source_session_is_the_latest_archive_published_by_the_decision(decided_at, expected):
    archive = [_date(2026, 9, 2), _date(2026, 9, 3), _date(2026, 9, 4)]
    got = PF.published_session(decided_at, archive_fn=lambda: archive)
    assert (got.isoformat() if got else None) == expected


def test_delivery_is_read_on_the_source_session_only():
    frame = _frame(n=80)
    frame["deliv_per"] = 40.0
    day = frame.index[60].date()
    frame.loc[frame.index[60], "deliv_per"] = 63.5
    frame.iloc[61:, frame.columns.get_loc("deliv_per")] = 99.0      # later sessions
    assert PF.delivery_on(frame, day) == pytest.approx(63.5)
    assert PF.delivery_on(frame, day + timedelta(days=1000)) is None  # no bar that session
    short = frame.iloc[55:61]                                         # 6 bars still has the value
    assert PF.delivery_on(short, day) == pytest.approx(63.5)


def test_delivery_is_never_read_from_a_live_overlay_row():
    frame = _frame(n=80)
    frame["deliv_per"] = 40.0
    day = frame.index[-1].date()
    frame.attrs["quantterm_live_overlay_date"] = day.isoformat()
    assert PF.delivery_on(frame, day) is None
    assert PF.delivery_on(frame, frame.index[-2].date()) == pytest.approx(40.0)


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 100.5])
def test_delivery_outside_its_physical_range_is_missing_not_clipped(bad):
    frame = _frame(n=80)
    frame["deliv_per"] = bad
    assert PF.delivery_on(frame, frame.index[-1].date()) is None


def test_vix_is_the_official_close_dated_exactly_the_source_session():
    vix = _index(n=40, start="2026-08-03", close=14.0)
    vix.loc[vix.index > "2026-09-04", "Close"] = 99.0                  # rows after the session
    calls = []

    def loader(ticker, as_of):
        calls.append((ticker, as_of))
        return vix if ticker == "^INDIAVIX" else None                   # deliberately unsliced
    got = PF.market_context(_date(2026, 9, 4), index_fn=loader)
    assert got == {"vix": pytest.approx(14.0)}
    assert ("^INDIAVIX", "2026-09-04") in calls
    # The index store lags the bhavcopy by a session: that is stale, not "close enough".
    assert PF.market_context(_date(2026, 9, 7), index_fn=lambda t, d: vix.loc[:"2026-09-04"]) == {}


@pytest.mark.parametrize("step,expected", [(1.0, "UP"), (-1.0, "DOWN"), (0.0, "FLAT")])
def test_index_trend_uses_the_point_in_time_session_classifier(step, expected):
    nifty = _index(n=260, start="2025-09-01", close=1000.0, step=step)
    session = nifty.index[-1].date()
    assert PF.market_context(session, index_fn=lambda t, d: nifty if t == "^NSEI" else None)["index_trend"] == expected


def test_index_trend_counts_distinct_sessions_and_needs_the_session_row():
    nifty = _index(n=199, start="2025-09-01", close=1000.0, step=1.0)
    padded = pd.concat([nifty, nifty.iloc[[-1]]])                     # 199 sessions + a duplicate
    session = nifty.index[-1].date()
    assert "index_trend" not in PF.market_context(session, index_fn=lambda t, d: padded if t == "^NSEI" else None)
    full = _index(n=260, start="2025-01-01", close=1000.0, step=1.0)
    shuffled = full.sample(frac=1.0, random_state=3)                  # unsorted input
    last = full.index[-1].date()
    assert PF.market_context(last, index_fn=lambda t, d: shuffled if t == "^NSEI" else None)["index_trend"] == "UP"
    assert "index_trend" not in PF.market_context(last + timedelta(days=3),
                                                  index_fn=lambda t, d: full if t == "^NSEI" else None)


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


def _breadth(session, frames, archive):
    PF.reset_breadth_cache()
    return PF.breadth_pct_above_50dma(session, symbols_fn=lambda: list(frames), ohlcv_fn=frames.get,
                                      archive_fn=lambda: [d.date() for d in archive])


def test_breadth_matches_scan_breadth_on_the_session():
    from scan.breadth import breadth_from_results

    idx, frames = _universe(240, 160)
    rows = []
    for df in frames.values():
        c = df["close"].to_numpy()
        rows.append({"change_pct": (c[-1] / c[-2] - 1) * 100, "above_sma50": bool(c[-1] > c[-50:].mean())})
    assert _breadth(idx[-1].date(), frames, idx) == pytest.approx(60.0)
    assert round(_breadth(idx[-1].date(), frames, idx), 1) == breadth_from_results(rows)["pct_above_50"]


def test_breadth_is_measured_only_on_verified_sessions_from_verified_bars():
    idx, frames = _universe(240, 160)
    assert _breadth(idx[-2].date(), frames, idx) == pytest.approx(0.0)      # everyone flat then
    assert _breadth(idx[-1].date(), frames, idx[:-1]) is None              # its archive is not verified
    # An archive missing in the middle: that row is not an input to anyone's mean.
    gap = idx.delete(70)
    spiked = {k: v.copy() for k, v in frames.items()}
    for v in spiked.values():
        v.iloc[70, 0] = 1e6
    assert _breadth(idx[-2].date(), spiked, gap) == pytest.approx(0.0)


def test_a_flat_series_is_a_tie_even_after_a_later_split_adjustment():
    idx = pd.bdate_range("2019-12-02", periods=1765)
    flat = pd.DataFrame({"close": 101.0}, index=idx)
    adjusted = pd.DataFrame({"close": 101.0 / 3.0}, index=idx)
    frames = {f"S{i:03d}": (adjusted if i % 2 else flat) for i in range(300)}
    assert _breadth(idx[-1].date(), frames, idx) == 0.0


def test_breadth_needs_300_stocks_and_excludes_marked_funds():
    idx, frames = _universe(150, 140)                     # 290 names
    assert _breadth(idx[-1].date(), frames, idx) is None
    funds = {name: pd.DataFrame({"close": [100.0] * 79 + [110.0]}, index=idx)
             for name in [f"GOLD{i}BEES" for i in range(10)] + [f"LIQUID{i}" for i in range(5)]
             + [f"NIFTY{i}ETF" for i in range(5)]}
    idx, frames = _universe(150, 140, extra=funds)        # 310 symbols, 290 stocks
    assert _breadth(idx[-1].date(), frames, idx) is None
    idx, frames = _universe(160, 140, extra=funds)        # 300 stocks
    assert _breadth(idx[-1].date(), frames, idx) == pytest.approx(160 / 300 * 100)


def test_breadth_needs_51_official_bars_per_name():
    idx, frames = _universe(300, 0, n_bars=51)
    assert _breadth(idx[-1].date(), frames, idx) == pytest.approx(100.0)
    assert _breadth(idx[-2].date(), frames, idx) is None   # only 50 bars there: nobody measured


def test_breadth_cache_follows_the_archive_and_the_store_generation(monkeypatch):
    idx, frames = _universe(240, 160)
    calls = []

    def ohlcv(sym):
        calls.append(sym)
        return frames.get(sym)
    archive = [d.date() for d in idx[:-1]]
    PF.reset_breadth_cache()
    kw = dict(symbols_fn=lambda: list(frames), ohlcv_fn=ohlcv, archive_fn=lambda: archive)
    assert PF.breadth_pct_above_50dma(idx[-2].date(), **kw) == pytest.approx(0.0)
    first = len(calls)
    assert PF.breadth_pct_above_50dma(idx[-3].date(), **kw) == pytest.approx(0.0)
    assert len(calls) == first                             # one pass per generation
    archive.append(idx[-1].date())                         # the session's archive lands
    assert PF.breadth_pct_above_50dma(idx[-1].date(), **kw) == pytest.approx(60.0)
    second = len(calls)
    generation = ["g1"]
    monkeypatch.setattr(PF, "_store_generation", lambda: tuple(generation))
    PF.breadth_pct_above_50dma(idx[-1].date(), **kw)
    third = len(calls)
    assert third > second                                  # new key: recomputed once
    PF.breadth_pct_above_50dma(idx[-1].date(), **kw)
    assert len(calls) == third
    generation[0] = "g2"                                   # the store installed new bars
    PF.breadth_pct_above_50dma(idx[-1].date(), **kw)
    assert len(calls) > third


def test_breadth_equals_the_scan_definition_on_every_session_of_a_random_market():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-01", periods=140)
    frames = {}
    for i in range(330):
        start = int(rng.integers(0, 30))
        closes = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx) - start))), 2)
        if i % 37 == 0:
            closes[int(rng.integers(0, len(closes)))] = np.nan     # a bad print
        if i % 41 == 0:
            closes[:] = 250.0                                       # never trades away from 250
        frames[f"S{i:03d}"] = pd.DataFrame({"close": closes}, index=idx[start:])
    checked = 0
    for k in (60, 80, 100, 120, 139):
        cut = idx[k]
        ups = n = 0
        for df in frames.values():
            hist = df.loc[:cut, "close"].to_numpy(dtype=float)
            if len(hist) < 51 or df.index[min(len(hist), len(df)) - 1] != cut:
                continue
            n += 1
            ups += bool(hist[-1] > hist[-50:].mean())               # scan.breadth: NaN -> not above
        got = _breadth(cut.date(), frames, idx)
        if n < 300:
            assert got is None
        else:
            checked += 1
            assert got == pytest.approx(ups / n * 100)
    assert checked >= 2


def test_fill_missing_treats_unknown_trend_as_missing_and_never_touches_regime():
    frame = _frame(n=80, start="2026-09-01")
    frame["deliv_per"] = 44.0
    session = frame.index[-1].date().isoformat()
    ctx = {"vix": 13.2, "index_trend": "UP", "breadth_pct_above_50dma": 55.0, "_session": session}
    merged, prov = PF.fill_missing(
        {"index_trend": "UNKNOWN", "regime": "UNKNOWN", "vix": None, "breadth_pct_above_50dma": 71.0},
        "X", "2026-12-31T16:00:00+05:30", frame=frame, context=ctx)
    assert merged["index_trend"] == "UP" and merged["vix"] == 13.2
    assert merged["breadth_pct_above_50dma"] == 71.0       # carried value wins
    assert merged["regime"] == "UNKNOWN"                   # live analogs weight stored regime
    assert merged["delivery_pct"] == pytest.approx(44.0)   # read on the source session
    assert set(prov["filled"]) >= {"index_trend", "vix", "delivery_pct"}
    assert "breadth_pct_above_50dma" not in prov["filled"] and "regime" not in prov["filled"]
    assert prov["context_session"] == session and prov["producer"] == PF.PRODUCER_VERSION


def test_a_legacy_producer_fills_only_the_technical_features_and_an_unknown_one_nothing():
    ctx = {"vix": 13.2, "index_trend": "UP", "breadth_pct_above_50dma": 55.0, "_session": "2026-12-30"}
    legacy, prov = PF.fill_missing({"vix": None, "atr_pct": None}, "X", "2026-12-31T16:00:00+05:30",
                                   frame=_frame(), context=ctx, producer=PF.LEGACY_PRODUCER)
    assert legacy["vix"] is None and legacy["atr_pct"] == pytest.approx(2.0)
    assert set(prov["filled"]) <= set(PF.PRODUCED)
    unknown, prov = PF.fill_missing({"vix": None, "atr_pct": None}, "X", "2026-12-31T16:00:00+05:30",
                                    frame=_frame(), context=ctx, producer="pit_v9")
    assert unknown == {"vix": None, "atr_pct": None} and prov["error"] == "UNKNOWN_PRODUCER"


def test_learning_record_gets_market_context_but_the_live_query_does_not(monkeypatch):
    from product import evidence_intelligence as EI
    from product.decision import Decision

    frame = _frame(n=80, start="2026-06-01")
    frame["deliv_per"] = 47.0
    session = frame.index[-1].date()
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: frame)
    monkeypatch.setattr(PF, "context_features", lambda decided_at: {
        "vix": 12.5, "index_trend": "DOWN", "breadth_pct_above_50dma": 38.0, "_session": session.isoformat()})
    decision = Decision(symbol="TCS", state="WATCH", setup="VCP",
                        generated_at=f"{session.isoformat()}T19:00:00+05:30", technical_evidence={"rsi": 55.0})
    live = EI.decision_features(decision)
    learning, prov = EI.learning_features(decision)
    for name in ("vix", "breadth_pct_above_50dma", "delivery_pct"):
        assert live[name] is None
    assert live["index_trend"] == "UNKNOWN"
    assert learning["vix"] == 12.5 and learning["index_trend"] == "DOWN"
    assert learning["breadth_pct_above_50dma"] == 38.0 and learning["delivery_pct"] == 47.0
    assert learning["regime"] == live["regime"]
    assert prov["context_session"] == session.isoformat()


def test_only_a_complete_context_is_cached_and_only_for_its_generation(monkeypatch):
    PF.reset_breadth_cache()
    archive = [_date(2026, 9, 3), _date(2026, 9, 4)]
    monkeypatch.setattr(PF, "_archive_dates", lambda: archive)
    calls = []
    state = {"vix": None, "gen": "g1"}
    monkeypatch.setattr(PF, "_store_generation", lambda: (state["gen"],))
    monkeypatch.setattr(PF, "market_context", lambda s, index_fn=None: calls.append(s) or (
        {"vix": state["vix"], "index_trend": "UP"} if state["vix"] else {"index_trend": "UP"}))
    monkeypatch.setattr(PF, "breadth_pct_above_50dma", lambda s, **kw: 52.0)
    at = "2026-09-04T19:00:00+05:30"
    for _ in range(3):
        assert "vix" not in PF.context_features(at)          # still loading: never cached
    assert len(calls) == 3
    state["vix"] = 11.0
    for _ in range(50):
        got = PF.context_features(at)
    assert got == {"vix": 11.0, "index_trend": "UP", "breadth_pct_above_50dma": 52.0, "_session": "2026-09-04"}
    assert len(calls) == 4                                   # one board, one read
    assert PF.context_features("2026-09-04T16:00:00+05:30")["_session"] == "2026-09-03"
    state["gen"] = "g2"                                      # the store moved on
    PF.context_features(at)
    assert len(calls) == 6
    PF.reset_breadth_cache()
