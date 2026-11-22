"""Market-wide rs_cs_v1 percentile in the learning record (product.pit_features, producer pit_v3_rs)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from product import pit_features as PF


def _market(n_names=320, n_bars=300, start="2025-01-01", seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_bars)
    frames = {}
    for i in range(n_names):
        closes = np.round(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_bars))), 2)
        frames[f"S{i:03d}"] = pd.DataFrame({"close": closes}, index=idx)
    return idx, frames


def _rs(sym, frames, idx, session, archive=None):
    archive = list(idx.date) if archive is None else archive
    return PF.rs_percentile_on(sym, frames.get(sym), session, symbols_fn=lambda: list(frames),
                               ohlcv_fn=frames.get, archive_fn=lambda: archive)


@pytest.fixture(autouse=True)
def _fresh_cache():
    PF.reset_breadth_cache()
    yield
    PF.reset_breadth_cache()


def test_equals_research_sepa_build_rs_table_when_the_universes_coincide():
    from research.sepa.config import DEFAULT_CONFIG
    from research.sepa.rs import build_rs_table

    idx, frames = _market()
    session = idx[-1].date()
    table = build_rs_table(frames, session, DEFAULT_CONFIG)
    assert table["n_ranked"] == len(frames)
    for sym in list(frames)[:40]:
        assert _rs(sym, frames, idx, session) == pytest.approx(table["percentiles"][sym], abs=1e-6)
    mid = idx[280].date()
    table = build_rs_table(frames, mid, DEFAULT_CONFIG)
    for sym in list(frames)[:10]:
        assert _rs(sym, frames, idx, mid) == pytest.approx(table["percentiles"][sym], abs=1e-6)


def test_future_bars_cannot_move_a_percentile():
    idx, frames = _market()
    session = idx[280].date()
    before = {s: _rs(s, frames, idx, session) for s in list(frames)[:15]}
    spiked = {k: v.copy() for k, v in frames.items()}
    for v in spiked.values():
        v.iloc[281:, 0] = v.iloc[281:, 0] * 50.0
    PF.reset_breadth_cache()
    assert {s: _rs(s, spiked, idx, session) for s in before} == before


def test_needs_a_bar_on_the_session_253_bars_a_stock_symbol_and_300_ranked_names():
    idx, frames = _market()
    session = idx[-1].date()
    frames["GAP"] = frames["S000"].iloc[:-1].copy()           # no bar on the session
    assert _rs("GAP", frames, idx, session) is None
    frames["YOUNG"] = frames["S001"].iloc[-200:].copy()       # 200 bars < 253
    assert _rs("YOUNG", frames, idx, session) is None
    frames["GOLDBEES"] = frames["S002"].copy()                # a marked fund
    assert _rs("GOLDBEES", frames, idx, session) is None
    assert _rs("S003", frames, idx, idx[250].date()) is None  # nobody has 253 bars there
    small = dict(list(frames.items())[:250])
    PF.reset_breadth_cache()
    assert _rs("S004", small, idx, session) is None


def test_horizons_count_verified_bars_only():
    idx, frames = _market()
    session = idx[-1].date()
    archive = [d for d in idx.date if d != idx[100].date()]   # one archive never verified
    spiked = {k: v.copy() for k, v in frames.items()}
    for v in spiked.values():
        v.iloc[100, 0] = 1e9                                    # that row is never an input
    from research.sepa.config import DEFAULT_CONFIG
    from research.sepa.rs import build_rs_table
    expected = build_rs_table({k: v.drop(v.index[100]) for k, v in spiked.items()}, session, DEFAULT_CONFIG)
    for sym in list(frames)[:10]:
        assert _rs(sym, spiked, idx, session, archive=archive) == pytest.approx(expected["percentiles"][sym], abs=1e-6)


def test_a_non_finite_composite_is_missing_not_ranked_at_zero():
    idx, frames = _market()
    session = idx[-1].date()
    bad = frames["S005"].copy()
    bad.iloc[-64, 0] = np.nan                                    # the r63 start bar
    frames["S005"] = bad
    assert _rs("S005", frames, idx, session) is None
    assert _rs("S006", frames, idx, session) is not None


def test_the_producer_pins_who_gets_rs(monkeypatch):
    idx, frames = _market()
    session = idx[-1].date()
    monkeypatch.setattr(PF, "_archive_dates", lambda: list(idx.date))
    monkeypatch.setattr(PF, "_store_symbols", lambda: list(frames))
    monkeypatch.setattr(PF, "_store_ohlcv", lambda s: frames.get(s))
    ctx = {"vix": 13.0, "index_trend": "UP", "breadth_pct_above_50dma": 50.0, "_session": session.isoformat()}
    at = f"{session.isoformat()}T19:00:00+05:30"
    v3, prov = PF.fill_missing({"rs_percentile": None}, "S007", at, context=ctx)
    assert v3["rs_percentile"] == pytest.approx(_rs("S007", frames, idx, session))
    assert "rs_percentile" in prov["filled"] and prov["producer"] == "pit_v3_rs"
    v2, prov = PF.fill_missing({"rs_percentile": None}, "S007", at, context=ctx, producer=PF.CONTEXT_PRODUCER)
    assert v2["rs_percentile"] is None and "rs_percentile" not in prov["filled"]
    carried, prov = PF.fill_missing({"rs_percentile": 12.5}, "S007", at, context=ctx)
    assert carried["rs_percentile"] == 12.5


def test_the_live_analog_query_never_sees_a_filled_rs_percentile():
    from product import evidence_intelligence as EI

    row = {"features": {"rs_percentile": 91.0, "rsi": 55.0}, "validation": [],
           "meta": {"pit_features": {"filled": ["rs_percentile"]}}}
    feats, problems = EI._live_projection(row)
    assert feats["rs_percentile"] is None and feats["rsi"] == 55.0 and problems == 1
