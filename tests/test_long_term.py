"""
Long-Term Picks — the screener + the tracker/revision loop.

The screener must only bless real long-term uptrends; the tracker must dedupe
calls and REVISE (exit) a pick when its thesis breaks. Network-free.
"""
import numpy as np
import pandas as pd
import pytest

from scan import long_term as LT
from core import long_term_tracker as TR


def _series(closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    c = np.asarray(closes, float)
    return pd.DataFrame({"close": c, "high": c * 1.01, "low": c * 0.99,
                         "volume": [800000] * n}, index=idx)


class TestScreener:
    def test_steady_uptrend_is_a_long_term_buy(self):
        df = _series(np.linspace(100, 210, 300)
                     + np.random.default_rng(0).normal(0, 1.0, 300))
        s = LT.long_term_score(df)
        assert s["verdict"] == "LONG_TERM_BUY"
        assert s["above_200dma"] and s["dma200_rising"]
        assert s["mom_12m_pct"] > 0

    def test_downtrend_is_skipped(self):
        df = _series(np.linspace(200, 120, 300))
        assert LT.long_term_score(df)["verdict"] == "SKIP"
        assert LT.long_term_score(df)["above_200dma"] is False

    def test_thin_history_is_skipped(self):
        assert LT.long_term_score(_series(np.linspace(100, 120, 80)))["verdict"] == "SKIP"

    def test_illiquid_is_skipped(self):
        df = _series(np.linspace(100, 210, 300))
        df["volume"] = 1                              # ~no turnover
        assert LT.long_term_score(df)["verdict"] == "SKIP"

    def test_scan_ranks_and_filters(self, monkeypatch):
        up = _series(np.linspace(100, 220, 300))
        mid = _series(np.linspace(100, 150, 300))
        dn = _series(np.linspace(200, 120, 300))
        data = {"WINNER": up, "OKAY": mid, "LOSER": dn}
        monkeypatch.setattr("data.bhavcopy_store.get_ohlcv",
                            lambda s: data.get(s))
        monkeypatch.setattr("data.bhavcopy_store.store_symbols",
                            lambda: list(data))
        picks = LT.scan_long_term()
        syms = [p["symbol"] for p in picks]
        assert "WINNER" in syms and "LOSER" not in syms
        assert picks == sorted(picks, key=lambda p: -p["score"])   # ranked
        assert all("thesis" in p for p in picks)


class TestRevision:
    def test_pure_revision_logic(self):
        assert LT and TR
        # fell below 200-DMA → exit
        ex, why = TR._revision({"verdict": "SKIP", "above_200dma": False})
        assert ex and "200-day" in why
        # trend flat + momentum negative → exit
        ex2, _ = TR._revision({"verdict": "WATCH", "dma200_rising": False,
                               "mom_12m_pct": -5, "score": 50})
        assert ex2
        # still healthy → hold
        hold, _ = TR._revision({"verdict": "LONG_TERM_BUY", "above_200dma": True,
                                "dma200_rising": True, "mom_12m_pct": 20, "score": 80})
        assert hold is False

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TR, "_DB_PATH", tmp_path / "lt.db")

    def test_record_dedupes_and_review_revises(self, monkeypatch):
        added = TR.record_picks([
            {"symbol": "ABC", "price": 100.0, "score": 80, "thesis": "uptrend",
             "factors": ["above 200-DMA"]},
            {"symbol": "XYZ", "price": 50.0, "score": 70, "thesis": "uptrend",
             "factors": ["above 200-DMA"]}])
        assert {p["symbol"] for p in added} == {"ABC", "XYZ"}
        # re-recording ABC does nothing (no duplicate active call)
        assert TR.record_picks([{"symbol": "ABC", "price": 110, "score": 82}]) == []
        assert len(TR.active_picks()) == 2

        # review with REAL data: ABC now a downtrend ending at 90 (entry 100 →
        # −10%, below 200-DMA → thesis broken); XYZ a healthy uptrend → holds.
        data = {
            "ABC": _series(np.linspace(120, 90, 300)),    # ends 90, below 200-DMA
            "XYZ": _series(np.linspace(35, 55, 300))}     # steady uptrend, holds
        monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: data.get(s))
        revs = TR.review_picks()
        rev_syms = {r["symbol"] for r in revs}
        assert "ABC" in rev_syms and "XYZ" not in rev_syms      # only ABC revised
        abc = next(r for r in revs if r["symbol"] == "ABC")
        assert abs(abc["return_pct"] - (-10.0)) < 1e-6         # ₹100 → ₹90
        assert "200-day" in abc["reason"]
        assert {p["symbol"] for p in TR.active_picks()} == {"XYZ"}  # ABC exited
