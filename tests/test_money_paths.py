"""
Money-critical path tests — the code that sizes, validates, and places
trades, and the data plumbing it depends on. These MUST stay green:
a regression here loses real money, not style points.

All tests are network-free (stubs/fixtures only) so they run anywhere.
"""
import sys
import json
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Position sizer — the 1% rule
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionSizer:
    def test_one_percent_rule_with_concentration_cap(self):
        from risk.position_sizer import size_position
        # risk budget 1000/44 → 22 sh, but 22×794 > 10% cap → 12 sh
        ps = size_position(794, 750, capital=100_000, risk_pct=0.01)
        assert ps["qty"] == 12 and ps["capped"]
        assert ps["invested"] <= 10_000

    def test_wide_stop_no_cap(self):
        from risk.position_sizer import size_position
        ps = size_position(100, 90, capital=100_000, risk_pct=0.01)
        assert ps["qty"] == 100 and not ps["capped"]
        assert ps["max_loss"] == 1000  # exactly 1% of capital

    def test_invalid_stop_gives_zero(self):
        from risk.position_sizer import size_position
        assert size_position(100, 105)["qty"] == 0     # stop above entry
        assert size_position(100, 100)["qty"] == 0     # stop == entry
        assert size_position(0, -5)["qty"] == 0        # nonsense prices

    def test_unaffordable_stock_gives_zero(self):
        from risk.position_sizer import size_position
        assert size_position(80_000, 70_000, capital=100_000)["qty"] == 0

    def test_max_loss_never_exceeds_risk_budget(self):
        from risk.position_sizer import size_position
        for entry, stop in [(500, 480), (1500, 1400), (90, 85), (2400, 2300)]:
            ps = size_position(entry, stop, capital=200_000, risk_pct=0.01)
            if ps["qty"]:
                assert ps["max_loss"] <= 200_000 * 0.01 + entry  # +1 share tolerance


# ══════════════════════════════════════════════════════════════════════════════
# 2. Trade executor — validation rails + paper journal
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeExecutor:
    def test_validation_rails(self):
        from execution.trade_executor import _validate
        assert _validate("HAL", 0, 4500, 4300, 4900)          # zero qty
        assert _validate("HAL", 10, 4500, 4600, 4900)         # stop above entry
        assert _validate("HAL", 10, 4500, 4300, 4400)         # target below entry
        assert _validate("HAL", 10, 4500, 3000, 4900)         # stop >20% away
        assert _validate("", 10, 4500, 4300, 4900)            # no symbol
        assert _validate("HAL", 10, 4500, 4300, 4900) is None  # valid passes

    def test_paper_mode_journals_and_never_places(self, tmp_path, monkeypatch):
        import execution.trade_executor as te
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        monkeypatch.setattr(te, "kite_ready", lambda: False)

        res = te.place_trade("HAL", qty=10, entry_type="LIMIT",
                             entry_price=4500, stop=4300, target=4900)
        assert res["ok"] and res["mode"] == "PAPER"
        rows = te.recent_trades(5)
        assert rows and rows[0]["symbol"] == "HAL"
        assert rows[0]["status"] == "PAPER_OPEN"

    def test_invalid_trade_blocked_before_any_side_effect(self, tmp_path, monkeypatch):
        import execution.trade_executor as te
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        res = te.place_trade("HAL", qty=10, entry_type="LIMIT",
                             entry_price=4500, stop=4600, target=4900)
        assert not res["ok"]
        assert te.recent_trades(5) == []   # nothing journaled


# ══════════════════════════════════════════════════════════════════════════════
# 3. Backtest simulation — fills, outcomes, mark-to-market
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestSimulate:
    def _sim(self, entry, stop, target, bars):
        from scan.signal_backtest import _simulate
        h = np.array([b[0] for b in bars], float)
        l = np.array([b[1] for b in bars], float)
        c = np.array([b[2] for b in bars], float)
        return _simulate(entry, stop, target, h, l, c)

    def test_no_fill_when_entry_never_reached(self):
        out, r = self._sim(110, 100, 130, [(105, 95, 100)] * 5)
        assert out == "NO_FILL" and r == 0.0

    def test_win_after_fill(self):
        bars = [(111, 105, 110), (131, 110, 130)]
        out, r = self._sim(110, 100, 130, bars)
        assert out == "WIN" and r == pytest.approx(2.0)

    def test_loss_after_fill(self):
        bars = [(111, 105, 110), (108, 99, 100)]
        out, r = self._sim(110, 100, 130, bars)
        assert out == "LOSS" and r == -1.0

    def test_flat_marked_to_market(self):
        bars = [(111, 105, 110), (115, 108, 114)]   # filled, neither hit
        out, r = self._sim(110, 100, 130, bars)
        assert out == "FLAT" and r == pytest.approx(0.4)  # (114-110)/10

    def test_stop_checked_before_target_same_bar(self):
        # gap-down bar that spans both: stop must win (conservative)
        bars = [(131, 99, 120)]
        out, r = self._sim(110, 100, 130, bars)
        assert out == "LOSS"

    def test_combo_edge_requires_evidence(self, tmp_path, monkeypatch):
        import scan.signal_backtest as sb
        monkeypatch.setattr(sb, "_OUT", tmp_path / "bt.json")
        (tmp_path / "bt.json").write_text(json.dumps({"signals": {
            "GOOD":  {"trades": 50, "expectancy_r": 0.30},
            "BAD":   {"trades": 50, "expectancy_r": -0.10},
            "THIN":  {"trades": 5,  "expectancy_r": 5.0},   # hype, no evidence
        }}))
        assert sb.combo_edge(["GOOD", "BAD"]) == pytest.approx(0.10)
        assert sb.combo_edge(["THIN"]) is None          # <30 trades excluded
        assert sb.combo_edge(["GOOD", "THIN"]) == 0.30  # thin one ignored


# ══════════════════════════════════════════════════════════════════════════════
# 4. Symbol filter — junk must not enter the universe
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolFilter:
    def test_valid_symbols_pass(self):
        from data.nse_universe import _is_valid_symbol
        for sym in ["RELIANCE", "BAJAJ-AUTO", "MCDOWELL-N", "M&M", "TATAMOTORS"]:
            assert _is_valid_symbol(sym), sym

    def test_junk_rejected(self):
        from data.nse_universe import _is_valid_symbol
        junk = ["AAFS29A-N0", "ACCORD-ST", "ALSDG-SF", "EBANKINAV",
                "EQU200INAV", "INDIA VIX", "ARSHIYA-BZ", "", "9INFRA-N3"]
        for sym in junk:
            assert not _is_valid_symbol(sym), sym


# ══════════════════════════════════════════════════════════════════════════════
# 5. Bhavcopy parsing — official data in, junk series out
# ══════════════════════════════════════════════════════════════════════════════

class TestBhavcopyParse:
    def test_read_day_filters_series_and_keeps_delivery(self, tmp_path, monkeypatch):
        import data.bhavcopy_store as bs
        monkeypatch.setattr(bs, "_BHAV_DIR", tmp_path)
        d = date(2026, 7, 3)
        (tmp_path / d.strftime("%d%m%Y")).with_suffix(".csv").write_text(
            "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, "
            "LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, "
            "TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
            "RELIANCE, EQ, 03-Jul-2026, 1370, 1375, 1395, 1372, 1390, 1390, "
            "1385, 5000000, 69250, 120000, 2500000, 50.0\n"
            "JUNKBOND, N1, 03-Jul-2026, 100, 100, 100, 100, 100, 100, 100, "
            "10, 0.01, 5, 5, 50.0\n")
        # _day_path uses .csv already — rename to expected name
        src = (tmp_path / d.strftime("%d%m%Y")).with_suffix(".csv")
        src.rename(tmp_path / f"{d.strftime('%d%m%Y')}.csv")

        df = bs._read_day(d)
        assert df is not None
        assert set(df["symbol"]) == {"RELIANCE"}          # N1 series dropped
        assert float(df["close"].iloc[0]) == 1390
        assert float(df["deliv_per"].iloc[0]) == 50.0     # delivery % kept


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sector heat — packs boost, loners don't
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorHeat:
    def test_pack_boost_and_loner_untouched(self):
        from scan.sector_heat import apply_sector_heat, _load_map
        assert len(_load_map()) > 200   # map parsed from nse_universe
        results = [
            {"symbol": "DLF",      "score": 60, "reasons": [], "checks": []},
            {"symbol": "GODREJPROP","score": 55, "reasons": [], "checks": []},
            {"symbol": "PRESTIGE", "score": 50, "reasons": [], "checks": []},
            {"symbol": "DABUR",    "score": 70, "reasons": [], "checks": []},
        ]
        hot = apply_sector_heat(results)
        assert any("Real Estate" in k for k in hot)
        dlf = next(r for r in results if r["symbol"] == "DLF")
        dabur = next(r for r in results if r["symbol"] == "DABUR")
        assert dlf["score"] > 60 and dlf["reasons"]
        assert dabur["score"] == 70 and not dabur["reasons"]


# ══════════════════════════════════════════════════════════════════════════════
# 7. Unified scanner — detects real setups, stays silent on garbage
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedScanner:
    def _df(self, close, vol=None):
        n = len(close)
        vol = vol if vol is not None else np.full(n, 2e6)
        idx = pd.date_range(end=pd.Timestamp(date.today()), periods=n, freq="B")
        return pd.DataFrame({"close": close, "high": close * 1.01,
                             "low": close * 0.99, "volume": vol}, index=idx)

    def test_breakout_detected_with_dated_reason(self):
        from scan.unified_scanner import UnifiedScanner
        close = np.linspace(100, 200, 260)
        close[-1] = 212
        vol = np.full(260, 2e6); vol[-1] = 5e6
        r = UnifiedScanner()._analyze("X", self._df(close, vol))
        assert r and "BREAKOUT_52W" in r.signals
        assert any("close)" in reason for reason in r.reasons)  # session-dated

    def test_downtrend_produces_nothing(self):
        from scan.unified_scanner import UnifiedScanner
        rng = np.random.default_rng(1)
        close = np.linspace(200, 100, 260) + rng.normal(0, 2, 260)
        assert UnifiedScanner()._analyze("X", self._df(close)) is None

    def test_stale_cup_handle_rejected(self):
        # SBCL case: rim 45% below price — pattern long finished
        from scan.unified_scanner import _detect_patterns
        close = np.concatenate([np.linspace(380, 549, 130),
                                np.linspace(549, 794, 130)])
        pats = _detect_patterns(close, close * 1.01, close * 0.99,
                                np.full(260, 1e6))
        assert "CUP_HANDLE" not in [p[0] for p in pats]

    def test_penny_and_illiquid_skipped(self):
        from scan.unified_scanner import UnifiedScanner
        sc = UnifiedScanner()
        close = np.linspace(5, 15, 260)                    # penny
        assert sc._analyze("X", self._df(close)) is None
        close2 = np.linspace(100, 200, 260)
        thin = np.full(260, 50)                            # ~₹10k/day turnover
        assert sc._analyze("X", self._df(close2, thin)) is None


# ══════════════════════════════════════════════════════════════════════════════
# 8. Error guard — crashes must be logged, log must not grow unbounded
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorGuard:
    def test_log_and_recall(self, tmp_path, monkeypatch):
        import core.error_guard as eg
        monkeypatch.setattr(eg, "_ERR_LOG", tmp_path / "errors.log")
        try:
            raise ValueError("atr blew up")
        except ValueError as e:
            eg.log_error("Scanner", e)
        errs = eg.recent_errors(5)
        assert len(errs) == 1
        assert "Scanner" in errs[0] and "ValueError" in errs[0]

    def test_rotation_caps_size(self, tmp_path, monkeypatch):
        import core.error_guard as eg
        monkeypatch.setattr(eg, "_ERR_LOG", tmp_path / "errors.log")
        monkeypatch.setattr(eg, "_MAX_LOG_BYTES", 500)
        for i in range(50):
            try:
                raise KeyError(f"e{i}")
            except KeyError as e:
                eg.log_error("rot", e)
        assert (tmp_path / "errors.log").stat().st_size < 5000

    def test_config_check_covers_integrations(self):
        from core.error_guard import check_config
        names = [c[0] for c in check_config()]
        assert "Kite token (daily)" in names
        assert "Telegram alerts" in names
