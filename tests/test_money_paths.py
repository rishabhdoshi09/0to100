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

    def test_regime_classifier_same_rule_history_and_today(self):
        from scan.signal_backtest import classify_regime
        rising = pd.Series(np.linspace(100, 200, 100))
        assert classify_regime(rising).iloc[-1] == "BULL"
        falling = pd.Series(np.linspace(200, 100, 100))
        assert classify_regime(falling).iloc[-1] == "BEAR"
        flat = pd.Series([100.0] * 100)
        assert classify_regime(flat).iloc[-1] == "CHOP"

    def test_wilson_ci_honest_on_thin_samples(self):
        from scan.signal_backtest import wilson_ci_pp
        assert wilson_ci_pp(0, 0) == 0.0
        thin = wilson_ci_pp(6, 12)          # 50% WR on 12 trades
        assert 25 <= thin <= 32             # ±~28pp — huge, and shown
        fat = wilson_ci_pp(300, 600)        # same WR on 600 trades
        assert fat < 5

    def test_signal_verdict_ladder(self):
        from scan.signal_backtest import signal_verdict
        assert signal_verdict(10, 2.0) == "THIN"       # hype without evidence
        assert signal_verdict(50, 0.30) == "PROVEN"
        assert signal_verdict(50, 0.10) == "POSITIVE"
        assert signal_verdict(50, 0.00) == "NEUTRAL"
        assert signal_verdict(50, -0.20) == "LOSER"

    def test_target_sweep_geometry(self):
        from scan.signal_backtest import sweep_targets
        h = np.array([100.5, 103.5]); l = np.array([98.0, 99.0])
        c = np.array([100.0, 103.0])
        out = sweep_targets(100, 95, h, l, c)
        assert out["+2%"][0] == "WIN" and out["+2%"][1] == pytest.approx(0.4)
        assert out["+3%"][0] == "WIN" and out["+3%"][1] == pytest.approx(0.6)
        assert out["+4%"][0] == "FLAT"      # 104 never printed — honest MTM

    def test_regime_edge_and_playbook(self, tmp_path, monkeypatch):
        import scan.signal_backtest as sb
        monkeypatch.setattr(sb, "_OUT", tmp_path / "bt.json")
        monkeypatch.setattr(sb, "current_regime_simple", lambda: "BULL")
        (tmp_path / "bt.json").write_text(json.dumps({
            "generated_at": "2026-07-12", "recommended_target_pct": 3.0,
            "target_sweep": {"+3%": {"trades": 500, "hit_rate": 60,
                                     "expectancy_r": 0.2}},
            "signals": {
                "VCP": {"trades": 50, "closed": 40, "expectancy_r": 0.20,
                        "verdict": "POSITIVE",
                        "by_regime": {"BULL": {"trades": 25,
                                               "expectancy_r": 0.40}}},
                "MOMENTUM": {"trades": 60, "closed": 50, "expectancy_r": 0.15,
                             "verdict": "POSITIVE",
                             "by_regime": {"BULL": {"trades": 5,
                                                    "expectancy_r": 3.0}}},
                "NR7_COIL": {"trades": 40, "closed": 35, "expectancy_r": -0.30,
                             "verdict": "LOSER", "by_regime": {}},
            }}))
        # regime bucket used when evidenced; thin bucket falls back to overall
        assert sb.edge_in_regime(["VCP"], "BULL") == 0.40
        assert sb.edge_in_regime(["MOMENTUM"], "BULL") == 0.15
        pb = sb.trading_playbook()
        assert pb["regime"] == "BULL"
        assert pb["best"][0]["signal"] == "VCP"          # 0.40R regime beats all
        assert pb["best"][0]["basis"] == "regime"
        assert pb["avoid"] == ["NR7_COIL"]
        assert pb["recommended_target_pct"] == 3.0

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


# ══════════════════════════════════════════════════════════════════════════════
# 9. Position manager — R-levels, auto-close, advice
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionManager:
    def _seed(self, tmp_path, monkeypatch, positions):
        import execution.trade_executor as te
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        monkeypatch.setattr(te, "kite_ready", lambda: False)
        for sym, entry, stop, target in positions:
            te.place_trade(sym, qty=10, entry_type="LIMIT",
                           entry_price=entry, stop=stop, target=target)

    def test_r_levels_and_auto_close(self, tmp_path, monkeypatch):
        import risk.position_manager as pm
        import data.live_quotes as lq
        self._seed(tmp_path, monkeypatch, [
            ("WINNER", 100, 95, 120), ("HALFWAY", 200, 190, 230),
            ("DANGER", 500, 480, 560), ("DEAD", 300, 290, 340)])
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms: {
            "WINNER": {"price": 112.0}, "HALFWAY": {"price": 211.0},
            "DANGER": {"price": 483.0}, "DEAD": {"price": 288.0}})
        alerts = {r["symbol"]: r["alert"] for r in pm.review_positions()}
        assert alerts["WINNER"] == "TARGET_ZONE"
        assert alerts["HALFWAY"] == "BOOK_HALF"
        assert alerts["DANGER"] == "NEAR_STOP"
        assert alerts["DEAD"] == "STOP_HIT"
        # DEAD auto-closed on next review
        assert "DEAD" not in {r["symbol"] for r in pm.review_positions()}


# ══════════════════════════════════════════════════════════════════════════════
# 10. Portfolio risk — concentration and total-risk verdicts
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioRisk:
    def test_verdict_ladder(self, tmp_path, monkeypatch):
        import execution.trade_executor as te
        import risk.portfolio_risk as prm
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        monkeypatch.setattr(te, "kite_ready", lambda: False)
        assert prm.portfolio_risk_report()["verdict"] == "OK"
        te.place_trade("DLF", 20, "LIMIT", 800, 780, 850)
        te.place_trade("GODREJPROP", 10, "LIMIT", 2500, 2450, 2650)
        assert prm.portfolio_risk_report()["verdict"] in ("CAUTION", "DANGER")
        te.place_trade("PRESTIGE", 10, "LIMIT", 1500, 1470, 1600)
        assert prm.portfolio_risk_report()["verdict"] == "DANGER"

    def test_simulation_never_mutates(self, tmp_path, monkeypatch):
        import execution.trade_executor as te
        import risk.portfolio_risk as prm
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        monkeypatch.setattr(te, "kite_ready", lambda: False)
        te.place_trade("HAL", 10, "LIMIT", 4500, 4300, 4900)
        prm.check_new_trade("INFY", 200, 1500, 1450)
        assert prm.portfolio_risk_report()["n_positions"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 11. Watchlist watcher — zone / broken / ran-away events
# ══════════════════════════════════════════════════════════════════════════════

class TestWatchlistWatcher:
    def test_events(self, tmp_path, monkeypatch):
        import sqlite3
        import risk.watchlist_watcher as ww
        import data.live_quotes as lq
        db = tmp_path / "watchlist.db"
        monkeypatch.setattr(ww, "_WL_DB", db)
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
            added_date TEXT NOT NULL, buy_zone_low REAL, buy_zone_high REAL,
            target_price REAL, stop_price REAL, notes TEXT, added_price REAL)""")
        for sym, lo, hi, tgt, stp in [("INZONE", 495, 505, 560, 470),
                                      ("BROKEN", 195, 205, 230, 185),
                                      ("QUIET", 295, 305, 340, 280)]:
            conn.execute("INSERT INTO watchlist (symbol, added_date, buy_zone_low,"
                         " buy_zone_high, target_price, stop_price) VALUES (?,?,?,?,?,?)",
                         (sym, "2026-07-01", lo, hi, tgt, stp))
        conn.commit(); conn.close()
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms: {
            "INZONE": {"price": 500.0}, "BROKEN": {"price": 180.0},
            "QUIET": {"price": 320.0}})
        by_sym = {e["symbol"]: e["event"] for e in ww.check_watchlist()}
        assert by_sym == {"INZONE": "IN_ZONE", "BROKEN": "BROKEN"}


# ══════════════════════════════════════════════════════════════════════════════
# 12. Trade coach — behavioral flags
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeCoach:
    def test_flags_bad_habits(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime, timedelta
        import execution.trade_executor as te
        import reports.trade_coach as tc
        db = tmp_path / "trades.db"
        monkeypatch.setattr(te, "_DB", db)
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, placed_at TEXT NOT NULL,
            mode TEXT NOT NULL, symbol TEXT NOT NULL, qty INTEGER NOT NULL,
            entry_type TEXT NOT NULL, entry_price REAL, stop_price REAL,
            target_price REAL, product TEXT, entry_order_id TEXT, gtt_id TEXT,
            status TEXT, note TEXT)""")
        def seed(sym, entry, stop, qty, days_ago, status="PAPER_OPEN"):
            ts = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
            conn.execute("INSERT INTO trades (placed_at, mode, symbol, qty, "
                         "entry_type, entry_price, stop_price, target_price, "
                         "product, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (ts, "PAPER", sym, qty, "LIMIT", entry, stop,
                          entry * 1.1, "CNC", status))
        seed("AAA", 100, 99.0, 10, 20, "PAPER_LOSS")
        seed("AAA", 101, 99.9, 10, 19)                 # revenge + tight stop
        seed("BBB", 500, 490, 100, 15, "PAPER_WIN")    # 100x risk of smallest
        conn.commit(); conn.close()
        joined = " ".join(tc.gather_stats(28)["insights"])
        assert "Risk inconsistent" in joined
        assert "Revenge" in joined


# ══════════════════════════════════════════════════════════════════════════════
# 13. Outcome tracker — the feedback loop must resolve via bulk quotes
# ══════════════════════════════════════════════════════════════════════════════

class TestOutcomeTracker:
    def test_outcomes_resolve_via_bulk_quotes(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta
        import core.signal_outcome_tracker as tr
        import data.live_quotes as lq
        monkeypatch.setattr(tr, "_DB_PATH", str(tmp_path / "outcomes.db"))
        conn = tr._get_conn()
        six_days = (datetime.now() - timedelta(days=6)).isoformat(timespec="seconds")
        for sym, entry in [("WINSYM", 100), ("LOSESYM", 200), ("FLATSYM", 300)]:
            conn.execute(
                "INSERT INTO signal_log (symbol, logged_at, signal_type, "
                "entry_price, stop_price) VALUES (?,?,?,?,?)",
                (sym, six_days, "UNIFIED_BUY", entry, entry * 0.96))
        conn.commit(); conn.close()
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms: {
            "WINSYM": {"price": 105.0}, "LOSESYM": {"price": 194.0},
            "FLATSYM": {"price": 301.5}})
        tr.update_outcomes()
        conn = tr._get_conn()
        d = {r["symbol"]: r["worked"] for r in conn.execute(
            "SELECT symbol, worked FROM signal_log").fetchall()}
        conn.close()
        assert d["WINSYM"] == 1
        assert d["LOSESYM"] == 0
        assert d["FLATSYM"] is None   # ±band → stays open

    def test_fresh_signals_not_judged_early(self, tmp_path, monkeypatch):
        from datetime import datetime
        import core.signal_outcome_tracker as tr
        import data.live_quotes as lq
        monkeypatch.setattr(tr, "_DB_PATH", str(tmp_path / "o2.db"))
        conn = tr._get_conn()
        conn.execute(
            "INSERT INTO signal_log (symbol, logged_at, signal_type, "
            "entry_price, stop_price) VALUES (?,?,?,?,?)",
            ("FRESH", datetime.now().isoformat(timespec="seconds"),
             "UNIFIED_BUY", 100, 96))
        conn.commit(); conn.close()
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms: {"FRESH": {"price": 150.0}})
        tr.update_outcomes()
        conn = tr._get_conn()
        row = conn.execute("SELECT worked FROM signal_log").fetchone()
        conn.close()
        assert row["worked"] is None   # <5 days old — too early to judge


# ══════════════════════════════════════════════════════════════════════════════
# 14. Telegram delivery — long messages must split, never silently drop
# ══════════════════════════════════════════════════════════════════════════════

class TestTelegramSplit:
    def test_short_message_unsplit(self):
        from alerts.telegram_alerts import AlertEngine
        assert AlertEngine._split_message("hello", 3800) == ["hello"]

    def test_long_message_splits_lossless(self):
        from alerts.telegram_alerts import AlertEngine
        blocks = [f"BLOCK{i} " + "x" * 790 for i in range(10)]
        msg = "\n\n".join(blocks)
        chunks = AlertEngine._split_message(msg, 3800)
        assert len(chunks) >= 2
        assert all(len(c) <= 3800 for c in chunks)
        joined = "".join(chunks)
        for i in range(10):
            assert f"BLOCK{i}" in joined     # nothing dropped

    def test_monster_block_hard_cut(self):
        from alerts.telegram_alerts import AlertEngine
        chunks = AlertEngine._split_message("y" * 9000, 3800)
        assert all(len(c) <= 3800 for c in chunks)
        assert sum(len(c) for c in chunks) == 9000


# ══════════════════════════════════════════════════════════════════════════════
# 15. Autopilot — every guardrail is money-critical
# ══════════════════════════════════════════════════════════════════════════════

class TestAutopilot:
    def _setup(self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        import execution.trade_executor as te
        import scan.sector_heat as sh
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        monkeypatch.setattr(ap, "_state", {}, raising=False)
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda msg: None)
        monkeypatch.setattr(ap, "_broker_cash", lambda: 500000.0)
        monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
        monkeypatch.setattr(sh, "sector_performance", lambda min_members=3: [
            {"sector": "Defence", "chg_1d": 1.5, "chg_5d": 3.0, "members": 4},
            {"sector": "IT / Software", "chg_1d": 0.8, "chg_5d": 1.0, "members": 5},
        ])
        ap.set_config(allocation=100000, mode="PAPER")
        return ap, te

    def test_not_armed_never_trades(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False
        assert te.recent_trades(2) == []

    def test_gates_and_three_pct_target(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        ok, _ = ap.arm()
        assert ok
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # weak sector / low score / negative edge all rejected
        assert ap.consider("X", 500, 480, 80, 0.2, "Cement", "t") is False
        assert ap.consider("HAL", 4500, 4300, 45, 0.2, "Defence", "t") is False
        assert ap.consider("HAL", 4500, 4300, 80, -0.1, "Defence", "t") is False
        # valid → placed, tagged, +3% target (not scanner target)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        t = te.recent_trades(1)[0]
        assert "AUTOPILOT" in t["note"]
        assert abs(float(t["target_price"]) - 4500 * 1.03) < 1.0
        # symbol once per day
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False

    def test_reserve_never_deployable_and_compounding(self, tmp_path, monkeypatch):
        import sqlite3
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t")
        st = ap.get_status()
        assert st["available"] <= st["pool"] * 0.90 - st["deployed"] + 1
        # close as WIN → pool compounds
        conn = sqlite3.connect(te._DB)
        conn.execute("UPDATE trades SET status='PAPER_WIN' WHERE symbol='HAL'")
        conn.commit(); conn.close()
        ap._account_closed_trades()
        st2 = ap.get_status()
        assert st2["realized_pnl"] > 0 and st2["pool"] > 100000

    def test_circuit_breaker_disarms(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec='seconds'), "PAPER", "CRASH",
             100, "MARKET", 500, 450, 515, "CNC", "PAPER_LOSS", "AUTOPILOT:t"))
        conn.commit(); conn.close()
        ap._circuit_breaker()
        st = ap.get_status()
        assert not st["armed"] and "circuit breaker" in st["disarmed_reason"]

    def test_live_arm_needs_exact_phrase(self, tmp_path, monkeypatch):
        ap, _ = self._setup(tmp_path, monkeypatch)
        ap.set_config(mode="LIVE")
        ok, _ = ap.arm("wrong phrase")
        assert not ok

    def test_sector_fallback_lookup(self, tmp_path, monkeypatch):
        """Signal without a sector tag must be judged by sector_of(), not
        auto-rejected — a lone strong stock in a top sector still trades."""
        import scan.sector_heat as sh
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        monkeypatch.setattr(sh, "sector_of", lambda sym: "Defence")
        assert ap.consider("BEL", 300, 290, 80, 0.2, "", "t") is True
        # and a lone stock whose real sector is weak still gets rejected
        monkeypatch.setattr(sh, "sector_of", lambda sym: "Cement")
        assert ap.consider("ACC", 2400, 2320, 80, 0.2, "", "t") is False

    def test_bad_time_format_rejected(self, tmp_path, monkeypatch):
        ap, _ = self._setup(tmp_path, monkeypatch)
        ap.set_config(start_time="around ten", end_time="25:99")
        st = ap.get_status()
        assert st["start_time"] == "09:30" and st["end_time"] == "14:45"

    def _insert_closed(self, te, symbol, entry, stop, target, qty, win,
                       source="scanner"):
        import sqlite3
        from datetime import datetime
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec='seconds'), "PAPER", symbol,
             qty, "MARKET", entry, stop, target, "CNC",
             "PAPER_WIN" if win else "PAPER_LOSS", f"AUTOPILOT:{source}"))
        conn.commit(); conn.close()

    def test_report_card_math_and_evidence_gate(self, tmp_path, monkeypatch):
        """P&L / R / equity curve exact; <30 trades can never claim READY."""
        ap, te = self._setup(tmp_path, monkeypatch)
        # win: (515-500)*10 = +150 @ risk (500-450)*10=500 → +0.3R
        self._insert_closed(te, "W1", 500, 450, 515, 10, win=True)
        # loss: (450-500)*10 = -500 → -1R
        self._insert_closed(te, "L1", 500, 450, 515, 10, win=False,
                            source="sniper")
        rc = ap.report_card()
        s = rc["stats"]
        assert s["n"] == 2 and s["wins"] == 1 and s["win_rate"] == 50.0
        assert s["total_pnl"] == -350
        assert rc["trades"][0]["pnl"] == 150 and rc["trades"][0]["r"] == 0.3
        assert rc["trades"][1]["pnl"] == -500 and rc["trades"][1]["r"] == -1.0
        assert rc["trades"][1]["equity"] == -350          # running curve
        assert rc["trades"][1]["source"] == "sniper"
        assert s["max_drawdown"] == 500                   # peak 150 → -350
        # evidence gate: profitable ya nahi, <30 trades = no claim
        assert rc["verdict"] == "COLLECTING_EVIDENCE"

    def test_report_card_verdict_after_evidence(self, tmp_path, monkeypatch):
        """30+ trades: positive expectancy → READY; negative → NOT_READY."""
        ap, te = self._setup(tmp_path, monkeypatch)
        for i in range(20):
            self._insert_closed(te, f"W{i}", 100, 95, 103, 10, win=True)
        for i in range(10):
            self._insert_closed(te, f"L{i}", 100, 95, 103, 10, win=False)
        rc = ap.report_card()
        assert rc["stats"]["n"] == 30
        # gross win 20*30=600, gross loss 10*50=500 → PF 1.2 < 1.3 → NOT_READY
        assert rc["verdict"] == "NOT_READY"
        for i in range(5):
            self._insert_closed(te, f"W2{i}", 100, 95, 103, 10, win=True)
        rc2 = ap.report_card()
        # gross win 750 / loss 500 → PF 1.5, expectancy > 0 → READY
        assert rc2["verdict"] == "READY_CANDIDATE"
        assert rc2["stats"]["profit_factor"] == 1.5

    def test_regime_gate_blocks_bad_tape(self, tmp_path, monkeypatch):
        """DISTRIBUTION / BEAR tape mein koi naya entry nahi; bull tape ok."""
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        monkeypatch.setattr(ap, "_market_regime", lambda: "DISTRIBUTION")
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False
        monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        # user can disable the gate — their call, their money
        ap.set_config(regime_gate=False)
        monkeypatch.setattr(ap, "_market_regime", lambda: "DISTRIBUTION")
        assert ap.consider("BEL", 300, 290, 80, 0.2, "Defence", "t") is True

    def test_breakeven_trail_moves_stop_keeps_original(self, tmp_path, monkeypatch):
        """+2% pe stop entry pe; orig_stop preserved so R stays honest."""
        import sqlite3
        import data.live_quotes as lq
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        # +1% — no trail yet
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms: {"HAL": {"price": 4545.0}})
        ap._trail_stops()
        t = te.recent_trades(1)[0]
        assert float(t["stop_price"]) == 4300
        # +2.2% — stop moves to breakeven, original stop preserved
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms: {"HAL": {"price": 4600.0}})
        ap._trail_stops()
        conn = sqlite3.connect(te._DB); conn.row_factory = sqlite3.Row
        row = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='HAL'").fetchone())
        conn.close()
        assert float(row["stop_price"]) == 4500.0
        assert float(row["orig_stop"]) == 4300.0
        assert "breakeven trail" in row["note"]
        # idempotent: second pass must not re-trail (stop no longer < entry)
        ap._trail_stops()

    def test_exchange_fill_preferred_over_planned_exit(self, tmp_path, monkeypatch):
        """Reconciled exit_price beats the planned GTT leg in P&L everywhere."""
        import sqlite3
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        ap._ensure_exit_col()
        conn = sqlite3.connect(te._DB)
        # actual exchange exit ₹4550 (planned target was 4635)
        conn.execute("UPDATE trades SET status='AUTO_WIN', exit_price=4550 "
                     "WHERE symbol='HAL'")
        conn.commit(); conn.close()
        ap._account_closed_trades()
        qty = int(te.recent_trades(1)[0]["qty"])
        assert ap.get_status()["realized_pnl"] == (4550 - 4500) * qty
        rc = ap.report_card()
        assert rc["trades"][0]["exit"] == 4550
        assert rc["trades"][0]["pnl"] == (4550 - 4500) * qty

    def test_no_trade_without_valid_stop(self, tmp_path, monkeypatch):
        """Sniper hits can carry stop=0 (watchlist rows without a plan) —
        a trade without a real stop must NEVER be placed (invariant #3)."""
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 0, 80, 0.2, "Defence", "t") is False
        assert ap.consider("HAL", 4500, 4600, 80, 0.2, "Defence", "t") is False
        assert ap.consider("HAL", 0, 0, 80, 0.2, "Defence", "t") is False
        assert te.recent_trades(3) == []
        # sniper hook path with stop=0 also rejected end-to-end
        ap.on_breakout({"symbol": "HAL", "ltp": 4500, "trigger": 4480,
                        "stop": 0, "target": 0})
        assert te.recent_trades(3) == []

    def test_eod_digest_once_per_day(self, tmp_path, monkeypatch):
        from datetime import datetime as dt
        ap, _ = self._setup(tmp_path, monkeypatch)
        sent = []
        monkeypatch.setattr(ap, "_notify", lambda msg: sent.append(msg))
        ap.arm()
        friday_close = dt(2026, 7, 10, 16, 0)
        assert ap.eod_digest(now=friday_close) is True
        assert len(sent) >= 1 and "EOD Digest" in sent[-1]
        assert ap.eod_digest(now=friday_close) is False      # once per day
        # market hours / weekend never fire
        assert ap.eod_digest(now=dt(2026, 7, 10, 14, 0)) is False
        assert ap.eod_digest(now=dt(2026, 7, 11, 16, 0)) is False

    def test_conviction_multiplier_evidence_only(self, tmp_path, monkeypatch):
        ap, _ = self._setup(tmp_path, monkeypatch)
        f = ap._conviction_multiplier
        assert f(85, 0.30) == 1.5      # strong score + proven edge
        assert f(60, 0.10) == 1.0      # ordinary evidence → base
        assert f(60, None) == 0.75     # unmeasured = low conviction
        assert f(85, None) == 1.0      # strong score, no backtest data
        assert f(60, 0.01) == 0.75     # marginal edge → half-ish
        assert f(85, 0.01) == 1.0
        assert 0.5 <= f(0, None) <= 1.5

    def test_conviction_sizing_changes_qty(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # base: pool 1L, risk 1% = 1000 / (500-450) = 20 qty
        ap.set_config(conviction_sizing=False)
        assert ap.consider("BASE", 500, 450, 85, 0.30, "Defence", "t") is True
        assert int(te.recent_trades(1)[0]["qty"]) == 20
        # conviction on: 1.5× → 30 qty (caps still respected)
        ap.set_config(conviction_sizing=True)
        assert ap.consider("CONV", 500, 450, 85, 0.30, "Defence", "t") is True
        assert int(te.recent_trades(1)[0]["qty"]) == 30

    def test_adaptive_source_gate_pauses_proven_loser(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # 14 sniper losses — below evidence threshold, still allowed
        for i in range(14):
            self._insert_closed(te, f"S{i}", 100, 95, 103, 10, win=False,
                                source="sniper")
        assert ap._source_paused("sniper") is None
        # 15th loss → sniper pauses itself; scanner unaffected
        self._insert_closed(te, "S14", 100, 95, 103, 10, win=False,
                            source="sniper")
        assert "paused" in (ap._source_paused("sniper") or "")
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence",
                           "sniper") is False
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence",
                           "scanner") is True
        # user override: gate off → source trades again
        ap.set_config(adaptive_source_gate=False)
        assert ap.consider("BEL", 300, 290, 80, 0.2, "Defence",
                           "sniper") is True

    def test_time_stop_paper_closes_live_nudges_once(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime as dt, timedelta
        import data.live_quotes as lq
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 500, 450, 80, 0.2, "Defence", "t") is True
        old = (dt.now() - timedelta(days=6)).isoformat(timespec="seconds")
        conn = sqlite3.connect(te._DB)
        conn.execute("UPDATE trades SET placed_at=? WHERE symbol='HAL'", (old,))
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (old, "LIVE", "BEL", 10, "MARKET", 300, 290, 309, "CNC",
             "PLACED", "AUTOPILOT:t"))
        conn.commit(); conn.close()
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 505.0},
                                                   "BEL": {"price": 301.0}})
        nudges = []
        monkeypatch.setattr(ap, "_notify", lambda m: nudges.append(m))
        ap._time_stop()
        conn = sqlite3.connect(te._DB); conn.row_factory = sqlite3.Row
        hal = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='HAL'").fetchone())
        bel = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='BEL'").fetchone())
        conn.close()
        # PAPER: closed at market with honest outcome + real exit price
        assert hal["status"] == "PAPER_WIN" and float(hal["exit_price"]) == 505.0
        assert "time-stop" in hal["note"]
        # LIVE: still open, nudged exactly once even across repeat cycles
        assert bel["status"] == "PLACED"
        assert len([n for n in nudges if "BEL" in n]) == 1
        ap._time_stop()
        assert len([n for n in nudges if "BEL" in n]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 16. Conviction tier — highest conviction first, everywhere
# ══════════════════════════════════════════════════════════════════════════════

class TestConvictionTier:
    def test_high_conviction_is_evidence_only(self):
        from scan.auto_scan import tag_conviction
        rows = [
            {"symbol": "A", "verdict": "STRONG BUY", "score": 80, "edge_r": 0.20},
            {"symbol": "B", "verdict": "BUY", "score": 75, "edge_r": 0.10},
            {"symbol": "C", "verdict": "BUY", "score": 90},              # no edge data
            {"symbol": "D", "verdict": "WATCH", "score": 95, "edge_r": 0.50},
            {"symbol": "E", "verdict": "BUY", "score": 74, "edge_r": 0.30},
        ]
        tag_conviction(rows)
        flags = {r["symbol"]: r["high_conviction"] for r in rows}
        assert flags == {"A": True, "B": True,          # boundary inclusive
                         "C": False,                    # unmeasured ≠ conviction
                         "D": False,                    # WATCH never high
                         "E": False}                    # score below 75
        # rank ordering: any BUY outranks the best WATCH
        ranks = {r["symbol"]: r["conviction_rank"] for r in rows}
        assert ranks["A"] > ranks["B"] > ranks["D"]
        assert min(ranks["B"], ranks["E"]) > ranks["D"]

    def test_autopilot_fills_slots_highest_conviction_first(
            self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        ap._state = {}
        ap.set_config(allocation=100000, mode="PAPER")
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        ap.arm()
        taken = []
        monkeypatch.setattr(
            ap, "consider",
            lambda symbol, **kw: taken.append(symbol) or True)
        ap.on_setups([
            {"symbol": "LOW", "verdict": "BUY", "score": 60,
             "conviction_rank": 160, "price": 100, "stop": 95},
            {"symbol": "TOP", "verdict": "STRONG BUY", "score": 88,
             "conviction_rank": 295, "price": 100, "stop": 95},
            {"symbol": "MID", "verdict": "BUY", "score": 80,
             "conviction_rank": 182, "price": 100, "stop": 95},
            {"symbol": "SKIP", "verdict": "WATCH", "score": 99,
             "conviction_rank": 99, "price": 100, "stop": 95},
        ])
        assert taken == ["TOP", "MID", "LOW"]           # priority order, WATCH out


# ══════════════════════════════════════════════════════════════════════════════
# 17. System Pulse + quote micro-cache — smoothness is money too
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemHealth:
    def test_beat_status_ladder(self, monkeypatch):
        import core.health as h
        monkeypatch.setattr(h, "_beats", {}, raising=False)
        assert h.daemon_status("auto_scan")[0] == "NEVER"
        h.beat("auto_scan", note="ready")
        status, age = h.daemon_status("auto_scan")
        assert status == "OK" and age < 5
        # age past warn → SLOW; past dead → DEAD (cadence 1200/4500)
        import time as t
        h._beats["auto_scan"]["ts"] = t.time() - 2000
        assert h.daemon_status("auto_scan")[0] == "SLOW"
        h._beats["auto_scan"]["ts"] = t.time() - 5000
        assert h.daemon_status("auto_scan")[0] == "DEAD"

    def test_latency_percentiles_and_pulse(self, monkeypatch):
        import core.health as h
        monkeypatch.setattr(h, "_lat", {}, raising=False)
        monkeypatch.setattr(h, "_beats", {}, raising=False)
        monkeypatch.setattr(h, "_counters", {}, raising=False)
        for ms in range(1, 101):                     # 1..100 ms
            h.record_latency("quote_fetch", ms / 1000)
        p = h.pulse()["latency"]["quote_fetch"]
        assert p["n"] == 100
        assert 45 <= p["p50_ms"] <= 55
        assert 90 <= p["p95_ms"] <= 100
        with h.timed("x"):
            pass
        assert h.pulse()["latency"]["x"]["n"] == 1


class TestLiveTicker:
    def test_tick_fold_and_age(self, monkeypatch):
        import data.live_ticker as lt
        monkeypatch.setattr(lt, "_store", {}, raising=False)
        ticks = [
            {"instrument_token": 1, "last_price": 4512.5,
             "volume_traded": 120000,
             "ohlc": {"close": 4450.0, "high": 4520.0, "low": 4440.0}},
            {"instrument_token": 2, "last_price": 0},          # junk → skipped
            {"instrument_token": 99, "last_price": 100},       # unknown token
        ]
        n = lt.update_store(ticks, {1: "HAL", 2: "BEL"}, now=1000.0)
        assert n == 1
        monkeypatch.setattr(lt.time, "time", lambda: 1003.0)
        snap = lt.get_ticks(["HAL"])
        assert snap["HAL"]["price"] == 4512.5
        assert snap["HAL"]["chg_pct"] == pytest.approx(1.4, abs=0.05)
        assert snap["HAL"]["age_s"] == 3.0        # stale must look stale
        assert "BEL" not in snap

    def test_status_without_stream(self, monkeypatch):
        import data.live_ticker as lt
        monkeypatch.setattr(lt, "_store", {}, raising=False)
        monkeypatch.setattr(lt, "_started", False, raising=False)
        s = lt.status()
        assert s["streaming"] is False and s["last_tick_age_s"] is None


class TestDeadSymbolRegistry:
    def _fresh(self, tmp_path, monkeypatch):
        import data.dead_symbols as ds
        monkeypatch.setattr(ds, "_FILE", tmp_path / "dead.json")
        monkeypatch.setattr(ds, "_cache", None, raising=False)
        return ds

    def test_mark_ttl_and_persistence(self, tmp_path, monkeypatch):
        ds = self._fresh(tmp_path, monkeypatch)
        assert not ds.is_dead("ACLGATI")
        ds.mark_dead("ACLGATI", "kite missing + yf 404")
        assert ds.is_dead("ACLGATI") and ds.is_dead("aclgati")
        # survives a process restart (reload from disk)
        monkeypatch.setattr(ds, "_cache", None, raising=False)
        assert ds.is_dead("ACLGATI")
        # TTL expiry → one fresh chance (relist/rename auto-heals)
        ds._load()["ACLGATI"]["ts"] = 0
        assert not ds.is_dead("ACLGATI")

    def test_history_path_registers_and_stops_asking(self, tmp_path, monkeypatch):
        import data.market_data as md
        ds = self._fresh(tmp_path, monkeypatch)
        calls = {"yf": 0}

        def _kite_fail(*a, **k):
            raise ValueError("Instrument ACLGATI not found in NSE instrument list")

        def _yf_fail(*a, **k):
            calls["yf"] += 1
            raise ValueError("No data returned from yfinance for ACLGATI")
        monkeypatch.setattr(md, "_kite_available", lambda: True)
        monkeypatch.setattr(md, "get_historical_data_kite", _kite_fail)
        monkeypatch.setattr(md, "get_historical_data_yfinance", _yf_fail)
        # first attempt: both fail → registered
        with pytest.raises(ValueError):
            md.get_historical_data("ACLGATI")
        assert calls["yf"] == 1 and ds.is_dead("ACLGATI")
        # every later attempt: instant skip, ZERO network calls
        with pytest.raises(ValueError, match="dead-symbol registry"):
            md.get_historical_data("ACLGATI")
        assert calls["yf"] == 1

    def test_live_symbol_never_registered(self, tmp_path, monkeypatch):
        """Kite down for a NORMAL reason (token expired) must not mark
        a healthy symbol dead even if yfinance also hiccups."""
        import data.market_data as md
        ds = self._fresh(tmp_path, monkeypatch)

        def _kite_fail(*a, **k):
            raise ValueError("Incorrect `api_key` or `access_token`")

        def _yf_fail(*a, **k):
            raise ValueError("timeout")
        monkeypatch.setattr(md, "_kite_available", lambda: True)
        monkeypatch.setattr(md, "get_historical_data_kite", _kite_fail)
        monkeypatch.setattr(md, "get_historical_data_yfinance", _yf_fail)
        with pytest.raises(ValueError):
            md.get_historical_data("RELIANCE")
        assert not ds.is_dead("RELIANCE")


class TestProviderPolicy:
    def test_provider_upgrades_to_kite_after_login(self, monkeypatch):
        """App opened before morning Kite login must NOT stay on Google
        scrape all day — the provider re-checks and upgrades to Kite."""
        import data.market_data as md

        class _FakeKite:
            pass
        monkeypatch.setattr(md, "_KiteProvider", _FakeKite)
        monkeypatch.setattr(md, "_provider", None, raising=False)
        # before login → Google provider
        monkeypatch.setattr(md, "_kite_available", lambda: False)
        assert isinstance(md.get_provider(), md._GoogleFinanceProvider)
        # user logs in mid-session → next call upgrades to Kite
        monkeypatch.setattr(md, "_kite_available", lambda: True)
        assert isinstance(md.get_provider(), _FakeKite)
        # and stays Kite (cached, no rebuild churn)
        p = md.get_provider()
        assert md.get_provider() is p


class TestQuoteMicroCache:
    def _fresh(self, monkeypatch):
        import data.live_quotes as lq
        monkeypatch.setattr(lq, "_qcache", {}, raising=False)
        calls = {"n": 0}

        def fake_kite(symbols):
            calls["n"] += 1
            return {s: {"price": 100.0, "chg_pct": 1.0, "source": "kite"}
                    for s in symbols if s != "MISSING"}
        monkeypatch.setattr(lq, "_kite_quotes", fake_kite)
        monkeypatch.setattr(lq, "_nse_quotes", lambda s: {})
        monkeypatch.setattr(lq, "_google_quotes", lambda s: {})
        return lq, calls

    def test_second_call_within_ttl_is_free(self, monkeypatch):
        lq, calls = self._fresh(monkeypatch)
        q1 = lq.get_live_quotes(["HAL", "BEL"])
        assert calls["n"] == 1 and q1["HAL"]["price"] == 100.0
        q2 = lq.get_live_quotes(["HAL", "BEL"])
        assert calls["n"] == 1                      # served from cache
        assert q2 == q1
        # partial overlap: only the new symbol hits the network
        lq.get_live_quotes(["HAL", "TCS"])
        assert calls["n"] == 2

    def test_ttl_zero_bypasses_and_misses_not_cached(self, monkeypatch):
        lq, calls = self._fresh(monkeypatch)
        lq.get_live_quotes(["HAL"])
        lq.get_live_quotes(["HAL"], ttl=0)
        assert calls["n"] == 2                      # forced fresh fetch
        # a symbol no source can answer is retried every call, never cached
        assert lq.get_live_quotes(["MISSING"]) == {}
        assert lq.get_live_quotes(["MISSING"]) == {}
        assert calls["n"] == 4

    def test_expiry_refetches(self, monkeypatch):
        lq, calls = self._fresh(monkeypatch)
        lq.get_live_quotes(["HAL"])
        # age the cache entry past the TTL
        ts, q = lq._qcache["HAL"]
        lq._qcache["HAL"] = (ts - 60, q)
        lq.get_live_quotes(["HAL"])
        assert calls["n"] == 2
