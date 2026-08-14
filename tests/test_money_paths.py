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


@pytest.fixture(autouse=True)
def _costs_off(monkeypatch):
    """Existing money-path tests assert GROSS R-math and rupee P&L; trading costs
    are an orthogonal layer tested on their own (TestTradingCosts). Zero them by
    default so an R-attribution test isn't coupled to the cost model — a cost
    test re-enables them explicitly."""
    try:
        monkeypatch.setattr("core.costs._ENABLED", False)
    except Exception:
        pass


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
# 2b. Trading costs — the honesty layer: net R < gross R
# ══════════════════════════════════════════════════════════════════════════════

class TestTradingCosts:
    def test_cost_model_values(self, monkeypatch):
        monkeypatch.setattr("core.costs._ENABLED", True)          # costs ON
        from core.costs import round_trip_cost_pct, cost_in_r, cost_rupees, net_r
        assert round_trip_cost_pct("CNC") == pytest.approx(0.32)   # 0.22 + 0.10 slip
        assert round_trip_cost_pct("MIS") == pytest.approx(0.20)   # 0.10 + 0.10
        # a 4% stop giving up 0.32% round-trip = 0.08R to costs
        assert cost_in_r(0.04, "CNC") == pytest.approx(0.08)
        assert net_r(2.0, 0.04, "CNC") == pytest.approx(1.92)
        assert cost_rupees(10, 100, "CNC") == pytest.approx(3.2)   # 1000 × 0.32%
        assert cost_in_r(0.0) == 0.0 and cost_in_r(-1) == 0.0      # safe

    def test_equity_curve_is_net_of_costs(self, tmp_path, monkeypatch):
        # re-enable real costs (override the module autouse) and check the Report
        # Card expectancy is NET, not gross.
        monkeypatch.setattr("core.costs._ENABLED", True)
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "c.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 10)
        # 6 wins @ +8% and 4 losses @ -4%, entry 100 / stop 96 (4% risk)
        rows = [(8.0, 1)] * 6 + [(-4.0, 0)] * 4
        for i, (pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i * 4)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,entry_price,"
                "stop_price,outcome_pct,worked,outcome_checked_at,outcome_price) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "breakout", 100.0, 96.0, pct, worked, la, 100 * (1 + pct / 100)))
        conn.commit(); conn.close()
        from reports.verdict_dashboard import build_equity_curve
        s = build_equity_curve(100000.0)["stats"]
        # gross expectancy = (6·2 − 4·1)/10 = +0.80R; net subtracts 0.08R/trade
        assert s["expectancy_r"] == pytest.approx(0.72, abs=0.01)
        assert s["expectancy_r"] < 0.80                    # strictly worse than gross


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

    def test_breakeven_trail_scratches_a_faded_pop(self):
        """The realism fix: a trade that pops past the breakeven trigger then
        fades back must exit at ~0R (SCRATCH) — modelling the live breakeven
        trail — NOT bleed to a FLAT loss like a naked-stop strawman would.
        This is the exact population that was dragging every expectancy
        mildly negative."""
        from scan.signal_backtest import _simulate
        entry, stop, target = 100.0, 94.0, 112.0        # risk 6, be at +2% = 102
        # pops to 102 (arms breakeven), then fades to tag entry
        hi = np.array([101, 102, 101, 100.5, 99.5])
        lo = np.array([99, 100.5, 99.5, 99, 98.5])
        cl = np.array([100.5, 101.5, 100, 99.8, 99.0])
        # naked stop (be off) → FLAT loss
        o_old, r_old = _simulate(entry, stop, target, hi, lo, cl, be_pct=0.0)
        assert o_old == "FLAT" and r_old < 0
        # breakeven trail on → SCRATCH at 0R (not a loss)
        o_new, r_new = _simulate(entry, stop, target, hi, lo, cl, be_pct=2.0)
        assert o_new == "SCRATCH" and r_new == 0.0
        assert r_new > r_old                              # strictly less pessimistic

    def test_breakeven_trail_leaves_clean_winners_and_quick_losers(self):
        """The trail must not distort the unambiguous cases: a clean run to
        target is still +2R, a fast stop-out before any pop is still -1R."""
        from scan.signal_backtest import _simulate
        entry, stop, target = 100.0, 94.0, 112.0
        win = _simulate(entry, stop, target,
                        np.array([101, 104, 108, 113]), np.array([99.5, 101, 104, 108]),
                        np.array([100.5, 103, 107, 112]), be_pct=2.0)
        assert win[0] == "WIN" and win[1] == pytest.approx(2.0)
        loss = _simulate(entry, stop, target,
                         np.array([100.5, 99, 96, 93]), np.array([99, 96, 93, 90]),
                         np.array([99.5, 97, 94, 91]), be_pct=2.0)
        assert loss[0] == "LOSS" and loss[1] == -1.0

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

    def test_edge_veto_only_demotes_proven_losers(self):
        """The edge gate must VETO a BUY only when the backtest calls the combo
        a LOSER (≤ -0.05, the system's own signal_verdict line) — NOT a
        NEUTRAL/breakeven combo whose slightly-negative point estimate is
        within measurement noise. The old -0.02 cutoff sat inside the NEUTRAL
        band and vetoed nearly everything."""
        from scan.auto_scan import _edge_vetoes
        # NEUTRAL band (the screenshot's -0.04R) must NOT veto
        assert _edge_vetoes(-0.04, "BUY") is False
        assert _edge_vetoes(-0.02, "BUY") is False
        # proven LOSER must veto
        assert _edge_vetoes(-0.05, "BUY") is True
        assert _edge_vetoes(-0.07, "BUY") is True         # UNITDSPR case
        # positive edge, no-evidence None, and already-WATCH never veto
        assert _edge_vetoes(0.10, "BUY") is False
        assert _edge_vetoes(None, "BUY") is False
        assert _edge_vetoes(-0.20, "WATCH") is False


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

    def test_instrument_cross_check_drops_stale_listings(self):
        """A symbol that LOOKS clean (passes _is_valid_symbol) but isn't a real
        Kite instrument — e.g. AMIRCHAND, a stale NSE listing — must be dropped
        so it stops missing on every fetch cycle. Pattern rules can't catch
        this; only the instrument map can."""
        from data.nse_universe import _filter_to_instruments
        tok = {f"SYM{i}": i for i in range(2000)}
        tok.update({"RELIANCE": 1, "INFY": 2, "TCS": 3})
        uni = ["RELIANCE", "INFY", "TCS"] + [f"SYM{i}" for i in range(97)] \
            + ["AMIRCHAND", "FAKESTK"]
        out = _filter_to_instruments(uni, tok)
        assert "AMIRCHAND" not in out and "FAKESTK" not in out
        assert all(s in out for s in ("RELIANCE", "INFY", "TCS"))

    def test_instrument_cross_check_is_fail_safe(self):
        """Guards: an unloaded/tiny map or an implausibly-large drop must be a
        no-op — a bad instrument load can never nuke the scanning universe."""
        from data.nse_universe import _filter_to_instruments
        uni = ["RELIANCE", "INFY", "TCS", "AMIRCHAND"]
        # tiny/unloaded map → unchanged
        assert _filter_to_instruments(uni, {"RELIANCE": 1}) == uni
        assert _filter_to_instruments(uni, {}) == uni
        # would drop >15% (map incomplete) → unchanged
        big_map = {f"SYM{i}": i for i in range(2000)}
        big_map["RELIANCE"] = 1
        junky = ["RELIANCE"] + [f"X{i}" for i in range(50)]   # only 1 in map
        assert _filter_to_instruments(junky, big_map) == junky


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
        # a few small pullback days so RSI isn't pinned at the theoretical
        # 100 ceiling (a pure straight-line ramp never happens in real
        # data) — keeps this test clear of the RSI hard-reject band
        close[-3] = close[-4] - 2.0
        close[-8] = close[-9] - 2.0
        close[-12] = close[-13] - 2.0
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

    def test_outcomes_resolve_via_true_first_touch(self, tmp_path, monkeypatch):
        # PREFERRED path: when a target + bhavcopy bars exist, the outcome is
        # judged by target-vs-stop first-touch (like the backtest), NOT the
        # crude ±band point-in-time mark.
        from datetime import datetime, timedelta
        import pandas as pd
        import core.signal_outcome_tracker as tr
        import data.live_quotes as lq
        import data.bhavcopy_store as bs
        monkeypatch.setattr(tr, "_DB_PATH", str(tmp_path / "ft.db"))
        six_days = (datetime.now() - timedelta(days=6)).isoformat(timespec="seconds")
        conn = tr._get_conn()
        # entry 100 / stop 96 / target 106 for each; the PATH decides the outcome
        for sym in ("WIN", "LOSS", "OPEN"):
            conn.execute(
                "INSERT INTO signal_log (symbol, logged_at, signal_type, "
                "entry_price, stop_price, target_price) VALUES (?,?,?,?,?,?)",
                (sym, six_days, "UNIFIED_BUY", 100.0, 96.0, 106.0))
        conn.commit(); conn.close()
        idx = pd.date_range(datetime.now() - timedelta(days=6), periods=16, freq="D")

        def _df(highs, lows, closes):
            n = len(highs)
            return pd.DataFrame({"high": highs, "low": lows, "close": closes},
                                index=idx[:n])
        paths = {
            # fills day0, target 106 hit day1 before any stop → WIN
            "WIN":  _df([101, 107, 108], [99, 100, 101], [100, 106, 107]),
            # fills day0, stop 96 hit day1 before target → LOSS
            "LOSS": _df([101, 104, 105], [99, 95, 94], [100, 96, 95]),
            # filled, drifts sideways, only 3 bars < horizon → still OPEN
            "OPEN": _df([101, 102, 103], [99, 100, 101], [100, 101, 102]),
        }
        monkeypatch.setattr(bs, "get_ohlcv", lambda s: paths.get(s.upper()))
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms: {})   # no network
        tr.update_outcomes()
        conn = tr._get_conn()
        d = {r["symbol"]: (r["worked"], r["outcome_pct"]) for r in conn.execute(
            "SELECT symbol, worked, outcome_pct FROM signal_log").fetchall()}
        conn.close()
        assert d["WIN"][0] == 1 and abs(d["WIN"][1] - 6.0) < 1e-6      # exited AT target
        assert d["LOSS"][0] == 0 and abs(d["LOSS"][1] - (-4.0)) < 1e-6  # exited AT stop
        assert d["OPEN"][0] is None                                    # still live

    def test_reresolve_history_corrects_old_crude_labels(self, tmp_path, monkeypatch):
        # back-data cleanup: a row the OLD ±band method mislabeled a LOSS is
        # re-judged by true first-touch and corrected to the WIN it actually was.
        from datetime import datetime, timedelta
        import pandas as pd
        import core.signal_outcome_tracker as tr
        import data.bhavcopy_store as bs
        monkeypatch.setattr(tr, "_DB_PATH", str(tmp_path / "rr.db"))
        old = (datetime.now() - timedelta(days=20)).isoformat(timespec="seconds")
        conn = tr._get_conn()
        # crude method stored worked=0 / outcome_pct=-1 (a −1% wiggle looked like a loss)
        conn.execute(
            "INSERT INTO signal_log (symbol, logged_at, signal_type, entry_price, "
            "stop_price, target_price, outcome_pct, worked) VALUES (?,?,?,?,?,?,?,?)",
            ("MIS", old, "UNIFIED_BUY", 100.0, 96.0, 106.0, -1.0, 0))
        conn.commit(); conn.close()
        idx = pd.date_range(datetime.now() - timedelta(days=20), periods=16, freq="D")
        # true path: fills day0, hits target 106 on day1 → WIN
        df = pd.DataFrame({"high": [101, 107, 108], "low": [99, 100, 101],
                           "close": [100, 106, 107]}, index=idx[:3])
        monkeypatch.setattr(bs, "get_ohlcv", lambda s: df)
        assert tr.reresolve_history() == 1
        conn = tr._get_conn()
        row = conn.execute("SELECT worked, outcome_pct FROM signal_log").fetchone()
        conn.close()
        assert row["worked"] == 1 and abs(row["outcome_pct"] - 6.0) < 1e-6
        # idempotent — a second pass changes nothing
        assert tr.reresolve_history() == 0

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
        # Brain gate is real + on by default; stub the posture so the money-path
        # tests stay fast + deterministic (its own test drives it explicitly).
        monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
        # fast-exit monitor spawns a real daemon thread on arm() — tests must
        # stay network-free/deterministic; its own test drives _book_tick.
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
        # tests signal price ko hi live maanein — real anchor apne dedicated
        # tests mein alag se verify hota hai
        self._real_anchor = ap._anchor_live
        monkeypatch.setattr(ap, "_anchor_live",
                            lambda sym, entry, stop, mc: (entry, ""))
        monkeypatch.setattr(sh, "sector_performance", lambda min_members=3: [
            {"sector": "Defence", "chg_1d": 1.5, "chg_5d": 3.0, "members": 4},
            {"sector": "IT / Software", "chg_1d": 0.8, "chg_5d": 1.0, "members": 5},
        ])
        ap.set_config(allocation=100000, mode="PAPER")
        return ap, te

    def _zero_costs(self, monkeypatch):
        """Neutralise slippage + charges so aggregation/verdict LOGIC tests
        stay deterministic; cost behaviour is covered by TestCostModel."""
        import execution.cost_model as cm
        monkeypatch.setattr(cm, "simulate_fill",
                            lambda price, side, is_stop=False: price)
        monkeypatch.setattr(cm, "zerodha_charges",
                            lambda *a, **k: {"stt": 0.0, "txn": 0.0,
                                             "sebi": 0.0, "stamp": 0.0,
                                             "dp": 0.0, "brokerage": 0.0,
                                             "gst": 0.0, "total": 0.0})

    def test_not_armed_never_trades(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False
        assert te.recent_trades(2) == []

    def test_gates_and_three_pct_target(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.set_config(thesis_hold=False, target_pct=3.0)  # scalp mode
        ok, _ = ap.arm()
        assert ok
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # weak sector / low score / negative edge all rejected
        assert ap.consider("X", 500, 480, 80, 0.2, "Cement", "t") is False
        assert ap.consider("HAL", 4500, 4300, 45, 0.2, "Defence", "t") is False
        assert ap.consider("HAL", 4500, 4300, 80, -0.1, "Defence", "t") is False
        # valid → placed, tagged, +3% target (scalp mode)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        t = te.recent_trades(1)[0]
        assert "AUTOPILOT" in t["note"]
        assert abs(float(t["target_price"]) - 4500 * 1.03) < 1.0
        # symbol once per day
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False

    def test_pnl_is_net_of_costs_everywhere(self, tmp_path, monkeypatch):
        """Report Card + compounding use NET P&L (slippage + Zerodha
        charges), not idealised gross — paper must not flatter itself."""
        import sqlite3
        ap, te = self._setup(tmp_path, monkeypatch)
        # WIN paper trade: entry 500, target 515, 100 sh → gross +1500
        self._insert_closed(te, "WIN1", 500, 480, 515, 100, win=True)
        rc = ap.report_card()
        t0 = rc["trades"][0]
        assert t0["gross"] == 1500                       # ideal
        assert t0["cost"] > 0                             # costs measured
        assert t0["pnl"] < 1500                           # net < gross
        assert t0["pnl"] == t0["gross"] - t0["cost"]
        assert rc["stats"]["total_costs"] > 0
        assert rc["stats"]["total_pnl"] < rc["stats"]["gross_pnl"]
        # compounding folds the NET number into the pool
        ap._account_closed_trades()
        assert 0 < ap.get_status()["realized_pnl"] < 1500

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
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        # placed_at defaults to now_ist_naive() — same IST clock the breaker's
        # day-filter uses, so this is machine-TZ-independent (C-13).
        self._insert_closed(te, "CRASH", 500, 450, 515, 100, win=False,
                            source="t")
        ap._circuit_breaker()
        st = ap.get_status()
        assert not st["armed"] and "circuit breaker" in st["disarmed_reason"]

    def test_live_arm_needs_exact_phrase(self, tmp_path, monkeypatch):
        ap, _ = self._setup(tmp_path, monkeypatch)
        ap.set_config(mode="LIVE")
        ok, _ = ap.arm("wrong phrase")
        assert not ok

    def test_live_disabled_during_overhaul(self, tmp_path, monkeypatch):
        # C-04b: LIVE arming fails closed while QT_LIVE_ENABLED is unset — even with
        # the correct phrase — but PAPER arming is unaffected.
        ap, _ = self._setup(tmp_path, monkeypatch)
        monkeypatch.delenv("QT_LIVE_ENABLED", raising=False)
        ap.set_config(mode="LIVE")
        ok, msg = ap.arm(ap.ARM_PHRASE)
        assert not ok and "DISABLED" in msg.upper()
        # paper still works
        ap.set_config(mode="PAPER")
        assert ap.arm()[0] is True
        # with the flag explicitly on, the gate opens (falls through to the phrase
        # check → a DIFFERENT refusal, proving the flag is what blocked it)
        ap.set_config(mode="LIVE")
        monkeypatch.setenv("QT_LIVE_ENABLED", "1")
        ok2, msg2 = ap.arm("wrong phrase")
        assert not ok2 and "DISABLED" not in msg2.upper()

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
                       source="scanner", placed_at=None, mode="PAPER"):
        """Insert a CLOSED trade. `placed_at` defaults to the IST storage clock
        (`now_ist_naive()`) so the write and the autopilot's IST day-filter share
        one clock — this is what keeps the day tests machine-TZ-independent (the
        C-13 fix). Boundary tests pass an explicit naive-IST timestamp instead."""
        import sqlite3
        from core.market_clock import now_ist_naive
        win_status = "PAPER_WIN" if win else "PAPER_LOSS"
        if mode == "LIVE":
            win_status = "AUTO_WIN" if win else "AUTO_LOSS"
        ts = placed_at or now_ist_naive().isoformat(timespec="seconds")
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ts, mode, symbol, qty, "MARKET", entry, stop, target, "CNC",
             win_status, f"AUTOPILOT:{source}"))
        conn.commit(); conn.close()

    def test_report_card_math_and_evidence_gate(self, tmp_path, monkeypatch):
        """P&L / R / equity curve exact; <30 trades can never claim READY."""
        ap, te = self._setup(tmp_path, monkeypatch)
        self._zero_costs(monkeypatch)
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
        self._zero_costs(monkeypatch)
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

    def test_sector_concentration_cap(self, tmp_path, monkeypatch):
        """Gate 14: ek sector mein max_per_sector se zyada open positions
        nahi — correlation cap for survival."""
        import scan.sector_heat as sh
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        ap.set_config(max_per_sector=2)
        # everything is Defence
        monkeypatch.setattr(sh, "sector_of", lambda sym: "Defence")
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        assert ap.consider("BEL", 300, 290, 80, 0.2, "Defence", "t") is True
        # third Defence name → blocked by the concentration cap
        assert ap.consider("BDL", 1200, 1150, 80, 0.2, "Defence", "t") is False
        # a different sector still gets through
        monkeypatch.setattr(sh, "sector_of",
                            lambda sym: "Defence" if sym in ("HAL", "BEL")
                            else "IT / Software")
        # (IT must be a top sector for the sector gate; it is in _setup)
        assert ap.consider("INFY", 1500, 1450, 80, 0.2, "IT / Software",
                           "t") is True

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
        self._zero_costs(monkeypatch)
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

    def test_live_anchor_rules(self, tmp_path, monkeypatch):
        """No live quote / broken setup / runaway price → NO trade."""
        import data.live_quotes as lq
        ap, _ = self._setup(tmp_path, monkeypatch)
        anchor = self._real_anchor
        # live quote hi nahi → None (EOD pe trade nahi)
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms, ttl=8.0: {})
        px, why = anchor("HAL", 4500, 4300, 1.0)
        assert px is None and "live quote nahi" in why
        # live stop ke neeche → setup toota
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 4250.0}})
        px, why = anchor("HAL", 4500, 4300, 1.0)
        assert px is None and "stop" in why
        # 1% se zyada upar → chase reject
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 4560.0}})
        px, why = anchor("HAL", 4500, 4300, 1.0)
        assert px is None and "chase" in why
        # theek range mein → live price hi entry banta hai
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 4530.0}})
        px, why = anchor("HAL", 4500, 4300, 1.0)
        assert px == 4530.0 and why == ""

    def test_order_anchored_to_live_not_signal(self, tmp_path, monkeypatch):
        """Placed trade's entry/target = LIVE price, not the signal's."""
        import data.live_quotes as lq
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.set_config(thesis_hold=False, target_pct=3.0)
        monkeypatch.setattr(ap, "_anchor_live", self._real_anchor)  # real one
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 4530.0}})
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        t = te.recent_trades(1)[0]
        assert float(t["entry_price"]) == 4530.0            # live, not 4500
        assert abs(float(t["target_price"]) - 4530 * 1.03) < 1.0

    def test_pnl_snapshot_live_and_day(self, tmp_path, monkeypatch):
        """Frontend P&L: unrealized from live quotes, day realized from
        today's closes, missing quote = None (never fake zero)."""
        import data.live_quotes as lq
        ap, te = self._setup(tmp_path, monkeypatch)
        self._zero_costs(monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 500, 450, 80, 0.2, "Defence", "t") is True
        qty = int(te.recent_trades(1)[0]["qty"])
        # closed WIN today: (515-500)*10 = +150
        self._insert_closed(te, "DONE", 500, 450, 515, 10, win=True)
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"HAL": {"price": 512.0}})
        pnl = ap.pnl_snapshot()
        assert pnl["unrealized"] == (512 - 500) * qty
        assert pnl["day_realized"] == 150 and pnl["day_closed"] == 1
        assert pnl["day_pnl"] == 150 + (512 - 500) * qty
        pos = {p["symbol"]: p for p in pnl["positions"]}
        assert pos["HAL"]["pnl"] == (512 - 500) * qty
        assert pos["HAL"]["pnl_pct"] == pytest.approx(2.4)
        # quote gayab → P&L None, zero nahi
        monkeypatch.setattr(lq, "get_live_quotes", lambda syms, ttl=8.0: {})
        pnl2 = ap.pnl_snapshot()
        assert pnl2["positions"][0]["pnl"] is None
        assert pnl2["unrealized"] == 0 and pnl2["day_realized"] == 150

    def test_fill_preserves_signal_price_once(self, tmp_path, monkeypatch):
        """Reconciled fill overwrites entry_price but the ORIGINAL signal
        price survives in signal_price — and a second fill can't clobber it."""
        import sqlite3
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        ap._ensure_exit_col()
        tid = te.recent_trades(1)[0]["id"]
        ap._apply_entry_fill(tid, 4507.35)
        ap._apply_entry_fill(tid, 4999.0)      # bogus second reconcile
        conn = sqlite3.connect(te._DB); conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM trades WHERE id=?",
                                (tid,)).fetchone())
        conn.close()
        assert float(row["signal_price"]) == 4500.0    # original, once
        assert float(row["entry_price"]) == 4999.0     # latest broker truth

    def test_execution_quality_math(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime as dt
        ap, te = self._setup(tmp_path, monkeypatch)
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        conn.commit(); conn.close()
        ap._ensure_exit_col()
        conn = sqlite3.connect(te._DB)
        # WIN: signal 100 → fill 100.5 (+0.5% entry slip), target 103,
        # actual exit 102.8 (−0.194% exit slip). qty 10.
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note, "
            "signal_price, exit_price) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (dt.now().isoformat(timespec='seconds'), "LIVE", "W", 10,
             "MARKET", 100.5, 95, 103, "CNC", "AUTO_WIN", "AUTOPILOT:t",
             100.0, 102.8))
        conn.commit(); conn.close()
        eq = ap.execution_quality()
        assert eq["n_entry"] == 1 and eq["n_exit"] == 1
        assert eq["avg_entry_slip_pct"] == pytest.approx(0.5)
        assert eq["avg_exit_slip_pct"] == pytest.approx(-0.194, abs=0.01)
        # cost = (100.5-100)*10 + (103-102.8)*10 = 5 + 2 = 7
        assert eq["slippage_cost"] == 7
        # PAPER trades kabhi ledger mein nahi aate
        self._insert_closed(te, "P", 100, 95, 103, 10, win=True)
        assert ap.execution_quality()["n_entry"] == 1

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

    def test_rsi_blowoff_and_thin_volume_gates(self, tmp_path, monkeypatch):
        """Fewer trades, higher win-rate: ignore RSI>70; volume<1× skip."""
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider(
            "HOT", 500, 480, 85, 0.2, "Defence", "scanner", rsi=82,
            volume_ratio=2.0,
        ) is False
        assert ap.consider(
            "THIN", 500, 480, 85, 0.2, "Defence", "scanner", rsi=55,
            volume_ratio=0.7,
        ) is False
        assert ap.consider(
            "OK", 500, 480, 85, 0.2, "Defence", "scanner", rsi=55,
            volume_ratio=1.5,
        ) is True
        assert te.recent_trades(1)[0]["symbol"] == "OK"

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
        ap.set_config(max_hold_days=5, thesis_hold=False)
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

    def test_reject_funnel_records_why_and_considered(self, tmp_path, monkeypatch):
        """Transparency: har reject ka reason category funnel mein girta hai,
        aur considered (denominator) sach bolta hai — 'itne kam trades kyun'."""
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # weak sector → reject; low score → reject; then a valid BUY
        assert ap.consider("X", 500, 480, 80, 0.2, "Cement", "t") is False
        assert ap.consider("HAL", 4500, 4300, 45, 0.2, "Defence", "t") is False
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        f = ap.reject_funnel()
        assert f["considered"] == 3                       # all three seen
        assert sum(f["rejects"].values()) == 2            # two rejected
        # reasons are categorised, not raw strings
        assert "sector strong nahi" in f["rejects"]
        assert "score/conviction kam" in f["rejects"]

    def test_invalid_stop_is_categorised(self, tmp_path, monkeypatch):
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4600, 80, 0.2, "Defence", "t") is False
        f = ap.reject_funnel()
        assert f["rejects"].get("invalid stop (data)") == 1

    def test_apply_preset_moves_breadth_not_safety(self, tmp_path, monkeypatch):
        """Presets sirf frequency knobs badalte hain — regime gate, conviction
        sizing, risk-per-trade jaise safety rails untouched."""
        ap, te = self._setup(tmp_path, monkeypatch)
        base = ap.get_status()
        ok, _ = ap.apply_preset("Aggressive")
        assert ok
        aggr = ap.get_status()
        assert aggr["preset"] == "Aggressive"
        assert aggr["min_score"] < base["min_score"]      # lower bar
        assert aggr["max_trades_per_day"] > base["max_trades_per_day"]
        assert aggr["max_open_positions"] >= base["max_open_positions"]
        # safety rails NOT touched by any preset
        assert aggr["risk_per_trade_pct"] == base["risk_per_trade_pct"]
        assert aggr["regime_gate"] == base["regime_gate"]
        assert aggr["daily_loss_limit_pct"] == base["daily_loss_limit_pct"]
        # Conservative tightens back
        ap.apply_preset("Conservative")
        cons = ap.get_status()
        assert cons["min_score"] > aggr["min_score"]
        assert cons["max_trades_per_day"] < aggr["max_trades_per_day"]
        # unknown preset is a no-op with a clear message
        ok2, msg2 = ap.apply_preset("Yolo")
        assert ok2 is False and "unknown" in msg2.lower()

    def test_brain_gate_pauses_new_entries_on_stand_aside(self, tmp_path,
                                                          monkeypatch):
        """Brain STAND_ASIDE (survival) blocks NEW entries; the funnel logs it;
        toggling brain_gate off restores trading. Existing positions untouched."""
        ap, te = self._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # Brain says stand aside → a normally-valid setup is rejected
        monkeypatch.setattr(ap, "_brain_posture",
                            lambda: ("STAND_ASIDE", "book DANGER"))
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is False
        assert te.recent_trades(1) == []
        f = ap.reject_funnel()
        assert f["rejects"].get("Brain STAND_ASIDE (survival)") == 1
        # same board, gate turned OFF → the trade goes through
        ap.set_config(brain_gate=False)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        # gate back ON but Brain NORMAL → trades normally
        ap.set_config(brain_gate=True)
        monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
        assert ap.consider("BEL", 300, 285, 80, 0.2, "Defence", "t") is True


# ══════════════════════════════════════════════════════════════════════════════
# 15a. Autopilot day-boundary safety (C-13) — UTC↔IST must never miscount a
#      trading day. These tests PIN the IST "today" (monkeypatching `_ist_today`)
#      so they are fully deterministic — independent of the machine timezone AND
#      of the wall-clock instant pytest runs. A trade's IST trading-day is decided
#      by `core.market_clock` alone; the circuit breaker, day-P&L and per-day
#      limits all read through it.
# ══════════════════════════════════════════════════════════════════════════════

class TestAutopilotDayBoundary:
    # naive-IST storage timestamps straddling the IST midnight of 2026-07-28
    JUST_AFTER_MIDNIGHT = "2026-07-28T00:01:00"     # IST 28th (today)
    JUST_BEFORE_MIDNIGHT = "2026-07-27T23:59:00"    # IST 27th (yesterday)

    def _pin_today(self, ap, monkeypatch, y=2026, m=7, d=28):
        """Freeze the autopilot's notion of 'today IST' — the ONLY input that
        must decide the trading day. Machine TZ / real date become irrelevant."""
        from datetime import date
        monkeypatch.setattr(ap, "_ist_today", lambda: date(y, m, d))

    # ── market_clock contract (the single source of the IST trading day) ──────
    def test_ist_day_of_contract(self):
        from core.market_clock import ist_day_of, is_ist_today
        from datetime import datetime, timezone, timedelta
        # naive-IST storage strings resolve to their own date (storage convention)
        assert ist_day_of(self.JUST_AFTER_MIDNIGHT) == "2026-07-28"
        assert ist_day_of(self.JUST_BEFORE_MIDNIGHT) == "2026-07-27"
        # a tz-aware UTC value is converted INTO IST before bucketing:
        # 2026-07-27 23:00 UTC == 2026-07-28 04:30 IST → the 28th
        utc = timezone.utc
        assert ist_day_of(datetime(2026, 7, 27, 23, 0, tzinfo=utc)) == "2026-07-28"
        # 2026-07-27 18:00 UTC == 2026-07-27 23:30 IST → still the 27th
        assert ist_day_of(datetime(2026, 7, 27, 18, 0, tzinfo=utc)) == "2026-07-27"
        # a tz-aware IST value round-trips
        ist = timezone(timedelta(hours=5, minutes=30))
        assert ist_day_of(datetime(2026, 7, 28, 0, 1, tzinfo=ist)) == "2026-07-28"
        # explicit `today` makes the comparison deterministic
        assert is_ist_today(self.JUST_AFTER_MIDNIGHT, "2026-07-28") is True
        assert is_ist_today(self.JUST_BEFORE_MIDNIGHT, "2026-07-28") is False

    # ── circuit breaker across the boundary (the money-critical control) ──────
    def test_breaker_counts_ist_today_even_when_utc_is_yesterday(
            self, tmp_path, monkeypatch):
        """A loss stamped 00:01 IST (28th) is TODAY even though its UTC instant
        is the 27th — the breaker MUST count it and fire."""
        ap, te = TestAutopilot()._setup(tmp_path, monkeypatch)
        self._pin_today(ap, monkeypatch)          # today = IST 2026-07-28
        ap.arm()
        # -5000 on a 100k pool = -5% < -3% limit
        TestAutopilot()._insert_closed(
            te, "CRASH", 500, 450, 515, 100, win=False,
            placed_at=self.JUST_AFTER_MIDNIGHT)
        ap._circuit_breaker()
        st = ap.get_status()
        assert not st["armed"] and "circuit breaker" in st["disarmed_reason"]

    def test_breaker_ignores_previous_ist_day_loss(self, tmp_path, monkeypatch):
        """Yesterday's loss (23:59 IST, 27th) must NOT bleed into today's
        breaker — a fresh IST day starts the day-loss count at zero."""
        ap, te = TestAutopilot()._setup(tmp_path, monkeypatch)
        self._pin_today(ap, monkeypatch)          # today = IST 2026-07-28
        ap.arm()
        TestAutopilot()._insert_closed(
            te, "CRASH", 500, 450, 515, 100, win=False,
            placed_at=self.JUST_BEFORE_MIDNIGHT)  # 27th, not today
        ap._circuit_breaker()
        st = ap.get_status()
        assert st["armed"] and not st["disarmed_reason"]

    # ── day-realised P&L: exactly today's closes, no double-count ─────────────
    def test_day_realized_only_ist_today_no_double_count(
            self, tmp_path, monkeypatch):
        helper = TestAutopilot()
        ap, te = helper._setup(tmp_path, monkeypatch)
        helper._zero_costs(monkeypatch)
        self._pin_today(ap, monkeypatch)
        # +150 today (00:01 IST) and +999 yesterday (23:59 IST)
        helper._insert_closed(te, "TODAY", 500, 450, 515, 10, win=True,
                              placed_at=self.JUST_AFTER_MIDNIGHT)
        helper._insert_closed(te, "YDAY", 100, 50, 1099, 1, win=True,
                              placed_at=self.JUST_BEFORE_MIDNIGHT)
        pnl = ap.pnl_snapshot()
        assert pnl["day_realized"] == 150 and pnl["day_closed"] == 1

    def test_day_realized_counts_paper_and_live_today(
            self, tmp_path, monkeypatch):
        """Both PAPER and LIVE closes on the IST day are day-realised; neither
        is double-counted and a prior-day row is excluded."""
        helper = TestAutopilot()
        ap, te = helper._setup(tmp_path, monkeypatch)
        helper._zero_costs(monkeypatch)
        self._pin_today(ap, monkeypatch)
        helper._insert_closed(te, "PAP", 500, 450, 515, 10, win=True,   # +150
                              placed_at=self.JUST_AFTER_MIDNIGHT, mode="PAPER")
        helper._insert_closed(te, "LIV", 100, 90, 110, 10, win=True,    # +100
                              placed_at="2026-07-28T09:30:00", mode="LIVE")
        helper._insert_closed(te, "OLD", 100, 50, 600, 1, win=True,     # y'day
                              placed_at=self.JUST_BEFORE_MIDNIGHT, mode="LIVE")
        pnl = ap.pnl_snapshot()
        assert pnl["day_realized"] == 250 and pnl["day_closed"] == 2

    # ── per-day trade limit resets on the IST day, not the machine day ────────
    def test_trades_per_day_limit_resets_on_ist_day(self, tmp_path, monkeypatch):
        ap, te = TestAutopilot()._setup(tmp_path, monkeypatch)
        ap.set_config(max_trades_per_day=1, max_open_positions=5, max_per_sector=5)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        self._pin_today(ap, monkeypatch, d=28)           # IST 28th
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        # second trade same IST day → blocked by the daily limit
        assert ap.consider("BEL", 300, 285, 80, 0.2, "Defence", "t") is False
        # advance to the next IST day → the daily counter resets, trading resumes
        self._pin_today(ap, monkeypatch, d=29)           # IST 29th
        assert ap.consider("BEL", 300, 285, 80, 0.2, "Defence", "t") is True


# ══════════════════════════════════════════════════════════════════════════════
# 15b. Verdict dashboard — the honest "real paisa laayak?" report card
# ══════════════════════════════════════════════════════════════════════════════

class TestMacroPulse:
    """The news radar: reads the stream for market-MOVING macro themes so the
    Brain isn't blind when the tape is news-driven. Keyword context radar,
    corroboration-gated, no trade calls."""

    def test_tariff_and_crude_spike_reads_risk_off(self):
        from core.macro_pulse import macro_pulse
        news = [
            {"headline": "US slaps fresh tariffs on Indian exports, trade war fears", "summary": ""},
            {"headline": "Tariff threat spooks IT and pharma stocks", "summary": "export duty"},
            {"headline": "Brent crude surges past $90 as OPEC cuts supply", "summary": ""},
            {"headline": "Crude oil jumps, OMCs under pressure", "summary": ""},
        ]
        p = macro_pulse(news)
        assert p["mood"] == "RISK_OFF" and p["risk_off"] is True
        names = {t["name"] for t in p["themes"]}
        assert "TARIFF" in names and "CRUDE" in names
        assert all(t["direction"] == "bearish" for t in p["themes"])

    def test_corroboration_gate_drops_single_headlines(self):
        """One headline is noise, not a market driver — a theme needs ≥2 fresh
        articles to count."""
        from core.macro_pulse import detect_macro_themes
        one = [{"headline": "Rupee falls to record low against dollar", "summary": ""},
               {"headline": "Reliance Q2 profit up 18%", "summary": "earnings"}]
        assert detect_macro_themes(one) == []      # rupee mentioned once → dropped

    def test_direction_flips_on_motion_and_de_escalation(self):
        from core.macro_pulse import macro_pulse
        # crude FALLING is bullish for India (import bill eases)
        soft = [{"headline": "Brent crude falls below $70 as demand eases", "summary": "oil drops"},
                {"headline": "Crude oil slumps on margin relief", "summary": "oil price plunge"}]
        assert macro_pulse(soft)["themes"][0]["direction"] == "bullish"
        # a trade DEAL (de-escalation) must not read bearish
        deal = [{"headline": "India US reach trade deal, tariff rollback agreed", "summary": "truce"},
                {"headline": "Tariff truce lifts export stocks", "summary": "agreement resolved"}]
        d = macro_pulse(deal)
        assert d["themes"][0]["direction"] == "bullish" and d["risk_off"] is False

    def test_calm_tape_is_neutral(self):
        from core.macro_pulse import macro_pulse
        calm = [{"headline": "TCS wins large deal", "summary": ""},
                {"headline": "HDFC Bank Q2 profit up 18%", "summary": ""}]
        p = macro_pulse(calm)
        assert p["mood"] == "NEUTRAL" and p["risk_off"] is False and p["heat"] == 0


def _isolate_brain_probes(monkeypatch):
    """Neutralize EVERY Brain probe to its fail-open default so a test machine
    with real internet (live news, bulk-deal flows, etc.) can't inject live
    directives that reorder/crowd the severity-sorted, capped-at-6 directive list.
    A test calls this first, then overrides only the probes it exercises."""
    import core.brain as brain
    for name, val in (
        ("_probe_regime", lambda: ""), ("_probe_edge", lambda cap: {}),
        ("_probe_setups", lambda m: ([], 0.0)), ("_probe_book", lambda: {}),
        ("_probe_autopilot", lambda m: {}), ("_probe_dead_daemons", lambda: []),
        ("_probe_rotation", lambda m: {}), ("_probe_correlation", lambda m: {}),
        ("_probe_breadth", lambda m: {}), ("_probe_options", lambda m: {}),
        ("_probe_flows", lambda m: {}), ("_probe_macro", lambda m: {}),
        ("_probe_drift", lambda m: []), ("_probe_calibration", lambda m: []),
        ("_probe_gates", lambda m: []), ("_probe_timeline", lambda m: []),
        ("_probe_beliefs", lambda m: []), ("_probe_rejections", lambda m: []),
    ):
        monkeypatch.setattr(brain, name, val)


class TestBrain:
    """The conductor: composes every subsystem into one posture + directives.
    Survival-first, evidence-gated, read-only."""

    def test_posture_matrix(self):
        from core.brain import decide_posture as d
        # survival first: over-risked book stands us aside despite a great tape
        assert d("TRENDING_BULL", "improving", 0.5, "DANGER", 100)[0] == "STAND_ASIDE"
        # hostile tape + negative edge → aside
        assert d("DISTRIBUTION", "stable", -0.2, "OK", 100)[0] == "STAND_ASIDE"
        # any single caution → defensive
        assert d("TRENDING_BULL", "decaying", 0.3, "OK", 100)[0] == "DEFENSIVE"
        assert d("CHOPPY", "stable", -0.05, "OK", 100)[0] == "DEFENSIVE"
        assert d("TRENDING_BULL", "improving", 0.3, "CAUTION", 100)[0] == "DEFENSIVE"
        # everything aligned AND enough evidence → lean in
        assert d("TRENDING_BULL", "improving", 0.3, "OK", 50)[0] == "AGGRESSIVE"
        # aligned but thin sample → NOT aggressive (evidence gate)
        assert d("TRENDING_BULL", "improving", 0.3, "OK", 10)[0] == "NORMAL"
        # unremarkable but fine → normal
        assert d("CHOPPY", "stable", 0.12, "OK", 50)[0] == "NORMAL"
        # macro news RISK_OFF is DEMOTE-ONLY: holds back AGGRESSIVE→NORMAL
        # (like breadth NARROW), never rescues a weak board, never forces a trade
        assert d("TRENDING_BULL", "improving", 0.3, "OK", 50,
                 macro_risk_off=True)[0] == "NORMAL"
        assert d("DISTRIBUTION", "stable", -0.2, "OK", 100,
                 macro_risk_off=True)[0] == "STAND_ASIDE"   # survival still wins

    def test_directives_priority_and_dedup(self):
        from core.brain import build_directives
        ds = build_directives({
            "book_verdict": "DANGER", "open_risk_pct": 6.1,
            "dead_daemons": ["auto_scan"], "edge_trend": "decaying",
            "recent_r": -0.1, "expectancy_r": 0.01, "regime": "DISTRIBUTION",
            "posture": "STAND_ASIDE"})
        assert ds[0]["severity"] == "critical"          # book risk first
        assert any("auto_scan" in x["text"] for x in ds)
        assert len(ds) <= 6
        # quiet board → a single reassuring info line
        calm = build_directives({"book_verdict": "OK", "regime": "CHOPPY",
                                 "edge_trend": "stable", "expectancy_r": 0.1,
                                 "posture": "NORMAL", "closed": 50})
        assert len(calm) == 1 and calm[0]["severity"] == "info"

    def test_assess_composes_probes(self, monkeypatch):
        import core.brain as brain
        _isolate_brain_probes(monkeypatch)          # no live feeds inject directives
        monkeypatch.setattr(brain, "_probe_regime", lambda: "TRENDING_BULL")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.3, "edge_trend": "improving", "closed": 60,
            "max_drawdown_pct": 8.0, "recent_avg_r": 0.35})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: (
            [{"symbol": "TOP", "verdict": "STRONG BUY", "score": 88,
              "conviction_rank": 290, "high_conviction": True,
              "entry": 101, "stop": 98, "target": 104},
             {"symbol": "MID", "verdict": "BUY", "score": 70,
              "conviction_rank": 170}], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {
            "verdict": "OK", "open_risk_pct": 1.2, "n_positions": 1})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {
            "armed": True, "trades_today": 2, "day_pnl": 450.0})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_breadth", lambda m: {"verdict": "HEALTHY"})
        a = brain.assess("IN", 100000.0)
        assert a["posture"] == "AGGRESSIVE"              # all aligned + evidence
        assert a["vitals"]["top_pick"]["symbol"] == "TOP"   # conviction-ranked
        assert a["vitals"]["n_buys"] == 2
        assert any(d["severity"] == "good" for d in a["directives"])

    def test_assess_degrades_when_subsystems_down(self, monkeypatch):
        """Every subsystem down (probes return their safe defaults) must NOT
        crash the brain — it returns a safe read, never a blank page."""
        import core.brain as brain
        monkeypatch.setattr(brain, "_probe_regime", lambda: "")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: ([], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        a = brain.assess("IN", 100000.0)
        assert a["posture"] in ("NORMAL", "DEFENSIVE", "STAND_ASIDE")
        assert a["directives"]                            # never empty
        assert a["vitals"]["regime"] == "UNKNOWN"

    def test_briefing_telegram_content(self, monkeypatch):
        import core.brain as brain
        _isolate_brain_probes(monkeypatch)          # no live feeds inject directives
        monkeypatch.setattr(brain, "_probe_regime", lambda: "TRENDING_BULL")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.3, "edge_trend": "improving", "closed": 60,
            "max_drawdown_pct": 8.0, "recent_avg_r": 0.35})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: (
            [{"symbol": "TATA", "verdict": "STRONG BUY", "score": 88,
              "conviction_rank": 290, "high_conviction": True,
              "entry": 1010, "stop": 980, "target": 1040}], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {
            "verdict": "OK", "open_risk_pct": 1.2, "n_positions": 1})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {
            "armed": True, "trades_today": 2, "day_pnl": 450.0})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_breadth", lambda m: {
            "verdict": "HEALTHY", "pct_above_50": 62.0, "line": "sehatmand"})
        monkeypatch.setattr(brain, "_probe_options", lambda m: {
            "bias": "BULLISH", "note": "PCR 1.4", "max_pain": 24400.0})
        monkeypatch.setattr(brain, "_probe_macro", lambda m: {})   # no live news
        msg = brain.briefing_telegram("IN")
        assert "QuantTerm Brain" in msg and "GREEN LIGHT" in msg
        assert "TATA" in msg and "TRENDING_BULL" in msg
        assert "HEALTHY" in msg and "62%>50DMA" in msg     # breadth on the phone
        assert "BULLISH" in msg                            # options vote on the phone
        assert "read-only" in msg                          # never implies it trades

    def test_briefing_pusher_window_dedupe_weekend(self, monkeypatch):
        import scan.auto_scan as a
        import datetime as _dt
        sent = []

        class _Eng:
            def is_configured(self): return True
            def send(self, msg, **k): sent.append(msg); return True
        monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", lambda: _Eng())
        monkeypatch.setattr("core.brain.briefing_telegram",
                            lambda market="IN": "BRIEF")

        def _clock(dtobj):
            class _DT(_dt.datetime):
                @classmethod
                def now(cls, tz=None): return dtobj
            monkeypatch.setattr(a, "datetime", _DT)

        a._brain_briefing_date = ""
        _clock(_dt.datetime(2026, 7, 13, 9, 0))            # Monday 09:00 → send
        a._maybe_push_brain_briefing()
        assert sent == ["BRIEF"]
        a._maybe_push_brain_briefing()                     # same day → deduped
        assert len(sent) == 1
        a._brain_briefing_date = ""; sent.clear()
        _clock(_dt.datetime(2026, 7, 18, 9, 0))            # Saturday → skip
        a._maybe_push_brain_briefing()
        assert sent == []
        a._brain_briefing_date = ""
        _clock(_dt.datetime(2026, 7, 13, 11, 0))           # outside window → skip
        a._maybe_push_brain_briefing()
        assert sent == []


class TestEVEngine:
    """North star: rank by expected value (measured), not points."""

    def _seed(self, tmp_path, monkeypatch, rows):
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1)
        for i, (arche, pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,"
                "entry_price,stop_price,quality_score,outcome_pct,worked,"
                "outcome_checked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "UNIFIED_BUY", arche, 100.0, 97.0, 70.0, pct,
                 worked, la))
        conn.commit(); conn.close()

    def test_ev_math_and_components(self, tmp_path, monkeypatch):
        from scan.ev_engine import signal_stats, estimate_ev
        # 30 wins @ +6% (2R), 20 losses @ -3% (1R) on a 3%-risk setup
        rows = [("SIG", 6.0, 1)] * 30 + [("SIG", -3.0, 0)] * 20
        self._seed(tmp_path, monkeypatch, rows)
        st = signal_stats()
        assert st["SIG"]["n"] == 50 and st["SIG"]["p_win"] == 0.6
        assert st["SIG"]["avg_win_r"] == 2.0 and st["SIG"]["avg_loss_r"] == 1.0
        ev = estimate_ev(["SIG"], entry=100, stop=97)
        # EV = 0.6*2R - 0.4*1R = 0.8R on 3% risk → +2.4%
        assert ev["ev_pct"] == 2.4 and ev["p_win"] == 60.0 and ev["n"] == 50

    def test_ev_gated_and_invalid_stop(self, tmp_path, monkeypatch):
        from scan.ev_engine import estimate_ev
        self._seed(tmp_path, monkeypatch, [("THIN", 6.0, 1)] * 10)   # n<30
        assert estimate_ev(["THIN"], 100, 97) is None                # no claim
        rows = [("OK", 6.0, 1)] * 40
        self._seed(tmp_path, monkeypatch, rows)
        assert estimate_ev(["OK"], 100, 105) is None                 # bad stop
        assert estimate_ev(["MISSING"], 100, 97) is None             # unknown sig

    def test_ev_rank_prefers_measured_over_points(self, tmp_path, monkeypatch):
        from scan.ev_engine import tag_ev, ev_rank_key
        self._seed(tmp_path, monkeypatch,
                   [("BREAKOUT_52W", 6.0, 1)] * 35 + [("BREAKOUT_52W", -3.0, 0)] * 15)
        rows = [
            {"symbol": "PTS", "signals": ["Unknown"], "entry": 100, "stop": 97,
             "verdict": "STRONG BUY", "conviction_rank": 290},
            {"symbol": "EV", "signals": ["52-week high breakout"], "entry": 100,
             "stop": 97, "verdict": "BUY", "conviction_rank": 150},
        ]
        tag_ev(rows)
        assert rows[1]["ev_pct"] is not None and rows[0].get("ev_pct") is None
        rows.sort(key=ev_rank_key, reverse=True)
        assert [r["symbol"] for r in rows] == ["EV", "PTS"]   # measured first


class TestPrimeFilter:
    """💎 Every data layer must pass before a setup earns the Telegram top
    slot — conviction, EV, liquidity, breadth, regime. Demote-only."""

    def _base(self):
        return {"verdict": "STRONG BUY", "categories": ["Breakout"],
                "high_conviction": True, "breakout_conviction": 72.0,
                "ev_pct": 3.1, "ev_lb_pct": 2.2, "ev_conf": "HIGH",
                "price": 500.0, "avg_vol20": 300000,
                "signals": ["52-week high breakout"]}

    def test_all_layers_pass_and_each_layer_blocks(self):
        from scan.prime_filter import prime_check
        ok, why, fail = prime_check(self._base())
        assert ok and fail == "" and any("liquid" in w for w in why)
        # each layer individually blocks
        assert not prime_check(dict(self._base(), verdict="WATCH"))[0]
        assert not prime_check(dict(self._base(), categories=["Pattern"]))[0]
        assert not prime_check(dict(self._base(), high_conviction=False,
                                    breakout_conviction=40))[0]
        assert "EV negative" in prime_check(
            dict(self._base(), ev_lb_pct=-0.5))[2]
        assert "illiquid" in prime_check(
            dict(self._base(), avg_vol20=5000))[2]
        assert "NARROW" in prime_check(
            self._base(), breadth_verdict="NARROW")[2]
        assert "leaky" in prime_check(
            self._base(), demoted_labels={"52-week high breakout"})[2]
        # no EV data + high conviction is still allowed (evidence pending)
        r = dict(self._base()); r.pop("ev_lb_pct")
        assert prime_check(r)[0]

    def test_tag_prime_counts_and_marks(self):
        from scan.prime_filter import tag_prime
        rows = [self._base(), dict(self._base(), verdict="WATCH")]
        n = tag_prime(rows)
        assert n == 1 and rows[0].get("prime") and "prime" not in rows[1]


class TestProfitBooking:
    """₹X aim + hard floor + trailing lock: PAPER auto-books, LIVE nudges
    once. 'Chhota profit pe kaat dena' structurally impossible — aim se
    neeche kabhi book nahi hota; aim cross → trail arm → peak se pullback
    pe lock (floor se kam kabhi nahi)."""

    def test_trail_stop_formula(self):
        from execution.autopilot import _trail_stop
        assert _trail_stop(peak=2000, floor=1000, giveback=300) == 1700
        assert _trail_stop(peak=1500, floor=1000, giveback=300) == 1200
        # giveback itna bada ki floor bind ho jaye (clamp up)
        assert _trail_stop(peak=1500, floor=1000, giveback=700) == 1000
        assert _trail_stop(peak=10000, floor=1000, giveback=300) == 9700

    def test_gross_for_net_math(self):
        from execution.cost_model import gross_for_net, zerodha_charges
        g = gross_for_net(1500, 1000, 40)
        ch = zerodha_charges(1000, 1000 + g / 40, 40)["total"]
        assert abs((g - ch) - 1500) < 5              # net lands on target
        assert g > 1500                              # charges ke upar
        assert gross_for_net(0, 1000, 40) == 0.0     # off → no-op
        assert gross_for_net(1500, 0, 0) == 1500     # degenerate-safe

    def test_below_aim_never_books_regardless_of_time(self, tmp_path, monkeypatch):
        """AIM se neeche — kitni bhi der chale — kabhi book nahi hota. Yahi
        'chhote profit pe kaat dena' ka structural fix hai."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ta._zero_costs(monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        ap.set_config(profit_book_rupees=1500.0)
        qty = int(te.recent_trades(1)[0]["qty"])
        for pct_net in (500, 900, 1200, 1450):    # sab AIM (1500) se neeche
            px = 4500 + pct_net / qty
            monkeypatch.setattr("data.live_quotes.get_live_quotes",
                                lambda syms, px=px: {"HAL": {"price": px}})
            ap._profit_book()
            assert te.recent_trades(1)[0]["status"] == "PAPER_OPEN"

    def test_arms_then_locks_on_pullback(self, tmp_path, monkeypatch):
        """AIM cross → arm (turant book nahi). Peak se giveback jitna pullback
        → lock. Runner ko chalne diya, phir bhi pakda."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ta._zero_costs(monkeypatch)          # exact math ke liye
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        ap.set_config(profit_book_rupees=1500.0, profit_book_min_rupees=1000.0,
                      profit_trail_giveback_rupees=300.0)
        qty = int(te.recent_trades(1)[0]["qty"])

        def _tick(net):
            px = 4500 + net / qty
            monkeypatch.setattr("data.live_quotes.get_live_quotes",
                                lambda syms, px=px: {"HAL": {"price": px}})
            ap._profit_book()

        _tick(1200)                          # AIM se neeche — chalne do
        assert te.recent_trades(1)[0]["status"] == "PAPER_OPEN"
        _tick(2000)                          # AIM cross, peak=2000, armed —
        assert te.recent_trades(1)[0]["status"] == "PAPER_OPEN"  # abhi book nahi (trail_stop=1700)
        _tick(1800)                          # neeche aaya par trail_stop(1700) se upar
        assert te.recent_trades(1)[0]["status"] == "PAPER_OPEN"
        _tick(1650)                          # trail_stop(1700) ke neeche → LOCK
        t = te.recent_trades(1)[0]
        assert t["status"] == "PAPER_WIN"
        assert "trail-book" in t["note"] and "peak" in t["note"]
        # ye 1650 hai — AIM(1500) se zyada, floor(1000) se zyada, purane
        # exact-threshold model se behtar (peak 2000 tak chalne diya)

    def test_off_by_default_and_live_nudges_once(self, tmp_path, monkeypatch):
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        # LIVE-mode trade in the journal (autopilot-tagged, open)
        te._journal({"mode": "LIVE", "symbol": "BEL", "qty": 100,
                     "entry_type": "MARKET", "entry_price": 300,
                     "stop_price": 290, "target_price": 315, "product": "CNC",
                     "status": "PLACED", "note": "AUTOPILOT:t"})
        notes = []
        monkeypatch.setattr(ap, "_notify", lambda m: notes.append(m))
        # AIM se neeche → koi notify nahi
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"BEL": {"price": 310.0}})  # +₹1000
        ap._profit_book()                       # default 0 → OFF
        assert notes == []
        ap.set_config(profit_book_rupees=1500.0)
        ap._profit_book()                       # peak=1000, armed nahi (< aim)
        assert notes == []
        # AIM cross (peak=2200) — armed, par abhi upar hi hai → notify nahi
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"BEL": {"price": 322.0}})  # +₹2200
        ap._profit_book()
        assert notes == []
        # pullback trail_stop(1900) se neeche → NOW notify
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"BEL": {"price": 318.5}})  # +₹1850
        ap._profit_book()
        ap._profit_book()                       # same day → single nudge
        assert len([n for n in notes if "BEL" in n]) == 1
        assert "book" in notes[0].lower()
        # LIVE kabhi auto-close nahi hota
        assert te.recent_trades(1)[0]["status"] == "PLACED"

    def test_stale_trail_state_pruned_on_close(self, tmp_path, monkeypatch):
        """Trade kisi aur raaste (stop/target) se band ho jaye toh uska
        trail_peaks entry agli tick pe khud saaf ho jaata hai."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ta._zero_costs(monkeypatch)
        ap.arm()
        ap.set_config(profit_book_rupees=1500.0)
        # fake stale entry jiska koi open trade nahi
        ap._state.setdefault("trail_peaks", {})["99999"] = 5000.0
        # koi open trades nahi → function turant return, state untouched
        # (prune sirf tab hota hai jab opens non-empty ho) — isliye ek asli
        # open trade bhi rakho taaki prune-path chale
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"HAL": {"price": 4500.5}})
        ap._profit_book()
        assert "99999" not in ap.get_status().get("trail_peaks", {})

    def test_fast_book_stop_geometry(self, tmp_path, monkeypatch):
        """Speed lever is GEOMETRY, not risk: confirmed A/B break + booking
        on → tight structural stop + bounded cap-relax → 2× qty, needed
        move halves-plus, and rupee-risk actually FALLS (tight stop)."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ap.set_config(max_per_sector=5)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # booking OFF — old wide 2×ATR stop, normal cap
        ap.set_config(profit_book_pct=0.0, profit_book_rupees=0.0)
        ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t", grade="A")
        t1 = te.recent_trades(1)[0]
        # booking ON + confirmed grade — tight stop, relaxed (bounded) cap
        ap.set_config(profit_book_rupees=1500.0)
        ap.consider("BEL", 1000, 960, 80, 0.2, "Defence", "t", grade="A")
        t2 = te.recent_trades(1)[0]
        assert t2["stop_price"] > t1["stop_price"]          # tightened
        assert t2["qty"] > t1["qty"]                        # bigger qty
        mv1 = 1500 / (t1["qty"] * 1000); mv2 = 1500 / (t2["qty"] * 1000)
        assert mv2 < mv1 / 1.8                              # ≥1.8× faster
        # rupee risk still inside the 1%-rule budget (tight stop pays for qty)
        assert (1000 - t2["stop_price"]) * t2["qty"] <= 100000 * 0.015 * 1.51
        # unconfirmed setup: wide stop + normal cap — discipline unchanged
        ap.consider("BHEL", 1000, 960, 80, 0.2, "Defence", "t")
        t3 = te.recent_trades(1)[0]
        assert t3["stop_price"] == 960.0 and t3["qty"] == t1["qty"]

    def test_fast_book_floor_never_overtightens(self, tmp_path, monkeypatch):
        """Already-tight stop (< 0.8% floor) is kept — noise-death guard."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        ap.set_config(profit_book_rupees=1500.0)
        ap.consider("HAL", 100, 99.5, 80, 0.2, "Defence", "t", grade="A")
        assert te.recent_trades(1)[0]["stop_price"] == 99.5

    def test_high_confidence_ev_multiplier(self):
        from execution.autopilot import _conviction_multiplier as m
        assert m(80, 0.25) == 1.5                    # old behaviour unchanged
        assert m(80, 0.25, "HIGH") == 2.0            # measured conviction → 2×
        assert m(60, 0.25, "HIGH") == 1.25           # weak score → no stretch
        assert m(80, 0.25, "MEDIUM") == 1.5          # only HIGH stretches

    def test_capital_pct_book_scales_with_quality(self, tmp_path, monkeypatch):
        """Default AIM is 3% of pool; stronger tech+fund → higher aim."""
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ta._zero_costs(monkeypatch)
        ap.set_config(profit_book_pct=3.0, profit_book_rupees=0.0,
                      profit_book_min_pct=1.5, profit_trail_giveback_pct=0.6)
        # 3% of ₹1L = ₹3000 base
        assert ap._base_book_aim_rupees() == 3000.0
        weak = ap._book_plan_for_trade(55, grade="", breakout_conviction=40)
        strong = ap._book_plan_for_trade(
            85, grade="A", breakout_conviction=80, fund_score=75,
        )
        assert weak["aim"] < 3000
        assert strong["aim"] > weak["aim"]
        assert strong["quality_mult"] > weak["quality_mult"]
        assert 0.5 <= weak["quality_mult"] <= 1.25
        assert 0.5 <= strong["quality_mult"] <= 1.25
        # Absolute ₹ still overrides pct
        ap.set_config(profit_book_rupees=1500.0)
        assert ap._base_book_aim_rupees() == 1500.0
        # Paper entry stores quality-scaled plan
        ap.set_config(profit_book_rupees=0.0, profit_book_pct=3.0)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider(
            "HAL", 4500, 4300, 85, 0.25, "Defence", "t",
            grade="A", breakout_conviction=80, fund_score=75,
        ) is True
        tid = str(te.recent_trades(1)[0]["id"])
        plan = ap.get_status()["book_plans"][tid]
        assert plan["aim"] == strong["aim"]
        # Book when NET crosses quality-scaled aim + trail
        qty = int(te.recent_trades(1)[0]["qty"])
        aim = float(plan["aim"])
        give = float(plan["giveback"])
        monkeypatch.setattr(
            "data.live_quotes.get_live_quotes",
            lambda syms: {"HAL": {"price": 4500 + (aim + give + 200) / qty}},
        )
        ap._profit_book()
        assert te.recent_trades(1)[0]["status"] == "PAPER_OPEN"
        # Pull below trail_stop = peak - giveback
        monkeypatch.setattr(
            "data.live_quotes.get_live_quotes",
            lambda syms: {"HAL": {"price": 4500 + (aim + 50) / qty}},
        )
        ap._profit_book()
        assert te.recent_trades(1)[0]["status"] == "PAPER_WIN"


class TestTelegramCommands:
    """📱 Phone commands: control anywhere, LIVE-arming never, chat-guarded."""

    def _setup(self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        import execution.trade_executor as te
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        monkeypatch.setattr(te, "_DB", tmp_path / "t.db")
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
        monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
        ap.set_config(allocation=100000, mode="PAPER")
        return ap

    def test_pause_resume_preset_book(self, tmp_path, monkeypatch):
        ap = self._setup(tmp_path, monkeypatch)
        from alerts.telegram_commands import handle_command as h
        assert "ARMED — PAPER" in h("/resume")
        assert ap.get_status()["armed"] is True
        assert "Aggressive" in h("/aggressive")
        assert ap.get_status()["max_trades_per_day"] == 10
        assert "1,500" in h("/book 1500")
        assert ap.get_status()["profit_book_rupees"] == 1500.0
        assert "3%" in h("/book 3%")
        assert ap.get_status()["profit_book_pct"] == 3.0
        assert ap.get_status()["profit_book_rupees"] == 0.0
        assert "PAUSED" in h("/pause")
        assert ap.get_status()["armed"] is False
        assert "Already OFF" in h("/pause")          # idempotent
        assert "OFF" in h("/status")

    def test_trade_now_command(self, tmp_path, monkeypatch):
        """📈 /trade: best store setup ko GATES KE SAATH place karta hai —
        force nahi. OFF → resume hint; LIVE → refuse; gates-fail → funnel."""
        ap = self._setup(tmp_path, monkeypatch)
        from alerts.telegram_commands import handle_command as h
        # OFF → pehle resume bolo
        assert "/resume" in h("/trade")
        ap.arm()
        store = [
            {"symbol": "MID", "verdict": "BUY", "score": 70, "entry": 500,
             "stop": 480, "conviction_rank": 170},
            {"symbol": "TOP", "verdict": "STRONG BUY", "score": 88,
             "entry": 1000, "stop": 960, "conviction_rank": 290,
             "prime": True, "ev_pct": 3.0, "ev_lb_pct": 2.1},
            {"symbol": "W", "verdict": "WATCH", "score": 95},
        ]
        monkeypatch.setattr("scan.auto_scan.get_results",
                            lambda: (store, 100, 0.0, "ready"))
        taken = []
        monkeypatch.setattr(ap, "consider",
                            lambda symbol, **kw: taken.append(symbol) or True)
        out = h("/trade")
        assert "TOP" in out and taken == ["TOP"]     # prime/EV-ranked first
        # sab gates fail → imaandaar funnel hint, koi force nahi
        monkeypatch.setattr(ap, "consider", lambda symbol, **kw: False)
        assert "/funnel" in h("/trade")
        # LIVE mode → refuse (paper-only invariant)
        ap.set_config(mode="LIVE")
        ap._state["armed"] = True                    # simulate armed LIVE
        assert "PAPER" in h("/trade")

    def test_help_lists_trade_command(self):
        from alerts.telegram_commands import _HELP
        assert "/trade" in _HELP and "NET" in _HELP

    def test_live_arming_always_refused(self, tmp_path, monkeypatch):
        ap = self._setup(tmp_path, monkeypatch)
        ap.set_config(mode="LIVE")
        from alerts.telegram_commands import handle_command as h
        assert "KABHI" in h("/resume")               # invariant #4 holds
        assert ap.get_status()["armed"] is False

    def test_non_commands_ignored_and_never_crash(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        from alerts.telegram_commands import handle_command as h
        assert h("hello kya haal") is None           # normal chat → silent
        assert h("") is None
        assert "Commands" in h("/help")
        assert "Commands" in h("/nonsense")          # unknown → help
        assert "/book 3%" in h("/book abc")        # bad arg → usage hint

    def test_chat_guard_blocks_strangers(self, monkeypatch):
        """Sirf configured chat-id se commands — baaki silently ignored."""
        import alerts.telegram_actions as ta
        sent = []
        monkeypatch.setattr("alerts.telegram_commands.handle_command",
                            lambda t: sent.append(t) or "ok")
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: None)
        ta._handle_message({"chat": {"id": 999}, "text": "/pause"},
                           "tok", "12345")           # stranger
        assert sent == []
        ta._handle_message({"chat": {"id": 12345}, "text": "/pause"},
                           "tok", "12345")           # owner
        assert sent == ["/pause"]

    def test_telegram_order_path_is_always_paper(self, monkeypatch):
        """Invariant #4 (C-04b verification): EVERY Telegram order path must
        force paper mode — a tap can NEVER place a live order, regardless of the
        app's configured mode or whether Kite is connected. This asserts the ONE
        order call in telegram_actions passes paper=True."""
        import alerts.telegram_actions as ta
        captured = {}

        def _spy(*a, **k):
            captured.update(k)
            return {"ok": True, "status": "PAPER", "id": 1,
                    "symbol": k.get("symbol")}

        monkeypatch.setattr("execution.trade_executor.place_trade", _spy)
        # even if the app is armed LIVE, the Telegram tap must stay paper
        import execution.autopilot as ap
        monkeypatch.setattr(ap, "set_config", lambda **kw: None, raising=False)
        msg = ta._do_paper_trade("HAL", 4500, 4300, 4600)
        assert captured.get("paper") is True
        assert "paper" in msg.lower() or "📝" in msg
        # guard against a second, un-audited order path sneaking in: the module
        # must import place_trade in exactly one place and always as paper.
        import inspect
        src = inspect.getsource(ta)
        assert src.count("place_trade(") == 1
        assert "paper=True" in src


class TestWatchdogAndBackup:
    """Operational trust: silence is suspicious; evidence is insured."""

    def test_watchdog_alerts_dead_once_per_day(self, monkeypatch):
        import core.watchdog as wd
        monkeypatch.setattr(wd, "_last_check", 0.0)
        monkeypatch.setattr(wd, "_alerted", {}, raising=False)
        wd._alerted = {}
        monkeypatch.setattr("core.health.pulse", lambda: {"daemons": {
            "auto_scan": {"status": "DEAD", "age_s": 5000, "note": ""},
            "autopilot": {"status": "OK", "age_s": 10, "note": ""}}})
        sent = []

        class _Eng:
            def is_configured(self): return True
            def send(self, m, **k): sent.append(m); return True
        monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", lambda: _Eng())
        fired = wd.check(force=True)
        assert fired == ["auto_scan"]                  # only the DEAD one
        assert len(sent) == 1 and "auto_scan" in sent[0]
        assert wd.check(force=True) == []              # same day → dedup
        # throttle: non-forced call inside window is a no-op
        wd._alerted = {}
        assert wd.check() == []

    def test_backup_snapshot_and_rotation(self, tmp_path, monkeypatch):
        import sqlite3
        import core.backup as bk
        monkeypatch.setattr(bk, "_LOGS", tmp_path)
        monkeypatch.setattr(bk, "_BACKUP_ROOT", tmp_path / "backup")
        conn = sqlite3.connect(tmp_path / "trades.db")
        conn.execute("CREATE TABLE t(x)"); conn.execute("INSERT INTO t VALUES (42)")
        conn.commit(); conn.close()
        (tmp_path / "autopilot.json").write_text('{"armed": true}')
        out = bk.snapshot("2026-07-20")
        assert set(out["copied"]) == {"trades.db", "autopilot.json"}
        # WAL-safe copy restores with data intact
        c2 = sqlite3.connect(tmp_path / "backup" / "2026-07-20" / "trades.db")
        assert c2.execute("SELECT x FROM t").fetchone() == (42,)
        c2.close()
        # rotation keeps only the newest KEEP_DAYS
        for i in range(1, 10):
            bk.snapshot(f"2026-07-{i:02d}")
        days = sorted(p.name for p in (tmp_path / "backup").iterdir())
        assert len(days) == bk.KEEP_DAYS
        assert days[-1] == "2026-07-20"                # newest survives

    def test_backup_never_raises_on_missing(self, tmp_path, monkeypatch):
        import core.backup as bk
        monkeypatch.setattr(bk, "_LOGS", tmp_path / "nowhere")
        monkeypatch.setattr(bk, "_BACKUP_ROOT", tmp_path / "b")
        out = bk.snapshot("2026-07-20")                # fresh install → no crash
        assert out["copied"] == []


class TestDailyScoreboard:
    """🎯 Machine ka purpose ek number mein: target vs NET booked. 1+1=2."""

    def test_states_and_math(self, tmp_path, monkeypatch):
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ta._zero_costs(monkeypatch)
        # OFF → target present par line bole machine band
        ap.set_config(profit_book_rupees=1500.0, max_trades_per_day=10)
        sb = ap.daily_scoreboard()
        assert sb["target"] == 15000.0 and "OFF" in sb["line"]
        # armed + booking off → purpose maango
        ap.set_config(profit_book_rupees=0.0, profit_book_pct=0.0)
        ap.arm()
        assert "/book" in ap.daily_scoreboard()["line"]
        # armed + booking on, koi trade nahi → slots ready
        ap.set_config(profit_book_rupees=1500.0)
        sb = ap.daily_scoreboard()
        assert sb["slots"] == 10 and "ready" in sb["line"]
        # ek trade lo — AIM cross (arm) phir pullback (trail-lock) → booked_net
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        qty = int(te.recent_trades(1)[0]["qty"])
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"HAL": {"price": 4500 + 2000 / qty}})
        ap._profit_book()                    # peak=2000, armed, trail_stop=1700
        assert ap.daily_scoreboard()["booked_n"] == 0   # abhi book nahi hua
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"HAL": {"price": 4500 + 1650 / qty}})
        ap._profit_book()                    # trail_stop se neeche → lock
        sb = ap.daily_scoreboard()
        assert sb["booked_n"] == 1
        assert 1500 <= sb["booked_net"] <= 1700       # NET (zero-cost stub)
        assert sb["taken"] == 1 and sb["slots"] == 9
        assert "Chal rahi hai" in sb["line"]
        assert 0 < sb["pct"] < 100

    def test_status_command_carries_scoreboard(self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        import execution.trade_executor as te
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        monkeypatch.setattr(te, "_DB", tmp_path / "t.db")
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
        monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
        ap.set_config(allocation=100000, mode="PAPER",
                      profit_book_rupees=1500.0, max_trades_per_day=10)
        from alerts.telegram_commands import handle_command as h
        out = h("/status")
        assert "🎯" in out and "₹15,000" in out       # target visible on phone


class TestFastExit:
    """⚡ Booking/exits must not wait for the 15-min scan cycle."""

    def test_tick_matrix(self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
        calls = {"book": 0, "review": 0}
        monkeypatch.setattr(ap, "_profit_book",
                            lambda: calls.__setitem__("book", calls["book"] + 1))
        monkeypatch.setattr("risk.position_manager.review_positions",
                            lambda: calls.__setitem__("review",
                                                      calls["review"] + 1))
        monkeypatch.setattr(ap, "_market_open_ist", lambda: True)
        ap.set_config(allocation=100000, mode="PAPER",
                      profit_book_rupees=1500.0)
        # not armed → idle, koi action nahi
        assert ap._book_tick() == 60 and calls == {"book": 0, "review": 0}
        ap.arm()
        # armed + market open → fast paper exits + booking dono
        ap._book_tick()
        assert calls == {"book": 1, "review": 1}
        # booking off → exits phir bhi fast (review chalta hai)
        ap.set_config(profit_book_rupees=0.0, profit_book_pct=0.0)
        ap._book_tick()
        assert calls == {"book": 1, "review": 2}
        # market closed → kuch nahi
        monkeypatch.setattr(ap, "_market_open_ist", lambda: False)
        ap._book_tick()
        assert calls == {"book": 1, "review": 2}

    def test_market_open_ist_window(self, monkeypatch):
        import execution.autopilot as ap
        import datetime as _dt
        from core.market_clock import IST

        def _fix(y, mo, d, h, mi):
            class _DT(_dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return _dt.datetime(y, mo, d, h, mi, tzinfo=IST)
            monkeypatch.setattr(ap, "datetime", _DT)

        _fix(2026, 7, 14, 10, 0)                    # Tue 10:00 IST
        assert ap._market_open_ist() is True
        _fix(2026, 7, 14, 16, 0)                    # post-close
        assert ap._market_open_ist() is False
        _fix(2026, 7, 18, 10, 0)                    # Saturday
        assert ap._market_open_ist() is False

    def test_aggressive_preset_matches_goal(self, tmp_path, monkeypatch):
        """Goal: 5-10 trades/day — Aggressive preset ka ceiling ab 10."""
        import execution.autopilot as ap
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        ap.apply_preset("Aggressive")
        s = ap.get_status()
        assert s["max_trades_per_day"] == 10
        assert s["max_open_positions"] == 5


class TestBreadth:
    """Data is gold: full-market internals from data we already compute —
    the truth BEHIND the index, at zero extra fetch."""

    def _rows(self, up, down, up50=None):
        up50 = up if up50 is None else up50
        rows = []
        for i in range(up):
            rows.append({"change_pct": 1.0, "above_sma50": i < up50,
                         "above_sma200": True})
        rows += [{"change_pct": -0.8, "above_sma50": False,
                  "above_sma200": False}] * down
        return rows

    def test_verdicts_and_math(self):
        from scan.breadth import breadth_from_results
        b = breadth_from_results(self._rows(250, 150))
        assert b["verdict"] == "HEALTHY"
        assert b["advancers"] == 250 and b["decliners"] == 150
        assert b["adv_ratio"] == round(250 / 150, 2)
        assert b["pct_above_50"] == round(250 / 400 * 100, 1)
        assert breadth_from_results(self._rows(100, 300))["verdict"] == "NARROW"
        # index-strong-but-hollow: advancers OK par 50-DMA ke neeche sab
        hollow = breadth_from_results(self._rows(250, 150, up50=50))
        assert hollow["verdict"] == "NARROW"           # p50 ~12% → kamzor

    def test_thin_sample_no_claim(self):
        from scan.breadth import breadth_from_results
        b = breadth_from_results(self._rows(100, 50))   # n=150 < 300
        assert b["verdict"] == "" and b["line"] == ""

    def test_posture_breadth_veto_demote_only(self):
        """NARROW breadth blocks AGGRESSIVE (→NORMAL); it never pushes a
        weak board up, and old callers without breadth are unchanged."""
        from core.brain import decide_posture as d
        aligned = ("TRENDING_BULL", "improving", 0.3, "OK", 50)
        assert d(*aligned, breadth="NARROW")[0] == "NORMAL"
        assert d(*aligned, breadth="HEALTHY")[0] == "AGGRESSIVE"
        assert d(*aligned)[0] == "AGGRESSIVE"          # backward compatible
        # breadth can't rescue a defensive board (demote-only)
        assert d("DISTRIBUTION", "stable", 0.1, "OK", 50,
                 breadth="HEALTHY")[0] == "DEFENSIVE"

    def test_brain_carries_breadth_and_options(self, monkeypatch):
        import core.brain as brain
        monkeypatch.setattr(brain, "_probe_regime", lambda: "TRENDING_BULL")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.3, "edge_trend": "improving", "closed": 60})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: ([], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {"verdict": "OK"})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_rotation", lambda m: {})
        monkeypatch.setattr(brain, "_probe_correlation", lambda m: {})
        monkeypatch.setattr(brain, "_probe_breadth", lambda m: {
            "verdict": "NARROW", "line": "sirf 30% above 50-DMA",
            "pct_above_50": 30.0})
        monkeypatch.setattr(brain, "_probe_options", lambda m: {
            "bias": "BEARISH", "note": "PCR 0.65 — call writers haavi",
            "max_pain": 24300.0})
        a = brain.assess("IN", 100000.0)
        assert a["posture"] == "NORMAL"                # breadth veto on lean-in
        assert a["vitals"]["breadth"] == "NARROW"
        assert any("📊" in x["text"] for x in a["directives"])
        assert any("Options positioning" in x["text"] for x in a["directives"])


class TestMarketClock:
    """Exchange time is truth: NSE gates must be IST-explicit so a UTC
    server can never shift market hours / daily limits by 5.5 hours."""

    def test_now_ist_is_aware_and_offset_correct(self):
        from core.market_clock import now_ist, today_ist, IST
        from datetime import timedelta
        n = now_ist()
        assert n.tzinfo is not None
        assert n.utcoffset() == timedelta(hours=5, minutes=30)
        assert today_ist() == n.date()
        assert IST is not None

    def test_market_hours_gate_uses_ist_not_local(self, monkeypatch):
        """Machine ka TZ kuch bhi ho — gate IST poochhta hai."""
        import scan.auto_scan as a
        import datetime as _dt
        from core.market_clock import IST

        def _fix(y, mo, d, h, mi):
            class _DT(_dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    assert tz is not None          # naive now() is the BUG
                    return _dt.datetime(y, mo, d, h, mi, tzinfo=IST)
            monkeypatch.setattr(a, "datetime", _DT)

        _fix(2026, 7, 14, 10, 30)                  # Tue 10:30 IST
        assert a._is_market_hours() is True
        _fix(2026, 7, 14, 8, 0)                    # Tue 08:00 IST — pre-open
        assert a._is_market_hours() is False
        _fix(2026, 7, 18, 10, 30)                  # Saturday
        assert a._is_market_hours() is False

    def test_autopilot_window_accepts_aware_ist(self, tmp_path, monkeypatch):
        import execution.autopilot as ap
        from core.market_clock import IST
        from datetime import datetime as dt
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        ap._state = {}
        ap.set_config(allocation=100000, mode="PAPER")   # window 09:30-14:45
        assert ap._in_window(dt(2026, 7, 14, 10, 0, tzinfo=IST)) is True
        assert ap._in_window(dt(2026, 7, 14, 9, 0, tzinfo=IST)) is False
        assert ap._in_window(dt(2026, 7, 18, 10, 0, tzinfo=IST)) is False  # Sat

    def test_daily_keys_use_ist_calendar(self, monkeypatch):
        """trades_today / funnel keys IST date se banti hain — UTC date se
        subah 5:30 pe limits reset ka bug ab impossible."""
        import execution.autopilot as ap
        from core.market_clock import today_ist
        assert ap._ist_today() == today_ist()


class TestEcoMode:
    """Shared-machine thermal relief: less CPU, identical trading logic."""

    def test_defaults_off_full_power(self, monkeypatch):
        import core.eco as eco
        monkeypatch.delenv("QT_ECO", raising=False)
        monkeypatch.delenv("QT_LOW_POWER", raising=False)
        assert eco.eco_on() is False
        assert eco.workers(8) == 8                      # untouched
        assert eco.scan_interval(900) == 900
        assert eco.should_scan_now(False) is True       # old behaviour

    def test_eco_clamps_and_skips_offhours(self, monkeypatch):
        import core.eco as eco
        monkeypatch.delenv("QT_LOW_POWER", raising=False)
        monkeypatch.setenv("QT_ECO", "1")
        assert eco.eco_on() is True
        assert eco.workers(8) == 2 and eco.workers(6) == 2
        assert eco.workers(1) == 1                      # never raises
        assert eco.scan_interval(900) == 1800           # 30-min floor
        assert eco.scan_interval(3600) == 3600          # floor, not cap
        assert eco.should_scan_now(True) is True        # market open → scan
        assert eco.should_scan_now(False) is False      # off-hours → thanda


class TestDecisionJournal:
    """Evidence infrastructure: every decision — even rejections — becomes a
    resolved prediction, so gates and probabilities get audited."""

    def _setup(self, tmp_path, monkeypatch):
        import core.decision_journal as dj
        monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "dec.db"))
        return dj

    def test_log_dedupe_and_no_fake_reference(self, tmp_path, monkeypatch):
        dj = self._setup(tmp_path, monkeypatch)
        dj.log_decision("HAL", "TAKEN", "", "scanner", 4500, 4300, 80,
                        ev_pct=3.2, p_win=65.0, confidence="MEDIUM")
        dj.log_decision("HAL", "TAKEN", "", "scanner", 4500, 4300, 80)
        dj.log_decision("HAL", "REJECTED", "sector", "scanner", 4500, 4300, 80)
        dj.log_decision("ZERO", "TAKEN", "", "scanner", 0, 0, 80)   # no price
        c = dj._conn()
        n = c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        c.close()
        assert n == 2            # dedup per symbol×day×decision; ZERO skipped

    def test_scanner_logs_decisions_for_calibration(self, tmp_path, monkeypatch):
        # the scan path must journal verdicts + predictions so calibration works
        # for MANUAL users too (not only when autopilot is armed)
        dj = self._setup(tmp_path, monkeypatch)
        import scan.auto_scan as a
        serialized = [
            {"symbol": "AAA", "verdict": "BUY", "entry": 100, "stop": 96,
             "score": 80, "ev_pct": 1.2, "p_win": 64.0, "confidence": "HIGH",
             "reasons": ["breakout"]},
            {"symbol": "BBB", "verdict": "WATCH", "entry": 50, "stop": 48,
             "score": 40, "reasons": ["too extended"]},
            {"symbol": "CCC", "verdict": "WATCH", "entry": 0, "reasons": []},
        ]
        a._log_decisions_for_calibration(serialized)
        c = dj._conn()
        got = {r["symbol"]: (r["decision"], r["p_win"]) for r in
               c.execute("SELECT symbol, decision, p_win FROM decisions")}
        c.close()
        assert got["AAA"] == ("TAKEN", 64.0)        # buy, with its prediction
        assert got["BBB"][0] == "REJECTED"          # watch → rejected decision
        assert "CCC" not in got                     # no entry price → no claim

    def test_outcomes_and_gate_audit(self, tmp_path, monkeypatch):
        dj = self._setup(tmp_path, monkeypatch)
        dj.log_decision("HAL", "TAKEN", "", "scanner", 4500, 4300, 80,
                        ev_pct=3.2, p_win=65.0)
        dj.log_decision("X", "REJECTED", "sector strong nahi", "scanner",
                        500, 480, 70, ev_pct=2.0, p_win=62.0)
        c = dj._conn()
        c.execute("UPDATE decisions SET decided_at='2026-07-01T10:00:00'")
        c.commit(); c.close()
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {"HAL": {"price": 4700.0},
                                          "X": {"price": 490.0}})
        assert dj.update_outcomes() == 2
        rep = dj.decision_report(min_n=1)
        assert rep["taken"]["n"] == 1 and rep["taken"]["win_rate"] == 100.0
        assert rep["rejected"]["avg_outcome_pct"] == -2.0
        assert "Gates kaam kar rahe" in rep["verdict"]     # gates earned money
        assert "sector strong nahi" in rep["by_reason"]
        cal = dj.calibration_report(min_n=1)
        scored = [b for b in cal["buckets"] if b["predicted"] is not None]
        assert scored and scored[0]["n"] == 2              # both predicted 60s

    def test_missing_quote_stays_pending(self, tmp_path, monkeypatch):
        """No price → no outcome written (no fake data), row stays pending."""
        dj = self._setup(tmp_path, monkeypatch)
        dj.log_decision("HAL", "TAKEN", "", "scanner", 4500, 4300, 80)
        c = dj._conn()
        c.execute("UPDATE decisions SET decided_at='2026-07-01T10:00:00'")
        c.commit(); c.close()
        monkeypatch.setattr("data.live_quotes.get_live_quotes",
                            lambda syms: {})
        assert dj.update_outcomes() == 0
        assert dj.decision_report()["taken"]["n"] == 0     # still unresolved

    def test_autopilot_journals_rejections_with_prediction(self, tmp_path,
                                                           monkeypatch):
        """A gate rejection lands in the journal with the EV prediction."""
        import execution.autopilot as ap
        import execution.trade_executor as te
        import scan.sector_heat as sh
        import core.decision_journal as dj
        monkeypatch.setattr(dj, "_DB_PATH", str(tmp_path / "dec.db"))
        monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
        monkeypatch.setattr(te, "_DB", tmp_path / "t.db")
        ap._state = {}
        monkeypatch.setattr(ap, "_notify", lambda m: None)
        monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
        monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
        monkeypatch.setattr(sh, "sector_performance", lambda min_members=3: [
            {"sector": "Defence", "chg_1d": 1.5, "chg_5d": 3.0, "members": 4}])
        ap.set_config(allocation=100000, mode="PAPER")
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # weak sector → rejected, with its prediction journaled
        ap.consider("X", 500, 480, 80, 0.2, "Cement", "scanner",
                    ev_pct=2.5, p_win=61.0, ev_conf="MEDIUM")
        c = dj._conn()
        row = c.execute("SELECT * FROM decisions").fetchone()
        c.close()
        assert row["decision"] == "REJECTED"
        assert row["reason"] == "sector strong nahi"       # categorised
        assert row["p_win"] == 61.0 and row["ev_pct"] == 2.5


class TestSimLab:
    """Thousands of futures before one live change — and earned capital."""

    def test_simulation_math_and_risk_tradeoff(self):
        from core.sim_lab import simulate
        rs = [2.0] * 40 + [-1.0] * 60                     # +0.2R edge
        lo = simulate(rs, 0.01, seed=7)
        hi = simulate(rs, 0.02, seed=7)
        assert lo["median_growth_pct"] > 0                 # edge compounds
        assert hi["median_growth_pct"] > lo["median_growth_pct"]
        assert hi["p95_max_dd_pct"] > lo["p95_max_dd_pct"]  # risk shows up
        assert hi["prob_dd20_pct"] > lo["prob_dd20_pct"]
        # deterministic under a seed
        assert simulate(rs, 0.01, seed=7) == lo

    def test_thin_evidence_refused(self):
        from core.sim_lab import simulate, compare
        assert simulate([1.0] * 10, 0.01) is None          # <30 = fiction
        assert compare([1.0] * 10, 0.01, 0.02) is None

    def test_scaling_advice_thresholds(self):
        from core.sim_lab import scaling_advice
        assert scaling_advice(1.9, 4.0, 250)["action"] == "INCREASE"
        assert scaling_advice(1.1, 8.0, 120)["action"] == "REDUCE"
        assert scaling_advice(1.5, 6.0, 120)["action"] == "HOLD"
        # evidence gate: great numbers on a thin sample still HOLD
        assert scaling_advice(2.5, 1.0, 30)["action"] == "HOLD"

    def test_rolling_expectancy_windows(self, tmp_path, monkeypatch):
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1)
        # first 50 winners (+2R), then 50 losers (-1R): last-50 window sinks
        rows = [("SIG", 6.0, 1)] * 50 + [("SIG", -3.0, 0)] * 50
        for i, (arche, pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,"
                "entry_price,stop_price,quality_score,outcome_pct,worked,"
                "outcome_checked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "UNIFIED_BUY", arche, 100.0, 97.0, 70.0, pct,
                 worked, la))
        conn.commit(); conn.close()
        from scan.live_edge import rolling_expectancy
        roll = rolling_expectancy()
        assert roll[50] == -1.0                            # recent window dead
        assert roll[100] == 0.5                            # lifetime still ok
        assert 250 not in roll                             # not enough data


class TestInstitutionalFlows:
    """🏦 FII/DII + bulk deals — bade paise ka footprint, NSE se free.
    Parsers pure; fetch-fail = no claim; context-only (gate nahi)."""

    def test_fii_dii_bias_matrix(self):
        from data.institutional_flows import parse_fii_dii
        def rows(fii, dii):
            return [{"category": "FII/FPI *", "netValue": str(fii),
                     "date": "22-Jul-2026"},
                    {"category": "DII **", "netValue": str(dii)}]
        assert parse_fii_dii(rows(2400, 1100))["bias"] == "SUPPORTIVE"
        r = parse_fii_dii(rows(-3200, 2100))
        assert r["bias"] == "FII_SELLING_DII_ABSORBING"
        assert "absorb" in r["note"]
        assert parse_fii_dii(rows(-1500, -900))["bias"] == "DISTRIBUTION"
        assert parse_fii_dii(rows(1500, -900))["bias"] == "MIXED"
        # comma-formatted NSE numbers parse hote hain
        assert parse_fii_dii([
            {"category": "FII/FPI", "netValue": "-2,412.35"},
            {"category": "DII", "netValue": "1,987.10"},
        ])["fii_net_cr"] == -2412.0
        assert parse_fii_dii([]) is None                 # no claim
        assert parse_fii_dii([{"category": "FII", "netValue": "5"}]) is None

    def test_fii_dii_store_parse_and_persist(self):
        from data.fii_dii_store import parse_trade_react_rows, reset_store, upsert_rows, summarize, get_history

        reset_store()
        rows = parse_trade_react_rows([
            {
                "date": "01-Aug-2026",
                "fiiBuyValue": "12,000",
                "fiiSellValue": "10,500",
                "diiBuyValue": "8,000",
                "diiSellValue": "7,200",
            },
            {
                "date": "31-Jul-2026",
                "category": "FII/FPI",
                "buyValue": "9,000",
                "sellValue": "9,500",
            },
            {
                "date": "31-Jul-2026",
                "category": "DII",
                "buyValue": "6,000",
                "sellValue": "5,500",
            },
        ])
        assert len(rows) == 2
        assert upsert_rows(rows) == 2
        hist = get_history(30)
        assert len(hist) == 2
        summary = summarize(30)
        assert summary["available"]
        assert summary["totals"]["fii_net_cr"] != 0
        reset_store()

    def test_market_brief_offline(self):
        from reporting.market_brief import build_institutional_market_brief

        brief = build_institutional_market_brief(days=30, symbol_limit=2)
        assert brief["report_type"] == "INSTITUTIONAL_MARKET_BRIEF"
        assert "narrative" in brief

    def test_fii_dii_lazy_refresh_mock(self):
        from data.fii_dii_store import reset_store, refresh_if_needed, count_rows

        reset_store()
        payload = refresh_if_needed(
            force=True,
            fetcher=lambda: [
                {
                    "date": "2026-08-01",
                    "fii_buy": 100.0,
                    "fii_sell": 90.0,
                    "fii_net": 10.0,
                    "dii_buy": 50.0,
                    "dii_sell": 40.0,
                    "dii_net": 10.0,
                }
            ],
        )
        assert payload.get("fetched")
        assert count_rows() == 1
        again = refresh_if_needed(fetcher=lambda: [])
        assert again.get("fetched") is False
        assert again.get("reason") == "fresh"
        reset_store()

    def test_fii_derivative_stats_fail_closed_without_legacy_endpoint(self):
        from data.fii_dii import (
            _reset_derivative_stats_cache_for_tests,
            get_fii_derivative_stats_uncached,
        )

        _reset_derivative_stats_cache_for_tests()
        stats = get_fii_derivative_stats_uncached()
        assert stats["available"] is False
        assert stats["total_net"] is None
        assert stats["index_futures_net"] is None
        assert stats.get("note")

    def test_fii_derivative_stats_parse_rows(self):
        from data.fii_dii import _parse_derivative_stats_rows

        parsed = _parse_derivative_stats_rows([
            {"category": "Index Futures", "netAmount": "100"},
            {"category": "Index Options", "netAmount": "-40"},
            {"category": "Stock Futures", "buyAmount": "50", "sellAmount": "30"},
        ])
        assert parsed["available"] is True
        assert parsed["index_futures_net"] == 100.0
        assert parsed["index_options_net"] == -40.0
        assert parsed["stock_futures_net"] == 20.0
        assert parsed["total_net"] == 80.0

    def test_lazy_fundamentals_ensure_mock(self):
        from fundamentals.cache import FundamentalsCache
        from fundamentals import fetcher as fetcher_mod
        from fundamentals.lazy import ensure_deep_fundamentals

        sym = "LAZYTEST"
        FundamentalsCache().invalidate(sym)

        class _MockScraper:
            def fetch_all(self, symbol: str) -> dict:
                return {"about": "mock company", "quarterly_results": [{"": "Q1"}]}

        original = fetcher_mod._scraper
        fetcher_mod._scraper = _MockScraper()
        try:
            data = ensure_deep_fundamentals(sym, force_refresh=False)
            assert data.get("about") == "mock company"
            assert FundamentalsCache().has(sym)
        finally:
            fetcher_mod._scraper = original
            FundamentalsCache().invalidate(sym)

    def test_lazy_fundamentals_cache_hit(self):
        from fundamentals.cache import FundamentalsCache
        from fundamentals.lazy import ensure_deep_fundamentals

        sym = "LAZYHIT"
        cache = FundamentalsCache()
        cache.invalidate(sym)
        cache.set(sym, {"about": "test co", "quarterly_results": []})
        hit = ensure_deep_fundamentals(sym, force_refresh=False)
        assert hit.get("about") == "test co"
        cache.invalidate(sym)

    def test_bulk_deals_and_net_buys(self):
        from data.institutional_flows import parse_bulk_deals, bulk_buy_symbols
        deals = parse_bulk_deals({"data": [
            {"BD_SYMBOL": "HAL", "BD_BUY_SELL": "BUY",
             "BD_QTY_TRD": "5,00,000".replace(",", ""), "BD_TP_WATP": "4500",
             "BD_CLIENT_NAME": "BIG FUND LLP"},
            {"BD_SYMBOL": "HAL", "BD_BUY_SELL": "SELL", "BD_QTY_TRD": "100000",
             "BD_TP_WATP": "4501"},
            {"BD_SYMBOL": "TRAP", "BD_BUY_SELL": "SELL", "BD_QTY_TRD": "900000",
             "BD_TP_WATP": "50"},
            {"BD_SYMBOL": "", "BD_BUY_SELL": "BUY", "BD_QTY_TRD": "1"},  # junk
        ]})
        assert len(deals) == 3
        buys = bulk_buy_symbols(deals)
        assert buys == {"HAL"}               # net buy sirf HAL; TRAP net sell

    def test_brain_carries_flows(self, monkeypatch):
        import core.brain as brain
        for p in ("_probe_regime",):
            monkeypatch.setattr(brain, p, lambda: "CHOPPY")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.1, "edge_trend": "stable", "closed": 60})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: ([], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {"verdict": "OK"})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_rotation", lambda m: {})
        monkeypatch.setattr(brain, "_probe_correlation", lambda m: {})
        monkeypatch.setattr(brain, "_probe_breadth", lambda m: {})
        monkeypatch.setattr(brain, "_probe_options", lambda m: {})
        monkeypatch.setattr(brain, "_probe_flows", lambda m: {
            "fii_dii": {"bias": "DISTRIBUTION",
                        "note": "FII -2400cr AUR DII -900cr dono bech rahe"},
            "bulk_buys": ["HAL", "BEL"]})
        a = brain.assess("IN", 100000.0)
        warns = [d for d in a["directives"] if d["severity"] == "warn"]
        assert any("dono bech" in d["text"] for d in warns)     # distribution=warn
        assert any("HAL, BEL" in d["text"] for d in a["directives"])


class TestSymbolMemory:
    """🧠 MOAT piece: har stock ka charitra yaad — serial false-breakers
    dobara entry nahi paate. Firms per-instrument models rakhti hain;
    retail nahi — ab hum rakhte hain, apne hi outcomes se."""

    def _seed(self, tmp_path, monkeypatch, rows):
        """rows = (symbol, outcome_pct, worked); entry 100 / stop 97."""
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1)
        for i, (sym, pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,"
                "entry_price,stop_price,quality_score,outcome_pct,worked,"
                "outcome_checked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sym, la, "UNIFIED_BUY", "BREAKOUT_52W", 100.0, 97.0, 70.0,
                 pct, worked, la))
        conn.commit(); conn.close()

    def test_symbol_edge_and_serial_losers(self, tmp_path, monkeypatch):
        from scan.live_edge import symbol_edge, serial_losers
        rows = ([("TRAP", -3.0, 0)] * 6            # 6 baar kata → -1R each
                + [("GOOD", 6.0, 1)] * 6           # respectful breakouts
                + [("THIN", -3.0, 0)] * 3)         # sirf 3 outcomes → no claim
        self._seed(tmp_path, monkeypatch, rows)
        se = symbol_edge()
        assert se["TRAP"]["expectancy_r"] == -1.0 and se["TRAP"]["n"] == 6
        assert se["GOOD"]["expectancy_r"] == 2.0
        assert "THIN" not in se                    # min_n gate
        losers = serial_losers()
        assert losers == {"TRAP"}                  # sirf proven repeat-offender

    def test_autopilot_blocks_serial_loser(self, tmp_path, monkeypatch):
        ta = TestAutopilot()
        ap, te = ta._setup(tmp_path, monkeypatch)
        ap.arm()
        monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
        # is test mein asli memory chahiye — setup-stub ko override karo
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: {"TRAP"})
        assert ap.consider("TRAP", 500, 480, 90, 0.3, "Defence", "t") is False
        f = ap.reject_funnel()
        assert f["rejects"].get("symbol memory (serial false-breaker)") == 1
        # doosra naam waise hi chalta hai
        assert ap.consider("HAL", 4500, 4300, 80, 0.2, "Defence", "t") is True
        # toggle off → memory bhoolo (user ka akhri faisla)
        ap.set_config(symbol_memory_gate=False)
        assert ap.consider("TRAP", 500, 480, 90, 0.3, "Defence", "t") is True


class TestBayesianConfidence:
    """Trust 520 trades @67% more than 52 @68% — shrink small samples."""

    def test_wilson_bound_trusts_large_samples(self):
        from scan.ev_engine import wilson_lb, confidence_tier
        small = wilson_lb(0.68, 52)
        large = wilson_lb(0.67, 520)
        assert large > small                     # the CTO's exact example
        assert confidence_tier(0.67, 520) == "HIGH"
        assert confidence_tier(0.68, 52) == "MEDIUM"
        assert confidence_tier(0.5, 0) == "LOW"
        assert wilson_lb(0.9, 0) == 0.0          # degenerate-safe

    def test_ranking_uses_conservative_ev(self):
        """A lucky thin sample must NOT outrank a deep workhorse."""
        from scan.ev_engine import ev_rank_key
        lucky = {"ev_pct": 4.2, "ev_lb_pct": 1.1, "conviction_rank": 200}
        work = {"ev_pct": 3.5, "ev_lb_pct": 3.1, "conviction_rank": 150}
        rows = [lucky, work]
        rows.sort(key=ev_rank_key, reverse=True)
        assert rows[0] is work                   # shrunk EV decides

    def test_estimate_ev_carries_lb_and_confidence(self, tmp_path, monkeypatch):
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1)
        rows = [("SIG", 6.0, 1)] * 30 + [("SIG", -3.0, 0)] * 20
        for i, (arche, pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,"
                "entry_price,stop_price,quality_score,outcome_pct,worked,"
                "outcome_checked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "UNIFIED_BUY", arche, 100.0, 97.0, 70.0, pct,
                 worked, la))
        conn.commit(); conn.close()
        from scan.ev_engine import estimate_ev
        ev = estimate_ev(["SIG"], 100, 97)
        assert ev["ev_lb_pct"] < ev["ev_pct"]    # conservative < point estimate
        assert ev["confidence"] in ("HIGH", "MEDIUM", "LOW")


class TestCorrelationEngine:
    """4 correlated positions = 1 real bet — the risk the sector cap misses."""

    def test_clusters_union_find_transitive(self):
        from risk.correlation import clusters_from_corr
        syms = ["HAL", "BEL", "BHEL", "LT", "INFY"]
        corr = {("BEL", "HAL"): 0.82, ("BHEL", "HAL"): 0.75,
                ("BHEL", "LT"): 0.71, ("BEL", "INFY"): 0.20}
        cl = clusters_from_corr(syms, corr)
        # HAL-BEL-BHEL-LT chain-merge (transitively) into one macro bet
        assert cl[0] == ["BEL", "BHEL", "HAL", "LT"]
        assert ["INFY"] in cl and len(cl) == 2

    def test_threshold_respected_and_empty_safe(self):
        from risk.correlation import clusters_from_corr
        # below threshold → everyone independent
        cl = clusters_from_corr(["A", "B"], {("A", "B"): 0.5})
        assert len(cl) == 2
        assert clusters_from_corr([], {}) == []
        # unknown symbols in corr ignored, no KeyError
        cl2 = clusters_from_corr(["A"], {("X", "Y"): 0.9})
        assert cl2 == [["A"]]

    def test_brain_warns_on_hidden_concentration(self, monkeypatch):
        import core.brain as brain
        monkeypatch.setattr(brain, "_probe_regime", lambda: "CHOPPY")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.12, "edge_trend": "stable", "closed": 60})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: ([], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {"verdict": "OK"})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_rotation", lambda m: {})
        monkeypatch.setattr(brain, "_probe_correlation", lambda m: {
            "n_positions": 4, "n_bets": 1,
            "clusters": [["BEL", "BHEL", "HAL", "LT"]],
            "biggest": ["BEL", "BHEL", "HAL", "LT"]})
        a = brain.assess("IN", 100000.0)
        warn = [d for d in a["directives"] if "asli bets" in d["text"]]
        assert warn and warn[0]["severity"] == "warn"
        assert "BEL/BHEL/HAL/LT" in warn[0]["text"]


class TestPortfolioIntel:
    """Capital as a portfolio: every holding competes daily vs best ideas."""

    def test_rotation_advice_and_churn_gate(self):
        from core.portfolio_intel import rotation_advice
        h = [{"symbol": "HAL", "ev_pct": 1.2}, {"symbol": "BEL", "ev_pct": 4.0}]
        c = [{"symbol": "TATVA", "ev_pct": 5.4, "verdict": "STRONG BUY"},
             {"symbol": "JUNK", "ev_pct": 9.0, "verdict": "WATCH"}]
        out = rotation_advice(h, c)
        assert out["portfolio_ev_pct"] == 2.6
        assert out["weakest"]["symbol"] == "HAL"
        assert out["swap"]["out"] == "HAL" and out["swap"]["in"] == "TATVA"
        assert out["swap"]["gap_pct"] == 4.2               # WATCH junk excluded
        assert "advice hai, order nahi" in out["swap"]["note"]
        # below the churn threshold → no swap advice
        assert rotation_advice(
            h, [{"symbol": "X", "ev_pct": 2.0, "verdict": "BUY"}])["swap"] is None

    def test_empty_book_or_candidates_safe(self):
        from core.portfolio_intel import rotation_advice
        c = [{"symbol": "T", "ev_pct": 5.0, "verdict": "BUY"}]
        assert rotation_advice([], c)["swap"] is None
        assert rotation_advice([], [])["portfolio_ev_pct"] is None
        # holdings without EV claims: no punishment swap invented
        h = [{"symbol": "A", "ev_pct": None}]
        assert rotation_advice(h, c)["swap"] is None

    def test_brain_carries_rotation_directive(self, monkeypatch):
        import core.brain as brain
        _isolate_brain_probes(monkeypatch)          # no live feeds crowd the list
        monkeypatch.setattr(brain, "_probe_regime", lambda: "CHOPPY")
        monkeypatch.setattr(brain, "_probe_edge", lambda cap: {
            "expectancy_r": 0.12, "edge_trend": "stable", "closed": 60})
        monkeypatch.setattr(brain, "_probe_setups", lambda m: ([], 0.0))
        monkeypatch.setattr(brain, "_probe_book", lambda: {"verdict": "OK"})
        monkeypatch.setattr(brain, "_probe_autopilot", lambda m: {})
        monkeypatch.setattr(brain, "_probe_dead_daemons", lambda: [])
        monkeypatch.setattr(brain, "_probe_rotation", lambda m: {
            "portfolio_ev_pct": 2.1,
            "swap": {"out": "HAL", "out_ev": 1.2, "in": "TATVA", "in_ev": 5.4,
                     "gap_pct": 4.2, "note": "HAL vs TATVA — advice hai, order nahi."}})
        a = brain.assess("IN", 100000.0)
        assert any("Opportunity cost" in d["text"] for d in a["directives"])
        assert a["vitals"]["portfolio_ev_pct"] == 2.1


class TestOptionsVerdict:
    """Raw chain metrics → one clear structural read (bias + range + IV)."""

    def test_bullish_bearish_range(self):
        from options.verdict import options_verdict
        # high PCR + spot below max pain → bullish
        b = options_verdict(spot=24180, pcr=1.4, max_pain=24400,
                            support=24000, resistance=24500, iv_pct=25)
        assert b["bias"] == "BULLISH"
        assert b["max_pain"]["pull"] == "up"
        assert b["iv"]["stance"] == "cheap"
        # low PCR + spot above max pain → bearish
        r = options_verdict(spot=24600, pcr=0.6, max_pain=24300,
                            support=24000, resistance=24500, iv_pct=82)
        assert r["bias"] == "BEARISH"
        assert r["iv"]["stance"] == "expensive"
        # neutral PCR + tight walls + spot at max pain → range
        g = options_verdict(spot=24250, pcr=1.0, max_pain=24250,
                            support=24200, resistance=24400, iv_pct=55)
        assert g["bias"] == "RANGE"

    def test_range_geometry_and_position(self):
        from options.verdict import options_verdict
        v = options_verdict(spot=24050, pcr=1.0, max_pain=24250,
                            support=24000, resistance=24500, iv_pct=50)
        assert v["range"]["support"] == 24000 and v["range"]["resistance"] == 24500
        assert 0.0 <= v["range"]["position"] <= 0.2   # spot near support
        assert any("support wall" in n for n in v["notes"])

    def test_insufficient_data_is_safe(self):
        from options.verdict import options_verdict
        v = options_verdict(spot=0, pcr=1.0, max_pain=0, support=0,
                            resistance=0, iv_pct=0)
        assert v["bias"] == "NEUTRAL" and "adhoori" in v["verdict_line"]
        # inverted walls (bad data) also degrade safely
        v2 = options_verdict(spot=100, pcr=1.0, max_pain=100, support=110,
                             resistance=90, iv_pct=50)
        assert v2["bias"] == "NEUTRAL"


class TestPulseCockpit:
    """The Daily Pulse cockpit surfaces setups + book + autopilot so the user
    lives on three tabs. These render helpers must not crash on the real store
    shape (wrong keys = a blank/So broken daily page)."""

    class _FakeCol:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _FakeSt:
        def __init__(self): self.session_state = {}
        def markdown(self, *a, **k): pass
        def caption(self, *a, **k): pass
        def success(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def button(self, *a, **k): return False        # never auto-fires paper
        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [TestPulseCockpit._FakeCol() for _ in range(n)]

    def test_today_setups_picks_buys_conviction_ranked(self, monkeypatch):
        import ui.street_pulse_page as sp
        fake = self._FakeSt()
        monkeypatch.setattr(sp, "st", fake)
        captured = {}
        # capture which symbols get a paper button (i.e. are rendered as setups)
        orig_button = fake.button
        def _btn(label, *a, **k):
            key = k.get("key", "")
            if key.startswith("pulse_paper_"):
                captured.setdefault("syms", []).append(key.split("_")[-1])
            return orig_button(label, *a, **k)
        fake.button = _btn
        results = [
            {"symbol": "AAA", "verdict": "WATCH", "score": 90, "price": 100,
             "entry": 101, "stop": 98, "target": 104},
            {"symbol": "BBB", "verdict": "BUY", "score": 70, "price": 100,
             "conviction_rank": 170, "entry": 101, "stop": 98, "target": 104},
            {"symbol": "CCC", "verdict": "STRONG BUY", "score": 80, "price": 100,
             "conviction_rank": 280, "entry": 101, "stop": 98, "target": 104},
        ]
        monkeypatch.setattr("scan.auto_scan.get_results",
                            lambda: (results, 100, 0.0, "ready"))
        sp._render_today_best_setups()               # must not raise
        # WATCH excluded; STRONG BUY ranked above BUY
        assert captured.get("syms") == ["CCC", "BBB"]

    def test_brain_hero_renders(self, monkeypatch):
        import ui.street_pulse_page as sp
        monkeypatch.setattr(sp, "st", self._FakeSt())
        monkeypatch.setattr("core.brain.assess", lambda market, capital: {
            "posture": "AGGRESSIVE", "posture_reason": "all good",
            "verdict_line": "green", "directives": [
                {"severity": "good", "text": "go"}],
            "vitals": {"cur": "₹", "regime": "TRENDING_BULL", "expectancy_r": 0.3,
                       "edge_trend": "improving", "drawdown_pct": 8.0,
                       "n_buys": 3, "n_high_conviction": 1, "book_verdict": "OK",
                       "open_risk_pct": 1.2, "autopilot_armed": True,
                       "day_pnl": 400.0}})
        sp._render_brain("IN")                           # must not raise

    def test_glance_and_book_do_not_crash(self, monkeypatch):
        import ui.street_pulse_page as sp
        monkeypatch.setattr(sp, "st", self._FakeSt())
        monkeypatch.setattr("execution.autopilot.get_status",
                            lambda: {"armed": True, "mode": "PAPER",
                                     "trades_today_count": 1,
                                     "max_trades_per_day": 4})
        monkeypatch.setattr("execution.autopilot.pnl_snapshot",
                            lambda: {"day_pnl": 250.0, "positions": [{}]})
        sp._render_autopilot_glance("IN")            # must not raise
        # US path: no pnl_snapshot exposed → glance still renders
        monkeypatch.setattr("execution.us_autopilot.get_status",
                            lambda: {"armed": False, "trades_today_count": 0,
                                     "max_trades_per_day": 4, "open_trades": []})
        sp._render_autopilot_glance("US")
        monkeypatch.setattr("risk.position_manager.review_positions", lambda: [])
        monkeypatch.setattr("execution.trade_executor.recent_trades",
                            lambda n: [])
        sp._render_your_book()

    def test_cockpit_paper_trade_is_paper_only(self, monkeypatch):
        """The one-tap action must place PAPER, never live."""
        import ui.street_pulse_page as sp
        monkeypatch.setattr(sp, "st", self._FakeSt())
        monkeypatch.setattr("risk.position_sizer.size_position",
                            lambda e, s: {"qty": 10})
        seen = {}
        def _pt(**kw):
            seen.update(kw)
            return {"ok": True, "mode": "PAPER"}
        monkeypatch.setattr("execution.trade_executor.place_trade", _pt)
        sp._paper_trade_from_setup(
            {"symbol": "ZZ", "entry": 101, "stop": 98, "target": 104})
        assert seen["paper"] is True and seen["symbol"] == "ZZ"
        assert seen["qty"] == 10


class TestVerdictDashboard:
    def _seed(self, tmp_path, monkeypatch, rows):
        """rows = list of (outcome_pct, worked). entry 100 / stop 97 (3% risk)."""
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 10)
        for i, (pct, worked) in enumerate(rows):
            la = (base + timedelta(hours=i * 4)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,"
                "entry_price,stop_price,outcome_pct,worked,outcome_checked_at,"
                "outcome_price) VALUES (?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "breakout", 100.0, 97.0, pct, worked, la,
                 100 * (1 + pct / 100)))
        conn.commit(); conn.close()

    def test_needs_five_signals(self, tmp_path, monkeypatch):
        from reports.verdict_dashboard import build_equity_curve
        self._seed(tmp_path, monkeypatch, [(4.0, 1), (-2.0, 0)])
        vd = build_equity_curve(100000.0)
        assert vd["points"] == [] and vd["stats"]["closed"] == 2
        assert "kam se kam 5" in vd["verdict"]

    def test_real_trade_curve_uses_actual_trades_not_signals(self, tmp_path,
                                                             monkeypatch):
        # the REAL curve reflects trades.db (actual placed trades) in rupees,
        # resolved by target-vs-stop first-touch — not the signal_log curve.
        import pandas as pd
        from datetime import datetime, timedelta
        import execution.trade_executor as te
        import data.bhavcopy_store as bs
        import reports.verdict_dashboard as vd
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        conn = te.connect(); conn.execute(te._DDL)
        pa = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%dT09:00:00")
        for sym, qty, e, s, t, status in [
            ("WIN", 10, 100, 96, 106, "PAPER_OPEN"),   # +₹60 at target
            ("LOSS", 10, 200, 190, 220, "PAPER_OPEN"),  # −₹100 at stop
            ("OPEN", 10, 50, 48, 55, "PLACED"),         # still live
            ("FAIL", 10, 30, 28, 33, "ENTRY_FAILED")]:  # never filled → excluded
            conn.execute(
                "INSERT INTO trades (placed_at,mode,symbol,qty,entry_type,"
                "entry_price,stop_price,target_price,status) VALUES (?,?,?,?,?,?,?,?,?)",
                (pa, "PAPER", sym, qty, "LIMIT", e, s, t, status))
        conn.commit(); conn.close()
        idx = pd.date_range(datetime.now() - timedelta(days=20), periods=16, freq="D")

        def mk(h, l, c):
            return pd.DataFrame({"high": h, "low": l, "close": c}, index=idx[:len(h)])
        paths = {
            "WIN": mk([101, 107, 108], [99, 100, 101], [100, 106, 107]),
            "LOSS": mk([201, 195, 193], [199, 189, 188], [200, 192, 190]),
            "OPEN": mk([51, 52, 53], [49, 50, 51], [50, 51, 52])}
        monkeypatch.setattr(bs, "get_ohlcv", lambda s: paths.get(s.upper()))
        r = vd.build_trade_equity_curve(100000.0)
        assert r["stats"]["closed"] == 2 and r["stats"]["open"] == 1   # FAIL excluded
        assert r["stats"]["realized_pnl"] == -40.0                     # +60 − 100
        assert r["stats"]["wins"] == 1 and r["stats"]["win_rate"] == 50.0

    def test_edge_metrics_and_curve_sequence(self, tmp_path, monkeypatch):
        """Enriched metrics compute; curve has one point per signal (plotted by
        SEQUENCE, so clustered dates can't flatten it)."""
        from reports.verdict_dashboard import build_equity_curve
        # 6 wins @ +6% (2R), 4 losses @ -3% (-1R) → PF = (6*2)/(4*1) = 3.0
        rows = [(6.0, 1)] * 6 + [(-3.0, 0)] * 4
        self._seed(tmp_path, monkeypatch, rows)
        s = build_equity_curve(100000.0)["stats"]
        assert s["closed"] == 10 and s["wins"] == 6
        assert s["win_rate"] == 60.0
        assert s["profit_factor"] == 3.0                 # (6*2R)/(4*1R)
        assert s["avg_win_r"] == 2.0 and s["avg_loss_r"] == 1.0
        assert s["payoff_ratio"] == 2.0
        assert s["expectancy_r"] == round((6 * 2 - 4 * 1) / 10, 2)  # +0.80R
        assert s["expectancy_rupees"] == round(0.8 * 0.01 * 100000, 0)
        assert s["max_loss_streak"] == 4                 # the 4 losers are last
        # one curve point per signal + the seed point → plotted by index
        vd = build_equity_curve(100000.0)
        assert len(vd["points"]) == 11

    def test_negative_edge_is_flagged_red(self, tmp_path, monkeypatch):
        from reports.verdict_dashboard import build_equity_curve
        rows = [(3.0, 1)] * 2 + [(-3.0, 0)] * 8       # mostly losers → neg edge
        self._seed(tmp_path, monkeypatch, rows)
        vd = build_equity_curve(100000.0)
        assert vd["stats"]["expectancy_r"] < 0
        assert vd["verdict"].startswith("🔴")

    def test_edge_trend_detects_decay(self, tmp_path, monkeypatch):
        """Strong early, weak lately → 'decaying' (size mat badhao)."""
        from reports.verdict_dashboard import build_equity_curve
        early = [(6.0, 1)] * 40                         # great first block
        recent = [(-3.0, 0)] * 40                       # rotten recent block
        self._seed(tmp_path, monkeypatch, early + recent)
        s = build_equity_curve(100000.0)["stats"]
        assert s["recent_avg_r"] < s["expectancy_r"]
        assert s["edge_trend"] == "decaying"


class TestLiveEdge:
    """The feedback loop that raises expectancy: learn from real tracked
    outcomes, demote proven-negative signals in the scanner."""
    def _seed(self, tmp_path, monkeypatch, rows):
        """rows = (archetype, outcome_pct, worked) or (archetype, regime,
        outcome_pct, worked). entry 100 / stop 97."""
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
        conn = tk._get_conn()
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1)
        for i, row in enumerate(rows):
            arche, reg, pct, worked = (
                (row[0], "", row[1], row[2]) if len(row) == 3 else row)
            la = (base + timedelta(hours=i)).isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,"
                "regime,entry_price,stop_price,quality_score,outcome_pct,worked,"
                "outcome_checked_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (f"S{i}", la, "UNIFIED_BUY", arche, reg, 100.0, 97.0, 70.0, pct,
                 worked, la))
        conn.commit(); conn.close()

    def test_attributes_r_to_each_constituent_signal(self, tmp_path, monkeypatch):
        from scan.live_edge import profile_edge
        # a combo trade credits BOTH signals; +6% on 3% risk = +2R
        self._seed(tmp_path, monkeypatch, [("BREAKOUT_52W|POCKET_PIVOT", 6.0, 1)])
        p = profile_edge()
        assert p["signals"]["BREAKOUT_52W"]["expectancy_r"] == 2.0
        assert p["signals"]["POCKET_PIVOT"]["expectancy_r"] == 2.0
        assert p["overall"]["n"] == 1

    def test_calibration_gates_on_sample_and_bands(self, tmp_path, monkeypatch):
        from scan.live_edge import live_calibration
        # GOOD: 40 winners (+2R) → boost 1.25; BAD: 40 losers (-1R) → 0.45;
        # THIN: 5 outcomes → no claim (absent)
        rows = ([("GOOD", 6.0, 1)] * 40 + [("BAD", -3.0, 0)] * 40
                + [("THIN", 6.0, 1)] * 5)
        self._seed(tmp_path, monkeypatch, rows)
        calib = live_calibration()
        assert calib["GOOD"] == 1.25
        assert calib["BAD"] == 0.45                    # proven loser demoted
        assert "THIN" not in calib                     # <30 = no claim

    def test_scanner_blend_is_conservative(self, tmp_path, monkeypatch):
        """Live data may DEMOTE but never inflate past the backtest's view."""
        import scan.unified_scanner as us
        # backtest already distrusts SIG (0.75); live is euphoric (would be 1.25)
        monkeypatch.setattr(us, "_load_calibration", lambda: {"SIG": 0.75})
        self._seed(tmp_path, monkeypatch, [("SIG", 6.0, 1)] * 40)   # live → 1.25
        sc = us.UnifiedScanner()
        assert sc._calib["SIG"] == 0.75                # min(0.75, 1.25) — no inflation
        # and a live-proven loser pulls a trusted signal DOWN
        monkeypatch.setattr(us, "_load_calibration", lambda: {"SIG2": 1.0})
        self._seed(tmp_path, monkeypatch, [("SIG2", -3.0, 0)] * 40)  # live → 0.45
        sc2 = us.UnifiedScanner()
        assert sc2._calib["SIG2"] == 0.45              # demoted by live evidence

    def test_no_data_no_change(self, tmp_path, monkeypatch):
        """Fresh install (no outcomes) → calibration untouched, nothing breaks."""
        import scan.unified_scanner as us
        monkeypatch.setattr(us, "_load_calibration", lambda: {"SIG": 1.1})
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "empty.db"))
        sc = us.UnifiedScanner()
        assert sc._calib.get("SIG") == 1.1

    def test_edge_split_by_regime(self, tmp_path, monkeypatch):
        """Same signal can be gold in one tape, poison in another — the
        per-signal average hides it, the regime split reveals it."""
        from scan.live_edge import profile_edge
        rows = ([("BREAKOUT_52W", "TRENDING_BULL", 6.0, 1)] * 20
                + [("BREAKOUT_52W", "DISTRIBUTION", -3.0, 0)] * 20)
        self._seed(tmp_path, monkeypatch, rows)
        p = profile_edge()
        assert p["regimes"]["TRENDING_BULL"]["expectancy_r"] == 2.0
        assert p["regimes"]["DISTRIBUTION"]["expectancy_r"] == -1.0
        assert p["signals"]["BREAKOUT_52W"]["expectancy_r"] == 0.5   # avg hides it
        assert p["combos"]["BREAKOUT_52W"]["n"] == 40

    def test_regime_absent_when_unrecorded(self, tmp_path, monkeypatch):
        """Old rows logged with empty regime don't pollute the regime table."""
        from scan.live_edge import profile_edge
        self._seed(tmp_path, monkeypatch, [("BREAKOUT_52W", 6.0, 1)] * 5)
        assert profile_edge()["regimes"] == {}

    def test_regime_calibration_bands_and_gates(self, tmp_path, monkeypatch):
        from scan.live_edge import regime_calibration
        rows = ([("BREAKOUT_52W", "TRENDING_BULL", 6.0, 1)] * 35
                + [("BREAKOUT_52W", "DISTRIBUTION", -3.0, 0)] * 35
                + [("VCP", "DISTRIBUTION", 6.0, 1)] * 5)      # thin → no claim
        self._seed(tmp_path, monkeypatch, rows)
        assert regime_calibration("TRENDING_BULL") == {"BREAKOUT_52W": 1.25}
        assert regime_calibration("DISTRIBUTION") == {"BREAKOUT_52W": 0.45}
        assert regime_calibration("UNKNOWN") == {}           # unknown tape
        assert "VCP" not in regime_calibration("DISTRIBUTION")  # <30 gated

    def test_scanner_regime_factor_is_demote_only(self, tmp_path, monkeypatch):
        """A regime demotion cuts the score; a regime 'boost' can't raise it
        (min 1.0) — regime can only add caution, never chase."""
        import numpy as np, pandas as pd
        import scan.unified_scanner as us
        monkeypatch.setattr(us, "_load_calibration", lambda: {})
        import core.signal_outcome_tracker as tk
        monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "empty.db"))
        # a clean confirmed breakout out of a tight base
        n = 120
        close = np.concatenate([np.full(90, 100.0), np.full(29, 101.0), [106.0]])
        high = close + 0.5; high[-1] = 106.5
        df = pd.DataFrame({"open": close - 0.3, "high": high, "low": close - 0.6,
                           "close": close, "volume": [1_000_000] * 119 + [3_000_000]},
                          index=pd.date_range("2025-01-01", periods=n, freq="D"))
        sc = us.UnifiedScanner(); sc._nifty_ret30 = 1.0
        base = sc._analyze("X", df)
        assert base is not None and base.signals
        # demote EVERY signal that fired → score must drop
        sc._regime_calib = {s: 0.45 for s in base.signals}
        assert sc._analyze("X", df).score < base.score
        # a regime 'boost' is clamped to 1.0 → score unchanged
        sc._regime_calib = {s: 1.25 for s in base.signals}
        assert sc._analyze("X", df).score == base.score


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
        monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
        monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
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


class TestStrictLiveDisplay:
    def test_every_card_live_or_honestly_eod(self, monkeypatch):
        import ui.scanner as sc
        monkeypatch.setattr(sc, "_live_quotes",
                            lambda key: {"A": {"price": 105.0, "chg_pct": 2.0}})
        monkeypatch.setattr(sc, "_mkt_open", lambda: True)
        rows = [{"symbol": "A", "price": 100.0, "entry": 104, "verdict": "BUY"},
                {"symbol": "B", "price": 200.0, "entry": 205, "verdict": "BUY"}]
        sc._apply_live_prices(rows)
        assert rows[0]["live"] is True and rows[0]["price"] == 105.0
        assert rows[1]["live"] is False          # quote nahi → EOD, loudly
        # market band → kuch bhi 'live' nahi, chahe quote mile
        monkeypatch.setattr(sc, "_mkt_open", lambda: False)
        rows2 = [{"symbol": "A", "price": 100.0, "entry": 104,
                  "verdict": "BUY"}]
        sc._apply_live_prices(rows2)
        assert rows2[0]["live"] is False

    def test_hero_requires_live_in_market_hours(self, monkeypatch):
        import ui.scanner as sc
        rows = [
            {"symbol": "STALE", "verdict": "STRONG BUY", "edge_r": 0.3,
             "live": False},
            {"symbol": "FRESH", "verdict": "BUY", "edge_r": 0.2, "live": True},
        ]
        monkeypatch.setattr(sc, "_mkt_open", lambda: True)
        assert sc._pick_best_trade(rows)["symbol"] == "FRESH"
        # off-hours: EOD hero theek hai (tag already honest)
        monkeypatch.setattr(sc, "_mkt_open", lambda: False)
        assert sc._pick_best_trade(rows)["symbol"] == "STALE"


class TestCostModel:
    def test_zerodha_cnc_charges_accurate(self):
        from execution.cost_model import zerodha_charges
        # buy 100@1000, sell 100@1050 — known Zerodha CNC round trip
        c = zerodha_charges(1000, 1050, 100, "CNC")
        assert c["stt"] == pytest.approx(205.0)          # 0.1% both sides
        assert c["stamp"] == pytest.approx(15.0)         # 0.015% buy
        assert c["dp"] == pytest.approx(13.5)            # flat per scrip sell
        assert c["brokerage"] == 0.0                     # free delivery
        assert c["total"] == pytest.approx(243.36, abs=0.1)

    def test_slippage_directions(self):
        from execution.cost_model import simulate_fill
        assert simulate_fill(1000, "BUY") > 1000         # buys slip up
        assert simulate_fill(1000, "SELL") < 1000        # sells slip down
        # stop sells slip MORE than target sells
        assert simulate_fill(1000, "SELL", is_stop=True) < \
               simulate_fill(1000, "SELL", is_stop=False)

    def test_net_result_paper_vs_live(self):
        from execution.cost_model import net_result
        paper = net_result(1000, 1050, 100, "CNC", exit_is_stop=False, paper=True)
        live = net_result(1000, 1050, 100, "CNC", exit_is_stop=False, paper=False)
        assert paper["gross"] == 5000
        assert paper["slippage"] > 0 and paper["charges"] > 0
        assert paper["net"] < paper["gross"]             # costs bite
        assert live["slippage"] == 0                      # real fills, no sim
        assert live["net"] < live["gross"]                # charges still apply

    def test_zero_qty_safe(self):
        from execution.cost_model import zerodha_charges, net_result
        assert zerodha_charges(1000, 1050, 0)["total"] == 0.0
        assert net_result(0, 0, 0)["net"] == 0.0


class TestBreakoutConviction:
    def test_conviction_rewards_delivery_and_rs(self):
        from scan.unified_scanner import breakout_conviction as bc
        # weak: min volume, no delivery data, lagging index, below 200-DMA
        low, _ = bc(vratio=1.5, deliv_now=None, deliv_base=None,
                    rs_outperf=-5, above_50=False, above_200=False)
        # strong: heavy volume, rising delivery, RS leader, Stage-2 uptrend
        high, factors = bc(vratio=3.0, deliv_now=68, deliv_base=45,
                           rs_outperf=12, above_50=True, above_200=True)
        assert high > low
        assert high >= 85
        assert any("delivery" in f for f in factors)
        assert any("RS leader" in f for f in factors)
        assert any("Stage 2" in f for f in factors)
        # delivery unknown is neutral, not zeroed
        mid, _ = bc(vratio=2.0, deliv_now=None, deliv_base=None,
                    rs_outperf=5, above_50=True, above_200=True)
        assert 40 <= mid <= 75

    def test_base_tightness_rewards_tight_penalises_extended(self):
        import numpy as np
        from scan.unified_scanner import base_tightness
        # tight base: 40 sessions in a ~4% range, price just above, near 50-DMA
        hi = np.array([102.0] * 40); lo = np.array([98.0] * 40)
        q_tight, note = base_tightness(hi, lo, price=103, sma50=101)
        assert q_tight >= 0.75 and "tight" in note
        # extended: wide swing base + price 25% above 50-DMA
        hi2 = np.linspace(80, 130, 40)
        lo2 = hi2 - 20
        q_ext, note2 = base_tightness(hi2, lo2, price=130, sma50=104)
        assert q_ext < q_tight
        assert "extended" in note2

    def test_low_conviction_break_is_watch_not_buy(self):
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner
        # clean confirmed break (volume + ATR) BUT lagging + below 200-DMA +
        # low delivery → conviction below threshold → must NOT be a BUY breakout
        n = 90
        close = np.concatenate([np.full(70, 100.0), np.linspace(100, 96, 20)])
        close[-1] = 103.0                         # breaks the 20-day high
        high = close + 0.5
        high[:70] = 102.0
        df = pd.DataFrame({
            "open": close - 0.3, "high": high, "low": close - 1.0,
            "close": close, "volume": [1_000_000] * 68 + [2_500_000] * 22,
            "deliv_per": [30.0] * n},               # weak delivery
            index=pd.date_range("2025-04-01", periods=n, freq="D"))
        sc = UnifiedScanner()
        sc._nifty_ret30 = 15.0                     # index roaring, stock lags
        r = sc._analyze("LAGGARD", df)
        if r is not None:
            assert "BREAKOUT_52W" not in r.signals
            assert "BREAKOUT_RES" not in r.signals


class TestShortScanner:
    """Bearish detection (paper-first, no live orders). The DOWNSIDE mirror
    of the breakout engine — every long-side quality gate has a flipped
    mirror because the risk flips (bounce, not blow-off)."""

    def test_grade_breakdown_mirror_rules(self):
        from scan.short_scanner import grade_breakdown as g
        # clean: 1.0xATR below on 2.5x volume, weak close, healthy RSI → A
        ok, grade, _ = g(price=95, level=100, atr=5, vratio=2.5,
                         day_change=-3, clv=0.1, rsi=45)
        assert ok and grade == "A"
        # strong close (near day HIGH) on a break = bear-trap → demote A→B
        ok, grade, note = g(price=95, level=100, atr=5, vratio=2.5,
                           day_change=-3, clv=0.85, rsi=45)
        assert ok and grade == "B" and "strong close" in note
        # RSI floor: already-crushed stock → reject (bounce risk), NOT a fresh short
        ok, _, note = g(price=95, level=100, atr=5, vratio=2.5,
                       day_change=-3, clv=0.1, rsi=15)
        assert not ok and "capitulation" in note
        # capitulation gap-down → reject (don't chase)
        ok, _, note = g(price=88, level=100, atr=5, vratio=3,
                       day_change=-12, clv=0.1, rsi=40)
        assert not ok and "capitulation" in note
        # price ABOVE the level → no breakdown
        assert g(price=101, level=100, atr=5, vratio=3, day_change=-1,
                 rsi=50)[0] is False

    def test_confirmed_breakdown_is_a_short_with_mirror_geometry(self):
        """A fresh break below support (RSI healthy-bearish, not capitulated)
        → verdict SHORT with stop ABOVE entry and target BELOW."""
        import numpy as np
        import pandas as pd
        from scan.short_scanner import analyze_short
        rng = np.random.default_rng(5)
        up = np.linspace(70, 104, 150) + rng.normal(0, 0.4, 150)
        top = 103 + np.abs(rng.normal(0, 1.0, 30))     # topping range, floor ~103
        close = np.concatenate([up, top, [100.5]])      # fresh break below ~102
        high = close + 1.0
        low = close - 0.8
        low[-1], high[-1] = 100.0, 103.2                # weak close near the low
        vol = np.concatenate([np.full(150, 1e6), np.full(30, 1e6), [2.4e6]])
        df = pd.DataFrame({
            "open": close + 0.5, "high": high, "low": low, "close": close,
            "volume": vol},
            index=pd.date_range("2024-01-01", periods=len(close), freq="D"))
        r = analyze_short("SHORTME", df)
        assert r is not None
        assert "BREAKDOWN_SUP" in r.signals
        assert r.verdict == "SHORT" and r.breakdown_grade in ("A", "B")
        assert r.stop > r.entry                          # short: stop is ABOVE
        assert r.target < r.entry                        # target is BELOW
        assert r.risk_reward > 1.5

    def test_rising_rocket_demotes_short_to_avoid(self):
        """A short is a WEAKNESS entry — bearish structure but UP big today
        must demote SHORT→AVOID (mirror of the long-side falling-knife)."""
        import numpy as np
        import pandas as pd
        from scan.short_scanner import analyze_short
        rng = np.random.default_rng(5)
        up = np.linspace(70, 104, 150) + rng.normal(0, 0.4, 150)
        top = 103 + np.abs(rng.normal(0, 1.0, 30))
        close = np.concatenate([up, top, [100.5]])
        close[-1] = close[-2] * 1.02                     # +2% GREEN day
        df = pd.DataFrame({
            "open": close - 0.3, "high": close + 0.6, "low": close - 0.6,
            "close": close,
            "volume": np.concatenate([np.full(150, 1e6), np.full(30, 1e6),
                                      [2.4e6]])},
            index=pd.date_range("2024-01-01", periods=len(close), freq="D"))
        r = analyze_short("ROCKET", df)
        if r is not None:
            assert r.verdict != "SHORT"                  # never short strength

    def test_no_short_in_healthy_uptrend(self):
        """A clean uptrend has no bearish setup → None (not a short)."""
        import numpy as np
        import pandas as pd
        from scan.short_scanner import analyze_short
        close = np.linspace(80, 140, 220)
        df = pd.DataFrame({
            "open": close - 0.3, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": [1_000_000] * 220},
            index=pd.date_range("2024-01-01", periods=220, freq="D"))
        assert analyze_short("UPONLY", df) is None


class TestSniperConfirmation:
    """The BAJEL bug: a wick that pokes the level then keeps falling must
    NEVER fire 'BREAKOUT CONFIRMED'."""
    WATCH = {101: {"symbol": "BAJEL", "trigger": 182.0,
                   "stop": 169.0, "target": 210.0, "avg_vol": 1_000_000}}

    def _tick(self, ltp, volume=1_200_000):
        return [{"instrument_token": 101, "last_price": ltp,
                 "volume_traded": volume}]

    def test_wick_then_reverse_never_fires(self):
        from scan.breakout_sniper import process_ticks
        arm, fired = {}, set()
        # t0: wick to 182.5 clears the level → armed, but NOT fired
        assert process_ticks(self._tick(182.5), self.WATCH, fired, arm,
                             now=0, hold_seconds=45) == []
        assert "BAJEL" in arm
        # t+10s: still up but hold window not met → no fire
        assert process_ticks(self._tick(182.6), self.WATCH, fired, arm,
                             now=10, hold_seconds=45) == []
        # t+20s: reverses below trigger → DISARM (false poke forgotten)
        assert process_ticks(self._tick(178.0), self.WATCH, fired, arm,
                             now=20, hold_seconds=45) == []
        assert "BAJEL" not in arm
        # t+120s: keeps falling → still nothing, ever
        assert process_ticks(self._tick(170.0), self.WATCH, fired, arm,
                             now=120, hold_seconds=45) == []

    def test_genuine_hold_confirms(self):
        from scan.breakout_sniper import process_ticks
        arm, fired = {}, set()
        # clears and holds above for the full window → CONFIRMED
        assert process_ticks(self._tick(183.0), self.WATCH, fired, arm,
                             now=0, hold_seconds=45, frac=0.5) == []
        hits = process_ticks(self._tick(184.0), self.WATCH, fired, arm,
                             now=50, hold_seconds=45, frac=0.5)
        assert len(hits) == 1
        assert hits[0]["symbol"] == "BAJEL" and hits[0]["held_s"] == 50

    def test_zero_volume_never_confirms(self):
        from scan.breakout_sniper import process_ticks
        arm, fired = {}, set()
        process_ticks(self._tick(183.0, volume=0), self.WATCH, fired, arm,
                      now=0, hold_seconds=45, frac=0.5)
        hits = process_ticks(self._tick(184.0, volume=0), self.WATCH, fired, arm,
                             now=50, hold_seconds=45, frac=0.5)
        assert hits == []

    def test_volume_confirms_pacing(self):
        from scan.breakout_sniper import volume_confirms
        # Pace-aware absolute floor — AUROPHARMA-style 0.1× never confirms
        assert volume_confirms(100_000, 1_000_000, 0.08) is False  # 0.1× < 0.25 early floor
        assert volume_confirms(100_000, 1_000_000, 0.01) is False  # early, still thin
        # halfway: floor = max(0.5, 0.25)=0.5× day; pace needs 1.2×0.5=0.6×
        assert volume_confirms(1_200_000, 1_000_000, 0.5) is True
        assert volume_confirms(700_000, 1_000_000, 0.5) is True    # 0.7 ≥ 0.6 pace + 0.5 floor
        assert volume_confirms(400_000, 1_000_000, 0.5) is False   # 0.4 < 0.5 floor
        # fail-closed: zero / unknown avg
        assert volume_confirms(0, 1_000_000, 0.5) is False
        assert volume_confirms(1_200_000, 0, 0.5) is False
        # early session: real open surge (≥0.25×) allowed; tiny print not
        assert volume_confirms(1_000_000, 1_000_000, 0.01) is True
        assert volume_confirms(250_000, 1_000_000, 0.01) is True
        assert volume_confirms(1, 1_000_000, 0.01) is False
        # mid-morning legitimate surge (0.35× day at ~20% session) can confirm
        assert volume_confirms(350_000, 1_000_000, 0.2) is True
        assert volume_confirms(200_000, 1_000_000, 0.2) is False  # below 0.25 early floor

    def test_thin_absolute_volume_break_does_not_fire(self):
        """0.1× avg-day must never fire BREAKOUT CONFIRMED (pace math trap)."""
        from scan.breakout_sniper import process_ticks
        watch = {101: {"symbol": "THIN", "trigger": 182.0, "stop": 169.0,
                       "target": 210.0, "avg_vol": 1_000_000}}
        arm, fired = {}, set()
        process_ticks([{"instrument_token": 101, "last_price": 183.0,
                        "volume_traded": 100_000}], watch, fired, arm,
                      now=0, hold_seconds=45, frac=0.08)
        hits = process_ticks([{"instrument_token": 101, "last_price": 184.0,
                               "volume_traded": 100_000}], watch, fired, arm,
                             now=50, hold_seconds=45, frac=0.08)
        assert hits == []

    def test_day_fraction_bounds(self):
        import pytz
        from datetime import datetime as dt
        from scan.breakout_sniper import day_fraction
        ist = pytz.timezone("Asia/Kolkata")
        assert day_fraction(ist.localize(dt(2026, 7, 14, 9, 15))) == 0.0
        assert day_fraction(ist.localize(dt(2026, 7, 14, 15, 30))) == 1.0
        mid = day_fraction(ist.localize(dt(2026, 7, 14, 12, 22)))
        assert 0.45 <= mid <= 0.55

    def test_dead_volume_break_does_not_fire(self):
        from scan.breakout_sniper import process_ticks
        watch = {101: {"symbol": "DEAD", "trigger": 182.0, "stop": 169.0,
                       "target": 210.0, "avg_vol": 1_000_000}}
        arm, fired = {}, set()
        # arm at t0, cleared + will hold, but volume dead
        process_ticks([{"instrument_token": 101, "last_price": 183.0,
                        "volume_traded": 100_000}], watch, fired, arm,
                      now=0, hold_seconds=45, frac=0.5)
        # held long enough BUT volume way below pace → still no fire
        hits = process_ticks([{"instrument_token": 101, "last_price": 184.0,
                               "volume_traded": 150_000}], watch, fired, arm,
                             now=50, hold_seconds=45, frac=0.5)
        assert hits == []
        # same break but volume running hot (≥ pace + floor) → confirms
        hits2 = process_ticks([{"instrument_token": 101, "last_price": 184.0,
                                "volume_traded": 700_000}], watch, fired, arm,
                              now=60, hold_seconds=45, frac=0.5)
        assert len(hits2) == 1 and hits2[0]["symbol"] == "DEAD"

    def test_touch_without_clearance_does_not_arm(self):
        from scan.breakout_sniper import process_ticks
        arm, fired = {}, set()
        # exactly at trigger (no clearance buffer) → not armed
        process_ticks(self._tick(182.0), self.WATCH, fired, arm,
                      now=0, hold_seconds=45)
        assert "BAJEL" not in arm

    def test_watch_map_skips_chase_risk_and_blowoff(self):
        """The sniper is a separate path from the scanner — it must apply the
        scanner's own quality demotes, not fire a green 'BREAKOUT CONFIRMED'
        (and auto-trade) on a stock the scanner flagged extended/overbought."""
        from scan.breakout_sniper import build_watch_map
        base = {"categories": ["PreBreakout"], "pivot_distance_pct": 1.0,
                "entry": 100.0, "stop": 95.0, "target": 112.0,
                "avg_vol20": 1_000_000}
        clean = {**base, "symbol": "CLEAN", "rsi": 60, "chase_risk": False}
        chased = {**base, "symbol": "CHASED", "rsi": 60, "chase_risk": True}
        blowoff = {**base, "symbol": "BLOWOFF", "rsi": 88, "chase_risk": False}
        zero_vol = {**base, "symbol": "NOVOL", "rsi": 55, "chase_risk": False,
                    "avg_vol20": 0}
        # tokens_for is best-effort; watch map keys off symbol regardless
        syms = {v["symbol"] for v in
                build_watch_map([clean, chased, blowoff, zero_vol]).values()}
        # even if instrument-token lookup is empty in the test env, the
        # PURE filter decision is what we assert — re-derive it directly:
        from scan.breakout_sniper import _quality_skip
        assert _quality_skip(clean) == ""
        assert "chase" in _quality_skip(chased)
        assert "blow-off" in _quality_skip(blowoff)
        assert "volume" in _quality_skip(zero_vol).lower()
        assert "NOVOL" not in syms
        assert "BLOWOFF" not in syms
        assert "CHASED" not in syms

    def test_quality_skip_backward_compatible(self):
        """Old scan dicts without chase/rsi flags still pass when volume is known;
        missing volume is never sniper-suggested."""
        from scan.breakout_sniper import _quality_skip
        assert _quality_skip({"symbol": "X", "avg_vol20": 500_000}) == ""
        assert "volume" in _quality_skip({"symbol": "X"}).lower()


class TestBreakoutConfirmation:
    def test_grade_breakout_rules(self):
        from scan.unified_scanner import grade_breakout as g
        # clean: 1.0xATR clearance on 2.5x volume → grade A
        ok, grade, _ = g(price=105, level=100, atr=5, vratio=2.5, day_change=3)
        assert ok and grade == "A"
        # decent: 0.6xATR on 1.7x volume → grade B (confirmed)
        ok, grade, _ = g(price=103, level=100, atr=5, vratio=1.7, day_change=3)
        assert ok and grade == "B"
        # marginal clearance: 0.2xATR → NOT confirmed (false-break risk)
        ok, grade, note = g(price=101, level=100, atr=5, vratio=2.0, day_change=3)
        assert not ok and "clearance" in note
        # no volume: 0.6xATR but 1.0x volume → NOT confirmed
        ok, _, note = g(price=103, level=100, atr=5, vratio=1.0, day_change=3)
        assert not ok and "volume" in note
        # exhaustion gap: +10% day → NOT confirmed (chase risk)
        ok, _, note = g(price=110, level=100, atr=5, vratio=3.0, day_change=10)
        assert not ok and "exhaustion" in note
        # price below level → nothing
        assert g(price=99, level=100, atr=5, vratio=3, day_change=1)[0] is False

    def test_close_location_value_math(self):
        """CLV pure function: Wyckoff close-of-range strength."""
        from scan.unified_scanner import close_location_value as clv
        assert clv(close=100, high=100, low=95) == 1.0     # closed at high
        assert clv(close=95, high=100, low=95) == 0.0       # closed at low
        assert clv(close=97.5, high=100, low=95) == 0.5     # mid-range
        assert clv(close=100, high=100, low=100) == 0.5     # flat day → neutral
        assert clv(close=200, high=100, low=95) == 1.0      # clamped, never >1

    def test_weak_close_demotes_grade_a_to_b(self):
        """ATR+volume clear cleanly (would be grade A), but the stock closed
        near the day's LOW (sellers took the day back) — demote A→B. ATR/
        volume alone can't see this; CLV can."""
        from scan.unified_scanner import grade_breakout as g
        strong = g(price=105, level=100, atr=3.33, vratio=3.0, day_change=2,
                  clv=0.8)
        assert strong == (True, "A", strong[2])             # strong close → A stays
        weak = g(price=105, level=100, atr=3.33, vratio=3.0, day_change=2,
                 clv=0.15)
        assert weak[0] is True and weak[1] == "B"            # demoted, still confirmed
        assert "demote" in weak[2] and "0.15" in weak[2]

    def test_weak_close_demotes_marginal_b_to_rejected(self):
        """A grade-B-level clearance WITH a weak close is a bull-trap
        pattern — rejected outright, not just downgraded."""
        from scan.unified_scanner import grade_breakout as g
        ok_strong, grade, _ = g(price=103, level=100, atr=5, vratio=1.7,
                               day_change=3, clv=0.8)
        assert ok_strong and grade == "B"                    # unaffected baseline
        ok_weak, grade_weak, note = g(price=103, level=100, atr=5, vratio=1.7,
                                      day_change=3, clv=0.15)
        assert ok_weak is False and grade_weak == ""
        assert "bull-trap" in note

    def test_clv_default_is_backward_compatible(self):
        """Callers that don't pass clv (default 1.0) behave exactly as
        before this feature — no silent behaviour change for old call sites."""
        from scan.unified_scanner import grade_breakout as g
        with_default = g(price=105, level=100, atr=5, vratio=2.5, day_change=3)
        explicit_strong = g(price=105, level=100, atr=5, vratio=2.5,
                            day_change=3, clv=1.0)
        assert with_default == explicit_strong

    def test_breakout_conviction_clv_demotion_and_floor(self):
        """Weak close cuts breakout_conviction (post weight-budget, demote-
        only) — enough to push a real setup below the BUY threshold, which
        A/B grade labelling alone would not have done."""
        from scan.unified_scanner import breakout_conviction as bc
        kwargs = dict(vratio=2.5, deliv_now=60, deliv_base=50, rs_outperf=5,
                      above_50=True, above_200=True, base_q=0.8)
        strong, _ = bc(**kwargs, clv=0.9)
        default, _ = bc(**kwargs)                    # clv=1.0 default
        threshold, _ = bc(**kwargs, clv=0.5)          # boundary — no cliff
        weak, factors = bc(**kwargs, clv=0.15)
        assert strong == default == threshold         # >=0.5 → identical, no penalty
        assert weak < strong                          # demoted
        assert weak == round(strong * (0.6 + 0.8 * 0.15), 0)
        assert any("weak close" in f for f in factors)
        worst, _ = bc(**kwargs, clv=0.0)               # closed at the day's low
        assert worst == round(strong * 0.6, 0)         # worst-case: 40% cut, floored

    def test_rsi_overbought_soft_demotes_grade_a_to_b(self):
        """Clean clearance that would grade A, but RSI is at the soft ceiling
        (70) — demote A→B. Above 70 hard-rejects; exactly 70 only demotes."""
        from scan.unified_scanner import grade_breakout as g
        strong = g(price=105, level=100, atr=3.33, vratio=3.0, day_change=2,
                  rsi=55)
        assert strong == (True, "A", strong[2])              # normal RSI → A stays
        extended = g(price=105, level=100, atr=3.33, vratio=3.0, day_change=2,
                     rsi=70)
        assert extended[0] is True and extended[1] == "B"    # demoted, still confirmed
        assert "demote" in extended[2] and "70" in extended[2]

    def test_rsi_overbought_hard_rejects_outright(self):
        """RSI >70 is a blow-off-top — rejected outright, same tier as the
        gap-exhaustion check, regardless of otherwise-clean clearance."""
        from scan.unified_scanner import grade_breakout as g
        ok, grade, note = g(price=105, level=100, atr=3.33, vratio=3.0,
                            day_change=2, rsi=85)
        assert ok is False and grade == ""
        assert "blow-off-top" in note

    def test_rsi_default_is_backward_compatible(self):
        """Callers that don't pass rsi (default 0.0) behave exactly as
        before this feature — no silent behaviour change for old call sites."""
        from scan.unified_scanner import grade_breakout as g
        with_default = g(price=105, level=100, atr=5, vratio=2.5, day_change=3)
        explicit = g(price=105, level=100, atr=5, vratio=2.5, day_change=3,
                     rsi=0.0)
        assert with_default == explicit

    def test_rsi_and_clv_combined_flags(self):
        """Weak close AND extended RSI can fire together — both flags land
        in one note, still just a demote (not a double-penalty reject)
        as long as the underlying clearance is clean and RSI is not >70."""
        from scan.unified_scanner import grade_breakout as g
        ok, grade, note = g(price=105, level=100, atr=3.33, vratio=3.0,
                            day_change=2, clv=0.2, rsi=70)
        assert ok is True and grade == "B"
        assert "CLV" in note and "RSI" in note and "70" in note

    def test_breakout_conviction_pct_below_high_demotion_and_floor(self):
        """A break happening well below the 52-week high is a laggard, not
        a leader — demote conviction (never a hard block; other factors can
        still carry an exceptional setup). Fresh 52-week-high breaks pass
        pct_below_high=0.0 by construction, so they're exempt."""
        from scan.unified_scanner import breakout_conviction as bc
        kwargs = dict(vratio=2.5, deliv_now=60, deliv_base=50, rs_outperf=5,
                      above_50=True, above_200=True, base_q=0.8)
        near, _ = bc(**kwargs, pct_below_high=5)
        default, _ = bc(**kwargs)                       # 0.0 default
        threshold, _ = bc(**kwargs, pct_below_high=30)   # boundary — no cliff
        assert near == default == threshold
        laggard, factors = bc(**kwargs, pct_below_high=45)
        assert laggard < near
        assert any("laggard zone" in f for f in factors)
        floor, _ = bc(**kwargs, pct_below_high=60)
        floor2, _ = bc(**kwargs, pct_below_high=90)      # deep past floor
        assert floor == floor2                            # floors at 30% cut, no further
        assert abs(floor - near * 0.70) <= 1              # ~30% cut (rounding tolerance)
        # even a DEEP laggard with a genuinely strong break must still be
        # able to clear the BUY conviction gate (>=50) — the filter should
        # thin out mediocre laggards, never blanket-veto the category
        assert floor >= 50

    def test_pct_below_high_default_is_backward_compatible(self):
        """Callers that don't pass pct_below_high (default 0.0, exempt by
        construction for fresh 52-week-high breaks) behave exactly as
        before this feature."""
        from scan.unified_scanner import breakout_conviction as bc
        kwargs = dict(vratio=2.5, deliv_now=60, deliv_base=50, rs_outperf=5,
                      above_50=True, above_200=True, base_q=0.8)
        with_default, _ = bc(**kwargs)
        explicit, _ = bc(**kwargs, pct_below_high=0.0)
        assert with_default == explicit

    def test_marginal_note_is_time_aware(self, monkeypatch):
        """Off-hours the close is IN — a marginal break must say 'close pe
        bhi confirm nahi hua', NOT 'close ka wait' (data humare paas hai)."""
        import scan.unified_scanner as us
        # market LIVE → 'wait'
        monkeypatch.setattr(us, "_market_is_live", lambda: True)
        _, _, note_live = us.grade_breakout(price=100.2, level=100, atr=5,
                                            vratio=1.0, day_change=2)
        assert "wait" in note_live
        # market CLOSED → definitive, no 'wait'
        monkeypatch.setattr(us, "_market_is_live", lambda: False)
        _, _, note_eod = us.grade_breakout(price=100.2, level=100, atr=5,
                                           vratio=1.0, day_change=2)
        assert "confirm nahi hua" in note_eod and "wait" not in note_eod

    def test_marginal_break_is_watch_not_buy(self):
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner
        n = 80
        # flat base at ~100, resistance ~102; last bar pokes 102.3 on FLAT
        # volume → must NOT fire a BREAKOUT (false-break guard)
        close = np.full(n, 100.0)
        high = np.full(n, 102.0)
        close[-1], high[-1] = 102.3, 102.4
        df = pd.DataFrame({
            "open": close - 0.2, "high": high, "low": close - 1.0,
            "close": close, "volume": [1_000_000] * n},
            index=pd.date_range("2025-06-01", periods=n, freq="D"))
        r = UnifiedScanner()._analyze("POKE", df)
        if r is not None:
            assert "BREAKOUT_52W" not in r.signals
            assert "BREAKOUT_RES" not in r.signals
            assert r.breakout_grade == ""


class TestExtensionGuard:
    def test_extended_momentum_is_watch_not_buy(self):
        """TATVA-type: big 5-day run, stretched above 20-EMA, no confirmed
        breakout → downgraded to WATCH with a 'don't chase' caveat."""
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner
        # calm base then a sharp +14% vertical spike on volume (extended)
        base = np.full(80, 100.0)
        spike = np.array([101, 103, 106, 110, 114.0])
        close = np.concatenate([base, spike])
        high = close + 0.4
        df = pd.DataFrame({
            "open": close - 0.3, "high": high, "low": close - 0.5,
            "close": close, "volume": [1_000_000] * 80 + [2_500_000] * 5},
            index=pd.date_range("2025-03-01", periods=85, freq="D"))
        r = UnifiedScanner()._analyze("TATVA", df)
        if r is not None and not r.breakout_grade:   # no confirmed breakout
            assert r.verdict == "WATCH"
            assert any("Extended" in x or "chase" in x.lower() for x in r.reasons)

    def test_steady_grinder_far_above_50dma_is_chase(self):
        """ADANIENSOL-type: a long base then a compressed run leaves the stock
        glued to its 20-EMA (short-term guard stays SILENT) but far above its
        50-DMA — a late-stage/climax entry. The 50-DMA extension view must
        catch what the 20-EMA view misses, and flag chase_risk so the sniper
        skips it and buzz/earnings enrichment can't promote it back to BUY."""
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner, _ema_np
        base = np.full(160, 100.0)
        run = [100.0]
        for i in range(28):                          # ~1.4%/day with dips
            run.append(run[-1] * (1 + (-0.010 if i % 5 == 4 else 0.017)))
        close = np.concatenate([base, np.array(run[1:])])
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.004, "low": close * 0.996,
            "close": close,
            "volume": list(np.concatenate([np.full(160, 1e6),
                            np.linspace(1e6, 1.3e6, len(run) - 1)]))},
            index=pd.date_range("2024-11-01", periods=len(close), freq="D"))
        price = close[-1]
        ema20 = _ema_np(close, 20)
        mom5 = (close[-1] / close[-6] - 1) * 100
        # precondition: the OLD short-term guard would NOT fire here
        assert not ((price / ema20 - 1) * 100 > 10 and mom5 > 10)
        r = UnifiedScanner()._analyze("GRINDER", df)
        if r is not None and not r.breakout_grade:
            assert r.chase_risk is True
            assert r.verdict == "WATCH"
            assert any("50-DMA" in x for x in r.reasons)

    def test_falling_knife_is_not_a_fresh_buy(self):
        """A stock red today / RSI rolling over is NOT breaking out — it's
        falling. Even when NOT extended, a fresh BUY must demote to WATCH
        ('RSI neeche girta hua stock mat recommend kar'). A pullback-to-
        support (buys weakness by design) is the one exemption."""
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner, _ema_np
        rng = np.random.default_rng(3)
        base = np.full(150, 100.0)
        up = np.linspace(100, 112, 25) + rng.normal(0, 0.4, 25)   # gentle, NOT extended
        tail = np.array([111.5, 110.5, 109.2])                     # last 3 days falling
        close = np.concatenate([base, up, tail])
        df = pd.DataFrame({
            "open": close * 1.002, "high": close * 1.006, "low": close * 0.99,
            "close": close,
            "volume": list(np.concatenate([np.full(150, 1e6),
                            np.linspace(1e6, 1.3e6, 25), [1.2e6, 1.25e6, 1.3e6]]))},
            index=pd.date_range("2024-11-01", periods=len(close), freq="D"))
        price = close[-1]
        # precondition: NOT extended (so this is the falling-knife guard's catch)
        assert (price / _ema_np(close, 20) - 1) * 100 <= 10
        assert (price / close[-50:].mean() - 1) * 100 <= 20
        r = UnifiedScanner()._analyze("KNIFE", df)
        if r is not None and "PULLBACK_SUPPORT" not in r.signals:
            assert r.verdict == "WATCH"
            assert r.chase_risk is True
            assert any("falling knife" in x.lower() for x in r.reasons)


class TestPullbackSetup:
    def test_pullback_detected_not_at_high(self):
        """A Stage-2 uptrend that pulled back to a rising EMA on drying
        volume + tightening = a buyable pullback, NOT a 52-week-high break."""
        import numpy as np
        from scan.unified_scanner import detect_pullback_support, _atr
        # long uptrend 60→120, then a mild pullback off the ~120 high to ~112,
        # tightening range + volume drying up
        up = np.linspace(60, 120, 200)
        pull = np.linspace(120, 112, 15)
        close = np.concatenate([up, pull])
        high = close + 0.6
        low = close - 0.6
        high[-15:] = close[-15:] + 0.3          # tightening
        low[-15:] = close[-15:] - 0.3
        vol = np.concatenate([np.full(200, 2_000_000), np.full(15, 800_000)])
        atr = _atr(high, low, close)
        ok, reason, pivot = detect_pullback_support(close, high, low, vol, atr)
        assert ok is True
        assert "pullback" in reason.lower()
        assert pivot > close[-1]                 # recent high above current px

    def test_no_pullback_when_extended(self):
        import numpy as np
        from scan.unified_scanner import detect_pullback_support, _atr
        # straight vertical run, no pullback, price AT the high → not a pullback
        close = np.linspace(60, 130, 215)
        high = close + 0.6; low = close - 0.6
        vol = np.full(215, 2_000_000)
        ok, _, _ = detect_pullback_support(close, high, low, vol, _atr(high, low, close))
        assert ok is False

    def test_no_pullback_in_downtrend(self):
        import numpy as np
        from scan.unified_scanner import detect_pullback_support, _atr
        # below a falling 50-EMA → never a buyable pullback
        close = np.linspace(130, 70, 215)
        high = close + 0.6; low = close - 0.6
        vol = np.full(215, 1_000_000)
        ok, _, _ = detect_pullback_support(close, high, low, vol, _atr(high, low, close))
        assert ok is False


class TestFallingKnifeFilter:
    def test_beaten_down_arr_threshold(self):
        from scan.unified_scanner import is_beaten_down_arr
        highs = [100.0] * 250
        assert is_beaten_down_arr(highs, 49.0) is True      # −51% → knife
        assert is_beaten_down_arr(highs, 51.0) is False     # −49% → ok
        assert is_beaten_down_arr(highs, 50.0) is True      # exactly −50%
        # too little history → never flagged (no guessing)
        assert is_beaten_down_arr([100.0] * 10, 5.0) is False
        assert is_beaten_down_arr(highs, 0.0) is False

    def test_beaten_down_dataframe(self):
        import pandas as pd
        from scan.unified_scanner import is_beaten_down
        idx = pd.date_range("2025-01-01", periods=250, freq="D")
        # peaked at 500, now 150 → −70%
        highs = [500.0] * 125 + [150.0] * 125
        df = pd.DataFrame({"high": highs, "close": [h - 1 for h in highs]},
                          index=idx)
        assert is_beaten_down(df) is True
        # recovered name: peaked 500, now 480 → −4%
        df2 = pd.DataFrame({"high": [500.0] * 250, "close": [480.0] * 250},
                           index=idx)
        assert is_beaten_down(df2) is False

    def test_scanner_skips_beaten_down_stock(self):
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner
        # a stock that peaked near 500 and now trades ~150 (−70%), liquid,
        # with an up day — would otherwise trip momentum, but must be skipped
        n = 260
        close = np.concatenate([np.linspace(480, 500, 130),
                                np.linspace(500, 150, 130)])
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.01,
            "low": close * 0.98, "close": close,
            "volume": [1_000_000] * n},
            index=pd.date_range("2025-01-01", periods=n, freq="D"))
        assert UnifiedScanner()._analyze("FALLEN", df) is None


class TestDailyPulseChart:
    def test_daily_ohlcv_is_1d_and_latest(self, monkeypatch):
        """Pulse chart data must be DAILY candles from the bhav store
        (which carries today's live-overlaid bar), not 15-min."""
        import pandas as pd
        import data.bhavcopy_store as bs
        import ui.street_pulse_page as sp
        idx = pd.date_range("2026-04-01", periods=120, freq="D")
        df = pd.DataFrame({
            "open": range(100, 220), "high": range(101, 221),
            "low": range(99, 219), "close": range(100, 220),
            "volume": [1000] * 120}, index=idx)
        monkeypatch.setattr(bs, "get_ohlcv", lambda s: df)
        sp._daily_ohlcv.clear()
        d = sp._daily_ohlcv("HAL")
        assert d is not None
        assert len(d["c"]) == 90                      # last 90 sessions
        assert d["idx"][-1] == "2026-07-29"           # latest bar last
        assert d["c"][-1] == 219                       # up-to-date close
        # too little data → None (caller falls back to text)
        monkeypatch.setattr(bs, "get_ohlcv",
                            lambda s: df.head(10))
        sp._daily_ohlcv.clear()
        assert sp._daily_ohlcv("THIN") is None


class TestUSCostModel:
    def test_us_charges_commission_free(self):
        from execution.cost_model import us_charges, net_result
        c = us_charges(100, 105, 100)
        assert c["commission"] == 0.0                    # Alpaca-style
        assert c["total"] < 1.0                          # only tiny SEC + TAF
        # net_result routes US market correctly
        us = net_result(100, 105, 100, exit_is_stop=False, paper=True, market="US")
        india = net_result(100, 105, 100, exit_is_stop=False, paper=True)
        assert us["net"] > india["net"]                  # US far cheaper
        assert us["net"] < us["gross"]                   # still slippage+fees


class TestUSAutopilot:
    def _setup(self, tmp_path, monkeypatch):
        import execution.us_autopilot as ua
        import execution.trade_executor as te
        import data.us_data as ud
        monkeypatch.setattr(ua, "_STATE_FILE", tmp_path / "us_ap.json")
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        ua._state = {}
        monkeypatch.setattr(ua, "_notify", lambda m: None)
        monkeypatch.setattr(ua, "_in_window", lambda now_et=None: True)
        monkeypatch.setattr(ud, "us_live_prices",
                            lambda syms: {s: {"price": 100.0} for s in syms})
        # anchor uses us_live_prices too; entry passed = 100 so no chase
        ua.set_config(allocation=50000)
        return ua, te

    def test_paper_only_and_gates(self, tmp_path, monkeypatch):
        ua, te = self._setup(tmp_path, monkeypatch)
        # not armed → no trade
        assert ua.consider("AAPL", 100, 95, 80, 60) is False
        ua.arm()
        # invalid stop → no trade
        assert ua.consider("AAPL", 100, 105, 80, 60) is False
        # low conviction → blocked
        assert ua.consider("AAPL", 100, 95, 80, 40) is False
        # valid → paper trade, tagged US_AUTOPILOT, +4% target
        assert ua.consider("AAPL", 100, 95, 80, 70) is True
        t = te.recent_trades(1)[0]
        assert "US_AUTOPILOT" in t["note"] and t["mode"] == "PAPER"
        assert abs(float(t["target_price"]) - 104.0) < 0.5
        # once per day
        assert ua.consider("AAPL", 100, 95, 80, 70) is False

    def test_funnel_and_preset_us(self, tmp_path, monkeypatch):
        ua, te = self._setup(tmp_path, monkeypatch)
        ua.arm()
        assert ua.consider("AAPL", 100, 105, 80, 60) is False   # invalid stop
        assert ua.consider("MSFT", 100, 95, 80, 30) is False    # low conviction
        assert ua.consider("NVDA", 100, 95, 80, 70) is True     # valid
        f = ua.reject_funnel()
        assert f["considered"] == 3 and sum(f["rejects"].values()) == 2
        assert f["rejects"].get("invalid stop") == 1
        assert f["rejects"].get("conviction kam") == 1
        # preset moves breadth, not safety
        base = ua.get_status()
        ua.apply_preset("Aggressive")
        aggr = ua.get_status()
        assert aggr["preset"] == "Aggressive"
        assert aggr["min_score"] < base["min_score"]
        assert aggr["risk_per_trade_pct"] == base["risk_per_trade_pct"]
        assert ua.apply_preset("Nope")[0] is False

    def test_concurrent_writes_no_lock_error(self, tmp_path, monkeypatch):
        """Both autopilots + position manager write the SAME db from
        threads. The WAL + busy-timeout connect() must survive concurrent
        writers without 'database is locked' losing a journal row."""
        import threading
        import execution.trade_executor as te
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        te._journal({"mode": "PAPER", "symbol": "SEED", "qty": 1,
                     "entry_type": "MARKET", "entry_price": 100, "stop_price": 95,
                     "target_price": 104, "product": "CNC",
                     "status": "PAPER_OPEN", "note": "seed"})
        errors = []

        def _writer(i):
            try:
                for j in range(10):
                    te._journal({"mode": "PAPER", "symbol": f"T{i}_{j}", "qty": 1,
                                 "entry_type": "MARKET", "entry_price": 100,
                                 "stop_price": 95, "target_price": 104,
                                 "product": "CNC", "status": "PAPER_OPEN",
                                 "note": "concurrent"})
            except Exception as exc:
                errors.append(str(exc))
        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(te.recent_trades(200)) == 61        # 1 seed + 60, none lost

    def test_us_breaker_trips_on_open_drawdown(self, tmp_path, monkeypatch):
        """US circuit breaker must fire on UNREALIZED loss of open
        positions, not only realized (NSE-parity)."""
        import sqlite3
        import data.us_data as ud
        from datetime import datetime as dt
        ua, te = self._setup(tmp_path, monkeypatch)
        ua.set_config(daily_loss_limit_pct=0.03)      # -3% of pool
        ua.arm()
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        # open position: 200 sh @ $100 = $20k; pool 50k → 3% = $1,500
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (dt.now().isoformat(timespec='seconds'), "PAPER", "TSLA", 200,
             "MARKET", 100, 90, 104, "CNC", "PAPER_OPEN", "US_AUTOPILOT:t"))
        conn.commit(); conn.close()
        # live $90 → open loss (90-100)*200 = -$2,000 < -$1,500 → trip
        monkeypatch.setattr(ud, "us_live_prices",
                            lambda syms: {"TSLA": {"price": 90.0}})
        ua._circuit_breaker()
        assert ua.get_status()["armed"] is False

    def test_us_report_card_net_and_evidence_gate(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import datetime as dt
        ua, te = self._setup(tmp_path, monkeypatch)
        conn = sqlite3.connect(te._DB)
        conn.execute(te._DDL)
        conn.execute(
            "INSERT INTO trades (placed_at, mode, symbol, qty, entry_type, "
            "entry_price, stop_price, target_price, product, status, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (dt.now().isoformat(timespec='seconds'), "PAPER", "AAPL", 100,
             "MARKET", 100, 95, 104, "CNC", "PAPER_WIN", "US_AUTOPILOT:t"))
        conn.commit(); conn.close()
        rc = ua.report_card()
        assert rc["stats"]["n"] == 1
        assert 0 < rc["stats"]["total_pnl"] < 400          # net < gross 400
        assert rc["verdict"] == "COLLECTING_EVIDENCE"      # <30 trades
        ua._account_closed()
        assert ua.get_status()["realized_pnl"] > 0


class TestUSScanner:
    def test_symbol_directory_parser(self):
        """Parse the real NASDAQ Trader file format → clean common stocks
        only (no ETFs, test issues, warrants/units/preferreds, headers)."""
        from data.us_universe import _parse_symbol_file
        nasdaq = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
            "Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
            "QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N\n"          # ETF → out
            "ZTEST|Test Issue|Q|Y|N|100|N|N\n"               # test → out
            "AAPLW|Apple Warrant|Q|N|N|100|N|N\n"            # 5-char but a warrant name; ticker clean → kept (acceptable)
            "BRKA|Berkshire|N|N|N|100|N|N\n"
            "File Creation Time: 0101202699:99|||||||\n")
        out = _parse_symbol_file(nasdaq, "nasdaq")
        assert "AAPL" in out and out["AAPL"].startswith("Apple")
        assert "QQQ" not in out                              # ETF excluded
        assert "ZTEST" not in out                            # test issue excluded
        assert "FILE CREATION" not in " ".join(out).upper()  # footer skipped
        # otherlisted format (NYSE etc.) — different column order
        other = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
                 "Test Issue|NASDAQ Symbol\n"
                 "IBM|International Business Machines|N|IBM|N|100|N|IBM\n"
                 "SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY\n")     # ETF → out
        out2 = _parse_symbol_file(other, "other")
        assert "IBM" in out2 and "SPY" not in out2

    def test_universe_fallback_curated(self, monkeypatch):
        """Directory unreachable → curated liquid list, never empty."""
        import data.us_universe as uu
        monkeypatch.setattr(uu, "_load_cached", lambda: None)
        monkeypatch.setattr(uu, "_fetch_all", lambda: {})   # network down
        u = uu.get_us_universe()
        assert "AAPL" in u and len(u) >= 30                  # curated fallback

    def test_index_constituents_parser(self):
        """Scope to the constituents table + keep only ticker-shaped tokens."""
        from data.us_indices import _extract_symbols
        html = (
            'junk <a>NOPE</a> <table id="other"><td>ZZZZ</td></table>'
            '<table id="constituents">'
            '<tr><td><a href="/q/MMM">MMM</a></td><td>3M</td></tr>'
            '<tr><td><a href="/q/AAPL">AAPL</a></td><td>Apple</td></tr>'
            '<tr><td>lowercase</td><td>TOOLONGTICKER</td></tr>'
            '</table>'
            '<table><td>AFTER</td></table>')          # outside → ignored
        syms = _extract_symbols(html, "constituents")
        assert "MMM" in syms and "AAPL" in syms
        assert "ZZZZ" not in syms                      # before the table
        assert "AFTER" not in syms                     # after </table>
        assert "TOOLONGTICKER" not in syms             # >5 chars

    def test_index_members_fallback_and_source(self, tmp_path, monkeypatch):
        """Live fetch down → curated members with an HONEST source label
        (stale must look stale). Dow 30 curated is complete."""
        import data.us_indices as ui
        monkeypatch.setattr(ui, "_CACHE_FILE", tmp_path / "us_indices.json")
        monkeypatch.setattr(ui, "_fetch_index", lambda index: {})   # network down
        dow, src = ui.get_index_members("Dow 30")
        assert "AAPL" in dow and "NVDA" in dow and len(dow) == 30
        assert "complete" in src
        ndx, src2 = ui.get_index_members("NASDAQ-100")
        assert "MSFT" in ndx and len(ndx) >= 50
        assert "subset" in src2                          # labeled honestly
        assert ui.get_index_members("Bogus")[0] == {}    # unknown index safe

    def test_index_members_prefers_live(self, tmp_path, monkeypatch):
        """A healthy live fetch is used verbatim and marked 'live'."""
        import data.us_indices as ui
        monkeypatch.setattr(ui, "_CACHE_FILE", tmp_path / "us_indices.json")
        fake = {f"S{i:03d}"[:5]: f"Co {i}" for i in range(90)}
        monkeypatch.setattr(ui, "_fetch_index", lambda index: fake)
        m, src = ui.get_index_members("S&P 500")
        assert src == "live" and len(m) == 90

    def test_us_quality_floor_kills_microcaps(self, monkeypatch):
        """'Naam jaanta bhi nahi' fix: $5 price + $10M/day turnover floor —
        micro-cap junk breakout list mein kabhi nahi aata."""
        import scan.us_scanner as us
        monkeypatch.setattr(us, "_US_MIN_TURNOVER_M", 10.0)
        monkeypatch.setattr(us, "_US_MIN_PRICE", 5.0)
        rows = [
            {"symbol": "AAPL", "price": 200.0, "avg_vol20": 50_000_000},  # $10B/d
            {"symbol": "PENY", "price": 2.5, "avg_vol20": 90_000_000},    # penny
            {"symbol": "GHST", "price": 40.0, "avg_vol20": 20_000},       # $0.8M/d
            {"symbol": "MIDC", "price": 25.0, "avg_vol20": 800_000},      # $20M/d
        ]
        kept = us._quality_floor(rows)
        assert [r["symbol"] for r in kept] == ["AAPL", "MIDC"]
        assert kept[0]["turnover_m"] == 10000.0        # card chip data
        # env off-switch: 0 = koi filter nahi
        monkeypatch.setattr(us, "_US_MIN_TURNOVER_M", 0.0)
        monkeypatch.setattr(us, "_US_MIN_PRICE", 0.0)
        assert len(us._quality_floor(rows)) == 4

    def test_scan_scope_by_index(self, monkeypatch):
        """Scanner can scope to one index; unknown/None → full listing, and
        it NEVER scans nothing (fail-safe)."""
        import scan.us_scanner as us
        monkeypatch.setattr("data.us_universe.get_us_universe",
                            lambda: ["AAA", "BBB", "CCC"])
        monkeypatch.setattr("data.us_indices.get_index_members",
                            lambda idx: ({"AAPL": "Apple", "MSFT": "Microsoft"}, "live"))
        syms, scope = us._index_universe("Dow 30")
        assert scope == "Dow 30" and set(syms) == {"AAPL", "MSFT"}
        syms2, scope2 = us._index_universe(None)                 # full listing
        assert scope2 == "All" and set(syms2) == {"AAA", "BBB", "CCC"}
        # empty index members → fail safe to full listing, not an empty scan
        monkeypatch.setattr("data.us_indices.get_index_members",
                            lambda idx: ({}, "curated (subset)"))
        syms3, scope3 = us._index_universe("NASDAQ-100")
        assert scope3 == "All" and syms3 == ["AAA", "BBB", "CCC"]

    def test_same_engine_runs_on_us_shaped_data(self, monkeypatch):
        """The NSE signal engine must analyse a US-shaped DataFrame — the
        whole point of the reuse. Delivery absent → conviction neutral,
        RS benchmarked to the injected S&P return."""
        import numpy as np
        import pandas as pd
        from scan.unified_scanner import UnifiedScanner
        # a clean confirmed breakout out of a tight base, strong volume
        n = 120
        close = np.concatenate([np.full(90, 100.0), np.full(29, 101.0),
                                [106.0]])
        # one small pullback day so RSI isn't pinned at the theoretical 100
        # ceiling (a pure flat-then-flat-then-up series never happens in
        # real data) — keeps this test clear of the RSI hard-reject band
        close[-8] -= 1.5
        high = close + 0.5
        high[-1] = 106.5
        df = pd.DataFrame({
            "open": close - 0.3, "high": high, "low": close - 0.6,
            "close": close, "volume": [1_000_000] * 119 + [3_000_000]},
            index=pd.date_range("2025-01-01", periods=n, freq="D"))
        sc = UnifiedScanner()
        sc._nifty_ret30 = 1.0                            # S&P benchmark (flat)
        r = sc._analyze("AAPL", df)                      # US ticker, no deliv col
        assert r is not None
        assert "Breakout" in {__import__("scan.unified_scanner",
                              fromlist=["SIGNAL_META"]).SIGNAL_META[s][1]
                              for s in r.signals}
        assert r.breakout_conviction > 0                 # engine graded it

    def test_us_telegram_push_dedup(self, monkeypatch):
        """US setups push to Telegram, highest-conviction first, once/stock/day."""
        import scan.us_scanner as us
        sent = []

        class _Eng:
            def is_configured(self): return True
            def send(self, msg): sent.append(msg)
        monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _Eng)
        monkeypatch.setattr(us, "_pushed", {}, raising=False)
        rows = [
            {"symbol": "NVDA", "verdict": "STRONG BUY", "price": 900,
             "entry": 900, "stop": 860, "target": 940, "score": 88,
             "conviction_rank": 290, "breakout_conviction": 82,
             "high_conviction": True, "reasons": ["clean base break"]},
            {"symbol": "AAPL", "verdict": "BUY", "price": 200, "entry": 200,
             "stop": 190, "target": 208, "score": 70, "conviction_rank": 160,
             "reasons": ["momentum"]},
        ]
        us._push_us_setups(rows)
        assert len(sent) == 1
        assert "NVDA" in sent[0] and "AAPL" in sent[0]
        assert sent[0].index("NVDA") < sent[0].index("AAPL")   # conviction order
        # second call same day → nothing new (dedup)
        us._push_us_setups(rows)
        assert len(sent) == 1

    def test_us_scan_serialize_shape(self, monkeypatch):
        """scan_us serializes results like the NSE store (cards reuse)."""
        import scan.us_scanner as us
        import numpy as np
        import pandas as pd
        n = 120
        close = np.concatenate([np.full(90, 100.0), np.full(29, 101.0), [106.0]])
        df = pd.DataFrame({
            "open": close - 0.3, "high": close + 0.5, "low": close - 0.6,
            "close": close, "volume": [1_000_000] * 119 + [3_000_000]},
            index=pd.date_range("2025-01-01", periods=n, freq="D"))
        monkeypatch.setattr("data.us_universe.get_us_universe",
                            lambda: ["AAPL"])
        monkeypatch.setattr("data.us_data.get_us_daily", lambda s, **k: df)
        monkeypatch.setattr("data.us_data.get_us_daily_batch",
                            lambda syms, **k: {s: df for s in syms})
        monkeypatch.setattr("data.us_data.sp500_return_30d", lambda: 0.5)
        out = us.scan_us(max_workers=1)
        # MUST produce results — an internal crash returning [] would hide
        # exactly the 'nothing shows on the terminal' bug
        assert isinstance(out, list) and len(out) >= 1
        r = out[0]
        assert {"symbol", "verdict", "score", "entry", "stop",
                "target"} <= set(r)


class TestMorningBrief:
    def test_conversational_brief_style(self, monkeypatch):
        import ui.command_center as cc
        import datetime as _dt
        # freeze to morning so the greeting is deterministic (suite runs at
        # any hour in CI; the brief's greeting is clock-driven)
        class _MorningDT(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime(2026, 7, 15, 8, 30)
        monkeypatch.setattr(cc, "datetime", _MorningDT)
        monkeypatch.setattr(cc, "_brief_cues", lambda: {
            "sp500": {"price": 5600, "chg": 0.20}, "nasdaq": {"price": 18000, "chg": 0.3},
            "kospi": {"price": 2800, "chg": 7.64}, "crude": {"price": 80.05, "chg": 0.29},
            "gold": {"price": 4031, "chg": -0.55}, "btc": {"price": 64564, "chg": -0.63},
        })
        monkeypatch.setattr(cc, "_top_market_news",
                            lambda n=2: ["Trump signals Gulf nations to invest big in the US"])
        # cache_data wrapper — call the underlying fn
        brief = cc._conversational_brief.__wrapped__(
            {"nifty_price": 24052, "nifty_change_1d": -0.66}) \
            if hasattr(cc._conversational_brief, "__wrapped__") \
            else cc._conversational_brief({"nifty_price": 24052,
                                           "nifty_change_1d": -0.66})
        assert "Good Morning" in brief
        assert "S&P" in brief and "Kospi" in brief.replace("KOSPI", "Kospi")
        assert "reversal" in brief.lower()           # KOSPI +7.6% → reversal
        assert "consolidation" in brief.lower()      # Nifty down → consolidation
        assert "80" in brief                          # crude level
        assert "Trump" in brief                        # market-moving news woven in
        assert "good day" in brief.lower()

    def test_news_ranker_prefers_macro(self):
        from ui.command_center import _rank_news
        headlines = [
            "Company XYZ appoints new marketing head for the region",
            "Trump signals Gulf nations to invest big in the US economy",
            "Fed hints at a rate cut as inflation cools sharply",
            "Local bakery opens third outlet in the suburb",
        ]
        out = _rank_news(headlines, 2)
        assert len(out) == 2
        assert any("Trump" in h or "Fed" in h for h in out)
        assert not any("marketing head" in h or "bakery" in h for h in out)


class TestLLMHealth:
    def test_balance_error_long_cooldown(self, monkeypatch):
        import ai.llm_health as h
        monkeypatch.setattr(h, "_down_until", 0.0, raising=False)
        assert h.available() is True
        h.note_failure("Error code: 402 - Insufficient Balance")
        assert h.available() is False                 # backed off
        st = h.status()
        assert st["available"] is False and st["cooldown_min"] > 60  # hard
        h.note_success()
        assert h.available() is True

    def test_transient_error_short_cooldown(self, monkeypatch):
        import ai.llm_health as h
        monkeypatch.setattr(h, "_down_until", 0.0, raising=False)
        h.note_failure("read timeout")
        st = h.status()
        assert not st["available"] and st["cooldown_min"] <= 10  # soft


class TestPositionCurrency:
    def test_us_position_shows_dollars(self, tmp_path, monkeypatch):
        import risk.position_manager as pm
        import execution.trade_executor as te
        import data.live_quotes as lq
        monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
        te._journal({"mode": "PAPER", "symbol": "AAPL", "qty": 10,
                     "entry_type": "MARKET", "entry_price": 100, "stop_price": 95,
                     "target_price": 104, "product": "CNC",
                     "status": "PAPER_OPEN", "note": "US_AUTOPILOT:t"})
        te._journal({"mode": "PAPER", "symbol": "HAL", "qty": 5,
                     "entry_type": "MARKET", "entry_price": 4500, "stop_price": 4300,
                     "target_price": 4600, "product": "CNC",
                     "status": "PAPER_OPEN", "note": "AUTOPILOT:t"})
        monkeypatch.setattr(lq, "get_live_quotes",
                            lambda syms, ttl=8.0: {"AAPL": {"price": 101.0},
                                                   "HAL": {"price": 4550.0}})
        rows = {r["symbol"]: r for r in pm.review_positions()}
        assert rows["AAPL"]["cur"] == "$"             # US trade → dollars
        assert rows["HAL"]["cur"] == "₹"              # NSE trade → rupees


class TestMarketSession:
    def test_active_market_by_hours(self):
        import pytz
        from datetime import datetime as dt
        from core.market_session import auto_market, resolve_market
        ist = pytz.timezone("Asia/Kolkata")
        # Tue 11:00 IST → NSE open → IN
        assert auto_market(ist.localize(dt(2026, 7, 14, 11, 0))) == "IN"
        # Tue 21:00 IST = 11:30 ET → US open, NSE closed → US
        assert auto_market(ist.localize(dt(2026, 7, 14, 21, 0))) == "US"
        # Tue 05:00 IST → both closed → home default IN
        assert auto_market(ist.localize(dt(2026, 7, 14, 5, 0))) == "IN"
        # weekend → IN default
        assert auto_market(ist.localize(dt(2026, 7, 11, 21, 0))) == "IN"

    def test_manual_override_wins(self):
        import pytz
        from datetime import datetime as dt
        from core.market_session import resolve_market
        ist = pytz.timezone("Asia/Kolkata")
        now = ist.localize(dt(2026, 7, 14, 11, 0))     # NSE open
        assert resolve_market("US", now) == "US"        # override to US
        assert resolve_market("IN", now) == "IN"
        assert resolve_market("AUTO", now) == "IN"      # auto → NSE (open)


class TestUSMarketClock:
    def test_us_session_boundaries(self):
        import pytz
        from datetime import datetime as dt
        from ui.live_watch import _us_market_open
        ny = pytz.timezone("America/New_York")
        # Tue 10:00 NY → open; Tue 09:00 → pre-market; Tue 16:30 → closed
        assert _us_market_open(ny.localize(dt(2026, 7, 14, 10, 0))) is True
        assert _us_market_open(ny.localize(dt(2026, 7, 14, 9, 0))) is False
        assert _us_market_open(ny.localize(dt(2026, 7, 14, 16, 30))) is False
        # weekend
        assert _us_market_open(ny.localize(dt(2026, 7, 12, 12, 0))) is False


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
