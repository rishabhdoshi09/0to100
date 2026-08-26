"""Retail-completion tests: persistent scans, market translation and honest funnels."""
from datetime import datetime, timezone

from product.market_view import build_market_view
from product.no_trade import build_no_trade_explanation
from product.retail_backtest import BacktestRequest, interpret_result
from product.scan_store import build_scan_payload, load_scan, save_scan, watchlist_rows


class Signal:
    def __init__(self, symbol, *, signals, verdict="WATCH", chase=False, score=60):
        self.symbol = symbol; self.signals = signals; self.verdict = verdict
        self.chase_risk = chase; self.score = score; self.reasons = ["plain reason"]
        self.price = 100; self.momentum_5d = 7; self.rsi = 62; self.volume_ratio = 1.6
        self.entry = 101; self.stop = 95; self.target = 113


def test_saved_scan_preserves_full_universe_and_watchlist(tmp_path):
    payload = build_scan_payload(
        {"AAA": "Alpha", "BBB": "Beta", "CCC": "Gamma"},
        [Signal("AAA", signals=["MOMENTUM"], verdict="BUY", score=80),
         Signal("BBB", signals=["PRE_BREAKOUT"], score=70),
         Signal("CCC", signals=["MOMENTUM"], chase=True, score=90)],
        {"AAA", "CCC"}, scanned_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    path = tmp_path / "scan.json"; save_scan(payload, path)
    loaded = load_scan(path)
    assert loaded["universe_size"] == 3
    assert loaded["summary"]["momentum"] == 2
    assert [row["symbol"] for row in watchlist_rows(loaded)] == ["AAA", "BBB", "CCC"]
    assert watchlist_rows(loaded)[0]["status"] == "Ready to trade"


def test_saved_scan_keeps_breakout_grade_and_change_pct():
    class Graded(Signal):
        def __init__(self):
            super().__init__("GRD", signals=["BREAKOUT_52W"], verdict="BUY", score=82)
            self.change_pct = 2.4
            self.breakout_grade = "A"
            self.above_sma50 = True
            self.avg_vol20 = 800000

    payload = build_scan_payload({"GRD": "Graded"}, [Graded()])
    row = payload["records"][0]
    assert row["breakout_grade"] == "A"
    assert row["change_pct"] == 2.4
    assert row["above_sma50"] is True
    assert row["avg_vol20"] == 800000


def test_market_view_uses_plain_language():
    view = build_market_view({
        "regime_score": 72, "risk_mode": "RISK_ON", "breakout_environment": "FAVORABLE",
        "breadth_label": "STRONG", "breadth_strength": 68,
        "leading_sectors": ["AUTO", "BANK"], "lagging_sectors": ["IT"],
        "nifty_change_1d": 0.7, "nifty_change_5d": 2.1, "vix": 13.4,
    })
    assert view.health == "Healthy"
    assert "New paper trades are allowed" in view.trade_stance
    assert "Auto" in view.summary or "AUTO" in view.summary


def test_no_trade_funnel_never_invents_unknown_proposal_count():
    payload = {"universe_size": 1842, "summary": {"with_any_setup": 214, "momentum": 36, "ready_to_trade": 4}}
    explanation = build_no_trade_explanation(payload, [("AAA", "total risk cap"), ("BBB", "total risk cap")], {})
    counts = {stage.label: stage.count for stage in explanation.stages}
    assert counts["Stocks scanned"] == 1842
    assert counts["Backend trade proposals"] is None
    assert explanation.top_reasons[0] == ("total risk cap", 2)


def test_backtest_interpretation_is_negative_when_strategy_loses():
    request = BacktestRequest(capital=100_000)
    result = {
        "final_equity": 90_000,
        "trade_journal": [{"action": "SELL", "realized_pnl": -10_000}],
        "fills": [{"transaction_cost": 100, "slippage_cost": 50}],
        "equity_curve": [],
    }
    summary = interpret_result(request, result, {"max_drawdown_pct": 15}, requested=1, loaded=1, nifty_return=5)
    assert summary.profit_loss == -10_000
    assert summary.trading_costs == 150
    assert "lost money" in summary.conclusion.lower()
    assert "Underperformed" in summary.comparison
