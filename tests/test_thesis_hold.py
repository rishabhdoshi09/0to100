"""Thesis hold — keep winners while technicals + fundamentals look good."""
from __future__ import annotations

from execution.thesis_hold import evaluate_thesis, runner_target


def test_runner_target_uses_wide_ceiling_and_signal():
    assert runner_target(100, 0, 10) == 110.0
    assert runner_target(100, 118, 10) == 118.0
    assert runner_target(100, 105, 10) == 110.0  # ceiling wins over small signal


def test_evaluate_thesis_holds_when_healthy():
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=108,
        scan_row={"rsi": 55, "verdict": "BUY", "status": "Ready to trade",
                  "momentum_5d": 2.0, "chase_risk": False},
        fund_row={"fundamental_coverage": 0.8, "classification": "QUALITY_COMPOUNDER",
                  "fundamental_score": 70},
    )
    assert ok and why == ""


def test_evaluate_thesis_breaks_on_rsi_blowoff():
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=108,
        scan_row={"rsi": 82, "verdict": "BUY", "status": "Ready to trade"},
    )
    assert not ok and "RSI" in why


def test_evaluate_thesis_breaks_on_failed_setup():
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=102,
        scan_row={"rsi": 50, "status": "Wait for pullback", "verdict": "WATCH"},
    )
    assert not ok and "pullback" in why.lower()


def test_evaluate_thesis_breaks_on_fund_avoid():
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=110,
        scan_row=None,
        fund_row={"fundamental_coverage": 0.9, "classification": "AVOID_REVIEW",
                  "fundamental_score": 20},
    )
    assert not ok and "AVOID_REVIEW" in why


def test_autopilot_thesis_hold_uses_runner_not_scalp(tmp_path, monkeypatch):
    import execution.autopilot as ap
    import execution.trade_executor as te
    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
    monkeypatch.setattr(te, "_DB", tmp_path / "t.db")
    ap._state = {}
    monkeypatch.setattr(ap, "_notify", lambda m: None)
    monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
    monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
    monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
    monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
    monkeypatch.setattr(ap, "_top_sectors", lambda: (["Defence"], "ok"))
    monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
    monkeypatch.setattr(
        "data.live_quotes.get_live_quotes",
        lambda syms: {s: {"price": 1000.0} for s in syms},
    )
    ap.set_config(allocation=100000, mode="PAPER", thesis_hold=True,
                  runner_target_pct=10.0, profit_book_pct=0.0)
    ap.arm()
    assert ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t",
                       signal_target=1120) is True
    t = te.recent_trades(1)[0]
    # Runner ceiling / signal target — NOT a +3% scalp cut
    assert float(t["target_price"]) >= 1100.0


def test_autopilot_thesis_exit_on_rsi_blowoff(tmp_path, monkeypatch):
    import execution.autopilot as ap
    import execution.trade_executor as te
    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "ap.json")
    monkeypatch.setattr(te, "_DB", tmp_path / "t.db")
    ap._state = {}
    monkeypatch.setattr(ap, "_notify", lambda m: None)
    monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
    monkeypatch.setattr(ap, "_serial_losers_cached", lambda: set())
    monkeypatch.setattr(ap, "_brain_posture", lambda: ("NORMAL", ""))
    monkeypatch.setattr(ap, "_in_window", lambda now=None: True)
    monkeypatch.setattr(ap, "_top_sectors", lambda: (["Defence"], "ok"))
    monkeypatch.setattr(ap, "_market_regime", lambda: "TRENDING_BULL")
    monkeypatch.setattr(ap, "_anchor_live",
                        lambda sym, entry, stop, mc: (entry, ""))
    prices = {"HAL": 1000.0}
    monkeypatch.setattr(
        "data.live_quotes.get_live_quotes",
        lambda syms: {s: {"price": prices.get(s, 1000.0)} for s in syms},
    )
    ap.set_config(allocation=100000, mode="PAPER", thesis_hold=True,
                  profit_book_pct=0.0, profit_book_rupees=0.0)
    ap.arm()
    assert ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t") is True
    prices["HAL"] = 1050.0
    monkeypatch.setattr(
        ap, "_lookup_thesis_context",
        lambda sym: ({"rsi": 85, "verdict": "BUY", "status": "Ready to trade"}, None),
    )
    ap._thesis_exits()
    t = te.recent_trades(1)[0]
    assert t["status"] == "PAPER_WIN"
    assert "thesis-exit" in (t.get("note") or "")
