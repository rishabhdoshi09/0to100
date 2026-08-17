"""Thesis hold — keep winners while technicals + fundamentals look good."""
from __future__ import annotations

from execution.thesis_hold import (
    clamp_rsi_protect_pct,
    evaluate_thesis,
    protective_stop,
    rsi_is_extended,
    runner_target,
)


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


def test_evaluate_thesis_holds_on_high_rsi():
    """Open-trade RSI spike is not a thesis break — protect, don't sell."""
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=108,
        scan_row={"rsi": 85, "verdict": "BUY", "status": "Ready to trade",
                  "momentum_5d": 3.0, "chase_risk": False},
    )
    assert ok and why == ""
    assert rsi_is_extended({"rsi": 85})
    assert not rsi_is_extended({"rsi": 65})
    assert not rsi_is_extended(None)


def test_evaluate_thesis_breaks_on_failed_setup():
    ok, why = evaluate_thesis(
        entry=100, stop=95, live_px=102,
        scan_row={"rsi": 85, "status": "Wait for pullback", "verdict": "WATCH"},
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


def test_protective_stop_never_loosens_or_crosses_ltp():
    # Entry 100, stop 96, LTP 110 → ~107.25, hold
    assert protective_stop(110, 96, 2.5) == 107.2
    # LTP 101, RSI high → 98.5 tighter than 96
    assert protective_stop(101, 96, 2.5) == 98.5
    # LTP 98 → 2.5% down is below original stop → keep 96
    assert protective_stop(98, 96, 2.5) == 96.0
    # Already trailed tighter than the band → never loosen
    assert protective_stop(108, 107, 2.5) == 107.0
    assert clamp_rsi_protect_pct(1.0) == 2.0
    assert clamp_rsi_protect_pct(9.0) == 3.0
    assert clamp_rsi_protect_pct(None) == 2.5


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
                       signal_target=1120, volume_ratio=1.5) is True
    t = te.recent_trades(1)[0]
    # Runner ceiling / signal target — NOT a +3% scalp cut
    assert float(t["target_price"]) >= 1100.0


def test_autopilot_high_rsi_holds_and_tightens_gtt(tmp_path, monkeypatch):
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
                  profit_book_pct=0.0, profit_book_rupees=0.0,
                  rsi_protect_pct=2.5)
    ap.arm()
    assert ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t",
                       volume_ratio=1.5) is True
    prices["HAL"] = 1100.0
    monkeypatch.setattr(
        ap, "_lookup_thesis_context",
        lambda sym: ({"rsi": 85, "verdict": "BUY", "status": "Ready to trade",
                      "momentum_5d": 4.0, "chase_risk": False}, None),
    )
    ap._protect_extended_rsi()
    ap._thesis_exits()
    t = te.recent_trades(1)[0]
    assert t["status"] == "PAPER_OPEN"
    assert "thesis-exit" not in (t.get("note") or "")
    assert abs(float(t["stop_price"]) - round(1100 * 0.975, 1)) < 0.05
    assert "RSI-protect GTT" in (t.get("note") or "")
    assert float(t.get("orig_stop") or 0) == 960.0
    # Second pass is idempotent — stop does not loosen if LTP dips a bit
    prices["HAL"] = 1090.0
    ap._protect_extended_rsi()
    t2 = te.recent_trades(1)[0]
    assert float(t2["stop_price"]) == float(t["stop_price"])
    # Higher LTP ratchets the stop up, never back down
    prices["HAL"] = 1200.0
    ap._protect_extended_rsi()
    t3 = te.recent_trades(1)[0]
    assert abs(float(t3["stop_price"]) - round(1200 * 0.975, 1)) < 0.05
    assert t3["status"] == "PAPER_OPEN"


def test_autopilot_thesis_exit_on_broken_setup_even_with_high_rsi(
        tmp_path, monkeypatch):
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
    assert ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t",
                       volume_ratio=1.5) is True
    prices["HAL"] = 1050.0
    monkeypatch.setattr(
        ap, "_lookup_thesis_context",
        lambda sym: ({"rsi": 85, "verdict": "WATCH",
                      "status": "Wait for pullback"}, None),
    )
    ap._protect_extended_rsi()
    ap._thesis_exits()
    t = te.recent_trades(1)[0]
    assert t["status"] == "PAPER_WIN"
    assert "thesis-exit" in (t.get("note") or "")
    assert "RSI-protect" not in (t.get("note") or "")


def test_autopilot_live_rsi_protect_places_gtt_when_missing(tmp_path, monkeypatch):
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
    # consider() still places PAPER; we flip the row to LIVE after entry
    prices = {"HAL": 1000.0}
    monkeypatch.setattr(
        "data.live_quotes.get_live_quotes",
        lambda syms: {s: {"price": prices.get(s, 1000.0)} for s in syms},
    )
    ap.set_config(allocation=100000, mode="PAPER", thesis_hold=True,
                  profit_book_pct=0.0, profit_book_rupees=0.0)
    ap.arm()
    assert ap.consider("HAL", 1000, 960, 80, 0.2, "Defence", "t",
                       volume_ratio=1.5) is True
    t = te.recent_trades(1)[0]
    ap._update_trade(t["id"], "mode=?, status=?, gtt_id=?",
                     ("LIVE", "PLACED", ""))
    placed = {}

    def _fake_place(t, new_stop, ltp):
        placed["stop"] = new_stop
        placed["ltp"] = ltp
        t["gtt_id"] = "gtt-99"
        return True

    monkeypatch.setattr(ap, "_ensure_live_protective_gtt", _fake_place)
    prices["HAL"] = 1100.0
    monkeypatch.setattr(
        ap, "_lookup_thesis_context",
        lambda sym: ({"rsi": 82, "verdict": "BUY", "status": "Ready to trade",
                      "momentum_5d": 2.0, "chase_risk": False}, None),
    )
    ap._protect_extended_rsi()
    row = te.recent_trades(1)[0]
    assert row["status"] == "PLACED"
    assert abs(placed["stop"] - round(1100 * 0.975, 1)) < 0.05
    assert abs(float(row["stop_price"]) - placed["stop"]) < 0.05
    assert str(row["gtt_id"]) == "gtt-99"
