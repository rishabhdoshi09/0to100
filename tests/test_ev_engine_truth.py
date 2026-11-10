"""EV evidence truth (2026-11-11 audit of production signal_outcomes.db).

The engine summed per-signal counts, so a setup with three signals claimed
n=4564 from 2884 closed trades; 59% of rows were the same stock re-logged
inside its own holding window; all outcomes came from 15 July sessions and
were 3.5 months old, yet every card said HIGH confidence.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest


def _seed(tmp_path, monkeypatch, rows):
    """rows: (symbol, logged_at datetime, archetype, outcome_pct)."""
    import core.signal_outcome_tracker as tk

    monkeypatch.setattr(tk, "_DB_PATH", str(tmp_path / "sig.db"))
    conn = tk._get_conn()
    for symbol, when, arche, pct in rows:
        la = when.isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO signal_log (symbol,logged_at,signal_type,archetype,entry_price,"
            "stop_price,quality_score,outcome_pct,worked,outcome_checked_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (symbol, la, "UNIFIED_BUY", arche, 100.0, 97.0, 70.0, pct, 1 if pct > 0 else 0, la))
    conn.commit()
    conn.close()


def _spread(n, arche, pct, *, start=datetime(2026, 9, 1), sessions=40, prefix="S"):
    return [(f"{prefix}{i}", start + timedelta(days=i % sessions), arche, pct) for i in range(n)]


def test_multi_signal_trade_counts_once(tmp_path, monkeypatch):
    from scan.ev_engine import estimate_ev

    _seed(tmp_path, monkeypatch, _spread(40, "A|B|C", 6.0) + _spread(20, "A|B|C", -3.0, prefix="L"))
    ev = estimate_ev(["A", "B", "C"], 100, 97, today=date(2026, 10, 20))
    assert ev["n"] == 60                       # not 180


def test_relogged_symbol_inside_window_is_one_trade(tmp_path, monkeypatch):
    from scan.ev_engine import outcome_table

    day = datetime(2026, 9, 1)
    _seed(tmp_path, monkeypatch, [
        ("PNB", day, "A", 5.0),
        ("PNB", day + timedelta(days=1), "A", 5.0),   # weekend/next-day re-log
        ("PNB", day + timedelta(days=3), "A", 5.0),
        ("PNB", day + timedelta(days=9), "A", -3.0),  # a new holding window
    ])
    assert [t["day"] for t in outcome_table()] == ["2026-09-01", "2026-09-10"]


def test_one_regime_window_is_never_high_confidence(tmp_path, monkeypatch):
    from scan.ev_engine import estimate_ev

    # 1,000 distinct trades but only 15 sessions (one three-week tape).
    rows = _spread(700, "A", -3.0, sessions=15) + _spread(300, "A", 6.0, sessions=15, prefix="W")
    _seed(tmp_path, monkeypatch, rows)
    ev = estimate_ev(["A"], 100, 97, today=date(2026, 9, 20))
    assert ev["n"] == 1000 and ev["sessions"] == 15
    assert ev["confidence"] == "LOW"


def test_stale_evidence_is_flagged_and_low(tmp_path, monkeypatch):
    from scan.ev_engine import STALE_AFTER_DAYS, estimate_ev

    _seed(tmp_path, monkeypatch, _spread(60, "A", -3.0, start=datetime(2026, 7, 1)))
    last = date(2026, 7, 1) + timedelta(days=39)
    fresh = estimate_ev(["A"], 100, 97, today=last + timedelta(days=STALE_AFTER_DAYS))
    stale = estimate_ev(["A"], 100, 97, today=last + timedelta(days=STALE_AFTER_DAYS + 1))
    assert fresh["stale"] is False
    assert stale["stale"] is True and stale["confidence"] == "LOW"
    assert stale["last_outcome"] == last.isoformat()


def test_effective_n_shrinks_when_sessions_move_together():
    from scan.ev_engine import effective_n

    # Every session is all-win or all-loss: each session is one draw.
    days = [f"d{i // 10}" for i in range(100)]
    wins = [1 if (i // 10) % 2 else 0 for i in range(100)]
    n_eff, sessions = effective_n(days, wins)
    assert sessions == 10 and n_eff == pytest.approx(10.0, rel=0.05)
    # Independent mixing inside sessions keeps (almost) the full sample.
    mixed = [i % 2 for i in range(100)]
    assert effective_n(days, mixed)[0] > 90


def test_stale_ev_card_says_unproven_not_negative():
    from product.decision_card import expected_payoff
    from product.reco_methods import _ev_method

    row = {"ev_n": 1174, "ev_lb_pct": -2.1, "ev_conf": "LOW",
           "ev_stale": True, "ev_last_outcome": "2026-07-23"}
    label, detail = expected_payoff(row)
    assert label == "Unproven" and "2026-07-23" in detail
    assert _ev_method(row)["status"] == "unknown"
    fresh = dict(row, ev_stale=False)
    assert expected_payoff(fresh)[0] == "Negative"
    assert _ev_method(fresh)["status"] == "fail"


def test_tag_ev_carries_freshness_fields(tmp_path, monkeypatch):
    from scan.ev_engine import tag_ev

    _seed(tmp_path, monkeypatch, _spread(40, "BREAKOUT_52W", 6.0, start=datetime(2026, 3, 1)))
    rows = [{"symbol": "X", "signals": ["BREAKOUT_52W"], "entry": 100, "stop": 97}]
    tag_ev(rows)
    assert rows[0]["ev_n"] == 40
    assert rows[0]["ev_stale"] is True and rows[0]["ev_last_outcome"]
    assert rows[0]["ev_conf"] == "LOW"
