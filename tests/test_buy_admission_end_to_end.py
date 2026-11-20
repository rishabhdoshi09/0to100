"""End-to-end BUY admission: scanner -> scan record -> recommendation tier -> paper -> Home -> Telegram.

Audit (2026-11-20, review of b60df3b2). The only producer of the "BUY"
verdict and of status "Ready to trade" is the live scanner's evidence gate
(scan.unified_scanner._evidence_verdict -> product.scan_store._record).
Every downstream consumer keys on one of the two:
- recommendation tiers via reco_ensemble.entry_state (READY needs "Ready to
  trade", or grade A/B with verdict BUY);
- paper autopilot via reco_tier;
- Home's BUY-only section and Telegram's "qualified setups" via verdict BUY.
So if the gate holds, nothing downstream can label or open a BUY.

Separate lanes, deliberately not a BUY:
- the paper learning lane (EXPLORATION: edge-gated cards, 1 per day, 0.5%
  risk, labelled, QT_PAPER_EXPLORATION=0 disables);
- F&O paper (its own directional scanner; no historical option evidence
  exists, labelled as a paper research candidate);
- the live-tick sniper alert, which now says WATCH/no proven edge for an
  unproven row instead of giving an entry plan.
The legacy breakout_sniper/auto_scan and the Telegram tap-to-paper listener
are not started by the canonical stack.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scan import unified_scanner as US

INF, NAN = float("inf"), float("nan")
FRESH = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _frame():
    close = np.linspace(100, 200, 260)
    close[-3] = close[-4] - 2.0
    close[-8] = close[-9] - 2.0
    close[-12] = close[-13] - 2.0
    close[-1] = 212
    vol = np.full(260, 2e6)
    vol[-1] = 5e6
    end = pd.offsets.BDay().rollback(pd.Timestamp(date.today()))
    idx = pd.date_range(end=end, periods=260, freq="B")
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99, "volume": vol}, index=idx)


def _ok(mean=0.30, **kw):
    row = {"mean": mean, "lower": 0.10, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST", "family": True}
    row.update(kw)
    return row


def _snap(edges, generated_at=FRESH):
    import scan.signal_backtest as SB
    from core.evaluation_spec import HORIZON_SESSIONS
    snap = {"snapshot_id": "e2e", "multipliers": {}, "signal_registry": {}, "backtest_edges": edges}
    if generated_at is not None:
        from tests.evidence_fixtures import proven_provenance
        snap["backtest_provenance"] = proven_provenance(generated_at)
    return snap


SIGNALS = ("BREAKOUT_52W", "VOL_SQUEEZE", "POCKET_PIVOT")
CONDITIONS = {
    "stale_snapshot": _snap({s: _ok() for s in SIGNALS}, generated_at="2026-01-02 10:00"),
    "undated_snapshot": _snap({s: _ok() for s in SIGNALS}, generated_at=None),
    "empty_evidence_for_these_signals": _snap({"OTHER_SIGNAL": _ok()}),
    "mixed_legacy_and_new": _snap({"BREAKOUT_52W": 0.9, "VOL_SQUEEZE": _ok(), "POCKET_PIVOT": _ok()}),
    "nan_mean": _snap({s: _ok(mean=NAN) for s in SIGNALS}),
    "inf_lower": _snap({s: _ok(lower=INF) for s in SIGNALS}),
    "low_n": _snap({s: _ok(n=12) for s in SIGNALS}),
    "too_few_independent_blocks": _snap({s: _ok(blocks=6) for s in SIGNALS}),
    "pre_rerun_bound": _snap({s: _ok(inference="") for s in SIGNALS}),
    "fails_family_correction": _snap({s: _ok(family=False) for s in SIGNALS}),
}


def _pipeline(scanner):
    from product import paper_autopilot as PA
    from product.reco_ensemble import ENTRY_READY, attach_ensemble, entry_state
    from product.scan_store import _record

    r = scanner._analyze("X", _frame())
    assert r is not None
    row = _record(r, {}, set())
    card = attach_ensemble(row)
    decision = PA.evaluate_candidate(card, book=None)
    return r, row, card, decision, entry_state(row) == ENTRY_READY


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_unproven_evidence_never_reaches_a_buy_label_or_an_entry(name):
    from product import paper_autopilot as PA

    r, row, card, decision, ready = _pipeline(US.UnifiedScanner(calibration_snapshot=CONDITIONS[name]))
    assert r.verdict == "WATCH" and r.edge_gated, name
    assert row["status"] != "Ready to trade" and not ready
    assert card["reco_tier"] not in PA.ELIGIBLE_TIERS
    assert decision.decision != PA.ENTER_NOW
    assert r.reasons[-1].startswith("⚖"), r.reasons[-1]                # the honest reason is on the card


def test_the_live_gate_with_no_report_loaded_fails_closed(monkeypatch):
    import scan.signal_backtest as SB
    monkeypatch.setattr(SB, "load_report", lambda: None)
    r, row, *_ = _pipeline(US.UnifiedScanner(calibration_snapshot={"snapshot_id": "none", "multipliers": {},
                                                                   "signal_registry": {}}))
    assert r.verdict == "WATCH" and row["status"] != "Ready to trade"
    assert "no walk-forward evidence is loaded" in r.reasons[-1]


def test_fully_verified_fresh_evidence_is_the_only_way_to_ready():
    r, row, card, _decision, ready = _pipeline(US.UnifiedScanner(calibration_snapshot=_snap({s: _ok() for s in SIGNALS})))
    assert r.verdict == "BUY" and row["status"] == "Ready to trade" and ready


def test_no_market_data_produces_no_card():
    sc = US.UnifiedScanner(calibration_snapshot=_snap({s: _ok() for s in SIGNALS}))
    assert sc._analyze("X", _frame().iloc[:30]) is None
    assert sc._analyze("X", _frame().iloc[0:0]) is None


def test_the_gate_does_not_depend_on_the_holiday_calendar(monkeypatch):
    import research.intelligence.data.nse_calendar as CAL
    monkeypatch.setattr(CAL, "load_holidays", lambda: set())
    unproven = US.UnifiedScanner(calibration_snapshot=CONDITIONS["low_n"])._analyze("X", _frame())
    proven = US.UnifiedScanner(calibration_snapshot=_snap({s: _ok() for s in SIGNALS}))._analyze("X", _frame())
    assert unproven.verdict == "WATCH" and proven.verdict == "BUY"


def test_home_buy_only_and_telegram_setups_take_only_gated_buys(tmp_path):
    from research.autonomy.telegram_notifications import TelegramNotifier
    from product.scan_store import _record

    class Engine:
        messages: list = []

        def is_configured(self):
            return True

        def send(self, message, reply_markup=None):
            self.messages.append(message)
            return True

    watch = _record(US.UnifiedScanner(calibration_snapshot=CONDITIONS["low_n"])._analyze("WATCHX", _frame()), {}, set())
    buy = _record(US.UnifiedScanner(calibration_snapshot=_snap({s: _ok() for s in SIGNALS}))._analyze("BUYX", _frame()),
                  {}, set())
    eng = Engine()
    n = TelegramNotifier(tmp_path, engine_factory=lambda: eng, now_fn=lambda: datetime(2026, 11, 20, 10, 0))
    out = n.notify_scan({"records": [watch, buy], "summary": {}}, phase="intraday")
    assert out["setup"] == 1
    text = "\n".join(eng.messages)
    assert "BUYX" in text.split("Breakout ke kareeb")[0]
    assert "WATCHX" not in text.split("Breakout ke kareeb")[0]


def test_a_live_cross_on_an_unproven_row_is_announced_as_research_not_an_entry(tmp_path):
    from research.autonomy.telegram_notifications import TelegramNotifier

    class Engine:
        def __init__(self):
            self.messages = []

        def is_configured(self):
            return True

        def send(self, message, reply_markup=None):
            self.messages.append(message)
            return True

    class Live:
        def entry_allowed(self, symbol):
            return True

        def price(self, symbol):
            return 101.0

    rows = [{"symbol": "GATED", "status": "Watch for breakout", "verdict": "WATCH", "edge_gated": True,
             "entry": 100.0, "stop": 95, "target": 110, "score": 80, "rsi": 60, "volume_ratio": 2.0,
             "signals": ["PRE_BREAKOUT"]},
            {"symbol": "PROVEN", "status": "Ready to trade", "verdict": "BUY",
             "entry": 100.0, "stop": 95, "target": 110, "score": 79, "rsi": 60, "volume_ratio": 2.0,
             "signals": ["MOMENTUM"]}]
    eng = Engine()
    clock = [1000.0]
    n = TelegramNotifier(tmp_path, engine_factory=lambda: eng, now_fn=lambda: datetime(2026, 11, 20, 10, 15),
                         epoch_fn=lambda: clock[0], breakout_confirmation_s=5, breakout_buffer_bps=10)
    n.observe_live_breakouts({"records": rows}, Live())
    clock[0] += 6
    n.observe_live_breakouts({"records": rows}, Live())
    msg = eng.messages[-1]
    gated, proven = msg.split("<b>GATED</b>")[1].split("<b>PROVEN</b>")[0], msg.split("<b>PROVEN</b>")[1]
    assert "no proven edge" in gated and "PAPER plan" not in gated
    assert "PAPER plan" in proven


def test_fo_paper_candidates_are_labelled_unproven_research():
    import inspect
    from research.autonomy import telegram_notifications as TN
    assert "No proven edge: F&amp;O paper is an unvalidated research lane" in inspect.getsource(TN)


def test_the_exploration_lane_is_the_only_paper_route_for_a_gated_card_and_it_is_labelled():
    import inspect
    from product import paper_autopilot as PA
    src = inspect.getsource(PA._exploration_pick)
    assert 'card.get("edge_gated")' in src and "EXPLORATION" in src
    assert PA.EXPLORATION_MAX_PER_DAY == 1 and PA.EXPLORATION_RISK_PCT == 0.5
