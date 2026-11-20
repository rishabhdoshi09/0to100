"""Operational acceptance of a legitimate NO_ELIGIBLE_BUY across Home, Opportunities, Telegram and paper.

Under the current evidence standard no BUY is admissible. Every surface must
then say so honestly, never fabricate a BUY, and keep the paper EXPLORATION
lane distinct from qualified trades, whether the evidence is fresh, stale, or
partial.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from scan import unified_scanner as US


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


def _snap(edges, prov):
    from tests.evidence_fixtures import seal_snapshot
    return seal_snapshot({"snapshot_id": "acc", "multipliers": {}, "signal_registry": {}, "backtest_edges": edges,
                          "backtest_provenance": prov})


def _real_negative_edges():
    """Shaped like today's production evidence: every signal measured negative."""
    row = lambda m: {"mean": m, "lower": m - 0.1, "n": 900, "blocks": 25, "inference": "OVERLAP_ROBUST",
                     "family": False}
    return {"BREAKOUT_52W": row(-0.17), "VOL_SQUEEZE": row(-0.06), "POCKET_PIVOT": row(-0.13)}


@pytest.fixture
def fresh_negative_scan():
    from product.scan_store import _record, build_scan_payload
    from tests.evidence_fixtures import proven_provenance
    sc = US.UnifiedScanner(calibration_snapshot=_snap(_real_negative_edges(), proven_provenance()))
    results = [sc._analyze(sym, _frame()) for sym in ("AAA", "BBB", "CCC")]
    return build_scan_payload({s: s for s in ("AAA", "BBB", "CCC")}, results, freshness={}), [_record(r, {}, set())
                                                                                             for r in results]


def test_a_fresh_scan_with_negative_evidence_produces_no_buy_anywhere(fresh_negative_scan, tmp_path):
    from datetime import timezone
    from product import paper_autopilot as PA
    from product.reco_ensemble import attach_ensemble
    from product.recommendations_workspace import build_recommendations_workspace
    from research.autonomy.telegram_notifications import TelegramNotifier

    payload, rows = fresh_negative_scan
    assert all(r["verdict"] == "WATCH" and r["edge_gated"] for r in rows)
    assert all(r["reasons"][-1].startswith("⚖") for r in rows)                  # honest reason on every card
    for card in (attach_ensemble(r) for r in rows):                               # paper: no qualified entry
        assert PA.evaluate_candidate(card, book=None).decision != PA.ENTER_NOW
    work = build_recommendations_workspace(scan_payload=payload, long_term_payload={"records": []})
    badges = [c.get("action_badge") for cat in work.get("categories") or [] for c in cat.get("cards") or []]
    assert "Buy" not in badges                                                    # Opportunities

    class Engine:
        def __init__(self):
            self.messages = []

        def is_configured(self):
            return True

        def send(self, message, reply_markup=None):
            self.messages.append(message)
            return True

    eng = Engine()
    out = TelegramNotifier(tmp_path, engine_factory=lambda: eng,
                           now_fn=lambda: datetime(2026, 11, 20, 10, 0)).notify_scan(payload, phase="intraday")
    assert out.get("setup", 0) == 0                                               # Telegram: no qualified setup
    assert all("qualified setups" not in m for m in eng.messages)


def test_home_reports_no_eligible_buy_and_the_rule(monkeypatch):
    import product.home_os as H
    monkeypatch.setattr(H, "_scan_evidence_index", lambda: {
        "by_symbol": {}, "withheld": [{"symbol": "AAA", "edge_r": -0.06}], "withheld_count": 3, "scanner_buys": 0,
        "session": "2026-11-19", "scanned_at": "", "evidence_note": "", "read_time_withheld": 0})
    out = H.build_buy_only([], verified=True)
    assert out["admission_state"] == "NO_ELIGIBLE_BUY" and out["rows"] == []
    assert "No BUY today" in out["empty_reason"] and "WATCH" in out["empty_reason"]


def test_a_stale_artifact_is_withheld_at_read_time_even_if_it_says_buy(tmp_path, monkeypatch):
    import product.scan_store as SS
    stale = {"schema_version": 2, "evidence_identity": {"id": "old"},
             "records": [{"symbol": "AAA", "verdict": "BUY", "status": "Ready to trade", "edge_evidence_id": "old",
                          "signals": ["VOL_SQUEEZE", "POCKET_PIVOT"], "reasons": ["old"]}]}
    path = SS.save_scan(stale, tmp_path / "scan.json")
    from tests.evidence_fixtures import proven_provenance
    monkeypatch.setattr(SS, "_evidence_authority", lambda: _snap(_real_negative_edges(), proven_provenance()))
    row = SS.load_scan(path)["records"][0]
    assert row["verdict"] == "WATCH" and row["status"] == "Watch" and row["edge_gated"]


@pytest.mark.parametrize("drop", ["alpha_accounting", "code_sha", "family_method", "generated_at"])
def test_partial_evidence_never_certifies(drop):
    from tests.evidence_fixtures import proven_provenance
    good = {"BREAKOUT_52W": {"mean": 0.4, "lower": 0.2, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST",
                             "family": True},
            "VOL_SQUEEZE": {"mean": 0.4, "lower": 0.2, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST",
                            "family": True},
            "POCKET_PIVOT": {"mean": 0.4, "lower": 0.2, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST",
                             "family": True}}
    prov = proven_provenance()
    prov.pop(drop)
    r = US.UnifiedScanner(calibration_snapshot=_snap(good, prov))._analyze("X", _frame())
    assert r.verdict == "WATCH" and r.edge_gated


def test_the_exploration_lane_stays_labelled_and_separate():
    import inspect
    from product import paper_autopilot as PA
    src = inspect.getsource(PA._exploration_pick)
    assert '"lane": EXPLORATION' in src and "EXPLORATION_RISK_PCT" in src
    assert PA.EXPLORATION not in PA.ELIGIBLE_TIERS


# ── distinct, actionable no-BUY statuses (product.buy_admission) ──────────────

def test_the_four_causes_of_no_buy_are_distinct_and_actionable():
    from product import buy_admission as BA
    cases = {
        BA.SCAN_NOT_CURRENT: BA.classify(buys=0, scan_current=False, evidence_note="x", withheld=3,
                                         login_required=True, scan_session="2026-11-17",
                                         expected_session="2026-11-19"),
        BA.EVIDENCE_UNAVAILABLE: BA.classify(buys=0, scan_current=True, evidence_note="the evidence is 30 days old",
                                             withheld=3),
        BA.NO_PROVEN_EDGE: BA.classify(buys=0, scan_current=True, withheld=3, closest="AAA at -0.06R/trade"),
        BA.NO_SETUP: BA.classify(buys=0, scan_current=True),
        BA.ELIGIBLE_BUY: BA.classify(buys=2, scan_current=True),
    }
    assert {k: v["code"] for k, v in cases.items()} == {k: k for k in cases}
    assert len({v["headline"] for v in cases.values()}) == len(cases)
    assert "main.py login" in cases[BA.SCAN_NOT_CURRENT]["action"]
    assert "2026-11-17" in cases[BA.SCAN_NOT_CURRENT]["detail"]
    assert "walk-forward" in cases[BA.EVIDENCE_UNAVAILABLE]["action"]
    assert "AAA" in cases[BA.NO_PROVEN_EDGE]["detail"] and "WATCH" in cases[BA.NO_PROVEN_EDGE]["detail"]
    # a stale scan with BUY rows is still "not current", never eligible
    assert BA.classify(buys=4, scan_current=False)["code"] == BA.SCAN_NOT_CURRENT


@pytest.mark.parametrize("index,verified,expected,code", [
    ({"withheld": [], "withheld_count": 0, "evidence_note": "", "session": "2026-11-19"}, False, "", "SCAN_NOT_CURRENT"),
    ({"withheld": [], "withheld_count": 2, "evidence_note": "", "session": "2026-11-17"}, True, "2026-11-19",
     "SCAN_NOT_CURRENT"),
    ({"withheld": [], "withheld_count": 2, "evidence_note": "the evidence is stale", "session": "2026-11-19"}, True,
     "2026-11-19", "EVIDENCE_UNAVAILABLE"),
    ({"withheld": [{"symbol": "AAA", "edge_r": -0.06}], "withheld_count": 2, "evidence_note": "",
      "session": "2026-11-19"}, True, "2026-11-19", "NO_PROVEN_EDGE"),
    ({"withheld": [], "withheld_count": 0, "evidence_note": "", "session": "2026-11-19"}, True, "2026-11-19",
     "NO_SETUP"),
])
def test_home_reports_why_there_is_no_buy(monkeypatch, index, verified, expected, code):
    import product.home_os as H
    full = {"by_symbol": {}, "scanner_buys": 0, "scanned_at": "", "read_time_withheld": 0, **index}
    monkeypatch.setattr(H, "_scan_evidence_index", lambda: full)
    out = H.build_buy_only([], verified=verified, expected_session=expected)
    assert out["admission_state"] == "NO_ELIGIBLE_BUY" and out["admission_reason"] == code
    assert out["admission_headline"] and out["empty_reason"]


def test_home_never_shows_a_buy_from_a_scan_older_than_the_current_session(monkeypatch):
    import product.home_os as H
    monkeypatch.setattr(H, "_scan_evidence_index", lambda: {
        "by_symbol": {}, "withheld": [], "withheld_count": 0, "scanner_buys": 1, "session": "2026-11-16",
        "scanned_at": "", "evidence_note": "", "read_time_withheld": 0})
    row = {"symbol": "AAA", "entry": 110, "stop": 100, "target": 130}
    out = H.build_buy_only([row], verified=True, expected_session="2026-11-19", login_required=True)
    assert out["rows"] == [] and out["admission_reason"] == "SCAN_NOT_CURRENT"
    assert "2026-11-16" in out["empty_reason"] and "main.py login" in out["admission_action"]


def test_opportunities_and_telegram_say_why_there_is_no_buy(fresh_negative_scan, tmp_path):
    from product.recommendations_workspace import build_recommendations_workspace
    from research.autonomy.telegram_notifications import TelegramNotifier

    payload, _ = fresh_negative_scan
    work = build_recommendations_workspace(scan_payload=payload, long_term_payload={"records": []})
    assert work["buy_admission"]["code"] == "NO_PROVEN_EDGE"
    stale = build_recommendations_workspace(scan_payload={**payload, "records_status": "PRIOR_DAY_SNAPSHOT"},
                                            long_term_payload={"records": []})
    assert stale["buy_admission"]["code"] == "SCAN_NOT_CURRENT"

    class Engine:
        messages: list = []

        def is_configured(self):
            return True

        def send(self, message, reply_markup=None):
            self.messages.append(message)
            return True

    eng = Engine()
    TelegramNotifier(tmp_path, engine_factory=lambda: eng,
                     now_fn=lambda: datetime(2026, 11, 20, 16, 0)).notify_scan(payload, phase="eod")
    eod = [m for m in eng.messages if "EOD scan complete" in m]
    assert eod, eng.messages
    assert "BUY (proven edge): 0" in eod[0] and "No BUY today: no proven edge" in eod[0]
    assert "Qualified setups" not in eod[0]
