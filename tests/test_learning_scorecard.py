"""The learning scorecard counts sessions, not rows, and never claims on thin evidence."""

from __future__ import annotations

import itertools
import random

from product import learning_scorecard as LS

_SYMBOL_IDS = itertools.count()

_MATURE = lambda _day: LS.MATURED   # these tests measure statistics, not maturation


def _row(session, outcome_pct, *, entry=100.0, stop=95.0, worked=None, score=60.0, archetype="VCP", regime="RISK_ON"):
    if worked is None:
        worked = 1 if outcome_pct > 0 else 0
    return {
        "symbol": f"S{next(_SYMBOL_IDS)}", "logged_at": f"{session}T10:00:00",       # distinct decisions
        "entry_price": entry, "stop_price": stop, "outcome_pct": outcome_pct, "worked": worked,
        "quality_score": score, "archetype": archetype, "regime": regime,
    }


def _sessions(n, start_month=1):
    """The first ``n`` real NSE sessions of 2026 from ``start_month`` (holidays skipped).

    Real sessions, because the verdicts now place each session on the exchange
    calendar: outcome windows overlap, so independence is counted in blocks
    of NSE sessions, never in rows or calendar dates.
    """
    from datetime import date, timedelta
    import research.intelligence.data.nse_calendar as CAL

    holidays, out = CAL.load_holidays(), []
    day = date(2026, start_month, 1) - timedelta(days=1)
    while len(out) < n:
        day = CAL.next_session(day, holidays)
        out.append(day.isoformat())
    return out


def test_thousands_of_rows_from_few_sessions_make_no_claim():
    rows = [_row(s, 3.0) for s in _sessions(5) for _ in range(400)]  # 2,000 rows, 5 sessions
    ev = LS.tracked_signal_evidence(rows, maturity=_MATURE)
    assert ev["overall_net"]["n"] == 2000
    assert ev["overall_net"]["sessions"] == 5
    assert ev["overall_net"]["verdict"] == LS.INSUFFICIENT


def test_consistent_positive_edge_over_enough_sessions_is_credible():
    random.seed(1)
    rows = [_row(s, random.gauss(4.0, 1.0)) for s in _sessions(210) for _ in range(3)]   # 21 blocks
    ov = LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]
    assert ov["verdict"] == LS.CREDIBLE_POSITIVE and ov["independent_blocks"] >= 20


def test_consistent_negative_edge_over_enough_sessions_is_credible_negative():
    random.seed(2)
    rows = [_row(s, random.gauss(-4.0, 1.0)) for s in _sessions(210) for _ in range(3)]
    assert LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]["verdict"] == LS.CREDIBLE_NEGATIVE


def test_thirty_consecutive_sessions_are_three_windows_not_thirty_observations():
    """The old floor (20 sessions) claimed on what is ~2-3 independent 10-session windows."""
    random.seed(9)
    rows = [_row(s, random.gauss(4.0, 1.0)) for s in _sessions(30) for _ in range(10)]
    ov = LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]
    assert ov["independent_blocks"] <= 4 and ov["verdict"] == LS.INSUFFICIENT


def test_noise_around_zero_is_not_distinguishable():
    random.seed(3)
    rows = [_row(s, random.gauss(0.32, 8.0)) for s in _sessions(210) for _ in range(5)]  # ~0 net of costs
    assert LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]["verdict"] in {LS.NOT_DISTINGUISHABLE}


def test_unknown_or_uncovered_calendar_makes_no_claim(monkeypatch):
    random.seed(10)
    rows = [_row(s, random.gauss(4.0, 1.0)) for s in _sessions(210) for _ in range(3)]
    import research.intelligence.data.nse_calendar as CAL
    monkeypatch.setattr(CAL, "load_holidays", lambda: {"2026-11-10"})       # a stray amendment, not a calendar
    ov = LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]
    assert ov["verdict"] == LS.INSUFFICIENT and ov["reason"].startswith("EXCHANGE_CALENDAR")


def test_costs_are_charged_in_r():
    rows = [_row(s, 0.0, worked=0) for s in _sessions(25)]  # flat gross, 5% stop
    ov = LS.tracked_signal_evidence(rows, maturity=_MATURE)["overall_net"]
    assert ov["mean_gross_R_naive"] == 0.0
    assert ov["mean_net_R_naive"] < 0  # 0.32% round trip / 5% risk = -0.064R


def test_tight_stop_cash_like_signals_are_excluded_from_r_but_counted():
    rows = [_row(s, 3.0) for s in _sessions(25)]
    rows += [dict(_row("2026-07-21", -0.5, entry=1000.0, stop=999.0), symbol="LIQUIDBEES")]
    ev = LS.tracked_signal_evidence(rows, maturity=_MATURE)
    assert ev["overall_net"]["n"] == 25
    assert ev["excluded_from_R_tight_stop"]["n"] == 1
    assert ev["excluded_from_R_tight_stop"]["symbols"] == {"LIQUIDBEES": 1}
    assert ev["overall_net_pct_all_filled"]["n"] == 26


def test_never_filled_and_open_rows_are_not_outcomes():
    rows = [_row("2026-01-01", 2.0)]
    rows.append({**_row("2026-01-02", 0.0), "worked": -1, "outcome_pct": None})
    rows.append({**_row("2026-01-03", 0.0), "worked": None, "outcome_pct": None})
    ev = LS.tracked_signal_evidence(rows, maturity=_MATURE)
    assert (ev["settled_filled"], ev["never_filled"], ev["open_or_unresolved"]) == (1, 1, 1)
    assert ev["fill_rate"] == 0.5


def test_a_score_that_does_not_rank_outcomes_is_called_out():
    random.seed(4)
    trades = [{"quality_score": random.uniform(30, 100), "won": random.random() < 0.2} for _ in range(600)]
    cal = LS.score_calibration(trades)
    assert cal["verdict"] == "NO_USEFUL_DISCRIMINATION"
    assert cal["brier_skill"] < 0  # reading score/100 as P(win) is worse than the base rate


def test_a_ranking_score_that_is_not_a_probability_is_labelled_so():
    random.seed(5)
    trades = []
    for _ in range(800):
        won = random.random() < 0.2
        trades.append({"quality_score": (85 if won else 70) + random.uniform(-5, 5), "won": won})
    cal = LS.score_calibration(trades)
    assert cal["auc"] > 0.9
    assert cal["verdict"] == "RANKS_BUT_NOT_A_PROBABILITY"


def test_calibration_needs_enough_predictions():
    assert LS.score_calibration([{"quality_score": 80, "won": True}] * 10)["verdict"] == LS.INSUFFICIENT


def test_trend_needs_two_independent_halves():
    rows = [_row(s, 2.0) for s in _sessions(30)]
    trades = [t for t in (LS._settled_trade(r) for r in rows) if t]
    assert LS.trend(trades)["verdict"] == LS.INSUFFICIENT
    random.seed(11)                                   # 60 sessions: 3 blocks a half, no claim
    rows = [_row(s, random.gauss(-3 if i < 30 else 4, 0.5)) for i, s in enumerate(_sessions(60))]
    assert LS.trend([t for t in (LS._settled_trade(r) for r in rows) if t])["verdict"] == LS.INSUFFICIENT
    random.seed(6)
    rows = [_row(s, random.gauss(-3 if i < 110 else 4, 0.5)) for i, s in enumerate(_sessions(220))]
    trades = [t for t in (LS._settled_trade(r) for r in rows) if t]
    assert LS.trend(trades)["verdict"] == "IMPROVING"


def test_learning_influence_reports_no_change_when_nothing_is_eligible():
    out = LS.learning_influence(
        [{"policy_id": "REJECT::X", "production_status": "REJECTED", "affects_selection": True}],
        [{"symbol": "A", "market_timestamp": "2026-01-01", "group": "REJECTED", "reason_code": "X",
          "policy_effect": "NEUTRAL"}],
    )
    assert out["decisions_changed_by_learning"] == 0
    assert out["answer"].startswith("NO")


def test_learning_influence_counts_each_decision_once():
    # An eligible decision that reached the ranking (NO_TRADE: not top of the cycle),
    # re-frozen by 9 cycles: one decision a score effect could change (QT-C2-09).
    rows = [{"symbol": "A", "market_timestamp": "2026-01-01", "group": "REJECTED", "reason_code": "NO_TRADE",
             "policy_effect": "PENALIZE", "cycle_id": f"c{i}"} for i in range(9)]
    out = LS.learning_influence(
        [{"policy_id": "SETUP::VCP", "production_status": "ELIGIBLE", "affects_selection": True,
          "sample_size": 40, "sessions_observed": 25}],
        rows,
    )
    assert out["decisions_changed_by_learning"] == 1
    assert out["answer"] == "YES"


def test_scorecard_on_empty_system_is_honest_and_live_stays_locked():
    card = LS.build_scorecard(signal_rows=[], forward_rows=[], policies=[], outcome_pipeline={},
                              include_challengers=False)
    g = card["graduation"]
    assert g["current_stage"] == 0
    assert g["verdicts"]["RESEARCH READY"] == "NOT READY"
    assert g["verdicts"]["FORWARD PAPER READY"] == "NOT READY"
    assert g["verdicts"]["OPERATIONALLY READY"] == "NOT READY"
    assert g["verdicts"]["SUPERVISED LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert g["verdicts"]["AUTONOMOUS LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert g["live_trading"] == "LOCKED"
    assert card["live_interlock"]["locked"] is True
    assert card["answers"]["5_net_of_costs"] == LS.INSUFFICIENT
    assert any("host soak" in u for u in g["unmet"]["OPERATIONALLY READY"])


def test_even_perfect_evidence_cannot_make_live_eligible():
    random.seed(7)
    rows = [_row(s, random.gauss(5.0, 0.5), score=90 if random.random() < 0.5 else 50)
            for s in _sessions(60) for _ in range(5)]
    card = LS.build_scorecard(signal_rows=rows, maturity=_MATURE, forward_rows=[], policies=[], outcome_pipeline={},
                              include_challengers=False,
                              institutional={"domains": [{"key": k, "status": "READY"} for k in
                                                         ("data", "operations", "reconciliation",
                                                          "execution", "protection", "risk")]})
    g = card["graduation"]
    assert g["verdicts"]["SUPERVISED LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert g["verdicts"]["AUTONOMOUS LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert g["verdicts"]["FORWARD PAPER READY"] == "NOT READY"  # no settled paper trades


def test_snapshot_history_has_one_line_per_session():
    card = LS.build_scorecard(signal_rows=[], forward_rows=[], policies=[], outcome_pipeline={},
                              include_challengers=False)
    first = LS.persist_snapshot(card, "2026-11-18")
    again = LS.persist_snapshot(card, "2026-11-18")
    nxt = LS.persist_snapshot(card, "2026-11-19")
    assert first["history_appended"] and not again["history_appended"] and nxt["history_appended"]
    assert [h["session"] for h in LS.load_history()][-2:] == ["2026-11-18", "2026-11-19"]
    assert LS.load_snapshot()["graduation"]["live_trading"] == "LOCKED"


def test_api_serves_snapshot_and_reevaluates_graduation(monkeypatch):
    import terminal_product_api as api

    card = LS.build_scorecard(signal_rows=[], forward_rows=[], policies=[], outcome_pipeline={},
                              include_challengers=False)
    LS.persist_snapshot(card, "2026-11-18")
    monkeypatch.setattr(api, "institutional_readiness", lambda: {"domains": [{"key": "data", "status": "PARTIAL"}]})
    api._SCORECARD_INSTITUTIONAL.update({"at": 0.0, "payload": None})
    out = api.learning_scorecard()
    assert out["source"] == "nightly_snapshot"
    assert out["graduation"]["verdicts"]["AUTONOMOUS LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert any("data" in u for u in out["graduation"]["unmet"]["RESEARCH READY"])
    assert out["history"]


def test_home_summary_before_first_snapshot_is_honest(monkeypatch):
    monkeypatch.setattr(LS, "load_snapshot", lambda: None)
    out = LS.home_summary()
    assert out["available"] is False
    assert "not computed yet" in out["headline"]
    assert out["live_trading"] == "LOCKED"


def test_home_summary_never_reads_unproven_evidence_as_an_edge():
    random.seed(8)
    rows = [_row(s, random.gauss(-0.5, 6.0)) for s in _sessions(16) for _ in range(20)]
    card = LS.build_scorecard(signal_rows=rows, maturity=_MATURE, forward_rows=[], policies=[], outcome_pipeline={},
                              include_challengers=False)
    out = LS.home_summary(card)
    assert out["available"] is True
    assert out["headline"].startswith("NO VALIDATED EDGE")
    assert out["net_edge"]["verdict"] == LS.INSUFFICIENT
    assert out["verdicts"]["AUTONOMOUS LIVE ELIGIBLE"] == "NOT ELIGIBLE"
    assert out["blockers"]["FORWARD PAPER READY"]


def _calendar(days):
    from datetime import date
    return [date.fromisoformat(d) for d in days]


def _holidays_2026():
    import json
    from pathlib import Path
    return set(json.loads((Path(__file__).resolve().parents[1] / "data" / "nse_holidays.json").read_text()))


_HOLIDAYS_2026 = _holidays_2026()

SESSIONS = ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-11",  # 10 Nov holiday
            "2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18"]


def test_maturity_counts_exchange_sessions_and_skips_holidays():
    from datetime import date
    check = LS.maturity_checker(verified_sessions=_calendar(SESSIONS), holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 11, 18), horizon=10)
    assert check("2026-11-02") == LS.MATURED                   # 10 sessions 3..17 Nov, holiday skipped
    assert check("2026-11-09") == LS.IMMATURE                  # only 6 sessions completed since
    assert check("2026-11-18") == LS.IMMATURE


def test_a_missing_official_bar_is_not_yet_verified_never_matured():
    from datetime import date
    verified = _calendar([d for d in SESSIONS if d != "2026-11-12"])
    check = LS.maturity_checker(verified_sessions=verified, holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 11, 18), horizon=10)
    assert check("2026-11-02") == LS.NOT_YET_VERIFIED


def test_unavailable_evidence_is_not_yet_verified(monkeypatch):
    from datetime import date
    assert LS.maturity_checker(verified_sessions=[], holidays=set(), last_completed=date(2026, 11, 18),
                               horizon=10)("2026-07-01") == LS.NOT_YET_VERIFIED
    import research.intelligence.data.nse_calendar as CAL
    monkeypatch.setattr(CAL, "load_holidays", lambda: (_ for _ in ()).throw(OSError("no calendar")))
    assert LS.maturity_checker()("2026-07-01") == LS.NOT_YET_VERIFIED


def test_an_unusable_holiday_calendar_fails_closed_never_weekdays():
    """Every weekday has a 'verified' bar; only the calendar is wrong. Never MATURED."""
    from datetime import date
    import pandas as pd

    weekdays = [d.date() for d in pd.bdate_range("2026-01-01", "2027-02-26")]
    for holidays, reason in ((set(), "EXCHANGE_CALENDAR_UNAVAILABLE"),
                             ({"2026-11-10"}, "EXCHANGE_CALENDAR_NOT_COVERED:2026"),
                             ({"2026-11-10", "garbage", "2026-13-40"}, "EXCHANGE_CALENDAR_NOT_COVERED:2026")):
        check = LS.maturity_checker(verified_sessions=weekdays, holidays=holidays,
                                    last_completed=date(2026, 11, 18), horizon=10)
        assert check("2026-11-02") == LS.NOT_YET_VERIFIED, holidays
        assert dict(check.reasons) == {reason: 1}
    # a covered year matures; a window reaching into an uncovered year does not
    check = LS.maturity_checker(verified_sessions=weekdays, holidays=_HOLIDAYS_2026,
                                last_completed=date(2027, 2, 26), horizon=10)
    assert check("2026-12-10") == LS.MATURED
    assert check("2026-12-22") == LS.NOT_YET_VERIFIED
    assert dict(check.reasons) == {"EXCHANGE_CALENDAR_NOT_COVERED:2027": 1}
    # with no calendar at all nothing is classified, not even a window still in the future
    check = LS.maturity_checker(verified_sessions=weekdays, holidays=set(),
                                last_completed=date(2026, 11, 18), horizon=10)
    assert check("2026-11-17") == LS.NOT_YET_VERIFIED


def test_only_matured_signals_enter_the_statistics_and_exclusions_are_reported():
    rows = [_row("2026-07-01", 3.0), _row("2026-07-02", -4.0), _row("2026-11-12", 5.0), _row("2026-11-13", 1.0)]
    status = {"2026-07-01": LS.MATURED, "2026-07-02": LS.MATURED, "2026-11-12": LS.IMMATURE,
              "2026-11-13": LS.NOT_YET_VERIFIED}
    ev = LS.tracked_signal_evidence(rows, maturity=lambda d: status[d])
    assert ev["maturity"] == {LS.MATURED: 2, LS.IMMATURE: 1, LS.NOT_YET_VERIFIED: 1}
    assert ev["rows_logged"] == 4 and ev["rows_matured"] == 2
    assert ev["settled_filled"] == 2


def test_a_decision_logged_twice_on_one_session_counts_once():
    random.seed(12)
    rows = [_row(s, random.gauss(4.0, 1.0)) for s in _sessions(210) for _ in range(3)]
    once = LS.tracked_signal_evidence(rows, maturity=_MATURE)
    copied = LS.tracked_signal_evidence(rows + [dict(r, logged_at=r["logged_at"][:10] + "T14:00:00") for r in rows],
                                        maturity=_MATURE)
    assert copied["duplicate_decision_rows_dropped"] == len(rows) and copied["rows_logged"] == 2 * len(rows)
    for key in ("n", "sessions", "mean_of_session_means_R", "ci95_R", "verdict"):
        assert copied["overall_net"][key] == once["overall_net"][key], key


def test_an_official_bar_on_a_calendar_closed_day_means_the_calendar_is_stale():
    """Offline freshness: the bhavcopy archive contradicts an unrecorded special session."""
    from datetime import date
    import pandas as pd

    weekdays = [d.date() for d in pd.bdate_range("2026-11-01", "2026-12-31") if d.strftime("%F") not in _HOLIDAYS_2026]
    check = LS.maturity_checker(verified_sessions=weekdays, holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 12, 31), horizon=10)
    assert check("2026-11-25") == LS.MATURED
    saturday = weekdays + [date(2026, 12, 5)]                       # an exceptional Saturday session traded
    check = LS.maturity_checker(verified_sessions=saturday, holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 12, 31), horizon=10)
    assert check("2026-11-25") == LS.NOT_YET_VERIFIED
    assert dict(check.reasons) == {"CALENDAR_CONTRADICTED_BY_OFFICIAL_BARS": 1}
    assert check("2026-12-08") == LS.MATURED                        # windows after it are unaffected
    muhurat = weekdays + [date(2026, 11, 8)]                         # documented in the manifest
    check = LS.maturity_checker(verified_sessions=muhurat, holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 12, 31), horizon=10)
    assert check("2026-11-05") == LS.MATURED


def test_an_emergency_closure_leaves_a_window_unverified():
    from datetime import date
    import pandas as pd

    closed = [d.date() for d in pd.bdate_range("2026-11-01", "2026-12-31")
              if d.strftime("%F") not in _HOLIDAYS_2026 and d.strftime("%F") != "2026-12-02"]
    check = LS.maturity_checker(verified_sessions=closed, holidays=_HOLIDAYS_2026,
                                last_completed=date(2026, 12, 31), horizon=10)
    assert check("2026-11-25") == LS.NOT_YET_VERIFIED and dict(check.reasons) == {"VERIFIED_BAR_MISSING": 1}


def test_the_scorecard_states_what_calendar_freshness_cannot_prove():
    ev = LS.tracked_signal_evidence([], maturity=_MATURE)
    prov = ev["calendar_provenance"]
    assert prov["years"]["2026"]["status"] == "VERIFIED_COMPLETE"
    assert prov["live_freshness"].startswith("BLOCKED")

