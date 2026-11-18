"""The learning scorecard counts sessions, not rows, and never claims on thin evidence."""

from __future__ import annotations

import random

from product import learning_scorecard as LS


def _row(session, outcome_pct, *, entry=100.0, stop=95.0, worked=None, score=60.0, archetype="VCP", regime="RISK_ON"):
    if worked is None:
        worked = 1 if outcome_pct > 0 else 0
    return {
        "symbol": f"S{random.randint(0, 10**6)}", "logged_at": f"{session}T10:00:00",
        "entry_price": entry, "stop_price": stop, "outcome_pct": outcome_pct, "worked": worked,
        "quality_score": score, "archetype": archetype, "regime": regime,
    }


def _sessions(n, start_month=1):
    return [f"2026-{start_month + i // 28:02d}-{i % 28 + 1:02d}" for i in range(n)]


def test_thousands_of_rows_from_few_sessions_make_no_claim():
    rows = [_row(s, 3.0) for s in _sessions(5) for _ in range(400)]  # 2,000 rows, 5 sessions
    ev = LS.tracked_signal_evidence(rows)
    assert ev["overall_net"]["n"] == 2000
    assert ev["overall_net"]["sessions"] == 5
    assert ev["overall_net"]["verdict"] == LS.INSUFFICIENT


def test_consistent_positive_edge_over_enough_sessions_is_credible():
    random.seed(1)
    rows = [_row(s, random.gauss(4.0, 1.0)) for s in _sessions(30) for _ in range(10)]
    assert LS.tracked_signal_evidence(rows)["overall_net"]["verdict"] == LS.CREDIBLE_POSITIVE


def test_consistent_negative_edge_over_enough_sessions_is_credible_negative():
    random.seed(2)
    rows = [_row(s, random.gauss(-4.0, 1.0)) for s in _sessions(30) for _ in range(10)]
    assert LS.tracked_signal_evidence(rows)["overall_net"]["verdict"] == LS.CREDIBLE_NEGATIVE


def test_noise_around_zero_is_not_distinguishable():
    random.seed(3)
    rows = [_row(s, random.gauss(0.32, 8.0)) for s in _sessions(30) for _ in range(5)]  # ~0 net of costs
    assert LS.tracked_signal_evidence(rows)["overall_net"]["verdict"] in {LS.NOT_DISTINGUISHABLE}


def test_costs_are_charged_in_r():
    rows = [_row(s, 0.0, worked=0) for s in _sessions(25)]  # flat gross, 5% stop
    ov = LS.tracked_signal_evidence(rows)["overall_net"]
    assert ov["mean_gross_R_naive"] == 0.0
    assert ov["mean_net_R_naive"] < 0  # 0.32% round trip / 5% risk = -0.064R


def test_tight_stop_cash_like_signals_are_excluded_from_r_but_counted():
    rows = [_row(s, 3.0) for s in _sessions(25)]
    rows += [dict(_row("2026-07-21", -0.5, entry=1000.0, stop=999.0), symbol="LIQUIDBEES")]
    ev = LS.tracked_signal_evidence(rows)
    assert ev["overall_net"]["n"] == 25
    assert ev["excluded_from_R_tight_stop"]["n"] == 1
    assert ev["excluded_from_R_tight_stop"]["symbols"] == {"LIQUIDBEES": 1}
    assert ev["overall_net_pct_all_filled"]["n"] == 26


def test_never_filled_and_open_rows_are_not_outcomes():
    rows = [_row("2026-01-01", 2.0)]
    rows.append({**_row("2026-01-02", 0.0), "worked": -1, "outcome_pct": None})
    rows.append({**_row("2026-01-03", 0.0), "worked": None, "outcome_pct": None})
    ev = LS.tracked_signal_evidence(rows)
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
    random.seed(6)
    rows = [_row(s, random.gauss(-3 if i < 25 else 4, 0.5)) for i, s in enumerate(_sessions(50))]
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
    rows = [{"symbol": "A", "market_timestamp": "2026-01-01", "group": "REJECTED", "reason_code": "X",
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
    card = LS.build_scorecard(signal_rows=rows, forward_rows=[], policies=[], outcome_pipeline={},
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
