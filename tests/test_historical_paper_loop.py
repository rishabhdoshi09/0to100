from __future__ import annotations

import json

import pytest

from product import historical_paper_loop as HPL
from research.autonomy import evidence_acquisition as EA


@pytest.mark.parametrize("reversal", [False, True])
def test_policy_comparison_isolated_learning_and_crash_resume(tmp_path, monkeypatch, reversal):
    from product.evidence_policy_engine import evaluate_policies
    from product.learning_policy_store import load_policies, HISTORICAL_COMPARISON_SOURCE
    from tests.test_paper_autopilot_money_path import _eligible_card

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    production = tmp_path / "untouched-forward-policies.json"
    production.write_text('{"policies":[]}')
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(production))
    external_reads = []
    def external_read(*args, **kwargs):
        external_reads.append(True)
        raise AssertionError("Historical selection read a present-day authority")

    monkeypatch.setattr("risk.portfolio_risk.account_exposure_gate", external_read)
    monkeypatch.setattr("product.challenger_learning.paper_selection_adjustment", external_read)
    days = [f"2025-04-{n:02}" for n in (1, 2, 3, 4, 7, 8, 9, 10)]
    good, bad = "TCS", "INFY"
    rows = []
    def decision(day, symbol):
        setup = "positive-control" if symbol == good else "negative-control"
        card = _eligible_card(symbol, setup_label=setup, sector=symbol)
        return {"decision": "BUY", "symbol": symbol, "as_of": day, "setup": setup,
                "regime": "RISK_ON", "target": card["target"], "selection_card": card}

    rows.extend(decision(d, symbol) for d, symbol in zip(days, (good, good, bad, bad, bad)))
    rows.extend([decision(days[5], good), decision(days[5], bad)])
    win, loss = (100, 116, 98, 115), (100, 101, 83, 88)
    bars = {d: {good: win, bad: loss} for d in days}
    if reversal:
        bars[days[6]][bad] = win
    floors = dict(experimental=2, eligible=2, active=2, sessions_eligible=2, sessions_active=2)
    options = dict(official_sessions=days, bars=bars, synthetic=True, floors=floors)
    original = json.dumps(rows, sort_keys=True)
    checkpoint = HPL._HistoricalPolicyComparison.save

    def interrupted(branch, payload):
        if branch.adaptive and payload["session"] == days[5]:
            raise OSError("interrupted after learning and decision freeze")
        return checkpoint(branch, payload)

    with monkeypatch.context() as patch:
        patch.setattr(HPL._HistoricalPolicyComparison, "save", interrupted)
        with pytest.raises(OSError, match="after learning"):
            HPL.compare_historical_policies(rows, directory=tmp_path / "resumed", **options)
    repaired = HPL.compare_historical_policies(rows, directory=tmp_path / "resumed", **options)
    assert HPL.compare_historical_policies(rows, directory=tmp_path / "resumed", **options) == repaired
    clean = HPL.compare_historical_policies(rows, directory=tmp_path / "clean", **options)
    assert repaired["comparison"] == clean["comparison"]
    assert repaired["branches"]["ADAPTIVE"]["trades"] == clean["branches"]["ADAPTIVE"]["trades"]
    changed = repaired["comparison"]["decisions_changed"]
    assert {r["symbol"]: r["effect"] for r in changed if r["day"] == days[5]} == {good: "SUPPORT", bad: "BLOCK"}
    assert (repaired["comparison"]["adaptive_minus_frozen_pct"] < 0) == reversal
    assert all(not r["affects_selection"] and r["production_status"] != "ACTIVE"
               for r in load_policies(tmp_path / "resumed/ADAPTIVE/policies.json")["policies"])
    assert production.read_text() == '{"policies":[]}'
    assert json.dumps(rows, sort_keys=True) == original
    assert external_reads == []
    simulated = {"policy_id": "comparison", "dimension": "setup", "bucket": "positive-control",
                 "production_status": "ACTIVE", "evidence_source": HISTORICAL_COMPARISON_SOURCE,
                 "affects_selection": True, "expectancy_difference_R": 2, "sample_size": 40}
    assert evaluate_policies({"setup_label": "positive-control"}, policies=[simulated])["final_effect"] == "NEUTRAL"
    assert repaired["comparison"]["ADAPTIVE"]["paired_positive_rejection_rate"] == float(reversal)


@pytest.mark.parametrize(("next_bar", "status", "reason"), [
    ((97, 99, 96, 98), "NO_FILL", None),
    ((102, 116, 94, 110), "FILLED", "LOSS"),
    ((117, 119, 116, 118), "FILLED", "SCRATCH"),
    ((114.99, 114.995, 114.9, 114.95), "FILLED", "SCRATCH"),
    (None, "UNAVAILABLE", None),
])
def test_comparison_executes_only_next_session(tmp_path, monkeypatch, next_bar, status, reason):
    from tests.test_paper_autopilot_money_path import _eligible_card

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    days = ["2025-04-01", "2025-04-02", "2025-04-03"]
    row = {"decision": "BUY", "symbol": "TCS", "as_of": days[0], "regime": "RISK_ON",
           "selection_card": _eligible_card()}
    # A signal-day high and a later trigger cannot fill the intervening order.
    bars = {day: {"TCS": (100, 116, 97, 115)} for day in days}
    bars[days[1]] = {"TCS": next_bar} if next_bar else {}
    output = HPL.compare_historical_policies(
        [row], official_sessions=days, bars=bars, directory=tmp_path / "comparison", synthetic=True,
    )
    for branch in output["branches"].values():
        assert [o["status"] for o in branch["orders"]] == [status]
        assert branch["orders"][0]["execution_date"] == days[1]
        assert [t["exit_reason"] for t in branch["trades"]] == ([reason] if reason else [])
        assert branch["book"]["equity_curve"][1:] == [row["equity"] for row in branch["daily"]]
        if reason:
            trade, = branch["trades"]
            assert trade["entry_price"] == pytest.approx(next_bar[0] * 1.0003)
            assert trade["net_pnl"] < 0


def test_bound_thesis_does_not_read_active_selection(monkeypatch):
    from product import trading_thesis

    reads = []
    def forbidden():
        reads.append(True)
        raise AssertionError("present-day state accessed")

    monkeypatch.setattr(trading_thesis, "_selection_policy_identity", forbidden)
    monkeypatch.setattr("product.challenger_learning.dashboard", forbidden)
    thesis = trading_thesis.manifest(include_active_selection=False)
    assert thesis["active_selection_learner"] == {}
    assert thesis["selection_policy_set"] == {}
    assert reads == []


def test_simulated_policy_does_not_change_production_thesis_identity(monkeypatch):
    from product import learning_policy_store, trading_thesis

    monkeypatch.setattr(learning_policy_store, "load_policies", lambda: {"policies": []})
    baseline = trading_thesis._selection_policy_identity()
    comparison = {
        "policy_id": "simulated", "dimension": "setup", "bucket": "example",
        "evidence_source": learning_policy_store.HISTORICAL_COMPARISON_SOURCE,
        "production_status": "ACTIVE", "confidence": "MEASURED",
        "affects_selection": True, "expectancy_difference_R": 2.0,
    }
    monkeypatch.setattr(learning_policy_store, "load_policies", lambda: {"policies": [comparison]})
    assert trading_thesis._selection_policy_identity() == baseline


def test_comparison_refuses_reserve_future_inputs_and_unsafe_path(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    target = tmp_path / "comparison"
    with pytest.raises(ValueError, match="reserved"):
        HPL.compare_historical_policies([], official_sessions=["2025-11-01"], bars={}, directory=target, synthetic=True)
    with pytest.raises(ValueError, match="future evidence"):
        HPL.compare_historical_policies([{"as_of": "2025-04-01", "symbol": "TCS", "pit": {"max_bar_date": "2025-04-02"}}],
                                        official_sessions=["2025-04-01"], bars={}, directory=target, synthetic=True)
    with pytest.raises(ValueError, match="dedicated isolated"):
        HPL.compare_historical_policies([], official_sessions=["2025-04-01"], bars={}, directory=tmp_path, synthetic=True)
    target.mkdir()
    (target / "ADAPTIVE").mkdir()
    with pytest.raises(ValueError, match="not empty"):
        HPL.compare_historical_policies([], official_sessions=["2025-04-01"], bars={}, directory=target, synthetic=True)


def test_comparison_rejects_tampered_frozen_selection_card(tmp_path, monkeypatch):
    from product.decision_freeze import freeze
    from product.historical_replay import selection_card_digest
    from research.intelligence.data import nse_calendar

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("QT_DECISION_FREEZE", str(tmp_path / "logs/decisions.sqlite"))
    monkeypatch.setattr(nse_calendar, "load_manifest", lambda _path: {})
    monkeypatch.setattr(nse_calendar, "calendar_gap", lambda *_args: None)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "nse_holidays.json").write_text("[]")
    day = "2025-04-01"
    row = {
        "as_of": day, "symbol": "TCS", "provenance": "HISTORICAL_REPLAY",
        "pit": {"max_bar_date": day}, "selection_card": {"entry": 100},
    }
    row["data_snapshot_id"] = selection_card_digest(row["selection_card"])
    row["freeze_id"] = freeze(row)["freeze_id"]
    row["selection_card"]["entry"] = 90
    options = {"bars": {}, "directory": tmp_path / "comparison"}
    with pytest.raises(ValueError, match="consecutive official session"):
        HPL.compare_historical_policies([row], official_sessions=[day, "2025-04-03"], **options)
    with pytest.raises(ValueError, match="immutable PIT freeze"):
        HPL.compare_historical_policies([row], official_sessions=[day], **options)
    row["data_snapshot_id"] = selection_card_digest(row["selection_card"])
    with pytest.raises(ValueError, match="immutable PIT freeze"):
        HPL.compare_historical_policies([row], official_sessions=[day], **options)


def test_virtual_trade_same_bar_stop_and_target_is_conservative_stop_first():
    row = {
        "run_id": "r1",
        "decision": "BUY",
        "symbol": "ABC",
        "as_of": "2026-01-05",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "setup": "VCP_BREAKOUT",
    }

    def later(_symbol, _as_of, *, horizon):
        assert horizon == HPL.DEFAULT_HORIZON_SESSIONS
        return [
            {"date": "2026-01-06", "high": 112.0, "low": 94.0, "close": 108.0},
        ]

    trade = HPL.simulate_virtual_trade(row, later_bars_fn=later)
    assert trade is not None
    assert trade["exit_price"] == 95.0
    assert trade["exit_reason"] == "STOP_AND_TARGET_SAME_BAR_STOP_FIRST_CONSERVATIVE"
    assert trade["realized_R"] == -1.0
    assert trade["same_bar_path_ambiguous"] is True
    assert trade["evidence_class"] == "HISTORICAL_REPLAY"
    assert trade["not_real_pnl"] is True
    assert trade["not_promotion_evidence"] is True


def test_historical_cursor_advances_only_after_learning_then_research(tmp_path):
    state = tmp_path / "state.json"
    HPL._save_state(
        {
            "phase": HPL.PHASE_AWAITING_LEARNING,
            "current_batch_id": "batch-1",
            "current_sessions": ["2026-01-05", "2026-01-06"],
            "last_completed_session": "",
        },
        state,
    )

    premature = HPL.mark_research_complete("batch-1", state_path=state)
    assert premature["phase"] == HPL.PHASE_AWAITING_LEARNING
    assert premature["last_completed_session"] == ""

    learned = HPL.mark_learning_complete("batch-1", state_path=state)
    assert learned["phase"] == HPL.PHASE_AWAITING_RESEARCH
    assert learned["last_completed_session"] == ""
    assert learned["last_learning_batch_id"] == "batch-1"
    assert learned["last_learning_completed_at"]

    completed = HPL.mark_research_complete("batch-1", state_path=state)
    assert completed["phase"] == HPL.PHASE_IDLE
    assert completed["last_completed_session"] == "2026-01-06"
    assert completed["current_batch_id"] == ""
    assert completed["current_sessions"] == []


def test_next_batch_uses_durable_cursor_and_does_not_repeat(tmp_path):
    state = tmp_path / "state.json"
    sessions = [f"2026-01-{day:02d}" for day in range(1, 13)]

    first = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert first["available"] is True
    assert first["sessions"] == ["2026-01-03", "2026-01-04"]

    HPL._save_state(
        {
            "phase": HPL.PHASE_IDLE,
            "last_completed_session": "2026-01-04",
            "thesis_hash": first["thesis_hash"],
        },
        state,
    )
    second = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert second["available"] is True
    assert second["sessions"] == ["2026-01-05", "2026-01-06"]
    assert second["batch_id"] != first["batch_id"]


def test_historical_virtual_ledger_is_idempotent_across_restart(tmp_path):
    ledger = tmp_path / "historical_paper.jsonl"
    row = {
        "trade_id": "hist-paper:abc",
        "symbol": "ABC",
        "evidence_class": "HISTORICAL_REPLAY",
        "not_real_pnl": True,
    }
    assert HPL._append_unique(ledger, [row]) == 1
    assert HPL._append_unique(ledger, [row]) == 0
    persisted = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(persisted) == 1
    assert persisted[0]["trade_id"] == "hist-paper:abc"



def test_historical_setup_confidence_policy_never_becomes_active(tmp_path):
    policy_path = tmp_path / "policies.json"
    trades = [
        {
            "trade_id": f"hist-paper-{i}",
            "symbol": f"S{i}",
            "setup": "VCP_BREAKOUT",
            "realized_R": 1.0,
            "evidence_class": "HISTORICAL_REPLAY",
            "thesis_hash": "thesis-a",
            "not_real_pnl": True,
        }
        for i in range(30)
    ]
    policies = HPL.update_historical_setup_policies(trades, path=policy_path)
    assert len(policies) == 1
    policy = policies[0]
    assert policy["policy_id"] == "HIST_SETUP::thesis-a::VCP_BREAKOUT"
    assert policy["sample_size"] == 30
    assert policy["historical_confidence_score"] == 95.0
    assert policy["historical_reproduced_positive"] is True
    assert policy["evidence_source"] == "backtest_historical_replay"
    assert policy["affects_selection"] is False
    assert policy["production_status"] != "ACTIVE"
    assert policy["not_promotion_evidence"] is True

    again = HPL.update_historical_setup_policies(trades, path=policy_path)
    assert again[0]["version"] == policy["version"]
    assert again[0]["generation_fingerprint"] == policy["generation_fingerprint"]



def test_historical_sequence_enforces_real_paper_book_overlap_caps(monkeypatch):
    from types import SimpleNamespace
    import product.paper_autopilot as PA

    days = [f"2026-01-{day:02d}" for day in range(5, 28)]
    decisions = []
    for index in range(6):
        day = days[0] if index < 3 else days[1]
        symbol = f"S{index}"
        decisions.append({
            "decision": "BUY",
            "symbol": symbol,
            "as_of": day,
            "entry": 100.0,
            "stop": 95.0,
            "target": 200.0,
            "sector": f"SEC{index}",
            "setup": "VCP",
            "regime": "RISK_ON",
            "thesis_hash": "thesis-1",
            "selection_score": 100.0 - index,
            "canonical_decision_id": f"d-{index}",
            "selection_card": {
                "symbol": symbol,
                "reco_tier": "high_conviction",
                "entry_state": "enter_now",
                "entry": 100.0,
                "stop": 95.0,
                "target": 200.0,
                "sector": f"SEC{index}",
                "setup_label": "VCP",
                "dd_status": "PASS",
                "volume_ratio": 1.2,
            },
        })

    class FakeDecision:
        def __init__(self, card):
            self.symbol = card["symbol"]
            self.decision = PA.ENTER_NOW
            self.reason_code = "ELIGIBLE"
            self.detail = "passed"
            self.card = dict(card)
            self.selection_score = 100.0 - int(self.symbol[1:])
            self.context = {}
            self.portfolio = {}
            self.policy_effect = "NEUTRAL"

        def as_dict(self):
            return {
                "symbol": self.symbol,
                "decision": self.decision,
                "reason_code": self.reason_code,
                "entry": self.card["entry"],
                "stop": self.card["stop"],
                "target": self.card["target"],
            }

    monkeypatch.setattr(
        PA,
        "evaluate_selection_candidate",
        lambda card, **_kwargs: FakeDecision(card),
    )

    def later(symbol, as_of, *, horizon):
        start = days.index(as_of)
        out = []
        for day in days[start + 1:start + 1 + horizon]:
            out.append({
                "date": day,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            })
        return out

    result = HPL.simulate_paper_book_sequence(
        decisions,
        official_sessions=days,
        later_bars_fn=later,
        horizon=20,
        max_new_per_session=3,
    )

    # Day 1 opens three. Day 2 can add only two because the real PaperBook caps
    # simultaneous positions at five. Independent per-trade backtesting would
    # incorrectly count all six.
    assert len(result["trades"]) == 5
    assert any(
        str(row.get("reason_code") or "") == "BOOK_REFUSED"
        or "MAX" in str(row.get("reason_code") or "")
        for row in result["rejections"]
    )
    assert result["open_unresolved"] == 0
    assert result["execution_model"] == "PaperBook"
    assert all(row["cost_model"] == "india_cash_costs" for row in result["trades"])
    assert all(row["slippage_bps"] == 3.0 for row in result["trades"])
    assert all(row["not_real_pnl"] is True for row in result["trades"])



def test_historical_batch_carries_only_eligible_evidence_request(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    request_path = tmp_path / "request.json"
    sessions = [f"2026-01-{day:02d}" for day in range(1, 13)]

    historical = EA.build_request(
        session_date="2026-09-21",
        strategy_id="QT_RECO_ENSEMBLE",
        gap_kind="poor_calibration",
        diagnosis="Need historical validation evidence.",
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=20,
        target_samples=30,
    )
    EA.save_request(historical, path=request_path)

    def plan_information_gain(request, sessions, **_kwargs):
        picked = list(sessions)[:2]
        return {
            "sessions": picked,
            "acquisitions": [
                {"session_date": day, "evidence_origin": "HISTORICAL_REPLAY"}
                for day in picked
            ],
            "selection_policy": "INFORMATION_GAIN",
            "reason": "selected",
            "stop": False,
        }

    monkeypatch.setattr(
        "research.autonomy.replay_acquisition_runtime.plan_runtime_acquisitions",
        plan_information_gain,
    )

    batch = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        evidence_request_path=request_path,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert batch["evidence_request_id"] == historical.request_id
    assert batch["evidence_request"]["sample_deficit"] == 10
    # The PIT curriculum is only a prefilter. The final evidence-directed batch
    # is selected and journaled by the information-gain planner.
    assert batch["selection_policy"] == "INFORMATION_GAIN"
    assert batch["selection_details"]["outcome_blind_selection"] is True
    assert batch["selection_details"]["acquisitions"]

    forward_only = EA.build_request(
        session_date="2026-09-21",
        strategy_id="MOM",
        gap_kind="insufficient_sample",
        diagnosis="Need forward paper outcomes.",
        evidence_origin="FORWARD_PAPER",
        current_samples=5,
        target_samples=30,
    )
    EA.save_request(forward_only, path=request_path)

    batch2 = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        evidence_request_path=request_path,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert batch2["evidence_request_id"] == ""
    assert batch2["evidence_request"] == {}
    assert batch2["selection_policy"] == "DURABLE_CURSOR"



def test_active_curriculum_noncontiguous_sessions_are_durable_and_never_repeat(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    request_path = tmp_path / "request.json"
    sessions = [f"2026-01-{day:02d}" for day in range(1, 16)]
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-active"},
    )

    req = EA.build_request(
        session_date="2026-09-21",
        strategy_id="QT_RECO_ENSEMBLE",
        gap_kind="poor_calibration",
        diagnosis="Balance historical regime evidence.",
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=10,
        target_samples=30,
        thesis_hash="thesis-active",
    )
    EA.save_request(req, path=request_path)

    def choose(eligible, *, processed_sessions=(), batch_size=8, classifier=None):
        remaining = [d for d in eligible if d not in set(processed_sessions)]
        picks = []
        if remaining:
            picks.append(remaining[-1])
        if len(remaining) > 2:
            picks.append(remaining[1])
        return {
            "sessions": picks[:batch_size],
            "selection_policy": "ACTIVE_REGIME_COVERAGE",
            "selection_objective": "test active selection",
            "outcome_blind_selection": True,
            "coverage_before": {},
            "coverage_after": {},
            "session_states": {},
        }

    monkeypatch.setattr(
        "research.autonomy.historical_curriculum.select_regime_balanced_sessions",
        choose,
    )

    def plan_information_gain(request, sessions, **_kwargs):
        picked = list(sessions)[:2]
        return {
            "sessions": picked,
            "acquisitions": [
                {"session_date": day, "evidence_origin": "HISTORICAL_REPLAY"}
                for day in picked
            ],
            "selection_policy": "INFORMATION_GAIN",
            "reason": "selected",
            "stop": False,
        }

    monkeypatch.setattr(
        "research.autonomy.replay_acquisition_runtime.plan_runtime_acquisitions",
        plan_information_gain,
    )

    first = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        evidence_request_path=request_path,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert first["selection_policy"] == "INFORMATION_GAIN"
    assert first["selection_details"]["curriculum_selection_policy"] == "ACTIVE_REGIME_COVERAGE"
    assert len(first["sessions"]) == 2
    assert first["sessions"] == sorted(first["sessions"])

    HPL._save_state(
        {
            "phase": HPL.PHASE_AWAITING_RESEARCH,
            "current_batch_id": first["batch_id"],
            "current_sessions": first["sessions"],
            "thesis_hash": "thesis-active",
            "evidence_request_id": req.request_id,
            "evidence_request": req.as_dict(),
            "selection_details": first["selection_details"],
        },
        state,
    )
    completed = HPL.mark_research_complete(first["batch_id"], state_path=state)
    assert set(first["sessions"]).issubset(set(completed["processed_sessions"]))

    second = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        evidence_request_path=request_path,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert second["available"] is True
    assert set(first["sessions"]).isdisjoint(set(second["sessions"]))
    assert second["processed_sessions_before"] == len(first["sessions"])


def test_throughput_metrics_measure_useful_examples_not_raw_activity():
    out = HPL._throughput_metrics(
        replay_elapsed_s=10.0,
        paper_elapsed_s=2.0,
        batch_elapsed_s=20.0,
        decisions=100,
        useful_examples=8,
    )

    assert out["decisions_per_replay_second"] == 10.0
    assert out["useful_examples_per_batch_second"] == 0.4
    assert out["useful_training_examples"] == 8
    assert "not evidence of profitability" in out["metric_note"]


def test_throughput_metrics_do_not_divide_by_zero():
    out = HPL._throughput_metrics(
        replay_elapsed_s=0.0,
        paper_elapsed_s=0.0,
        batch_elapsed_s=0.0,
        decisions=10,
        useful_examples=2,
    )
    assert out["decisions_per_replay_second"] is None
    assert out["useful_examples_per_batch_second"] is None
