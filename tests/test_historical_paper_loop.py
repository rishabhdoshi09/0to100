from __future__ import annotations

import json

from product import historical_paper_loop as HPL
from research.autonomy import evidence_acquisition as EA


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



def test_historical_batch_carries_only_eligible_evidence_request(tmp_path):
    state = tmp_path / "state.json"
    request_path = tmp_path / "request.json"
    sessions = [f"2026-01-{day:02d}" for day in range(1, 13)]

    historical = EA.build_request(
        session_date="2026-09-21",
        strategy_id="MOM",
        gap_kind="poor_calibration",
        diagnosis="Need historical validation evidence.",
        evidence_origin="RESEARCH_VALIDATION",
        current_samples=20,
        target_samples=30,
    )
    EA.save_request(historical, path=request_path)

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
    # Test environment has no official index cache, so the active curriculum
    # truthfully falls back instead of inventing historical regime labels.
    assert batch["selection_policy"] == "DURABLE_CURSOR"
    assert batch["selection_details"]["outcome_blind_selection"] is True

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
        strategy_id="MOM",
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

    first = HPL.peek_next_batch(
        sessions_fn=lambda: sessions,
        state_path=state,
        evidence_request_path=request_path,
        batch_size=2,
        warmup_sessions=2,
        horizon_sessions=2,
        universe_limit=10,
    )
    assert first["selection_policy"] == "ACTIVE_REGIME_COVERAGE"
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
