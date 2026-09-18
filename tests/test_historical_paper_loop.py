from __future__ import annotations

import json

from product import historical_paper_loop as HPL


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
