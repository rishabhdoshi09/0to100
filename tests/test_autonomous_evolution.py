from __future__ import annotations

from product.autonomous_evolution import (
    _aggregate_splits,
    _publish_history_policies,
    _split_plan,
    summarize_report,
)
from product.evidence_confidence import confidence_from_policies


def _buy(setup: str, r_value: float) -> dict:
    return {
        "symbol": "TEST",
        "decision": "BUY",
        "setup": setup,
        "pit_grade": "PIT_STRONG",
        "pit": {"comparable_to_forward": True},
        "outcome_status": "MATURED",
        "r_multiple": r_value,
    }


def _historical_policy(setup: str = "VCP", score: float = 72.0) -> dict:
    return {
        "policy_id": f"HIST_SETUP::{setup}",
        "dimension": "setup",
        "bucket": setup,
        "sample_size": 30,
        "expectancy_R": 0.45,
        "expectancy_difference_R": 0.45,
        "production_status": "ELIGIBLE",
        "confidence": "REPRODUCED_BACKTEST",
        "affects_selection": True,
        "historical_reproduced_positive": True,
        "historical_confidence_score": score,
        "splits_tested": 3,
        "positive_splits": 3,
        "generation_fingerprint": "gen-A",
    }


def test_reproduced_history_requires_multiple_positive_splits(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_MIN_HIST_N", "8")
    monkeypatch.setenv("QT_EVOLUTION_MIN_POSITIVE_SPLITS", "2")
    monkeypatch.setenv("QT_EVOLUTION_MIN_MEAN_R", "0.15")

    reports = [
        {"decisions": [_buy("VCP", 0.8), _buy("VCP", 0.6), _buy("VCP", 0.4)]},
        {"decisions": [_buy("VCP", 0.7), _buy("VCP", 0.5), _buy("VCP", 0.3)]},
        {"decisions": [_buy("VCP", 0.9), _buy("VCP", 0.2), _buy("VCP", 0.4)]},
    ]
    summaries = [summarize_report(report, split_id=f"s{i}") for i, report in enumerate(reports)]
    setups, _ = _aggregate_splits(summaries)

    row = setups["VCP"]
    assert row["n"] == 9
    assert row["tested_splits"] == 3
    assert row["positive_splits"] == 3
    assert row["reproduced"] is True
    assert 0 < row["historical_confidence_score"] <= 79


def test_one_good_backtest_slice_is_not_reproduction(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_MIN_HIST_N", "3")
    monkeypatch.setenv("QT_EVOLUTION_MIN_POSITIVE_SPLITS", "2")
    monkeypatch.setenv("QT_EVOLUTION_MIN_MEAN_R", "0.10")

    summaries = [
        summarize_report({"decisions": [_buy("BREAKOUT", 1.0)] * 3}, split_id="positive"),
        summarize_report({"decisions": [_buy("BREAKOUT", -0.6)] * 3}, split_id="negative"),
    ]
    setups, _ = _aggregate_splits(summaries)

    assert setups["BREAKOUT"]["positive_splits"] == 1
    assert setups["BREAKOUT"]["reproduced"] is False
    assert setups["BREAKOUT"]["historical_confidence_score"] <= 49


def test_forward_paper_can_strengthen_or_decay_historical_confidence():
    historical = _historical_policy()
    positive_forward = {
        "policy_id": "SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 20,
        "expectancy_R": 0.40,
        "expectancy_difference_R": 0.40,
        "evidence_source": "paper_forward_taken_execution_adjusted",
        "affects_selection": True,
    }
    negative_forward = {
        **positive_forward,
        "sample_size": 10,
        "expectancy_R": -0.50,
        "expectancy_difference_R": -0.50,
    }

    base = confidence_from_policies({"setup_label": "VCP"}, [historical])
    strengthened = confidence_from_policies(
        {"setup_label": "VCP"}, [historical, positive_forward]
    )
    decayed = confidence_from_policies(
        {"setup_label": "VCP"}, [historical, negative_forward]
    )

    assert base["confidence_stage"] == "HISTORICAL_BASE"
    assert strengthened["evidence_confidence_score"] > base["evidence_confidence_score"]
    assert strengthened["confidence_stage"] == "FORWARD_CONFIRMED"
    assert decayed["evidence_confidence_score"] < base["evidence_confidence_score"]
    assert decayed["confidence_stage"] == "FORWARD_DECAYED"
    assert decayed["paper_eligible"] is False
    assert decayed["live_locked"] is True


def test_positive_gross_only_paper_does_not_fake_confidence_boost():
    historical = _historical_policy(score=70.0)
    gross_only = {
        "policy_id": "SETUP::VCP",
        "sample_size": 15,
        "expectancy_R": 0.60,
        "expectancy_difference_R": 0.60,
        "evidence_source": "paper_forward_taken_gross_only",
        "affects_selection": False,
    }

    result = confidence_from_policies({"setup_label": "VCP"}, [historical, gross_only])

    assert result["evidence_confidence_score"] == 70.0
    assert result["forward_n"] == 0
    assert result["forward_observed_n"] == 15
    assert result["confidence_stage"] == "FORWARD_EVIDENCE_UNTRUSTED"
    assert result["forward_trusted_positive"] is False


def test_production_history_gate_blocks_until_bootstrap_completes(monkeypatch):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard
    from product.evidence_policy_engine import _historical_gate

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "RUNNING",
            "analysis_complete": False,
            "paper_ready_setups": 0,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "RUNNING"})
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {"fingerprint": "test-gen", "historical_replay_required": True, "changed": False},
    )

    gate = _historical_gate({"setup_label": "VCP"}, [], enabled=True)

    assert gate["paper_eligible"] is False
    assert gate["bootstrap_complete"] is False
    assert gate["confidence_stage"] == "HISTORICAL_BOOTSTRAP"
    assert gate["live_locked"] is True


def test_production_history_gate_releases_only_reproduced_setup(monkeypatch):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard
    from product.evidence_policy_engine import _historical_gate

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "SUCCEEDED",
            "analysis_complete": True,
            "paper_ready_setups": 1,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "SUCCEEDED"})
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {"fingerprint": "gen-A", "historical_replay_required": False, "changed": False},
    )

    gate = _historical_gate(
        {"setup_label": "VCP"},
        [_historical_policy()],
        enabled=True,
    )
    unknown = _historical_gate(
        {"setup_label": "UNSEEN"},
        [_historical_policy()],
        enabled=True,
    )

    assert gate["paper_eligible"] is True
    assert gate["historical_ready"] is True
    assert gate["bootstrap_complete"] is True
    assert unknown["paper_eligible"] is False
    assert unknown["historical_ready"] is False


def test_split_plan_is_disjoint_and_leaves_forward_outcome_buffer(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_SPLITS", "3")
    monkeypatch.setenv("QT_EVOLUTION_SESSIONS_PER_SPLIT", "4")
    monkeypatch.setenv("QT_EVOLUTION_OUTCOME_BUFFER", "5")
    sessions = [f"2026-01-{day:02d}" for day in range(1, 25)]

    plan = _split_plan(sessions)

    assert len(plan) == 3
    flattened = [day for split in plan for day in split["sessions"]]
    assert len(flattened) == len(set(flattened)) == 12
    assert max(flattened) < sessions[-5]



def test_generation_mismatch_fails_closed():
    historical = _historical_policy()
    result = confidence_from_policies(
        {"setup_label": "VCP"},
        [historical],
        generation_fingerprint="gen-B",
    )
    assert result["historical_generation_match"] is False
    assert result["historical_ready"] is False
    assert result["paper_eligible"] is False
    assert result["confidence_stage"] == "HISTORICAL_GENERATION_MISMATCH"
    assert result["evidence_confidence_score"] == 0.0


def test_store_backed_evaluate_policies_enforces_history_during_pytest(monkeypatch, tmp_path):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard
    from product.evidence_policy_engine import BLOCK, evaluate_policies
    from product.learning_policy_store import upsert_policy

    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION", str(tmp_path / "evolution.json"))
    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={
            "production_status": "ELIGIBLE",
            "confidence": "REPRODUCED_BACKTEST",
            "affects_selection": True,
            "historical_reproduced_positive": True,
            "historical_confidence_score": 72.0,
            "splits_tested": 3,
            "positive_splits": 3,
            "generation_fingerprint": "gen-A",
        },
    )
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {
            "fingerprint": "gen-A",
            "historical_replay_required": False,
            "changed": False,
        },
    )
    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "SUCCEEDED",
            "analysis_complete": True,
            "paper_ready_setups": 1,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "SUCCEEDED"})

    allowed = evaluate_policies({"setup_label": "VCP"})
    blocked = evaluate_policies({"setup_label": "UNSEEN"})

    assert allowed["historical_forward_confidence"]["required"] is True
    assert allowed["historical_forward_confidence"]["historical_ready"] is True
    assert blocked["historical_forward_confidence"]["required"] is True
    assert blocked["final_effect"] == BLOCK
    assert any(
        row.get("policy_id") == "AUTONOMOUS_HISTORY_FIRST_GATE"
        for row in blocked["blocking"]
    )


def test_generation_change_purges_only_historical_policies(monkeypatch, tmp_path):
    import json
    import product.evolution_generation_guard as generation_guard
    from product.learning_policy_store import load_policies, upsert_policy

    policy_path = tmp_path / "policies.json"
    state_path = tmp_path / "evolution.json"
    run_dir = tmp_path / "runs"
    identity_path = tmp_path / "identity.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION", str(state_path))
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION_DIR", str(run_dir))
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION_IDENTITY", str(identity_path))

    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={"historical_reproduced_positive": True, "generation_fingerprint": "gen-A"},
    )
    upsert_policy(
        policy_id="SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.20,
        source="paper_forward_taken_execution_adjusted",
    )
    state_path.write_text(json.dumps({"analysis_complete": True}), encoding="utf-8")
    run_dir.mkdir(parents=True)
    (run_dir / "artifact.json").write_text("{}", encoding="utf-8")
    identity_path.write_text(json.dumps({"fingerprint": "gen-A"}), encoding="utf-8")
    monkeypatch.setattr(
        generation_guard,
        "current_generation",
        lambda: {
            "versions": {},
            "champion": {},
            "warehouse_fingerprint": {},
            "fingerprint": "gen-B",
        },
    )

    verdict = generation_guard.ensure_current_generation()
    policies = load_policies()["policies"]
    ids = {row["policy_id"] for row in policies}

    assert verdict["changed"] is True
    assert "HIST_SETUP::VCP" not in ids
    assert "SETUP::VCP" in ids
    assert not state_path.exists()
    assert not run_dir.exists()


def test_degraded_bootstrap_does_not_publish_partial_history(monkeypatch, tmp_path):
    import product.historical_replay as replay
    import product.evolution_generation_guard as generation_guard
    from product.autonomous_evolution import run_bootstrap
    from product.learning_policy_store import load_policies, upsert_policy

    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("QT_EVOLUTION_SPLITS", "2")
    monkeypatch.setenv("QT_EVOLUTION_SESSIONS_PER_SPLIT", "2")
    monkeypatch.setenv("QT_EVOLUTION_OUTCOME_BUFFER", "2")
    sessions = [f"2026-01-{day:02d}" for day in range(1, 12)]
    monkeypatch.setattr(replay, "official_sessions", lambda: sessions)
    calls = {"n": 0}

    def fake_replay(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("split failure")
        return {
            "run_id": "one",
            "status": "SUCCEEDED",
            "decisions": [_buy("VCP", 0.8), _buy("VCP", 0.6)],
        }

    monkeypatch.setattr(replay, "run_historical_replay", fake_replay)
    monkeypatch.setattr(
        generation_guard,
        "current_generation",
        lambda: {"fingerprint": "gen-A"},
    )
    upsert_policy(
        policy_id="SETUP::KEEP",
        dimension="setup",
        bucket="KEEP",
        sample_size=5,
        expectancy_R=0.1,
        source="paper_forward_taken_execution_adjusted",
    )

    result = run_bootstrap(force=True, path=tmp_path / "state.json")
    ids = {row["policy_id"] for row in load_policies()["policies"]}

    assert result["analysis_complete"] is False
    assert result["status"] == "DEGRADED"
    assert result["published_policies"] == []
    assert not any(policy_id.startswith("HIST_") for policy_id in ids)
    assert "SETUP::KEEP" in ids


def test_historical_over_rejection_links_to_forward_counterfactual(monkeypatch, tmp_path):
    from product.learning_policy_store import load_policies
    from product.paper_learning_loop import ingest_counterfactual
    from product.counterfactual_learning import MISSED_WINNER

    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    _publish_history_policies(
        {},
        {
            "SOFT_FILTER": {
                "n": 10,
                "mean_quality": -0.4,
                "classifications": {"MISSED_WINNER": 4, "FLAT": 6},
                "split_metrics": [],
            }
        },
        generation_fingerprint="gen-A",
    )

    ingest_counterfactual(
        {
            "symbol": "XYZ",
            "reason_code": "SOFT_FILTER",
            "classification": MISSED_WINNER,
            "evidence": {"regime": "RISK_ON"},
        },
        floors={"experimental": 1, "eligible": 2, "active": 3},
    )
    policies = load_policies()["policies"]
    forward = next(row for row in policies if row["policy_id"] == "REJECT::SOFT_FILTER")

    assert forward["historical_hypothesis_linked"] is True
    assert forward["historical_policy_id"] == "HIST_REJECT::SOFT_FILTER"
    assert forward["historical_generation_fingerprint"] == "gen-A"
    assert forward["historical_over_rejection_candidate"] is True
    assert forward["historical_missed_rate"] == 0.4
    assert forward["not_pnl"] is True


def test_blank_outcome_status_is_not_usable_buy():
    from product.autonomous_evolution import _usable_buy

    matured = _buy("VCP", 0.8)
    blank = {**matured, "outcome_status": ""}
    missing = {k: v for k, v in matured.items() if k != "outcome_status"}
    unresolved = {**matured, "outcome_status": "UNRESOLVED"}

    assert _usable_buy(matured) == ("VCP", 0.8)
    assert _usable_buy(blank) is None
    assert _usable_buy(missing) is None
    assert _usable_buy(unresolved) is None


def test_history_gate_api_has_no_pytest_or_env_bypass():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "product" / "evidence_policy_engine.py").read_text(
        encoding="utf-8"
    )
    assert "PYTEST_CURRENT_TEST" not in src
    assert "QT_SKIP" not in src
    assert "enforce_history = bool(store_backed and path is None)" not in src


def test_injected_policies_skip_history_unless_explicitly_enforced(monkeypatch):
    import product.autonomous_evolution as evolution
    from product.evidence_policy_engine import BLOCK, evaluate_policies

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {"status": "NOT_STARTED", "analysis_complete": False, "paper_ready_setups": 0},
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "NOT_STARTED"})

    skipped = evaluate_policies({"setup_label": "VCP"}, policies=[])
    assert skipped["historical_forward_confidence"]["required"] is False
    assert skipped["final_effect"] != BLOCK
    assert not any(
        row.get("policy_id") == "AUTONOMOUS_HISTORY_FIRST_GATE" for row in skipped["blocking"]
    )

    forced = evaluate_policies({"setup_label": "VCP"}, policies=[], enforce_history=True)
    assert forced["historical_forward_confidence"]["required"] is True
    assert forced["final_effect"] == BLOCK


def test_explicit_policy_path_does_not_bypass_history_gate(monkeypatch, tmp_path):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard
    from product.evidence_policy_engine import BLOCK, evaluate_policies
    from product.paper_autopilot import EVIDENCE_POLICY_BLOCK, run_reco_paper_cycle
    from research.auto_research.paper_book import PaperBook

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {"status": "SUCCEEDED", "analysis_complete": True, "paper_ready_setups": 0},
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "SUCCEEDED"})
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {"fingerprint": "gen-A", "historical_replay_required": False, "changed": False},
    )

    path = tmp_path / "policies.json"
    path.write_text('{"policies": []}', encoding="utf-8")
    effect = evaluate_policies({"setup_label": "VCP"}, path=path)
    assert effect["historical_forward_confidence"]["required"] is True
    assert effect["final_effect"] == BLOCK

    now = __import__("datetime").datetime(2026, 9, 1, 10, tzinfo=__import__("datetime").timezone.utc)
    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }
    out = run_reco_paper_cycle(
        book=PaperBook(capital=100_000),
        cards=[card],
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": 1, "cards": [card]}],
        },
        now=now,
        as_of="2026-09-01",
        persist_journal=False,
        policy_path=path,
    )
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def _vcp_card(**over):
    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }
    card.update(over)
    return card


def _ready_generation(monkeypatch):
    import product.autonomous_evolution as evolution
    import product.evolution_generation_guard as generation_guard

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "SUCCEEDED",
            "analysis_complete": True,
            "paper_ready_setups": 1,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "SUCCEEDED"})
    monkeypatch.setattr(
        generation_guard,
        "ensure_current_generation",
        lambda: {"fingerprint": "gen-A", "historical_replay_required": False, "changed": False},
    )


def test_reproduced_setup_becomes_paper_eligible_on_money_path(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from product.learning_policy_store import upsert_policy
    from product.paper_autopilot import run_reco_paper_cycle
    from research.auto_research.paper_book import PaperBook

    _ready_generation(monkeypatch)
    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={
            "production_status": "ELIGIBLE",
            "confidence": "REPRODUCED_BACKTEST",
            "affects_selection": True,
            "historical_reproduced_positive": True,
            "historical_confidence_score": 72.0,
            "splits_tested": 3,
            "positive_splits": 3,
            "generation_fingerprint": "gen-A",
        },
    )
    now = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    card = _vcp_card()
    book = PaperBook(capital=100_000)
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": 1, "cards": [card]}],
        },
        now=now,
        as_of="2026-09-01",
        persist_journal=False,
    )
    assert out["taken"]
    assert out["taken"][0]["symbol"] == "TCS"
    assert next(iter(book.open.values())).symbol == "TCS"


def test_generation_mismatch_still_blocks_paper_money_path(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from product.learning_policy_store import upsert_policy
    from product.paper_autopilot import EVIDENCE_POLICY_BLOCK, run_reco_paper_cycle
    from research.auto_research.paper_book import PaperBook

    _ready_generation(monkeypatch)
    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={
            "production_status": "ELIGIBLE",
            "historical_reproduced_positive": True,
            "historical_confidence_score": 72.0,
            "generation_fingerprint": "gen-OLD",
        },
    )
    now = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    card = _vcp_card()
    out = run_reco_paper_cycle(
        book=PaperBook(capital=100_000),
        cards=[card],
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": 1, "cards": [card]}],
        },
        now=now,
        as_of="2026-09-01",
        persist_journal=False,
    )
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == EVIDENCE_POLICY_BLOCK


def test_history_gate_does_not_hide_invalid_stop_when_setup_is_reproduced(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from product.learning_policy_store import upsert_policy
    from product.paper_autopilot import INVALID_STOP, run_reco_paper_cycle
    from research.auto_research.paper_book import PaperBook

    _ready_generation(monkeypatch)
    policy_path = tmp_path / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))
    upsert_policy(
        policy_id="HIST_SETUP::VCP",
        dimension="setup",
        bucket="VCP",
        sample_size=30,
        expectancy_R=0.45,
        source="backtest_reproduced",
        extra={
            "production_status": "ELIGIBLE",
            "historical_reproduced_positive": True,
            "historical_confidence_score": 72.0,
            "generation_fingerprint": "gen-A",
            "affects_selection": True,
        },
    )
    now = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    card = _vcp_card(stop=101.0)
    out = run_reco_paper_cycle(
        book=PaperBook(capital=100_000),
        cards=[card],
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": 1, "cards": [card]}],
        },
        now=now,
        as_of="2026-09-01",
        persist_journal=False,
    )
    assert not out["taken"]
    assert out["rejections"][0]["reason_code"] == INVALID_STOP


def test_bootstrap_lock_prevents_concurrent_overwrite(monkeypatch, tmp_path):
    import product.autonomous_evolution as evolution

    state = tmp_path / "evolution.json"
    monkeypatch.setenv("QT_AUTONOMOUS_EVOLUTION", str(state))
    held = evolution._try_acquire_bootstrap_lock(state)
    assert held is not None
    try:
        blocked = evolution.run_bootstrap(force=True)
        assert blocked.get("bootstrap_lock") == "held_elsewhere"
        assert blocked.get("analysis_complete") is False
    finally:
        evolution._release_bootstrap_lock(held)
