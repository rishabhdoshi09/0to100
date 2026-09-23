from __future__ import annotations

from product import research_director_status as RD


def _safety():
    return {
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "live_lock_status": "LOCKED",
        "live_lock_reason": "paper-only deployment",
        "live_lock_source": "product.live_execution_interlock",
    }


def _patch_common(monkeypatch):
    monkeypatch.setattr(RD, "live_safety_projection", _safety)
    monkeypatch.setattr(
        RD,
        "_autonomy",
        lambda: {
            "state": "RESEARCHING",
            "current_activity": "HISTORICAL_REPLAY",
            "activity_truth": {
                "activity": "HISTORICAL_REPLAY",
                "busy": True,
                "resource_governor": {},
            },
            "resource_governor": {
                "decision": "ALLOW_HISTORICAL_REPLAY",
                "reason": "no higher-priority due/running work",
            },
        },
    )
    monkeypatch.setattr(
        RD,
        "_calibration",
        lambda: {
            "snapshot_id": "cal-1",
            "immutable": True,
            "identities": {"data_identity": "nse:2026-09-21"},
        },
    )
    monkeypatch.setattr(
        RD,
        "_decision_calibration",
        lambda: {
            "settled_observations": 40,
            "explicit_probability_observations": 24,
            "probability_coverage": 0.6,
            "overall": {
                "status": "MEASURED",
                "probability_status": "MEASURED",
                "probability_sample_size": 24,
                "brier": 0.21,
                "calibration_actionable": False,
            },
            "buckets": {},
            "probability_drift": {
                "status": "STABLE_WITHIN_UNCERTAINTY",
                "degradation_detected": False,
                "recent_brier": 0.20,
                "baseline_brier": 0.21,
            },
            "affects_production": False,
            "live_locked": True,
        },
    )
    monkeypatch.setattr(
        RD,
        "_signal_registry",
        lambda: {
            "registry_version": "sig-v1",
            "summary": {
                "scanner_catalog": 17,
                "forward_calibrated": 16,
                "effective_calibrated": 17,
                "count_difference_explained": True,
            },
        },
    )
    monkeypatch.setattr(
        RD,
        "_learned_challenger",
        lambda: {
            "current": {
                "model_version": "clf-1",
                "status": "SHADOW_CANDIDATE",
                "trained_n": 100,
                "real_forward_n": 30,
                "promotion_dossier": {"decision": "KEEP_SHADOW"},
            }
        },
    )
    monkeypatch.setattr(
        RD,
        "_rule_challengers",
        lambda: {
            "challengers": [
                {"challenger_id": "r1", "status": "TESTING"},
                {"challenger_id": "dead", "status": "REJECTED"},
            ]
        },
    )
    monkeypatch.setattr(
        RD,
        "_research_overview",
        lambda: {
            "knowledge_growth": {"net_knowledge_gain": 2},
            "knowledge_growth_1d": {
                "validated_in_window": 0,
                "retired_in_window": 0,
            },
            "edge_health": {},
            "gate_scorecard": [],
            "data_health": {},
            "research_debt": {},
        },
    )
    monkeypatch.setattr(
        RD,
        "_forward_soak",
        lambda: {
            "FORWARD_SOAK_STATUS": "COLLECTING",
            "real_forward_observations": 12,
            "settled_trades": 3,
            "rejected_candidates_settled": 8,
            "missed_winners": 2,
            "avoided_losers": 1,
        },
    )


def test_director_reports_exact_evidence_gap_and_batch_rationale(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(RD, "_today_ist", lambda: "2026-09-22")
    monkeypatch.setattr(
        RD,
        "_request",
        lambda: {
            "request_id": "er-1",
            "status": "OPEN",
            "diagnosis": "VCP has too little high-volatility evidence.",
            "gap_kind": "REGIME_SCARCITY",
            "evidence_origin": "RESEARCH_VALIDATION",
            "allowed_lanes": ["HISTORICAL_REPLAY"],
            "current_samples": 20,
            "target_samples": 40,
            "sample_deficit": 20,
            "missing_metrics": ["deflated_sharpe"],
            "acquisition_tasks": ["Acquire high-volatility VCP sessions"],
            "stop_conditions": ["target_samples_reached"],
        },
    )
    monkeypatch.setattr(
        RD,
        "_progress",
        lambda _request_id: {
            "request_id": "er-1",
            "status": "OPEN",
            "sample_count": 28,
            "target_samples": 40,
            "sample_deficit": 12,
            "last_batch_yield": 8,
            "stagnant_batches": 0,
            "resolved_metrics": [],
            "unresolved_metrics": ["deflated_sharpe"],
            "next_action": "ACQUIRE_MORE_EVIDENCE",
            "history": [
                {
                    "recorded_at": "2026-09-22T01:00:00+05:30",
                    "samples_acquired": 8,
                }
            ],
        },
    )
    monkeypatch.setattr(
        RD,
        "_historical_state",
        lambda: {
            "phase": "RUNNING",
            "current_batch_id": "hist-1",
            "current_sessions": ["2026-04-01", "2026-05-15"],
            "selection_details": {
                "selection_policy": "ACTIVE_REGIME_COVERAGE",
                "selection_objective": "reduce point-in-time regime coverage imbalance",
                "outcome_blind_selection": True,
                "coverage_before": {"BULL_TREND": 20, "BEAR": 3},
                "coverage_after": {"BULL_TREND": 20, "BEAR": 5},
            },
        },
    )

    out = RD.build_research_director_status()

    assert out["current_question"] == "VCP has too little high-volatility evidence."
    assert out["evidence_request"]["sample_deficit"] == 12
    assert out["next_evidence_batch"]["selection_policy"] == "ACTIVE_REGIME_COVERAGE"
    assert out["next_evidence_batch"]["outcome_blind_selection"] is True
    assert out["next_action"] == "ACQUIRE_MORE_EVIDENCE"
    assert out["learning_delta"]["measurable_change_today"] is True
    assert out["learning_delta"]["last_batch_evidence_added"] == 8
    assert out["signals"]["scanner_catalog"] == 17
    assert out["signals"]["forward_calibrated"] == 16
    assert out["calibration"]["snapshot_id"] == "cal-1"
    assert out["decision_calibration"]["settled_observations"] == 40
    assert out["decision_calibration"]["explicit_probability_observations"] == 24
    assert out["decision_calibration"]["probability_coverage"] == 0.6
    assert out["decision_calibration"]["overall"]["probability_status"] == "MEASURED"
    assert out["decision_calibration"]["probability_drift"]["status"] == "STABLE_WITHIN_UNCERTAINTY"
    assert out["decision_calibration"]["affects_production"] is False
    assert out["resource_governor"]["decision"] == "ALLOW_HISTORICAL_REPLAY"
    assert out["live_locked"] is True
    assert out["live_execution_authorized"] is False


def test_activity_alone_never_claims_learning(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(RD, "_today_ist", lambda: "2026-09-22")
    monkeypatch.setattr(RD, "_request", lambda: {})
    monkeypatch.setattr(RD, "_progress", lambda _request_id: {})
    monkeypatch.setattr(
        RD,
        "_historical_state",
        lambda: {
            "phase": "RUNNING",
            "current_batch_id": "hist-activity-only",
            "current_sessions": ["2026-01-01"],
            "selection_details": {},
        },
    )

    out = RD.build_research_director_status()

    assert out["current_activity"] == "HISTORICAL_REPLAY"
    assert out["learning_delta"]["measurable_change_today"] is False
    assert out["current_question"] == "No unresolved evidence request is currently persisted."
    assert out["evidence_request"]["request_id"] == ""


def test_missing_snapshots_are_blockers_not_synthetic_values(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(RD, "_request", lambda: {})
    monkeypatch.setattr(RD, "_progress", lambda _request_id: {})
    monkeypatch.setattr(RD, "_historical_state", lambda: {})
    monkeypatch.setattr(RD, "_calibration", lambda: {})
    monkeypatch.setattr(RD, "_signal_registry", lambda: {})

    out = RD.build_research_director_status()

    assert out["calibration"]["snapshot_id"] == ""
    assert out["signals"]["registry_version"] == ""
    assert "No immutable calibration snapshot is currently persisted." in out["blockers"]
    assert "No persisted Signal Registry snapshot is available." in out["blockers"]


def test_runtime_state_mismatch_is_visible_in_director(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        RD,
        "_autonomy",
        lambda: {
            "state": "RESEARCHING",
            "current_activity": "IDLE",
            "activity_truth": {"activity": "IDLE", "busy": False},
            "resource_governor": {},
        },
    )
    monkeypatch.setattr(RD, "_request", lambda: {})
    monkeypatch.setattr(RD, "_progress", lambda _request_id: {})
    monkeypatch.setattr(RD, "_historical_state", lambda: {})

    out = RD.build_research_director_status()

    assert out["state_truth"]["mismatch"] is True
    assert any("RESEARCHING" in blocker for blocker in out["blockers"])
