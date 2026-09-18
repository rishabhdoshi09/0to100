from __future__ import annotations

import json
from types import SimpleNamespace

from product import historical_replay as HR
from product import trading_thesis as TT


def test_selection_affecting_policy_change_invalidates_thesis_hash(tmp_path, monkeypatch):
    policy_path = tmp_path / "learning_policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policy_path))

    policy_path.write_text(
        json.dumps({"schema_version": 1, "policies": [], "live_locked": True}),
        encoding="utf-8",
    )
    empty = TT.manifest()

    # Observation-only state cannot affect selection and therefore should not
    # force the operator to approve the same thesis again.
    policy_path.write_text(
        json.dumps({
            "schema_version": 1,
            "policies": [{
                "policy_id": "SETUP::VCP",
                "version": 1,
                "dimension": "setup",
                "bucket": "VCP",
                "sample_size": 7,
                "expectancy_difference_R": 0.4,
                "shrunk_expectancy_R": 0.2,
                "confidence": "INSUFFICIENT_EVIDENCE",
                "production_status": "OBSERVING",
                "evidence_source": "paper_forward",
                "affects_selection": True,
            }],
            "live_locked": True,
        }),
        encoding="utf-8",
    )
    observing = TT.manifest()
    assert observing["thesis_hash"] == empty["thesis_hash"]

    # Once the policy has a real selection effect it becomes part of the
    # versioned thesis identity. Startup permission is separate and remains
    # one-time; this hash is for evidence/batch comparability.
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["policies"][0]["version"] = 2
    payload["policies"][0]["sample_size"] = 35
    payload["policies"][0]["confidence"] = "MEASURED"
    payload["policies"][0]["production_status"] = "ACTIVE"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")

    active = TT.manifest()
    assert active["thesis_hash"] != observing["thesis_hash"]
    assert active["selection_policy_set"]["count"] == 1
    assert active["selection_policy_set"]["effective_policies"] == [{
        "policy_id": "SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "effective_status": "ACTIVE",
        "effect": "SUPPORT",
    }]

    # More observations inside the same effective SUPPORT region do not create
    # a new thesis generation or force historical replay back to session one.
    stable_hash = active["thesis_hash"]
    payload["policies"][0]["version"] = 3
    payload["policies"][0]["sample_size"] = 48
    payload["policies"][0]["expectancy_difference_R"] = 0.31
    payload["policies"][0]["shrunk_expectancy_R"] = 0.27
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    same_effect = TT.manifest()
    assert same_effect["thesis_hash"] == stable_hash

    # Crossing an actual selection-behavior threshold is a real thesis change.
    payload["policies"][0]["version"] = 4
    payload["policies"][0]["expectancy_difference_R"] = -0.45
    payload["policies"][0]["shrunk_expectancy_R"] = -0.40
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    blocked = TT.manifest()
    assert blocked["thesis_hash"] != stable_hash
    assert blocked["selection_policy_set"]["effective_policies"][0]["effect"] == "BLOCK"


def test_historical_replay_defaults_to_present_paper_decider(monkeypatch):
    card = {
        "symbol": "INFY",
        "reco_tier": "high_conviction",
        "setup_label": "VCP_BREAKOUT",
        "entry_state": "enter_now",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "dd_status": "PASS",
        "volume_ratio": 1.2,
    }
    workspace = {
        "scan_scanned_at": "2026-09-01T15:30:00+05:30",
        "categories": [{"id": "momentum_breakouts", "cards": [card]}],
    }
    scan = {
        "as_of_session": "2026-09-01",
        "regime": "RISK_ON",
    }

    monkeypatch.setattr(
        "product.recommendations_workspace.build_recommendations_workspace",
        lambda **_kwargs: dict(workspace),
    )
    monkeypatch.setattr("product.pit_query.attach_pit_to_card", lambda row, **_kwargs: dict(row))
    monkeypatch.setattr(
        "product.pit_coverage.overall_replay_grade",
        lambda *_a, **_k: {
            "grade": "MARKET_ONLY",
            "reason": "test",
            "coverage": {},
            "comparable_to_forward": False,
            "production_comparable": False,
        },
    )
    monkeypatch.setattr(
        "product.pit_coverage.explain_downgrade",
        lambda *_a, **_k: {"unavailable": [], "unverified": [], "available": []},
    )
    monkeypatch.setattr(
        "product.pit_versions.current_versions",
        lambda: SimpleNamespace(as_dict=lambda: {"test": "v1"}),
    )
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-1", "objective_id": "test"},
    )

    calls = []

    class Decision:
        selection_score = 91.5

        def as_dict(self):
            return {
                "symbol": "INFY",
                "decision": "NO_TRADE",
                "reason_code": "NO_ELIGIBLE_TRADE",
            }

    def production_selection(candidate, **kwargs):
        calls.append((dict(candidate), dict(kwargs)))
        assert kwargs["enforce_history"] is False
        assert kwargs["entries_allowed"] is True
        assert kwargs["paper_enabled"] is True
        assert kwargs["regime"] == "RISK_ON"
        return Decision()

    monkeypatch.setattr(HR, "evaluate_selection_candidate", production_selection)

    rows = HR.decide_session("2026-09-01", scan)

    assert len(calls) == 1
    assert rows
    assert rows[0]["selection_score"] == 91.5
    assert rows[0]["thesis_hash"] == "thesis-1"
    assert rows[0]["pit"]["history_bootstrap_gate_applied"] is False

