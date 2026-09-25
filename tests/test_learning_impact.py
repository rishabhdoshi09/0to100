from __future__ import annotations


def test_learning_impact_distinguishes_active_selection_from_shadow(monkeypatch):
    import product.autonomous_learning as auto
    import product.challenger_learning as challenger
    import product.decision_discovery_store as discovery
    import product.learning_policy_store as policies
    from product.learning_impact import build_learning_impact

    monkeypatch.setattr(
        challenger,
        "dashboard",
        lambda: {
            "current": {
                "status": "PAPER_ACTIVE",
                "model_version": "clf_test",
                "trained_n": 120,
                "real_forward_n": 35,
                "historical_n": 70,
                "counterfactual_n": 15,
                "affects_selection": True,
                "forward_validation": {"n": 35, "improvement": 0.02},
            }
        },
    )
    monkeypatch.setattr(
        policies,
        "load_policies",
        lambda: {
            "policies": [
                {
                    "policy_id": "SETUP::VCP",
                    "production_status": "ACTIVE",
                    "affects_selection": True,
                    "final_effect": "SUPPORT",
                }
            ]
        },
    )
    monkeypatch.setattr(
        discovery,
        "load_current",
        lambda: {
            "decisions": [
                {
                    "symbol": "TEST",
                    "state": "BUY",
                    "base_score": 70,
                    "ranking_score": 72,
                    "learning_adjustment": 2,
                    "evidence_adjustment": 0,
                    "why": "forward-proven learner changed rank",
                    "learning": {"model_version": "clf_test"},
                }
            ]
        },
    )
    monkeypatch.setattr(
        auto,
        "dashboard",
        lambda: {
            "counts": {
                "historical_decisions_simulated": 500,
                "historical_virtual_paper_trades": 90,
                "forward_paper_decisions": 40,
                "correct_rejects": 12,
                "avoided_losers": 5,
                "missed_winners": 3,
            }
        },
    )

    impact = build_learning_impact()

    assert impact["status"] == "ACTIVE_IN_PAPER_SELECTION"
    assert impact["selection_is_currently_changed"] is True
    assert impact["current_decisions_influenced"] == 1
    assert impact["challenger"]["affects_selection"] is True
    assert impact["policies"]["production_effective"] == 1
    assert impact["contract"]["historical_only_can_promote"] is False
    assert impact["live_locked"] is True


def test_learning_impact_does_not_claim_historical_replay_improves_live_ranking(monkeypatch):
    import product.autonomous_learning as auto
    import product.challenger_learning as challenger
    import product.decision_discovery_store as discovery
    import product.learning_policy_store as policies
    from product.learning_impact import build_learning_impact

    monkeypatch.setattr(
        challenger,
        "dashboard",
        lambda: {"current": {"status": "OBSERVING", "trained_n": 100, "historical_n": 100, "real_forward_n": 0}},
    )
    monkeypatch.setattr(policies, "load_policies", lambda: {"policies": []})
    monkeypatch.setattr(discovery, "load_current", lambda: {"decisions": []})
    monkeypatch.setattr(
        auto,
        "dashboard",
        lambda: {"counts": {"historical_decisions_simulated": 1000}},
    )

    impact = build_learning_impact()

    assert impact["status"] == "LEARNING_BUT_NOT_PROMOTED"
    assert impact["selection_is_currently_changed"] is False
    assert impact["challenger"]["affects_selection"] is False
    assert impact["contract"]["historical_only_can_promote"] is False
