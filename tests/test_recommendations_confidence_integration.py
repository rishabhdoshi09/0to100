from __future__ import annotations

import product.confidence_breakdown as CB
import product.recommendations_workspace as RW


def _card():
    return {
        "symbol": "TCS",
        "company": "TCS",
        "category_id": "wealth_builders",
        "category_label": "Wealth Builders",
        "action_badge": "Watch",
        "risk_tier": "Medium",
        "setup_label": "VCP",
        "sector": "Technology",
        "score": 80,
        "reco_tier": "watch",
        "reco_tier_label": "Watch",
        "entry_state": "ready",
        "market_support": "Unmeasured",
        "lifecycle": "active",
        "methods": [],
        "families": [],
    }


def _buckets():
    return {
        "wealth_builders": [_card()],
        "super_trends": [],
        "momentum_breakouts": [],
        "recovery_setups": [],
    }


def _context():
    return {
        "policies": [],
        "generation": {"fingerprint": "gen-1"},
        "thesis": {"thesis_hash": "thesis-1"},
        "calibration": {
            "snapshot_id": "cal-1",
            "immutable": True,
            "identities": {"thesis_hash": "thesis-1"},
        },
        "signal_registry": {},
    }


def _patch_workspace(monkeypatch):
    monkeypatch.setattr(RW, "_bucket_rows", lambda *a, **k: _buckets())
    monkeypatch.setattr(RW, "_tracker_lifecycle", lambda: ([], []))
    monkeypatch.setattr(
        RW,
        "ensemble_summary",
        lambda cards: {
            "high_conviction_count": 0,
            "good_setup_count": 0,
            "watch_count": len(cards),
            "avoid_count": 0,
            "empty_high_conviction": True,
            "empty_line": "No high-conviction names.",
        },
    )


def test_current_workspace_attaches_confidence_breakdown(monkeypatch):
    _patch_workspace(monkeypatch)
    monkeypatch.setattr(CB, "load_context", _context)

    payload = RW.build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": "2026-09-22"},
        long_term_payload={"records": []},
        point_in_time=False,
    )

    card = payload["categories"][0]["cards"][0]
    breakdown = card["confidence_breakdown"]
    assert breakdown["available"] is True
    assert breakdown["identities"]["generation_fingerprint"] == "gen-1"
    assert breakdown["identities"]["thesis_hash"] == "thesis-1"
    assert breakdown["identities"]["calibration_snapshot_id"] == "cal-1"
    assert breakdown["final"]["is_win_probability"] is False


def test_point_in_time_workspace_refuses_current_confidence_context(monkeypatch):
    _patch_workspace(monkeypatch)

    def forbidden():
        raise AssertionError("PIT workspace must not load today's confidence context")

    monkeypatch.setattr(CB, "load_context", forbidden)

    payload = RW.build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": "2026-04-01"},
        long_term_payload={"records": []},
        point_in_time=True,
        as_of="2026-04-01",
    )

    card = payload["categories"][0]["cards"][0]
    breakdown = card["confidence_breakdown"]
    assert breakdown["available"] is False
    assert breakdown["reason"] == "POINT_IN_TIME_CURRENT_CONFIDENCE_SKIPPED"
    assert breakdown["final"]["evidence_strength_score"] is None
