"""Network-free tests for Phase A.5 scoped certification (no strategy rerun)."""
from __future__ import annotations

from pathlib import Path

from research.phase_a5 import scoped_certification as sc


def test_frozen_panel_exact_size_and_unique():
    assert len(sc.FROZEN_PANEL) == 29
    assert len(set(sc.FROZEN_PANEL)) == 29
    assert "RELIANCE" in sc.FROZEN_PANEL
    assert "BAJAJFINSV" in sc.FROZEN_PANEL


def test_dependency_matrix_covers_five_experiments_without_invented_index():
    rows = sc.protocol_dependency_matrix()
    exps = {r["experiment"] for r in rows}
    assert exps == {
        "EXP-A5-01", "EXP-A6-01", "EXP-A2-01", "EXP-A3-01", "EXP-A5A6-01",
    }
    index_rows = [r for r in rows if r["required_data_asset"] == "index_vix"]
    assert index_rows
    assert all(r["status"] == "NOT_REQUIRED" for r in index_rows)
    univ = [r for r in rows if r["required_data_asset"] == "universe_membership_history"]
    assert all(r["status"] == "NOT_REQUIRED" for r in univ)
    assert all(r["universe_mode"] == "FIXED_PREREGISTERED_29" for r in rows)


def test_sector_requirement_split():
    assert sc.SECTOR_UNUSED == {"EXP-A2-01", "EXP-A3-01"}
    assert "EXP-A5-01" in sc.SECTOR_STATIC_CONSUMERS
    assert "EXP-A6-01" in sc.SECTOR_STATIC_CONSUMERS
    assert "EXP-A5A6-01" in sc.SECTOR_STATIC_CONSUMERS


def test_repo_root_points_at_workspace():
    assert (sc.REPO_ROOT / "PHASE_A5_FROZEN_PROTOCOLS.md").exists()
    assert (sc.REPO_ROOT / "research" / "phase_a5" / "scoped_certification.py").exists()


def test_render_markdown_includes_required_sections_and_matrix():
    cert = {
        "global_trust_class": "OPERATIONAL_ONLY",
        "certification": "READY_FOR_SCIENTIFIC_RERUN",
        "phase_a5_rerun_executed": False,
        "phase_b_started": False,
        "panel": list(sc.FROZEN_PANEL),
        "date_start": sc.FROZEN_DATE_START,
        "date_end": sc.FROZEN_DATE_END,
        "hypothesis_ids": list(sc.FROZEN_HYPOTHESIS_IDS),
        "protocol_version": sc.PROTOCOL_VERSION,
        "validator_version": sc.VALIDATOR_VERSION,
        "adjustment_policy_version": "ca_sharecount_v1",
        "git_sha": "abc",
        "evaluated_at": "2026-08-11T00:00:00+00:00",
        "dependency_matrix": sc.protocol_dependency_matrix()[:3],
        "identity": {
            "ok": True,
            "blockers": [],
            "security_ids": {s: f"isin:X{s}" for s in sc.FROZEN_PANEL},
            "rows": [
                {
                    "symbol": s, "security_id": f"isin:X{s}", "isin": f"X{s}",
                    "listing_date": "2000-01-01", "delisting_date": None,
                    "status": "VERIFIED", "lineage_status": "NOT_APPLICABLE",
                }
                for s in sc.FROZEN_PANEL
            ],
        },
        "ca": {
            "ok": True,
            "consecutive_events_in_window": 0,
            "verified_ca_transitions": 0,
            "unresolved_consecutive": [],
            "adjustment_policy_version": "ca_sharecount_v1",
            "ca_event_ledger": [],
        },
        "universe": {
            "ok": True,
            "universe_mode": "FIXED_PREREGISTERED_29",
            "mode": "A",
            "dynamic_pit_membership_required": False,
            "ledger_source": "test",
            "protocol_note": "fixed panel",
        },
        "sector": {
            "ok": True,
            "requirement": "STATIC_MAP_ONLY",
            "pit_sector_history_required": False,
            "protocol_note": "static only",
        },
        "price": {
            "metric": "unresolved_consecutive_session_symbol_rate",
            "threshold": 0.002,
            "total_consecutive_session_transitions": 100,
            "verified_ca_transitions": 0,
            "genuine_large_market_moves": 0,
            "unresolved_discontinuities": 0,
            "unresolved_rate": 0.0,
            "unresolved_event_rate_vs_all_transitions": 0.0,
            "sparse_or_suspension_events": 0,
            "thin_history": [],
            "note": "test",
        },
        "pit": {"ok": True, "mode": "FIXED_PANEL_ASOF_BARS", "note": "test"},
        "snapshot": {"snapshot_id": "deadbeef", "root": "/tmp", "verify_ok": True},
        "hashes": {"ca_events": "aa"},
        "per_experiment": [
            {
                "experiment": eid,
                "IDENTITY": "VERIFIED",
                "CA": "VERIFIED",
                "UNIVERSE": "VERIFIED",
                "SECTOR": "NOT_REQUIRED" if eid in sc.SECTOR_UNUSED else "VERIFIED_STATIC",
                "PRICE": "VERIFIED",
                "PIT": "VERIFIED",
                "SNAPSHOT": "COMMITTED",
                "CERTIFICATION": "READY_FOR_SCIENTIFIC_RERUN",
            }
            for eid, _ in sc.EXPERIMENT_ROWS
        ],
        "user_facing": {
            "layer1": {
                "explanation": (
                    "QuantTerm's full historical database is not yet certified for "
                    "scientific research. However, we separately checked the exact "
                    "historical data needed for this specific frozen test. The specific "
                    "data used by this test passed the required historical checks, so "
                    "the test can now be rerun scientifically."
                ),
                "implication": "Global trust stays OPERATIONAL_ONLY.",
            }
        },
    }
    md = sc.render_certification_markdown(cert)
    assert "Global trust state" in md
    assert "OPERATIONAL_ONLY" in md
    assert "READY_FOR_SCIENTIFIC_RERUN" in md
    assert "EXPERIMENT | IDENTITY | CA | UNIVERSE | SECTOR | PRICE" in md
    assert "Do **not** begin Phase B" in md
    assert "full historical database is not yet certified" in md
    assert "PASS/FAIL" in md  # explicitly says not strategy PASS/FAIL


def test_unresolved_rate_threshold_unchanged():
    assert sc.UNRESOLVED_RATE_MAX == 0.002
