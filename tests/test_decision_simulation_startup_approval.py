from __future__ import annotations

from product import decision_simulation_gate as G


def test_one_startup_approval_survives_autonomous_thesis_evolution(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-1")

    thesis = {"value": "thesis-a"}
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": thesis["value"], "objective_id": "test"},
    )
    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": True,
            "scan_scanned_at": "2026-09-18T10:00:00+00:00",
            "best_trades": [{"symbol": "INFY"}],
            "decisions": [{"symbol": "INFY"}],
            "actionable": 1,
        },
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": ""},
    )
    monkeypatch.setattr(
        "product.historical_paper_loop.reset_for_thesis",
        lambda *_a, **_k: {},
    )

    G.begin_startup("startup-1", path=state)
    approved = G.approve(path=state)
    assert approved["accepted"] is True
    assert approved["approved"] is True
    assert approved["approved_thesis_hash"] == "thesis-a"
    assert G.is_approved(path=state) is True

    thesis["value"] = "thesis-b"

    evolved = G.status(path=state)
    assert evolved["approved"] is True
    assert evolved["approval_required"] is False
    assert evolved["approved_thesis_hash"] == "thesis-a"
    assert evolved["current_thesis_hash"] == "thesis-b"
    assert evolved["thesis_changed_since_approval"] is True
    assert G.is_approved(path=state) is True

    # Re-running startup initialization for the same complete-stack process must
    # not turn learning into another operator prompt.
    same = G.begin_startup("startup-1", path=state)
    assert same["approved"] is True
    assert G.is_approved(path=state) is True


def test_new_complete_stack_startup_requires_one_new_approval(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-1")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-a", "objective_id": "test"},
    )
    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": True,
            "scan_scanned_at": "2026-09-18T10:00:00+00:00",
            "best_trades": [{"symbol": "INFY"}],
            "decisions": [{"symbol": "INFY"}],
            "actionable": 1,
        },
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": ""},
    )
    monkeypatch.setattr(
        "product.historical_paper_loop.reset_for_thesis",
        lambda *_a, **_k: {},
    )

    G.begin_startup("startup-1", path=state)
    assert G.approve(path=state)["accepted"] is True

    monkeypatch.setenv("QT_STARTUP_ID", "startup-2")
    fresh = G.begin_startup("startup-2", path=state)

    assert fresh["approved"] is False
    assert fresh["approval_required"] is True
    assert G.is_approved(path=state) is False
