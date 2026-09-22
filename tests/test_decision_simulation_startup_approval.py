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



def test_gate_board_uses_precomputed_discovery_projection(monkeypatch):
    import product.decision_discovery_store as discovery
    import product.recommendations_workspace as workspace_mod
    import product.scan_store as scan_store
    import product.long_term_store as long_term_store
    import product.trading_thesis as thesis_mod

    monkeypatch.setattr(
        scan_store,
        "load_scan",
        lambda: {
            "scanned_at": "2026-09-18T10:00:00+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )
    monkeypatch.setattr(
        long_term_store,
        "load_long_term_scan",
        lambda: {"scanned_at": "2026-09-18T09:00:00+00:00", "records": []},
    )
    monkeypatch.setattr(
        thesis_mod,
        "manifest",
        lambda: {"thesis_hash": "thesis-a"},
    )
    cached = {
        "available": True,
        "scan_scanned_at": "2026-09-18T10:00:00+00:00",
        "best_trades": [{"symbol": "INFY"}],
        "decisions": [{"symbol": "INFY"}],
        "actionable": 1,
    }
    monkeypatch.setattr(discovery, "load", lambda **_kwargs: dict(cached))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("gate must not rebuild recommendations when discovery cache matches")

    monkeypatch.setattr(workspace_mod, "build_recommendations_workspace", forbidden)

    assert G._board() == cached



def test_stale_gate_never_rebuilds_decision_board(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-stale")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-a", "objective_id": "test"},
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: False)
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-09-17T10:00:00+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )

    def forbidden():
        raise AssertionError("stale gate must not rebuild/rank a decision board")

    monkeypatch.setattr(G, "_board", forbidden)

    G.begin_startup("startup-stale", path=state)
    payload = G.status(path=state)

    assert payload["scan_fresh"] is False
    assert payload["discovery_ready"] is False
    assert payload["best_trades"] == []
    assert payload["phase"] == "SEARCHING_BEST_TRADES"


def test_startup_discovery_runs_during_truthful_publication_grace(tmp_path, monkeypatch):
    """Weekend/publication grace is usable official history, not a discovery deadlock."""
    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {"discovery_ready": False},
    )
    monkeypatch.setattr(
        "product.decision_simulation_gate.current_startup_id",
        lambda: "startup-weekend",
    )
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {
            "current": False,
            "usable_for_scan": True,
            "publication_pending": True,
            "reason_code": "HISTORY_PUBLICATION_PENDING",
            "available_session": "2026-09-17",
            "expected_latest_completed_session": "2026-09-18",
        },
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    sup._ensure_startup_trade_discovery()

    queued = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.MARKET_SCAN
    ]
    assert len(queued) == 1
    assert queued[0].idempotency_key == (
        "startup_discovery_scan:startup-weekend:market:official_nse:2026-09-17"
    )


def test_startup_discovery_still_blocks_genuinely_stale_history(tmp_path, monkeypatch):
    """The liveness fix must not weaken the canonical freshness safety gate."""
    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {"discovery_ready": False},
    )
    monkeypatch.setattr(
        "product.decision_simulation_gate.current_startup_id",
        lambda: "startup-stale",
    )
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {
            "current": False,
            "usable_for_scan": False,
            "publication_pending": False,
            "reason_code": "HISTORY_STALE",
            "available_session": "2026-09-16",
            "expected_latest_completed_session": "2026-09-18",
        },
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    sup._ensure_startup_trade_discovery()

    assert not [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.MARKET_SCAN
    ]


def test_scan_fresh_prefers_schema_v2_provenance_identity(monkeypatch):
    from product import desk_pipeline as desk

    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda **_kwargs: {
            "current": True,
            "usable_for_scan": True,
            "available_session": "2026-09-18",
            "expected_latest_completed_session": "2026-09-18",
        },
    )
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda *_a, **_k: {
            "schema_version": 2,
            "scanned_at": "2026-09-18T12:00:00+00:00",
            "records": [{"symbol": "INFY"}],
            "provenance": {
                "market_session_date": "2026-09-18",
                "price_data_as_of": "2026-09-18",
            },
        },
    )

    assert desk.scan_is_fresh() is True


def test_scan_fresh_rejects_stale_schema_v2_provenance_identity(monkeypatch):
    from product import desk_pipeline as desk

    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda **_kwargs: {
            "current": True,
            "usable_for_scan": True,
            "available_session": "2026-09-18",
            "expected_latest_completed_session": "2026-09-18",
        },
    )
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda *_a, **_k: {
            "schema_version": 2,
            "scanned_at": "2026-09-18T12:00:00+00:00",
            "records": [{"symbol": "INFY"}],
            "provenance": {"market_session_date": "2026-09-17"},
        },
    )

    assert desk.scan_is_fresh() is False



def test_autonomous_approval_persists_provenance(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-auto")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-auto", "objective_id": "test"},
    )
    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": True,
            "scan_scanned_at": "2026-09-21T16:22:28+00:00",
            "best_trades": [{"symbol": "INFY"}],
            "decisions": [{"symbol": "INFY"}],
            "actionable": 1,
        },
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": "thesis-auto"},
    )

    G.begin_startup("startup-auto", path=state)
    approved = G.ensure_autonomous_approval(path=state)

    assert approved["accepted"] is True
    assert approved["approved"] is True
    assert approved["approval_source"] == "AUTONOMY"
    assert approved["approval_required"] is False
    assert G.is_approved(path=state) is True


def test_supervisor_auto_authorizes_when_discovery_is_ready(tmp_path, monkeypatch):
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    calls = []
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {"discovery_ready": True, "approved": False},
    )
    monkeypatch.setattr(
        "product.decision_simulation_gate.ensure_autonomous_approval",
        lambda: calls.append("auto") or {
            "accepted": True,
            "approved": True,
            "approval_source": "AUTONOMY",
        },
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    sup._ensure_startup_trade_discovery()

    assert calls == ["auto"]



def test_autonomous_approval_refuses_stale_or_unready_discovery(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-auto-stale")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-auto", "objective_id": "test"},
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: False)
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-09-20T10:00:00+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )

    G.begin_startup("startup-auto-stale", path=state)
    result = G.ensure_autonomous_approval(path=state)

    assert result["accepted"] is False
    assert result["approved"] is False
    assert result["reason"] == "BEST_TRADE_DISCOVERY_NOT_READY"
    assert result["scan_fresh"] is False
    assert G.is_approved(path=state) is False



def test_gate_board_cache_miss_is_nonblocking_and_never_builds(monkeypatch):
    import product.decision_discovery_store as discovery
    import product.decision_service as decision_service
    import product.long_term_store as long_term_store
    import product.recommendations_workspace as workspace_mod
    import product.scan_store as scan_store
    import product.trading_thesis as thesis_mod

    monkeypatch.setattr(
        scan_store,
        "load_scan",
        lambda: {
            "scanned_at": "2026-09-22T10:00:00+00:00",
            "records": [{"symbol": "INFY"}],
        },
    )
    monkeypatch.setattr(
        long_term_store,
        "load_long_term_scan",
        lambda: {"scanned_at": "2026-09-22T09:00:00+00:00", "records": []},
    )
    monkeypatch.setattr(
        thesis_mod,
        "manifest",
        lambda: {"thesis_hash": "thesis-a"},
    )
    monkeypatch.setattr(discovery, "load", lambda **_kwargs: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("status GET must never perform discovery computation")

    monkeypatch.setattr(workspace_mod, "build_recommendations_workspace", forbidden)
    monkeypatch.setattr(decision_service, "decision_board", forbidden)

    payload = G._board()

    assert payload["available"] is False
    assert payload["state"] == "SEARCHING_BEST_TRADES"
    assert payload["scan_scanned_at"] == "2026-09-22T10:00:00+00:00"
    assert payload["best_trades"] == []
    assert payload["decisions"] == []
    assert payload["status_source"] == "persisted_discovery_missing"
    assert "does not rebuild recommendations" in payload["reason"]
