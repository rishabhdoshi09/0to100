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


def test_approved_current_scan_stays_discovery_ready_if_projection_cache_temporarily_missing(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-approved-cache-gap")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-a", "objective_id": "test"},
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)

    board = {
        "available": True,
        "scan_scanned_at": "2026-09-22T10:00:00+00:00",
        "best_trades": [{"symbol": "INFY"}],
        "decisions": [{"symbol": "INFY"}],
        "actionable": 1,
    }
    monkeypatch.setattr(G, "_board", lambda: dict(board))
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": "thesis-a"},
    )

    G.begin_startup("startup-approved-cache-gap", path=state)
    approved = G.approve(path=state)
    assert approved["approved"] is True
    assert approved["discovery_ready"] is True

    # Simulate an atomic projection replacement window after approval. The scan
    # itself has not changed, so durable approval provenance proves discovery
    # already completed for this exact scan.
    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": False,
            "state": "SEARCHING_BEST_TRADES",
            "reason": "projection refresh in progress",
            "scan_scanned_at": "2026-09-22T10:00:00+00:00",
            "best_trades": [],
            "decisions": [],
            "actionable": 0,
        },
    )

    status = G.status(path=state)
    assert status["approved"] is True
    assert status["discovery_ready"] is True
    assert status["phase"] == "APPROVED"


def test_approved_old_scan_does_not_mark_new_scan_discovery_ready(tmp_path, monkeypatch):
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-new-scan")
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": "thesis-a", "objective_id": "test"},
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": "thesis-a"},
    )

    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": True,
            "scan_scanned_at": "2026-09-22T10:00:00+00:00",
            "best_trades": [{"symbol": "INFY"}],
            "decisions": [{"symbol": "INFY"}],
            "actionable": 1,
        },
    )
    G.begin_startup("startup-new-scan", path=state)
    assert G.approve(path=state)["approved"] is True

    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": False,
            "state": "SEARCHING_BEST_TRADES",
            "reason": "new scan projection not published yet",
            "scan_scanned_at": "2026-09-22T11:00:00+00:00",
            "best_trades": [],
            "decisions": [],
            "actionable": 0,
        },
    )

    status = G.status(path=state)
    assert status["approved"] is True
    assert status["discovery_ready"] is False


def test_current_scan_with_new_thesis_queues_one_discovery_refresh_not_another_scan(
    tmp_path, monkeypatch
):
    """Learning identity changes re-project the saved scan without network/scan churn."""
    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": False,
            "approved": True,
            "scan_fresh": True,
            "scan_scanned_at": "2026-09-22T19:02:49+00:00",
            "current_thesis_hash": "thesis-b",
        },
    )

    def forbidden_history():
        raise AssertionError("fresh saved scan must be re-projected, not re-scanned")

    monkeypatch.setattr("product.readiness.official_history", forbidden_history)

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    sup._ensure_startup_trade_discovery()
    sup._ensure_startup_trade_discovery()

    refreshes = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.DISCOVERY_REFRESH
    ]
    scans = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.MARKET_SCAN
    ]
    assert len(refreshes) == 1
    assert scans == []
    assert refreshes[0].critical is True
    assert refreshes[0].input_snapshot_id == "2026-09-22T19:02:49+00:00"
    assert refreshes[0].idempotency_key == SCH.discovery_refresh_key(
        "2026-09-22T19:02:49+00:00",
        "",
        "thesis-b",
    )


def test_durable_startup_approval_does_not_authorize_stale_current_projection(
    tmp_path, monkeypatch
):
    """Approval survives learning, but new simulation waits for current projection truth."""
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    gate = {
        "discovery_ready": False,
        "approved": True,
        "scan_fresh": True,
        "scan_scanned_at": "2026-09-22T19:02:49+00:00",
        "current_thesis_hash": "thesis-b",
    }
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: dict(gate),
    )

    def forbidden_approval():
        raise AssertionError("stale current discovery must not be re-approved")

    monkeypatch.setattr(
        "product.decision_simulation_gate.ensure_autonomous_approval",
        forbidden_approval,
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    assert sup._ensure_decision_simulation_authority() is False

    gate["discovery_ready"] = True
    assert sup._ensure_decision_simulation_authority() is True


def test_manual_paper_control_waits_durably_for_discovery_refresh(tmp_path, monkeypatch):
    """An accepted control is not lost when immutable discovery is between identities."""
    from datetime import datetime, timezone

    from research.autonomy import controls as CTRL
    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return "snapshot-1"

        def now_ist(self):
            return datetime(2026, 9, 22, 18, 0, tzinfo=timezone.utc)

    approval = {"accepted": False}
    monkeypatch.setattr(
        "product.decision_simulation_gate.approve",
        lambda: dict(approval),
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    refresh_calls = []
    monkeypatch.setattr(
        sup,
        "_ensure_startup_trade_discovery",
        lambda: refresh_calls.append("refresh"),
    )
    control = sup.controls.request(CTRL.RUN_CYCLE_NOW, requested_by="test")

    sup._process_controls()

    assert refresh_calls == ["refresh"]
    assert [row.control_id for row in sup.controls.pending()] == [control.control_id]
    assert not [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.PAPER_CYCLE
    ]

    approval["accepted"] = True
    sup._process_controls()

    assert sup.controls.pending() == []
    recent = {row.control_id: row for row in sup.controls.recent(limit=20)}
    assert recent[control.control_id].status == CTRL.PROCESSED
    paper = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.PAPER_CYCLE
    ]
    assert len(paper) == 1
    assert paper[0].critical is True
    assert paper[0].idempotency_key == (
        f"manual:cycle:snapshot-1:{control.control_id}"
    )


def test_approved_scan_cache_gap_does_not_mask_changed_thesis_identity(tmp_path, monkeypatch):
    """An old approval cannot make a missing new-thesis projection look ready."""
    state = tmp_path / "gate.json"
    monkeypatch.setenv("QT_STARTUP_ID", "startup-thesis-gap")

    thesis = {"value": "thesis-a"}
    monkeypatch.setattr(
        "product.trading_thesis.manifest",
        lambda: {"thesis_hash": thesis["value"], "objective_id": "test"},
    )
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {"thesis_hash": "thesis-a"},
    )

    board = {
        "available": True,
        "scan_scanned_at": "2026-09-22T10:00:00+00:00",
        "best_trades": [{"symbol": "INFY"}],
        "decisions": [{"symbol": "INFY"}],
        "actionable": 1,
    }
    monkeypatch.setattr(G, "_board", lambda: dict(board))

    G.begin_startup("startup-thesis-gap", path=state)
    approved = G.approve(path=state)
    assert approved["approved"] is True
    assert approved["discovery_ready"] is True
    assert approved["approved_thesis_hash"] == "thesis-a"

    thesis["value"] = "thesis-b"
    monkeypatch.setattr(
        G,
        "_board",
        lambda: {
            "available": False,
            "state": "SEARCHING_BEST_TRADES",
            "reason": "projection for changed thesis not published yet",
            "scan_scanned_at": "2026-09-22T10:00:00+00:00",
            "best_trades": [],
            "decisions": [],
            "actionable": 0,
        },
    )

    evolved = G.status(path=state)
    assert evolved["approved"] is True
    assert evolved["thesis_changed_since_approval"] is True
    assert evolved["discovery_ready"] is False
    assert evolved["best_trades"] == []


def test_long_term_identity_change_gets_a_new_discovery_refresh_job(tmp_path, monkeypatch):
    """Discovery idempotency includes every store identity used by the cache key."""
    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class DiscoveryDeps:
        def active_snapshot_id(self):
            return ""

    gate = {
        "discovery_ready": False,
        "approved": True,
        "scan_fresh": True,
        "scan_scanned_at": "2026-09-22T19:02:49+00:00",
        "long_term_scanned_at": "2026-09-22T18:00:00+00:00",
        "current_thesis_hash": "thesis-b",
    }
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: dict(gate),
    )

    sup = Supervisor(tmp_path / "auto", deps=DiscoveryDeps())
    sup._ensure_startup_trade_discovery()
    sup._ensure_startup_trade_discovery()

    first = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.DISCOVERY_REFRESH
    ]
    assert len(first) == 1

    gate["long_term_scanned_at"] = "2026-09-22T18:30:00+00:00"
    sup._ensure_startup_trade_discovery()

    refreshes = [
        job for job in sup.jobs.list(limit=20)
        if job.job_type == SCH.DISCOVERY_REFRESH
    ]
    assert len(refreshes) == 2
    assert {job.idempotency_key for job in refreshes} == {
        SCH.discovery_refresh_key(
            "2026-09-22T19:02:49+00:00",
            "2026-09-22T18:00:00+00:00",
            "thesis-b",
        ),
        SCH.discovery_refresh_key(
            "2026-09-22T19:02:49+00:00",
            "2026-09-22T18:30:00+00:00",
            "thesis-b",
        ),
    }


def test_intraday_fresh_scan_repairs_discovery_even_when_broker_auth_is_unavailable(
    tmp_path, monkeypatch
):
    """Intraday auth/data flow must not strand a fresh scan with stale discovery."""
    from datetime import datetime, timezone

    from research.autonomy import schedules as SCH
    from research.autonomy.supervisor import Supervisor

    class IntradayDeps:
        def holidays(self):
            return set()

        def active_snapshot_id(self):
            return ""

    monkeypatch.setattr(SCH, "market_is_open", lambda *_a, **_k: True)
    monkeypatch.setattr(SCH, "in_scan_window", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": False,
            "approved": False,
            "scan_fresh": True,
            "scan_scanned_at": "2026-09-23T06:35:49.779226+00:00",
            "long_term_scanned_at": "2026-09-23T06:36:56.081161+00:00",
            "current_thesis_hash": "thesis-intraday",
        },
    )

    sup = Supervisor(tmp_path / "auto", deps=IntradayDeps())
    monkeypatch.setattr(sup, "_release_stale_official_blocks", lambda: None)

    sup.enqueue_due(datetime(2026, 9, 23, 6, 45, tzinfo=timezone.utc))

    refreshes = [
        job for job in sup.jobs.list(limit=50)
        if job.job_type == SCH.DISCOVERY_REFRESH
    ]
    assert len(refreshes) == 1
    assert refreshes[0].critical is True
    assert refreshes[0].input_snapshot_id == "2026-09-23T06:35:49.779226+00:00"
    assert refreshes[0].idempotency_key == SCH.discovery_refresh_key(
        "2026-09-23T06:35:49.779226+00:00",
        "2026-09-23T06:36:56.081161+00:00",
        "thesis-intraday",
    )
