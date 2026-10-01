from __future__ import annotations

import json
import inspect
import time

import terminal_api
from scripts.run_product_acceptance import grade_canonical_health, product_acceptance_verdict


def test_capability_strings_are_not_coerced_by_python_truthiness():
    assert terminal_api._capability("allowed") == "allowed"
    assert terminal_api._capability("limited") == "limited"
    assert terminal_api._capability("blocked") == "blocked"
    assert terminal_api._capability("") == "blocked"
    assert terminal_api._capability(None) == "blocked"


def test_terminal_controls_have_no_live_broker_or_order_action():
    assert terminal_api._ALLOWED_CONTROLS == {
        "RUN_SCAN_NOW",
        "RUN_LONG_TERM_SCAN_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_MARKET_REPORT_NOW",
        "REFRESH_FNO_NOW",
        "RUN_CYCLE_NOW",
        "REFRESH_DATA_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
        "OBSERVE_ONLY_TODAY",
        "CLEAR_OBSERVE_ONLY",
        "RUN_HISTORICAL_REPLAY",
        "RUN_LEARNING_NOW",
    }
    source = inspect.getsource(terminal_api.control).lower()
    assert "broker" not in source
    assert "order" not in source


def test_paper_payload_exposes_daily_learning_and_keeps_live_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_MEMORY", str(tmp_path / "paper_memory.json"))
    from product.paper_learning import remember_paper_book
    remember_paper_book(
        [
            {"symbol": "TCS", "pnl": -110, "realized_R": -1.1, "exit_date": "2026-08-20", "exit_reason": "STOP"},
            {"symbol": "TCS", "pnl": -40, "realized_R": -0.4, "exit_date": "2026-08-22", "exit_reason": "STOP"},
        ],
        as_of="2026-08-24",
    )
    payload = terminal_api._paper_payload()
    learning = payload["learning"]
    assert learning["live_locked"] is True
    assert learning["closed_trades"] == 2
    assert learning["cooldown"][0]["symbol"] == "TCS"
    assert "explicit deployment authorization" in learning["disclaimer"].lower()


def test_paper_payload_projects_backend_field_names_for_the_frontend(monkeypatch):
    """research.auto_research.paper_book's PaperPosition/ClosedTrade dataclasses
    use qty/stop_price/target_price/strategy_id/bars_held/realized_R --
    frontend/src/types.ts's PaperPosition (rendered by PositionsTable) reads
    quantity/stop/target/strategy/days_held/result_r. Without a projection,
    the Paper Portfolio table renders QTY 0 and blank stop/target for data
    that is present, just under the backend's own field names."""
    from product.paper_status import PaperStatus

    open_position = {
        "strategy_id": "swing_breakout", "symbol": "HAL", "entry_price": 4500.0,
        "stop_price": 4300.0, "target_price": 4900.0, "qty": 10,
        "entry_date": "2026-08-20", "max_holding_days": 10, "risk_amount": 2000.0,
        "bars_held": 3,
    }
    closed_trade = {
        "strategy_id": "swing_breakout", "symbol": "TCS", "entry_price": 3500.0,
        "exit_price": 3400.0, "stop_price": 3400.0, "qty": 5,
        "entry_date": "2026-08-10", "exit_date": "2026-08-15",
        "exit_reason": "STOP", "realized_R": -1.0, "pnl": -500.0,
    }
    import product.paper_status as paper_status_module
    monkeypatch.setattr(
        paper_status_module, "read_paper_status",
        lambda: PaperStatus(open_positions=(open_position,), closed_trades=(closed_trade,)),
    )
    payload = terminal_api._paper_payload()

    projected_open = payload["open_positions"][0]
    assert projected_open["quantity"] == 10
    assert projected_open["stop"] == 4300.0
    assert projected_open["target"] == 4900.0
    assert projected_open["strategy"] == "swing_breakout"
    assert projected_open["days_held"] == 3
    # Originals stay too -- other consumers of this same payload may read them.
    assert projected_open["qty"] == 10

    projected_closed = payload["closed_trades"][0]
    assert projected_closed["quantity"] == 5
    assert projected_closed["stop"] == 3400.0
    assert projected_closed["current_price"] == 3400.0  # exit_price, honestly
    assert projected_closed["result_r"] == -1.0
    assert projected_closed["pnl"] == -500.0
    assert projected_closed["exit_reason"] == "STOP"


def test_market_controls_are_dispatched_outside_paper_autonomy():
    assert terminal_api._OPERATION_CONTROLS == {
        "RUN_SCAN_NOW": "MARKET_SCAN",
        "RUN_LONG_TERM_SCAN_NOW": "MARKET_SCAN",
        "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
        "REFRESH_NEWS_NOW": "NEWS_REFRESH",
        "REFRESH_MARKET_REPORT_NOW": "MARKET_REPORT",
        "REFRESH_FNO_NOW": "FNO_REFRESH",
        "REFRESH_DATA_NOW": "DATA_PREPARE",
    }
    assert terminal_api._AUTONOMY_CONTROLS == {
        "RUN_CYCLE_NOW",
        "PAUSE_NEW_PAPER_ENTRIES",
        "RESUME_NEW_PAPER_ENTRIES",
        "OBSERVE_ONLY_TODAY",
        "CLEAR_OBSERVE_ONLY",
        "RUN_HISTORICAL_REPLAY",
        "RUN_LEARNING_NOW",
    }


def test_ops_runtime_treats_live_lock_owner_as_running(tmp_path, monkeypatch):
    import json
    import os

    ops = tmp_path / "market_ops"
    ops.mkdir()
    (ops / "worker.lock").write_text(str(os.getpid()), encoding="utf-8")
    (ops / "runtime.json").write_text(
        json.dumps({"process_running": False, "worker_pid": 1, "heartbeat_epoch": 0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(terminal_api, "OPS_ROOT", ops)
    monkeypatch.setattr(terminal_api, "OPS_RUNTIME", ops / "runtime.json")
    payload = terminal_api._ops_runtime_payload()
    assert payload["running"] is True
    assert payload["worker_pid"] == os.getpid()
    assert payload.get("recovering") is True


def test_health_is_a_cheap_liveness_probe():
    import inspect

    src = inspect.getsource(terminal_api.health)
    assert "_autonomy_payload" not in src
    assert "_operations_payload" not in src
    payload = terminal_api.health()
    assert payload["ok"] is True
    assert payload["service"] == "quantterm-terminal-api"


def test_health_surfaces_inspect_runtime_ready_flags_without_inventing_them(monkeypatch):
    monkeypatch.setattr(
        "product.runtime_lifecycle.inspect_runtime",
        lambda **_k: {
            "lifecycle": "READY",
            "reason": "Required services are alive and official history is current",
            "reasons": [],
            "components": [],
            "history": {},
            "resources": {},
            "operational_ready": True,
            "evidence_ready": True,
            "live_locked": True,
            "live_lock_verified": True,
            "live_execution_authorized": False,
        },
    )
    payload = terminal_api.health()
    assert payload["ok"] is True
    assert payload["lifecycle"] == "READY"
    assert payload["operational_ready"] is True
    assert payload["evidence_ready"] is True
    assert payload["live_locked"] is True
    assert payload["live_lock_verified"] is True
    assert payload["live_execution_authorized"] is False
    graded = grade_canonical_health(payload)
    assert graded["status"] == "PASS"
    assert product_acceptance_verdict(
        [{"feature": "Canonical stack / readiness", "status": graded["status"]}],
        live_locked=graded["live_locked"],
        live_lock_verified=graded["live_lock_verified"],
    )["verdict"] == "PRODUCT ACCEPTANCE PASS"

    monkeypatch.setattr(
        "product.runtime_lifecycle.inspect_runtime",
        lambda **_k: {
            "lifecycle": "READY",
            "reason": "Required services are alive and official history is current",
            "reasons": [],
            "components": [],
            "history": {},
            "resources": {},
        },
    )
    omitted = terminal_api.health()
    assert omitted.get("operational_ready") is not True
    assert omitted.get("evidence_ready") is not True
    assert omitted.get("live_locked") is not True
    omitted_grade = grade_canonical_health(omitted)
    assert omitted_grade["status"] == "FAIL"
    assert product_acceptance_verdict(
        [{"feature": "Canonical stack / readiness", "status": omitted_grade["status"]}],
        live_locked=omitted_grade["live_locked"],
        live_lock_verified=omitted_grade["live_lock_verified"],
    )["verdict"] == "PRODUCT ACCEPTANCE HOLD"


def test_json_safe_strips_nan_and_inf():
    payload = terminal_api._json_safe({"ok": 1.0, "bad": float("nan"), "rows": [float("inf"), 2.0]})
    assert payload == {"ok": 1.0, "bad": None, "rows": [None, 2.0]}


def test_dashboard_keeps_last_scan_when_a_lane_explodes(monkeypatch):
    scan = {
        "available": True,
        "scanned_at": "2026-08-24T04:41:16+00:00",
        "universe_size": 2,
        "summary": {},
        "records": [{"symbol": "TCS", "score": 80, "signals": ["MOMENTUM"]}],
    }
    monkeypatch.setattr(terminal_api, "_scan_payload", lambda: scan)
    monkeypatch.setattr(terminal_api, "_market_payload", lambda: {"available": True, "health": "Neutral"})
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})

    def boom(*_args, **_kwargs):
        raise RuntimeError("bhavcopy cache exploded")

    monkeypatch.setattr(terminal_api, "_data_payload", boom)
    payload = terminal_api.dashboard()
    assert payload["scan"]["records"][0]["symbol"] == "TCS"
    assert payload["data"]["scan_saved"] is True
    assert "degraded" in str(payload.get("error", "")).lower()


def test_radar_home_keeps_watchlist_when_sepa_ranking_fails(monkeypatch):
    from product import observer_api

    scan = {
        "available": True,
        "scanned_at": "2026-08-24T04:41:16+00:00",
        "universe_size": 2,
        "summary": {},
        "records": [
            {
                "symbol": "TCS",
                "score": 80,
                "signals": ["MOMENTUM"],
                "price": 100,
                "entry": 101,
                "stop": 95,
                "target": 120,
                "verdict": "WATCH",
                "chase_risk": False,
                "reasons": ["trend"],
            }
        ],
    }
    monkeypatch.setattr(observer_api.core, "_scan_payload", lambda: scan)
    monkeypatch.setattr(
        observer_api.core,
        "_market_payload",
        lambda: {
            "available": True,
            "health": "Neutral",
            "summary": "ok",
            "trade_stance": "Wait",
            "breadth": "mixed",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": 0.2,
            "nifty_change_5d": 0.1,
            "vix": 12.0,
            "technical_details": {},
        },
    )
    monkeypatch.setattr(observer_api.core, "_long_term_payload", lambda: {"available": False, "records": []})

    def boom(*_args, **_kwargs):
        raise RuntimeError("sepa unavailable")

    monkeypatch.setattr("product.sepa_setup.public_best_setups", boom)
    home = observer_api.radar_home_workspace()
    assert home["lanes"]["momentum"][0]["symbol"] == "TCS"
    assert home["best_setups"] == []
    assert "unavailable" in home["best_setups_note"].lower()
    assert "telegram" in home
    assert "headline" in home["telegram"]
    assert "desk_pipeline" in home


def test_dashboard_slims_scan_keeps_universe_and_conviction_input(monkeypatch):
    records = [{"symbol": f"S{i:03d}", "score": i, "composite": i} for i in range(120)]
    scan = {
        "available": True,
        "scanned_at": "2026-09-02T05:00:00+00:00",
        "universe_size": 1842,
        "summary": {"with_any_setup": 12},
        "records": records,
    }
    seen: dict[str, int] = {}
    monkeypatch.setattr(terminal_api, "_scan_payload", lambda: scan)
    monkeypatch.setattr(
        terminal_api,
        "_market_payload",
        lambda: {
            "available": True,
            "health": "Mixed",
            "summary": "ok",
            "trade_stance": "Wait",
            "breadth": "mixed",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": 0.1,
            "nifty_change_5d": 0.2,
            "vix": 12.0,
            "nifty_price": 25000,
            "technical_details": {},
        },
    )
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False, "articles": []})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})
    monkeypatch.setattr(terminal_api, "_scan_progress_payload", lambda: {})

    def fake_data(scan_arg, *_args):
        return {
            "ready": False,
            "snapshot": {},
            "bhavcopy": {},
            "scan_saved": True,
            "scan_records": len(scan_arg.get("records") or []),
            "long_term_saved": False,
            "long_term_records": 0,
            "blockers": [],
        }

    def fake_conviction(scan_arg, _market):
        seen["n"] = len(scan_arg.get("records") or [])
        return [{"symbol": "S119"}]

    monkeypatch.setattr(terminal_api, "_data_payload", fake_data)
    monkeypatch.setattr(terminal_api, "_conviction", fake_conviction)

    payload = terminal_api.dashboard()
    assert payload["scan"]["universe_size"] == 1842
    assert payload["scan"]["dashboard_records_shown"] == 80
    assert len(payload["scan"]["records"]) == 80
    assert payload["scan"]["records"][0]["symbol"] == "S119"
    assert payload["data"]["scan_records"] == 120
    assert seen["n"] == 120


def test_dashboard_returns_without_waiting_for_regime_fetch(monkeypatch):
    def hang():
        time.sleep(8)
        raise AssertionError("regime fetch must stay off the dashboard request")

    monkeypatch.setattr("core.regime_engine.compute_regime", hang)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: None)
    monkeypatch.setattr(
        terminal_api,
        "_scan_payload",
        lambda: {"available": True, "scanned_at": "2026-09-02T05:00:00+00:00", "universe_size": 10, "summary": {}, "records": []},
    )
    monkeypatch.setattr(terminal_api, "_long_term_payload", lambda: {"available": False, "records": [], "summary": {}, "job": {}})
    monkeypatch.setattr(terminal_api, "_paper_payload", lambda: {"available": False})
    monkeypatch.setattr(terminal_api, "_autonomy_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_operations_payload", lambda: {"available": False, "running": False})
    monkeypatch.setattr(terminal_api, "_news_payload", lambda: {"available": False, "articles": []})
    monkeypatch.setattr(terminal_api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": []})
    monkeypatch.setattr(terminal_api, "_scan_progress_payload", lambda: {})
    monkeypatch.setattr(
        terminal_api,
        "_data_payload",
        lambda *_args: {
            "ready": False,
            "snapshot": {},
            "bhavcopy": {},
            "scan_saved": True,
            "scan_records": 0,
            "long_term_saved": False,
            "long_term_records": 0,
            "blockers": [],
        },
    )
    started = time.monotonic()
    payload = terminal_api.dashboard()
    assert time.monotonic() - started < 1.5
    assert payload["market"]["available"] is False
    assert "assembl" in payload["market"]["summary"].lower()


def test_market_payload_does_not_block_on_regime_fetch(monkeypatch):
    monkeypatch.setattr(terminal_api, "_warm_regime", lambda: None)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: None)
    payload = terminal_api._market_payload()
    assert payload["available"] is False
    assert "assembl" in payload["summary"].lower()
    assert "do not infer" in payload["trade_stance"].lower()


def test_data_payload_does_not_unpickle_bhavcopy_inline(monkeypatch):
    seen: list[bool] = []

    def fake_status(*, load_cache: bool = False):
        seen.append(load_cache)
        return {
            "ready": False,
            "cache_exists": True,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_files": 0,
            "minimum_sessions": 60,
        }

    monkeypatch.setattr(terminal_api, "_warm_bhavcopy_cache", lambda: None)
    monkeypatch.setattr("data.bhavcopy_runtime.status", fake_status)
    monkeypatch.setattr(terminal_api, "_snapshot_payload", lambda: {"ready": False})
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda history, load_cache=False, **_kwargs: {"current": True},
    )
    payload = terminal_api._data_payload(
        {"available": True, "records": [1, 2]},
        {"available": False, "records": []},
        {"running": True},
        {"available": True},
        {"available": True},
    )
    assert seen == [False]
    assert payload["scan_records"] == 2
    assert payload["ready"] is True
    assert payload["history_current"] is True
    assert payload["ready"] == payload["history_current"]
    blockers = " ".join(payload["blockers"]).lower()
    assert "not loaded the bhavcopy store" in blockers
    assert "history is not ready" not in blockers


def test_peek_cached_regime_is_missing_until_computed():
    from core import regime_engine

    regime_engine._CACHE.clear()
    assert regime_engine.peek_cached_regime() is None
    assert terminal_api._slim_ranked_records(
        {"universe_size": 3, "records": [{"symbol": "A", "score": 1}, {"symbol": "B", "score": 9}]}
    )["records"][0]["symbol"] == "B"


def test_home_and_forward_verifier_share_scan_artifact_override(tmp_path, monkeypatch):
    scan_path = tmp_path / "one-runtime-scan.json"
    scan_path.write_text(
        json.dumps({
            "schema_version": 2,
            "scanned_at": "2026-09-18T12:05:00+00:00",
            "approved_universe": 598,
            "scanned": 598,
            "qualified_rows": 0,
            "universe_size": 598,
            "records": [],
            "summary": {"qualified": 0, "with_any_setup": 0},
            "provenance": {
                "market_session_date": "2026-09-18",
                "data_current": True,
                "provenance_available": True,
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("QT_SCAN_PATH", str(scan_path))

    home_scan = terminal_api._scan_payload()
    from product import forward_soak
    verifier_scan = forward_soak._scan_payload()

    assert home_scan["scanned_at"] == "2026-09-18T12:05:00+00:00"
    assert home_scan["universe_size"] == 598
    assert home_scan["records"] == []
    assert verifier_scan["path"] == str(scan_path)
    assert verifier_scan["payload"]["scanned"] == 598
    assert verifier_scan["payload"]["scanned_at"] == home_scan["scanned_at"]


def test_fno_payload_exposes_durable_paper_state_without_live_authority(monkeypatch, tmp_path):
    from product import fo_paper_store

    real_cls = fo_paper_store.FoPaperStore
    db = tmp_path / "fo-paper.sqlite3"

    class TempStore(real_cls):
        def __init__(self):
            super().__init__(db)

    with TempStore() as store:
        store.replace_positions([{
            "trade_id": "T1",
            "option_symbol": "TEST26OCTCE",
            "underlying": "TEST",
            "option_type": "CE",
            "entry_price": 50.0,
            "stop_price": 40.0,
            "target_price": 70.0,
            "lot_size": 50,
            "lots": 1,
            "quantity": 50,
            "opened_at": "2026-09-25T10:00:00+05:30",
            "max_holding_sessions": 2,
            "risk_amount": 500.0,
            "setup_score": 75.0,
            "option_score": 82.0,
            "context_key": "CTX",
            "bars_held": 0,
            "last_mark_session": "2026-09-25",
            "max_mark": 50.0,
            "min_mark": 50.0,
        }])

    monkeypatch.setattr(fo_paper_store, "FoPaperStore", TempStore)
    monkeypatch.setattr(terminal_api, "_fo_directional_payload", lambda: {
        "available": True,
        "status": "READY",
        "candidate_count": 1,
        "candidates": [],
        "paper_only": True,
        "live_execution_allowed": False,
    })
    monkeypatch.setattr(terminal_api, "_json_file", lambda *_a, **_k: {})

    payload = terminal_api._fno_payload()

    assert payload["paper"]["available"] is True
    assert payload["paper"]["status"]["open_positions"] == 1
    assert payload["paper"]["open_positions"][0]["option_symbol"] == "TEST26OCTCE"
    assert payload["paper"]["production_evidence_enabled"] is False
    assert payload["paper"]["paper_only"] is True
    assert payload["paper"]["live_execution_allowed"] is False


def _fo_context(
    *,
    direction: str = "LONG",
    oi: str = "LONG_BUILDUP",
    rvol: str = "RVOL_2_2.5",
    adx: str = "ADX_GE30",
    delta: str = "D_55_65",
    dte: str = "DTE_8_14",
    iv: str = "IV_NORMAL",
) -> str:
    from product.fo_evidence import FO_CONTEXT_SCHEMA_VERSION

    return "|".join((FO_CONTEXT_SCHEMA_VERSION, direction, oi, rvol, adx, delta, dte, iv))


def _fo_evidence_rows(*, context_key: str, n: int, production_eligible: bool = True):
    rows = []
    for i in range(n):
        win = i % 3 != 0
        row = {
            "settled": True,
            "evidence_lane": "FORWARD_PAPER",
            "context_key": context_key,
            "net_option_return_pct": 8.0 if win else -4.0,
            "production_evidence_eligible": production_eligible,
            "mfe_pct": 10.0 if win else 2.0,
            "mae_pct": -2.0 if win else -5.0,
            "settled_at": f"2026-09-{(i % 28) + 1:02d}T10:00:00",
        }
        if not production_eligible:
            row["cost_model_status"] = "UNCONFIGURED_GROSS_ONLY"
        rows.append(row)
    return rows


def test_fno_candidate_evidence_overlay_promotes_only_exact_fully_costed_context():
    directional = {
        "available": True,
        "candidates": [{
            "symbol": "TEST",
            "direction": "LONG",
            "selected_contract": {"symbol": "TESTCE", "context_key": _fo_context()},
        }],
    }
    outcomes = (
        _fo_evidence_rows(context_key=_fo_context(), n=40)
        + _fo_evidence_rows(context_key=_fo_context(delta="D_65_75"), n=40)
    )

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    evidence = payload["candidates"][0]["forward_evidence"]

    assert evidence["status"] == "EVIDENCE_READY"
    assert evidence["context_key"] == _fo_context()
    assert evidence["n"] == 40
    assert evidence["observed_n"] == 40
    assert evidence["probability_claim_available"] is True
    assert evidence["win_probability_pct"] is not None
    assert evidence["win_probability_wilson_lb_pct"] < evidence["win_probability_pct"]
    assert evidence["production_influence_allowed"] is True
    assert payload["candidate_evidence_policy"]["live_execution_allowed"] is False


def test_fno_candidate_evidence_overlay_holds_probability_for_uncosted_or_small_samples():
    directional = {
        "available": True,
        "candidates": [
            {
                "symbol": "UNCOSTED",
                "selected_contract": {"context_key": _fo_context(iv="IV_HIGH")},
            },
            {
                "symbol": "SMALL",
                "selected_contract": {"context_key": _fo_context(dte="DTE_15_30")},
            },
            {
                "symbol": "NOCTX",
                "selected_contract": {},
            },
        ],
    }
    outcomes = (
        _fo_evidence_rows(
            context_key=_fo_context(iv="IV_HIGH"),
            n=40,
            production_eligible=False,
        )
        + _fo_evidence_rows(context_key=_fo_context(dte="DTE_15_30"), n=12)
    )

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    by_symbol = {row["symbol"]: row["forward_evidence"] for row in payload["candidates"]}

    assert by_symbol["UNCOSTED"]["status"] == "COST_MODEL_REQUIRED"
    assert by_symbol["UNCOSTED"]["observed_n"] == 40
    assert by_symbol["UNCOSTED"]["n"] == 0
    assert by_symbol["UNCOSTED"]["probability_claim_available"] is False
    assert by_symbol["UNCOSTED"]["production_influence_allowed"] is False

    assert by_symbol["SMALL"]["status"] == "ACCUMULATING"
    assert by_symbol["SMALL"]["n"] == 12
    assert by_symbol["SMALL"]["win_probability_pct"] is None

    assert by_symbol["NOCTX"]["status"] == "NO_CONTEXT_KEY"
    assert by_symbol["NOCTX"]["probability_claim_available"] is False



def test_fno_candidate_evidence_overlay_exposes_broader_counts_as_research_only():
    exact = _fo_context()
    same_thesis = _fo_context(delta="D_65_75", dte="DTE_15_30", iv="IV_HIGH")
    same_direction_oi = _fo_context(rvol="RVOL_1.5_2", adx="ADX_20_25")
    same_direction = _fo_context(oi="SHORT_COVERING")

    directional = {
        "available": True,
        "candidates": [{
            "symbol": "TEST",
            "direction": "LONG",
            "selected_contract": {"symbol": "TESTCE", "context_key": exact},
        }],
    }
    outcomes = (
        _fo_evidence_rows(context_key=exact, n=12)
        + _fo_evidence_rows(context_key=same_thesis, n=8)
        + _fo_evidence_rows(context_key=same_direction_oi, n=6)
        + _fo_evidence_rows(context_key=same_direction, n=4)
    )

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    evidence = payload["candidates"][0]["forward_evidence"]
    coverage = evidence["coverage"]

    assert evidence["status"] == "ACCUMULATING"
    assert evidence["n"] == 12
    assert evidence["probability_claim_available"] is False
    assert evidence["production_influence_allowed"] is False

    assert coverage["valid_context"] is True
    assert coverage["exact_n"] == 12
    assert coverage["thesis_n"] == 20
    assert coverage["direction_oi_n"] == 26
    assert coverage["direction_n"] == 30
    assert coverage["remaining_to_exact_min_n"] == 18
    assert coverage["counts_only"] is True
    assert coverage["probability_claim_available"] is False
    assert coverage["production_influence_allowed"] is False
    assert payload["candidate_evidence_policy"]["probability_requires_current_context_version"] is True
    assert payload["candidate_evidence_policy"]["probability_requires_exact_context"] is True
    assert payload["candidate_evidence_policy"]["probability_requires_complete_observed_path"] is True
    assert payload["candidate_evidence_policy"]["broader_context_counts_research_only"] is True


def test_fno_candidate_evidence_overlay_rejects_legacy_context_version():
    legacy = "LONG|LONG_BUILDUP|RVOL_2_2.5|ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    directional = {
        "available": True,
        "candidates": [{
            "symbol": "LEGACY",
            "direction": "LONG",
            "selected_contract": {"symbol": "LEGACYCE", "context_key": legacy},
        }],
    }
    outcomes = _fo_evidence_rows(context_key=legacy, n=100)

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    evidence = payload["candidates"][0]["forward_evidence"]

    assert evidence["status"] == "CONTEXT_VERSION_REQUIRED"
    assert evidence["valid_context"] is False
    assert evidence["n"] == 100
    assert evidence["probability_claim_available"] is False
    assert evidence["production_influence_allowed"] is False
    assert evidence["coverage"]["valid_context"] is False



def test_fno_candidate_evidence_overlay_labels_path_observation_holdout():
    context = _fo_context()
    directional = {
        "available": True,
        "candidates": [{
            "symbol": "PATHHELD",
            "direction": "LONG",
            "selected_contract": {"symbol": "PATHCE", "context_key": context},
        }],
    }
    outcomes = _fo_evidence_rows(context_key=context, n=40)
    for row in outcomes:
        row["production_evidence_eligible"] = False
        row["cost_model_status"] = "CONFIGURED:TEST_COSTS"
        row["path_observation_complete"] = False
        row["entry_minute_status"] = "AMBIGUOUS_BOUNDARY_TOUCH"
        row["evidence_exclusion_reason"] = "ENTRY_MINUTE_AMBIGUOUS_BOUNDARY_TOUCH"

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    evidence = payload["candidates"][0]["forward_evidence"]

    assert evidence["status"] == "PATH_OBSERVATION_REQUIRED"
    assert evidence["observed_n"] == 40
    assert evidence["n"] == 0
    assert evidence["excluded_unpriced_costs"] == 0
    assert evidence["excluded_path_observation"] == 40
    assert evidence["probability_claim_available"] is False
    assert evidence["production_influence_allowed"] is False
    assert payload["candidate_evidence_policy"]["probability_requires_complete_observed_path"] is True



def test_fno_candidate_evidence_overlay_labels_combined_evidence_inputs():
    context = _fo_context()
    directional = {
        "available": True,
        "candidates": [{
            "symbol": "BOTHHELD",
            "direction": "LONG",
            "selected_contract": {"symbol": "BOTHCE", "context_key": context},
        }],
    }
    outcomes = _fo_evidence_rows(
        context_key=context,
        n=40,
        production_eligible=False,
    )
    for row in outcomes:
        row["path_observation_complete"] = False
        row["entry_minute_status"] = "UNAVAILABLE"

    payload = terminal_api._fo_forward_evidence_overlay(
        directional, outcomes, min_n=30,
    )
    evidence = payload["candidates"][0]["forward_evidence"]

    assert evidence["status"] == "EVIDENCE_INPUTS_REQUIRED"
    assert evidence["n"] == 0
    assert evidence["excluded_unpriced_costs"] == 40
    assert evidence["excluded_path_observation"] == 40
    assert evidence["probability_claim_available"] is False
    assert evidence["production_influence_allowed"] is False
