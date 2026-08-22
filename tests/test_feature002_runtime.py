"""FEATURE-002 runtime integration: production scan path, fail-open, IST identity.

All ledger writes stay in tmp_path. Implementation-test / tmp live_scan rows
are not the production primary ledger.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from research.feature002.acceptance import (
    HEALTHY_COLLECTING,
    NO_POST_ACTIVATION_SCAN,
    evaluate_first_real_scan,
    operational_state,
)
from research.feature002.constants import (
    FEATURE_SET_VERSION,
    FORWARD_START_TS_IST,
    event_id,
    protocol_hash,
)
from research.feature002.ledger import feature_snapshot, get_observation
from research.feature002.observe import set_enabled
from research.feature002.resolve import resolve_event


IST = ZoneInfo("Asia/Kolkata")


def _hist(n=280, start=80.0, step=0.4, last="2026-08-21"):
    end = pd.Timestamp(last)
    idx = pd.bdate_range(end=end, periods=n)
    close = pd.Series([start + i * step for i in range(n)], index=idx)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.8, "low": close - 0.6,
         "close": close, "volume": [200000] * n},
        index=idx,
    )


def _signal(symbol, verdict="BUY", score=82.0, signals=None):
    return SimpleNamespace(
        symbol=symbol,
        price=100.0,
        momentum_5d=3.0,
        volume_ratio=1.5,
        rsi=58.0,
        signals=signals or ["MOMENTUM"],
        reasons=["test"],
        score=score,
        verdict=verdict,
        entry=100.0,
        stop=95.0,
        target=110.0,
        chase_risk=False,
    )


def _redirect_feature002(monkeypatch, tmp_path):
    db = tmp_path / "shadow.db"
    hook = tmp_path / "hook_log.jsonl"
    product = tmp_path / "latest_momentum_scan.json"
    scan_store = tmp_path / "scan_store.json"
    monkeypatch.setenv("PYTEST_FEATURE002_DB", str(db))
    monkeypatch.setattr("research.feature002.ledger.DB_PATH", db)
    monkeypatch.setattr("research.feature002.observe.LEDGER_DIR", tmp_path)
    monkeypatch.setattr("research.feature002.observe.HOOK_LOG", hook)
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    monkeypatch.setattr("research.feature002.observe._regime", lambda: "TEST")
    monkeypatch.setattr(
        "research.feature002.observe._ist_now_iso",
        lambda: "2026-08-24T10:30:00+05:30",
    )
    monkeypatch.setattr(
        "research.feature002.observe._session_date",
        lambda cards: "2026-08-24",
    )
    monkeypatch.setattr("research.feature002.health.SCAN_STATE", scan_store)
    monkeypatch.setattr("research.feature002.health.PRODUCT_SCAN", product)
    monkeypatch.setattr("research.feature002.health.HOOK_LOG", hook)
    monkeypatch.setattr("research.feature002.health.DB_PATH", db)
    monkeypatch.setattr("research.feature002.health.LEDGER_DIR", tmp_path)
    monkeypatch.setattr("research.feature002.watchdog.HOOK_LOG", hook)
    monkeypatch.setattr("research.feature002.ledger.DB_PATH", db)
    return db, hook, product


def _money_view(records):
    from alerts.telegram_actions import build_setup_keyboard
    from product.scan_store import watchlist_rows
    from product.trade_plan import build_trade_plan
    from risk.position_sizer import size_position

    ready = [r for r in records if r.get("status") == "Ready to trade"]
    tickets = []
    gtts = []
    for r in records:
        sized = size_position(float(r["entry"]), float(r["stop"]), capital=100000.0)
        plan = build_trade_plan(
            r["symbol"], r["entry"], r["stop"], r["target"],
            capital=100000.0, sizer=lambda e, s, capital, risk_pct: sized,
            portfolio_report=lambda: {"open_risk_pct": 0.0, "verdict": "OK", "warnings": []},
        )
        tickets.append({
            "symbol": r["symbol"],
            "verdict": r["verdict"],
            "score": r["score"],
            "signals": list(r["signals"]),
            "entry": r["entry"],
            "stop": r["stop"],
            "target": r["target"],
            "qty": sized["qty"],
            "plan_qty": plan.qty,
        })
        gtts.append({
            "symbol": r["symbol"],
            "qty": sized["qty"],
            "trigger_values": [round(float(r["stop"]), 1), round(float(r["target"]), 1)],
            "last_price": float(r["entry"]),
        })
    queue = [r["symbol"] for r in records if r["verdict"] in ("BUY", "STRONG BUY")][:20]
    return {
        "symbols": [r["symbol"] for r in records],
        "scores": [r["score"] for r in records],
        "signals": [list(r["signals"]) for r in records],
        "verdicts": [r["verdict"] for r in records],
        "ready": [r["symbol"] for r in ready],
        "watchlist": [r["symbol"] for r in watchlist_rows({"records": records})],
        "tickets": tickets,
        "gtts": gtts,
        "telegram": build_setup_keyboard(ready or records),
        "autopilot_queue": queue,
    }


def _run_runtime_scan(monkeypatch, tmp_path, *, enabled: bool, explode=False):
    from scan import market_scan_service as MSS
    import product.scan_store as store

    db, hook, product = _redirect_feature002(monkeypatch, tmp_path)
    set_enabled(enabled)
    if explode:
        def boom(*a, **k):
            raise RuntimeError("shadow exploded")
        monkeypatch.setattr("research.feature002.observe.build_shadow_records", boom)

    saved = {}
    real_save = store.save_scan

    def fake_save(payload, path=product):
        saved["payload"] = copy.deepcopy(dict(payload))
        return real_save(payload, product)

    class FakeScanner:
        def scan(self, symbols, progress=None, **k):
            return [
                _signal("AAA", "BUY", 82.0, ["MOMENTUM"]),
                _signal("BBB", "WATCH", 55.0, ["BREAKOUT_52W"]),
            ]

    monkeypatch.setattr(store, "save_scan", fake_save)
    report = MSS.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Aaa", "BBB": "Bbb"},
        prefetch_fn=lambda symbols, progress=None: len(symbols),
        scanner=FakeScanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    set_enabled(True)
    return report, saved.get("payload"), db, hook, product


def test_runtime_scan_off_vs_on_money_path_identical(monkeypatch, tmp_path):
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir(); on_dir.mkdir()
    report_off, payload_off, _, _, _ = _run_runtime_scan(monkeypatch, off_dir, enabled=False)
    report_on, payload_on, db, _, _ = _run_runtime_scan(monkeypatch, on_dir, enabled=True)

    assert report_off.status == report_on.status
    rec_off = payload_off["records"]
    rec_on = payload_on["records"]
    assert rec_off == rec_on
    assert _money_view(rec_off) == _money_view(rec_on)
    assert get_observation(event_id("2026-08-24", "AAA"), path=db) is not None


def test_runtime_scan_fail_open(monkeypatch, tmp_path):
    report, payload, db, _, _ = _run_runtime_scan(
        monkeypatch, tmp_path, enabled=True, explode=True,
    )
    assert report.ok
    assert [r["symbol"] for r in payload["records"]] == ["AAA", "BBB"]
    assert payload["records"][0]["score"] == 82.0
    assert get_observation(event_id("2026-08-24", "AAA"), path=db) is None
    view = _money_view(payload["records"])
    assert view["tickets"][0]["entry"] == 100.0
    assert view["tickets"][0]["stop"] == 95.0
    assert view["tickets"][0]["target"] == 110.0
    assert view["gtts"][0]["trigger_values"] == [95.0, 110.0]


def test_monday_scan_uses_monday_ist_not_friday_hist(monkeypatch, tmp_path):
    from research.feature002.observe import build_shadow_records

    _redirect_feature002(monkeypatch, tmp_path)
    cards = [{
        "symbol": "AAA", "score": 80, "verdict": "BUY",
        "signals": ["Strong momentum"], "entry": 100, "stop": 95,
        "target": 110, "chase_risk": False,
    }]
    cset, rows = build_shadow_records(
        cards, session_date="2026-08-24",
        recorded_at="2026-08-24T10:30:00+05:30",
        scan_cycle_id="monday-live", source="implementation_test",
    )
    assert cset["session_date"] == "2026-08-24"
    assert rows[0]["event_id"] == event_id("2026-08-24", "AAA")
    assert rows[0]["event_id"] != event_id("2026-08-21", "AAA")
    assert rows[0]["feature_snapshot"]["hist_as_of"] == "2026-08-21"


def test_persistence_restart_and_idempotency(monkeypatch, tmp_path):
    from research.feature002.observe import observe_production_scan

    db, _, _ = _redirect_feature002(monkeypatch, tmp_path)
    cards = [{
        "symbol": "AAA", "score": 80, "verdict": "BUY",
        "signals": ["Strong momentum"], "entry": 100, "stop": 95,
        "target": 110, "chase_risk": False,
    }]
    set_enabled(True)
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    eid = event_id("2026-08-24", "AAA")
    first = get_observation(eid, path=db)
    assert first is not None
    snap = json.loads(first["feature_snapshot"])
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    second = get_observation(eid, path=db)
    assert json.loads(second["feature_snapshot"]) == snap
    assert second["production_score"] == first["production_score"]
    assert second["production_rank"] == first["production_rank"]


def test_resolver_leaves_frozen_fields(monkeypatch, tmp_path):
    from research.feature002.observe import observe_production_scan

    db, _, _ = _redirect_feature002(monkeypatch, tmp_path)
    cards = [{
        "symbol": "AAA", "score": 80, "verdict": "BUY",
        "signals": ["Strong momentum"], "entry": 100, "stop": 95,
        "target": 110, "chase_risk": False,
    }]
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    eid = event_id("2026-08-24", "AAA")
    before = get_observation(eid, path=db)
    snap_before = feature_snapshot(eid, path=db)
    resolve_event(eid, path=db)
    after = get_observation(eid, path=db)
    assert feature_snapshot(eid, path=db) == snap_before
    for key in (
        "recorded_at", "candidate_set_id", "production_score", "production_rank",
        "n_structure_passed", "rs_percentile", "rs_rank", "trend_rank",
        "combined_shadow_rank", "feature_set_version",
    ):
        assert after[key] == before[key]


def test_first_real_scan_evaluator_rejects_synthetic(monkeypatch, tmp_path):
    db, _, product = _redirect_feature002(monkeypatch, tmp_path)
    out = evaluate_first_real_scan(ledger_path=db)
    assert out["accepted"] is False
    assert out["operational"]["operational_state"] == NO_POST_ACTIVATION_SCAN


def test_first_real_scan_evaluator_accepts_genuine_tmp_live_scan(monkeypatch, tmp_path):
    """Evaluator path using a tmp ledger. Not the host production ledger."""
    from research.feature002.observe import observe_production_scan
    from product.scan_store import build_scan_payload, save_scan

    db, hook, product = _redirect_feature002(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "research.feature002.health._today_ist", lambda: "2026-08-24",
    )
    cards = [
        _signal("AAA", "BUY", 82.0, ["MOMENTUM"]),
        _signal("BBB", "WATCH", 55.0, ["BREAKOUT_52W"]),
    ]
    payload = build_scan_payload({"AAA": "Aaa", "BBB": "Bbb"}, cards)
    payload["scanned_at"] = "2026-08-24T05:00:00+00:00"
    save_scan(payload, product)
    observe_production_scan(
        payload["records"], source="live_scan", background=False, path=db,
    )
    hook.write_text(
        json.dumps({
            "kind": "hook_received", "source": "live_scan",
            "ts": "2026-08-24T10:30:00+05:30", "n_cards": 2,
        }) + "\n",
        encoding="utf-8",
    )
    out = evaluate_first_real_scan(ledger_path=db)
    assert out["checks"]["production_scan_occurred"] is True
    assert out["checks"]["source_live_scan"] is True
    assert out["checks"]["candidate_set_persisted"] is True
    assert out["checks"]["recorded_after_activation"] is True
    assert out["checks"]["unresolved_null_not_zero"] is True
    assert out["operational"]["operational_state"] != NO_POST_ACTIVATION_SCAN
    assert out["accepted"] is True


def test_operational_states_empty_and_collecting(monkeypatch, tmp_path):
    db, _, product = _redirect_feature002(monkeypatch, tmp_path)
    empty = operational_state(ledger_path=db)
    assert empty["operational_state"] == NO_POST_ACTIVATION_SCAN
    assert "INSUFFICIENT NEW DATA" in empty["research_maturity"]

    from research.feature002.observe import observe_production_scan
    from product.scan_store import build_scan_payload, save_scan
    payload = build_scan_payload({"AAA": "Aaa"}, [_signal("AAA")])
    payload["scanned_at"] = "2026-08-24T05:00:00+00:00"
    save_scan(payload, product)
    observe_production_scan(
        payload["records"], source="live_scan", background=False, path=db,
    )
    collecting = operational_state(ledger_path=db)
    assert collecting["operational_state"] == HEALTHY_COLLECTING
    assert collecting["combined"].startswith("HEALTHY_COLLECTING")


def test_sepa_runtime_port_has_no_strategy_engine():
    import research.sepa as sepa
    assert hasattr(sepa, "DEFAULT_CONFIG")
    assert not hasattr(sepa, "evaluate_sepa_eligibility")
    text = (sepa.__file__ and open(sepa.__file__).read()) or ""
    assert "evaluate_sepa_eligibility" not in text
    assert "from research.sepa.engine" not in text


def test_feature002_not_imported_by_money_modules():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "scan/unified_scanner.py",
        "execution/trade_executor.py",
        "execution/autopilot.py",
        "risk/position_sizer.py",
        "alerts/telegram_actions.py",
        "product/trade_plan.py",
    ):
        assert "feature002" not in (root / rel).read_text()
