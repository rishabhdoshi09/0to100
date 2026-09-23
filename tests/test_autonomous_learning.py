"""Autonomous learning control is persisted, lane-separated, and live-locked."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from product.autonomous_learning import (
    EVIDENCE_FORWARD,
    EVIDENCE_REPLAY,
    MODE_AUTO,
    MODE_FORWARD_PAPER,
    MODE_HISTORICAL_REPLAY,
    MODE_PAUSED,
    dashboard,
    intended_evidence_lane,
    load_control,
    maybe_run_closed_market_replay,
    save_control,
    set_control,
)
from product.evidence_class import HISTORICAL_REPLAY, may_promote


def test_control_persists_and_survives_reload(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    saved = save_control({"enabled": True, "mode": MODE_HISTORICAL_REPLAY})
    assert saved["mode"] == MODE_HISTORICAL_REPLAY
    assert saved["live_locked"] is True
    loaded = load_control()
    assert loaded["mode"] == MODE_HISTORICAL_REPLAY
    assert loaded["enabled"] is True


def test_paused_control_has_no_evidence_lane(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    save_control({"enabled": True, "mode": MODE_PAUSED})
    assert intended_evidence_lane() == ""


def test_auto_uses_replay_when_cash_session_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    save_control({"enabled": True, "mode": MODE_AUTO})
    closed = datetime(2026, 9, 17, 22, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert intended_evidence_lane(now=closed) == EVIDENCE_REPLAY


def test_forward_paper_mode_stays_forward_even_when_closed(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    save_control({"enabled": True, "mode": MODE_FORWARD_PAPER})
    closed = datetime(2026, 9, 17, 22, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert intended_evidence_lane(now=closed) == EVIDENCE_FORWARD


def test_replay_is_never_promotion_evidence():
    assert may_promote(HISTORICAL_REPLAY) is False
    assert may_promote(EVIDENCE_REPLAY) is False


def test_dashboard_does_not_invent_counts(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    monkeypatch.setenv("QT_HISTORICAL_REPLAY_DIR", str(tmp_path / "replay"))
    save_control({"enabled": True, "mode": MODE_AUTO})
    view = dashboard()
    assert view["available"] is True
    assert view["live_locked"] is True
    assert view["counts"]["forward_evidence_count"] == 0
    assert view["counts"]["replay_evidence_count"] == 0
    assert "No persisted learning evidence yet." in view["missing"]


def test_closed_market_replay_never_opens_paper(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    save_control({"enabled": True, "mode": MODE_HISTORICAL_REPLAY})
    monkeypatch.setattr(
        "product.historical_replay.start_replay_async",
        lambda **_k: {"status": "STARTED", "opens_paper_trades": False},
    )
    monkeypatch.setattr(
        "product.historical_replay.load_latest",
        lambda: {"status": "IDLE"},
    )
    result = maybe_run_closed_market_replay(force=True)
    assert result.get("opens_paper_trades") is False
    assert result.get("evidence_class") == EVIDENCE_REPLAY
    assert result.get("not_promotion_evidence") is True


def test_learning_now_and_replay_are_safe_controls():
    from product.backend_control_plane import FORBIDDEN_CONTROLS, SAFE_CONTROLS

    assert "RUN_LEARNING_NOW" in SAFE_CONTROLS
    assert "RUN_HISTORICAL_REPLAY" in SAFE_CONTROLS
    assert "UNLOCK_LIVE_MONEY" in FORBIDDEN_CONTROLS
    assert "LIVE_BUY" in FORBIDDEN_CONTROLS


def test_dashboard_prefers_durable_historical_learning_completion(tmp_path, monkeypatch):
    path = tmp_path / "autonomous_learning.json"
    monkeypatch.setenv("QT_AUTONOMOUS_LEARNING", str(path))
    save_control({
        "enabled": True,
        "mode": MODE_AUTO,
        "last_cycle_at": "2026-09-18T06:57:54+00:00",
    })
    monkeypatch.setattr(
        "product.historical_paper_loop.load_state",
        lambda: {
            "phase": "IDLE",
            "last_learning_completed_at": "2026-09-23T16:12:23+00:00",
            "last_learning_batch_id": "hist_97395d4909d00f77",
        },
    )

    view = dashboard()

    assert view["last_learning_cycle"] == "2026-09-23T16:12:23+00:00"
    assert view["last_learning_batch_id"] == "hist_97395d4909d00f77"
