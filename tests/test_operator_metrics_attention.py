from __future__ import annotations

from product import operator_metrics as OM


def _row(kind: str, requested_by: str, status: str, at: float = 100.0) -> dict:
    return {
        "source": "test",
        "kind": kind,
        "requested_by": requested_by,
        "at": at,
        "status": status,
        "class": OM._classify(requested_by, kind),
    }


def test_missing_broker_is_optional_capability_not_human_alarm(monkeypatch):
    monkeypatch.setattr(OM, "_ops_rows", lambda: [_row("MARKET_SCAN", "autonomy", "SUCCEEDED")])
    monkeypatch.setattr(OM, "_run_started_s", lambda: 50.0)
    monkeypatch.setattr(
        OM.RDY,
        "broker_live",
        lambda: {
            "ready": False,
            "reason_code": "TOKEN_MISSING",
            "login_required": True,
        },
    )

    state = OM.build_operator_metrics(session="2026-09-03")

    assert state["operator_attention_required_now"] is False
    assert state["necessary_human"] == []
    assert state["kite_needed_for_paper_entry_only"] is True
    assert state["optional_capabilities"]
    assert "Zerodha" in state["optional_capabilities"][0]


def test_persisted_active_kite_login_is_real_human_attention(monkeypatch):
    monkeypatch.setattr(
        OM,
        "_ops_rows",
        lambda: [
            _row("MARKET_SCAN", "autonomy", "SUCCEEDED"),
            _row("KITE_LOGIN", "terminal", "PENDING", at=110.0),
        ],
    )
    monkeypatch.setattr(OM, "_run_started_s", lambda: 50.0)
    monkeypatch.setattr(
        OM.RDY,
        "broker_live",
        lambda: {
            "ready": False,
            "reason_code": "TOKEN_MISSING",
            "login_required": True,
        },
    )

    state = OM.build_operator_metrics(session="2026-09-03")

    assert state["operator_attention_required_now"] is True
    assert state["necessary_human"] == ["Zerodha authentication"]
    assert state["human_required_actions"] >= 1


def test_completed_kite_login_does_not_remain_attention(monkeypatch):
    monkeypatch.setattr(
        OM,
        "_ops_rows",
        lambda: [_row("KITE_LOGIN", "terminal", "SUCCEEDED", at=110.0)],
    )
    monkeypatch.setattr(OM, "_run_started_s", lambda: 50.0)
    monkeypatch.setattr(
        OM.RDY,
        "broker_live",
        lambda: {"ready": True, "reason_code": "READY", "login_required": False},
    )

    state = OM.build_operator_metrics(session="2026-09-03")

    assert state["operator_attention_required_now"] is False
    assert state["necessary_human"] == []
    assert state["optional_capabilities"] == []
