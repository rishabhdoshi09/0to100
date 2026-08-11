"""Corporate-action honesty: detect gaps, never invent factors, verify adjust-on-read."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _bonus_frame():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    close = [100.0, 102.0, 101.0, 51.0, 52.0, 51.5]  # 1:1 bonus gap
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000.0] * 6,
        },
        index=idx,
    )


def test_dividend_rows_are_rejected(tmp_path, monkeypatch):
    from data import corporate_actions as CA

    dest = tmp_path / "ca.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))
    status = CA.write_events(
        [
            {"symbol": "INFY", "ex_date": "2024-01-04", "factor": 2.0, "type": "bonus"},
            {"symbol": "INFY", "ex_date": "2024-06-01", "factor": 1.5, "type": "dividend"},
        ],
        path=dest,
    )
    assert status["events"] == 1
    assert "INFY" in CA.load_events(dest)
    assert all(e["type"] != "dividend" for e in CA.load_events(dest)["INFY"])


def test_export_gap_todo_never_invents_factor(tmp_path, monkeypatch):
    from data import bhavcopy_store as BS
    from data import corporate_actions as CA

    dest = tmp_path / "ca.json"
    todo = tmp_path / "todo.csv"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))
    monkeypatch.setenv("QT_CA_TODO_FILE", str(todo))
    monkeypatch.setattr(BS, "_store", {"TESTCO": _bonus_frame()}, raising=False)

    report = CA.export_gap_todo(sample=10, path=todo)
    assert report["written"] is True
    assert report["gaps"] >= 1
    assert report["never_invents"] is True
    text = todo.read_text(encoding="utf-8")
    assert "TESTCO" in text
    # factor column blank for operator
    lines = [ln for ln in text.splitlines() if ln.startswith("TESTCO")]
    assert lines
    cols = lines[0].split(",")
    # symbol,ex_date,factor,type,...
    assert cols[2] == ""


def test_ledger_status_surfaces_verify_cache(tmp_path, monkeypatch):
    import time

    from data import corporate_actions as CA

    dest = tmp_path / "ca.json"
    cache = tmp_path / "verify.json"
    monkeypatch.setenv("QT_CA_EVENTS_FILE", str(dest))
    monkeypatch.setenv("QT_CA_VERIFY_CACHE_FILE", str(cache))
    CA.write_events(
        [{"symbol": "TCS", "ex_date": "2023-06-01", "factor": 5.0, "type": "split"}],
        path=dest,
    )
    cache.write_text(
        json.dumps({
            "passed": True,
            "gap_rate": 0.0,
            "checked": 50,
            "still_flagged": 0,
            "note": "PASS",
            "checked_at_unix": time.time(),
            "flagged": [],
        }),
        encoding="utf-8",
    )
    status = CA.ledger_status(path=dest, verify=False)
    assert status["research_grade"] is True
    assert status["adjustment_verified"] is True
    assert status["gap_rate"] == 0.0


def test_checklist_ca_partial_until_verified():
    from product.retail_research_checklist import build_retail_research_checklist

    partial = build_retail_research_checklist(
        ca={"research_grade": True, "events": 3, "symbols": 2, "adjustment_verified": False},
    )
    ca_item = next(i for i in partial["items"] if i["key"] == "corporate_actions")
    assert ca_item["status"] == "PARTIAL"

    ready = build_retail_research_checklist(
        ca={
            "research_grade": True,
            "events": 3,
            "symbols": 2,
            "adjustment_verified": True,
            "gap_rate": 0.0,
        },
    )
    ca_ready = next(i for i in ready["items"] if i["key"] == "corporate_actions")
    assert ca_ready["status"] == "READY"


def test_corporate_actions_api(monkeypatch):
    from fastapi.testclient import TestClient

    import terminal_product_api as api
    from data import corporate_actions as CA

    monkeypatch.setattr(
        CA,
        "ledger_status",
        lambda verify=False, sample=80: {
            "available": True,
            "symbols": 1,
            "events": 2,
            "research_grade": True,
            "adjustment_verified": False,
            "never_invents": True,
        },
    )
    monkeypatch.setattr(
        CA,
        "export_gap_todo",
        lambda sample=400: {"written": True, "gaps": 3, "path": "logs/ca_events.todo.csv", "never_invents": True},
    )
    monkeypatch.setattr(
        CA,
        "refresh_adjustment_verify",
        lambda sample=80: {"passed": False, "gap_rate": 0.1, "note": "FAIL"},
    )

    client = TestClient(api.app)
    r = client.get("/api/corporate-actions")
    assert r.status_code == 200
    assert r.json()["never_invents"] is True

    g = client.post("/api/corporate-actions/from-gaps?sample=50")
    assert g.status_code == 200
    assert g.json()["gaps"] == 3

    v = client.post("/api/corporate-actions/verify?sample=40")
    assert v.status_code == 200
    assert "verify" in v.json()
