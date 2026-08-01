from __future__ import annotations

import json

import terminal_product_api


def test_capability_registry_requires_certified_at_and_evidence(tmp_path, monkeypatch):
    path = tmp_path / "certifications.json"
    path.write_text(
        json.dumps(
            {
                "certifications": {
                    "durable_oms": {
                        "certified": True,
                        "certified_at": "2026-08-01T06:30:00+00:00",
                        "evidence": ["tests/test_oms.py"],
                    },
                    "broker_event_ingestion": {
                        "certified": True,
                        "certified_at": "",
                        "evidence": ["missing timestamp"],
                    },
                    "idempotent_submission": {
                        "certified": "true",
                        "certified_at": "2026-08-01T06:30:00+00:00",
                        "evidence": ["truthy strings must not pass"],
                    },
                    "partial_fill_recovery": {
                        "certified": True,
                        "certified_at": "2026-08-01T06:30:00+00:00",
                        "evidence": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(terminal_product_api, "INSTITUTIONAL_CERTIFICATIONS", path)

    assert terminal_product_api._institutional_capabilities() == {"durable_oms": True}


def test_institutional_endpoint_is_fail_closed_without_certifications(monkeypatch):
    monkeypatch.setattr(
        terminal_product_api,
        "_current_product_payloads",
        lambda: {
            "data": {
                "bhavcopy": {"ready": True, "sessions": 500, "minimum_sessions": 60},
                "snapshot": {"ready": True},
            },
            "market": {"available": True},
            "scan": {"available": True},
            "long_term": {},
            "operations": {"running": True},
            "news": {},
            "fno": {},
        },
    )
    monkeypatch.setattr(terminal_product_api.core, "_paper_payload", lambda: {"available": True})
    monkeypatch.setattr(terminal_product_api.core, "_autonomy_payload", lambda: {"available": True})
    monkeypatch.setattr(terminal_product_api, "_institutional_capabilities", lambda: {})

    report = terminal_product_api.institutional_readiness()

    assert report["system_state"] == "PAPER_ONLY"
    assert report["deployment"]["limited_live"]["allowed"] is False
    assert report["deployment"]["live"]["allowed"] is False
    assert "execution" in report["hard_blockers"]
    assert "risk" in report["hard_blockers"]
    assert "reconciliation" in report["hard_blockers"]
