from __future__ import annotations

import json
from datetime import datetime, timezone

from product.scan_store import scan_age_hours, scan_artifact_is_fresh


def test_future_scan_timestamp_has_unknown_age():
    payload = {
        "schema_version": 2,
        "scanned_at": "2026-09-13T12:05:00+00:00",
        "records": [],
        "provenance": {},
    }
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)

    assert scan_age_hours(payload, now=now) is None


def test_future_scan_timestamp_can_never_pass_freshness(tmp_path):
    payload = {
        "schema_version": 2,
        "scanned_at": "2026-09-13T12:05:00+00:00",
        "records": [],
        "provenance": {},
    }
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)

    assert scan_artifact_is_fresh(path, max_age_s=3600, now=now) is False
