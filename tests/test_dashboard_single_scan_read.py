"""Hot-path scan projection uses one atomic saved-scan read per request.

The cached scan may be large (hundreds of deeply nested records); the API
must not parse it again simply to supply coverage metadata.
"""
from __future__ import annotations


def test_dashboard_scan_projection_reads_atomic_artifact_once(monkeypatch):
    import api.runtime as runtime
    from product import scan_store

    seen = []
    raw = {
        "schema_version": 2,
        "scanned_at": "2026-10-10T16:32:00+00:00",
        "universe_size": 2000,
        "requested_universe": 2384,
        "coverage_state": "DEGRADED",
        "coverage_warning": "Missing histories",
        "coverage": {"checked": 2158, "data_unavailable": 226},
        "summary": {"with_any_setup": 1},
        "provenance": {"price_data_as_of": "2026-10-09"},
        "records": [{"symbol": "TCS", "score": 55, "signals": ["MOMENTUM"]}],
    }

    def load(path=None):
        seen.append(path)
        return raw

    monkeypatch.setattr(scan_store, "load_scan", load)
    result = runtime.core._scan_payload()
    assert len(seen) == 1, "Do not parse saved scan again in api.runtime"
    assert result["coverage"] == raw["coverage"]
    assert result["coverage_state"] == "DEGRADED"
    assert result["coverage_warning"] == "Missing histories"
    assert result["requested_universe"] == 2384
    assert result["records"] == raw["records"]
    assert result["provenance"]["price_data_as_of"] == "2026-10-09"


def test_scan_projection_isolated_from_input_mutations(monkeypatch):
    import terminal_api
    from product import scan_store

    raw = {
        "schema_version": 2,
        "records": [{"symbol": "INFY", "score": 66}],
        "summary": {},
        "provenance": {"price_data_as_of": "2026-10-09"},
        "coverage": {"checked": 2000},
    }
    monkeypatch.setattr(scan_store, "load_scan", lambda *args, **kwargs: raw)
    projected = terminal_api._scan_payload()
    projected["records"][0]["score"] = 0
    projected["coverage"]["checked"] = 0
    assert raw["records"][0]["score"] == 66
    assert raw["coverage"]["checked"] == 2000
