from __future__ import annotations

from scripts.verify_quantterm_stack import _fd_counts, _pick_symbol, _pick_symbols


def test_fd_counts_reads_canonical_health_resource_shape():
    health = {
        "resources": {
            "state": "OK",
            "api": {"fd_count": 41},
            "market_ops": {"fd_count": 19},
        }
    }

    assert _fd_counts(health) == (41, 19, "OK")


def test_fd_counts_fail_soft_when_os_cannot_report_descriptors():
    assert _fd_counts({"resources": {"state": "OK"}}) == (None, None, "OK")


def test_pick_symbol_prefers_explicit_symbol():
    dashboard = {"scan": {"records": [{"symbol": "TCS"}]}}
    assert _pick_symbol(dashboard, " reliance ") == "RELIANCE"


def test_pick_symbol_uses_saved_scan_then_long_term():
    dashboard = {
        "scan": {"records": []},
        "long_term": {"records": [{"symbol": "HDFCBANK"}]},
    }
    assert _pick_symbol(dashboard, "") == "HDFCBANK"


def test_pick_symbol_is_empty_when_no_saved_symbol_exists():
    assert _pick_symbol({"scan": {"records": []}, "long_term": {"records": []}}, "") == ""


def test_pick_symbols_returns_two_unique_names_for_compare_probe():
    dashboard = {
        "scan": {"records": [{"symbol": "TCS"}, {"symbol": "TCS"}, {"symbol": "INFY"}]},
        "long_term": {"records": [{"symbol": "HDFCBANK"}]},
    }
    assert _pick_symbols(dashboard) == ["TCS", "INFY"]


def test_pick_symbols_keeps_explicit_symbol_then_adds_saved_name():
    dashboard = {"scan": {"records": [{"symbol": "TCS"}, {"symbol": "INFY"}]}}
    assert _pick_symbols(dashboard, " reliance ") == ["RELIANCE", "TCS"]
