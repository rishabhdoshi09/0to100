from __future__ import annotations


def test_long_term_control_routes_to_long_term_operation():
    import terminal_api as core
    import terminal_product_api_parallel  # noqa: F401 - applies canonical routing patch

    assert core._OPERATION_CONTROLS["RUN_SCAN_NOW"] == "MARKET_SCAN"
    assert core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] == "LONG_TERM_SCAN"
    assert core._OPERATION_CONTROLS["REFRESH_LONG_TERM_NOW"] == "LONG_TERM_REFRESH"
