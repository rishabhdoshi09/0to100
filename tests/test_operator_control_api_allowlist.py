from __future__ import annotations

import terminal_api


def test_final_operator_surface_is_backed_by_real_allowlisted_controls() -> None:
    expected = {
        "RUN_SCAN_NOW": "MARKET_SCAN",
        "REFRESH_DATA_NOW": "DATA_PREPARE",
        "REFRESH_NEWS_NOW": "NEWS_REFRESH",
        "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
        "REFRESH_FNO_NOW": "FNO_REFRESH",
        "REFRESH_MARKET_REPORT_NOW": "MARKET_REPORT",
    }
    for control, operation_kind in expected.items():
        assert terminal_api._OPERATION_CONTROLS.get(control) == operation_kind
        assert control in terminal_api._ALLOWED_CONTROLS

    assert "RUN_CYCLE_NOW" in terminal_api._AUTONOMY_CONTROLS
    assert "RUN_CYCLE_NOW" in terminal_api._ALLOWED_CONTROLS


def test_terminal_api_exposes_no_live_money_mutation_control() -> None:
    forbidden_fragments = ("LIVE", "BUY", "SELL", "UNLOCK", "BROKER")
    exposed = {str(name).upper() for name in terminal_api._ALLOWED_CONTROLS}
    assert not {
        control
        for control in exposed
        if any(fragment in control for fragment in forbidden_fragments)
    }
