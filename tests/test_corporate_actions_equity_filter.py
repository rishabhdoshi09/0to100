from __future__ import annotations

import json
from pathlib import Path

from data import corporate_actions_equity_filter as EQF


def test_preference_share_bonus_is_not_an_equity_share_count_action():
    assert EQF.is_non_equity_distribution(
        "Bonus - 4 (Four) 9% Cumulative Non-Convertible Redeemable Preference Shares for every 1 Equity Share"
    )
    assert EQF.is_non_equity_distribution(
        "Bonus Preference Shares - Series II 3:1"
    )
    assert not EQF.is_non_equity_distribution("Bonus Equity Shares 1:1")
    assert not EQF.is_non_equity_distribution(
        "Sub-division of each equity share from face value Rs 10 to Rs 2"
    )


def test_sanitizer_removes_misclassified_preference_adjustments_and_forces_refresh(tmp_path: Path):
    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    events.write_text(json.dumps([
        {
            "symbol": "SIYSIL",
            "ex_date": "2026-08-21",
            "type": "bonus",
            "factor": 5.0,
            "subject": "Bonus 4:1 Cumulative Non-Convertible Redeemable Preference Shares",
        },
        {
            "symbol": "RELIANCE",
            "ex_date": "2024-10-28",
            "type": "bonus",
            "factor": 2.0,
            "subject": "Bonus Equity Shares 1:1",
        },
    ]), encoding="utf-8")
    coverage.write_text(json.dumps({
        "version": 1,
        "last_refresh_at": "2026-08-28T10:00:00+05:30",
        "conflicts": [{"symbol": "SIYSIL", "factors": [4.0, 5.0]}],
    }), encoding="utf-8")

    result = EQF.sanitize_persisted_adjustments(events_path=events, coverage_path=coverage)
    cleaned = json.loads(events.read_text(encoding="utf-8"))
    status = json.loads(coverage.read_text(encoding="utf-8"))

    assert result == {"removed": 1, "kept": 1}
    assert [row["symbol"] for row in cleaned] == ["RELIANCE"]
    assert status["last_refresh_at"] == ""
    assert status["equity_security_filter_version"] == 1
    assert status["security_filter_removed_adjustments"] == 1


def test_autonomy_bootstrap_installs_security_filter_before_bse_adapter():
    source = (Path(__file__).resolve().parents[1] / "research" / "autonomy" / "__init__.py").read_text(
        encoding="utf-8"
    )
    security = source.index("install_ca_equity_filter()")
    bse = source.index("install_bse_ca_adapter()")
    parallel = source.index("install_parallel_runtime()")
    assert security < bse < parallel
