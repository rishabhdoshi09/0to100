from __future__ import annotations

import json

import research.intelligence.data.nse_calendar as CAL


def test_load_holidays_unions_bundled_and_runtime_amendments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    (data_dir / "nse_holidays.json").write_text(json.dumps(["2026-01-26"]))
    (logs_dir / "nse_holidays.json").write_text(json.dumps(["2026-01-15", "2026-01-26"]))
    monkeypatch.setattr(CAL, "logs_dir", lambda: logs_dir)

    assert CAL.load_holidays() == {"2026-01-15", "2026-01-26"}


def test_load_holidays_keeps_valid_source_when_other_source_is_malformed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()
    (data_dir / "nse_holidays.json").write_text("{not-json")
    (logs_dir / "nse_holidays.json").write_text(json.dumps(["2026-01-15"]))
    monkeypatch.setattr(CAL, "logs_dir", lambda: logs_dir)

    assert CAL.load_holidays() == {"2026-01-15"}
