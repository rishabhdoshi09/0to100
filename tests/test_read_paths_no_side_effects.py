"""Reading the autonomy desk must not create or depend on the job stores.

On a runtime where the autonomy supervisor has not created its job store yet
(first start, autonomy paused), the first GET /api/autonomy-desk opened
logs/autonomy/jobs.db read-write, which created an EMPTY database. From then
on operator_metrics read "jobs" from it and raised "no such table: jobs":
every later autonomy-desk read was an HTTP 500 until the supervisor ran.
"""
from __future__ import annotations

import importlib


def _fresh_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    import product.operator_metrics as OM
    import product.autonomous_loop as AL
    OM = importlib.reload(OM)          # its store paths are import-time constants
    return OM, AL


def test_desk_read_creates_no_job_store(monkeypatch, tmp_path):
    OM, AL = _fresh_modules(monkeypatch, tmp_path)
    AL.project_events(24)
    OM.build_operator_metrics(session="2026-11-09")
    assert not (tmp_path / "logs" / "autonomy" / "jobs.db").exists()
    assert not (tmp_path / "logs" / "market_ops" / "jobs.db").exists()


def test_an_empty_job_store_left_by_an_older_build_is_read_as_no_jobs(monkeypatch, tmp_path):
    OM, AL = _fresh_modules(monkeypatch, tmp_path)
    for lane in ("autonomy", "market_ops"):
        (tmp_path / "logs" / lane).mkdir(parents=True)
        (tmp_path / "logs" / lane / "jobs.db").write_bytes(b"")
    metrics = OM.build_operator_metrics(session="2026-11-09")
    assert isinstance(metrics, dict)
    assert isinstance(AL.project_events(24), list)
