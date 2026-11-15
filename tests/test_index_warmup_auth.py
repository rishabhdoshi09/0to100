"""Index warm-up without the day's Zerodha session waits for login instead of
failing (owner's Mac 2026-11-15: PERMANENT_FAILED -> Home "Automation: Problem")."""
from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy import auth as AUTH


def _ctx(indices: int, *, valid: bool):
    status = AUTH.SESSION_VALID if valid else AUTH.SESSION_EXPIRED
    deps = SimpleNamespace(
        warm_indices=lambda: {"indices": indices},
        auth_health=lambda: AUTH.AuthHealth(status, "test"),
    )
    return SimpleNamespace(deps=deps)


def test_logged_out_kite_index_build_blocks_on_auth(monkeypatch):
    monkeypatch.setattr(JOBS, "_index_store_needs_broker", lambda: True)
    monkeypatch.setattr("data.nse_industry.refresh_industry_map", lambda **k: {})
    res = JOBS.run_index_warmup(_ctx(0, valid=False))
    assert res.status == JS.BLOCKED and res.blocked_on == JOBS.DEP_AUTH


def test_a_real_data_gap_with_a_valid_session_still_retries(monkeypatch):
    monkeypatch.setattr(JOBS, "_index_store_needs_broker", lambda: True)
    monkeypatch.setattr("data.nse_industry.refresh_industry_map", lambda **k: {})
    res = JOBS.run_index_warmup(_ctx(0, valid=True))
    assert res.status == JS.RETRYABLE_FAILED and res.error_code == "INDEX_DATA_UNAVAILABLE"


def test_without_kite_credentials_the_gap_is_a_data_failure(monkeypatch):
    monkeypatch.setattr(JOBS, "_index_store_needs_broker", lambda: False)
    monkeypatch.setattr("data.nse_industry.refresh_industry_map", lambda **k: {})
    assert JOBS.run_index_warmup(_ctx(0, valid=False)).status == JS.RETRYABLE_FAILED
    assert JOBS.run_index_warmup(_ctx(11, valid=False)).status == JS.SUCCEEDED


def test_the_block_is_a_broker_wait_not_a_system_problem():
    from product.operator_health import is_broker_scoped_job
    assert is_broker_scoped_job({"job_type": "index_warmup", "status": "BLOCKED", "blocked_on": JOBS.DEP_AUTH})
