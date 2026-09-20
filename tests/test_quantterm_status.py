from __future__ import annotations

import scripts.quantterm_status as status


def test_http_probe_retries_one_transient_timeout(monkeypatch) -> None:
    calls = {"n": 0}

    class _Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False

    def fake_urlopen(_url, timeout):
        calls["n"] += 1
        assert timeout == 0.01
        if calls["n"] == 1:
            raise TimeoutError("busy")
        return _Response()

    monkeypatch.setattr(status.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(status.time, "sleep", lambda _seconds: None)

    assert status._http_ok("http://127.0.0.1:8765/api/health", timeout=0.01, attempts=2) is True
    assert calls["n"] == 2


def test_runtime_status_requires_api_desk_and_fresh_worker(monkeypatch) -> None:
    monkeypatch.setattr(status, "_http_ok", lambda url, timeout=1.5, attempts=2: True)
    monkeypatch.setattr(status, "_pid_alive", lambda pid: pid == 42)
    monkeypatch.setattr(
        status,
        "_read_json",
        lambda path: {"worker_pid": 42, "heartbeat_epoch": 995.0},
    )

    observed = status.runtime_status(now=1000.0)

    assert observed["market_ops"]["healthy"] is True
    assert observed["market_ops"]["heartbeat_age_seconds"] == 5.0
    assert observed["ready"] is True


def test_runtime_status_fails_closed_on_stale_worker(monkeypatch) -> None:
    monkeypatch.setattr(status, "_http_ok", lambda url, timeout=1.5, attempts=2: True)
    monkeypatch.setattr(status, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        status,
        "_read_json",
        lambda path: {"worker_pid": 42, "heartbeat_epoch": 900.0},
    )

    observed = status.runtime_status(now=1000.0)

    assert observed["market_ops"]["pid_alive"] is True
    assert observed["market_ops"]["healthy"] is False
    assert observed["ready"] is False


def test_runtime_status_does_not_invent_missing_worker(monkeypatch) -> None:
    monkeypatch.setattr(status, "_http_ok", lambda url, timeout=1.5, attempts=2: False)
    monkeypatch.setattr(status, "_read_json", lambda path: {})
    monkeypatch.setattr(status, "_pid_alive", lambda pid: False)

    observed = status.runtime_status(now=1000.0)

    assert observed["market_ops"]["pid"] is None
    assert observed["market_ops"]["heartbeat_age_seconds"] is None
    assert observed["market_ops"]["healthy"] is False
    assert observed["ready"] is False


def test_runtime_status_fails_closed_on_malformed_numeric_evidence(monkeypatch) -> None:
    monkeypatch.setattr(status, "_http_ok", lambda url, timeout=1.5, attempts=2: True)
    monkeypatch.setattr(status, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        status,
        "_read_json",
        lambda path: {"worker_pid": "not-a-pid", "heartbeat_epoch": "not-an-epoch"},
    )

    observed = status.runtime_status(now=1000.0)

    assert observed["market_ops"]["pid"] is None
    assert observed["market_ops"]["heartbeat_age_seconds"] is None
    assert observed["market_ops"]["healthy"] is False
    assert observed["ready"] is False


def test_runtime_status_rejects_non_finite_heartbeat(monkeypatch) -> None:
    monkeypatch.setattr(status, "_http_ok", lambda url, timeout=1.5, attempts=2: True)
    monkeypatch.setattr(status, "_pid_alive", lambda pid: pid == 42)
    monkeypatch.setattr(
        status,
        "_read_json",
        lambda path: {"worker_pid": 42, "heartbeat_epoch": "nan"},
    )

    observed = status.runtime_status(now=1000.0)

    assert observed["market_ops"]["pid_alive"] is True
    assert observed["market_ops"]["heartbeat_age_seconds"] is None
    assert observed["market_ops"]["healthy"] is False
    assert observed["ready"] is False
