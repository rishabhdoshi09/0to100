from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_stack.py"
_SPEC = importlib.util.spec_from_file_location("quantterm_local_stack", _PATH)
LS = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(LS)


def test_autonomy_pid_reads_runtime_then_status(tmp_path: Path, monkeypatch):
    root = tmp_path
    auto = root / "logs" / "autonomy"
    auto.mkdir(parents=True)
    (auto / "status.json").write_text(json.dumps({"scheduler_owner_pid": 111}), encoding="utf-8")
    assert LS.autonomy_pid(root) == 111
    (auto / "runtime.json").write_text(json.dumps({"scheduler_owner_pid": 222}), encoding="utf-8")
    assert LS.autonomy_pid(root) == 222


def test_market_ops_pid_reads_runtime(tmp_path: Path):
    ops = tmp_path / "logs" / "market_ops"
    ops.mkdir(parents=True)
    (ops / "runtime.json").write_text(json.dumps({"worker_pid": 333}), encoding="utf-8")
    assert LS.market_ops_pid(tmp_path) == 333


def test_pids_on_port_uses_lsof_when_proc_is_empty(monkeypatch):
    monkeypatch.setattr(LS, "_pids_on_port_proc", lambda port: [])
    monkeypatch.setattr(LS, "_pids_on_port_lsof", lambda port: [444] if port == 8765 else [])
    assert LS.pids_on_port(8765) == [444]
    assert LS.pids_on_port(5173) == []


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


def test_queue_scan_now_posts_run_scan_now(monkeypatch):
    seen: list[object] = []

    def fake_urlopen(req, timeout=0):
        seen.append((req.full_url, req.get_method(), timeout))
        return _FakeResponse({"accepted": True, "control": "RUN_SCAN_NOW", "created": True})

    monkeypatch.setattr(LS.urllib.request, "urlopen", fake_urlopen)
    payload = LS.queue_scan_now()
    assert payload["accepted"] is True
    assert payload["control"] == "RUN_SCAN_NOW"
    assert seen[0][0] == "http://127.0.0.1:8765/api/controls/RUN_SCAN_NOW"
    assert seen[0][1] == "POST"


def test_scan_cli_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        LS,
        "queue_desk_jobs",
        lambda **kwargs: {
            "accepted": True,
            "queued": 3,
            "jobs": [{"accepted": True, "control": "RUN_SCAN_NOW"}],
        },
    )
    assert LS.main(["scan", "--origin", "http://127.0.0.1:8765"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["accepted"] is True
    assert printed["queued"] == 3


def test_queue_desk_jobs_posts_scan_news_and_funds(monkeypatch):
    seen: list[str] = []

    def fake_control(control, **kwargs):
        seen.append(control)
        return {"accepted": True, "control": control}

    monkeypatch.setattr(LS, "queue_control", fake_control)
    payload = LS.queue_desk_jobs()
    assert payload["accepted"] is True
    assert seen == ["RUN_SCAN_NOW", "REFRESH_NEWS_NOW", "REFRESH_LONG_TERM_NOW"]


def test_scan_cli_returns_error_when_api_is_down(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(LS, "queue_desk_jobs", boom)
    assert LS.main(["scan"]) == 1
    err = capsys.readouterr().err
    assert "not queued yet" in err
    assert "connection refused" in err
