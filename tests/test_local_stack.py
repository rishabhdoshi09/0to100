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
