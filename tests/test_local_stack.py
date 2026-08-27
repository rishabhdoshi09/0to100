"""Local stack restart helpers — resolve PIDs, never pkill -f."""
from __future__ import annotations

import importlib.util
import json
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("local_stack", ROOT / "scripts" / "local_stack.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pids_on_port_finds_this_process_listener():
    ls = _mod()
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert os.getpid() in ls.pids_on_port(port)
    finally:
        sock.close()


def test_autonomy_pid_reads_status_json(tmp_path):
    ls = _mod()
    (tmp_path / "status.json").write_text(json.dumps({"scheduler_owner_pid": 4242}), encoding="utf-8")
    assert ls.autonomy_pid(tmp_path) == 4242


def test_stop_pid_refuses_self_and_pid1():
    ls = _mod()
    assert ls.stop_pid(1) is False
    assert ls.stop_pid(os.getpid()) is False
