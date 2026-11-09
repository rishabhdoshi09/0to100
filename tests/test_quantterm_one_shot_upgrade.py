"""Offline tests: a recovery must not kill unrelated processes or upgrade unsafely."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import quantterm_one_shot_upgrade as qt


def test_classification_is_specific_to_installed_quantterm():
    assert qt.owned_command("/Library/Frameworks/Python/Python -u main.py autonomy")
    assert qt.owned_command("python -u -m operations.market_ops")
    assert qt.owned_command("python -m product.host_entrypoint")
    assert qt.owned_command("python -m product.host_supervisor")
    assert qt.owned_command("python -u -m uvicorn terminal_product_api_parallel:app --port 8765")
    assert qt.owned_command("python -m uvicorn report_api:app --port 8766")
    assert qt.owned_command("npm --prefix /Users/apple/0to100/frontend run dev -- --host 127.0.0.1")
    assert qt.owned_command("/usr/bin/node /tmp/repo/node_modules/.bin/vite --host 127.0.0.1")
    assert not qt.owned_command("python main.py login")
    assert not qt.owned_command("python -m pytest")
    assert not qt.owned_command("python unrelated.py autonomy")
    assert not qt.owned_command("python other_service.py")


def test_ps_parsing_excludes_unrelated_and_zombies():
    ps = """
 12    1 apple S  /Python -u main.py autonomy
 13    1 apple Z  /Python -u main.py autonomy
 14    1 apple S  /Python scripts/my_personal_task.py
 15    1 apple S  /Python -m product.host_supervisor
"""
    assert [x[0] for x in qt.parse_owned_processes(ps)] == [12, 15]


def test_unknown_cwd_is_blocked_without_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(qt, "launchd_unloaded", lambda: True)
    monkeypatch.setattr(qt, "remaining_processes", lambda: [(52, "apple", "python -u main.py autonomy")])
    monkeypatch.setattr(qt.pwd, "getpwuid", lambda _id: SimpleNamespace(pw_name="apple"))
    monkeypatch.setattr(qt.os, "getuid", lambda: 501)
    monkeypatch.setattr(qt, "process_cwd", lambda _id: Path("/totally/unrelated"))
    sent = []
    monkeypatch.setattr(qt.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    with pytest.raises(qt.Blocked, match="could not establish"):
        qt.stop_orphans(tmp_path)
    assert sent == []


def test_no_process_mutation_while_launchd_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(qt, "launchd_unloaded", lambda: False)
    inspected = []
    monkeypatch.setattr(qt, "remaining_processes", lambda: inspected.append(1))
    with pytest.raises(qt.Blocked, match="still loaded"):
        qt.stop_orphans(tmp_path)
    assert not inspected


def test_verified_orphan_receives_only_sigterm(monkeypatch, tmp_path):
    monkeypatch.setattr(qt, "launchd_unloaded", lambda: True)
    monkeypatch.setattr(qt, "remaining_processes", lambda: [(52, "apple", "python -u main.py autonomy")])
    monkeypatch.setattr(qt.pwd, "getpwuid", lambda _id: SimpleNamespace(pw_name="apple"))
    monkeypatch.setattr(qt.os, "getuid", lambda: 501)
    monkeypatch.setattr(qt, "process_cwd", lambda _id: tmp_path)
    monkeypatch.setattr(qt, "run", lambda _args, **_kw: SimpleNamespace(
        returncode=0, stdout="apple S python -u main.py autonomy"
    ))
    alive = [True]
    monkeypatch.setattr(qt, "pid_present", lambda _id: alive[0])
    sent = []
    def stop(pid, sig):
        sent.append((pid, sig))
        alive[0] = False
    monkeypatch.setattr(qt.os, "kill", stop)
    qt.stop_orphans(tmp_path)
    assert sent == [(52, qt.signal.SIGTERM)]


def test_unknown_pid_identity_blocks_before_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(qt, "launchd_unloaded", lambda: True)
    monkeypatch.setattr(qt, "remaining_processes", lambda: [(52, "apple", "python -u main.py autonomy")])
    monkeypatch.setattr(qt.pwd, "getpwuid", lambda _id: SimpleNamespace(pw_name="apple"))
    monkeypatch.setattr(qt.os, "getuid", lambda: 501)
    monkeypatch.setattr(qt, "process_cwd", lambda _id: tmp_path)
    monkeypatch.setattr(qt, "run", lambda _args, **_kw: SimpleNamespace(
        returncode=0, stdout="apple S python unrelated.py"
    ))
    monkeypatch.setattr(qt, "pid_present", lambda _id: True)
    signaled = []
    monkeypatch.setattr(qt.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    with pytest.raises(qt.Blocked, match="changed identity"):
        qt.stop_orphans(tmp_path)
    assert not signaled


def test_up_to_date_checkout_does_not_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(qt, "stop_orphans", lambda *_: pytest.fail("must not stop a healthy host"))
    qt.upgrade(tmp_path, tmp_path, qt.BRANCH, "same-sha", "same-sha")
