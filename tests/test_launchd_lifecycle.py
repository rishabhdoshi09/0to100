from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from product import host_install_existing_v2 as install_v2
from product import launchd_control as lc


def cp(rc: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["launchctl"], rc, out, err)


def test_stop_is_idempotent_only_when_absence_is_explicit(monkeypatch):
    monkeypatch.setattr(
        lc, "_run",
        lambda *args, **kwargs: cp(113, err="Could not find service com.quantterm.desk"),
    )
    result = lc.stop_verified(timeout_s=0)
    assert result.returncode == 113


def test_unexpected_print_error_fails_closed(monkeypatch):
    monkeypatch.setattr(
        lc, "_run",
        lambda *args, **kwargs: cp(5, err="Operation not permitted"),
    )
    with pytest.raises(lc.LaunchdControlError, match="cannot establish launchd state"):
        lc.stop_verified(timeout_s=0)


def test_stop_proves_target_absent_after_bootout(monkeypatch):
    responses = iter([
        cp(0, out="loaded"),                      # initial print
        cp(0, out="bootout accepted"),            # bootout
        cp(113, err="Could not find service"),     # verification print
    ])
    monkeypatch.setattr(lc, "_run", lambda *args, **kwargs: next(responses))
    result = lc.stop_verified(timeout_s=0)
    assert result.returncode == 0


def test_stop_fails_when_bootout_does_not_unload_target(monkeypatch):
    responses = iter([
        cp(0, out="loaded"),
        cp(0, out="bootout accepted"),
        cp(0, out="still loaded"),
    ])
    monkeypatch.setattr(lc, "_run", lambda *args, **kwargs: next(responses))
    with pytest.raises(lc.LaunchdControlError, match="did not unload"):
        lc.stop_verified(timeout_s=0)


def test_restart_never_bootstraps_when_verified_stop_fails(monkeypatch):
    called = {"bootstrap": False}

    def fail_stop(**kwargs):
        raise lc.LaunchdControlError("still loaded")

    def bootstrap(**kwargs):
        called["bootstrap"] = True
        return cp(0)

    monkeypatch.setattr(lc, "stop_verified", fail_stop)
    monkeypatch.setattr(lc, "_bootstrap_verified", bootstrap)
    with pytest.raises(lc.LaunchdControlError, match="still loaded"):
        lc.restart_verified(plist=Path("/tmp/nope"))
    assert called["bootstrap"] is False


def test_strict_plist_makes_python_the_launchd_owned_process(tmp_path):
    rendered = install_v2._render_launchd_plist_direct_host(
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        python="/usr/bin/python3",
        build_sha="deadbeef",
        env_file="",
        bootstrap_log_dir=tmp_path / "logs",
    )
    assert "<string>product.host_launchd_entrypoint</string>" in rendered
    assert "<string>product.host_entrypoint</string>" not in rendered
    assert "<string>/usr/bin/caffeinate</string><string>-i</string>" not in rendered


def test_bootstrap_does_not_force_kill_fresh_runatload_host(tmp_path, monkeypatch):
    plist = tmp_path / "com.quantterm.desk.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["launchctl", "print"]:
            return cp(0, out="loaded")
        return cp(0)

    monkeypatch.setattr(lc, "_run", fake_run)
    lc._bootstrap_verified(plist=plist, timeout_s=0)

    kickstarts = [call for call in calls if call[:2] == ["launchctl", "kickstart"]]
    assert kickstarts
    assert ["launchctl", "kickstart", lc.target()] in kickstarts
    assert not any("-k" in call for call in kickstarts)


def test_start_on_loaded_service_never_uses_force_kill(monkeypatch):
    calls = []
    responses = iter([
        cp(0, out="loaded"),   # initial query_loaded
        cp(0, out="kick"),     # kickstart
        cp(0, out="loaded"),   # verification query_loaded
    ])

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return next(responses)

    monkeypatch.setattr(lc, "_run", fake_run)
    lc.start_verified(label=lc.DEFAULT_LABEL, plist=Path("/tmp/unused.plist"))

    kickstarts = [call for call in calls if call[:2] == ["launchctl", "kickstart"]]
    assert kickstarts == [["launchctl", "kickstart", lc.target(lc.DEFAULT_LABEL)]]
    assert not any("-k" in call for call in kickstarts)
