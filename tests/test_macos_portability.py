"""QT-PRA-077: host safety checks must hold on macOS, the real production OS.

Found by running the canonical suite on the owner's Mac (CI is Linux only):
- the installer/preflight "ephemeral runtime root" guard compared resolved
  paths against ``/tmp/``, but on macOS ``/tmp`` resolves to ``/private/tmp``
  and ``$TMPDIR`` is ``/private/var/folders/...`` (purged by dirhelper), so a
  temp directory could be adopted as the durable runtime;
- ``stack_supervision.pid_alive`` read ``/proc`` and reported every process,
  including the caller, as dead on macOS.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import product.host_install as HI
import product.host_preflight as HP
from product import stack_supervision as SS


def test_installer_and_preflight_share_one_ephemeral_definition():
    assert HI.EPHEMERAL_PREFIXES == HP.EPHEMERAL_PREFIXES
    for path in ("/tmp/qt", "/private/tmp/qt", "/var/tmp/qt", "/private/var/tmp/qt",
                 "/private/var/folders/ab/cd/T/qt", "/var/folders/ab/cd/T/qt", "/dev/shm/qt"):
        assert (path + "/").startswith(HI.EPHEMERAL_PREFIXES), path
    for path in ("/Volumes/QuantTermStorage/QuantTerm/runtime", "/home/q/.local/state/quantterm/runtime",
                 "/Users/apple/Library/Application Support/QuantTerm/runtime"):
        assert not (path + "/").startswith(HI.EPHEMERAL_PREFIXES), path


def test_installer_refuses_the_platform_temp_dir_as_durable_runtime():
    candidate = Path(tempfile.gettempdir()) / "qt-should-never-be-adopted"
    with pytest.raises(HI.HostInstallError, match="ephemeral"):
        HI.ensure_persistent_runtime_root(candidate)
    assert not candidate.exists()


def test_preflight_warns_on_the_platform_temp_dir(monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="qt-preflight-"))
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    check = HP.check_runtime_root()
    assert check.status == HP.WARN
    assert check.evidence["persistent"] is False


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


@pytest.mark.parametrize("procfs", [True, False])
def test_pid_alive_is_portable(monkeypatch, procfs):
    if procfs and not os.path.exists("/proc/self/stat"):
        pytest.skip("no procfs on this OS")
    if not procfs:
        real_exists = os.path.exists
        monkeypatch.setattr(SS.os.path, "exists", lambda p: False if p == "/proc/self/stat" else real_exists(p))
    assert SS.pid_alive(os.getpid())
    proc = _sleeper()
    try:
        assert SS.pid_alive(proc.pid)
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.time() + 5
        while SS.pid_alive(proc.pid) and time.time() < deadline:  # zombie until reaped
            time.sleep(0.05)
        assert not SS.pid_alive(proc.pid)
    finally:
        proc.kill()
        proc.wait(timeout=5)
    assert not SS.pid_alive(proc.pid)
