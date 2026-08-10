"""Cross-platform process file lock + stack launcher presence."""
from __future__ import annotations

from pathlib import Path

from utils.process_lock import ProcessFileLock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_process_file_lock_exclusive(tmp_path):
    path = tmp_path / "owner.lock"
    a = ProcessFileLock(path)
    b = ProcessFileLock(path)
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_windows_stack_wrappers_exist():
    for name in (
        "quantterm_stack.py",
        "setup_windows.ps1",
        "run_quantterm.ps1",
        "run_quantterm_low_power.ps1",
        "run_quantterm_complete.ps1",
        "stop_quantterm.ps1",
        "run_quantterm.bat",
        "run_quantterm_low_power.bat",
        "stop_quantterm.bat",
        "run_quantterm_low_power.sh",
        "_windows_common.ps1",
    ):
        assert (SCRIPTS / name).is_file(), name


def test_quantterm_stack_cli_help():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "quantterm_stack.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    assert "setup" in result.stdout
    assert "run" in result.stdout
    assert "stop" in result.stdout
