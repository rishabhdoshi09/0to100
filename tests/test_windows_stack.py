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
        "run_quantterm_lean.ps1",
        "run_quantterm_complete.ps1",
        "stop_quantterm.ps1",
        "run_quantterm.bat",
        "run_quantterm_low_power.bat",
        "run_quantterm_lean.bat",
        "stop_quantterm.bat",
        "run_quantterm_low_power.sh",
        "run_quantterm_lean.sh",
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


def test_lean_flag_in_stack_help():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "quantterm_stack.py"), "run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    assert "--lean" in result.stdout
    assert "3GB" in result.stdout or "report API" in result.stdout.lower()


def test_lean_script_skips_complete_report_api():
    lean_ps1 = (SCRIPTS / "run_quantterm_lean.ps1").read_text(encoding="utf-8")
    lean_bat = (SCRIPTS / "run_quantterm_lean.bat").read_text(encoding="utf-8")
    lean_sh = (SCRIPTS / "run_quantterm_lean.sh").read_text(encoding="utf-8")
    assert "--lean" in lean_ps1
    assert "--lean" in lean_bat
    assert "--lean" in lean_sh
    assert "run --complete" not in lean_ps1
    assert "run --complete" not in lean_bat
    assert "Remove-Item Env:QT_DISABLE_AUTO_MARKET_SCAN" in lean_ps1
