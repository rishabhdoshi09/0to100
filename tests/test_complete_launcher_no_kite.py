from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_quantterm_complete.sh"


def test_complete_launcher_does_not_require_kite_credentials() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "Zerodha credentials are optional" in text
    assert "Broker live-data/execution lanes are disabled" in text
    assert "research, official-data scans, replay, settlement and learning continue" in text

    missing_env_block = text.split("if [[ ! -f .env ]]; then", 1)[1].split("auth_rc=0", 1)[0]
    assert "exit 2" not in missing_env_block

    missing_credentials_block = text.split('if [[ "$auth_rc" -eq 2 ]]; then', 1)[1].split(
        'elif [[ "$auth_rc" -eq 1 ]]', 1
    )[0]
    assert "exit 2" not in missing_credentials_block


def test_complete_launcher_keeps_daily_login_optional_for_noninteractive_runs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    login_block = text.split('elif [[ "$auth_rc" -eq 1 ]]; then', 1)[1].split("port_open()", 1)[0]
    assert "QT_NONINTERACTIVE" in login_block
    assert "non-broker autonomy continues" in login_block
    assert "python main.py login" in login_block


def test_complete_launcher_machine_lock_is_portable_to_macos() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "try_machine_lock()" in text
    assert "try-fd-lock --fd 200" in text
    assert "if try_machine_lock; then" in text
    assert "if flock -n 200; then" not in text
    assert "if flock -n 201; then" not in text


def test_complete_launcher_bounds_persistent_runtime_probe() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "Startup probe: persistent runtime storage" in text
    assert "Runtime storage probe timed out after 10s" in text
    assert 'kill -TERM "$runtime_probe_pid"' in text
    assert 'kill -KILL "$runtime_probe_pid"' in text
    assert "missing, unmounted, or unresponsive" in text
    assert "Refusing to create a replacement runtime" in text

    assert "Runtime volume present in mount table" in text
    assert 'mount | grep -F " on $runtime_volume "' in text
    assert '[[ -s "$RUNTIME_PROBE_STATUS" ]]' in text
    assert "kill -0" not in text.split('echo "[COMPLETE STACK] Startup probe: persistent runtime storage"', 1)[1].split('STACK_LOG_DIR=', 1)[0]
