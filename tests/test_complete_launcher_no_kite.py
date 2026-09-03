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
