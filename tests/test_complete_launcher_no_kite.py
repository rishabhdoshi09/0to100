from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_quantterm_complete.sh"


def test_complete_launcher_does_not_require_kite_credentials() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "Zerodha credentials are optional" in text
    assert "Broker-live/F&O/reconciliation lanes are disabled" in text
    assert "official-data scanning, paper execution, replay, settlement and learning continue" in text

    missing_env_block = text.split("if [[ ! -f .env ]]; then", 1)[1].split("auth_rc=0", 1)[0]
    assert "exit 2" not in missing_env_block

    missing_credentials_block = text.split('if [[ "$auth_rc" -eq 2 ]]; then', 1)[1].split(
        'elif [[ "$auth_rc" -eq 1 ]]', 1
    )[0]
    assert "exit 2" not in missing_credentials_block


def test_complete_launcher_never_waits_for_optional_daily_broker_login() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    login_block = text.split('elif [[ "$auth_rc" -eq 1 ]]; then', 1)[1].split("port_open()", 1)[0]
    assert "Core QuantTerm startup will not wait for optional broker auth" in login_block
    assert "python main.py login" in login_block
    assert "paper execution" in login_block
    assert "python main.py login ||" not in login_block
    assert "Browser will open" not in login_block


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
    assert "missing or unmounted, or the mounted filesystem is unresponsive" in text
    assert "Refusing to create a replacement runtime" in text

    assert "Runtime volume present in mount table" in text
    assert 'mount | grep -F " on $runtime_volume "' in text
    assert '[[ -s "$RUNTIME_PROBE_STATUS" ]]' in text
    probe_block = text.split('echo "[COMPLETE STACK] Startup probe: persistent runtime storage"', 1)[1].split('STACK_LOG_DIR=', 1)[0]
    assert 'if ! kill -0 "$runtime_probe_pid"' not in probe_block


def test_complete_launcher_reconciles_stale_launchd_sha_on_restart() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "read_plist_env()" in text
    assert "QT_BUILD_SHA" in text
    assert "QT_RUNTIME_ROOT" in text
    assert "QT_HOST_ENV_FILE" in text
    assert "PYTHONPATH" in text
    assert "git rev-parse HEAD" in text
    assert "Installed host identity differs" in text
    assert "Refusing to run stale installed code" in text
    assert "Cannot reconcile launchd from a dirty checkout" in text
    assert 'local reconcile_runtime="${installed_runtime:-${QT_RUNTIME_ROOT:-}}"' in text
    assert 'local install_args=(install --runtime-root "$reconcile_runtime" --manager launchd)' in text
    assert 'install_args+=(--env-file "$installed_env_file")' in text
    assert '"$installed_env_file" == "$installed_repo/.env"' in text
    assert 'installed_env_file="$ROOT/.env"' in text
    assert "QT_RUNTIME_ROOT_REQUIRE_EXISTING=1" in text
    assert 'PYTHON="$py"' in text
    assert '"$ROOT/scripts/install_quantterm_host.sh"' in text
    assert "Exact-SHA host reconciliation failed" in text


def test_complete_launcher_does_not_silently_restart_old_launchd_build() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    launchd = text.split("run_installed_launchd_console() {", 1)[1].split(
        'if [[ "$(uname -s)" == "Darwin"', 1
    )[0]

    stale_guard = launchd.index('if [[ -n "$current_sha" && ( "$installed_sha" != "$current_sha" || "$installed_repo" != "$ROOT" ) ]]')
    control_call = launchd.index('"$py" -m product.launchd_control "$action"')
    assert stale_guard < control_call
    assert 'if [[ "$requested" != "--restart" ]]' in launchd
    assert 'action="start"' in launchd


def test_complete_launcher_shell_syntax_is_valid() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
