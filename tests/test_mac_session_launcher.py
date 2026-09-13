"""Regression coverage for the Catalina macOS process-session launcher."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mac_wrapper_supplies_setsid_without_extra_system_dependency():
    src = (ROOT / "scripts" / "run_quantterm_mac.sh").read_text(encoding="utf-8")
    assert "if ! command -v setsid" in src
    assert "setsid()" in src
    assert "export -f setsid" in src
    assert "exec python - \"$@\"" in src
    assert "os.setsid()" in src
    assert 'command = os.environ.get("QT_NPM_BIN") or command' in src
    assert "os.execvpe(command, argv, os.environ)" in src


def test_mac_wrapper_resolves_npm_without_shell_profile():
    src = (ROOT / "scripts" / "run_quantterm_mac.sh").read_text(encoding="utf-8")
    assert "resolve_npm()" in src
    assert '"$HOME"/.nvm/versions/node/*/bin/npm' in src
    assert '"$HOME"/.volta/bin/npm' in src
    assert '"$HOME"/.asdf/shims/npm' in src
    assert 'export QT_NPM_BIN' in src
    assert 'export PATH="$NPM_BIN_DIR:' in src


def test_mac_launchagent_enters_through_compatibility_wrapper():
    setup = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    assert "scripts/run_quantterm_mac.sh" in setup
    assert "scripts/run_quantterm_complete.sh" not in setup.split("cat > \"$APP_PLIST\"", 1)[1].split("PLIST", 1)[0]


def test_mac_installer_pins_npm_and_node_directory_into_launchd_environment():
    setup = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    assert 'NPM_BIN="${QT_NPM_BIN:-$(command -v npm' in setup
    assert '<key>QT_NPM_BIN</key><string>$NPM_BIN</string>' in setup
    assert 'LAUNCH_PATH="$NPM_BIN_DIR:' in setup
    assert '<key>PATH</key><string>$LAUNCH_PATH</string>' in setup
