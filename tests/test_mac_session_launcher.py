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


def test_mac_installer_delegates_to_canonical_host_owner():
    setup = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    assert "install_quantterm_host.sh" in setup
    assert "--manager launchd" in setup
    assert "QT_RUNTIME_ROOT_REQUIRE_EXISTING=1" in setup
    assert "product.host_install" in setup
    assert "product.host_supervisor" in setup
    assert "run_quantterm_mac.sh" not in setup
    assert "run_quantterm_complete.sh" not in setup


def test_mac_installer_persists_npm_and_storage_contract_for_launchd():
    setup = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    assert 'NPM_BIN="${QT_NPM_BIN:-$(command -v npm' in setup
    assert '"QT_NPM_BIN=$NPM_BIN"' in setup
    assert '"QT_STORAGE_PREFLIGHT_REQUIRED=1"' in setup
    assert '"QT_STORAGE_EXTERNAL_VOLUME=$EXTERNAL_VOLUME"' in setup
    assert '"QT_STORAGE_BUNDLE=$STORAGE_BUNDLE"' in setup
    assert '"QT_STORAGE_MOUNT=$STORAGE_MOUNT"' in setup
    assert '"QT_STORAGE_RUNTIME=$STORAGE_RUNTIME"' in setup
    assert '"QT_RUNTIME_LINK=$RUNTIME_LINK"' in setup
    assert "chmod 600 \"$APP_DIR/.env\"" in setup
