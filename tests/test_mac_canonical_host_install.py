from __future__ import annotations

import inspect
import os
from pathlib import Path

import product.host_entrypoint as HE


def test_host_entrypoint_pins_configured_npm_and_node_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "node-bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    npm.chmod(0o755)

    monkeypatch.setenv("QT_NPM_BIN", str(npm))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    resolved = HE.prepare_frontend_toolchain()

    assert resolved == str(npm.absolute())
    assert os.environ["QT_NPM_BIN"] == str(npm.absolute())
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)


def test_host_entrypoint_resolves_frontend_toolchain_before_supervisor_import():
    source = inspect.getsource(HE.main)
    assert source.index("prepare_frontend_toolchain()") < source.index(
        "from product.host_supervisor import main as supervisor_main"
    )


def test_setup_mac_delegates_to_single_canonical_host_installer():
    script = Path("deploy/setup_mac.sh").read_text(encoding="utf-8")

    assert "scripts/install_quantterm_host.sh" in script
    assert "--manager launchd" in script
    assert "QT_NPM_BIN" in script
    assert 'cat > "$APP_PLIST"' not in script
    assert "<key>Label</key>" not in script

    # Historical labels may appear only so setup can stop/remove them.  The
    # compatibility installer must not load or kickstart any of them again.
    assert "com.quantterm.ui.plist" in script
    assert 'launchctl kickstart -k "gui/$(id -u)/com.quantterm.ui"' not in script
    assert "run_quantterm_mac.sh" not in script
