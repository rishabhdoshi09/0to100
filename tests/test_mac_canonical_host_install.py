from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess

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


def test_required_mac_storage_preflight_attaches_before_runtime_use(tmp_path, monkeypatch):
    from core import runtime_paths

    repo = tmp_path / "repo"
    script = repo / "scripts" / "quantterm_storage_preflight.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    monkeypatch.setattr(HE.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_paths, "REPO_ROOT", repo)
    monkeypatch.setenv("QT_STORAGE_PREFLIGHT_REQUIRED", "1")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="[STORAGE PREFLIGHT] PASS: APFS runtime verified\n",
            stderr="",
        )

    monkeypatch.setattr(HE.subprocess, "run", fake_run)

    result = HE.prepare_runtime_storage_for_startup()

    assert result == "[STORAGE PREFLIGHT] PASS: APFS runtime verified"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ["/bin/bash", str(script)]
    assert kwargs["cwd"] == repo
    assert kwargs["env"]["QT_STORAGE_PREFLIGHT_REQUIRED"] == "1"
    assert kwargs["env"]["QT_STORAGE_PREFLIGHT_ATTACH"] == "1"


def test_mac_storage_preflight_failure_is_fail_closed(tmp_path, monkeypatch):
    from core import runtime_paths

    repo = tmp_path / "repo"
    script = repo / "scripts" / "quantterm_storage_preflight.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\nexit 78\n", encoding="utf-8")

    monkeypatch.setattr(HE.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_paths, "REPO_ROOT", repo)
    monkeypatch.setenv("QT_STORAGE_PREFLIGHT_REQUIRED", "1")
    monkeypatch.setattr(
        HE.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=78,
            stdout="",
            stderr="[STORAGE PREFLIGHT] FAIL: runtime volume is not mounted\n",
        ),
    )

    try:
        HE.prepare_runtime_storage_for_startup()
    except RuntimeError as exc:
        assert "macOS storage preflight failed with rc=78" in str(exc)
    else:
        raise AssertionError("required failed storage preflight must block host startup")


def test_host_entrypoint_prepares_storage_and_toolchain_before_supervisor_import():
    source = inspect.getsource(HE.main)
    storage_index = source.index("prepare_runtime_storage_for_startup()")
    strict_index = source.index('os.environ["QT_RUNTIME_ROOT_REQUIRE_EXISTING"] = "1"')
    toolchain_index = source.index("prepare_frontend_toolchain()")
    supervisor_index = source.index("from product.host_supervisor import main as supervisor_main")

    assert storage_index < strict_index
    assert storage_index < toolchain_index < supervisor_index
    assert "prepare=prepare_runtime_storage_for_startup" in source


def test_setup_mac_delegates_to_single_canonical_host_installer():
    script = Path("deploy/setup_mac.sh").read_text(encoding="utf-8")

    assert "scripts/install_quantterm_host.sh" in script
    assert "--manager launchd" in script
    assert "QT_NPM_BIN" in script
    assert 'cat > "$APP_PLIST"' not in script
    assert "<key>Label</key>" not in script

    # Historical labels may appear only so setup can stop/remove them. The
    # compatibility installer must evict orphaned jobs by service target and
    # must never load/kickstart those historical owners again.
    assert "com.quantterm.ui.plist" in script
    assert 'launchctl bootout "gui/$UID_VALUE/$label"' in script
    assert 'launchctl kickstart -k "gui/$(id -u)/com.quantterm.ui"' not in script
    assert "run_quantterm_mac.sh" not in script

    # The one canonical owner is stopped before Python/package mutation so an
    # update cannot leave already-running children backed by a half-updated venv.
    assert "com.quantterm.desk" in script
    assert script.index("com.quantterm.desk") < script.index('"$PYTHON_BIN" -m pip install')

    # The secure host env must contain enough information to attach and verify
    # the same APFS sparsebundle on login/reboot and after removable-disk loss.
    for key in (
        "QT_STORAGE_PREFLIGHT_REQUIRED=1",
        "QT_STORAGE_EXTERNAL_VOLUME=$EXTERNAL_VOLUME",
        "QT_STORAGE_BUNDLE=$STORAGE_BUNDLE",
        "QT_STORAGE_MOUNT=$STORAGE_MOUNT",
        "QT_STORAGE_RUNTIME=$STORAGE_RUNTIME",
        "QT_RUNTIME_LINK=$RUNTIME_LINK",
    ):
        assert key in script

    # This existing external runtime is adopted, never created. If the volume
    # disappears after preflight, the strict installer must block instead of
    # recreating /Volumes/... on the internal disk.
    assert "export QT_RUNTIME_ROOT_REQUIRE_EXISTING=1" in script
    assert script.index("export QT_RUNTIME_ROOT_REQUIRE_EXISTING=1") < script.index(
        'exec "$APP_DIR/scripts/install_quantterm_host.sh"'
    )


def test_install_wrapper_routes_strict_runtime_to_fail_closed_adapter():
    script = Path("scripts/install_quantterm_host.sh").read_text(encoding="utf-8")
    assert "QT_RUNTIME_ROOT_REQUIRE_EXISTING" in script
    assert "product.host_install_existing" in script
    assert "product.host_install install" in script


def test_setup_server_delegates_to_single_canonical_host_installer():
    script = Path("deploy/setup_server.sh").read_text(encoding="utf-8")

    assert "scripts/install_quantterm_host.sh" in script
    assert "--manager systemd" in script
    assert "product.host_entrypoint" in script
    assert "quantterm-autonomy.service" in script  # cleanup only
    assert "quantterm-ui.service" in script        # cleanup only
    assert "ExecStart=" not in script
    assert "main.py autonomy --interval" not in script
    assert "cursor/live-terminal-contract-858e" not in script


def test_legacy_split_service_templates_are_not_shippable():
    retired = (
        "deploy/com.quantterm.autonomy.plist",
        "deploy/com.quantterm.ui.plist",
        "deploy/quantterm-autonomy.service",
        "deploy/quantterm-ui.service",
        "deploy/quantterm.service",
    )
    assert not [path for path in retired if Path(path).exists()]
