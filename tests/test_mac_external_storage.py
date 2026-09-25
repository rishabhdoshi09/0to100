"""Regression coverage for the fail-closed macOS external-runtime contract."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "quantterm_storage_preflight.sh"
MAC_RUNNER = ROOT / "scripts" / "run_quantterm_mac.sh"
SETUP = ROOT / "deploy" / "setup_mac.sh"
INNER = ROOT / "scripts" / "run_quantterm.sh"


def _fake_diskutil(bin_dir: Path) -> None:
    path = bin_dir / "diskutil"
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"info\" ]; then exit 2; fi\n"
        "if [ ! -d \"$QT_STORAGE_MOUNT\" ]; then exit 1; fi\n"
        "printf '   Mount Point:              %s\\n' \"$QT_STORAGE_MOUNT\"\n"
        "printf '   File System Personality:  APFS\\n'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_hdiutil(bin_dir: Path) -> None:
    path = bin_dir / "hdiutil"
    path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    path.chmod(0o755)


def _env_for_layout(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    external = tmp_path / "Expansion"
    bundle = external / "QuantTermStorage.sparsebundle"
    mount = tmp_path / "QuantTermStorage"
    runtime = mount / "QuantTerm" / "runtime"
    canonical = tmp_path / "home" / "Library" / "Application Support" / "QuantTerm" / "runtime"
    bundle.mkdir(parents=True)
    runtime.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(runtime)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_diskutil(fake_bin)
    _fake_hdiutil(fake_bin)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "QT_PREFLIGHT_UNAME": "Darwin",
            "QT_STORAGE_PREFLIGHT_REQUIRED": "1",
            "QT_STORAGE_EXTERNAL_VOLUME": str(external),
            "QT_STORAGE_BUNDLE": str(bundle),
            "QT_STORAGE_MOUNT": str(mount),
            "QT_STORAGE_RUNTIME": str(runtime),
            "QT_RUNTIME_LINK": str(canonical),
        }
    )
    return env, external, runtime, canonical


def test_preflight_accepts_exact_apfs_runtime(tmp_path: Path):
    env, _, runtime, _ = _env_for_layout(tmp_path)
    proc = subprocess.run(
        ["bash", str(PREFLIGHT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS: APFS runtime verified" in proc.stdout
    assert not list(runtime.glob(".quantterm-storage-preflight.*"))


def test_preflight_fails_closed_when_external_disk_is_absent(tmp_path: Path):
    env, external, runtime, canonical = _env_for_layout(tmp_path)
    # Removing the external root also removes the sparsebundle. The mounted
    # runtime is deliberately left behind to prove that mount presence alone is
    # not enough: the backing external disk is part of the startup contract.
    for child in (external / "QuantTermStorage.sparsebundle",):
        child.rmdir()
    external.rmdir()

    proc = subprocess.run(
        ["bash", str(PREFLIGHT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 78
    assert "external volume is not mounted" in proc.stderr
    assert canonical.is_symlink()
    assert runtime.is_dir()


def test_preflight_rejects_wrong_runtime_symlink(tmp_path: Path):
    env, _, runtime, canonical = _env_for_layout(tmp_path)
    canonical.unlink()
    wrong = tmp_path / "wrong-runtime"
    wrong.mkdir()
    canonical.symlink_to(wrong)

    proc = subprocess.run(
        ["bash", str(PREFLIGHT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 78
    assert "runtime symlink target mismatch" in proc.stderr
    assert runtime.is_dir()


def test_preflight_never_creates_runtime_paths():
    src = PREFLIGHT.read_text(encoding="utf-8")
    assert 'mkdir -p "$CANONICAL_RUNTIME"' not in src
    assert 'mkdir -p "$EXPECTED_RUNTIME"' not in src
    assert "runtime symlink target mismatch" in src
    assert "File System Personality" in src
    assert 'FS_PERSONALITY" == "APFS"' in src


def test_mac_service_has_one_owner_and_storage_gate():
    setup = SETUP.read_text(encoding="utf-8")
    runner = MAC_RUNNER.read_text(encoding="utf-8")

    # The compatibility runner remains fail-closed, but setup no longer installs
    # it as a second owner. setup_mac delegates to the canonical host installer.
    assert "quantterm_storage_preflight.sh" in setup
    assert setup.index("quantterm_storage_preflight.sh") < setup.index("pip install")
    assert "install_quantterm_host.sh" in setup
    assert "QT_RUNTIME_ROOT_REQUIRE_EXISTING=1" in setup
    assert "--manager launchd" in setup
    assert "run_quantterm_mac.sh" not in setup

    # Every historical launchd owner is explicitly booted out/removed.
    for label in ("com.quantterm.desk", "com.quantterm.ui", "com.quantterm.app", "com.quantterm.autonomy"):
        assert label in setup
    assert 'cat > "$AUTO_PLIST"' not in setup
    assert "kickstart -k \"gui/$(id -u)/com.quantterm.autonomy\"" not in setup

    # Storage/npm contract is persisted for host_entrypoint reboot/reconnect use.
    assert '"QT_NPM_BIN=$NPM_BIN"' in setup
    assert '"QT_STORAGE_PREFLIGHT_REQUIRED=1"' in setup
    assert '"QT_STORAGE_BUNDLE=$STORAGE_BUNDLE"' in setup
    assert '"QT_STORAGE_RUNTIME=$STORAGE_RUNTIME"' in setup

    # The old manual runner itself still gates storage before launching its stack.
    assert "quantterm_storage_preflight.sh" in runner
    assert runner.index("quantterm_storage_preflight.sh") < runner.index("run_quantterm_complete.sh")


def test_inner_supervisor_reads_market_ops_truth_from_runtime_root():
    inner = INNER.read_text(encoding="utf-8")
    assert 'from core.runtime_paths import logs_path' in inner
    assert 'logs_path("market_ops", "worker.lock")' in inner
    assert 'logs_path("market_ops", "runtime.json")' in inner
    assert 'Path("logs/market_ops/worker.lock")' not in inner
    assert 'Path("logs/market_ops/runtime.json")' not in inner


def test_preflight_preserves_hdiutil_attach_failure_detail(tmp_path: Path):
    env, _, runtime, _ = _env_for_layout(tmp_path)
    # Force the mount to appear absent so the preflight attempts hdiutil.
    runtime_root = runtime.parents[1]
    if runtime_root.exists():
        import shutil
        shutil.rmtree(runtime_root)

    hdiutil = Path(env["PATH"].split(":")[0]) / "hdiutil"
    hdiutil.write_text(
        "#!/bin/sh\necho 'hdiutil: attach failed - Resource temporarily unavailable' >&2\nexit 16\n",
        encoding="utf-8",
    )
    hdiutil.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(PREFLIGHT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 78
    assert "could not attach sparsebundle (rc=16)" in proc.stderr
    assert "Resource temporarily unavailable" in proc.stderr
