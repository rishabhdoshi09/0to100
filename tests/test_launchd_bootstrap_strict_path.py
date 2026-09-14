"""Reboot-survivability regressions for the macOS launchd bootstrap path.

launchd opens ``StandardOutPath``/``StandardErrorPath`` *before* it execs
``product.host_entrypoint``. If those targets live on the detachable QuantTerm
runtime, a reboot with the sparsebundle not yet attached stops the entrypoint
from ever running its APFS preflight, so the machine can never recover itself.

``product.host_install.install_service_definition`` was fixed for this. The
macOS production installer is not that function -- ``scripts/
install_quantterm_host.sh`` routes Darwin to ``product.host_install_existing``,
whose ``strict_install_service_definition`` replaces it. These tests pin the
whole contract on the path the owner's Mac actually executes.
"""

from __future__ import annotations

import json
import plistlib
import re
from pathlib import Path

import pytest

from product import host_install as HI
from product import host_install_existing as HIE


@pytest.fixture(autouse=True)
def _strict_test_isolation(monkeypatch):
    HIE._close_pinned_root()
    monkeypatch.setattr(HI, "EPHEMERAL_PREFIXES", ("/definitely-not-this-prefix/",))
    yield
    HIE._close_pinned_root()


def _initialised_runtime(root: Path) -> Path:
    root.mkdir(parents=True)
    marker = root / HI.MANIFEST_REL
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({
            "schema_version": 1,
            "initialized_at": "2026-09-14T00:00:00+00:00",
            "runtime_root": str(root),
            "build_sha": "existing-sha",
        }),
        encoding="utf-8",
    )
    return root


def _strict_install(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(
        tmp_path / "Volumes" / "QuantTermStorage" / "QuantTerm" / "runtime"
    )
    home = tmp_path / "home"
    definition = HIE.strict_install_service_definition(
        runtime_root=runtime,
        build_sha="exact-sha",
        repo_root=repo,
        python="/usr/bin/python3",
        manager="launchd",
        home=home,
    )
    return definition, Path(definition["path"]), home, runtime


def _std_paths(plist_text: str) -> tuple[str, str]:
    parsed = plistlib.loads(plist_text.encode("utf-8"))
    return parsed["StandardOutPath"], parsed["StandardErrorPath"]


# 1 + 2: the plist launchd actually reads on macOS.
def test_strict_plist_bootstrap_logs_are_internal_not_runtime(tmp_path: Path) -> None:
    definition, plist_path, home, runtime = _strict_install(tmp_path)
    out, err = _std_paths(plist_path.read_text(encoding="utf-8"))
    bootstrap = (home / "Library" / "Logs" / "QuantTerm").resolve()

    assert out == str(bootstrap / "launchd.out.log")
    assert err == str(bootstrap / "launchd.err.log")
    # No launchd-opened path may sit under the detachable runtime.
    assert not Path(out).is_relative_to(runtime.resolve())
    assert not Path(err).is_relative_to(runtime.resolve())
    assert str(runtime.resolve()) not in out
    assert str(runtime.resolve()) not in err


# 3: launchd's log parent must be materialized by the installer, not by launchd.
def test_strict_install_creates_the_bootstrap_log_parent(tmp_path: Path) -> None:
    definition, plist_path, home, _runtime = _strict_install(tmp_path)
    bootstrap = (home / "Library" / "Logs" / "QuantTerm").resolve()

    # launchd creates the log FILE but never the intermediate directory. A plist
    # pointing into a directory nobody made is the same deadlock, relocated.
    assert bootstrap.is_dir()
    assert definition["bootstrap_log_dir"] == str(bootstrap)


# 2 (cont.): the plist and its log targets must resolve against the same home.
def test_strict_install_honours_home_for_both_plist_and_logs(tmp_path: Path) -> None:
    definition, plist_path, home, _runtime = _strict_install(tmp_path)
    out, err = _std_paths(plist_path.read_text(encoding="utf-8"))

    assert plist_path.is_relative_to(home.resolve())
    assert Path(out).is_relative_to(home.resolve())
    assert Path(err).is_relative_to(home.resolve())


# 4 + 5: the non-strict definition builder must not fabricate a runtime either.
def test_launchd_definition_never_mkdirs_a_missing_runtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    runtime = tmp_path / "detached" / "QuantTermStorage" / "runtime"

    definition = HI.install_service_definition(
        runtime_root=runtime, build_sha="exact-sha", repo_root=repo,
        python="/usr/bin/python3", manager="launchd", home=home,
    )

    assert Path(definition["path"]).is_file()          # 5: still generated
    assert (home / "Library" / "Logs" / "QuantTerm").is_dir()
    assert not runtime.exists()                        # 4: never recreated
    assert not (tmp_path / "detached").exists()        # nor any parent of it


# 6: once Python starts, the APFS preflight runs before strict runtime use.
def test_entrypoint_attaches_storage_before_requiring_runtime_root() -> None:
    from product import host_entrypoint

    source = Path(host_entrypoint.__file__).read_text(encoding="utf-8")
    body = source.split("def main()", 1)[1]

    attach = body.index("prepare_runtime_storage_for_startup()")
    require_root = body.index('QT_RUNTIME_ROOT')
    strict_latch = body.index('QT_RUNTIME_ROOT_REQUIRE_EXISTING')
    supervisor = body.index("host_supervisor")

    assert attach < require_root < strict_latch < supervisor


# 7: an unrecoverable external runtime must fail closed, never substitute one.
def test_unrecoverable_storage_preflight_fails_closed(monkeypatch, tmp_path: Path) -> None:
    from product import host_entrypoint

    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "attach failed: no backing sparsebundle"

    monkeypatch.setattr(host_entrypoint.sys, "platform", "darwin")
    monkeypatch.setenv("QT_STORAGE_PREFLIGHT_REQUIRED", "1")
    monkeypatch.setattr(host_entrypoint.subprocess, "run", lambda *a, **k: _Failed())

    with pytest.raises(RuntimeError, match="storage preflight failed"):
        host_entrypoint.prepare_runtime_storage_for_startup()

    # Fail-closed means nothing was fabricated to make startup "work".
    assert not (tmp_path / "Volumes").exists()


def test_strict_installer_refuses_a_missing_external_runtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    missing = tmp_path / "Volumes" / "QuantTermStorage" / "QuantTerm" / "runtime"

    with pytest.raises(HI.HostInstallError):
        HIE.strict_install_service_definition(
            runtime_root=missing, build_sha="exact-sha", repo_root=repo,
            python="/usr/bin/python3", manager="launchd", home=tmp_path / "home",
        )
    assert not missing.exists()


# 8: Linux/systemd is untouched -- it logs to the journal, not to a file path.
def test_systemd_unit_has_no_file_log_targets_and_no_bootstrap_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    runtime = tmp_path / "srv" / "runtime"

    unit = HI.render_systemd_unit(
        repo_root=repo, runtime_root=runtime, python="/usr/bin/python3", build_sha="exact-sha",
    )
    assert "StandardOutput" not in unit
    assert "StandardError" not in unit
    assert "Library/Logs/QuantTerm" not in unit

    definition = HI.install_service_definition(
        runtime_root=runtime, build_sha="exact-sha", repo_root=repo,
        python="/usr/bin/python3", manager="systemd", home=home,
    )
    assert definition["manager"] == "systemd"
    assert definition["bootstrap_log_dir"] == ""
    assert not (home / "Library" / "Logs" / "QuantTerm").exists()


# 9: only launchd's own stdout/stderr moved. No durable state followed it.
def test_no_durable_state_is_redirected_to_the_internal_bootstrap_dir(tmp_path: Path) -> None:
    definition, plist_path, home, runtime = _strict_install(tmp_path)
    bootstrap = (home / "Library" / "Logs" / "QuantTerm").resolve()

    # The bootstrap dir holds launchd diagnostics only -- the installer puts no
    # database, evidence, scan, manifest or paper state there.
    assert sorted(p.name for p in bootstrap.iterdir()) == []

    parsed = plistlib.loads(plist_path.read_bytes())
    # Durable state is still addressed by QT_RUNTIME_ROOT, on the external disk.
    assert parsed["EnvironmentVariables"]["QT_RUNTIME_ROOT"] == str(runtime.resolve())
    assert (runtime / HI.MANIFEST_REL).is_file()

    # No production module may resolve a durable path under the bootstrap tree.
    offenders = [
        f"{path}:{n}"
        for path in Path("product").rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'Library.{0,4}Logs.{0,4}QuantTerm', line.split("#", 1)[0])
        and "_launchd_bootstrap_log_dir" not in line
    ]
    assert offenders == []


# 10: the pre-existing external-runtime identity protections still hold.
def test_existing_runtime_identity_protections_survive(tmp_path: Path) -> None:
    definition, _plist, _home, runtime = _strict_install(tmp_path)

    # Install still anchored the strict pinned-descriptor service log dir on the
    # external runtime, and still adopted the *existing* manifest identity.
    assert (runtime / "logs" / "service").is_dir()
    manifest = json.loads((runtime / HI.MANIFEST_REL).read_text(encoding="utf-8"))
    assert manifest["runtime_root"] == str(runtime)
    assert manifest["build_sha"] == "existing-sha"

    # And a different manager is still refused outright.
    with pytest.raises(HI.HostInstallError, match="only supports launchd"):
        HIE.strict_install_service_definition(
            runtime_root=runtime, build_sha="exact-sha", repo_root=tmp_path / "repo",
            python="/usr/bin/python3", manager="systemd", home=tmp_path / "home2",
        )
