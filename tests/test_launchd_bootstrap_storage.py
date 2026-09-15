from __future__ import annotations

from pathlib import Path

import product.host_install as host_install


def test_launchd_plist_keeps_bootstrap_logs_off_detachable_runtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "Volumes" / "QuantTermStorage" / "runtime"
    bootstrap_logs = tmp_path / "home" / "Library" / "Logs" / "QuantTerm"

    plist = host_install.render_launchd_plist(
        repo_root=repo,
        runtime_root=runtime,
        python="/usr/bin/python3",
        build_sha="exact-sha",
        bootstrap_log_dir=bootstrap_logs,
    )

    assert str(bootstrap_logs.resolve() / "launchd.out.log") in plist
    assert str(bootstrap_logs.resolve() / "launchd.err.log") in plist
    assert str(runtime.resolve() / "logs" / "service") not in plist
    # The detachable runtime remains only the application runtime contract.
    assert "QT_RUNTIME_ROOT" in plist
    assert str(runtime.resolve()) in plist


def test_launchd_definition_does_not_create_missing_detachable_runtime(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    runtime = tmp_path / "detached" / "QuantTermStorage" / "runtime"

    definition = host_install.install_service_definition(
        runtime_root=runtime,
        build_sha="exact-sha",
        repo_root=repo,
        python="/usr/bin/python3",
        manager="launchd",
        home=home,
    )

    plist_path = Path(definition["path"])
    bootstrap_logs = (home / "Library" / "Logs" / "QuantTerm").resolve()
    plist = plist_path.read_text(encoding="utf-8")

    assert plist_path.is_file()
    assert bootstrap_logs.is_dir()
    assert definition["bootstrap_log_dir"] == str(bootstrap_logs)
    assert str(bootstrap_logs / "launchd.out.log") in plist
    assert str(bootstrap_logs / "launchd.err.log") in plist
    assert str(runtime.resolve() / "logs" / "service") not in plist
    # Critical reboot invariant: preparing launchd itself must not fabricate a
    # replacement runtime on the internal disk while external storage is absent.
    assert not runtime.exists()
