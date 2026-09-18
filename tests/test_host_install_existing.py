from __future__ import annotations

import json
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


def test_missing_strict_runtime_is_never_created(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    missing = tmp_path / "external" / "QuantTerm" / "runtime"

    with pytest.raises(HI.HostInstallError, match="must already exist"):
        HIE.ensure_existing_persistent_runtime_root(missing, repo_root=repo)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_existing_strict_runtime_is_pinned_and_probe_leaves_no_residue(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = tmp_path / "external" / "QuantTerm" / "runtime"
    runtime.mkdir(parents=True)

    resolved = HIE.ensure_existing_persistent_runtime_root(runtime, repo_root=repo)

    assert resolved == runtime.resolve(strict=True)
    assert HIE._PINNED_ROOT == resolved
    assert HIE._PINNED_ROOT_FD is not None
    assert not list(runtime.glob(".quantterm-install-probe.*"))


def test_strict_runtime_inside_checkout_is_rejected(tmp_path: Path):
    repo = tmp_path / "repo"
    runtime = repo / "runtime"
    runtime.mkdir(parents=True)

    with pytest.raises(HI.HostInstallError, match="outside the source checkout"):
        HIE.ensure_existing_persistent_runtime_root(runtime, repo_root=repo)


def test_strict_adoption_requires_existing_manifest_and_never_initialises(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = tmp_path / "external" / "QuantTerm" / "runtime"
    runtime.mkdir(parents=True)

    with pytest.raises(HI.HostInstallError, match="strict runtime file is unreadable"):
        HIE.adopt_existing_runtime(runtime, repo_root=repo, build_sha="new-sha")

    assert not (runtime / HI.MANIFEST_REL).exists()
    assert not (repo / ".quantterm_runtime_root").exists()


def test_strict_adoption_accepts_only_existing_initialised_runtime(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(tmp_path / "external" / "QuantTerm" / "runtime")

    result = HIE.adopt_existing_runtime(runtime, repo_root=repo, build_sha="new-sha")

    assert result["state"] == "ADOPTED"
    assert result["runtime_root"] == str(runtime.resolve(strict=True))
    assert result["manifest"]["build_sha"] == "existing-sha"
    assert (repo / ".quantterm_runtime_root").read_text(encoding="utf-8").strip() == str(runtime.resolve())


def test_strict_adoption_reprobes_pinned_storage_before_pointer(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(tmp_path / "external" / "QuantTerm" / "runtime")
    calls = {"count": 0}
    original = HIE._anchored_probe

    def disappearing(prefix=".quantterm-install-probe"):
        calls["count"] += 1
        if calls["count"] == 1:
            return original(prefix)
        raise HI.HostInstallError("runtime disappeared during strict adoption")

    monkeypatch.setattr(HIE, "_anchored_probe", disappearing)

    with pytest.raises(HI.HostInstallError, match="disappeared"):
        HIE.adopt_existing_runtime(runtime, repo_root=repo)

    assert calls["count"] == 2
    assert not (repo / ".quantterm_runtime_root").exists()


def test_strict_atomic_json_writes_inside_pin_without_creating_missing_parent(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(tmp_path / "external" / "QuantTerm" / "runtime")
    HIE.ensure_existing_persistent_runtime_root(runtime, repo_root=repo)

    target = runtime / "state" / "host_deployment.json"
    HIE.strict_atomic_json(target, {"build_sha": "abc"})
    assert json.loads(target.read_text(encoding="utf-8"))["build_sha"] == "abc"

    missing_parent = runtime / "never-create-me"
    with pytest.raises(Exception):
        HIE.strict_atomic_json(missing_parent / "state.json", {"x": 1})
    assert not missing_parent.exists()


def test_strict_service_definition_creates_runtime_logs_via_pinned_descriptor(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(tmp_path / "external" / "QuantTerm" / "runtime")
    home = tmp_path / "home"

    result = HIE.strict_install_service_definition(
        runtime_root=runtime,
        build_sha="deadbeef",
        repo_root=repo,
        python="/usr/bin/python3",
        manager="launchd",
        home=home,
    )

    assert (runtime / "logs" / "service").is_dir()
    plist = Path(result["path"])
    assert plist.is_file()
    assert HI.LAUNCHD_LABEL in plist.read_text(encoding="utf-8")


def test_strict_adapter_routes_all_runtime_mutation_surfaces(monkeypatch):
    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["root_fn"] = HI.ensure_persistent_runtime_root
        observed["migrate_fn"] = HI.migrate_repo_runtime
        observed["atomic_fn"] = HI._atomic_json
        observed["service_fn"] = HI.install_service_definition
        observed["preflight_fn"] = HI.run_required_preflight
        return 17

    originals = {
        "root": HI.ensure_persistent_runtime_root,
        "migrate": HI.migrate_repo_runtime,
        "atomic": HI._atomic_json,
        "service": HI.install_service_definition,
        "preflight": HI.run_required_preflight,
    }
    monkeypatch.setattr(HI, "main", fake_main)

    rc = HIE.main(["--runtime-root", "/already-mounted/runtime", "--manager", "launchd"])

    assert rc == 17
    assert observed["argv"] == [
        "install",
        "--runtime-root",
        "/already-mounted/runtime",
        "--manager",
        "launchd",
    ]
    assert observed["root_fn"] is HIE.ensure_existing_persistent_runtime_root
    assert observed["migrate_fn"] is HIE.adopt_existing_runtime
    assert observed["atomic_fn"] is HIE.strict_atomic_json
    assert observed["service_fn"] is HIE.strict_install_service_definition
    assert observed["preflight_fn"] is HIE.strict_run_required_preflight
    assert HI.ensure_persistent_runtime_root is originals["root"]
    assert HI.migrate_repo_runtime is originals["migrate"]
    assert HI._atomic_json is originals["atomic"]
    assert HI.install_service_definition is originals["service"]
    assert HI.run_required_preflight is originals["preflight"]
    assert HIE._PINNED_ROOT_FD is None
