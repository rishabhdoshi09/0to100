from __future__ import annotations

import json
from pathlib import Path

import pytest

from product import host_install as HI
from product import host_install_existing as HIE


@pytest.fixture(autouse=True)
def _not_ephemeral(monkeypatch):
    monkeypatch.setattr(HI, "EPHEMERAL_PREFIXES", ("/definitely-not-this-prefix/",))


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


def test_existing_strict_runtime_is_verified_without_creating_probe_residue(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = tmp_path / "external" / "QuantTerm" / "runtime"
    runtime.mkdir(parents=True)

    resolved = HIE.ensure_existing_persistent_runtime_root(runtime, repo_root=repo)

    assert resolved == runtime.resolve(strict=True)
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

    with pytest.raises(HI.HostInstallError, match="requires existing runtime manifest"):
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


def test_strict_adoption_rechecks_storage_before_persisting_pointer(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = _initialised_runtime(tmp_path / "external" / "QuantTerm" / "runtime")
    calls = {"count": 0}
    original = HIE.ensure_existing_persistent_runtime_root

    def disappearing(path, *, repo_root=HI.REPO_ROOT):
        calls["count"] += 1
        if calls["count"] == 1:
            return original(path, repo_root=repo_root)
        raise HI.HostInstallError("runtime disappeared during strict adoption")

    monkeypatch.setattr(HIE, "ensure_existing_persistent_runtime_root", disappearing)

    with pytest.raises(HI.HostInstallError, match="disappeared"):
        HIE.adopt_existing_runtime(runtime, repo_root=repo)

    assert not (repo / ".quantterm_runtime_root").exists()


def test_strict_adapter_routes_install_through_fail_closed_root_and_adoption(monkeypatch):
    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["root_fn"] = HI.ensure_persistent_runtime_root
        observed["migrate_fn"] = HI.migrate_repo_runtime
        return 17

    original_root = HI.ensure_persistent_runtime_root
    original_migrate = HI.migrate_repo_runtime
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
    assert HI.ensure_persistent_runtime_root is original_root
    assert HI.migrate_repo_runtime is original_migrate
