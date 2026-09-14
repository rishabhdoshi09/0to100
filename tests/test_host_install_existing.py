from __future__ import annotations

from pathlib import Path

import pytest

from product import host_install as HI
from product import host_install_existing as HIE


@pytest.fixture(autouse=True)
def _not_ephemeral(monkeypatch):
    monkeypatch.setattr(HI, "EPHEMERAL_PREFIXES", ("/definitely-not-this-prefix/",))


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


def test_strict_adapter_routes_install_through_fail_closed_root(monkeypatch):
    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["root_fn"] = HI.ensure_persistent_runtime_root
        return 17

    original = HI.ensure_persistent_runtime_root
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
    assert HI.ensure_persistent_runtime_root is original
