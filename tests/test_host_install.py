from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import product.host_install as HI
from product.host_entrypoint import load_env_file


@pytest.fixture(autouse=True)
def _not_ephemeral(monkeypatch):
    monkeypatch.setattr(HI, "EPHEMERAL_PREFIXES", ("/definitely-not-this-prefix/",))


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_first_migration_copies_and_verifies_without_deleting_source(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    source = _write(repo, "logs/product/evidence.json", "real-evidence")
    _write(repo, "db/paper.sqlite", "db-bytes")

    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc123")

    assert out["state"] == "MIGRATED"
    assert source.exists()
    assert (runtime / "logs/product/evidence.json").read_text() == "real-evidence"
    assert (runtime / "db/paper.sqlite").read_text() == "db-bytes"
    assert (repo / ".quantterm_runtime_root").read_text().strip() == str(runtime.resolve())
    manifest = json.loads((runtime / HI.MANIFEST_REL).read_text())
    assert manifest["source_preserved"] is True
    assert manifest["build_sha"] == "abc123"
    assert manifest["copied_files"] == 2


def test_partial_uninitialised_destination_resumes_exact_subset(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/evidence.json", "real-evidence")
    _write(repo, "db/paper.sqlite", "db-bytes")
    _write(runtime, "logs/product/evidence.json", "real-evidence")

    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc123")

    assert out["state"] == "MIGRATED"
    assert (runtime / "logs/product/evidence.json").read_text() == "real-evidence"
    assert (runtime / "db/paper.sqlite").read_text() == "db-bytes"
    manifest = json.loads((runtime / HI.MANIFEST_REL).read_text())
    assert manifest["initial_target_files"] == 1
    assert manifest["resumed_partial_migration"] is True
    assert manifest["source_preserved"] is True


def test_migration_disk_preflight_blocks_before_first_copy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/evidence.json", "x" * 4096)

    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=8192, used=8191, free=1),
    )

    with pytest.raises(HI.HostInstallError, match="insufficient disk space") as err:
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert "required_bytes=" in str(err.value)
    assert "safety_headroom_bytes=" in str(err.value)
    assert not (runtime / "logs/product/evidence.json").exists()
    assert not (runtime / HI.MANIFEST_REL).exists()


def test_source_mutation_during_copy_is_reinventoried_and_reconciled(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    source = _write(repo, "logs/product/state.json", "before-copy")
    original_copy = HI._copy_runtime_file
    mutated = {"done": False}

    def mutating_copy(repo_root, target, rel):
        original_copy(repo_root, target, rel)
        if not mutated["done"] and rel == "logs/product/state.json":
            source.write_text("after-copy", encoding="utf-8")
            mutated["done"] = True

    monkeypatch.setattr(HI, "_copy_runtime_file", mutating_copy)

    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert out["state"] == "MIGRATED"
    assert source.read_text() == "after-copy"
    assert (runtime / "logs/product/state.json").read_text() == "after-copy"
    manifest = json.loads((runtime / HI.MANIFEST_REL).read_text())
    assert manifest["source_changed_during_copy"] is True
    assert manifest["reconcile_passes"] == 1
    assert manifest["copied_files"] == 2


def test_split_brain_diagnostics_include_file_and_byte_deltas(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/state.json", "source-longer")
    _write(runtime, "logs/product/state.json", "dst")

    with pytest.raises(HI.HostInstallError, match="split-brain") as err:
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    detail = str(err.value)
    assert "conflicts=['logs/product/state.json']" in detail
    assert "source_bytes=" in detail
    assert "target_bytes=" in detail
    assert "byte_delta=" in detail
    assert "conflict_bytes=" in detail
    assert not (runtime / HI.MANIFEST_REL).exists()


def test_interrupted_staging_is_cleaned_and_does_not_poison_resume(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/state.json", "authoritative")
    staging = runtime / HI.MIGRATION_STAGING_DIR
    staging.mkdir(parents=True)
    (staging / "abandoned.tmp").write_text("partial-copy", encoding="utf-8")

    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert out["state"] == "MIGRATED"
    assert (runtime / "logs/product/state.json").read_text() == "authoritative"
    assert not (staging / "abandoned.tmp").exists()


def test_divergent_uninitialised_roots_block_instead_of_guessing(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/state.json", "old")
    _write(runtime, "logs/product/state.json", "different")

    with pytest.raises(HI.HostInstallError, match="split-brain"):
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")


def test_initialized_persistent_root_stays_authoritative_after_repo_diverges(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/product/state.json", "initial")
    first = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")
    assert first["state"] == "MIGRATED"

    _write(runtime, "logs/product/state.json", "new-market-evidence")
    _write(repo, "logs/product/state.json", "stale-checkout-copy")
    second = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="def")

    assert second["state"] == "ADOPTED"
    assert (runtime / "logs/product/state.json").read_text() == "new-market-evidence"
    assert (repo / ".quantterm_runtime_root").read_text().strip() == str(runtime.resolve())


def test_runtime_root_inside_checkout_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(HI.HostInstallError, match="outside the source checkout"):
        HI.ensure_persistent_runtime_root(repo / "runtime", repo_root=repo)


def test_systemd_unit_is_pinned_to_exact_sha_and_persistent_root(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "persistent"
    unit = HI.render_systemd_unit(
        repo_root=repo,
        runtime_root=runtime,
        python="/usr/bin/python3",
        build_sha="deadbeef",
        env_file="/secure/quantterm.env",
    )
    assert "product.host_entrypoint" in unit
    assert f"WorkingDirectory={repo.resolve()}" in unit
    assert f'WorkingDirectory="{repo.resolve()}"' not in unit
    assert f"QT_RUNTIME_ROOT={runtime.resolve()}" in unit
    assert "QT_BUILD_SHA=deadbeef" in unit
    assert "QT_HOST_ENV_FILE=/secure/quantterm.env" in unit
    assert "Restart=on-failure" in unit
    assert "KITE_API_SECRET" not in unit


def test_launchd_plist_is_pinned_without_embedding_secret_values(tmp_path):
    repo = tmp_path / "repo"
    runtime = tmp_path / "persistent"
    bootstrap_logs = tmp_path / "home" / "Library" / "Logs" / "QuantTerm"
    plist = HI.render_launchd_plist(
        repo_root=repo,
        runtime_root=runtime,
        python="/usr/bin/python3",
        build_sha="deadbeef",
        env_file="/secure/quantterm.env",
        bootstrap_log_dir=bootstrap_logs,
    )
    assert HI.LAUNCHD_LABEL in plist
    assert "product.host_entrypoint" in plist
    assert "deadbeef" in plist
    assert str(runtime.resolve()) in plist
    assert str(bootstrap_logs.resolve() / "launchd.out.log") in plist
    assert str(bootstrap_logs.resolve() / "launchd.err.log") in plist
    assert str(runtime.resolve() / "logs" / "service") not in plist
    assert str(repo.resolve() / "logs") not in plist
    assert "KITE_API_SECRET" not in plist


def test_environment_loader_never_overwrites_service_manager_values(tmp_path, monkeypatch):
    env = tmp_path / "host.env"
    env.write_text("KITE_API_KEY=file-key\nKITE_API_SECRET='file secret'\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("KITE_API_KEY", "manager-key")
    monkeypatch.delenv("KITE_API_SECRET", raising=False)

    loaded = load_env_file(env)

    assert "KITE_API_KEY" not in loaded
    assert "KITE_API_SECRET" in loaded
    assert __import__("os").environ["KITE_API_KEY"] == "manager-key"
    assert __import__("os").environ["KITE_API_SECRET"] == "file secret"


def test_status_render_is_truthful_without_service_logs():
    text = HI.render_status({
        "runtime_root": "/persistent/qt",
        "deployment": {"build_sha": "abc"},
        "supervisor": {"state": "RUNNING", "pid": 10, "heartbeat_at": "now", "bootstrap": {"state": "READY"}},
        "effective_state": "RUNNING",
        "heartbeat_age_s": 1.0,
        "children": {"market_ops": {"alive": True, "healthy": True, "health_failures": 0}},
        "service": {"manager": "systemd", "returncode": 0},
        "live_execution": {"status": "LOCKED", "locked": True, "verified": True, "authorized": False},
        "healthy": True,
    })
    assert "abc" in text
    assert "RUNNING" in text
    assert "market_ops" in text
    assert "LOCKED" in text
    assert "authorized=False" in text
    assert "PAPER/SHADOW ONLY" not in text
