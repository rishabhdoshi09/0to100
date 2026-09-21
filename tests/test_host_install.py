from __future__ import annotations

import json
from pathlib import Path

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



def test_migration_preflights_disk_capacity_before_copy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    source = _write(repo, "logs/large.bin", "x" * 1024)
    monkeypatch.setattr(HI, "MIGRATION_MIN_HEADROOM_BYTES", 1024)
    monkeypatch.setattr(HI, "MIGRATION_HEADROOM_FRACTION", 0.0)
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 1024})(),
    )

    with pytest.raises(HI.HostInstallError, match="insufficient disk space"):
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert source.exists()
    assert not (runtime / HI.MANIFEST_REL).exists()


def test_interrupted_partial_target_resumes_without_recoping_verified_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/a.json", "A")
    _write(repo, "logs/b.json", "B")
    _write(runtime, "logs/a.json", "A")
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    copied = []
    original = HI.shutil.copy2

    def tracking_copy(src, dst):
        copied.append(Path(src).name)
        return original(src, dst)

    monkeypatch.setattr(HI.shutil, "copy2", tracking_copy)
    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert out["state"] == "MIGRATED"
    assert copied == ["b.json"]
    assert (runtime / "logs/a.json").read_text() == "A"
    assert (runtime / "logs/b.json").read_text() == "B"
    assert (runtime / HI.MANIFEST_REL).exists()


def test_live_source_mutation_fails_closed_and_quarantines_unaccepted_copy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    source = _write(repo, "logs/cron.log", "before")
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024**3})(),
    )
    original = HI.shutil.copy2

    def mutate_after_copy(src, dst):
        result = original(src, dst)
        if Path(src) == source:
            source.write_text("after-writer-ran", encoding="utf-8")
        return result

    monkeypatch.setattr(HI.shutil, "copy2", mutate_after_copy)

    with pytest.raises(HI.HostInstallError, match="source changed during migration"):
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert source.read_text() == "after-writer-ran"
    assert not (runtime / HI.MANIFEST_REL).exists()
    quarantines = list(tmp_path.glob("runtime.migration-quarantine-*"))
    assert quarantines
    assert (quarantines[0] / "logs/cron.log").read_text() == "before"


def test_unmanifested_sqlite_sidecars_and_db_log_conflicts_are_quarantined(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/news_curator.sqlite3", "source-db")
    _write(repo, "logs/streamlit.log", "source-log")
    _write(runtime, "logs/news_curator.sqlite3", "target-db")
    _write(runtime, "logs/news_curator.sqlite3-wal", "stale-wal")
    _write(runtime, "logs/news_curator.sqlite3-shm", "stale-shm")
    _write(runtime, "logs/streamlit.log", "target-log")
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    out = HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert out["state"] == "MIGRATED"
    assert (runtime / "logs/news_curator.sqlite3").read_text() == "source-db"
    assert (runtime / "logs/streamlit.log").read_text() == "source-log"
    assert not (runtime / "logs/news_curator.sqlite3-wal").exists()
    assert not (runtime / "logs/news_curator.sqlite3-shm").exists()
    quarantine = Path(out["manifest"]["quarantine_path"])
    assert quarantine.exists()
    assert (quarantine / "logs/news_curator.sqlite3").read_text() == "target-db"
    assert (quarantine / "logs/news_curator.sqlite3-wal").read_text() == "stale-wal"
    assert (quarantine / "logs/news_curator.sqlite3-shm").read_text() == "stale-shm"
    assert (quarantine / "logs/streamlit.log").read_text() == "target-log"


def test_arbitrary_unmanifested_target_only_evidence_still_blocks_split_brain(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write(repo, "logs/source.json", "source")
    _write(runtime, "logs/foreign.json", "target-only")
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    with pytest.raises(HI.HostInstallError) as exc:
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    msg = str(exc.value)
    assert "split-brain" in msg
    assert "target_only_count=1" in msg
    assert "logs/foreign.json" in msg
    assert (runtime / "logs/foreign.json").read_text() == "target-only"
    assert not (runtime / HI.MANIFEST_REL).exists()


def test_empty_source_never_silently_adopts_unmanifested_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    _write(runtime, "logs/foreign.json", "target-only")
    monkeypatch.setattr(
        HI.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10 * 1024**3})(),
    )

    with pytest.raises(HI.HostInstallError, match="refusing to guess authority"):
        HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")

    assert not (runtime / HI.MANIFEST_REL).exists()


def test_macos_first_migration_quiesces_canonical_and_legacy_launchd(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(HI.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(HI.os, "getuid", lambda: 501)
    monkeypatch.setattr(HI.Path, "home", classmethod(lambda cls: tmp_path / "home"))

    def fake_run(args, *, check=True, timeout=30.0):
        calls.append(list(args))
        if args[:2] == ["launchctl", "print"]:
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if args[:3] == ["ps", "-axo", "pid=,command="]:
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(HI, "_run", fake_run)
    stopped = HI._quiesce_macos_quantterm_writers(tmp_path / "repo")

    assert set(stopped) == set(HI.LEGACY_LAUNCHD_LABELS)
    for label in HI.LEGACY_LAUNCHD_LABELS:
        assert ["launchctl", "bootout", f"gui/501/{label}"] in calls



def test_macos_writer_probe_ignores_installer_shell_but_catches_runtime(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(HI.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(HI.os, "getpid", lambda: 999)
    ps = "\n".join([
        f"100 /bin/bash {repo}/scripts/install_quantterm_host.sh --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime",
        f"101 /bin/bash {repo}/scripts/run_quantterm_complete.sh --restart",
        f"102 {repo}/venv/bin/python -u -m operations.market_ops",
        "103 /usr/bin/python3 some_unrelated_quantterm_notes.py",
    ])

    monkeypatch.setattr(
        HI,
        "_run",
        lambda args, check=True, timeout=30.0: type(
            "Proc", (), {"returncode": 0, "stdout": ps, "stderr": ""}
        )(),
    )

    rows = HI._macos_quantterm_processes(repo)
    pids = {row["pid"] for row in rows}

    assert 100 not in pids
    assert 101 in pids
    assert 102 in pids
    assert 103 not in pids
