from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from product.runtime_storage_guard import (
    MANIFEST_REL,
    RuntimeStorageGuard,
    RuntimeStorageUnavailable,
    establish_runtime_storage_identity,
    verify_runtime_storage,
)


def _runtime(root: Path) -> Path:
    root.mkdir(parents=True)
    manifest = root / MANIFEST_REL
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "initialized_at": "2026-09-14T00:00:00+00:00",
            "runtime_root": str(root),
            "build_sha": "test-sha",
        }),
        encoding="utf-8",
    )
    return root


def test_missing_runtime_is_never_created(tmp_path: Path):
    missing = tmp_path / "external" / "QuantTerm" / "runtime"

    with pytest.raises(RuntimeStorageUnavailable, match="runtime root is missing"):
        establish_runtime_storage_identity(missing)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_manifest_identity_is_verified_and_tampering_fails_closed(tmp_path: Path):
    root = _runtime(tmp_path / "runtime")
    identity = establish_runtime_storage_identity(root)

    ok, reason = verify_runtime_storage(identity)
    assert ok is True
    assert reason == ""
    assert not list(root.glob(".quantterm-runtime-probe.*"))

    manifest = root / MANIFEST_REL
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["build_sha"] = "different-sha"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    ok, reason = verify_runtime_storage(identity)
    assert ok is False
    assert "manifest changed unexpectedly" in reason


def test_broken_runtime_symlink_detects_loss_and_same_runtime_can_recover(tmp_path: Path):
    external = tmp_path / "external"
    runtime = _runtime(external / "QuantTerm" / "runtime")
    canonical = tmp_path / "Application Support" / "QuantTerm" / "runtime"
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(runtime, target_is_directory=True)

    identity = establish_runtime_storage_identity(canonical)
    ok, _ = verify_runtime_storage(identity)
    assert ok is True

    offline = external / "QuantTerm" / "runtime.offline"
    runtime.rename(offline)
    ok, reason = verify_runtime_storage(identity)
    assert ok is False
    assert "disappeared" in reason or "verification failed" in reason
    assert canonical.is_symlink()

    offline.rename(runtime)
    ok, reason = verify_runtime_storage(identity)
    assert ok is True, reason


def test_installed_runtime_path_resolution_fails_before_mkdir_on_storage_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from core import runtime_paths

    external = tmp_path / "external"
    runtime = _runtime(external / "QuantTerm" / "runtime")
    canonical = tmp_path / "Application Support" / "QuantTerm" / "runtime"
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(runtime, target_is_directory=True)
    monkeypatch.setenv(runtime_paths.ENV_VAR, str(canonical))
    monkeypatch.setenv(runtime_paths.REQUIRE_EXISTING_ENV, "1")

    assert runtime_paths.runtime_root() == runtime.resolve()
    assert runtime_paths.ensure_logs_path("probe", "ok.json") == runtime.resolve() / "logs" / "probe" / "ok.json"

    # Simulate the external runtime disappearing while the canonical symlink
    # remains. Shared path helpers must fail before mkdir can create anything.
    offline = external / "QuantTerm" / "runtime.offline"
    runtime.rename(offline)
    with pytest.raises(RuntimeError, match="runtime root is missing"):
        runtime_paths.ensure_logs_path("would-be-split", "state.json")
    assert canonical.is_symlink()
    assert not runtime.exists()

    offline.rename(runtime)
    assert runtime_paths.runtime_root() == runtime.resolve()


def test_strict_runtime_requires_configuration_and_non_strict_dev_mode_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from core import runtime_paths

    missing = tmp_path / "not-mounted" / "runtime"
    monkeypatch.setenv(runtime_paths.ENV_VAR, str(missing))
    monkeypatch.delenv(runtime_paths.REQUIRE_EXISTING_ENV, raising=False)

    # Ordinary developer/test callers still receive the configured path; only
    # the installed host opts into fail-closed existence enforcement.
    assert runtime_paths.runtime_root() == missing
    assert not missing.exists()

    monkeypatch.setenv(runtime_paths.REQUIRE_EXISTING_ENV, "true")
    with pytest.raises(RuntimeError, match="runtime root is missing"):
        runtime_paths.runtime_root()
    assert not missing.exists()


def test_strict_runtime_without_override_or_pointer_has_no_repo_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    from core import runtime_paths

    monkeypatch.delenv(runtime_paths.ENV_VAR, raising=False)
    monkeypatch.setenv(runtime_paths.REQUIRE_EXISTING_ENV, "1")
    monkeypatch.setattr(runtime_paths, "read_runtime_pointer", lambda: None)

    with pytest.raises(RuntimeError, match="requires an existing configured persistent runtime root"):
        runtime_paths.runtime_root()


def test_loss_event_stays_latched_across_fast_recovery_until_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = _runtime(tmp_path / "runtime")
    guard = RuntimeStorageGuard(root=root, interval_s=0.2)
    checks = iter([False, True, True])
    monkeypatch.setattr(guard, "check", lambda: next(checks))
    callbacks: list[str] = []
    guard._on_loss = callbacks.append

    class ThreeTicks:
        def __init__(self):
            self.count = 0

        def wait(self, _timeout: float) -> bool:
            self.count += 1
            return self.count > 3

        def is_set(self) -> bool:
            return self.count > 3

        def set(self) -> None:
            self.count = 99

    guard.shutdown = ThreeTicks()  # type: ignore[assignment]
    guard._loop()

    # Storage was healthy again on ticks 2 and 3, but the loss must remain
    # latched until the entrypoint has observed the supervisor shutdown.
    assert guard.lost.is_set() is True
    assert len(callbacks) == 1

    guard.shutdown = threading.Event()
    monkeypatch.setattr(guard, "check", lambda: True)
    assert guard.wait_until_recovered() is True
    assert guard.lost.is_set() is False


def test_host_entrypoint_restarts_supervisor_after_latched_storage_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import product.host_entrypoint as entrypoint
    import product.host_supervisor as host_supervisor
    import product.runtime_storage_guard as storage_module

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("QT_BUILD_SHA", "exact-test-sha")
    # entrypoint.main intentionally overwrites this to 1. Register the key with
    # monkeypatch first so pytest restores the pre-test process environment even
    # though the production assignment itself is direct os.environ mutation.
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "0")

    class FakeGuard:
        def __init__(self, *, interval_s: float):
            self.interval_s = interval_s
            self.lost = threading.Event()
            self.shutdown = threading.Event()
            self.recoveries = 0

        def start(self, _on_loss):
            return None

        def check(self) -> bool:
            return True

        def wait_until_recovered(self, *, should_stop=None) -> bool:
            assert self.lost.is_set() is True
            self.recoveries += 1
            self.lost.clear()
            return not (should_stop and should_stop())

        def close(self) -> None:
            self.shutdown.set()

    guard = FakeGuard(interval_s=15)
    monkeypatch.setattr(storage_module, "RuntimeStorageGuard", lambda interval_s: guard)

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(entrypoint.threading, "Thread", NoopThread)
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args, **kwargs: None)

    calls = {"count": 0}

    def fake_supervisor_main() -> int:
        calls["count"] += 1
        if calls["count"] == 1:
            guard.lost.set()
            return 0
        return 7

    monkeypatch.setattr(host_supervisor, "main", fake_supervisor_main)

    rc = entrypoint.main()

    assert rc == 7
    assert calls["count"] == 2
    assert guard.recoveries == 1
    assert __import__("os").environ["QT_RUNTIME_ROOT_REQUIRE_EXISTING"] == "1"
