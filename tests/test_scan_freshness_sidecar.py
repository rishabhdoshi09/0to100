"""Decision-gate scan freshness cannot reparse giant scan JSON every poll."""
from __future__ import annotations

import json
import os

from product import scan_store
from product import desk_pipeline


def _scan(*, session="2026-10-09"):
    return {
        "schema_version": 2,
        "scanned_at": "2026-10-11T09:15:00+00:00",
        "as_of_session": session,
        "history_latest_date": session,
        "provenance": {
            "market_session_date": session,
            "price_data_as_of": session,
            "data_current": True,
            "scan_id": "1234567890abcdef",
        },
        "records": [{"symbol": "TCS", "status_scope": "SCANNER_SETUP"}],
        "universe_size": 1,
        "summary": {"with_any_setup": 1},
    }


def test_scan_sidecar_matches_exact_atomic_publication(tmp_path):
    path = tmp_path / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    metadata = scan_store.load_scan_metadata(path)
    assert metadata is not None
    assert metadata["provenance"]["price_data_as_of"] == "2026-10-09"
    assert metadata["records_count"] == 1
    assert not metadata.get("records")

    # A second atomic publication at the same filename invalidates the
    # earlier sidecar. Do not borrow session provenance from older content.
    replacement = path.with_suffix(".replacement")
    replacement.write_text(json.dumps(_scan(session="2026-10-08")), encoding="utf-8")
    os.replace(replacement, path)
    assert scan_store.load_scan_metadata(path) is None
    assert scan_store.load_scan(path)["provenance"]["price_data_as_of"] == "2026-10-08"


def test_metadata_does_not_pretend_missing_json_is_current(tmp_path):
    path = tmp_path / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    path.unlink()
    assert scan_store.load_scan_metadata(path) is None


def test_corrupt_or_old_sidecar_fails_closed(tmp_path):
    path = tmp_path / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    sidecar = scan_store._scan_metadata_path(path)
    sidecar.write_text('{"schema_version": 2, "file_identity": {}}', encoding="utf-8")
    assert scan_store.load_scan_metadata(path) is None


def test_decision_freshness_avoids_full_scan_json_when_sidecar_verified(tmp_path, monkeypatch):
    path = tmp_path / "product" / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *args, **kwargs: {
            "current": True, "usable_for_scan": True,
            "available_session": "2026-10-09",
        },
    )
    monkeypatch.setattr(
        scan_store, "load_scan", lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("Full saved scan JSON reloaded during metadata-only freshness")
        ),
    )
    assert desk_pipeline.scan_is_fresh() is True


def test_sidecar_never_bypasses_stale_authoritative_exchange_session(tmp_path, monkeypatch):
    path = tmp_path / "product" / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *args, **kwargs: {
            "current": False, "usable_for_scan": False,
            "available_session": "2026-10-10",
        },
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_changed_json_makes_gate_use_authoritative_fallback(tmp_path, monkeypatch):
    path = tmp_path / "product" / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    replacement = path.with_suffix(".new.json")
    replacement.write_text(json.dumps(_scan(session="2026-10-08")), encoding="utf-8")
    os.replace(replacement, path)
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *args, **kwargs: {
            "current": True, "usable_for_scan": True,
            "available_session": "2026-10-09",
        },
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_official_history_lookup_error_cannot_be_replaced_by_scan_wallclock(tmp_path, monkeypatch):
    path = tmp_path / "product" / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(), path)
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)

    def unavailable(*_args, **_kwargs):
        raise OSError("external data volume unavailable")

    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness", unavailable,
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_future_dated_scan_session_never_overrides_exchange_history(tmp_path, monkeypatch):
    path = tmp_path / "product" / "latest_momentum_scan.json"
    scan_store.save_scan(_scan(session="2026-10-12"), path)
    monkeypatch.setattr(desk_pipeline, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *args, **kwargs: {
            "current": True, "usable_for_scan": True,
            "available_session": "2026-10-09",
        },
    )
    assert desk_pipeline.scan_is_fresh() is False


def test_concurrent_scan_writer_cannot_bind_own_sidecar_to_other_writer_file(tmp_path, monkeypatch):
    path = tmp_path / "latest_momentum_scan.json"
    replaced = []
    original_replace = scan_store.os.replace

    def competing_publish(src, dst):
        original_replace(src, dst)
        if str(dst) == str(path) and not replaced:
            # Another writer atomically publishes newer canonical content
            # before this writer checks the published inode identity.
            rival = tmp_path / ".rival.json"
            rival.write_text(json.dumps(_scan(session="2026-10-08")), encoding="utf-8")
            original_replace(rival, path)
            replaced.append(True)

    monkeypatch.setattr(scan_store.os, "replace", competing_publish)
    scan_store.save_scan(_scan(session="2026-10-09"), path)

    assert replaced == [True]
    assert scan_store.load_scan_metadata(path) is None
    assert scan_store.load_scan(path)["provenance"]["price_data_as_of"] == "2026-10-08"


def test_scan_publications_use_distinct_temp_names_and_no_stray_files(tmp_path, monkeypatch):
    path = tmp_path / "latest_momentum_scan.json"
    original_replace = scan_store.os.replace
    source_files = []

    def note_publish(src, dst):
        if str(dst) == str(path):
            source_files.append(str(src))
        return original_replace(src, dst)

    monkeypatch.setattr(scan_store.os, "replace", note_publish)
    scan_store.save_scan(_scan(session="2026-10-08"), path)
    scan_store.save_scan(_scan(session="2026-10-09"), path)

    assert len(source_files) == 2 and source_files[0] != source_files[1]
    assert scan_store.load_scan_metadata(path)["provenance"]["price_data_as_of"] == "2026-10-09"
    assert not any(item.name.endswith(".tmp") for item in tmp_path.iterdir())


def test_installed_host_scan_writer_never_recreates_missing_external_mount(tmp_path, monkeypatch):
    import pytest

    missing_root = tmp_path / "disconnected-volume" / "QuantTerm" / "runtime"
    destination = missing_root / "logs" / "product" / "latest_momentum_scan.json"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(missing_root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    with pytest.raises(RuntimeError, match="runtime root"):
        scan_store.save_scan(_scan(), destination)
    assert not missing_root.exists()
    assert not destination.parent.exists()


def test_installed_host_scan_writer_blocks_unredirected_checkout_path(tmp_path, monkeypatch):
    import pytest

    mounted = tmp_path / "QuantTerm" / "runtime"
    mounted.mkdir(parents=True)
    other = tmp_path / "checkout" / "logs" / "product" / "latest_momentum_scan.json"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(mounted))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    with pytest.raises(RuntimeError, match="outside the installed persistent runtime"):
        scan_store.save_scan(_scan(), other)
    assert not other.parent.exists()


def test_installed_host_scan_writer_allows_existing_persistent_runtime(tmp_path, monkeypatch):
    mounted = tmp_path / "QuantTerm" / "runtime"
    mounted.mkdir(parents=True)
    destination = mounted / "logs" / "product" / "latest_momentum_scan.json"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(mounted))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    assert scan_store.save_scan(_scan(), destination) == destination
    assert scan_store.load_scan_metadata(destination) is not None
