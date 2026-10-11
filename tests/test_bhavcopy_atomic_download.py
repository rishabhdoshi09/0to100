"""NSE whole-market CSV is committed only after verified, atomic download."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from data import bhavcopy_store as store


DAY = date(2026, 10, 9)
HEADER = (
    "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,"
    "LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY\n"
)
GOOD = (HEADER + ("TCS,EQ,09-Oct-2026,100,101,99,100,500\n" * 100)).encode()


def test_good_nse_download_publishes_via_atomic_rename(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: SimpleNamespace(status_code=200, content=GOOD),
    )
    target = store._day_path(DAY)
    original_replace = store.os.replace
    saw = []

    def inspect_atomic(src, dst):
        assert Path(src).is_file()
        assert Path(dst) == target
        assert not target.exists()
        assert Path(src).read_bytes() == GOOD
        saw.append(True)
        return original_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", inspect_atomic)
    assert store._download_day(DAY, retries=0) is True
    assert target.read_bytes() == GOOD
    assert saw == [True]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["09102026.csv"]


def test_incomplete_existing_recent_archive_is_replaced_only_after_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    target = store._day_path(DAY)
    target.write_bytes(b"incomplete download")
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: SimpleNamespace(status_code=200, content=GOOD),
    )
    assert store._download_day(DAY, retries=0) is True
    assert target.read_bytes() == GOOD


def test_bad_new_download_never_creates_final_session_file(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: SimpleNamespace(status_code=200, content=b"INVALID" * 200),
    )
    assert store._download_day(DAY, retries=0) is False
    assert not store._day_path(DAY).exists()
    assert list(tmp_path.iterdir()) == []


def test_wrong_embedded_date_never_publishes_as_another_exchange_session(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    renamed = GOOD.replace(b"09-Oct-2026", b"08-Oct-2026")
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: SimpleNamespace(status_code=200, content=renamed),
    )
    assert store._download_day(DAY, retries=0) is False
    assert not store._day_path(DAY).exists()


def test_already_valid_csv_is_not_redownloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    store._day_path(DAY).write_bytes(GOOD)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Valid local archive should skip network")
        ),
    )
    assert store._download_day(DAY, retries=0) is True


def test_invalid_exchange_response_uses_retry_backoff_before_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    replies = iter([
        SimpleNamespace(status_code=200, content=b"INVALID" * 200),
        SimpleNamespace(status_code=200, content=GOOD),
    ])
    delays = []
    monkeypatch.setattr("requests.get", lambda *a, **k: next(replies))
    monkeypatch.setattr(store.time, "sleep", lambda seconds: delays.append(seconds))
    assert store._download_day(DAY, retries=1) is True
    assert store._day_path(DAY).read_bytes() == GOOD
    assert delays == [1.5]


def test_missing_installed_external_runtime_is_never_recreated_by_download(monkeypatch, tmp_path):
    root = tmp_path / "vanished-volume" / "QuantTerm" / "runtime"
    target_dir = root / "logs" / "bhav"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    monkeypatch.setattr(store, "_BHAV_DIR", target_dir)
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: SimpleNamespace(status_code=200, content=GOOD),
    )
    assert store._download_day(DAY, retries=0) is False
    assert not root.exists()
    assert not target_dir.exists()


def test_missing_installed_external_runtime_is_never_recreated_by_pickle_writer(monkeypatch, tmp_path):
    root = tmp_path / "vanished-volume" / "QuantTerm" / "runtime"
    folder = root / "logs" / "bhav"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    monkeypatch.setattr(store, "_PKL", folder / "store_cache.pkl")
    monkeypatch.setattr(store, "_store", {"TCS": "cached"})
    store._save_pkl()  # the writer logs failure but must not create any path
    assert not root.exists()


def test_pickle_cache_publishes_with_unique_atomic_temp_file(monkeypatch, tmp_path):
    monkeypatch.delenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", raising=False)
    destination = tmp_path / "bhav" / "store_cache.pkl"
    monkeypatch.setattr(store, "_PKL", destination)
    monkeypatch.setattr(store, "_store", {"TCS": "cached"})
    monkeypatch.setattr(store, "_store_last_day", DAY)
    monkeypatch.setattr(store, "_store_sessions", 1)
    observed = []
    actual_replace = store.os.replace

    def record_replace(src, dst):
        assert Path(src).parent == destination.parent
        assert Path(src).name.startswith(".store_cache-")
        assert Path(dst) == destination
        assert Path(src).is_file()
        observed.append(Path(src).name)
        return actual_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", record_replace)
    store._save_pkl()
    store._save_pkl()
    assert destination.is_file()
    assert len(set(observed)) == 2
    assert sorted(p.name for p in destination.parent.iterdir()) == ["store_cache.pkl"]


def test_installed_runtime_bhavcopy_writer_rejects_symlink_to_internal_disk(monkeypatch, tmp_path):
    import pytest

    root = tmp_path / "QuantTerm" / "runtime"
    root.mkdir(parents=True)
    external = tmp_path / "local-internal-dir"
    external.mkdir()
    (root / "logs").symlink_to(external, target_is_directory=True)
    directory = root / "logs" / "bhav"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    with pytest.raises(RuntimeError, match="no longer matches"):
        store._prepare_bhav_write_directory(directory)
    assert not (external / "bhav").exists()
