"""QT-PRA-085: a restarted process must not re-read every official archive.

On the owner's Mac each QuantTerm process spent ~18 s re-validating all 1,754
bhavcopy archives on its first history check after every restart. Verdicts are
persisted by file identity and validator version; anything that changes is
re-validated, and nothing is trusted that the validator has not seen.
"""
from __future__ import annotations

import os
import time
from datetime import date

import pytest

from data import bhavcopy_store as BS


def _csv(day: str) -> bytes:
    lines = [
        "SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY",
        f"INFY, EQ, {day}, 100, 102, 99, 101, 150000",
        f"TCS, EQ, {day}, 200, 202, 199, 201, 100000",
    ]
    return ("\n".join(lines) + "\n").encode("latin-1")


def _age(path, seconds=60):
    t = time.time() - seconds
    os.utime(path, (t, t))


def _cold_process(monkeypatch):
    """Forget everything a fresh interpreter would not have in memory."""
    BS._validated_disk_file.cache_clear()
    monkeypatch.setattr(BS, "_VERDICTS", None)
    monkeypatch.setattr(BS, "_VERDICTS_SIG", None)
    monkeypatch.setattr(BS, "_PENDING_VERDICTS", {})
    monkeypatch.setattr(BS, "_SESSION_LISTING", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    bhav = tmp_path / "bhav"
    bhav.mkdir()
    monkeypatch.setattr(BS, "_BHAV_DIR", bhav)
    monkeypatch.setattr(BS, "_holidays", lambda: {"2026-10-20"})
    _cold_process(monkeypatch)
    calls = {"n": 0}
    real = BS.validate_archive_path

    def counting(path, day):
        calls["n"] += 1
        return real(path, day)

    monkeypatch.setattr(BS, "validate_archive_path", counting)
    for name, stamp in (("19102026", "19-Oct-2026"), ("21102026", "21-Oct-2026"), ("22102026", "19-Oct-2026")):
        f = bhav / f"{name}.csv"
        f.write_bytes(_csv(stamp))
        _age(f)
    _age(bhav)
    return bhav, calls


def test_a_cold_process_reuses_persisted_verdicts(store, monkeypatch):
    bhav, calls = store
    first = BS._dates_on_disk()
    assert first == [date(2026, 10, 19), date(2026, 10, 21)]  # 22nd is mislabelled
    assert calls["n"] == 3
    assert BS._verdict_cache_path().exists()

    _cold_process(monkeypatch)
    calls["n"] = 0
    assert BS._dates_on_disk() == first
    assert calls["n"] == 0  # nothing re-read, including the invalid verdict


def test_a_rewritten_archive_is_revalidated(store, monkeypatch):
    bhav, calls = store
    BS._dates_on_disk()
    target = bhav / "21102026.csv"
    target.write_bytes(_csv("18-Oct-2026") + b"x\n")  # in-place rewrite, now invalid
    _age(target)
    _cold_process(monkeypatch)
    calls["n"] = 0
    assert BS._dates_on_disk() == [date(2026, 10, 19)]
    assert calls["n"] == 1


def test_a_changed_validator_ignores_old_verdicts(store, monkeypatch):
    _bhav, calls = store
    BS._dates_on_disk()
    _cold_process(monkeypatch)
    monkeypatch.setattr(BS, "_validator_id", lambda: "a-different-validator")
    calls["n"] = 0
    BS._dates_on_disk()
    assert calls["n"] == 3


def test_just_modified_archives_are_never_persisted(store, monkeypatch):
    bhav, calls = store
    fresh = bhav / "23102026.csv"
    fresh.write_bytes(_csv("23-Oct-2026"))  # mtime = now
    _age(bhav)
    assert date(2026, 10, 23) in BS._dates_on_disk()
    import json

    persisted = json.loads(BS._verdict_cache_path().read_text())["entries"]
    assert "23102026.csv" not in persisted
    assert "19102026.csv" in persisted
