"""Paper journal concurrency, crash safety, and installed-runtime boundaries."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from product import autopilot_journal as journal


def test_concurrent_paper_cycles_do_not_lose_or_corrupt_history(tmp_path):
    target = tmp_path / "journal.json"
    def write_one(n):
        return journal.record_cycle(
            {"cycle_id": f"cycle-{n}", "as_of": "2026-10-13", "taken": []},
            path=target,
        )

    # Each worker acquires a separate flock descriptor and reads only while
    # holding that lock. The prior unlocked RMW lost concurrent cycles.
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write_one, range(48)))
    rows = journal.load_journal(target)["cycles"]
    assert len(rows) == 48
    assert {row["cycle_id"] for row in rows} == {
        f"cycle-{n}" for n in range(48)
    }
    assert journal.load_journal(target)["latest"] == rows[-1]
    assert not list(tmp_path.glob(".journal.json.*.tmp"))


def test_journal_retains_only_last_120_cycles(tmp_path):
    target = tmp_path / "journal.json"
    for n in range(123):
        journal.record_cycle({"cycle_id": str(n)}, path=target)
    rows = journal.load_journal(target)["cycles"]
    assert len(rows) == 120
    assert [row["cycle_id"] for row in rows] == [
        str(n) for n in range(3, 123)
    ]


def test_corrupt_journal_refuses_overwrite(tmp_path):
    target = tmp_path / "journal.json"
    target.write_text('{"cycles": [broken', encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.record_cycle({"cycle_id": "new"}, path=target)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".journal.json.*.tmp"))


def test_invalid_journal_structure_refuses_overwrite(tmp_path):
    target = tmp_path / "journal.json"
    target.write_text(json.dumps({"cycles": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid cycle structure"):
        journal.record_cycle({"cycle_id": "new"}, path=target)
    assert json.loads(target.read_text())["cycles"] == {"bad": True}


def test_failed_atomic_publication_preserves_previous_journal(tmp_path, monkeypatch):
    target = tmp_path / "journal.json"
    journal.record_cycle({"cycle_id": "before"}, path=target)
    before = target.read_bytes()
    def fail_replace(_src, _dst):
        raise OSError("simulated interrupted rename")
    monkeypatch.setattr(journal.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted rename"):
        journal.record_cycle({"cycle_id": "after"}, path=target)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".journal.json.*.tmp"))


def test_journal_path_tracks_runtime_root_after_import(tmp_path, monkeypatch):
    root = tmp_path / "external-runtime"
    root.mkdir()
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.delenv("QT_PAPER_AUTOPILOT_JOURNAL", raising=False)
    expected = root / "logs" / "product" / "paper_autopilot_journal.json"
    assert journal.journal_path() == expected
    journal.record_cycle({"cycle_id": "root-1"})
    assert expected.exists()


def test_unmounted_runtime_fails_before_creating_any_directory(tmp_path, monkeypatch):
    root = tmp_path / "unmounted" / "runtime"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    monkeypatch.delenv("QT_PAPER_AUTOPILOT_JOURNAL", raising=False)
    with pytest.raises(RuntimeError, match="runtime root is missing"):
        journal.record_cycle({"cycle_id": "should-not-write"})
    assert not root.parent.exists()


def test_installed_journal_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "logs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    monkeypatch.setenv("QT_RUNTIME_ROOT_REQUIRE_EXISTING", "1")
    with pytest.raises(RuntimeError, match="outside the installed persistent runtime"):
        journal.record_cycle({"cycle_id": "should-not-write"})
    assert not (outside / "product").exists()
