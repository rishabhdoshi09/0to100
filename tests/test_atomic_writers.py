"""Durable JSON writers must survive concurrent writers.

Every writer used a fixed ``<name>.tmp`` path. Two concurrent writers (the scan
pipeline and the API's background rebuild; market_ops and the autonomy job
both running MARKET_SCAN) shared that inode: one writer's os.replace moved the
other's half-written file, and the loser raised FileNotFoundError -- an
HTTP 500 or a silently failed publication. No network.
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _race(monkeypatch, write, target: Path, writers: int = 6):
    """Hold every writer between 'temp written' and 'os.replace' so all collide."""
    barrier = threading.Barrier(writers)
    real_replace = os.replace

    def held_replace(src, dst, *a, **k):
        if Path(dst) == target:
            barrier.wait(timeout=15)
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", held_replace)
    with ThreadPoolExecutor(max_workers=writers) as pool:
        results = list(pool.map(write, range(writers)))
    return results


def test_concurrent_recommendation_publications_all_commit(tmp_path, monkeypatch):
    from product.recommendations_store import load_recommendations, save_recommendations

    target = tmp_path / "latest_recommendations.json"

    def write(i):
        save_recommendations({"schema_version": 4, "categories": [], "generation": i}, target)
        return "ok"

    assert _race(monkeypatch, write, target) == ["ok"] * 6
    assert load_recommendations(target)["generation"] in range(6)
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_scan_saves_all_commit(tmp_path, monkeypatch):
    from product.scan_store import save_scan

    target = tmp_path / "latest_momentum_scan.json"

    def write(i):
        save_scan({"schema_version": 2, "records": [], "provenance": {}, "generation": i}, target)
        return "ok"

    assert _race(monkeypatch, write, target) == ["ok"] * 6
    assert json.loads(target.read_text(encoding="utf-8"))["generation"] in range(6)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_write_leaves_the_previous_file_and_no_temp(tmp_path):
    from core.atomic_io import atomic_write_json

    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": 1})
    with pytest.raises(TypeError):
        atomic_write_json(target, {"v": object()}, default=None)
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_keeps_an_existing_files_permissions(tmp_path):
    from core.atomic_io import atomic_write_text

    target = tmp_path / "shared.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o640)
    atomic_write_text(target, "{\"a\": 1}")
    assert (target.stat().st_mode & 0o777) == 0o640


_FIXED_TEMP = re.compile(
    r"""with_suffix\(\s*(?:[\w.]+\.suffix\s*\+\s*)?["']\.tmp["']\s*\)"""
)


def test_no_production_writer_uses_a_fixed_temp_name():
    """Guard the class: a fixed <name>.tmp is shared by every concurrent writer."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".", "frontend/", "legacy_app", "ui/", "archive")) or "/." in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if _FIXED_TEMP.search(line):
                offenders.append(f"{rel}:{n}")
    assert offenders == [], "use core.atomic_io instead of a fixed .tmp path:\n" + "\n".join(offenders)


def test_shared_parse_is_reused_until_the_artifact_is_replaced(tmp_path):
    from core import json_cache
    from core.atomic_io import atomic_write_json

    json_cache.clear()
    target = tmp_path / "latest.json"
    atomic_write_json(target, {"generation": 1})
    calls = []

    def loader(path):
        calls.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    first = json_cache.read_json_shared(target, loader)
    assert json_cache.read_json_shared(target, loader) is first and len(calls) == 1
    atomic_write_json(target, {"generation": 2})          # same size, new inode
    assert json_cache.read_json_shared(target, loader) == {"generation": 2} and len(calls) == 2
    target.unlink()
    assert json_cache.read_json_shared(target, loader) is None and len(calls) == 2


def test_a_version_replaced_mid_parse_is_not_cached(tmp_path):
    from core import json_cache
    from core.atomic_io import atomic_write_json

    json_cache.clear()
    target = tmp_path / "latest.json"
    atomic_write_json(target, {"generation": 1})

    def racing_loader(path):
        value = json.loads(path.read_text(encoding="utf-8"))
        atomic_write_json(path, {"generation": 2})        # a writer lands during the parse
        return value

    assert json_cache.read_json_shared(target, racing_loader) == {"generation": 1}
    assert json_cache.read_json_shared(target) == {"generation": 2}
