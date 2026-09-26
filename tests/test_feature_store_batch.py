from __future__ import annotations

import pytest

from research import feature_store as fs


def test_feature_write_batch_preserves_write_once_and_durability(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feature_store.db")

    with fs.feature_write_batch():
        first = fs.snapshot("decision::a", "AAA", "DECISION", {"rsi": 50.0})
        second = fs.snapshot("decision::b", "BBB", "DECISION", {"rsi": 60.0})
        duplicate = fs.snapshot("decision::a", "AAA", "DECISION", {"rsi": 99.0})

    assert first["status"] == "frozen"
    assert second["status"] == "frozen"
    assert duplicate["status"] == "exists"

    a = fs.get_observation("decision::a")
    b = fs.get_observation("decision::b")
    assert a is not None and a["features"]["rsi"] == 50.0
    assert b is not None and b["features"]["rsi"] == 60.0


def test_feature_write_batch_rolls_back_on_outer_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feature_store.db")

    with pytest.raises(RuntimeError, match="abort board"):
        with fs.feature_write_batch():
            assert fs.snapshot(
                "decision::rollback",
                "AAA",
                "DECISION",
                {"rsi": 55.0},
            )["status"] == "frozen"
            raise RuntimeError("abort board")

    assert fs.get_observation("decision::rollback") is None


def test_nested_feature_write_batch_reuses_outer_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feature_store.db")

    with fs.feature_write_batch():
        with fs.feature_write_batch():
            assert fs.snapshot(
                "decision::nested",
                "AAA",
                "DECISION",
                {"rsi": 45.0},
            )["status"] == "frozen"

    assert fs.get_observation("decision::nested") is not None
