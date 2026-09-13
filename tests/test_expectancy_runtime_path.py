from __future__ import annotations

from pathlib import Path


def _close(engine) -> None:
    conn = getattr(engine, "_conn", None)
    if conn is not None:
        conn.close()


def test_default_expectancy_db_follows_runtime_root(tmp_path, monkeypatch):
    """Default durable expectancy state must never fall back into the checkout."""
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(runtime_root))

    from expectancy.expectancy_engine import ExpectancyEngine

    engine = ExpectancyEngine()
    try:
        expected = runtime_root / "logs" / "expectancy.db"
        assert Path(engine._db_path) == expected
        assert expected.exists()
    finally:
        _close(engine)


def test_explicit_expectancy_db_override_is_preserved(tmp_path, monkeypatch):
    """Callers that deliberately supply an isolated DB path retain that contract."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    explicit = tmp_path / "fixture" / "custom-expectancy.db"

    from expectancy.expectancy_engine import ExpectancyEngine

    engine = ExpectancyEngine(db_path=str(explicit))
    try:
        assert Path(engine._db_path) == explicit
        assert explicit.exists()
    finally:
        _close(engine)
