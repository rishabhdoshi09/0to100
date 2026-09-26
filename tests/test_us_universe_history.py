from __future__ import annotations

from data import us_universe as uu


def _symbols(prefix: str, n: int = 600) -> dict[str, str]:
    return {f"{prefix}{i:04d}": f"Company {i}" for i in range(n)}


def test_official_us_universe_archive_is_write_once(tmp_path, monkeypatch):
    monkeypatch.setattr(uu, "_HISTORY_DIR", tmp_path / "us_universe_history")

    first = _symbols("A")
    second = _symbols("B")

    path = uu._archive_official_snapshot(first, session_date="2026-09-25")
    assert path is not None and path.exists()
    assert uu.load_us_universe_snapshot("2026-09-25") == first

    # Same date can never be rewritten by a later refresh/race.
    uu._archive_official_snapshot(second, session_date="2026-09-25")
    assert uu.load_us_universe_snapshot("2026-09-25") == first
    assert uu.available_us_universe_snapshots() == ["2026-09-25"]


def test_us_universe_archive_refuses_curated_or_partial_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(uu, "_HISTORY_DIR", tmp_path / "us_universe_history")

    assert uu._archive_official_snapshot(
        {"AAPL": "Apple", "MSFT": "Microsoft"},
        session_date="2026-09-25",
    ) is None
    assert uu.available_us_universe_snapshots() == []


def test_official_cache_write_also_archives_pit_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(uu, "_CACHE_FILE", tmp_path / "us_symbols.json")
    monkeypatch.setattr(uu, "_HISTORY_DIR", tmp_path / "us_universe_history")
    monkeypatch.setattr(uu, "_snapshot_date_et", lambda: "2026-09-26")

    symbols = _symbols("C")
    uu._save_cache(symbols, source="NASDAQ_TRADER_SYMBOL_DIRECTORY")

    assert uu.load_us_universe_snapshot("2026-09-26") == symbols
    assert uu.available_us_universe_snapshots() == ["2026-09-26"]
