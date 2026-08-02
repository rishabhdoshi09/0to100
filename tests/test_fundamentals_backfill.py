"""Tests for resumable fundamentals backfill."""
from __future__ import annotations

from fundamentals.backfill import run_fundamentals_backfill, resolve_universe


def test_resolve_universe_nse_non_empty():
    syms = resolve_universe("nse")
    assert len(syms) > 100
    assert "RELIANCE" in syms


def test_backfill_skips_cached_and_records_success(tmp_path, monkeypatch):
    from fundamentals import backfill as bf

    monkeypatch.setattr(bf, "_STATE_PATH", tmp_path / "state.json")
    calls: list[str] = []

    def fake_fetch(sym: str, force: bool) -> dict:
        calls.append(sym)
        return {"symbol": sym, "about": "test co"}

    monkeypatch.setattr(
        "fundamentals.cache.FundamentalsCache.get",
        lambda self, sym: {"about": "cached"} if sym == "AAA" else None,
    )
    monkeypatch.setattr(
        "fundamentals.cache.FundamentalsCache.set",
        lambda self, sym, data: None,
    )

    report = run_fundamentals_backfill(
        symbols=["AAA", "BBB"],
        force=False,
        resume=False,
        delay_seconds=0,
        fetcher=fake_fetch,
    )
    assert report["succeeded_count"] >= 2
    assert "BBB" in calls
    assert "AAA" not in calls
