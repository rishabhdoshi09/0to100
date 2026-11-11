"""Official filings are acquired for Opportunities cards that have none
(owner's Mac 2026-11-11: 85 cards "Unmeasured", warehouse stuck at 375)."""
from __future__ import annotations


def _patch(monkeypatch, have=("TCS",), parsed=1):
    import fundamentals.fetcher as F
    import product.pit_backfill as B
    import product.pit_warehouse as W

    calls = []
    monkeypatch.setattr(F, "_network_backfill_allowed", lambda: True)
    monkeypatch.setattr(W, "get_evidence", lambda s, **kw: [{"x": 1}] if s in have else [])
    monkeypatch.setattr(B, "backfill_structured_financials",
                        lambda s, **kw: (calls.append((s, kw)), {"parsed": parsed})[1])
    return F, calls


def test_only_symbols_without_filings_are_fetched_and_bounded(monkeypatch):
    F, calls = _patch(monkeypatch)
    report = F.acquire_official_for_symbols(["tcs", "RPEL", "RPEL", "", "PNB", "SPARC"], limit=2)
    assert report["missing"] == 3 and report["attempted"] == 2
    assert [c[0] for c in calls] == ["RPEL", "PNB"]
    assert report["acquired"] == ["RPEL", "PNB"]
    assert calls[0][1]["sleep_s"] > 0                      # polite pacing


def test_unparsed_or_raising_backfill_counts_as_failed(monkeypatch):
    F, _ = _patch(monkeypatch, parsed=0)
    assert F.acquire_official_for_symbols(["RPEL"])["failed"] == ["RPEL"]
    import product.pit_backfill as B
    monkeypatch.setattr(B, "backfill_structured_financials",
                        lambda s, **kw: (_ for _ in ()).throw(RuntimeError("nse 403")))
    assert F.acquire_official_for_symbols(["PNB"])["failed"] == ["PNB"]


def test_network_gate_is_respected(monkeypatch):
    import fundamentals.fetcher as F

    monkeypatch.setattr(F, "_network_backfill_allowed", lambda: False)
    assert F.acquire_official_for_symbols(["RPEL"])["skipped"] == "network_disabled"


def test_refresh_pass_acquires_for_current_cards(monkeypatch):
    import product.recommendations_store as RS
    import scan.long_term_service as LT

    F, calls = _patch(monkeypatch)
    monkeypatch.setattr(RS, "load_recommendations", lambda *a, **k: {"categories": [
        {"id": "momentum_breakouts", "cards": [{"symbol": "RPEL"}, {"symbol": "TCS"}]}]})
    report = LT.acquire_card_fundamentals(limit=5)
    assert [c[0] for c in calls] == ["RPEL"] and report["acquired"] == ["RPEL"]
