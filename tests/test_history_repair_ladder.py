"""Source-authority tests for whole-market historical repair.

The current contract is explicit:
- With Kite authority active, Kite supplies Kite-covered OHLCV. A partial or
  failed Kite repair stays visible; NSE/Yahoo do not silently fill the same gap.
- Without Kite authority, QuantTerm retains an offline continuity ladder:
  official NSE bhavcopy first, then Yahoo as last resort.

This prevents one scan from silently mixing equivalent market-history fields
from multiple providers while preserving operation when broker data is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scan.bulk_fetcher as BF
from data.acquisition import ACQUIRED


def _frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [10.0] * rows,
        "high": [11.0] * rows,
        "low": [9.0] * rows,
        "close": [10.5] * rows,
        "volume": [1000] * rows,
    })


@pytest.fixture(autouse=True)
def _clean_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(BF, "_kite_cache", {})
    monkeypatch.setattr(BF, "_yf_cache", {})
    monkeypatch.setattr(BF, "_bhav_symbols", lambda: set())
    monkeypatch.setattr(BF, "_adopt_active_kite_snapshot", lambda symbols: 0)


def test_kite_authority_supplies_the_gap_when_it_can(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(
        BF,
        "_repair_via_kite",
        lambda missing, client=None, now=None: (
            {s: _frame() for s in missing},
            {"attempted": len(missing), "failed": 0},
        ),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_bhavcopy",
        lambda missing: (_ for _ in ()).throw(AssertionError("NSE fallback forbidden")),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_yfinance",
        lambda missing: (_ for _ in ()).throw(AssertionError("Yahoo fallback forbidden")),
    )

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["loaded"] == 2
    assert report["unresolved"] == 0
    assert report["sources"] == ["zerodha_kite_data_only"]
    assert report["state"] == ACQUIRED
    assert {"AAA", "BBB"} <= set(BF._kite_cache)


def test_kite_partial_result_stays_visible_without_same_field_substitution(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    reached = {"nse": 0, "yf": 0}
    monkeypatch.setattr(
        BF,
        "_repair_via_kite",
        lambda missing, client=None, now=None: (
            {"AAA": _frame()},
            {"attempted": len(missing), "failed": 1},
        ),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_bhavcopy",
        lambda missing: reached.__setitem__("nse", reached["nse"] + 1),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_yfinance",
        lambda missing: reached.__setitem__("yf", reached["yf"] + 1),
    )

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["loaded"] == 1
    assert report["unresolved"] == 1
    assert report["sources"] == ["zerodha_kite_data_only"]
    assert reached == {"nse": 0, "yf": 0}


def test_kite_total_failure_is_reported_without_cross_source_replacement(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(
        BF,
        "_repair_via_kite",
        lambda missing, client=None, now=None: (_ for _ in ()).throw(
            RuntimeError("Kite session rejected")
        ),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_bhavcopy",
        lambda missing: (_ for _ in ()).throw(AssertionError("NSE fallback forbidden")),
    )
    monkeypatch.setattr(
        BF,
        "_repair_via_yfinance",
        lambda missing: (_ for _ in ()).throw(AssertionError("Yahoo fallback forbidden")),
    )

    report = BF.backfill_missing(["AAA"])

    assert report["loaded"] == 0
    assert report["unresolved"] == 1
    assert report["state"] == "DATA_UNAVAILABLE"
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["source"] == "zerodha_kite_data_only"


def test_kite_snapshot_hit_is_a_cheap_no_op(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(BF, "_kite_cache", {"AAA": _frame(), "BBB": _frame()})
    called = []
    monkeypatch.setattr(
        BF,
        "_repair_via_kite",
        lambda *a, **k: called.append(1),
    )

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["missing"] == 0
    assert report["loaded"] == 0
    assert called == []


def test_no_kite_authority_uses_official_store_first(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: False)
    reached = {"yf": 0}
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {s: _frame() for s in missing})
    monkeypatch.setattr(
        BF,
        "_repair_via_yfinance",
        lambda missing: reached.__setitem__("yf", reached["yf"] + 1) or {},
    )

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["loaded"] == 2
    assert report["unresolved"] == 0
    assert report["sources"] == ["nse_bhavcopy"]
    assert reached["yf"] == 0


def test_no_kite_authority_uses_yahoo_only_for_official_remainder(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {"AAA": _frame()})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {"BBB": _frame()})

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["loaded"] == 2
    assert report["unresolved"] == 0
    assert report["sources"] == ["nse_bhavcopy", "yfinance_daily"]


def test_no_kite_authority_keeps_unresolved_gap_visible(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {"AAA": _frame()})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {})

    report = BF.backfill_missing(["AAA", "BBB"])

    assert report["loaded"] == 1
    assert report["unresolved"] == 1


def test_no_kite_authority_total_failure_is_reported(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(
        BF,
        "_repair_via_bhavcopy",
        lambda missing: (_ for _ in ()).throw(LookupError("store is current")),
    )
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {})

    report = BF.backfill_missing(["AAA"])

    assert report["loaded"] == 0
    assert report["unresolved"] == 1
    assert report["state"] == "DATA_UNAVAILABLE"
    assert [row["source"] for row in report["attempts"]] == [
        "nse_bhavcopy",
        "yfinance_daily",
    ]


def test_public_source_contribution_is_named_when_kite_is_unavailable(monkeypatch):
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {s: _frame() for s in missing})

    report = BF.backfill_missing(["AAA"])

    assert report["sources"] == ["yfinance_daily"]
    assert report["loaded"] == 1


def test_an_up_to_date_store_says_so_instead_of_re_downloading(monkeypatch):
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: True)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["ZZZ"])
    monkeypatch.setattr(BS, "latest_two_eq_sessions", lambda: ["2099-01-01", "2099-01-02"])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))
    monkeypatch.setattr(CAL, "load_holidays", lambda *a, **k: set())
    monkeypatch.setattr(
        CAL,
        "latest_required_session",
        lambda *a, **k: __import__("datetime").date(2098, 1, 1),
    )

    with pytest.raises(LookupError):
        BF._repair_via_bhavcopy(["AAA"])
    assert built == []


def test_a_behind_store_is_brought_up_to_date_before_giving_up(monkeypatch):
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: True)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["ZZZ"])
    monkeypatch.setattr(BS, "latest_two_eq_sessions", lambda: ["2020-01-01"])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))
    monkeypatch.setattr(BS, "get_ohlcv", lambda symbol: _frame())
    monkeypatch.setattr(CAL, "load_holidays", lambda *a, **k: set())
    monkeypatch.setattr(
        CAL,
        "latest_required_session",
        lambda *a, **k: __import__("datetime").date(2026, 9, 11),
    )

    frames = BF._repair_via_bhavcopy(["AAA"])

    assert built == [1]
    assert set(frames) == {"AAA"}


def test_symbols_already_in_the_store_are_not_re_fetched(monkeypatch):
    from data import bhavcopy_store as BS

    monkeypatch.setattr(BS, "store_symbols", lambda: ["AAA"])
    assert BF._repair_via_bhavcopy(["AAA"]) == {}


def test_an_unbuilt_store_is_a_bootstrap_problem_not_a_scan_problem(monkeypatch):
    from data import bhavcopy_store as BS

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: False)
    monkeypatch.setattr(BS, "store_symbols", lambda: [])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))

    with pytest.raises(LookupError, match="bootstrap"):
        BF._repair_via_bhavcopy(["AAA"])
    assert built == []
