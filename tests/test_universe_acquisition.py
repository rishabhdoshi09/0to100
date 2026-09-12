"""The NSE universe comes off a ladder, and the desk records which rung.

The universe used to fall through three tiers logging "Tier 2 failed" and
nothing else. If the Kite cache was absent and NSE was unreachable the desk
quietly ran on a CSV shipped with the checkout, and no screen and no ledger
said so. A universe from months-old bundled data is a different product from a
universe pulled from the exchange this morning, and the difference has to be
visible.
"""
from __future__ import annotations

import pytest

from data import nse_universe as U
from data.acquisition import (
    ACQUIRED,
    EMPTY,
    LAST_KNOWN_GOOD_STATE,
    NOT_PRESENT,
    PARSER_CHANGED,
    read_provenance,
)


@pytest.fixture(autouse=True)
def _fresh_universe(monkeypatch, tmp_path):
    """The universe is process-cached; each test needs a clean load."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(U, "_universe_loaded", False)
    monkeypatch.setattr(U, "_cached_universe", [])
    monkeypatch.setattr(U, "_cached_names", {})
    monkeypatch.setattr(U, "_last_acquisition", None)
    # The instrument cross-check reaches for the broker; not what these test.
    monkeypatch.setattr(U, "_filter_to_instruments", lambda syms, _m: syms)


def _payload(symbols, schema_ok=True, detail=""):
    return {"symbols": list(symbols), "names": {}, "schema_ok": schema_ok,
            "detail": detail}


def test_the_broker_cache_is_preferred_when_it_is_there(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache", lambda: _payload(["AAA", "BBB"]))
    symbols = U.get_nse_universe()
    result = U.last_universe_acquisition()
    assert symbols == ["AAA", "BBB"]
    assert result.state == ACQUIRED
    assert result.source == "kite_instrument_cache"
    assert result.fallback_level == 0


def test_a_missing_cache_is_not_present_and_the_ladder_continues(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no cache")))
    monkeypatch.setattr(U, "_fetch_nse_equity_list", lambda: _payload(["CCC"]))
    assert U.get_nse_universe() == ["CCC"]
    result = U.last_universe_acquisition()
    assert result.attempts[0].outcome == NOT_PRESENT
    assert result.source == "nse_equity_list"


def test_a_moved_column_is_a_parser_change_not_an_empty_market(monkeypatch):
    """The defect class: a scraper breaks and the desk reads it as no data."""
    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: _payload([], schema_ok=False, detail="missing columns"))
    monkeypatch.setattr(U, "_fetch_nse_equity_list", lambda: _payload(["DDD"]))
    assert U.get_nse_universe() == ["DDD"]
    result = U.last_universe_acquisition()
    assert result.attempts[0].outcome == PARSER_CHANGED
    assert result.parser_changed, "the break must stay visible after recovery"


def test_a_source_that_parses_but_lists_nothing_is_empty_not_broken(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache", lambda: _payload([]))
    monkeypatch.setattr(U, "_fetch_nse_equity_list", lambda: _payload(["EEE"]))
    U.get_nse_universe()
    assert U.last_universe_acquisition().attempts[0].outcome == EMPTY


def test_running_on_bundled_data_is_reported_as_last_known_good(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("x")))
    monkeypatch.setattr(U, "_fetch_nse_equity_list",
                        lambda: (_ for _ in ()).throw(ConnectionError("nse down")))
    symbols = U.get_nse_universe()
    result = U.last_universe_acquisition()
    assert symbols, "the desk still has a universe"
    assert result.state == LAST_KNOWN_GOOD_STATE, (
        "but it must not claim the exchange gave it this"
    )
    assert result.fallback_level >= 2


def test_the_builtin_constant_is_the_last_rung_not_a_silent_default(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("x")))
    monkeypatch.setattr(U, "_fetch_nse_equity_list",
                        lambda: (_ for _ in ()).throw(ConnectionError("x")))
    monkeypatch.setattr(U, "_fetch_bundled_csv",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("x")))
    symbols = U.get_nse_universe()
    result = U.last_universe_acquisition()
    assert len(symbols) > 100
    assert result.source == "builtin_nifty500"
    assert result.state == LAST_KNOWN_GOOD_STATE


def test_the_universe_records_its_provenance_on_disk(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache", lambda: _payload(["AAA", "BBB", "CCC"]))
    U.get_nse_universe()
    stored = read_provenance(U.UNIVERSE_DATASET)
    assert stored["source"] == "kite_instrument_cache"
    assert stored["record_count"] == 3
    assert stored["parser_version"] == U.UNIVERSE_PARSER_VERSION
    assert stored["content_hash"]


def test_symbol_validation_still_applies_to_every_rung(monkeypatch):
    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: {"symbols": [s for s in ["AAA", "BBB"]], "names": {},
                                 "schema_ok": True, "detail": ""})
    assert all(U._is_valid_symbol(s) for s in U.get_nse_universe())


# ---------------------------------------------------------------------------
# "Whole market scanned" is a claim about a universe. The coverage evidence
# must therefore say WHICH universe, and how the desk got hold of it.
# ---------------------------------------------------------------------------
def test_coverage_evidence_names_the_universe_it_walked(monkeypatch):
    from scan.scan_coverage import ScanCoverageProbe, _universe_provenance

    monkeypatch.setattr(U, "_fetch_kite_cache", lambda: _payload(["AAA", "BBB"]))
    U.get_nse_universe()

    provenance = _universe_provenance()
    assert provenance["source"] == "kite_instrument_cache"
    assert provenance["state"] == ACQUIRED
    assert provenance["fallback_level"] == 0
    assert provenance["content_hash"]

    probe = ScanCoverageProbe(["AAA", "BBB"], instrumented=True,
                              universe_provenance=provenance)
    summary = probe.finalize()["summary"]
    assert summary["universe_provenance"]["source"] == "kite_instrument_cache"


def test_a_scan_on_bundled_data_says_so_in_its_coverage(monkeypatch):
    """The dishonest case: a full-market claim standing on months-old data."""
    from scan.scan_coverage import _universe_provenance

    monkeypatch.setattr(U, "_fetch_kite_cache",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("x")))
    monkeypatch.setattr(U, "_fetch_nse_equity_list",
                        lambda: (_ for _ in ()).throw(ConnectionError("x")))
    U.get_nse_universe()

    provenance = _universe_provenance()
    assert provenance["state"] == LAST_KNOWN_GOOD_STATE
    assert provenance["fallback_level"] >= 2


def test_missing_provenance_is_unrecorded_not_invented(monkeypatch):
    from scan.scan_coverage import _universe_provenance

    monkeypatch.setattr(U, "last_universe_acquisition", lambda: None)
    assert _universe_provenance()["state"] == "UNRECORDED"


def test_provenance_lookup_never_breaks_a_scan(monkeypatch):
    from scan.scan_coverage import _universe_provenance

    def explode():
        raise RuntimeError("provenance store is on fire")

    monkeypatch.setattr(U, "last_universe_acquisition", explode)
    assert _universe_provenance()["state"] == "UNRECORDED"
