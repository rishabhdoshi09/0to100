"""Tests for evidence-backed security identity ledger (no invented transitions)."""
from data.security_identity import (
    build_identity_ledger,
    resolve_as_of,
    write_identity_ledger,
    load_identity_ledger,
)


def test_build_identity_does_not_invent_delisting():
    equity = [{
        "security_id": "isin:INE001A01036",
        "symbol": "RELIANCE",
        "series": "EQ",
        "isin": "INE001A01036",
        "valid_from": "1977-11-29",
        "valid_to": None,
        "listing_date": "1977-11-29",
        "delisting_date": None,
        "tradable": True,
        "provenance": "nse_equity_l",
    }]
    ledger = build_identity_ledger(equity, [])
    assert ledger["completeness"]["has_official_delistings"] is False
    assert ledger["completeness"]["symbol_lineage_complete"] is False
    assert ledger["securities"][0]["delisting_date"] is None


def test_symbol_change_closes_old_interval_only_with_evidence():
    equity = [
        {
            "security_id": "isin:INE123",
            "symbol": "OLDCO",
            "series": "EQ",
            "isin": "INE123",
            "valid_from": "2010-01-01",
            "valid_to": None,
            "listing_date": "2010-01-01",
            "delisting_date": None,
            "tradable": True,
            "provenance": "nse_equity_l",
        },
        {
            "security_id": "isin:INE123",
            "symbol": "NEWCO",
            "series": "EQ",
            "isin": "INE123",
            "valid_from": "2020-06-01",
            "valid_to": None,
            "listing_date": "2020-06-01",
            "delisting_date": None,
            "tradable": True,
            "provenance": "nse_equity_l",
        },
    ]
    changes = [{
        "event": "symbol_change",
        "old_symbol": "OLDCO",
        "new_symbol": "NEWCO",
        "effective_date": "2020-06-01",
        "company": "Example",
        "provenance": "nse_symbolchange",
    }]
    ledger = build_identity_ledger(equity, changes)
    by = {r["symbol"]: r for r in ledger["securities"]}
    assert by["OLDCO"]["valid_to"] == "2020-06-01"
    assert by["OLDCO"]["tradable"] is False
    assert resolve_as_of("OLDCO", "2019-01-01", ledger)["status"] == "OK"
    assert resolve_as_of("OLDCO", "2020-06-01", ledger)["status"] == "SYMBOL_ENDED"
    assert resolve_as_of("NEWCO", "2021-01-01", ledger)["status"] == "OK"
    assert resolve_as_of("MISSING", "2021-01-01", ledger)["status"] == "UNKNOWN"


def test_roundtrip_write_load(tmp_path):
    equity = [{
        "security_id": "isin:INE002",
        "symbol": "TCS",
        "series": "EQ",
        "isin": "INE002",
        "valid_from": "2004-08-25",
        "valid_to": None,
        "listing_date": "2004-08-25",
        "delisting_date": None,
        "tradable": True,
        "provenance": "nse_equity_l",
    }]
    ledger = build_identity_ledger(equity, [])
    p = write_identity_ledger(ledger, path=tmp_path / "security_identity.json")
    loaded = load_identity_ledger(p)
    assert loaded["securities"][0]["symbol"] == "TCS"
