"""Getting the data is the product's job, and it must say how it got it.

Two failures this contract exists to prevent, both of which the codebase had:

* A parser that stops matching its page returns zero rows, and zero rows read
  as "there is nothing today". The desk cannot tell a quiet market from a
  broken scraper, so it believes the broken scraper.
* Fallback data looks exactly like primary data once it is on disk. Nothing
  downstream can tell whether a number came from the exchange this morning or
  from a CSV shipped with the checkout months ago.
"""
from __future__ import annotations

import json

import pytest

from data.acquisition import (
    ACQUIRED,
    DATA_UNAVAILABLE,
    EMPTY,
    ERROR,
    HTTP_ERROR,
    LAST_KNOWN_GOOD_STATE,
    NOT_PRESENT,
    OK,
    PARSER_CHANGED,
    REFUSED_BYPASS,
    SOURCE_CONFLICT,
    TIMEOUT,
    UNREACHABLE,
    Source,
    SourceTier,
    Validation,
    acquire,
    read_provenance,
)


def _src(name, tier, fetch, validate=None, **kw):
    return Source(name, tier, fetch, validate, **kw)


def _rows(n=3):
    return lambda: [{"symbol": f"S{i}"} for i in range(n)]


def _raises(exc):
    def fetch():
        raise exc
    return fetch


# ── the happy path ─────────────────────────────────────────────────────────
def test_the_first_working_source_wins_and_says_so():
    result = acquire("d", [
        _src("official", SourceTier.OFFICIAL_API, _rows(3)),
        _src("backup", SourceTier.REPUTABLE_PUBLIC, _rows(9)),
    ], persist=False)
    assert result.state == ACQUIRED
    assert result.source == "official"
    assert result.fallback_level == 0
    assert result.is_primary
    assert result.record_count == 3
    assert result.content_hash


def test_a_later_source_is_recorded_as_a_fallback_not_as_primary():
    result = acquire("d", [
        _src("official", SourceTier.OFFICIAL_API, _raises(ConnectionError("down"))),
        _src("backup", SourceTier.REPUTABLE_PUBLIC, _rows(2)),
    ], persist=False)
    assert result.state == ACQUIRED
    assert result.source == "backup"
    assert result.fallback_level == 1
    assert not result.is_primary


# ── failure classification: the point of the whole exercise ────────────────
@pytest.mark.parametrize("exc,expected", [
    (ConnectionError("no route"), UNREACHABLE),
    (TimeoutError("timed out"), TIMEOUT),
    (FileNotFoundError("no such file"), NOT_PRESENT),
    (ValueError("something else"), ERROR),
])
def test_failures_are_classified_not_collapsed(exc, expected):
    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, _raises(exc)),
        _src("b", SourceTier.REPUTABLE_PUBLIC, _rows(1)),
    ], persist=False)
    assert result.attempts[0].outcome == expected


def test_an_empty_answer_is_not_a_success():
    """Zero records is a claim about the world; it needs a source to survive."""
    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, lambda: []),
        _src("b", SourceTier.REPUTABLE_PUBLIC, _rows(4)),
    ], persist=False)
    assert result.attempts[0].outcome == EMPTY
    assert result.source == "b"


def test_a_schema_change_is_distinguishable_from_an_empty_market():
    def moved_columns():
        return {"unexpected": "shape"}

    def validator(payload):
        if "rows" not in payload:
            return Validation.bad(PARSER_CHANGED, "no rows key")
        return Validation.good(len(payload["rows"]))

    result = acquire("d", [
        _src("scraper", SourceTier.PUBLIC_SCRAPE, moved_columns, validator),
        _src("official", SourceTier.OFFICIAL_API, _rows(2)),
    ], persist=False)
    assert result.attempts[0].outcome == PARSER_CHANGED
    assert result.state == ACQUIRED
    assert result.parser_changed, "a covered-for parser break must still be visible"


def test_a_validator_that_throws_is_a_parser_problem_not_a_crash():
    def bad_validator(payload):
        raise KeyError("column moved")

    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, _rows(1), bad_validator),
        _src("b", SourceTier.REPUTABLE_PUBLIC, _rows(1)),
    ], persist=False)
    assert result.attempts[0].outcome == PARSER_CHANGED
    assert result.source == "b"


# ── exhaustion ─────────────────────────────────────────────────────────────
def test_every_source_failing_is_stated_not_returned_as_empty_data():
    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, _raises(ConnectionError("x"))),
        _src("b", SourceTier.REPUTABLE_PUBLIC, lambda: []),
    ], persist=False)
    assert result.state == DATA_UNAVAILABLE
    assert result.value is None
    assert not result.ok
    assert [a.outcome for a in result.attempts] == [UNREACHABLE, EMPTY]


def test_cache_tier_is_reported_as_last_known_good_not_as_fresh():
    result = acquire("d", [
        _src("live", SourceTier.OFFICIAL_API, _raises(ConnectionError("x"))),
        _src("cache", SourceTier.LAST_KNOWN_GOOD, _rows(5)),
    ], persist=False)
    assert result.state == LAST_KNOWN_GOOD_STATE
    assert result.ok, "stale data is usable — it just must not claim to be fresh"


# ── access boundaries ──────────────────────────────────────────────────────
def test_a_source_needing_a_bypass_is_never_called():
    called = []

    def must_not_run():
        called.append(1)
        return _rows(1)()

    result = acquire("d", [
        _src("paywalled", SourceTier.REPUTABLE_PUBLIC, must_not_run, requires_bypass=True),
        _src("public", SourceTier.PUBLIC_SCRAPE, _rows(1)),
    ], persist=False)
    assert called == [], "a source behind an access control must not be fetched"
    assert result.attempts[0].outcome == REFUSED_BYPASS
    assert result.source == "public"


# ── disagreement ───────────────────────────────────────────────────────────
def test_two_trusted_sources_disagreeing_is_a_conflict_not_a_coin_toss():
    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, lambda: {"close": 100.0}),
        _src("b", SourceTier.ALTERNATE_AUTHORITATIVE, lambda: {"close": 141.0}),
    ], corroborate=lambda x, y: abs(x["close"] - y["close"]) < 1.0, persist=False)
    assert result.state == SOURCE_CONFLICT
    assert result.value is None, "a conflict must not silently return one side"
    assert result.conflict["a"] == "a"
    assert result.conflict["b"] == "b"


def test_agreeing_sources_pass_corroboration():
    result = acquire("d", [
        _src("a", SourceTier.OFFICIAL_API, lambda: {"close": 100.0}),
        _src("b", SourceTier.ALTERNATE_AUTHORITATIVE, lambda: {"close": 100.2}),
    ], corroborate=lambda x, y: abs(x["close"] - y["close"]) < 1.0, persist=False)
    assert result.state == ACQUIRED
    assert result.source == "a"


# ── provenance ─────────────────────────────────────────────────────────────
def test_provenance_records_what_was_used_and_what_was_tried(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    acquire("prices", [
        _src("official", SourceTier.OFFICIAL_API, _raises(ConnectionError("down")),
             identifier="https://example.invalid/api"),
        _src("cache", SourceTier.LAST_KNOWN_GOOD, _rows(7)),
    ])
    stored = read_provenance("prices")
    assert stored["source"] == "cache"
    assert stored["state"] == LAST_KNOWN_GOOD_STATE
    assert stored["fallback_level"] == 1
    assert stored["record_count"] == 7
    assert stored["content_hash"]
    assert [a["outcome"] for a in stored["attempts"]] == [UNREACHABLE, OK]


def test_provenance_appends_a_history_not_only_the_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    for _ in range(3):
        acquire("prices", [_src("a", SourceTier.OFFICIAL_API, _rows(1))])
    log = tmp_path / "logs" / "provenance" / "prices.jsonl"
    assert len([ln for ln in log.read_text().splitlines() if ln.strip()]) == 3


def test_provenance_failure_never_fails_the_acquisition(monkeypatch):
    import data.acquisition as A

    monkeypatch.setattr(A, "provenance_path", lambda d: (_ for _ in ()).throw(OSError("nope")))
    result = A.acquire("d", [_src("a", SourceTier.OFFICIAL_API, _rows(1))])
    assert result.state == ACQUIRED


def test_attempt_order_is_the_order_tried():
    result = acquire("d", [
        _src("first", SourceTier.OFFICIAL_API, _raises(ConnectionError("x"))),
        _src("second", SourceTier.OFFICIAL_FILE, lambda: []),
        _src("third", SourceTier.PUBLIC_SCRAPE, _rows(1)),
    ], persist=False)
    assert [a.source for a in result.attempts] == ["first", "second", "third"]
