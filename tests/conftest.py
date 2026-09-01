"""
Pytest collection policy.

The CANONICAL network-free unit suite is simply:

    python -m pytest

`tests/integration/` is EXCLUDED from that default run by classification (not an ad-hoc
`--ignore`): it holds tests whose import chain reaches heavy, environment-dependent
operational modules (e.g. `scan/*`, which make lazy data/network calls at import time
and stall without network). Those are integration tests, not deterministic unit tests.

To run the integration suite explicitly (may be slow / need network):

    QT_INTEGRATION=1 python -m pytest tests/integration

`collect_ignore` prevents pytest from even importing the integration directory during
the default run, so the network-free suite cannot stall on their import chain.
"""
import os
from datetime import datetime, timezone

import pytest

# During the default (network-free) run, do not collect/import tests/integration.
collect_ignore = [] if os.getenv("QT_INTEGRATION") else ["integration"]

_LONG_TERM_PROJECTOR = None


def pytest_sessionstart(session):
    """Freeze the canonical saved-scan projector identity for leak detection.

    Tests may monkeypatch it through pytest's monkeypatch fixture, but the fixture
    must restore it at teardown. A direct assignment leak is a test-isolation bug
    because later tests (and long-lived processes) would see altered behavior.
    """
    global _LONG_TERM_PROJECTOR
    from scan import long_term_service
    _LONG_TERM_PROJECTOR = long_term_service.technical_rows_from_market_scan


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail at the test that leaves the canonical projector mutated."""
    yield
    if _LONG_TERM_PROJECTOR is None:
        return
    from scan import long_term_service
    current = long_term_service.technical_rows_from_market_scan
    if current is not _LONG_TERM_PROJECTOR:
        pytest.fail(
            "test leaked scan.long_term_service.technical_rows_from_market_scan "
            f"after teardown: {item.nodeid}",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def isolate_mutable_runtime_state(tmp_path_factory, monkeypatch, request):
    """Hermetic suite: never inherit warmed bhavcopy, analog corpus, or paper memory."""
    from data.bhavcopy_store import reset_in_memory_store
    from research.market_memory import reset_analog_corpus_cache

    reset_in_memory_store()
    reset_analog_corpus_cache()
    paper_mem = tmp_path_factory.mktemp("paper_memory") / "paper_memory.json"
    monkeypatch.setenv("QT_PAPER_MEMORY", str(paper_mem))
    auto_journal = tmp_path_factory.mktemp("autopilot_journal") / "journal.json"
    monkeypatch.setenv("QT_PAPER_AUTOPILOT_JOURNAL", str(auto_journal))
    policies = tmp_path_factory.mktemp("learning_policies") / "policies.json"
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(policies))
    counter = tmp_path_factory.mktemp("counterfactuals") / "cf.jsonl"
    monkeypatch.setenv("QT_COUNTERFACTUALS", str(counter))
    taken = tmp_path_factory.mktemp("taken_evidence") / "taken.jsonl"
    monkeypatch.setenv("QT_TAKEN_EVIDENCE", str(taken))
    ingested = tmp_path_factory.mktemp("learning_ingested") / "ingested.json"
    monkeypatch.setenv("QT_LEARNING_INGESTED", str(ingested))

    # This legacy smart-acquire test intentionally writes an Aug-26 cache and
    # asserts that the 3-day filings lane is still fresh. Without an explicit
    # clock it becomes date-dependent and started failing on Aug-29 even though
    # production correctly treats >72h exchange data as stale. Freeze ONLY that
    # test; never weaken the live freshness policy to satisfy a calendar test.
    if request.node.name == "test_smart_acquire_skips_fresh_lanes":
        import product.due_diligence.acquire as acquire_module

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                fixed = cls(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)
                return fixed if tz is not None else fixed.replace(tzinfo=None)

        monkeypatch.setattr(acquire_module, "datetime", _FrozenDateTime)

    yield
    reset_in_memory_store()
    reset_analog_corpus_cache()
