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


@pytest.fixture(autouse=True)
def isolate_mutable_runtime_state(tmp_path_factory, monkeypatch, request):
    """Hermetic suite: never inherit warmed bhavcopy, analog corpus, or paper memory."""
    from data.bhavcopy_store import reset_in_memory_store
    from research.market_memory import reset_analog_corpus_cache

    reset_in_memory_store()
    reset_analog_corpus_cache()
    paper_mem = tmp_path_factory.mktemp("paper_memory") / "paper_memory.json"
    monkeypatch.setenv("QT_PAPER_MEMORY", str(paper_mem))

    # Some import-safety tests deliberately reload modules. test_universal_scan
    # imports run_long_term_scan at collection time, so without this rebind it can
    # retain a function whose __globals__ belongs to an obsolete module instance
    # while string-based monkeypatches target the current sys.modules instance.
    # Keep the test exercising the current production module; do not weaken its
    # saved-scan/no-secondary-walk assertion.
    if request.module.__name__.endswith("test_universal_scan"):
        from scan import long_term_service as current_long_term_service

        monkeypatch.setattr(
            request.module,
            "run_long_term_scan",
            current_long_term_service.run_long_term_scan,
            raising=True,
        )

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
