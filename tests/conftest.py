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

import pytest

# During the default (network-free) run, do not collect/import tests/integration.
collect_ignore = [] if os.getenv("QT_INTEGRATION") else ["integration"]


@pytest.fixture(autouse=True)
def isolate_mutable_runtime_state():
    """Hermetic suite: never inherit warmed bhavcopy or analog corpus from prior tests."""
    from data.bhavcopy_store import reset_in_memory_store
    from research.market_memory import reset_analog_corpus_cache

    reset_in_memory_store()
    reset_analog_corpus_cache()
    yield
    reset_in_memory_store()
    reset_analog_corpus_cache()
