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

# During the default (network-free) run, do not collect/import tests/integration.
collect_ignore = [] if os.getenv("QT_INTEGRATION") else ["integration"]
