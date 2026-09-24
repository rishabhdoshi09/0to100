"""Deprecated compatibility import for the canonical QuantTerm API runtime.

Runtime ownership moved to :mod:`api.runtime`.  This file intentionally contains
no implementation so there is only one API runtime architecture.
"""
from __future__ import annotations

import api.runtime as _canonical

for _name in dir(_canonical):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_canonical, _name)

app = _canonical.app
