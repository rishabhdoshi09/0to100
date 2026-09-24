"""Deprecated compatibility import for the canonical QuantTerm API.

New code and launchers must import :mod:`api.app`.  This module intentionally
contains no route or runtime implementation; it only preserves older imports.
"""
from __future__ import annotations

import api.app as _canonical

for _name in dir(_canonical):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_canonical, _name)

app = _canonical.app
