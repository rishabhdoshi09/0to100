"""Stale institutional flows are never presented as today's context
(owner's cache was last written 2026-08-26 while the brief reads it as "aaj")."""
from __future__ import annotations

from data.institutional_flows import USABLE_MAX_AGE_S, usable_flows

NOW = 1_800_000_000.0
FLOWS = {"fii_dii": {"date": "25-Aug-2026", "bias": "SUPPORTIVE", "note": "FII buying"},
         "bulk_buys": ["ASPIRE"]}


def test_recent_flows_are_usable():
    data = {**FLOWS, "ts": NOW - 3600}
    assert usable_flows(data, now=NOW) is data


def test_old_or_stale_cache_is_dropped():
    assert usable_flows({**FLOWS, "ts": NOW - USABLE_MAX_AGE_S - 1}, now=NOW) == {}
    assert usable_flows({**FLOWS, "ts": NOW - 81 * 86400, "stale": True}, now=NOW) == {}


def test_missing_or_bad_timestamps_are_dropped():
    assert usable_flows({**FLOWS}, now=NOW) == {}
    assert usable_flows({**FLOWS, "ts": "x"}, now=NOW) == {}
    assert usable_flows({**FLOWS, "ts": NOW + 3600}, now=NOW) == {}
    assert usable_flows(None, now=NOW) == {} and usable_flows({}, now=NOW) == {}


def test_brain_probe_hides_august_flows(monkeypatch):
    import core.brain as brain
    import data.institutional_flows as flows

    monkeypatch.setattr(flows, "get_flows", lambda *a, **k: {**FLOWS, "ts": 1_787_000_000.0, "stale": True})
    assert brain._probe_flows("IN") == {}
