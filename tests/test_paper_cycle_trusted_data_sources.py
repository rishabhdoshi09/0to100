"""Paper-cycle entry gating must follow trusted market data, not Kite snapshot existence.

A successful autonomous scan may be backed by an active snapshot, current official NSE
completed-session history, or an explicitly ready live source.  Paper execution consumes the
saved recommendation card and does not require a broker-created snapshot.  These tests pin that
contract so a daily/absent broker session cannot re-enter through ``active_snapshot_id()``.
"""
from __future__ import annotations

from datetime import datetime

from research.autonomy import jobs as JOBS


_NOW = datetime(2026, 7, 31, 10, 0)  # Friday, inside 09:30-15:15 entry window.


class _Deps:
    def __init__(self, *, snapshot=None, official=False, live=None):
        self.snapshot = snapshot
        self.official = official
        self.live = dict(live or {})
        self.seen = {}
        self.live_probes = 0

    def now_ist(self):
        return _NOW

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return self.snapshot

    def official_history(self):
        return {"current": self.official, "source": "official_nse"}

    def live_market_ready(self):
        self.live_probes += 1
        return self.live

    def run_paper_cycle(self, entries_allowed, reason="", phase="", failures=()):
        self.seen = {
            "allowed": bool(entries_allowed),
            "reason": str(reason or ""),
            "phase": str(phase or ""),
            "failures": set(failures or ()),
        }
        return {"eligibility": "NO_ELIGIBLE_TRADE"}


def test_current_official_history_allows_paper_without_snapshot_or_broker_probe():
    deps = _Deps(snapshot=None, official=True)
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))

    assert result.status == JOBS.JS.SUCCEEDED
    assert result.new_entries_allowed is True
    assert deps.seen["allowed"] is True
    assert deps.seen["reason"] == ""
    assert deps.live_probes == 0
    assert result.metadata["market_data_source"] == "official_nse"


def test_active_snapshot_remains_a_valid_paper_data_source():
    deps = _Deps(snapshot="snap-real", official=False)
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))

    assert result.new_entries_allowed is True
    assert deps.seen["allowed"] is True
    assert deps.live_probes == 0
    assert result.metadata["market_data_source"] == "snapshot:snap-real"


def test_explicitly_ready_live_source_is_valid_when_official_history_is_not_current():
    deps = _Deps(snapshot=None, official=False, live={"ready": True, "source": "kite_quotes"})
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))

    assert result.new_entries_allowed is True
    assert deps.seen["allowed"] is True
    assert deps.live_probes == 1
    assert result.metadata["market_data_source"] == "kite_quotes"


def test_no_trusted_market_source_still_blocks_new_paper_entries():
    deps = _Deps(snapshot=None, official=False, live={})
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps))

    assert result.status == JOBS.JS.SUCCEEDED
    assert result.new_entries_allowed is False
    assert deps.seen["allowed"] is False
    assert deps.seen["reason"] == "NO_DATA_SNAPSHOT"
    assert result.metadata["market_data_source"] == "unavailable"
