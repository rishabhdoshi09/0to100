"""Regressions for the parallel release hardening imported into PR #149."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import types

import pytest

from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
import research.autonomy.paper_cycle_truth as PCT
import scan.market_scan_service as MSS


ROOT = Path(__file__).resolve().parents[1]


def test_generic_installer_fails_closed_by_default_on_darwin_without_breaking_linux_bootstrap():
    src = (ROOT / "scripts" / "install_quantterm_host.sh").read_text(encoding="utf-8")
    assert 'elif [[ "$(uname -s)" == "Darwin" ]]' in src
    assert "REQUIRE_EXISTING=1" in src
    assert "REQUIRE_EXISTING=0" in src
    assert "product.host_install_existing" in src
    assert "product.host_install install" in src
    assert "Invalid QT_RUNTIME_ROOT_REQUIRE_EXISTING" in src


def test_macbook_runbook_uses_storage_preflight_entrypoint_not_generic_installer():
    src = (ROOT / "docs" / "MACBOOK_RUNBOOK.md").read_text(encoding="utf-8")
    assert "bash deploy/setup_mac.sh" in src
    assert "bash scripts/install_quantterm_host.sh" not in src
    assert "/Volumes/QuantTermStorage/QuantTerm/runtime" in src
    assert "--manager launchd" in src


class _Brain:
    def __init__(self):
        self.intel_book = object()
        self.state = types.SimpleNamespace(last_intel_cycle=None)
        self.saved = 0

    def run_intelligence_cycle_day(self, **kwargs):
        return {
            "eligibility": "BLOCKED_SAFETY",
            "entry_block_reason": "RECO_SELECTION_AUTHORITY",
            "positions_opened": [],
            "as_of_date": "2026-09-11",
        }

    def is_paper_auto_enabled(self):
        return True

    def _save_intel_book(self):
        self.saved += 1


class _Telegram:
    def __init__(self):
        self.last = None

    def notify_paper_cycle(self, result, book=None):
        self.last = dict(result)


class _Deps:
    def __init__(self, telegram):
        self.live_feed = None
        self.telegram = telegram

    def now_ist(self):
        return datetime(2026, 7, 31, 10, 0)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snap-real"

    def run_paper_cycle(self, *args):
        return JOBS.Deps.run_paper_cycle(self, *args)


def _install_truth_with(monkeypatch, brain, reco_callable):
    fake_sched = types.ModuleType("research.auto_research.scheduler")
    fake_sched.get_brain = lambda: brain
    monkeypatch.setitem(sys.modules, "research.auto_research.scheduler", fake_sched)

    fake_autopilot = types.ModuleType("product.paper_autopilot")
    fake_autopilot.run_reco_paper_cycle = reco_callable
    monkeypatch.setitem(sys.modules, "product.paper_autopilot", fake_autopilot)

    PCT._INSTALLED = False
    PCT.install_paper_cycle_truth()


def test_executor_crash_retires_uncertain_mutation_not_succeeded_or_safety_block(monkeypatch):
    original = JOBS.Deps.run_paper_cycle
    brain = _Brain()
    telegram = _Telegram()

    def boom(**kwargs):
        raise RuntimeError("paper book unavailable")

    try:
        _install_truth_with(monkeypatch, brain, boom)
        result = JOBS.run_paper_cycle(JOBS._Ctx(_Deps(telegram)))
    finally:
        JOBS.Deps.run_paper_cycle = original
        PCT._INSTALLED = False

    assert result.status == JS.PERMANENT_FAILED
    assert result.error_code == "PAPER_CYCLE_UNCERTAIN_MUTATION"
    assert PCT.PAPER_EXECUTION_FAILED in result.error_message
    assert telegram.last["eligibility"] == PCT.PAPER_EXECUTION_FAILED
    assert telegram.last["management_eligibility"] == "BLOCKED_SAFETY"
    assert brain.state.last_intel_cycle["eligibility"] == PCT.PAPER_EXECUTION_FAILED
    assert brain.saved >= 1


def test_traded_without_persisted_fill_fails_durable_job(monkeypatch):
    original = JOBS.Deps.run_paper_cycle
    brain = _Brain()
    telegram = _Telegram()

    def inconsistent(**kwargs):
        return {"eligibility": "TRADED", "positions_opened": []}

    try:
        _install_truth_with(monkeypatch, brain, inconsistent)
        result = JOBS.run_paper_cycle(JOBS._Ctx(_Deps(telegram)))
    finally:
        JOBS.Deps.run_paper_cycle = original
        PCT._INSTALLED = False

    assert result.status == JS.PERMANENT_FAILED
    assert result.error_code == "PAPER_CYCLE_UNCERTAIN_MUTATION"
    assert PCT.EXECUTION_INCONSISTENT in result.error_message
    assert telegram.last["eligibility"] == PCT.EXECUTION_INCONSISTENT
    assert brain.state.last_intel_cycle["execution_truth_error"] == "TRADED_WITHOUT_PERSISTED_POSITION"


class _InstrumentedButSkippedScanner:
    # Presence of _analyze makes scan_coverage instrument the scanner. scan()
    # deliberately never invokes it, reproducing a warm-cache pass that actually
    # evaluated zero symbols.
    def _analyze(self, symbol, frame):
        return None

    def scan(self, symbols, **kwargs):
        return []


def test_warm_cache_zero_evaluated_scan_cannot_overwrite_last_real_artifact(monkeypatch):
    saved = []
    monkeypatch.setattr(MSS, "_saved_priority_inputs", lambda: (None, None, None, []))
    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(
        "scan.bulk_fetcher.backfill_missing",
        lambda symbols: {"requested": len(symbols), "attempted": 0, "loaded": 0},
    )
    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, *a, **k: saved.append(payload))

    report = MSS.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "BBB": "Beta"},
        prefetch_fn=lambda *a, **k: 2,
        scanner=_InstrumentedButSkippedScanner(),
        fno_provider=lambda: set(),
        save=True,
        snapshot_id="snap-real",
    )

    assert report.status == MSS.DATA_UNAVAILABLE
    assert report.error_code == "NO_SYMBOL_EVALUATED"
    assert report.scanned == 0
    assert not saved
    assert "previous scan was kept" in report.error_message.lower()
