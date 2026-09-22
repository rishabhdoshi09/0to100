from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from product.evidence_class import (
    OPERATIONAL_DECISION_ONLY,
    is_market_evidence,
    may_promote,
)
from product.forward_evidence import load_ledger
from product.forward_soak import COLLECTING, verify_persisted_soak
from product.paper_autopilot import ENTER_NOW, WAIT, run_reco_decision_cycle
from research.auto_research.paper_book import PaperBook
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS
from research.autonomy.supervisor import Supervisor


def _clock():
    return datetime(2026, 9, 22, 2, 0, tzinfo=timezone.utc)


def _card(symbol="TCS", **over):
    row = {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "reco_tier_label": "High Conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": 3,
        "score": 82,
        "primary_thesis": "VCP + quality",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90},
            {"id": "sepa", "status": "pass", "points": 70},
            {"id": "funds", "status": "pass", "points": 80},
            {"id": "trend", "status": "pass", "points": 85},
            {"id": "rs", "status": "pass", "points": 75},
            {"id": "ev", "status": "unknown", "points": None},
            {"id": "conviction", "status": "pass", "points": 80},
            {"id": "case", "status": "unknown", "points": None},
            {"id": "sector", "status": "pass", "points": 80},
        ],
    }
    row.update(over)
    return row


def _workspace(cards):
    stamp = _clock().isoformat()
    return {
        "schema_version": 4,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }


def _write_scan_and_reco(cards):
    stamp = _clock().isoformat()
    scan_path = Path(os.environ["QT_SCAN_PATH"])
    reco_path = Path(os.environ["QT_RECO_PATH"])
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    reco_path.parent.mkdir(parents=True, exist_ok=True)
    scan_path.write_text(
        json.dumps({
            "schema_version": 2,
            "available": True,
            "scanned_at": stamp,
            "scanned": len(cards),
            "universe_size": len(cards),
            "records": [{"symbol": row["symbol"], "score": row.get("score", 0)} for row in cards],
            "provenance": {"market_session_date": "2026-09-22", "data_current": True},
        }),
        encoding="utf-8",
    )
    reco_path.write_text(json.dumps(_workspace(cards)), encoding="utf-8")


def _verified_lock():
    return {
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "live_lock_status": "LOCKED",
        "live_lock_reason": "paper-only test boundary",
        "live_lock_source": "product.live_execution_interlock",
    }


def test_operational_decision_only_is_never_market_or_promotion_evidence():
    assert is_market_evidence(OPERATIONAL_DECISION_ONLY) is False
    assert may_promote(OPERATIONAL_DECISION_ONLY) is False


def test_decision_only_reuses_selector_but_cannot_execute_or_write_forward_evidence(monkeypatch):
    card = _card()
    book = PaperBook(capital=100_000)

    monkeypatch.setattr(
        "product.paper_autopilot._execute",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decision-only must never execute")),
    )
    monkeypatch.setattr(
        "product.counterfactual_learning.freeze_decision",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decision-only must not freeze counterfactuals")),
    )
    monkeypatch.setattr(
        "product.forward_soak.record_cycle_evidence",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decision-only must not write forward evidence")),
    )

    out = run_reco_decision_cycle(
        book=book,
        cards=[card],
        workspace=_workspace([card]),
        as_of="2026-09-22",
        now=_clock(),
        paper_enabled=True,
        persist_journal=False,
        enforce_history=False,
    )

    assert out["decision_only"] is True
    assert out["decision_only_valid"] is True
    assert len(out["decision_only_judgments"]) == 1
    judgment = out["decision_only_judgments"][0]
    assert judgment["decision"] == ENTER_NOW
    assert judgment["status"] == "WOULD_ENTER"
    assert judgment["evidence_class"] == OPERATIONAL_DECISION_ONLY
    assert judgment["decision_fingerprint"]
    assert judgment["opens_position"] is False
    assert judgment["trade_intent_created"] is False
    assert out["taken"] == []
    assert out["positions_opened"] == []
    assert not book.open
    assert load_ledger() == []


def test_decision_only_preserves_wait_instead_of_manufacturing_buy():
    card = _card(chase_risk=True, entry_state="extended")
    out = run_reco_decision_cycle(
        book=PaperBook(capital=100_000),
        cards=[card],
        workspace=_workspace([card]),
        as_of="2026-09-22",
        now=_clock(),
        paper_enabled=True,
        persist_journal=False,
        enforce_history=False,
    )

    assert out["decision_only_valid"] is True
    assert out["would_enter"] == []
    assert len(out["decision_only_judgments"]) == 1
    assert out["decision_only_judgments"][0]["decision"] == WAIT
    assert out["decision_only_judgments"][0]["reason_code"] == "ENTRY_TOO_EXTENDED"


def test_empty_decision_only_cycle_cannot_close_selection_gap():
    out = run_reco_decision_cycle(
        book=PaperBook(capital=100_000),
        cards=[],
        workspace=_workspace([]),
        as_of="2026-09-22",
        now=_clock(),
        persist_journal=False,
        enforce_history=False,
    )

    assert out["decision_only_valid"] is False
    assert out["decision_only_judgments"] == []
    assert out["eligibility"] == "DECISION_ONLY_INVALID"


def test_persisted_decision_truth_closes_selection_gap_without_forward_evidence(monkeypatch):
    card = _card(chase_risk=True, entry_state="extended")
    _write_scan_and_reco([card])
    monkeypatch.setattr("product.forward_soak.live_safety_projection", _verified_lock)
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )

    out = run_reco_decision_cycle(
        book=PaperBook(capital=100_000),
        cards=[card],
        workspace=_workspace([card]),
        as_of="2026-09-22",
        now=_clock(),
        persist_journal=True,
        enforce_history=False,
    )
    assert out["decision_only_valid"] is True
    assert load_ledger() == []

    verified = verify_persisted_soak()
    assert verified["lanes"]["SELECTION"] == "PASS"
    assert verified["lanes"]["AUTOPILOT"] == "PASS"
    assert verified["lanes"]["PAPER EXECUTION"] == "MARKET_CLOSED_DECISION_ONLY"
    assert verified["decision_only"] is True
    assert verified["decision_only_valid"] is True
    assert verified["decision_only_judgment_count"] == 1
    assert verified["real_forward_n"] == 0
    assert verified["soak_status"] == COLLECTING
    assert verified["lanes"]["LIVE MONEY"] == "LOCKED"


def test_invalid_persisted_decision_only_cycle_remains_selection_failure(monkeypatch):
    _write_scan_and_reco([])
    monkeypatch.setattr("product.forward_soak.live_safety_projection", _verified_lock)
    monkeypatch.setattr(
        "product.autonomy_status.read_autonomy_status",
        lambda *a, **k: {"running": True},
    )
    run_reco_decision_cycle(
        book=PaperBook(capital=100_000),
        cards=[],
        workspace=_workspace([]),
        as_of="2026-09-22",
        now=_clock(),
        persist_journal=True,
        enforce_history=False,
    )

    verified = verify_persisted_soak()
    assert verified["lanes"]["SELECTION"] == "FAIL"
    assert verified["lanes"]["PAPER EXECUTION"] == "FAIL"
    assert verified["decision_only"] is True
    assert verified["decision_only_valid"] is False


class _JobDeps:
    def __init__(self):
        self.calls = []

    def now_ist(self):
        return datetime(2026, 9, 22, 23, 0)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snap-current"

    def run_decision_only_cycle(self, reason, phase):
        self.calls.append((reason, phase))
        return {
            "eligibility": "DECISION_ONLY_JUDGED",
            "decision_only": True,
            "decision_only_valid": True,
            "decision_only_judgments": [{"symbol": "TCS", "decision": "WAIT"}],
        }


def test_decision_only_job_is_structurally_non_executing():
    deps = _JobDeps()
    job = SimpleNamespace(idempotency_key="snapshot_decision:snap:thesis:cal")
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps, job=job))

    assert result.status == JS.SUCCEEDED
    assert result.new_entries_allowed is False
    assert result.metadata["decision_only"] is True
    assert result.metadata["not_forward_evidence"] is True
    assert result.metadata["judgment_count"] == 1
    assert deps.calls == [("ENTRY_WINDOW_CLOSED_DECISION_ONLY", "off_session")]


def test_decision_only_job_does_not_journal_without_trusted_market_data():
    class NoData(_JobDeps):
        def active_snapshot_id(self):
            return ""

    deps = NoData()
    job = SimpleNamespace(idempotency_key="snapshot_decision:missing")
    result = JOBS.run_paper_cycle(JOBS._Ctx(deps, job=job))

    assert result.status == JS.BLOCKED
    assert result.error_code == "NO_DATA_SNAPSHOT"
    assert result.new_entries_allowed is False
    assert result.metadata["decision_only_valid"] is False
    assert result.metadata["judgment_count"] == 0
    assert deps.calls == []


class _CaptureJobs:
    def __init__(self):
        self.calls = []

    def enqueue(self, job_type, **kwargs):
        self.calls.append((job_type, dict(kwargs)))
        return SimpleNamespace(status=JS.PENDING)


class _ClosedDeps:
    def now_ist(self):
        return datetime(2026, 9, 22, 23, 0)

    def holidays(self):
        return set()

    def active_snapshot_id(self):
        return "snap-current"


class _SchedulerHarness:
    def __init__(self):
        self.deps = _ClosedDeps()
        self.jobs = _CaptureJobs()

    def _ensure_decision_simulation_authority(self):
        return True

    def _snapshot_token(self, *args, **kwargs):
        return "snap-current"


def test_scheduler_identity_pins_data_thesis_and_calibration(monkeypatch):
    monkeypatch.setattr(
        "product.decision_simulation_gate.status",
        lambda: {
            "discovery_ready": True,
            "scan_scanned_at": "2026-09-22T15:30:00+05:30",
            "thesis_hash": "thesis-abc",
        },
    )
    monkeypatch.setattr(
        "scan.calibration_snapshot.load_current",
        lambda: {"snapshot_id": "cal-xyz"},
    )
    policies = [{"policy_id": "P1", "version": 2, "production_status": "ACTIVE"}]
    monkeypatch.setattr(
        "product.learning_policy_store.load_policies",
        lambda: {"policies": policies},
    )
    sup = _SchedulerHarness()

    Supervisor._ensure_closed_market_decision_cycle(sup)

    assert len(sup.jobs.calls) == 1
    job_type, kwargs = sup.jobs.calls[0]
    assert job_type == "paper_cycle"
    canonical = json.dumps(policies, sort_keys=True, separators=(",", ":"), default=str)
    policy_hash = "policy-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert kwargs["idempotency_key"] == (
        "snapshot_decision:snap-current:"
        "2026-09-22T15:30:00+05:30:"
        "thesis-abc:cal-xyz:" + policy_hash
    )
    assert kwargs["input_snapshot_id"] == "snap-current"
    assert kwargs["critical"] is False
