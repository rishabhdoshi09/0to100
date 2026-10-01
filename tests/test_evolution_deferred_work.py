"""Durability and process-isolation tests for deferred Evolution work."""
from __future__ import annotations

import sys
import time

from product import decision_context
from product.evolution import deferred_work as DW
from product.evolution import tournament
from research.auto_research.paper_book import PaperBook


def _hang_command(_root: str, _work_id: str) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def _card(symbol: str = "TCS") -> dict:
    return {
        "symbol": symbol,
        "entry": 100.0,
        "stop": 95.0,
        "target": 112.0,
        "setup_label": "VCP",
        "primary_thesis": "VCP",
        "sector": "Technology",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "volume_ratio": 1.5,
        "rs_percentile": 85,
        "family_confirms": 3,
        "score": 90,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90},
            {"id": "sepa", "status": "pass", "points": 85},
            {"id": "funds", "status": "pass", "points": 80},
            {"id": "trend", "status": "pass", "points": 85},
            {"id": "rs", "status": "pass", "points": 90},
            {"id": "sector", "status": "pass", "points": 80},
        ],
    }


def _prepared_real_item(tmp_path):
    card = _card()
    book = PaperBook(capital=500_000)
    ctx = decision_context.snapshot(card, book=book, regime="RISK_ON")
    base = decision_context.score_breakdown(
        card,
        {"final_effect": "NEUTRAL", "sample_size": 0},
        ctx,
    )
    individual = {
        "TCS": {
            "symbol": "TCS",
            "decision": "ENTER_NOW",
            "reason_code": "ELIGIBLE",
            "selection_score": float(base["selection_rank"]),
            "detail": "passed all gates",
            "breakdown": {
                **base,
                "pre_evolution_breakdown": {
                    **base,
                    "parts": [dict(p) for p in base["parts"]],
                },
                "learning_challenger": {
                    "available": False,
                    "affects_selection": False,
                    "adjustment": 0.0,
                    "live_locked": True,
                },
            },
        }
    }
    bundle = tournament.freeze_premutation_bundle(
        [card],
        {
            "TCS": {
                "decision": "ENTER_NOW",
                "reason_code": "ELIGIBLE",
                "selection_score": float(base["selection_rank"]),
                "breakdown": individual["TCS"]["breakdown"],
            }
        },
        champion_policy_id="CHAMP",
        as_of="2026-09-30",
        book=book,
        regime="RISK_ON",
    )
    return DW.prepare_work(
        bundle=bundle,
        champion_policy_fingerprint="champ-fingerprint",
        challenger_policies=[
            {
                "policy_id": "CHAL",
                "domain": "EQUITY",
                "version": 1,
                "status": "CHALLENGER",
                "hypothesis": "test frozen replay",
                "weights": {"relative_strength_mult": 1.25},
            }
        ],
        individual_decisions_by_symbol=individual,
        pre_mutation_book_snapshot=book.snapshot(),
        held_sector_by_symbol={},
        correlations={},
        max_new=1,
        regime="RISK_ON",
    )


def test_prepared_work_survives_restart_and_uses_only_frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    prepared = _prepared_real_item(tmp_path)
    work_id = prepared["work_id"]

    # No in-memory closure is required. Re-read the durable work item as a
    # restarted process would, then release it and evaluate.
    reloaded = DW.get_work(work_id)
    assert reloaded is not None
    assert reloaded["status"] == "PREPARED"
    DW.mark_ready(work_id)

    result = DW.process_work_item(work_id)
    assert result["challengers_evaluated"] == ["CHAL"]
    assert result["results"][0]["challengers"]["CHAL"]["symbol"] == "TCS"
    assert DW.get_work(work_id)["status"] == "SUCCEEDED"

    # Idempotent retry cannot duplicate research or mutate a PAPER book.
    second = DW.process_work_item(work_id)
    assert second == result


def test_isolated_worker_kills_nonreturning_child_and_returns_control(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    root = tmp_path / "isolated"
    prepared = DW.prepare_work(
        bundle={
            "domain": "EQUITY",
            "as_of": "2026-09-30",
            "champion_policy_id": "CHAMP",
            "snapshots": {},
            "champion_rows": {},
        },
        champion_policy_fingerprint="fp",
        challenger_policies=[],
        individual_decisions_by_symbol={},
        pre_mutation_book_snapshot={},
        held_sector_by_symbol={},
        correlations={},
        max_new=1,
        regime="RISK_ON",
        root=root,
    )
    DW.mark_ready(prepared["work_id"], root=root)

    started = time.monotonic()
    outcome = DW.process_ready_isolated(
        root=root,
        limit=1,
        timeout_seconds=0.15,
        command_factory=_hang_command,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert len(outcome) == 1
    assert outcome[0]["timed_out"] is True
    assert outcome[0]["status"] == "RETRYABLE"
    assert DW.get_work(prepared["work_id"], root=root)["status"] == "RETRYABLE"


def test_three_timeouts_fail_closed_instead_of_blocking_forever(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    root = tmp_path / "retry"
    prepared = DW.prepare_work(
        bundle={
            "domain": "EQUITY",
            "as_of": "2026-09-30",
            "champion_policy_id": "CHAMP",
            "snapshots": {},
            "champion_rows": {},
        },
        champion_policy_fingerprint="fp2",
        challenger_policies=[],
        individual_decisions_by_symbol={},
        pre_mutation_book_snapshot={},
        held_sector_by_symbol={},
        correlations={},
        max_new=1,
        regime="RISK_ON",
        root=root,
    )
    DW.mark_ready(prepared["work_id"], root=root)

    for expected in ("RETRYABLE", "RETRYABLE", "FAILED"):
        outcome = DW.process_ready_isolated(
            root=root,
            limit=1,
            timeout_seconds=0.05,
            command_factory=_hang_command,
        )
        assert outcome[0]["status"] == expected

    assert DW.ready_work(root=root) == []
    assert DW.get_work(prepared["work_id"], root=root)["attempts"] == 3


def test_challenger_can_differ_from_champion_overlay_without_rescuing_hard_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    ctx = decision_context.snapshot(card, book=None, regime="RISK_ON")
    base = decision_context.score_breakdown(
        card,
        {"final_effect": "NEUTRAL", "sample_size": 0},
        ctx,
    )
    item = {
        "bundle": {"domain": "EQUITY"},
        "individual_decisions_by_symbol": {
            "TCS": {
                # Final Champion decision was blocked by its own Evolution
                # sample-floor policy, but canonical production eligibility
                # before that overlay was ENTER_NOW.
                "decision": "BLOCK",
                "reason_code": "EVOLUTION_MIN_SAMPLE_NOT_MET",
                "selection_score": 0.0,
                "breakdown": {
                    **base,
                    "pre_evolution_breakdown": {
                        **base,
                        "parts": [dict(p) for p in base["parts"]],
                    },
                    "pre_evolution_decision": {
                        "decision": "ENTER_NOW",
                        "reason_code": "ELIGIBLE",
                        "detail": "passed canonical hard gates",
                    },
                    "learning_challenger": {"adjustment": 0.0},
                },
            }
        },
        "pre_mutation_book_snapshot": PaperBook(capital=500_000).snapshot(),
        "held_sector_by_symbol": {},
        "correlations": {},
        "max_new": 1,
        "regime": "RISK_ON",
    }
    snapshot = {
        "symbol": "TCS",
        "market_snapshot_id": "snap-1",
        "context": ctx,
        "card": card,
    }
    permissive = {
        "policy_id": "PERMISSIVE",
        "version": 1,
        "status": "CHALLENGER",
        "weights": {},
    }
    verdict = DW._frozen_policy_batch(item, [snapshot], permissive)[0]
    assert verdict["decision"] == "ENTER_NOW"

    # A genuine canonical hard reject stays rejected regardless of policy.
    item["individual_decisions_by_symbol"]["TCS"]["breakdown"]["pre_evolution_decision"] = {
        "decision": "BLOCK",
        "reason_code": "INVALID_STOP",
        "detail": "hard gate",
    }
    hard = DW._frozen_policy_batch(item, [snapshot], permissive)[0]
    assert hard["decision"] == "BLOCK"
    assert hard["reason_code"] == "INVALID_STOP"


def test_default_isolated_worker_processes_real_prepared_work(tmp_path, monkeypatch):
    """Exercise the exact production subprocess command, module entrypoint and
    repo-root cwd used by launchd/supervisor integration."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    prepared = _prepared_real_item(tmp_path)
    DW.mark_ready(prepared["work_id"])

    outcome = DW.process_ready_isolated(
        root=DW._root(),
        limit=1,
        timeout_seconds=10.0,
    )
    assert len(outcome) == 1
    assert outcome[0]["work_id"] == prepared["work_id"]
    assert outcome[0]["status"] == "SUCCEEDED"
    assert outcome[0]["timed_out"] is False
    assert DW.get_work(prepared["work_id"])["status"] == "SUCCEEDED"
