"""Production authority tests for the Evolution Engine.

These tests intentionally drive product.paper_autopilot.run_reco_paper_cycle:
the policy recorded as Current Champion must be the policy that controls
actual future PAPER selection. Challenger shadows must see the same
pre-mutation account state and must never rescue a canonical hard reject.
"""
from __future__ import annotations

from datetime import datetime, timezone

from product.evolution import policy_registry as PR
from product.evolution import shadow_decisions as SD
from product.evolution import snapshot as ES
from product.paper_autopilot import (
    BLOCK,
    ENTER_NOW,
    WAIT,
    evaluate_selection_candidate,
    run_reco_paper_cycle,
)
from research.auto_research.paper_book import PaperBook


def _now(day: int = 30) -> datetime:
    return datetime(2026, 9 if day == 30 else 10, day if day == 30 else 1, 10, 0, tzinfo=timezone.utc)


def _card(
    symbol: str,
    *,
    score: float,
    rs: float,
    confirms: int,
    entry_state: str = "ready",
    empirical_n: int = 40,
) -> dict:
    return {
        "symbol": symbol,
        "reco_tier": "high_conviction",
        "reco_tier_label": "High Conviction",
        "entry_state": entry_state,
        "entry": 100.0,
        "stop": 95.0,
        "target": 112.0,
        "cmp": 100.0,
        "chase_risk": entry_state == "extended",
        "volume_ratio": 1.4,
        "sector": "Technology",
        "family_confirms": confirms,
        "score": score,
        "rs_percentile": rs,
        "empirical_n": empirical_n,
        "primary_thesis": "VCP + quality",
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90, "detail": "clean"},
            {"id": "sepa", "status": "pass", "points": 70, "detail": "base"},
            {"id": "funds", "status": "pass", "points": 80, "detail": "quality"},
            {"id": "trend", "status": "pass", "points": 85, "detail": "up"},
            {"id": "rs", "status": "pass", "points": 75, "detail": "leader"},
            {"id": "ev", "status": "unknown", "points": None, "detail": "collecting"},
            {"id": "case", "status": "unknown", "points": None, "detail": "collecting"},
            {"id": "sector", "status": "pass", "points": 80, "detail": "leader"},
        ],
    }


def _workspace(cards, stamp: str) -> dict:
    return {
        "schema_version": 4,
        "point_in_time": True,
        "generated_at": stamp,
        "scan_scanned_at": stamp,
        "categories": [{"id": "wealth_builders", "count": len(cards), "cards": cards}],
    }


def _cycle(book, cards, *, as_of: str, now: datetime):
    return run_reco_paper_cycle(
        book=book,
        cards=cards,
        workspace=_workspace(cards, now.isoformat()),
        as_of=as_of,
        now=now,
        entries_allowed=True,
        paper_enabled=True,
        persist_journal=False,
        max_new=1,
        enforce_history=False,
        regime="RISK_ON",
    )


def test_seed_population_is_idempotent_and_nonempty(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    first = PR.ensure_seed_population(PR.EQUITY)
    first_ids = [p["policy_id"] for p in PR.list_policies(domain=PR.EQUITY)]
    second = PR.ensure_seed_population(PR.EQUITY)
    second_ids = [p["policy_id"] for p in PR.list_policies(domain=PR.EQUITY)]

    assert first["champion"]["policy_id"] == second["champion"]["policy_id"]
    assert len(first["challengers"]) >= 5
    assert first_ids == second_ids


def test_neutral_evolution_policy_matches_baseline_candidate_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card("TCS", score=100, rs=20, confirms=3)
    book = PaperBook(capital=500_000)
    baseline = evaluate_selection_candidate(
        card, book=book, workspace=_workspace([card], _now().isoformat()),
        now=_now(), enforce_history=False, regime="RISK_ON",
    )
    neutral = {
        "policy_id": "NEUTRAL",
        "domain": PR.EQUITY,
        "version": 1,
        "status": PR.CHAMPION,
        "weights": {},
        "hypothesis": "neutral parity",
    }
    evolved = evaluate_selection_candidate(
        card, book=book, workspace=_workspace([card], _now().isoformat()),
        now=_now(), enforce_history=False, regime="RISK_ON",
        evolution_policy=neutral,
    )
    assert evolved.decision == baseline.decision
    assert evolved.reason_code == baseline.reason_code
    assert evolved.selection_score == baseline.selection_score


def test_current_champion_controls_real_future_paper_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="AUTH_CHAMP_A", domain=PR.EQUITY, hypothesis="neutral baseline",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="AUTH_CHAL_B", domain=PR.EQUITY,
        hypothesis="strongly prefer relative-strength leadership",
        weights={"relative_strength_mult": 3.0}, status=PR.CHALLENGER,
        parent_policy_id="AUTH_CHAMP_A",
    )

    # Baseline: TCS wins narrowly on evidence-family strength.
    cards = [
        _card("TCS", score=100, rs=20, confirms=3),
        _card("RELIANCE", score=80, rs=90, confirms=2),
    ]
    before = _cycle(PaperBook(capital=500_000), cards, as_of="2026-09-30", now=_now())
    assert before["taken"][0]["symbol"] == "TCS"
    assert before["taken"][0]["champion_policy_id"] == "AUTH_CHAMP_A"

    # Simulate a completed, qualified promotion state transition; promotion
    # science itself is separately tested in test_evolution_promotion.py.
    PR.set_status("AUTH_CHAMP_A", PR.PROBATION, reason="test promotion handoff")
    PR.set_status("AUTH_CHAL_B", PR.CHAMPION, reason="test promotion handoff")

    after = _cycle(PaperBook(capital=500_000), cards, as_of="2026-10-01", now=_now(1))
    assert after["taken"][0]["symbol"] == "RELIANCE"
    assert after["taken"][0]["champion_policy_id"] == "AUTH_CHAL_B"
    assert after["evolution"]["controls_paper_decisions"] is True


def test_evolution_policy_cannot_rescue_hard_ineligible_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="HARD_CHAMP", domain=PR.EQUITY, hypothesis="aggressive rank only",
        weights={"relative_strength_mult": 3.0, "sector_confirmation_bonus": 5.0},
        status=PR.CHAMPION,
    )
    card = _card("TCS", score=100, rs=99, confirms=4, entry_state="extended")
    decision = evaluate_selection_candidate(
        card,
        book=PaperBook(capital=500_000),
        workspace=_workspace([card], _now().isoformat()),
        now=_now(),
        enforce_history=False,
        regime="RISK_ON",
        evolution_policy=PR.current_champion(PR.EQUITY),
    )
    assert decision.decision in {WAIT, BLOCK}
    assert decision.decision != ENTER_NOW


def test_market_twin_snapshot_is_pre_mutation_and_remains_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    book = PaperBook(capital=500_000)
    cycle = _cycle(book, cards, as_of="2026-09-30", now=_now())
    assert len(book.open) == 1

    snapshot_ids = cycle["evolution"]["market_snapshot_ids"]
    assert snapshot_ids
    frozen = ES.get_snapshot_record(snapshot_ids[0])
    assert frozen is not None
    assert frozen["pre_decision_book"]["open_count"] == 0

    # Mutating the real book after the freeze cannot rewrite the Market Twin.
    frozen_again = ES.get_snapshot_record(snapshot_ids[0])
    assert frozen_again["pre_decision_book"]["open_count"] == 0


def test_normal_paper_cycle_prepares_restart_safe_challenger_work(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    cycle = _cycle(PaperBook(capital=500_000), cards, as_of="2026-09-30", now=_now())

    from product.evolution import deferred_work as DW

    work_id = cycle["evolution"]["deferred_work_id"]
    assert work_id
    assert cycle["evolution"]["deferred_work_status"] == "PREPARED"
    work = DW.get_work(work_id)
    assert work is not None
    assert work["status"] == "PREPARED"
    assert len(work["challenger_policies"]) >= 5
    # Champion rows are frozen immediately; Challenger rows are intentionally
    # absent until the non-critical research lane runs.
    challenger_ids = {p["policy_id"] for p in PR.active_challengers(PR.EQUITY)}
    frozen_ids = {row["policy_id"] for row in SD.list_shadow_decisions()}
    assert challenger_ids.isdisjoint(frozen_ids)


def test_paper_cycle_never_invokes_challenger_evaluation(tmp_path, monkeypatch):
    """A non-returning Challenger cannot block PAPER because PAPER never calls
    the Challenger evaluator at all.  It only persists a PREPARED work item."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="DETACH_CHAMP", domain=PR.EQUITY, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="DETACH_CHAL", domain=PR.EQUITY, hypothesis="shadow",
        weights={}, status=PR.CHALLENGER, parent_policy_id="DETACH_CHAMP",
    )

    from product.evolution import tournament as T

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("PAPER path must never execute Challenger research")

    monkeypatch.setattr(T, "evaluate_challengers_from_bundle", _must_not_run)

    book = PaperBook(capital=500_000)
    cycle = _cycle(
        book,
        [_card("TCS", score=100, rs=80, confirms=3)],
        as_of="2026-09-30",
        now=_now(),
    )
    assert len(book.open) == 1
    assert cycle["taken"][0]["symbol"] == "TCS"
    assert cycle["evolution"]["deferred_work_id"]
    assert cycle["evolution"]["challengers_evaluated"] == []


def test_deferred_challenger_uses_exact_premutation_snapshot_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="TWIN_CHAMP", domain=PR.EQUITY, hypothesis="neutral baseline",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="TWIN_CHAL", domain=PR.EQUITY, hypothesis="test challenger",
        weights={}, status=PR.CHALLENGER, parent_policy_id="TWIN_CHAMP",
    )

    book = PaperBook(capital=500_000)
    cycle = _cycle(
        book,
        [_card("TCS", score=100, rs=80, confirms=3)],
        as_of="2026-09-30",
        now=_now(),
    )
    assert len(book.open) == 1

    from product.evolution import deferred_work as DW

    work_id = cycle["evolution"]["deferred_work_id"]
    # Simulated process restart: work is re-read from disk; no in-memory
    # closure/book reference is required.
    reloaded = DW.get_work(work_id)
    assert reloaded is not None and reloaded["status"] == "PREPARED"
    DW.mark_ready(work_id)
    result = DW.process_work_item(work_id)
    assert "TWIN_CHAL" in result["challengers_evaluated"]

    champion_row = next(
        r for r in SD.list_shadow_decisions(policy_id="TWIN_CHAMP") if r["symbol"] == "TCS"
    )
    challenger_row = next(
        r for r in SD.list_shadow_decisions(policy_id="TWIN_CHAL") if r["symbol"] == "TCS"
    )
    assert challenger_row["market_snapshot_id"] == champion_row["market_snapshot_id"]

    frozen_snapshot = ES.get_snapshot_record(challenger_row["market_snapshot_id"])
    assert frozen_snapshot is not None
    assert frozen_snapshot["pre_decision_book"]["open_count"] == 0
    assert len(book.open) == 1


def test_deferred_research_retry_cannot_duplicate_paper_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="RETRY_CHAMP", domain=PR.EQUITY, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="RETRY_CHAL", domain=PR.EQUITY, hypothesis="shadow",
        weights={}, status=PR.CHALLENGER, parent_policy_id="RETRY_CHAMP",
    )
    book = PaperBook(capital=500_000)
    cycle = _cycle(
        book,
        [_card("TCS", score=100, rs=80, confirms=3)],
        as_of="2026-09-30",
        now=_now(),
    )
    assert len(book.open) == 1

    from product.evolution import deferred_work as DW

    work_id = cycle["evolution"]["deferred_work_id"]
    DW.mark_ready(work_id)
    first = DW.process_work_item(work_id)
    second = DW.process_work_item(work_id)
    assert first == second
    assert len(book.open) == 1


def test_live_money_remains_locked_through_the_deferred_tournament_path(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="LOCK_CHAMP", domain=PR.EQUITY, hypothesis="neutral baseline",
        weights={}, status=PR.CHAMPION,
    )
    book = PaperBook(capital=500_000)
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    cycle = _cycle(book, cards, as_of="2026-09-30", now=_now())

    assert cycle["execution_reality"]["shadow_mode"] is True
    assert cycle["execution_reality"]["affects_paper_orders"] is False

    import ast
    import inspect
    from product.evolution import deferred_work
    from product.evolution import tournament as evolution_tournament_engine

    forbidden = {
        "execution", "zerodha_broker", "telegram_actions",
        "telegram_commands", "kite_client",
    }
    for module in (evolution_tournament_engine, deferred_work):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (names & forbidden), f"forbidden import found: {names}"
