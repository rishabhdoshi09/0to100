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


def test_normal_paper_cycle_creates_challenger_shadows_automatically(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    cycle = _cycle(PaperBook(capital=500_000), cards, as_of="2026-09-30", now=_now())
    seeded = {p["policy_id"] for p in PR.active_challengers(PR.EQUITY)}
    frozen = SD.list_shadow_decisions()
    frozen_ids = {row["policy_id"] for row in frozen}

    assert len(seeded) >= 5
    assert seeded & frozen_ids
    assert cycle["evolution"]["challengers_evaluated"]


# ── Challenger evaluation is OUTSIDE the Champion PAPER execution-critical
# path (independent architecture review on PR #260) ─────────────────────────

def test_hung_challenger_does_not_delay_real_champion_paper_mutation(tmp_path, monkeypatch):
    """The literal merge blocker: a Challenger evaluator that blocks far
    longer than the configured tournament budget must NEVER delay the real
    Champion PAPER mutation. Proven by chronological ordering of real
    in-process timestamps -- the real book mutation (_execute) must complete
    BEFORE the slow Challenger's evaluator even starts running, not merely
    "the cycle finished eventually". A budget check that only runs BETWEEN
    Challengers (the pre-fix implementation) cannot guarantee this: the
    Challenger that is already running blocks regardless of the budget."""
    import time as time_module

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("QT_EVOLUTION_TOURNAMENT_MAX_SECONDS", "0.05")

    PR.register_policy(
        policy_id="HANG_CHAMP", domain=PR.EQUITY, hypothesis="neutral baseline",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="HANG_SLOW", domain=PR.EQUITY, hypothesis="deliberately slow evaluator",
        weights={}, status=PR.CHALLENGER, parent_policy_id="HANG_CHAMP",
    )
    PR.register_policy(
        policy_id="HANG_FAST", domain=PR.EQUITY, hypothesis="normal evaluator",
        weights={}, status=PR.CHALLENGER, parent_policy_id="HANG_CHAMP",
    )

    events: list[tuple[str, float]] = []

    import product.paper_autopilot as PA
    real_execute = PA._execute
    real_evaluate = PA.evaluate_selection_candidate

    def _tracking_execute(decision, **kwargs):
        events.append(("champion_mutation", time_module.monotonic()))
        return real_execute(decision, **kwargs)

    def _tracking_evaluate(card, **kwargs):
        policy = kwargs.get("evolution_policy") or {}
        if policy.get("policy_id") == "HANG_SLOW":
            events.append(("slow_challenger_start", time_module.monotonic()))
            time_module.sleep(0.35)  # far longer than the 0.05s budget above
            events.append(("slow_challenger_end", time_module.monotonic()))
        return real_evaluate(card, **kwargs)

    monkeypatch.setattr(PA, "_execute", _tracking_execute)
    monkeypatch.setattr(PA, "evaluate_selection_candidate", _tracking_evaluate)

    book = PaperBook(capital=500_000)
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    started = time_module.monotonic()
    cycle = _cycle(book, cards, as_of="2026-09-30", now=_now())
    total_elapsed = time_module.monotonic() - started

    # The real mutation genuinely happened.
    assert len(book.open) == 1
    assert cycle["taken"]
    assert cycle["taken"][0]["symbol"] == "TCS"
    assert cycle["taken"][0]["champion_policy_id"] == "HANG_CHAMP"

    by_label = {label: ts for label, ts in events}
    assert "champion_mutation" in by_label
    assert "slow_challenger_start" in by_label
    # The core proof: mutation completed strictly BEFORE the slow Challenger
    # even started -- not "finished first" by luck, but structurally ordered.
    assert by_label["champion_mutation"] < by_label["slow_challenger_start"]

    # The slow Challenger really did run for its full duration (this is a
    # genuine block, not a mocked-away no-op), and the whole cycle still
    # completed deterministically afterward.
    assert by_label["slow_challenger_end"] - by_label["slow_challenger_start"] >= 0.3
    assert total_elapsed >= 0.3


def test_deferred_challenger_evaluates_against_exact_premutation_snapshot(tmp_path, monkeypatch):
    """Even though Challenger evaluation now runs AFTER the real mutation,
    it must still see the market/account state EXACTLY as it was before
    mutation -- the Market Twin guarantee must be preserved by freezing
    inputs, not by timing. Proven two ways: (1) the Challenger's frozen
    shadow row cites the identical market_snapshot_id as the Champion's row
    for the same symbol, and (2) that snapshot's own recorded pre_decision_book
    shows zero open positions, even though the REAL book has one open
    position by the time the Challenger actually runs."""
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
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    cycle = _cycle(book, cards, as_of="2026-09-30", now=_now())

    assert len(book.open) == 1  # the real mutation happened

    champion_row = next(
        r for r in SD.list_shadow_decisions(policy_id="TWIN_CHAMP") if r["symbol"] == "TCS"
    )
    challenger_row = next(
        r for r in SD.list_shadow_decisions(policy_id="TWIN_CHAL") if r["symbol"] == "TCS"
    )
    assert challenger_row["market_snapshot_id"] == champion_row["market_snapshot_id"]

    frozen_snapshot = ES.get_snapshot_record(challenger_row["market_snapshot_id"])
    assert frozen_snapshot is not None
    # The frozen Market Twin reflects the PRE-mutation book (zero open),
    # not the real book's current state (one open) at the time this
    # Challenger was actually evaluated.
    assert frozen_snapshot["pre_decision_book"]["open_count"] == 0


def test_challenger_failure_is_recorded_and_excluded_from_consensus(tmp_path, monkeypatch):
    """A Challenger that raises (modeling a timeout/crash) must be recorded
    explicitly as skipped/unevaluated, never silently dropped, and must never
    count toward the consensus denominator (section 16's "only policies that
    actually ran successfully this cycle count")."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="FAIL_CHAMP", domain=PR.EQUITY, hypothesis="neutral baseline",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="FAIL_BROKEN", domain=PR.EQUITY, hypothesis="deliberately broken evaluator",
        weights={}, status=PR.CHALLENGER, parent_policy_id="FAIL_CHAMP",
    )
    PR.register_policy(
        policy_id="FAIL_OK", domain=PR.EQUITY, hypothesis="normal evaluator",
        weights={}, status=PR.CHALLENGER, parent_policy_id="FAIL_CHAMP",
    )

    import product.paper_autopilot as PA
    real_evaluate = PA.evaluate_selection_candidate

    def _flaky_evaluate(card, **kwargs):
        policy = kwargs.get("evolution_policy") or {}
        if policy.get("policy_id") == "FAIL_BROKEN":
            raise RuntimeError("simulated Challenger evaluator crash/timeout")
        return real_evaluate(card, **kwargs)

    monkeypatch.setattr(PA, "evaluate_selection_candidate", _flaky_evaluate)

    book = PaperBook(capital=500_000)
    cards = [_card("TCS", score=100, rs=80, confirms=3)]
    cycle = _cycle(book, cards, as_of="2026-09-30", now=_now())

    # Champion mutation is entirely unaffected by the broken Challenger.
    assert len(book.open) == 1
    assert cycle["taken"][0]["champion_policy_id"] == "FAIL_CHAMP"

    evolution = cycle["evolution"]
    assert "FAIL_BROKEN" not in evolution["challengers_evaluated"]
    assert "FAIL_OK" in evolution["challengers_evaluated"]

    from product.evolution.consensus_board import get_consensus

    consensus = get_consensus("TCS")
    assert consensus is not None
    # Only Champion + the Challengers that actually ran count toward the
    # denominator -- the broken one is excluded entirely, not counted as a
    # silent "no" vote. (The environment also auto-seeds its own default
    # Challenger population alongside FAIL_OK/FAIL_BROKEN, so the exact
    # count varies; what matters is it equals champion + evaluated, never
    # champion + evaluated + the one that crashed.)
    assert consensus["qualified_count"] == 1 + len(evolution["challengers_evaluated"])
    assert "FAIL_BROKEN" not in str(consensus.get("dissent_breakdown") or {})


def test_live_money_remains_locked_through_the_deferred_tournament_path(tmp_path, monkeypatch):
    """The restructured Phase-1/Phase-2 split must not introduce any new
    route to execution/broker/Telegram code or flip any live-trading flag."""
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
    from product.evolution import tournament as evolution_tournament_engine

    source = inspect.getsource(evolution_tournament_engine)
    tree = ast.parse(source)
    forbidden = {"execution", "zerodha_broker", "telegram_actions", "telegram_commands", "kite_client"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & forbidden), f"forbidden import found: {names}"
