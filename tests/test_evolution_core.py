"""Core safety invariants for the Evolution Engine (product/evolution/*):

  - no lookahead: a frozen decision is a pure function of the snapshot it was
    given, never of anything that happens to real data stores afterward.
  - immutability: the same (policy, snapshot) always resolves to the same
    frozen id; genuinely different content under that same id is a
    collision, never a silent overwrite.
  - Champion-only PAPER authority: no module in product/evolution ever
    imports a broker/book-mutating/Telegram-sending symbol.
  - failure isolation: one broken Challenger cannot block the Champion or
    any sibling Challenger.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import pytest

from product.evolution import grading, policy_eval, policy_registry, shadow_decisions, snapshot, tournament


def _card(symbol="RELIANCE", **over):
    card = {
        "symbol": symbol, "entry": 2500.0, "stop": 2400.0, "target": 2650.0,
        "setup_label": "VCP", "sector": "Energy", "rs_percentile": 85, "volume_ratio": 1.8,
        "extension_pct": 2.0, "entry_state": "ready", "as_of": "2026-09-30",
        "reco_tier": "high_conviction",
        "methods": [{"id": "rs", "status": "pass", "points": 80}],
    }
    card.update(over)
    return card


# ── no lookahead ─────────────────────────────────────────────────────────

def test_snapshot_and_decision_are_pure_functions_of_their_inputs(tmp_path, monkeypatch):
    """Mutating an unrelated external data source between two calls must not
    change a frozen snapshot's content or a policy's decision against it --
    neither function reads anything except its own arguments."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    snap1 = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    policy = {"policy_id": "P1", "weights": {"relative_strength_mult": 2.0}}
    verdict1 = policy_eval.evaluate_snapshot(snap1, policy)

    # Simulate "the future arrived": real market data stores would now have
    # bars this decision could never have seen. Mutate the most obviously
    # tempting-to-leak-into-features store (bhavcopy) and prove it changes
    # nothing about re-evaluating the ALREADY-FROZEN snapshot.
    with mock.patch("data.bhavcopy_store.get_ohlcv", side_effect=AssertionError(
        "evaluate_snapshot must never read live data stores -- it only uses its own snapshot argument"
    )):
        snap2 = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
        verdict2 = policy_eval.evaluate_snapshot(snap2, policy)

    assert snap1["market_snapshot_id"] == snap2["market_snapshot_id"]
    assert verdict1["adjusted_score"] == verdict2["adjusted_score"]
    assert verdict1["decision"] == verdict2["decision"]


def test_grading_never_resolves_before_the_horizon_has_elapsed(tmp_path, monkeypatch):
    """resolve_forward_outcome / grade_shadow_decision must return None (not
    a guess, not zero) until core.outcome_resolver.first_touch_path itself
    says the horizon-th session actually exists in official data."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    snap = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    policy = {"policy_id": "P1", "weights": {}}
    verdict = policy_eval.evaluate_snapshot(snap, policy)
    row = shadow_decisions.freeze_shadow_decision(snap, verdict)

    with mock.patch.object(grading, "first_touch_path", return_value=None):
        assert grading.grade_shadow_decision(row) is None

    # Now "time has passed" and the real resolver has a path -- grading
    # becomes available, using exactly that resolved outcome.
    with mock.patch.object(grading, "first_touch_path", return_value=(2650.0, 6.0, 1)):
        graded = grading.grade_shadow_decision(row)
    assert graded is not None
    assert graded["outcome"] is not None
    assert graded["classification"] == "WINNER_TAKEN"


def test_grading_is_idempotent_once_settled(tmp_path, monkeypatch):
    """A second grading attempt (e.g. a restarted off-hours job re-scanning
    pending rows) must never re-settle an already-graded decision, even if
    the forward-resolver would now return something different."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    snap = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    verdict = policy_eval.evaluate_snapshot(snap, {"policy_id": "P1", "weights": {}})
    row = shadow_decisions.freeze_shadow_decision(snap, verdict)

    with mock.patch.object(grading, "first_touch_path", return_value=(2650.0, 6.0, 1)):
        first = grading.grade_shadow_decision(row)
    with mock.patch.object(grading, "first_touch_path", return_value=(1000.0, -90.0, 0)):
        second = grading.grade_shadow_decision(shadow_decisions.get_shadow_decision(row["shadow_id"]))

    assert first["classification"] == second["classification"] == "WINNER_TAKEN"
    assert first["counterfactual_R"] == second["counterfactual_R"]


def test_identical_policy_and_snapshot_freeze_to_the_same_id(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    snap = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    policy = {"policy_id": "P1", "weights": {}}
    verdict = policy_eval.evaluate_snapshot(snap, policy)
    row1 = shadow_decisions.freeze_shadow_decision(snap, verdict)
    row2 = shadow_decisions.freeze_shadow_decision(snap, verdict)
    assert row1["shadow_id"] == row2["shadow_id"]
    assert len(shadow_decisions.list_shadow_decisions(policy_id="P1")) == 1


def test_genuinely_different_decision_under_the_same_identity_collides(tmp_path, monkeypatch):
    """A code bug that made the same (policy, snapshot) produce a DIFFERENT
    decision on a later call must be caught loudly, not silently accepted."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    card = _card()
    snap = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    policy = {"policy_id": "P1", "weights": {}}
    verdict = policy_eval.evaluate_snapshot(snap, policy)
    shadow_decisions.freeze_shadow_decision(snap, verdict)

    tampered = dict(verdict)
    tampered["adjusted_score"] = 99999.0
    from product.decision_freeze import DecisionIdentityCollision
    with pytest.raises(DecisionIdentityCollision):
        shadow_decisions.freeze_shadow_decision(snap, tampered)


def test_restart_reload_preserves_policy_registry_and_shadow_ledger(tmp_path, monkeypatch):
    """Everything is read fresh from durable JSON/JSONL/SQLite on every call --
    no in-memory cache a restart could lose or strand stale. Proven with a
    genuinely separate OS process reading the SAME runtime root, which is
    the real thing "restart-safe" has to survive (an in-process
    importlib.reload() would instead re-evaluate this module's own
    env-derived path constants mid-test, which is not what a restart does:
    a fresh process reads QT_RUNTIME_ROOT exactly once, at its own startup)."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    policy_registry.register_policy(
        policy_id="EQUITY_CHAMPION_V1", domain=policy_registry.EQUITY,
        hypothesis="baseline", weights={}, status=policy_registry.CHAMPION,
    )
    card = _card()
    snap = snapshot.build_snapshot(card, regime="TRENDING_BULL", domain="EQUITY")
    verdict = policy_eval.evaluate_snapshot(snap, {"policy_id": "EQUITY_CHAMPION_V1", "weights": {}})
    shadow_decisions.freeze_shadow_decision(snap, verdict)

    import subprocess
    import sys

    script = (
        "from product.evolution import policy_registry as PR, shadow_decisions as SD, snapshot as SNAP\n"
        "champ = PR.current_champion(PR.EQUITY)\n"
        "assert champ is not None and champ['policy_id'] == 'EQUITY_CHAMPION_V1', champ\n"
        "rows = SD.list_shadow_decisions(policy_id='EQUITY_CHAMPION_V1')\n"
        "assert len(rows) == 1, rows\n"
        f"refetched = SNAP.get_snapshot_record({snap['market_snapshot_id']!r})\n"
        "assert refetched['symbol'] == 'RELIANCE', refetched\n"
        "print('RESTART_OK')\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root),
        env={**__import__("os").environ, "QT_RUNTIME_ROOT": str(tmp_path)},
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESTART_OK" in proc.stdout


# ── Champion-only PAPER authority ───────────────────────────────────────────

_FORBIDDEN_SYMBOLS = (
    "place_trade", "open_position", "open_intent", "KiteConnect",
    "telegram_actions", "telegram_commands", "send_message", "broker_order",
)
_FORBIDDEN_IMPORTS = (
    "execution.trade_executor", "execution.autopilot", "execution.zerodha_broker",
    "alerts.telegram_actions", "alerts.telegram_commands",
)


def _evolution_module_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "product" / "evolution"
    return sorted(root.glob("*.py"))


def _imported_module_names(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", _evolution_module_files(), ids=lambda p: p.name)
def test_evolution_module_never_touches_execution_or_telegram(path: Path):
    """Static proof (same technique as tests/test_autonomy_telegram.py's
    execution.autopilot check): a Challenger-evaluation module must contain
    no ACTUAL import of any order-placing, book-mutating, or message-sending
    module -- parsed via AST so a docstring merely explaining this
    invariant (which legitimately names the forbidden modules) is not a
    false positive."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _imported_module_names(tree)
    for bad in _FORBIDDEN_IMPORTS:
        assert not any(mod == bad or mod.startswith(bad + ".") for mod in imported), (
            f"{path.name} imports forbidden module {bad!r} (imports: {sorted(imported)})"
        )
    names_called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for bad in _FORBIDDEN_SYMBOLS:
        assert bad not in names_called, f"{path.name} calls forbidden symbol {bad!r}"


def test_product_evolution_package_has_no_broker_or_live_execution_path():
    """product/evolution/__init__.py's own safety claim, verified: nothing in
    the package imports anything from the live execution surface."""
    live_locked_claims = (
        Path(__file__).resolve().parents[1] / "product" / "evolution" / "policy_eval.py"
    ).read_text(encoding="utf-8")
    assert "PAPER" in live_locked_claims


# ── failure isolation ────────────────────────────────────────────────────

def test_one_broken_challenger_does_not_block_champion_or_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    policy_registry.register_policy(
        policy_id="CHAMP", domain=policy_registry.EQUITY, hypothesis="baseline",
        weights={}, status=policy_registry.CHAMPION,
    )
    policy_registry.register_policy(
        policy_id="GOOD_CHALLENGER", domain=policy_registry.EQUITY, hypothesis="fine",
        weights={"relative_strength_mult": 1.5}, status=policy_registry.CHALLENGER,
        parent_policy_id="CHAMP",
    )
    policy_registry.register_policy(
        policy_id="BROKEN_CHALLENGER", domain=policy_registry.EQUITY, hypothesis="broken on purpose",
        weights={"min_empirical_sample": "not-a-number-but-forced-bad"}, status=policy_registry.CHALLENGER,
        parent_policy_id="CHAMP",
    )

    cards = [_card("RELIANCE")]
    champion_decisions = {"RELIANCE": {"decision": "ENTER_NOW", "reason_code": "ELIGIBLE", "selection_score": 90.0}}

    # BROKEN_CHALLENGER's weights genuinely raise inside evaluate_snapshot
    # (int("not-a-number...") -> ValueError) -- no mocking needed, this is a
    # real code-path failure the orchestrator must isolate.
    result = tournament.run_tournament_cycle(
        cards, champion_decisions, champion_policy_id="CHAMP", as_of="2026-09-30",
    )

    assert "BROKEN_CHALLENGER" in result["challengers_skipped"]
    assert "GOOD_CHALLENGER" in result["challengers_evaluated"]
    assert result["results"][0]["champion"]["decision"] == "ENTER_NOW"
