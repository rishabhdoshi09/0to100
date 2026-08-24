"""Paper-memory overlay: daily self-learn on PAPER, live still locked."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from product.paper_learning import (
    LIVE_STILL_LOCKED,
    PROMOTION_LADDER,
    build_paper_memory,
    load_paper_memory,
    remember_paper_book,
    select_paper_signal,
)
from research.auto_research.paper_book import ClosedTrade, PaperBook
from research.autonomy.promotion import LIVE_EXECUTION_LOCKED
from research.intelligence.allocation_brain import AllocationConfig
from research.intelligence.runtime.runtime_state import RuntimeState
from research.intelligence.runtime.target_portfolio import (
    BLOCKED,
    TARGETED,
    build_target_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]


def _ctx(**overrides):
    values = {
        "as_of_date": "2026-08-01",
        "mode": "PAPER_AUTO",
        "data_snapshot_id": "snapshot-1",
        "data_ok": True,
        "market_regime": "RISK_ON",
        "clusters": {},
        "strategies": (),
        "fresh_live_symbols": frozenset(),
        "live_confirmation_required": False,
        "pending_quantities": {},
        "pending_risk_amounts": {},
        "pending_capital_amounts": {},
        "cycle_id": lambda: "cycle-1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _decision(strategy_id="s1", symbol="AAA", risk=0.5, score=1.0, family="momentum"):
    return SimpleNamespace(
        strategy_id=strategy_id,
        strategy_version=1,
        rules_hash=f"rules-{strategy_id}",
        family=family,
        action="DEPLOY",
        target_risk_pct=risk,
        score=score,
        record_id=f"allocation-{strategy_id}",
        reasons=("evidence-qualified",),
        symbol=symbol,
    )


def _card(strategy_id="s1"):
    return SimpleNamespace(strategy_id=strategy_id, record_id=f"card-{strategy_id}")


def _loss(symbol, exit_date, r=-1.0):
    return {
        "symbol": symbol, "pnl": r * 100, "realized_R": r,
        "exit_date": exit_date, "exit_reason": "STOP", "entry_date": exit_date,
    }


def _win(symbol, exit_date, r=1.0):
    return {
        "symbol": symbol, "pnl": r * 100, "realized_R": r,
        "exit_date": exit_date, "exit_reason": "TARGET", "entry_date": exit_date,
    }


def _sig(symbol):
    return {"symbol": symbol, "entry": 100, "stop": 90, "target": 120, "max_hold": 20}


def test_two_consecutive_losses_enter_cooldown():
    memory = build_paper_memory(
        [_loss("AAA", "2026-08-01"), _loss("AAA", "2026-08-02")],
        as_of="2026-08-03",
    )
    assert memory["live_locked"] is True
    assert "AAA" in {row["symbol"] for row in memory["cooldown"]}
    assert memory["prefer"] == []


def test_win_after_losses_is_not_cooldown():
    memory = build_paper_memory(
        [_loss("AAA", "2026-08-01"), _loss("AAA", "2026-08-02"), _win("AAA", "2026-08-03")],
        as_of="2026-08-04",
    )
    assert memory["cooldown"] == []


def test_single_loss_is_recorded_not_banned():
    memory = build_paper_memory([_loss("AAA", "2026-08-01")], as_of="2026-08-02")
    assert memory["cooldown"] == []
    assert memory["closed_trades"] == 1
    assert memory["symbols"][0]["losses"] == 1


def test_prefer_requires_three_trades_and_positive_mean():
    memory = build_paper_memory(
        [_win("BBB", "2026-08-01"), _win("BBB", "2026-08-02"), _win("BBB", "2026-08-03")],
        as_of="2026-08-04",
    )
    assert memory["prefer"] == ["BBB"]
    assert memory["cooldown"] == []


def test_select_skips_cooldown_and_takes_next():
    memory = build_paper_memory(
        [_loss("AAA", "2026-08-01"), _loss("AAA", "2026-08-02")],
        as_of="2026-08-03",
    )
    picked, skipped = select_paper_signal([_sig("AAA"), _sig("BBB")], memory, as_of="2026-08-03")
    assert picked["symbol"] == "BBB"
    assert any(item.startswith("AAA:") for item in skipped)


def test_select_prefers_proven_winner_among_same_strategy_signals():
    memory = build_paper_memory(
        [_win("BBB", "2026-08-01"), _win("BBB", "2026-08-02"), _win("BBB", "2026-08-03")],
        as_of="2026-08-04",
    )
    picked, skipped = select_paper_signal([_sig("AAA"), _sig("BBB")], memory, as_of="2026-08-04")
    assert picked["symbol"] == "BBB"
    assert skipped == ()


def test_select_without_memory_keeps_first_signal():
    picked, skipped = select_paper_signal([_sig("AAA"), _sig("BBB")], None, as_of="2026-08-04")
    assert picked["symbol"] == "AAA"
    assert skipped == ()


def test_target_portfolio_skips_cooldown_name_for_next_signal():
    memory = build_paper_memory(
        [_loss("AAA", "2026-08-01"), _loss("AAA", "2026-08-02")],
        as_of="2026-08-03",
    )
    build = build_target_portfolio(
        _ctx(paper_memory=memory, as_of_date="2026-08-03"),
        book=PaperBook(capital=100_000),
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals={"s1": [_sig("AAA"), _sig("BBB")]},
        cards=[_card()],
        cfg=AllocationConfig(),
    )
    assert len(build.executable) == 1
    assert build.executable[0].symbol == "BBB"
    assert build.executable[0].status == TARGETED
    assert any("AAA:" in reason for reason in build.executable[0].reasons)


def test_target_portfolio_blocks_when_every_name_is_on_cooldown():
    memory = build_paper_memory(
        [_loss("AAA", "2026-08-01"), _loss("AAA", "2026-08-02")],
        as_of="2026-08-03",
    )
    build = build_target_portfolio(
        _ctx(paper_memory=memory, as_of_date="2026-08-03"),
        book=PaperBook(capital=100_000),
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals={"s1": [_sig("AAA")]},
        cards=[_card()],
        cfg=AllocationConfig(),
    )
    assert len(build.executable) == 0
    assert build.positions[0].symbol == "AAA"
    assert build.positions[0].status == BLOCKED
    assert "PAPER_LESSON_COOLDOWN" in build.positions[0].blocked_by


def test_target_portfolio_without_memory_keeps_first_signal():
    build = build_target_portfolio(
        _ctx(),
        book=PaperBook(capital=100_000),
        runtime_state=RuntimeState(),
        decisions=[_decision(risk=0.5)],
        today_signals={"s1": [_sig("AAA"), _sig("BBB")]},
        cards=[_card()],
        cfg=AllocationConfig(),
    )
    assert build.executable[0].symbol == "AAA"


def test_run_learning_writes_symbol_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_MEMORY", str(tmp_path / "paper_memory.json"))
    book = SimpleNamespace(closed=[
        ClosedTrade("s1", "AAA", 100, 90, 90, 10, "2026-08-01", "2026-08-02", "STOP", -1.0, -100.0),
        ClosedTrade("s1", "AAA", 100, 90, 90, 10, "2026-08-03", "2026-08-04", "STOP", -1.0, -100.0),
    ])
    brain = SimpleNamespace(
        intel_book=book,
        knowledge=SimpleNamespace(remember_forward=lambda *a, **k: None, save=lambda: None),
        strategy_registry=None,
    )
    from research.autonomy.research_loop import run_learning
    out = run_learning(brain, session_date="2026-08-05")
    assert out["paper_closed"] == 2
    assert out["paper_cooldown"] == 1
    loaded = load_paper_memory()
    assert loaded["cooldown"][0]["symbol"] == "AAA"


def test_remember_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_PAPER_MEMORY", str(tmp_path / "paper_memory.json"))
    memory = remember_paper_book([_win("INFY", "2026-08-01", 0.8)], as_of="2026-08-02")
    loaded = load_paper_memory()
    assert loaded["closed_trades"] == memory["closed_trades"] == 1
    assert loaded["live_locked"] is True
    assert LIVE_STILL_LOCKED in loaded["disclaimer"]


def test_learning_job_summary_includes_paper_memory_counts():
    from research.autonomy import jobs as JOBS
    src = Path(JOBS.__file__).read_text(encoding="utf-8")
    assert "paper_cooldown" in src
    assert "paper_prefer" in src
    assert "place_order" not in src


def test_live_stays_locked_and_no_broker_or_workers():
    assert LIVE_EXECUTION_LOCKED is True
    src = (ROOT / "product" / "paper_learning.py").read_text(encoding="utf-8")
    assert ".start()" not in src
    assert "start_background_scan" not in src
    assert "place_order" not in src
    assert "LIVE_STILL_LOCKED" in src
    assert PROMOTION_LADDER
    ui = (ROOT / "ui" / "desk_board.py").read_text(encoding="utf-8")
    assert "render_bot_learning" in ui
    assert "start_background_scan" not in ui
