"""Checkpointed batch driver for F&O historical walk-forward simulation.

Proves the specific operational properties required of a job this safe to
run unattended, off-market, repeatedly:
  - never reprocesses an already-graded session
  - a bounded batch size per call (never one giant blocking run)
  - honest status when there is no universe / no history / nothing new yet
  - one symbol's failure does not lose the whole batch
  - the cursor and counters survive a fresh reload (restart durability)
  - it only ever produces COUNTERFACTUAL evidence, never touches live money
"""
from __future__ import annotations

import product.conditional_evidence as CE
from product.evidence_class import COUNTERFACTUAL
from product.fno_historical_loop import load_checkpoint, run_next_batch, status
from tests.test_fno_walk_forward_learning import _repeated_losing_breakouts


def _provider_for(frames: dict[str, "object"]):
    def provider(symbol: str):
        return frames.get(symbol)
    return provider


def test_no_universe_is_reported_honestly(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    result = run_next_batch(universe=[], path=path)
    assert result["status"] == "NO_UNIVERSE"
    assert result["candidates_evaluated"] == 0
    s = status(path)
    assert s["available"] is False, "a call that did no real work must not claim to have run"


def test_no_history_is_reported_honestly(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    result = run_next_batch(universe=["RELIANCE"], history_provider=lambda s: None, path=path)
    assert result["status"] == "NO_HISTORY"


def test_first_run_processes_a_bounded_batch_and_advances_the_cursor(tmp_path, monkeypatch):
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))
    checkpoint_path = tmp_path / "checkpoint.json"

    from datetime import date
    frame, breakout_dates = _repeated_losing_breakouts(10, start=date(2026, 1, 1))
    provider = _provider_for({"LOSER": frame})

    result = run_next_batch(
        universe=["LOSER"], history_provider=provider,
        max_sessions_per_run=999,  # 10 cycles * 23 calendar days: let one call cover it all
        path=checkpoint_path,
    )
    assert result["status"] == "OK"
    assert result["candidates_evaluated"] == 10
    assert result["settled"] == 10
    assert result["errors"] == []

    s = status(checkpoint_path)
    assert s["available"] is True
    assert s["total_candidates_evaluated"] == 10
    assert s["total_settled"] == 10
    assert s["evidence_class"] == "COUNTERFACTUAL"
    assert s["affects_ranking"] is False
    assert s["affects_live_money"] is False

    # The evidence really landed where fno_context_key says it should.
    from product.fno_historical_walkforward import evaluate_point_in_time_candidate
    sample = evaluate_point_in_time_candidate("LOSER", breakout_dates[0], frame)
    cell = CE.read(sample.context_key, evidence_class=COUNTERFACTUAL, path=evidence_path)
    assert cell["count"] == 10

    # Every one of these engineered breakouts crashes on the very next bar --
    # a real, honestly graded outcome the forward move went against, not a
    # fabricated label.
    assert sum(result["classification_counts"].values()) == 10
    assert sum(s["classification_counts"].values()) == 10


def test_repeated_calls_never_reprocess_the_same_session(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    checkpoint_path = tmp_path / "checkpoint.json"

    from datetime import date
    # 6 cycles * 23 calendar days = 138 days; batches of 10 days each mean a
    # cycle's own breakout (needs ~22 quiet days first) will not necessarily
    # land in the very first batch -- that is realistic, not a bug.
    frame, _ = _repeated_losing_breakouts(6, start=date(2026, 1, 1))
    provider = _provider_for({"LOSER": frame})

    seen_sessions: set[str] = set()
    cursors: list[str] = []
    total_evaluated = 0
    exhausted = None
    for _ in range(30):
        result = run_next_batch(universe=["LOSER"], history_provider=provider,
                                 max_sessions_per_run=10, path=checkpoint_path)
        assert seen_sessions.isdisjoint(result["sessions_processed"]), (
            "no session may be graded twice"
        )
        seen_sessions.update(result["sessions_processed"])
        total_evaluated += result["candidates_evaluated"]
        if result["status"] == "UP_TO_DATE":
            exhausted = result
            break
        if result["cursor_date"]:
            cursors.append(result["cursor_date"])

    assert exhausted is not None, "the batch loop must eventually reach UP_TO_DATE"
    assert exhausted["sessions_processed"] == []
    assert cursors == sorted(cursors), "the cursor must move forward monotonically, never repeat"
    assert len(cursors) == len(set(cursors)), "the same cursor value must never be reported twice"
    assert total_evaluated == 6, "all six engineered breakouts must be found exactly once, cumulatively"


def test_awaiting_forward_bar_when_nothing_can_be_settled_yet(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    checkpoint_path = tmp_path / "checkpoint.json"

    from datetime import date
    # A single cycle's frame has 23 calendar rows total. Demanding a horizon
    # longer than the entire frame means literally nothing on disk has a
    # real forward bar far enough out to settle against yet.
    frame, _ = _repeated_losing_breakouts(1, start=date(2026, 1, 1))
    provider = _provider_for({"LOSER": frame})

    result = run_next_batch(universe=["LOSER"], history_provider=provider,
                             horizon_sessions=30, path=checkpoint_path)
    assert result["status"] == "AWAITING_FORWARD_BAR"


def test_one_symbols_failure_does_not_lose_the_whole_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    checkpoint_path = tmp_path / "checkpoint.json"

    from datetime import date
    good_frame, _ = _repeated_losing_breakouts(5, start=date(2026, 1, 1))

    class _BoomFrame:
        @property
        def index(self):
            raise RuntimeError("simulated corrupt frame")
        empty = False

    provider = _provider_for({"GOOD": good_frame, "BROKEN": _BoomFrame()})

    result = run_next_batch(universe=["GOOD", "BROKEN"], history_provider=provider,
                             max_sessions_per_run=999, path=checkpoint_path)
    # GOOD's candidates still get evaluated and settled despite BROKEN.
    assert result["candidates_evaluated"] == 5
    assert result["settled"] == 5
    assert result["status"] == "PARTIAL_ERROR"
    assert any("BROKEN" in e for e in result["errors"]), "the failure must be visible, not swallowed"

    s = status(checkpoint_path)
    assert s["total_candidates_evaluated"] == 5
    assert "BROKEN" in s["last_error"]


def test_checkpoint_survives_a_fresh_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "evidence.json"))
    checkpoint_path = tmp_path / "checkpoint.json"

    from datetime import date
    frame, _ = _repeated_losing_breakouts(4, start=date(2026, 1, 1))
    provider = _provider_for({"LOSER": frame})
    run_next_batch(universe=["LOSER"], history_provider=provider,
                    max_sessions_per_run=999, path=checkpoint_path)

    before = load_checkpoint(checkpoint_path)
    # Simulate a process restart: nothing but the file on disk carries state.
    after = load_checkpoint(checkpoint_path)
    assert after == before
    assert after["total_candidates_evaluated"] == 4
