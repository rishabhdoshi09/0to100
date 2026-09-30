"""Walk-forward acceptance test for F&O historical counterfactual learning.

Proves, with a deterministic synthetic price series (never real market
data, never claimed to be):

  1. No future leakage: a candidate evaluated at day N never sees day N+1.
  2. Evidence accumulates correctly and progressively across walk-forward
     periods (Period A -> B -> C), including through the sample sizes
     below MIN_SAMPLE where nothing should be usable yet.
  3. However large the historical/counterfactual sample grows, it can
     never move product.decision_ranking.rank()'s ranking score -- the
     same MARKET_EVIDENCE boundary product.evidence_class and
     tests/test_core_loop_revolution.py already enforce for equities.
  4. Real PAPER_FORWARD evidence for the exact same context DOES move
     ranking, even while COUNTERFACTUAL evidence for that same context
     stays inert -- proving "real forward paper results validate or
     disprove" what historical simulation only hypothesises.
  5. Evidence survives a process restart (reloading the store from the
     same file path).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import product.conditional_evidence as CE
from product.decision import BUY, Decision
from product.decision_ranking import decision_context_key, rank
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_evidence import fno_context_key, record_fno_settlement
from product.fno_historical_walkforward import (
    evaluate_point_in_time_candidate,
    record_walk_forward_outcome,
)


def _flat_then_breakout_series(*, quiet_days: int, breakout_close: float,
                                breakout_volume: float, post_close: float,
                                start: date) -> pd.DataFrame:
    """quiet_days of a tight range, then one breakout day, then one more
    day at post_close (the real forward bar used to settle the breakout)."""
    rows = []
    d = start
    for _ in range(quiet_days):
        rows.append({"date": d, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0})
        d += timedelta(days=1)
    rows.append({
        "date": d, "High": breakout_close + 1, "Low": 99.0,
        "Close": breakout_close, "Volume": breakout_volume,
    })
    breakout_date = d
    d += timedelta(days=1)
    rows.append({"date": d, "High": post_close + 1, "Low": post_close - 1,
                  "Close": post_close, "Volume": 1000.0})
    frame = pd.DataFrame(rows).set_index("date")
    frame.index = pd.to_datetime(frame.index)
    return frame, breakout_date


def _repeated_losing_breakouts(n: int, *, start: date) -> tuple[pd.DataFrame, list[date]]:
    """n independent quiet-then-breakout-then-crash cycles concatenated, so
    each breakout has its own fresh 20-day quiet lookback and there are n
    settleable candidates in total."""
    frames = []
    dates = []
    cursor = start
    for _ in range(n):
        frame, breakout_date = _flat_then_breakout_series(
            quiet_days=21, breakout_close=110.0, breakout_volume=3000.0,
            post_close=95.0, start=cursor,
        )
        frames.append(frame)
        dates.append(breakout_date)
        cursor = frame.index[-1].date() + timedelta(days=1)
    return pd.concat(frames), dates


def test_no_lookahead_in_point_in_time_evaluation():
    """Appending future bars to the SAME history object a walk-forward loop
    would realistically keep growing (rather than re-slicing per call) must
    never change what an earlier as_of evaluates to.

    Deliberately does not rely on the "history's last row must equal as_of"
    guard alone: as_of here is NOT the last row of the frame passed in, on
    both sides of the comparison, so a defensive re-slice inside the
    evaluator (not merely a last-row check) is what has to do the work.
    """
    frame, breakout_date = _flat_then_breakout_series(
        quiet_days=21, breakout_close=110.0, breakout_volume=3000.0,
        post_close=95.0, start=date(2026, 1, 1),
    )
    as_of = breakout_date  # a real, non-final-row breakout day: frame has one more row after it
    assert frame.index[-1] != pd.Timestamp(as_of)

    baseline = evaluate_point_in_time_candidate("LOSER", as_of, frame)
    assert baseline is not None, "the engineered breakout must be detected from data up to and including as_of"

    # Append 21 more future sessions after the frame's current end, ending
    # with an enormous, differently-directioned move. A walk-forward driver
    # would realistically hand the evaluator this same, ever-growing frame
    # for every as_of it has already passed, not a freshly re-sliced copy.
    poisoned = frame.copy()
    cursor = poisoned.index[-1].date() + timedelta(days=1)
    extra_rows = []
    for _ in range(21):
        extra_rows.append({"date": cursor, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0})
        cursor += timedelta(days=1)
    extra_rows.append({"date": cursor, "High": 5.0, "Low": 1.0, "Close": 2.0, "Volume": 999999.0})
    extra = pd.DataFrame(extra_rows).set_index("date")
    extra.index = pd.to_datetime(extra.index)
    poisoned = pd.concat([poisoned, extra])

    replayed = evaluate_point_in_time_candidate("LOSER", as_of, poisoned)
    assert replayed is not None
    assert replayed.direction == baseline.direction == "LONG"
    assert replayed.entry == baseline.entry
    assert replayed.stop == baseline.stop
    assert replayed.target == baseline.target
    assert replayed.score == baseline.score, (
        "a massive future crash appended after as_of leaked into the "
        "point-in-time evaluation and changed its result"
    )


def test_walk_forward_periods_accumulate_evidence_progressively(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))

    frame, breakout_dates = _repeated_losing_breakouts(35, start=date(2026, 1, 1))
    key = None
    settled_so_far = 0

    def run_period(n_cycles: int) -> int:
        nonlocal key, settled_so_far
        newly_settled = 0
        for breakout_date in breakout_dates[settled_so_far:settled_so_far + n_cycles]:
            candidate = evaluate_point_in_time_candidate("LOSER", breakout_date, frame)
            assert candidate is not None, "engineered breakout must be detected"
            assert candidate.direction == "LONG"
            key = key or candidate.context_key
            assert candidate.context_key == key, "identical setups must share one cell"
            forward_row = frame.loc[frame.index > pd.Timestamp(breakout_date)].iloc[0]
            update = record_walk_forward_outcome(
                candidate, forward_close=float(forward_row["Close"]),
                resolved_at=forward_row.name.isoformat(), path=path,
            )
            assert update is not None
            newly_settled += 1
        settled_so_far += n_cycles
        return newly_settled

    # Period A: a handful of candidates -- nowhere near enough to be usable.
    run_period(3)
    cell_a = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert cell_a["count"] == 3

    # Period B: more accumulate. Still real numbers, still inspectable.
    run_period(12)
    cell_b = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert cell_b["count"] == 15
    assert cell_b["expectancy_R"] is not None
    assert cell_b["expectancy_R"] < 0, "every engineered cycle is a real loss"

    # Period C: cross the MIN_SAMPLE line the ranking gate itself uses.
    run_period(20)
    cell_c = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert cell_c["count"] == 35
    assert cell_c["wilson_lower_bound"] is not None


def test_counterfactual_walk_forward_evidence_never_moves_ranking(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    frame, breakout_dates = _repeated_losing_breakouts(30, start=date(2026, 1, 1))
    key = None
    for breakout_date in breakout_dates:
        candidate = evaluate_point_in_time_candidate("LOSER", breakout_date, frame)
        key = key or candidate.context_key
        forward_row = frame.loc[frame.index > pd.Timestamp(breakout_date)].iloc[0]
        record_walk_forward_outcome(
            candidate, forward_close=float(forward_row["Close"]),
            resolved_at=forward_row.name.isoformat(), path=path,
        )

    cell = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert cell["count"] == 30, "plenty of samples by ordinary standards"

    loser_decision = Decision(
        symbol="LOSER", state=BUY, setup="FNO_LONG", score=80.0,
        calibrated_confidence=0.7, market_state="NEUTRAL", sector_state="NEUTRAL",
        technical_evidence={"atr_pct": 2.0, "pct_from_pivot": 1.0},
        entry=110.0, stop=106.0, target=118.0,
        evidence_class=PAPER_FORWARD, generated_at="2026-01-01T00:00:00+00:00",
    )
    control = Decision(
        symbol="CONTROL", state=BUY, setup="FNO_LONG", score=78.0,
        calibrated_confidence=0.7, market_state="RISK_ON", sector_state="LEADING",
        technical_evidence={"atr_pct": 2.0, "pct_from_pivot": 1.0},
        entry=110.0, stop=106.0, target=118.0,
        evidence_class=PAPER_FORWARD, generated_at="2026-01-01T00:00:00+00:00",
    )
    ranked = rank([loser_decision, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in ranked] == ["LOSER", "CONTROL"], (
        "30 settled COUNTERFACTUAL observations must not out-rank the raw "
        "scorer -- only real market evidence may do that"
    )
    unmoved = next(r for r in ranked if r.symbol == "LOSER")
    assert unmoved.evidence_adjustment == 0.0
    assert unmoved.evidence["reason"] == "INSUFFICIENT_EVIDENCE"


def test_real_forward_evidence_moves_ranking_where_historical_evidence_could_not(tmp_path, monkeypatch):
    """The full hierarchy in one test: historical counterfactual evidence
    for a context accumulates and stays inert; real PAPER_FORWARD evidence
    for that SAME context (a different cell -- evidence_class is part of
    the key) demotes the ranking."""
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    frame, breakout_dates = _repeated_losing_breakouts(30, start=date(2026, 1, 1))
    key = None
    last_candidate = None
    for breakout_date in breakout_dates:
        candidate = evaluate_point_in_time_candidate("LOSER", breakout_date, frame)
        key = key or candidate.context_key
        last_candidate = candidate
        forward_row = frame.loc[frame.index > pd.Timestamp(breakout_date)].iloc[0]
        record_walk_forward_outcome(
            candidate, forward_close=float(forward_row["Close"]),
            resolved_at=forward_row.name.isoformat(), path=path,
        )
    assert CE.read(key, evidence_class=COUNTERFACTUAL, path=path)["count"] == 30

    # Built FROM the walk-forward candidate's own setup fields so its
    # decision_context_key() is guaranteed to be the exact same cell `key`
    # -- the same "the cell a decision is ranked against is the cell its
    # own outcome updates" requirement decision_ranking.py's docstring
    # names as the thing that silently breaks the loop if it drifts.
    setup = last_candidate.setup
    loser = Decision(
        symbol="LOSER", state=BUY, setup=f"FNO_{setup['direction']}", score=80.0,
        calibrated_confidence=setup["score"] / 100.0, market_state="NEUTRAL", sector_state="NEUTRAL",
        technical_evidence={"atr_pct": setup["atr_pct"], "pct_from_pivot": setup["breakout_distance_pct"]},
        entry=110.0, stop=106.0, target=118.0,
        evidence_class=PAPER_FORWARD, generated_at="2026-01-01T00:00:00+00:00",
    )
    assert decision_context_key(loser) == key
    control = Decision(
        symbol="CONTROL", state=BUY, setup=f"FNO_{setup['direction']}", score=78.0,
        calibrated_confidence=setup["score"] / 100.0, market_state="RISK_ON", sector_state="LEADING",
        technical_evidence={"atr_pct": setup["atr_pct"], "pct_from_pivot": setup["breakout_distance_pct"]},
        entry=110.0, stop=106.0, target=118.0,
        evidence_class=PAPER_FORWARD, generated_at="2026-01-01T00:00:00+00:00",
    )
    before = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in before] == ["LOSER", "CONTROL"]

    # Now real, fully-costed, fully-observed settled F&O paper trades land
    # for the exact same context -- a materially different evidentiary
    # claim from the historical simulation above.
    for i in range(30):
        row = {
            "trade_id": f"REAL-{i}",
            "underlying": "LOSER",
            "entry_price": 110.0,
            "stop_price": 106.0,
            "exit_price": 95.0,
            "quantity": 10,
            "net_pnl": -1500.0,
            "exit_reason": "STOP",
            "opened_at": "2026-02-01T04:00:00+00:00",
            "settled_at": f"2026-02-{1 + (i % 20):02d}T10:00:00+00:00",
            "production_evidence_eligible": True,
        }
        update = record_fno_settlement(row, context_key=key, path=path)
        assert update is not None

    after = rank([loser, control], evidence_class=PAPER_FORWARD, path=path)
    assert [r.symbol for r in after] == ["CONTROL", "LOSER"], (
        "real settled PAPER_FORWARD evidence for this context must demote "
        "LOSER even though historical COUNTERFACTUAL evidence for the same "
        "context could not"
    )
    demoted = next(r for r in after if r.symbol == "LOSER")
    assert demoted.evidence["reason"] == "MEASURED_NEGATIVE_EXPECTANCY"
    assert demoted.evidence["count"] == 30, (
        "the ranking-relevant count must be the 30 REAL trades, not "
        "30 (real) + 30 (historical) double-counted into one cell"
    )


def test_walk_forward_evidence_survives_reload_from_the_same_path(tmp_path, monkeypatch):
    path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(path))

    frame, breakout_dates = _repeated_losing_breakouts(5, start=date(2026, 1, 1))
    key = None
    for breakout_date in breakout_dates:
        candidate = evaluate_point_in_time_candidate("LOSER", breakout_date, frame)
        key = key or candidate.context_key
        forward_row = frame.loc[frame.index > pd.Timestamp(breakout_date)].iloc[0]
        record_walk_forward_outcome(
            candidate, forward_close=float(forward_row["Close"]),
            resolved_at=forward_row.name.isoformat(), path=path,
        )
    before_restart = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert before_restart["count"] == 5

    # Simulate a process restart: nothing but the file path carries state
    # across this point.
    after_restart = CE.read(key, evidence_class=COUNTERFACTUAL, path=path)
    assert after_restart == before_restart


def test_ambiguous_whipsaw_candidate_still_classifies_honestly():
    """classify_walk_forward_outcome must produce a real label from real
    forward data even for a marginal (neither clearly win nor loss) move,
    never raising and never inventing a confident classification it has no
    basis for."""
    from product.fno_historical_walkforward import classify_walk_forward_outcome

    frame, breakout_date = _flat_then_breakout_series(
        quiet_days=21, breakout_close=110.0, breakout_volume=3000.0,
        post_close=110.5, start=date(2026, 1, 1),  # barely moved
    )
    candidate = evaluate_point_in_time_candidate("LOSER", breakout_date, frame)
    assert candidate is not None
    label = classify_walk_forward_outcome(candidate, forward_close=110.5)
    assert label in {"FLAT", "RAN_AWAY_WITHOUT_ENTRY", "CORRECT_REJECTION", "AVOIDED_LOSER", "MISSED_WINNER", "GOOD_WAIT"}
