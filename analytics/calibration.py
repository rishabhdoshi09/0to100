"""
Conviction Score Calibration — are our confidence scores accurate?

When ConvictionScorer says 80, it should win ~80% of the time.
Systematic over/under-confidence is detected and reported here.

Run: python -m analytics.calibration
Or call compute_calibration() from journal_analytics UI.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_BUCKETS = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


@dataclass
class CalibrationBucket:
    score_lo: int
    score_hi: int
    trade_count: int
    claimed_win_rate: float   # midpoint of bucket / 100
    actual_win_rate: float
    calibration_error: float  # actual - claimed (positive = better than claimed)
    grade: str                # WELL_CALIBRATED | OVERCONFIDENT | UNDERCONFIDENT


@dataclass
class CalibrationReport:
    buckets: list[CalibrationBucket]
    overall_brier_score: float   # lower is better; 0.25 = random, 0.0 = perfect
    verdict: str                 # one-line summary
    sample_size: int


def compute_calibration(db_path: Optional[str] = None) -> Optional[CalibrationReport]:
    """
    Read trade journal from SQLite, compute calibration per score bucket.
    Returns None if insufficient data (< 20 closed trades).
    """
    path = db_path or str(Path("logs") / "jarvis_memory.db")
    try:
        con = sqlite3.connect(path)
        rows = con.execute(
            "SELECT conviction_score, pnl FROM trade_journal "
            "WHERE pnl IS NOT NULL AND conviction_score IS NOT NULL"
        ).fetchall()
        con.close()
    except Exception as e:
        log.debug("calibration_db_error", error=str(e))
        # Try paper trading journal as fallback
        try:
            from paper_trading import PaperTrader
            trades = PaperTrader().get_closed_trades()
            rows = [(t.get("conviction_score", 70), t.get("pnl", 0)) for t in trades
                    if t.get("pnl") is not None]
        except Exception:
            return None

    if len(rows) < 20:
        log.info("calibration_insufficient_data", trades=len(rows))
        return None

    buckets: list[CalibrationBucket] = []
    brier_sum = 0.0

    for lo, hi in _BUCKETS:
        in_bucket = [(score, pnl) for score, pnl in rows if lo <= score < hi]
        if not in_bucket:
            continue
        n = len(in_bucket)
        claimed = (lo + min(hi, 100)) / 2 / 100
        wins = sum(1 for _, pnl in in_bucket if pnl > 0)
        actual = wins / n
        error = actual - claimed
        # Brier contribution
        for score, pnl in in_bucket:
            predicted = score / 100
            outcome = 1.0 if pnl > 0 else 0.0
            brier_sum += (predicted - outcome) ** 2

        if abs(error) < 0.08:
            grade = "WELL_CALIBRATED"
        elif error < -0.08:
            grade = "OVERCONFIDENT"
        else:
            grade = "UNDERCONFIDENT"

        buckets.append(CalibrationBucket(
            score_lo=lo, score_hi=hi, trade_count=n,
            claimed_win_rate=claimed, actual_win_rate=actual,
            calibration_error=error, grade=grade,
        ))

    if not buckets:
        return None

    brier = brier_sum / len(rows)
    overconfident = sum(1 for b in buckets if b.grade == "OVERCONFIDENT")
    if brier < 0.15:
        verdict = "Excellent calibration — scores reflect reality."
    elif overconfident >= 2:
        verdict = f"Overconfident in {overconfident} buckets — reduce position sizes by 10-15%."
    elif brier < 0.22:
        verdict = "Reasonable calibration — minor adjustments suggested."
    else:
        verdict = "Poor calibration — conviction scores not reliable. Review signal logic."

    return CalibrationReport(
        buckets=buckets,
        overall_brier_score=round(brier, 4),
        verdict=verdict,
        sample_size=len(rows),
    )


def render_calibration_ui() -> None:
    """Streamlit UI for calibration report."""
    import streamlit as st

    st.markdown(
        "<h3 style='color:#00d4ff'>📐 Conviction Score Calibration</h3>"
        "<p style='color:#8892a4;font-size:.82rem'>When we say 80% confidence — do we win 80% of the time?</p>",
        unsafe_allow_html=True,
    )

    report = compute_calibration()
    if report is None:
        st.info("Need at least 20 closed trades to compute calibration. Keep trading!")
        return

    grade_color = {"WELL_CALIBRATED": "#00d4a0", "OVERCONFIDENT": "#ff4b4b", "UNDERCONFIDENT": "#f59e0b"}

    # Brier score header
    brier_color = "#00d4a0" if report.overall_brier_score < 0.15 else "#f59e0b" if report.overall_brier_score < 0.22 else "#ff4b4b"
    col1, col2 = st.columns(2)
    col1.metric("Brier Score", f"{report.overall_brier_score:.3f}", help="Lower = better. 0.25 = random. 0.0 = perfect.")
    col2.metric("Sample Size", f"{report.sample_size} trades")

    st.markdown(
        f"<div style='background:#161b22;border-radius:10px;padding:.7rem 1rem;margin:.5rem 0'>"
        f"<span style='color:{brier_color};font-weight:700'>{report.verdict}</span></div>",
        unsafe_allow_html=True,
    )

    # Per-bucket table
    for b in report.buckets:
        color = grade_color.get(b.grade, "#8892a4")
        error_str = f"+{b.calibration_error:.1%}" if b.calibration_error >= 0 else f"{b.calibration_error:.1%}"
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:.4rem;"
            f"background:#0d1117;border:1px solid {color}33;border-radius:8px;"
            f"padding:.5rem .8rem;margin:.25rem 0;font-size:.78rem'>"
            f"<div style='color:#8892a4'>Score {b.score_lo}–{b.score_hi}</div>"
            f"<div style='color:#c8cfe0'>{b.trade_count} trades</div>"
            f"<div style='color:#8892a4'>Claimed: {b.claimed_win_rate:.0%}</div>"
            f"<div style='color:#c8cfe0'>Actual: {b.actual_win_rate:.0%}</div>"
            f"<div style='color:{color};font-weight:700'>{error_str} · {b.grade.replace('_',' ')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    report = compute_calibration()
    if report:
        print(f"\nBrier Score: {report.overall_brier_score} | {report.verdict}")
        print(f"{'Score':12} {'Trades':8} {'Claimed':10} {'Actual':10} {'Error':10} Grade")
        for b in report.buckets:
            print(f"{b.score_lo}-{b.score_hi}{'':6} {b.trade_count:<8} {b.claimed_win_rate:<10.1%} {b.actual_win_rate:<10.1%} {b.calibration_error:+.1%}{'':4} {b.grade}")
    else:
        print("Insufficient data.")
