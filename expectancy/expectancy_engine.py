"""
Expectancy Engine — SQLite-backed per-setup, per-playbook performance tracker.

Computes evidence-based expected value for NSE India institutional trading setups.
Tracks win rate, R-multiples, MAE/MFE, and regime-conditional expectancy.
"""
from __future__ import annotations

import sqlite3
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "expectancy.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS setup_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    playbook_id     TEXT NOT NULL,
    quality_tier    TEXT NOT NULL,
    entry_date      TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    stop_price      REAL NOT NULL,
    target_price    REAL NOT NULL,
    exit_date       TEXT,
    exit_price      REAL,
    outcome         TEXT NOT NULL DEFAULT 'PENDING',
    return_pct      REAL,
    hold_days       INTEGER,
    mae_pct         REAL NOT NULL DEFAULT 0.0,
    mfe_pct         REAL NOT NULL DEFAULT 0.0,
    market_regime   TEXT NOT NULL,
    vol_regime      TEXT NOT NULL,
    breadth         TEXT NOT NULL,
    created_at      TEXT NOT NULL
)
"""


@dataclass
class PlaybookStats:
    playbook_id: str
    total_setups: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float           # win_rate × avg_win − (1 − wr) × |avg_loss|
    avg_hold_days: float
    avg_mae: float
    avg_mfe: float
    risk_reward: float
    best_regime: str            # regime with highest expectancy
    sample_size: int
    regime_breakdown: dict = field(default_factory=dict)   # {regime: expectancy}


@dataclass
class ExpectedValueReport:
    playbook_id: str
    quality_tier: str
    market_regime: str
    expected_value_r: float         # in R units
    historical_win_rate: float
    regime_win_rate: float          # win rate specifically in this regime
    regime_alignment: str           # HIGH | MEDIUM | LOW
    failure_risk: str               # LOW | MODERATE | HIGH
    sample_size: int
    confidence: str                 # HIGH (>30) | MEDIUM (10-30) | LOW (<10)
    recommendation: str             # "FAVORABLE" | "SELECTIVE" | "AVOID"

    def format_institutional(self) -> str:
        tier_display = self.quality_tier.replace("_", " ")
        lines = [
            f"SETUP: {self.playbook_id}",
            f"QUALITY: {tier_display}",
            f"EXPECTED VALUE: {self.expected_value_r:+.1f}R",
            f"HISTORICAL WIN RATE: {self.historical_win_rate:.1%}",
            f"REGIME ({self.market_regime}) WIN RATE: {self.regime_win_rate:.1%}",
            f"REGIME ALIGNMENT: {self.regime_alignment}",
            f"FAILURE RISK: {self.failure_risk}",
            f"SAMPLE SIZE: {self.sample_size}",
            f"CONFIDENCE: {self.confidence}",
            f"RECOMMENDATION: {self.recommendation}",
        ]
        return "\n".join(lines)


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _compute_expectancy(rows: list) -> tuple[float, float, float, float]:
    """Return (win_rate, avg_win_pct, avg_loss_pct, expectancy) from closed rows."""
    closed = [r for r in rows if r["outcome"] in ("WIN", "LOSS", "BREAKEVEN")]
    if not closed:
        return 0.0, 0.0, 0.0, 0.0
    wins = [r["return_pct"] for r in closed if r["outcome"] == "WIN" and r["return_pct"] is not None]
    losses = [abs(r["return_pct"]) for r in closed if r["outcome"] == "LOSS" and r["return_pct"] is not None]
    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return win_rate, avg_win, avg_loss, expectancy


class ExpectancyEngine:
    """Tracks per-setup, per-playbook performance and computes evidence-based EV."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._conn = _connect(db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_setup(
        self,
        symbol: str,
        playbook_id: str,
        quality_tier: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        market_regime: str,
        vol_regime: str,
        breadth: str,
    ) -> int:
        """Insert a new PENDING setup and return its row id."""
        try:
            now = datetime.utcnow().isoformat(timespec="seconds")
            cur = self._conn.execute(
                """
                INSERT INTO setup_outcomes
                    (symbol, playbook_id, quality_tier, entry_date, entry_price,
                     stop_price, target_price, outcome, mae_pct, mfe_pct,
                     market_regime, vol_regime, breadth, created_at)
                VALUES (?,?,?,?,?,?,?,'PENDING',0.0,0.0,?,?,?,?)
                """,
                (symbol, playbook_id, quality_tier, date.today().isoformat(),
                 entry_price, stop_price, target_price,
                 market_regime, vol_regime, breadth, now),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as exc:
            return -1

    def close_setup(
        self,
        setup_id: int,
        exit_price: float,
        exit_date: Optional[str] = None,
    ) -> None:
        """Compute outcome, return_pct, hold_days and mark setup closed."""
        try:
            row = self._conn.execute(
                "SELECT * FROM setup_outcomes WHERE id = ?", (setup_id,)
            ).fetchone()
            if row is None:
                return

            entry_price: float = row["entry_price"]
            stop_price: float = row["stop_price"]
            target_price: float = row["target_price"]

            return_pct = (exit_price - entry_price) / entry_price * 100.0

            risk = entry_price - stop_price
            if abs(return_pct) < 0.05:
                outcome = "BREAKEVEN"
            elif exit_price >= target_price or return_pct > 0:
                outcome = "WIN"
            else:
                outcome = "LOSS"

            today = exit_date or date.today().isoformat()
            try:
                entry_dt = date.fromisoformat(row["entry_date"])
                exit_dt = date.fromisoformat(today)
                hold_days = (exit_dt - entry_dt).days
            except Exception:
                hold_days = 0

            self._conn.execute(
                """
                UPDATE setup_outcomes
                SET exit_price = ?, exit_date = ?, outcome = ?,
                    return_pct = ?, hold_days = ?
                WHERE id = ?
                """,
                (exit_price, today, outcome, return_pct, hold_days, setup_id),
            )
            self._conn.commit()
        except Exception:
            pass

    def update_mae_mfe(self, setup_id: int, current_price: float) -> None:
        """Update max adverse / favorable excursion for a live position."""
        try:
            row = self._conn.execute(
                "SELECT entry_price, mae_pct, mfe_pct FROM setup_outcomes WHERE id = ?",
                (setup_id,),
            ).fetchone()
            if row is None:
                return
            entry_price: float = row["entry_price"]
            move_pct = (current_price - entry_price) / entry_price * 100.0
            new_mae = min(row["mae_pct"], move_pct)   # most negative
            new_mfe = max(row["mfe_pct"], move_pct)   # most positive
            self._conn.execute(
                "UPDATE setup_outcomes SET mae_pct = ?, mfe_pct = ? WHERE id = ?",
                (new_mae, new_mfe, setup_id),
            )
            self._conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def _rows_for_playbook(self, playbook_id: str, market_regime: Optional[str] = None) -> list:
        try:
            if market_regime:
                return self._conn.execute(
                    "SELECT * FROM setup_outcomes WHERE playbook_id = ? AND market_regime = ?",
                    (playbook_id, market_regime),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM setup_outcomes WHERE playbook_id = ?", (playbook_id,)
            ).fetchall()
        except Exception:
            return []

    def _build_playbook_stats(self, playbook_id: str, rows: list) -> PlaybookStats:
        closed = [r for r in rows if r["outcome"] in ("WIN", "LOSS", "BREAKEVEN")]
        win_rate, avg_win, avg_loss, expectancy = _compute_expectancy(rows)

        hold_vals = [r["hold_days"] for r in closed if r["hold_days"] is not None]
        mae_vals = [r["mae_pct"] for r in rows if r["mae_pct"] is not None]
        mfe_vals = [r["mfe_pct"] for r in rows if r["mfe_pct"] is not None]

        avg_hold = sum(hold_vals) / len(hold_vals) if hold_vals else 0.0
        avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
        avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Per-regime breakdown
        regimes: dict[str, list] = {}
        for r in rows:
            reg = r["market_regime"] or "UNKNOWN"
            regimes.setdefault(reg, []).append(r)

        regime_breakdown: dict[str, float] = {}
        best_regime = ""
        best_ev = float("-inf")
        for reg, reg_rows in regimes.items():
            _, _, _, ev = _compute_expectancy(reg_rows)
            regime_breakdown[reg] = round(ev, 3)
            if ev > best_ev:
                best_ev = ev
                best_regime = reg

        return PlaybookStats(
            playbook_id=playbook_id,
            total_setups=len(rows),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 3),
            avg_loss_pct=round(avg_loss, 3),
            expectancy=round(expectancy, 3),
            avg_hold_days=round(avg_hold, 1),
            avg_mae=round(avg_mae, 3),
            avg_mfe=round(avg_mfe, 3),
            risk_reward=round(risk_reward, 2),
            best_regime=best_regime,
            sample_size=len(closed),
            regime_breakdown=regime_breakdown,
        )

    def get_playbook_stats(self, playbook_id: str, market_regime: Optional[str] = None) -> PlaybookStats:
        rows = self._rows_for_playbook(playbook_id, market_regime)
        return self._build_playbook_stats(playbook_id, rows)

    def get_quality_tier_stats(self, quality_tier: str) -> PlaybookStats:
        try:
            rows = self._conn.execute(
                "SELECT * FROM setup_outcomes WHERE quality_tier = ?", (quality_tier,)
            ).fetchall()
        except Exception:
            rows = []
        return self._build_playbook_stats(quality_tier, rows)

    def get_expected_value_report(
        self,
        playbook_id: str,
        quality_tier: str,
        market_regime: str,
    ) -> ExpectedValueReport:
        # All rows for this playbook + tier
        try:
            all_rows = self._conn.execute(
                "SELECT * FROM setup_outcomes WHERE playbook_id = ? AND quality_tier = ?",
                (playbook_id, quality_tier),
            ).fetchall()
        except Exception:
            all_rows = []

        # Regime-specific rows
        try:
            regime_rows = self._conn.execute(
                "SELECT * FROM setup_outcomes WHERE playbook_id = ? AND quality_tier = ? AND market_regime = ?",
                (playbook_id, quality_tier, market_regime),
            ).fetchall()
        except Exception:
            regime_rows = []

        hist_wr, hist_avg_win, hist_avg_loss, hist_ev = _compute_expectancy(all_rows)
        reg_wr, reg_avg_win, reg_avg_loss, reg_ev = _compute_expectancy(regime_rows)

        # Expected value in R units: EV / avg_risk_per_trade
        # Use EV as a fraction of the average stop distance
        try:
            risk_vals = [
                abs(r["entry_price"] - r["stop_price"]) / r["entry_price"] * 100.0
                for r in all_rows
                if r["entry_price"] and r["stop_price"]
            ]
            avg_risk_pct = sum(risk_vals) / len(risk_vals) if risk_vals else 1.0
        except Exception:
            avg_risk_pct = 1.0

        ev_r = hist_ev / avg_risk_pct if avg_risk_pct > 0 else 0.0

        # Regime alignment
        if regime_rows and reg_wr >= hist_wr * 1.1:
            regime_alignment = "HIGH"
        elif regime_rows and reg_wr >= hist_wr * 0.9:
            regime_alignment = "MEDIUM"
        else:
            regime_alignment = "LOW"

        # Failure risk
        closed_all = [r for r in all_rows if r["outcome"] in ("WIN", "LOSS", "BREAKEVEN")]
        loss_rate = 1 - hist_wr
        if loss_rate < 0.35:
            failure_risk = "LOW"
        elif loss_rate < 0.55:
            failure_risk = "MODERATE"
        else:
            failure_risk = "HIGH"

        sample_size = len(closed_all)
        if sample_size >= 30:
            confidence = "HIGH"
        elif sample_size >= 10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Recommendation
        if ev_r >= 1.5 and regime_alignment in ("HIGH", "MEDIUM") and failure_risk != "HIGH":
            recommendation = "FAVORABLE"
        elif ev_r >= 0.5 and failure_risk != "HIGH":
            recommendation = "SELECTIVE"
        else:
            recommendation = "AVOID"

        return ExpectedValueReport(
            playbook_id=playbook_id,
            quality_tier=quality_tier,
            market_regime=market_regime,
            expected_value_r=round(ev_r, 2),
            historical_win_rate=round(hist_wr, 4),
            regime_win_rate=round(reg_wr, 4),
            regime_alignment=regime_alignment,
            failure_risk=failure_risk,
            sample_size=sample_size,
            confidence=confidence,
            recommendation=recommendation,
        )

    def get_all_stats_summary(self) -> dict:
        """Overall system performance across all playbooks."""
        try:
            rows = self._conn.execute("SELECT * FROM setup_outcomes").fetchall()
        except Exception:
            return {}

        closed = [r for r in rows if r["outcome"] in ("WIN", "LOSS", "BREAKEVEN")]
        win_rate, avg_win, avg_loss, expectancy = _compute_expectancy(rows)

        playbooks = list({r["playbook_id"] for r in rows})
        tiers = list({r["quality_tier"] for r in rows})

        return {
            "total_setups": len(rows),
            "closed_setups": len(closed),
            "pending_setups": len(rows) - len(closed),
            "overall_win_rate": round(win_rate, 4),
            "overall_expectancy": round(expectancy, 3),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "playbooks_tracked": playbooks,
            "quality_tiers_tracked": tiers,
        }

    def get_recent_setups(self, limit: int = 20) -> list[dict]:
        """Recent setup outcomes for UI display, newest first."""
        try:
            rows = self._conn.execute(
                """
                SELECT id, symbol, playbook_id, quality_tier, entry_date, entry_price,
                       exit_date, exit_price, outcome, return_pct, hold_days,
                       mae_pct, mfe_pct, market_regime
                FROM setup_outcomes
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
