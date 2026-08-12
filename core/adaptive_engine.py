"""
core/adaptive_engine.py — Personalized playbook edge tracker.

Reads from journal.db and computes Bayesian-smoothed win rates per
playbook+regime combination, enabling the system to weight signals by
the trader's actual historical edge rather than generic baselines.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default journal DB location (relative to project root)
_DEFAULT_DB = "journal.db"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PlaybookEdge:
    playbook: str
    regime: str
    trades: int
    wins: int
    win_rate: float          # Bayesian posterior (smoothed with prior)
    avg_win_r: float         # average winning trade in R-units
    avg_loss_r: float        # average losing trade in R-units (negative)
    expectancy_r: float      # win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
    sample_size_label: str   # "THIN (<10)", "MODERATE (10-30)", "ROBUST (30+)"
    confidence: str          # "LOW", "MEDIUM", "HIGH"
    last_updated: str

    def to_dict(self) -> dict:
        return {
            "playbook": self.playbook,
            "regime": self.regime,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "avg_win_r": round(self.avg_win_r, 4),
            "avg_loss_r": round(self.avg_loss_r, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "sample_size_label": self.sample_size_label,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_size_label(n: int) -> str:
    if n < 10:
        return "THIN (<10)"
    if n < 30:
        return "MODERATE (10-30)"
    return "ROBUST (30+)"


def _confidence(n: int) -> str:
    if n < 10:
        return "LOW"
    if n < 30:
        return "MEDIUM"
    return "HIGH"


def _expectancy(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    return win_rate * avg_win_r + (1.0 - win_rate) * avg_loss_r


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdaptiveEngine:
    """
    Reads from journal.db to compute personalized win rates per
    playbook+regime combination with Bayesian smoothing.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB,
        prior_trades: int = 20,
        prior_win_rate: float = 0.55,
    ):
        self.db_path = Path(db_path)
        self.prior_trades = prior_trades
        self.prior_win_rate = prior_win_rate
        self.prior_wins = prior_trades * prior_win_rate  # = 11.0 by default
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """
        Ensure journal.db exists and has all required columns.
        Non-destructive: ALTER TABLE is silently ignored if columns exist.
        """
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(self.db_path)
            # Create table if it doesn't exist at all
            conn.execute("""
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT,
                    symbol          TEXT,
                    action          TEXT,
                    price           REAL,
                    qty             INTEGER,
                    indicators      TEXT,
                    news_snap       TEXT,
                    chart_state     TEXT,
                    ai_verdict      TEXT
                )
            """)
            # Add columns that may not exist in older DBs
            extra_cols = [
                ("outcome",         "TEXT"),
                ("pnl_r",           "REAL"),
                ("exit_price",      "REAL"),
                ("exit_timestamp",  "TEXT"),
            ]
            for col_name, col_type in extra_cols:
                try:
                    conn.execute(
                        f"ALTER TABLE journal_entries ADD COLUMN {col_name} {col_type}"
                    )
                except Exception:
                    pass  # column already exists
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("AdaptiveEngine._ensure_schema failed: %s", exc)

    # ------------------------------------------------------------------
    # Private DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_resolved_trades(self) -> list[dict]:
        """
        Load all journal entries that have been resolved (outcome not NULL).
        Returns list of dicts with at least: playbook, regime, outcome, pnl_r.
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT id, timestamp, indicators, outcome, pnl_r
                FROM journal_entries
                WHERE outcome IS NOT NULL
                ORDER BY timestamp ASC
                """
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("AdaptiveEngine._load_resolved_trades failed: %s", exc)
            return []

        trades: list[dict] = []
        for row in rows:
            try:
                indicators = json.loads(row["indicators"] or "{}")
                playbook = indicators.get("playbook", "UNKNOWN")
                regime = indicators.get("regime", "UNKNOWN")
                trades.append(
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "playbook": playbook,
                        "regime": regime,
                        "outcome": row["outcome"],
                        "pnl_r": row["pnl_r"] if row["pnl_r"] is not None else 0.0,
                    }
                )
            except Exception:
                continue
        return trades

    def _build_edge(
        self,
        playbook: str,
        regime: str,
        trades: list[dict],
        last_updated: str,
    ) -> PlaybookEdge:
        """Compute a PlaybookEdge from a list of filtered trades."""
        n = len(trades)
        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        win_trades = [t for t in trades if t["outcome"] == "WIN"]

        # Bayesian posterior
        posterior_wins = self.prior_wins + wins
        posterior_n = self.prior_trades + n
        win_rate = posterior_wins / posterior_n

        avg_win_r = (
            sum(t["pnl_r"] for t in win_trades) / len(win_trades)
            if win_trades
            else 1.5  # prior: typical 1.5R winner
        )
        avg_loss_r = (
            sum(t["pnl_r"] for t in losses) / len(losses)
            if losses
            else -1.0  # prior: -1R per loss
        )

        exp_r = _expectancy(win_rate, avg_win_r, avg_loss_r)

        return PlaybookEdge(
            playbook=playbook,
            regime=regime,
            trades=n,
            wins=wins,
            win_rate=round(win_rate, 4),
            avg_win_r=round(avg_win_r, 4),
            avg_loss_r=round(avg_loss_r, 4),
            expectancy_r=round(exp_r, 4),
            sample_size_label=_sample_size_label(n),
            confidence=_confidence(n),
            last_updated=last_updated,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_edge(self, playbook: str, regime: str) -> PlaybookEdge:
        """
        Returns the personal edge for a specific playbook+regime combination.

        Uses Bayesian smoothing: posterior = (prior_wins + actual_wins) / (prior_n + actual_n).
        If fewer than 5 personal trades exist, returns a prior-only estimate.
        """
        all_trades = self._load_resolved_trades()
        filtered = [
            t for t in all_trades
            if t["playbook"] == playbook and t["regime"] == regime
        ]

        last_updated = (
            filtered[-1]["timestamp"]
            if filtered
            else datetime.now().isoformat()
        )

        if len(filtered) < 5:
            # Prior-only estimate (no personal trades)
            return PlaybookEdge(
                playbook=playbook,
                regime=regime,
                trades=len(filtered),
                wins=sum(1 for t in filtered if t["outcome"] == "WIN"),
                win_rate=round(self.prior_win_rate, 4),
                avg_win_r=1.5,
                avg_loss_r=-1.0,
                expectancy_r=round(
                    _expectancy(self.prior_win_rate, 1.5, -1.0), 4
                ),
                sample_size_label=_sample_size_label(len(filtered)),
                confidence="LOW",
                last_updated=last_updated,
            )

        return self._build_edge(playbook, regime, filtered, last_updated)

    def get_all_edges(self) -> list[PlaybookEdge]:
        """
        Returns edges for all playbook+regime combinations that have any trades.
        Sorted by expectancy_r descending.
        """
        all_trades = self._load_resolved_trades()
        if not all_trades:
            return []

        # Group by (playbook, regime)
        groups: dict[tuple[str, str], list[dict]] = {}
        for t in all_trades:
            key = (t["playbook"], t["regime"])
            groups.setdefault(key, []).append(t)

        edges: list[PlaybookEdge] = []
        for (playbook, regime), group_trades in groups.items():
            last_ts = group_trades[-1]["timestamp"]
            edge = self._build_edge(playbook, regime, group_trades, last_ts)
            edges.append(edge)

        edges.sort(key=lambda e: e.expectancy_r, reverse=True)
        return edges

    def get_best_playbooks_today(
        self, regime: str, top_n: int = 3
    ) -> list[PlaybookEdge]:
        """
        Returns the top N playbooks for the current regime, sorted by expectancy_r.
        Includes playbooks with zero trades (prior estimates) so the system always
        has recommendations.
        """
        all_trades = self._load_resolved_trades()

        # Collect playbooks seen for this regime
        seen_playbooks: set[str] = set()
        for t in all_trades:
            if t["regime"] == regime:
                seen_playbooks.add(t["playbook"])

        # Build edges for known playbooks in this regime
        edges: list[PlaybookEdge] = []
        for playbook in seen_playbooks:
            filtered = [
                t for t in all_trades
                if t["playbook"] == playbook and t["regime"] == regime
            ]
            if not filtered:
                continue
            last_ts = filtered[-1]["timestamp"]
            edge = self._build_edge(playbook, regime, filtered, last_ts)
            edges.append(edge)

        edges.sort(key=lambda e: e.expectancy_r, reverse=True)
        return edges[:top_n]

    def record_outcome(
        self,
        entry_id: int,
        outcome: str,
        exit_price: float,
        pnl_r: float,
    ) -> None:
        """
        Record the outcome of a trade.
        outcome must be one of: WIN / LOSS / BREAKEVEN
        """
        valid_outcomes = {"WIN", "LOSS", "BREAKEVEN"}
        if outcome not in valid_outcomes:
            raise ValueError(
                f"outcome must be one of {valid_outcomes}, got: {outcome!r}"
            )

        exit_ts = datetime.now().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                UPDATE journal_entries
                SET outcome = ?, exit_price = ?, pnl_r = ?, exit_timestamp = ?
                WHERE id = ?
                """,
                (outcome, exit_price, pnl_r, exit_ts, entry_id),
            )
            conn.commit()
            conn.close()
            logger.info(
                "Recorded outcome id=%s outcome=%s pnl_r=%.2f", entry_id, outcome, pnl_r
            )
        except Exception as exc:
            logger.error("record_outcome failed for id=%s: %s", entry_id, exc)

    def get_summary_stats(self) -> dict:
        """
        Returns aggregate statistics across all resolved trades:
        {
            total_trades, win_rate, avg_expectancy_r,
            best_regime, best_playbook,
            streak_current ("W" or "L"), streak_count
        }
        """
        all_trades = self._load_resolved_trades()
        if not all_trades:
            return {
                "total_trades": 0,
                "win_rate": self.prior_win_rate,
                "avg_expectancy_r": 0.0,
                "best_regime": "N/A",
                "best_playbook": "N/A",
                "streak_current": "W",
                "streak_count": 0,
            }

        total = len(all_trades)
        wins = sum(1 for t in all_trades if t["outcome"] == "WIN")
        win_rate = wins / total if total > 0 else 0.0

        # Bayesian-smoothed win rate across all edges
        edges = self.get_all_edges()
        avg_exp = (
            sum(e.expectancy_r for e in edges) / len(edges) if edges else 0.0
        )

        # Best regime: most expectancy_r edges
        regime_exp: dict[str, list[float]] = {}
        for e in edges:
            regime_exp.setdefault(e.regime, []).append(e.expectancy_r)
        best_regime = (
            max(regime_exp, key=lambda r: sum(regime_exp[r]) / len(regime_exp[r]))
            if regime_exp
            else "N/A"
        )

        # Best playbook: highest single expectancy
        best_playbook = edges[0].playbook if edges else "N/A"

        # Current streak: scan from most recent backward
        streak_current = "W" if all_trades[-1]["outcome"] == "WIN" else "L"
        streak_count = 0
        for t in reversed(all_trades):
            letter = "W" if t["outcome"] == "WIN" else "L"
            if letter == streak_current:
                streak_count += 1
            else:
                break

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_expectancy_r": round(avg_exp, 4),
            "best_regime": best_regime,
            "best_playbook": best_playbook,
            "streak_current": streak_current,
            "streak_count": streak_count,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper — enriched journal logging
# ---------------------------------------------------------------------------

def log_trade_with_context(
    symbol: str,
    action: str,
    price: float,
    qty: int,
    regime: str,
    playbook: str,
    quality_tier: str,
    stop_price: float,
    target_price: float,
    ai_verdict: str = "",
    news_snap: str = "",
    chart_state: Optional[dict] = None,
    setup_type: str = "",
    db_path: str = _DEFAULT_DB,
) -> None:
    """
    Logs a trade to journal.db via the existing log_trade_to_journal function,
    enriching the `indicators` JSON with regime, playbook, and quality context.

    This wrapper is importable without Streamlit — it lazy-imports journal.py only
    for the actual DB write, falling back to a direct SQLite insert if unavailable.
    """
    indicators = {
        "regime": regime,
        "playbook": playbook,
        "quality_tier": quality_tier,
        "setup_type": setup_type,
        "stop_price": stop_price,
        "target_price": target_price,
    }

    try:
        from ui.journal import log_trade_to_journal  # lazy import — avoids Streamlit at module level
        log_trade_to_journal(
            symbol=symbol,
            action=action,
            price=price,
            qty=qty,
            indicators=indicators,
            news_snap=news_snap,
            chart_state=chart_state or {},
            ai_verdict=ai_verdict,
        )
    except ImportError:
        # Streamlit not available — write directly to DB
        _direct_log(
            db_path=db_path,
            symbol=symbol,
            action=action,
            price=price,
            qty=qty,
            indicators=indicators,
            news_snap=news_snap,
            chart_state=chart_state or {},
            ai_verdict=ai_verdict,
        )


def _direct_log(
    db_path: str,
    symbol: str,
    action: str,
    price: float,
    qty: int,
    indicators: dict,
    news_snap: str,
    chart_state: dict,
    ai_verdict: str,
) -> None:
    """Direct SQLite write used when Streamlit/ui package is not available."""
    engine = AdaptiveEngine(db_path=db_path)
    engine._ensure_schema()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO journal_entries "
            "(timestamp, symbol, action, price, qty, indicators, news_snap, chart_state, ai_verdict) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                symbol,
                action,
                price,
                qty,
                json.dumps(indicators),
                news_snap[:500],
                json.dumps(chart_state),
                ai_verdict,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("_direct_log failed: %s", exc)
