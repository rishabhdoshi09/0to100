"""
Prediction Ledger — Self-Validating Signal Tracker.

Every analysis creates a set of forward predictions, each with:
  - A specific testable metric (e.g. CFO/NI ratio)
  - The baseline value at prediction time
  - A direction (DETERIORATE | IMPROVE | EVENT)
  - A confidence level from empirical literature
  - A check date (signal_date + horizon_days)

The ledger stores these in SQLite (logs/predictions.db). Running
`python main.py ledger check SYMBOL` re-fetches fundamentals and
auto-resolves predictions whose check dates have passed.

`python main.py ledger stats` prints hit rate, open predictions, and
the Brier score (a proper scoring rule for probability forecasts).

Signal → Prediction mapping is deterministic. Same inputs → same
predictions. No ambiguity about what constitutes a correct outcome.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logger import get_logger

log = get_logger("forensics.ledger")

DB_PATH = Path("logs/predictions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    signal_title    TEXT NOT NULL,
    prediction      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    horizon_days    INTEGER NOT NULL,
    signal_date     TEXT NOT NULL,
    check_date      TEXT NOT NULL,
    baseline_metric TEXT NOT NULL,
    baseline_value  REAL,
    prediction_direction TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    outcome_notes   TEXT,
    checked_date    TEXT,
    checked_value   REAL
);
CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_check_date ON predictions(check_date);
"""

# ── Signal → Prediction mapping ───────────────────────────────────────────────
# Keys match the flag titles produced by the forensic layers.
# Each entry: (prediction_text, confidence_0_1, horizon_days,
#              baseline_metric_key, direction, signal_type_code)

SIGNAL_PREDICTIONS: List[Tuple[str, str, float, int, str, str]] = [
    (
        "High accruals — earnings not backed by cash",
        "Cash conversion (CFO/NI) deteriorates over next 4 quarters as accruals reverse",
        0.72, 365, "cash_conversion", "DETERIORATE", "HIGH_ACCRUALS",
    ),
    (
        "Weak cash conversion",
        "Earnings miss or downward revision likely within 12 months",
        0.68, 365, "cash_conversion", "DETERIORATE", "WEAK_CASH_CONVERSION",
    ),
    (
        "Receivables growing unusually fast",
        "Revenue growth decelerates or receivables written down within 4 quarters",
        0.72, 365, "receivables_to_revenue", "DETERIORATE", "RECEIVABLES_GAP",
    ),
    (
        "Revenue growing while operating cash flow shrinks",
        "Cash conversion worsens further; potential revenue restatement or correction",
        0.76, 365, "cash_conversion", "DETERIORATE", "REV_CFO_DIVERGENCE",
    ),
    (
        "Beneish M-Score in manipulator zone",
        "Accounting enforcement action, restatement, or large earnings revision "
        "within 24 months",
        0.65, 730, "m_score", "EVENT", "BENEISH_MANIPULATOR",
    ),
    (
        "Altman Z-Score in distress zone",
        "Credit event (default, restructuring, dilutive capital raise) within "
        "24 months",
        0.60, 730, "z_score", "EVENT", "ALTMAN_DISTRESS",
    ),
    (
        "Empire Building Detector: scale over value",
        "ROIC continues declining and incremental ROIC remains below WACC "
        "over next 2 fiscal years",
        0.70, 548, "roic", "DETERIORATE", "EMPIRE_BUILDING",
    ),
    (
        "Debt increasing rapidly",
        "Debt/equity ratio increases further or credit rating downgrade "
        "within 18 months",
        0.66, 548, "debt_to_equity", "DETERIORATE", "DEBT_SURGE",
    ),
    (
        "Persistent negative free cash flow",
        "Company raises capital (equity or debt) within 18 months",
        0.72, 548, "fcf", "EVENT", "NEGATIVE_FCF",
    ),
    (
        "Heavy insider selling",
        "Underperformance vs sector index over next 12 months",
        0.62, 365, "price_return", "DETERIORATE", "INSIDER_SELLING",
    ),
    (
        "Margin deterioration",
        "Gross margin continues declining in next two annual reports",
        0.67, 365, "gross_margin", "DETERIORATE", "MARGIN_DETERIORATION",
    ),
    (
        "Goodwill concentration — write-down risk",
        "Goodwill impairment charge or acquisition write-down within 24 months",
        0.55, 730, "goodwill_ratio", "EVENT", "GOODWILL_RISK",
    ),
]

# Metric keys → how to compute current value for outcome check
METRIC_EXTRACTORS: Dict[str, str] = {
    "cash_conversion": "CFO / Net Income (latest year)",
    "receivables_to_revenue": "Receivables / Revenue (latest year)",
    "m_score": "Beneish M-Score",
    "z_score": "Altman Z-Score",
    "roic": "ROIC (NOPAT / Invested Capital)",
    "debt_to_equity": "Total Debt / Equity",
    "fcf": "Free Cash Flow",
    "price_return": "1-year total return vs signal-date price",
    "gross_margin": "Gross Profit / Revenue",
    "goodwill_ratio": "Goodwill / Total Assets",
}


# ── Database helpers ───────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _existing_ids(conn: sqlite3.Connection, symbol: str,
                  signal_types: List[str]) -> set:
    today = date.today().isoformat()
    rows = conn.execute(
        f"SELECT signal_type FROM predictions "
        f"WHERE symbol=? AND signal_date=? AND status='PENDING'",
        (symbol.upper(), today),
    ).fetchall()
    return {r["signal_type"] for r in rows}


# ── Prediction generation ─────────────────────────────────────────────────────

@dataclass
class NewPrediction:
    id: str
    symbol: str
    ticker: str
    signal_type: str
    signal_title: str
    prediction: str
    confidence: float
    horizon_days: int
    signal_date: str
    check_date: str
    baseline_metric: str
    baseline_value: Optional[float]
    prediction_direction: str


def generate_predictions(
    symbol: str,
    ticker: str,
    flags,           # List[RedFlag]
    baseline_metrics: Dict[str, Optional[float]],
) -> List[NewPrediction]:
    today = date.today()
    flag_titles = {f.title for f in flags}
    preds: List[NewPrediction] = []

    for (title, prediction, conf, horizon,
         metric, direction, code) in SIGNAL_PREDICTIONS:
        if title not in flag_titles:
            continue
        check_dt = (today + timedelta(days=horizon)).isoformat()
        preds.append(NewPrediction(
            id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            ticker=ticker,
            signal_type=code,
            signal_title=title,
            prediction=prediction,
            confidence=conf,
            horizon_days=horizon,
            signal_date=today.isoformat(),
            check_date=check_dt,
            baseline_metric=metric,
            baseline_value=baseline_metrics.get(metric),
            prediction_direction=direction,
        ))
    return preds


def save_predictions(preds: List[NewPrediction]) -> int:
    if not preds:
        return 0
    conn = _conn()
    skip = _existing_ids(conn, preds[0].symbol,
                         [p.signal_type for p in preds])
    saved = 0
    for p in preds:
        if p.signal_type in skip:
            continue
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.symbol, p.ticker, p.signal_type, p.signal_title,
             p.prediction, p.confidence, p.horizon_days, p.signal_date,
             p.check_date, p.baseline_metric, p.baseline_value,
             p.prediction_direction, "PENDING", None, None, None),
        )
        saved += 1
    conn.commit()
    conn.close()
    log.info("predictions_saved", symbol=preds[0].symbol, saved=saved)
    return saved


# ── Baseline metric extraction ────────────────────────────────────────────────

def extract_baseline_metrics(layers: Dict) -> Dict[str, Optional[float]]:
    """Pull the key metric values from already-computed layers."""
    fraud = layers.get("fraud")
    cap = layers.get("capital_allocation")
    forensics = layers.get("forensics")

    m: Dict[str, Optional[float]] = {
        "m_score": fraud.extras.get("m_score") if fraud else None,
        "z_score": fraud.extras.get("z_score") if fraud else None,
        "roic": cap.extras.get("roic") if cap else None,
    }

    def _find(layer, name: str) -> Optional[float]:
        if layer is None:
            return None
        for met in layer.metrics:
            if met.name == name and met.value is not None:
                return float(met.value)
        return None

    m["cash_conversion"] = _find(forensics, "Cash Conversion (CFO/NI)")
    m["gross_margin"] = _find(forensics, "Gross Margin Trend")

    for met in (forensics.metrics if forensics else []):
        if met.name == "Debt / Equity" and met.value is not None:
            m["debt_to_equity"] = float(met.value)
        if met.name == "Goodwill & Intangibles / Assets" and met.value is not None:
            m["goodwill_ratio"] = float(met.value)

    fcf_m = _find(forensics, "Free Cash Flow (3y)")
    m["fcf"] = fcf_m

    recv_m = _find(forensics, "Receivables vs Revenue Growth")
    m["receivables_to_revenue"] = recv_m

    return m


# ── Outcome checking ──────────────────────────────────────────────────────────

def check_outcomes(symbol: str, current_metrics: Dict[str, Optional[float]]) -> Dict:
    """
    For all PENDING predictions for symbol whose check_date has passed,
    compare current_metrics against baseline and mark CORRECT / INCORRECT /
    INCONCLUSIVE. Returns summary dict.
    """
    today = date.today().isoformat()
    conn = _conn()
    due = conn.execute(
        "SELECT * FROM predictions WHERE symbol=? AND status='PENDING' AND check_date<=?",
        (symbol.upper(), today),
    ).fetchall()

    results = {"checked": 0, "correct": 0, "incorrect": 0, "inconclusive": 0}

    for row in due:
        metric = row["baseline_metric"]
        baseline = row["baseline_value"]
        direction = row["prediction_direction"]
        current = current_metrics.get(metric)
        status = "INCONCLUSIVE"
        notes = "Current metric unavailable"
        checked_val = None

        if current is not None:
            checked_val = current
            if direction == "EVENT":
                # Events cannot be auto-checked from metric alone; mark inconclusive
                status = "INCONCLUSIVE"
                notes = (f"Event prediction — manual verification required. "
                         f"Baseline {baseline}, current {current:.4f}")
            elif baseline is not None:
                deteriorated = current < baseline if direction == "DETERIORATE" else current > baseline
                threshold = abs(baseline) * 0.05 if baseline else 0.01
                if abs(current - baseline) < threshold:
                    status = "INCONCLUSIVE"
                    notes = f"Change below 5% threshold: {baseline:.4f} → {current:.4f}"
                elif deteriorated:
                    status = "CORRECT"
                    notes = f"Metric moved as predicted: {baseline:.4f} → {current:.4f}"
                else:
                    status = "INCORRECT"
                    notes = f"Metric did NOT move as predicted: {baseline:.4f} → {current:.4f}"

        conn.execute(
            "UPDATE predictions SET status=?, outcome_notes=?, checked_date=?, "
            "checked_value=? WHERE id=?",
            (status, notes, today, checked_val, row["id"]),
        )
        results["checked"] += 1
        if status == "CORRECT":
            results["correct"] += 1
        elif status == "INCORRECT":
            results["incorrect"] += 1
        else:
            results["inconclusive"] += 1

    conn.commit()
    conn.close()
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats(symbol: Optional[str] = None) -> Dict:
    conn = _conn()
    where = "WHERE symbol=?" if symbol else ""
    params = (symbol.upper(),) if symbol else ()
    rows = conn.execute(
        f"SELECT status, confidence, COUNT(*) as n FROM predictions {where} "
        f"GROUP BY status", params
    ).fetchall()
    conn.close()

    counts = {r["status"]: r["n"] for r in rows}
    total = sum(counts.values())
    resolved = counts.get("CORRECT", 0) + counts.get("INCORRECT", 0)
    hit_rate = counts.get("CORRECT", 0) / resolved if resolved else None

    # Brier score (mean squared error of probability forecasts, lower = better)
    conn2 = _conn()
    resolved_rows = conn2.execute(
        f"SELECT confidence, status FROM predictions {where} "
        f"WHERE status IN ('CORRECT','INCORRECT')", params
    ).fetchall()
    conn2.close()
    brier = None
    if resolved_rows:
        brier = sum(
            (r["confidence"] - (1.0 if r["status"] == "CORRECT" else 0.0)) ** 2
            for r in resolved_rows
        ) / len(resolved_rows)

    return {
        "total": total,
        "pending": counts.get("PENDING", 0),
        "correct": counts.get("CORRECT", 0),
        "incorrect": counts.get("INCORRECT", 0),
        "inconclusive": counts.get("INCONCLUSIVE", 0),
        "hit_rate": hit_rate,
        "brier_score": brier,
    }


def get_open_predictions(symbol: str) -> List[sqlite3.Row]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE symbol=? AND status='PENDING' "
        "ORDER BY check_date",
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    return rows
