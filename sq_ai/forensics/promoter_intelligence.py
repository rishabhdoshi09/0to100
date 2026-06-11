"""
Promoter Intelligence Engine — Indian Equity Advantage Layer.

Data sources (in priority order):
  1. NSE shareholding pattern API (session-based, quarterly data)
     → trend: promoter stake QoQ, pledging %, institutional changes
  2. yfinance major_holders / institutional_holders
     → single snapshot only
  3. Ledger SQLite (logs/predictions.db)
     → stores quarterly snapshots so trend builds over time

Output:
  Promoter Alignment Score (0–100)
  Ownership Trend Score (0–100)
  Shareholding pattern metrics with sources tagged
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logger import get_logger
from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
    SOURCE_RELIABILITY,
)
from forensics.data_reliability import CoverageTracker

log = get_logger("forensics.promoter")

NSE_API_BASE = "https://www.nseindia.com/api"
SNAPSHOT_DB = Path("logs/predictions.db")


# ── NSE Shareholding Pattern Fetcher ─────────────────────────────────────────

def _nse_shareholding(symbol: str) -> Optional[Dict]:
    """
    Fetch latest shareholding pattern from NSE API.
    Requires a fresh session cookie from nseindia.com.
    Falls back gracefully to None if blocked or offline.
    """
    try:
        import requests
        import time

        session = requests.Session()
        session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.nseindia.com/",
        })
        session.get("https://www.nseindia.com", timeout=8)
        time.sleep(0.6)
        r = session.get(
            f"{NSE_API_BASE}/shareholding-patterns",
            params={"index": "equities", "symbol": symbol},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            log.info("nse_shareholding_fetched", symbol=symbol)
            return data
    except Exception as exc:
        log.debug("nse_shareholding_unavailable", symbol=symbol, reason=str(exc))
    return None


def _parse_nse_pattern(data: Dict) -> Optional[Dict]:
    """Extract the latest quarter from NSE shareholding JSON."""
    try:
        records = data.get("data", [])
        if not records:
            return None
        latest = records[0]
        promoter_pct = float(latest.get("promoterAndPromoterGroupShareholding", 0) or 0)
        pledged_pct = float(latest.get("promoterGroupEncumberedSharesPercentage", 0) or 0)
        fii_pct = float(latest.get("foreignPortfolioInvestors", 0) or 0)
        dii_pct = float(latest.get("mutualFunds", 0) or 0)
        quarter = latest.get("date", "")
        return {
            "promoter_pct": promoter_pct,
            "pledged_pct": pledged_pct,
            "fii_pct": fii_pct,
            "dii_pct": dii_pct,
            "quarter": quarter,
            "source": "NSE Shareholding Pattern",
        }
    except Exception:
        return None


# ── Quarterly snapshot store ──────────────────────────────────────────────────

_SNAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS shareholding_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    quarter     TEXT NOT NULL,
    fetch_date  TEXT NOT NULL,
    promoter_pct REAL,
    pledged_pct  REAL,
    fii_pct      REAL,
    dii_pct      REAL,
    source      TEXT,
    UNIQUE(symbol, quarter)
);
"""


def _snap_conn() -> sqlite3.Connection:
    SNAPSHOT_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(SNAPSHOT_DB)
    c.row_factory = sqlite3.Row
    c.executescript(_SNAP_SCHEMA)
    return c


def _save_snapshot(symbol: str, snap: Dict) -> None:
    conn = _snap_conn()
    conn.execute(
        "INSERT OR REPLACE INTO shareholding_snapshots "
        "(symbol, quarter, fetch_date, promoter_pct, pledged_pct, "
        "fii_pct, dii_pct, source) VALUES (?,?,?,?,?,?,?,?)",
        (symbol.upper(), snap.get("quarter", date.today().isoformat()),
         date.today().isoformat(), snap.get("promoter_pct"),
         snap.get("pledged_pct"), snap.get("fii_pct"),
         snap.get("dii_pct"), snap.get("source")),
    )
    conn.commit()
    conn.close()


def _load_snapshots(symbol: str, n: int = 4) -> List[sqlite3.Row]:
    conn = _snap_conn()
    rows = conn.execute(
        "SELECT * FROM shareholding_snapshots WHERE symbol=? "
        "ORDER BY quarter DESC LIMIT ?",
        (symbol.upper(), n),
    ).fetchall()
    conn.close()
    return rows


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(d: FundamentalData,
            tracker: Optional[CoverageTracker] = None) -> LayerResult:
    res = LayerResult(layer="Promoter & Ownership Intelligence", score=None)
    symbol_clean = d.symbol.replace(".NS", "").upper()
    scores: List[float] = []
    source = "Market Data (yfinance)"

    # Try NSE shareholding API for fresh data
    nse_data = _nse_shareholding(symbol_clean)
    snap = _parse_nse_pattern(nse_data) if nse_data else None
    if snap:
        _save_snapshot(symbol_clean, snap)
        source = "NSE Shareholding Pattern"

    # Historical snapshots for trend
    snaps = _load_snapshots(symbol_clean, 4)
    has_trend = len(snaps) >= 2

    # Fallback to yfinance
    info = d.info
    mh = d.major_holders

    # ── Promoter / Insider ownership ──────────────────────────────────────────
    promoter_pct: Optional[float] = None
    if snap:
        promoter_pct = snap.get("promoter_pct")
    if promoter_pct is None and mh is not None:
        try:
            import pandas as pd
            # yfinance major_holders has Value / % Shares columns
            row = mh[mh.index.astype(str).str.lower().str.contains("insider")]
            if not row.empty:
                promoter_pct = float(row.iloc[0, 0]) * 100
        except Exception:
            pass
    if promoter_pct is None:
        v = info.get("heldPercentInsiders")
        if v is not None:
            promoter_pct = float(v) * 100

    if promoter_pct is not None:
        rel = SOURCE_RELIABILITY.get(source, 75.0)
        if tracker:
            tracker.declare("Ownership", source, "promoter_pct")
        s_p = max(0.0, min(100.0, 70.0 if 20 <= promoter_pct <= 65 else
                  (50.0 if promoter_pct < 20 else 55.0)))
        scores.append(s_p)
        res.metrics.append(Metric(
            name="Promoter / Insider Stake", value=promoter_pct,
            display=f"{promoter_pct:.1f}%",
            what="Percentage of shares held by promoters / insiders.",
            why="High promoter ownership aligns incentives; above 75% risks minority "
                "suppression and low float manipulation. Below 20% signals "
                "promoter disengagement.",
            good="20–65% — meaningful skin in the game without entrenchment.",
            implication=("Good alignment zone." if 20 <= promoter_pct <= 65
                         else "Above 65% — check for pledging and related-party risk."
                         if promoter_pct > 65
                         else "Below 20% — promoter commitment is weak."),
            source=source, reliability=rel,
        ))

    # ── Promoter pledging ──────────────────────────────────────────────────────
    pledged_pct: Optional[float] = snap.get("pledged_pct") if snap else None
    if pledged_pct is not None and promoter_pct:
        if tracker:
            tracker.declare("Ownership", source, "pledged_pct")
        pledge_ratio = pledged_pct / max(promoter_pct, 1)
        s_pl = max(0.0, min(100.0, 100 - pledge_ratio * 150))
        scores.append(s_pl)
        res.metrics.append(Metric(
            name="Promoter Pledging", value=pledged_pct,
            display=f"{pledged_pct:.1f}% pledged  ({pledge_ratio:.0%} of promoter stake)",
            what="Fraction of promoter shares pledged as collateral for loans.",
            why="Pledged shares create forced-sale risk: if the stock falls, lenders "
                "sell promoter shares, accelerating the decline.",
            good="Below 10% of promoter stake.",
            implication=("Pledging is minimal." if pledge_ratio < 0.10
                         else "Significant pledging — forced-sale risk present."),
            source=source, reliability=SOURCE_RELIABILITY.get(source, 75),
        ))
        if pledge_ratio > 0.25:
            res.flags.append(RedFlag(
                title="Promoter pledging elevated",
                severity=Severity.HIGH,
                evidence=f"{pledged_pct:.1f}% of shares pledged "
                         f"({pledge_ratio:.0%} of promoter holding).",
                why_it_matters="A stock price decline triggers margin calls, forcing "
                               "promoter share sales that push the price down further — "
                               "a reflexive doom loop.",
                precedent="Zee Entertainment and ADAG group stocks collapsed partly "
                          "due to pledging-triggered cascades.",
                evidence_rows=[
                    ("Promoter holding", f"{promoter_pct:.1f}%"),
                    ("Shares pledged", f"{pledged_pct:.1f}%"),
                    ("Pledge as % of promoter stake", f"{pledge_ratio:.0%}"),
                ],
                sources=[source],
            ))
    elif snap is None:
        if tracker:
            tracker.mark_missing("Ownership", "Promoter pledging (requires NSE API)")
        res.notes.append("Promoter pledging: not available from yfinance — "
                         "requires NSE shareholding pattern API.")

    # ── Institutional ownership trend ─────────────────────────────────────────
    fii_pct: Optional[float] = snap.get("fii_pct") if snap else None
    dii_pct: Optional[float] = snap.get("dii_pct") if snap else None
    if fii_pct is None:
        v = info.get("heldPercentInstitutions")
        if v is not None:
            fii_pct = float(v) * 100
            source_inst = "Market Data (yfinance)"
        else:
            source_inst = "Not Available"
    else:
        source_inst = source

    if fii_pct is not None:
        if tracker:
            tracker.declare("Ownership", source_inst, "institutional_pct")
        s_inst = max(0.0, min(100.0, fii_pct * 1.5))
        scores.append(s_inst)
        res.metrics.append(Metric(
            name="Institutional Ownership (FII/DII)", value=fii_pct,
            display=f"{fii_pct:.1f}%" + (f" FII  +  {dii_pct:.1f}% DII" if dii_pct else ""),
            what="Percentage held by foreign and domestic institutional investors.",
            why="Institutional ownership is a diligence proxy; sudden exits are a "
                "lead indicator before bad news becomes public.",
            good="Above 20% for a liquid large-cap.",
            implication=("Strong institutional conviction." if fii_pct > 20
                         else "Institutions are under-weight this name — ask why."),
            source=source_inst, reliability=SOURCE_RELIABILITY.get(source_inst, 75),
        ))
    else:
        if tracker:
            tracker.mark_missing("Ownership", "Institutional ownership")

    # ── QoQ ownership trend from snapshots ────────────────────────────────────
    if has_trend:
        latest_snap = snaps[0]
        prev_snap = snaps[1]
        delta_promo = (latest_snap["promoter_pct"] or 0) - (prev_snap["promoter_pct"] or 0)
        delta_inst = ((latest_snap["fii_pct"] or 0) - (prev_snap["fii_pct"] or 0))
        if tracker:
            tracker.declare("Ownership", "NSE Shareholding Pattern", "ownership_trend")
        trend_label = ("Promoters accumulating" if delta_promo > 0.5
                       else "Promoters divesting" if delta_promo < -0.5
                       else "Promoter stake stable")
        inst_label = ("Institutions buying" if delta_inst > 1.0
                      else "Institutions selling" if delta_inst < -1.0
                      else "Institutional holding stable")
        res.metrics.append(Metric(
            name="Ownership Trend (QoQ)", value=delta_promo,
            display=f"Promoter {delta_promo:+.1f}pp  |  Inst. {delta_inst:+.1f}pp",
            what="Quarter-over-quarter change in promoter and institutional stake.",
            why="Promoter and institutional accumulation/distribution often leads "
                "price moves by 1–2 quarters.",
            good="Promoters stable or buying; institutions buying.",
            implication=f"{trend_label}; {inst_label}.",
            source="NSE Shareholding Pattern",
            reliability=95.0,
        ))
        if delta_promo < -2.0:
            res.flags.append(RedFlag(
                title="Promoter stake declining",
                severity=Severity.MEDIUM,
                evidence=f"Promoter holding fell {delta_promo:.1f}pp QoQ "
                         f"({prev_snap['promoter_pct']:.1f}% → "
                         f"{latest_snap['promoter_pct']:.1f}%).",
                why_it_matters="Promoter reduction of >2pp in a quarter is "
                               "unusual and often precedes negative disclosures.",
                precedent="Promoter exits preceded the collapses at Yes Bank, "
                          "DHFL, and multiple NBFC failures.",
                evidence_rows=[
                    ("Quarter (prev)", str(prev_snap["quarter"])),
                    ("Promoter stake (prev)", f"{prev_snap['promoter_pct']:.1f}%"),
                    ("Quarter (latest)", str(latest_snap["quarter"])),
                    ("Promoter stake (latest)", f"{latest_snap['promoter_pct']:.1f}%"),
                    ("Change", f"{delta_promo:+.1f}pp"),
                ],
                sources=["NSE Shareholding Pattern"],
            ))

    elif not snap:
        res.notes.append("Ownership trend: only one snapshot available. "
                         "Re-run quarterly to build trend data in the ledger.")

    res.score = sum(scores) / len(scores) if scores else None
    return res
