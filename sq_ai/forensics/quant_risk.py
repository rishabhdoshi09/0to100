"""
Layer 2 — Quantitative Risk Engine.

Price-based stability and risk metrics plus statistical anomaly detection
on the fundamental series (outlier quarters, abnormal growth spikes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity, stmt_row,
)
from forensics.statement_forensics import REVENUE

TRADING_DAYS = 252
RISK_FREE = 0.065  # approximate Indian 10y yield


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Quantitative Risk Engine", score=None)
    px = d.prices
    if px.empty or "Close" not in px or len(px) < 60:
        res.notes.append("Insufficient price history — layer skipped.")
        return res

    ret = px["Close"].pct_change().dropna()
    ann_vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float((1 + ret.mean()) ** TRADING_DAYS - 1)

    # Max drawdown
    cum = (1 + ret).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = float(dd.min())

    # Downside deviation & Sortino
    downside = ret[ret < 0]
    dd_dev = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 5 else None
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else None
    sortino = (ann_ret - RISK_FREE) / dd_dev if dd_dev else None

    # Beta vs benchmark
    beta = None
    if not d.benchmark.empty and "Close" in d.benchmark:
        bret = d.benchmark["Close"].pct_change().dropna()
        joined = pd.concat([ret, bret], axis=1, join="inner").dropna()
        if len(joined) > 60 and joined.iloc[:, 1].var() > 0:
            beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var())

    scores = []

    s_vol = max(0.0, min(100.0, 100 - (ann_vol - 0.15) * 200))
    scores.append(s_vol)
    res.metrics.append(Metric(
        name="Annualized Volatility", value=ann_vol, display=f"{ann_vol:.1%}",
        what="Standard deviation of daily returns, annualized.",
        why="Volatility is the price of admission: it determines drawdowns, position "
            "sizing, and how violently bad news is repriced.",
        good="Below ~25% for large caps; above 45% is speculative territory.",
        implication="Stable price behavior." if ann_vol < 0.30
        else "The market itself treats this stock as high-uncertainty.",
        score=s_vol,
    ))

    s_dd = max(0.0, min(100.0, 100 + max_dd * 130))
    scores.append(s_dd)
    res.metrics.append(Metric(
        name="Max Drawdown (5y)", value=max_dd, display=f"{max_dd:.1%}",
        what="Largest peak-to-trough decline over the price history.",
        why="Shows how badly holders have been hurt before — deep drawdowns reveal "
            "fragile ownership and leverage in the story.",
        good="Shallower than −30%. Worse than −60% marks a blowup history.",
        implication="Drawdowns have been contained." if max_dd > -0.35
        else "Holders have endured severe losses in this name before.",
        score=s_dd,
    ))

    if sharpe is not None:
        res.metrics.append(Metric(
            name="Sharpe Ratio", value=sharpe, display=f"{sharpe:.2f}",
            what="Excess return over the risk-free rate per unit of volatility.",
            why="The standard institutional measure of risk-adjusted return.",
            good="Above 1.0 is good; below 0 means cash beat the stock.",
            implication="Risk has been compensated." if sharpe > 0.5
            else "Returns have not justified the volatility taken.",
        ))
    if sortino is not None:
        res.metrics.append(Metric(
            name="Sortino Ratio", value=sortino, display=f"{sortino:.2f}",
            what="Like Sharpe, but penalizes only downside volatility.",
            why="Separates 'good' upside variance from harmful downside swings.",
            good="Above 1.5.",
            implication="Downside risk well-paid." if sortino > 1
            else "Downside swings dominate the return profile.",
        ))
    if beta is not None:
        res.metrics.append(Metric(
            name="Beta", value=beta, display=f"{beta:.2f}",
            what="Sensitivity of the stock to the benchmark index.",
            why="High-beta names amplify market stress; in a drawdown they fall hardest.",
            good="0.7–1.2 for core holdings.",
            implication="Market-like sensitivity." if 0.6 <= beta <= 1.3
            else ("Defensive, low market sensitivity." if beta < 0.6
                  else "Amplifies every market move — fragile in stress regimes."),
        ))

    # ── Fundamental anomaly detection: revenue series ─────────────────────────
    rev_q = stmt_row(d.quarterly_income, REVENUE)
    if rev_q is not None and len(rev_q.dropna()) >= 5:
        series = rev_q.dropna().astype(float)[::-1]  # oldest → newest
        growth = series.pct_change().dropna()
        if len(growth) >= 4 and growth.std() > 0:
            z = (growth - growth.mean()) / growth.std()
            outliers = z[abs(z) > 2]
            rev_vol = float(growth.std())
            s_stab = max(0.0, min(100.0, 100 - rev_vol * 300))
            scores.append(s_stab)
            res.metrics.append(Metric(
                name="Revenue Volatility (QoQ)", value=rev_vol, display=f"{rev_vol:.1%}",
                what="Standard deviation of quarter-over-quarter revenue growth.",
                why="Lumpy revenue is harder to forecast and easier to manipulate — "
                    "smooth-but-fake beats lumpy-but-real in fraud archaeology.",
                good="Below ~10% QoQ std-dev for a mature business.",
                implication="Revenue stream is stable." if rev_vol < 0.10
                else "Revenue is erratic quarter to quarter.",
                score=s_stab,
            ))
            for ts, zval in outliers.items():
                g = growth.loc[ts]
                res.flags.append(RedFlag(
                    title="Statistical anomaly: outlier revenue quarter",
                    severity=Severity.MEDIUM if abs(zval) < 3 else Severity.HIGH,
                    period=str(getattr(ts, "date", lambda: ts)()),
                    evidence=f"QoQ revenue growth {g:+.1%} (z-score {zval:+.1f}).",
                    why_it_matters="Abnormal spikes can mark one-off pull-forwards, "
                                   "channel stuffing, or a genuine regime change — "
                                   "either way the trend is not what it appears.",
                    precedent="Quarter-end revenue spikes were the tell in the "
                              "Autonomy/HP and Ricoh India cases.",
                ))

    res.score = sum(scores) / len(scores) if scores else None
    res.extras["ann_return"] = ann_ret
    return res
