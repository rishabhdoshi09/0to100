"""
Layer 8 — Market Microstructure Intelligence.

Volume/price-structure signals computable from daily OHLCV: volume anomalies,
VWAP deviation, Accumulation/Distribution trend, and an OBV-based smart money
divergence. Combined into the Institutional Accumulation Score (0–100).

India-specific feeds (delivery %, block/bulk deals, options PCR / OI
concentration) need an NSE data subscription and are disclosed as not wired in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forensics.models import (
    FundamentalData, LayerResult, Metric, RedFlag, Severity,
)


def _slope(series: pd.Series) -> float:
    """Normalized linear slope: total fitted change over the window ÷ mean level."""
    y = series.dropna().values.astype(float)
    if len(y) < 10:
        return 0.0
    x = np.arange(len(y))
    m = np.polyfit(x, y, 1)[0]
    denom = abs(np.mean(y)) or 1.0
    return float(m * len(y) / denom)


def analyze(d: FundamentalData) -> LayerResult:
    res = LayerResult(layer="Market Microstructure Intelligence", score=None)
    px = d.prices
    need = {"Close", "High", "Low", "Volume"}
    if px.empty or not need.issubset(px.columns) or len(px) < 120 \
            or px["Volume"].tail(120).sum() == 0:
        res.notes.append("Insufficient OHLCV/volume history — layer skipped.")
        return res

    close, high, low, vol = px["Close"], px["High"], px["Low"], px["Volume"]
    scores = []

    # ── Volume anomaly detection ──────────────────────────────────────────────
    v1y = vol.tail(252)
    base_mean, base_std = v1y.mean(), v1y.std()
    recent = vol.tail(20)
    if base_std > 0:
        z_recent = (recent - base_mean) / base_std
        spikes = z_recent[z_recent > 3]
        ret_on_spikes = close.pct_change().reindex(spikes.index)
        res.metrics.append(Metric(
            name="Volume Anomalies (20d)", value=float(len(spikes)),
            display=f"{len(spikes)} spike(s) >3σ vs 1y baseline",
            what="Days in the last month with volume more than 3 standard "
                 "deviations above the 1-year average.",
            why="Institutions cannot hide size — accumulation and distribution "
                "leave volume footprints before price confirms.",
            good="Occasional spikes on up-days (accumulation); clusters of spikes "
                 "on down-days signal distribution.",
            implication="No abnormal volume." if len(spikes) == 0
            else "Abnormal volume detected — someone large is active.",
        ))
        down_spikes = int((ret_on_spikes < -0.01).sum())
        if down_spikes >= 2:
            res.flags.append(RedFlag(
                title="Clustered high-volume selling",
                severity=Severity.MEDIUM,
                evidence=f"{down_spikes} days in 20 with >3σ volume on >1% declines.",
                why_it_matters="Heavy volume on declines is the signature of "
                               "institutional distribution into retail bids.",
                precedent="Distribution clusters preceded the major legs down in "
                          "Yes Bank (2019) and Paytm post-listing.",
            ))

    # ── VWAP deviation (20d rolling) ──────────────────────────────────────────
    typical = (high + low + close) / 3
    pv = (typical * vol).rolling(20).sum()
    vv = vol.rolling(20).sum()
    vwap = (pv / vv).iloc[-1]
    dev = float(close.iloc[-1] / vwap - 1) if vwap and not np.isnan(vwap) else None
    if dev is not None:
        res.metrics.append(Metric(
            name="VWAP Deviation (20d)", value=dev, display=f"{dev:+.1%}",
            what="Last close versus the 20-day volume-weighted average price.",
            why="VWAP is the institutional execution benchmark; persistent trading "
                "above it means buyers are willing to pay up.",
            good="Modestly above (0 to +5%) during accumulation.",
            implication="Price holding above institutional cost basis."
            if dev > 0 else "Price below VWAP — recent buyers are underwater.",
        ))

    # ── Accumulation/Distribution score ───────────────────────────────────────
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    ad_line = (clv.fillna(0) * vol).cumsum()
    ad_trend = _slope(ad_line.tail(60))
    px_trend = _slope(close.tail(60))
    s_ad = max(0.0, min(100.0, 50 + ad_trend * 250))
    scores.append(s_ad)
    res.metrics.append(Metric(
        name="Accumulation/Distribution (60d)", value=ad_trend,
        display=f"trend {ad_trend:+.2f} → "
                f"{'accumulation' if ad_trend > 0.02 else 'distribution' if ad_trend < -0.02 else 'neutral'}",
        what="Slope of the A/D line (close-location-value × volume, cumulative) "
             "over 60 sessions.",
        why="Measures whether volume is flowing in near daily highs (buyers in "
            "control) or near lows (sellers in control).",
        good="Rising A/D, ideally rising faster than price.",
        implication="Volume is flowing into the stock." if ad_trend > 0
        else "Volume pattern is distributive.",
        score=s_ad,
    ))

    # ── Smart Money Flow: OBV vs price divergence ─────────────────────────────
    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    obv_trend = _slope(obv.tail(60))
    divergence = obv_trend - px_trend
    s_smf = max(0.0, min(100.0, 50 + divergence * 150))
    scores.append(s_smf)
    label = ("Stealth accumulation" if divergence > 0.1 and px_trend <= 0.02
             else "Distribution into strength" if divergence < -0.1 and px_trend >= 0
             else "Confirming")
    res.metrics.append(Metric(
        name="Smart Money Flow (OBV divergence)", value=divergence,
        display=f"OBV {obv_trend:+.2f} vs price {px_trend:+.2f} → {label}",
        what="On-balance-volume trend compared against price trend over 60 days.",
        why="When OBV rises while price stalls, size is being absorbed quietly; "
            "when OBV falls while price holds, smart money is exiting into bids.",
        good="OBV confirming or leading price upward.",
        implication=f"{label}.",
        score=s_smf,
    ))
    if label == "Distribution into strength":
        res.flags.append(RedFlag(
            title="Smart money distribution divergence",
            severity=Severity.MEDIUM,
            evidence=f"OBV trend {obv_trend:+.2f} vs price trend {px_trend:+.2f} "
                     "over 60 sessions.",
            why_it_matters="A flat-to-rising price on falling cumulative volume "
                           "flow means the stock is being handed from strong to "
                           "weak hands.",
            precedent="OBV divergence preceded the tops in most 2021 IPO-era "
                      "high-flyers.",
        ))

    # ── Up-day volume share ───────────────────────────────────────────────────
    last60 = px.tail(60)
    ret60 = last60["Close"].pct_change()
    up_vol = float(last60["Volume"][ret60 > 0].sum())
    tot_vol = float(last60["Volume"].iloc[1:].sum())
    if tot_vol > 0:
        share = up_vol / tot_vol
        s_uv = max(0.0, min(100.0, share * 200 - 50))
        scores.append(s_uv)
        res.metrics.append(Metric(
            name="Up-Day Volume Share (60d)", value=share, display=f"{share:.0%}",
            what="Fraction of total volume that traded on up days.",
            why="The simplest demand/supply scoreboard — institutions buy "
                "strength and create it.",
            good="Above 55%.",
            implication="Demand dominates." if share > 0.55
            else "Supply dominates recent trade.",
            score=s_uv,
        ))

    res.notes.append("Not wired in (needs NSE bhavcopy / options feed): "
                     "delivery % trend, block & bulk deal detection, put/call "
                     "skew, OI concentration. The score below uses "
                     "volume-structure proxies only.")
    res.score = sum(scores) / len(scores) if scores else None
    res.extras["accumulation_score"] = res.score
    return res
