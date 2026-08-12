"""Shared evaluation helpers for phase-next experiments."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.harness import evaluate, effective_sample_size
from research.phase_a5 import metrics as M
from research.phase_next import protocol as P


def cost_pct() -> float:
    return float(round_trip_cost_pct(P.COST_PRODUCT))


def pack_stream(net: pd.Series, *, n_trials: int) -> dict:
    arr = net.to_numpy(dtype=float)
    pack = M.harness_pack(arr, n_trials=n_trials, min_n=30)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gw = float(wins.sum()) if wins.size else 0.0
    gl = float(abs(losses.sum())) if losses.size else 0.0
    pf = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    if arr.size:
        eq = np.cumsum(arr)
        peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())
    else:
        dd = 0.0
    hit = float((arr > 0).mean()) if arr.size else 0.0
    if arr.size >= 2 and arr.std(ddof=1) > 0:
        se = arr.std(ddof=1) / np.sqrt(arr.size)
        ci = [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)]
    else:
        m = float(arr.mean()) if arr.size else 0.0
        ci = [m, m]
    return {
        **pack,
        "hit_rate": round(hit, 4),
        "profit_factor": None if pf == float("inf") else round(pf, 4),
        "max_drawdown": round(dd, 4),
        "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        "mean_gross": None,  # filled by caller
        "cost_drag": round(M.cost_drag(P.TURNOVER_ONE_WAY, cost_pct()), 4),
        "mean_net": pack["mean_r"],
        "n_eff": pack["n_eff"],
    }


def long_short_period(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    invert: bool = False,
    top_q: float = 0.2,
) -> pd.Series:
    """Long high scores (or low if invert) vs short opposite, on given dates only."""
    common = scores.index.intersection(fwd.index).intersection(dates)
    port = []
    out_dates = []
    for dt in common:
        s = scores.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 6:
            continue
        n = max(1, int(len(s) * top_q))
        if invert:
            # reversal: long losers (low formation return) / short winners
            long = s.nsmallest(n).index
            short = s.nlargest(n).index
        else:
            long = s.nlargest(n).index
            short = s.nsmallest(n).index
        r = float(f.loc[long].mean() - f.loc[short].mean())
        port.append(r)
        out_dates.append(dt)
    return pd.Series(port, index=pd.Index(out_dates), dtype=float)


def net_of_costs(gross: pd.Series) -> pd.Series:
    drag = M.cost_drag(P.TURNOVER_ONE_WAY, cost_pct())
    return gross - drag


def realized_vol(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    rets = closes.pct_change()
    return rets.rolling(lookback).std()


def result_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def map_discovery_verdict(harness_verdict: str, *, mean_net: float, fdr_ok: bool) -> str:
    """Map harness + economic gates to research cycle vocabulary."""
    hv = str(harness_verdict or "").upper()
    if mean_net <= 0:
        return "FAIL"
    if hv == "PROMOTE" and fdr_ok:
        return "PASS"
    if hv in {"UNDERPOWERED", "INCONCLUSIVE"}:
        return "INCONCLUSIVE"
    if hv == "REJECT":
        return "FAIL"
    return "INCONCLUSIVE"


def final_after_confirm(discovery: str, confirm: str | None) -> str:
    if discovery in {"FAIL", "INCONCLUSIVE", "BLOCKED"}:
        return discovery if discovery != "PASS" else discovery
    if discovery == "PASS" and confirm is None:
        return "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION"
    if discovery == "PASS" and confirm == "PASS":
        return "CONFIRMED"
    if discovery == "PASS" and confirm == "FAIL":
        return "FAILED_CONFIRMATION"
    if discovery == "PASS" and confirm == "INCONCLUSIVE":
        return "INCONCLUSIVE"
    return discovery
