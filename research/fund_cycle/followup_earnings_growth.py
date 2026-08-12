"""EXP-FUND-03-FOLLOWUP — earnings-growth robustness validation (research only).

Does NOT alter EXP-FUND-03. Does NOT tune the factor. Does NOT grant production
authority. No ML / Phase B.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.fund_cycle import data as D
from research.intelligence.data.snapshot_store import SnapshotStore
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_next import eval_utils as E

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "logs" / "research_expansion" / "fund_cycle" / "followup_03"
CYCLE_RESULT = REPO_ROOT / "logs" / "research_expansion" / "fund_cycle" / "cycle_result.json"
FROZEN_FU = REPO_ROOT / "docs" / "overhaul" / "EXP_FUND_03_FOLLOWUP_FROZEN_PROTOCOL.json"
REPORT_PATH = REPO_ROOT / "EXP_FUND_03_EARNINGS_GROWTH_FOLLOWUP.md"

EXPERIMENT_ID = "EXP-FUND-03-FOLLOWUP"
PARENT_ID = "EXP-FUND-03"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()[:16]
    except Exception:
        return "unknown"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 8)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def _pack_summary(gross: pd.Series, *, cost_drag: float | None = None) -> dict:
    drag = D.cost_drag() if cost_drag is None else float(cost_drag)
    net = gross - drag
    pack = E.pack_stream(net, n_trials=D.N_TRIALS)
    pack["mean_gross"] = round(float(gross.mean()) if len(gross) else 0.0, 6)
    pack["mean_net"] = round(float(net.mean()) if len(net) else 0.0, 6)
    pack["cost_drag"] = round(drag, 6)
    return {
        "n": int(pack.get("n") or 0),
        "mean_gross": pack["mean_gross"],
        "mean_net": pack["mean_net"],
        "cost_drag": pack["cost_drag"],
        "max_drawdown": pack.get("max_drawdown"),
        "hit_rate": pack.get("hit_rate"),
        "p_value": pack.get("p_value"),
        "harness_verdict": pack.get("verdict"),
        "direction": (
            "POSITIVE" if pack["mean_net"] > 0
            else ("NEGATIVE" if pack["mean_net"] < 0 else "ZERO")
        ),
    }


def _load_volumes() -> pd.DataFrame:
    store = SnapshotStore(D.SNAP_ROOT)
    snap = store.open_snapshot(D.OHLCV_ID)
    by_sym: dict[str, dict[str, float]] = {}
    for r in snap._equity:
        by_sym.setdefault(str(r["symbol"]).upper(), {})[r["date"]] = float(r["volume"] or 0)
    vol = pd.DataFrame({s: pd.Series(v) for s, v in by_sym.items()})
    vol.index = pd.to_datetime(vol.index)
    return vol.sort_index()


def _eval_partition(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    cost_drag: float | None = None,
) -> dict:
    fwd = M.forward_returns(closes, D.HOLD)
    reb = D.rebalance_dates(dates, D.REB_CS)
    gross, ew = D.long_short_from_scores(scores, fwd, reb, invert=False)
    summary = _pack_summary(gross, cost_drag=cost_drag)
    summary["n_rebalances"] = int(len(gross))
    summary["median_names"] = (
        int(scores.loc[reb].notna().sum(axis=1).median()) if len(reb) else 0
    )
    summary["ew_mean_gross"] = round(float(ew.mean()) if len(ew) else 0.0, 6)
    summary["turnover_assumption"] = D.TURNOVER
    summary["gross_series"] = gross
    summary["ew_series"] = ew
    return summary


def _compare_to_parent(repro: dict, parent: dict, *, label: str, frozen: dict) -> dict:
    tol = frozen["reproduction"]["tolerances"]
    p_pack = parent["pack"]
    checks = {
        "n_match": int(repro["n"]) == int(p_pack["n"]),
        "mean_net_match": abs(float(repro["mean_net"]) - float(p_pack["mean_net"]))
            <= float(tol["mean_net_abs"]),
        "mean_gross_match": abs(float(repro["mean_gross"]) - float(p_pack["mean_gross"]))
            <= float(tol["mean_gross_abs"]),
        "parent_partition_pass": parent.get("verdict") == "PASS",
    }
    ok = all([checks["n_match"], checks["mean_net_match"], checks["mean_gross_match"]])
    return {
        "label": label,
        "ok": ok,
        "checks": checks,
        "reproduced": {
            "n": repro["n"],
            "mean_gross": repro["mean_gross"],
            "mean_net": repro["mean_net"],
            "harness_verdict": repro["harness_verdict"],
        },
        "stored": {
            "n": p_pack["n"],
            "mean_gross": p_pack.get("mean_gross"),
            "mean_net": p_pack.get("mean_net"),
            "harness_verdict": p_pack.get("verdict"),
            "partition_verdict": parent.get("verdict"),
        },
    }


def _cohort_members(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    top_q: float = D.Q,
) -> list[dict]:
    rows = []
    for dt in dates:
        if dt not in scores.index or dt not in fwd.index:
            continue
        s = scores.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 10:
            continue
        n = max(1, int(len(s) * top_q))
        long = list(s.nlargest(n).index)
        short = list(s.nsmallest(n).index)
        long_r = float(f.loc[long].mean())
        short_r = float(f.loc[short].mean())
        ls = long_r - short_r
        for side, names in (("long", long), ("short", short)):
            for sym in names:
                r = float(f.loc[sym])
                contrib = (r / n) if side == "long" else (-r / n)
                rows.append({
                    "date": dt,
                    "symbol": sym,
                    "side": side,
                    "growth": float(s.loc[sym]),
                    "fwd_ret": r,
                    "contrib": contrib,
                    "ls": ls,
                })
    return rows


def _monotonicity(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict:
    fwd = M.forward_returns(closes, D.HOLD)
    reb = D.rebalance_dates(dates, D.REB_CS)
    bucket_rets: dict[int, list[float]] = {i: [] for i in range(1, 6)}
    for dt in reb:
        if dt not in scores.index or dt not in fwd.index:
            continue
        s = scores.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 25:
            continue
        try:
            q = pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        except ValueError:
            ranks = s.rank(method="first")
            q = pd.cut(ranks, 5, labels=[1, 2, 3, 4, 5])
        if getattr(q, "nunique", lambda: 0)() < 5:
            ranks = s.rank(method="first")
            q = pd.cut(ranks, 5, labels=[1, 2, 3, 4, 5])
        for b in range(1, 6):
            mask = q.astype(int) == b
            if mask.any():
                bucket_rets[b].append(float(f.loc[mask].mean()))
    summary = {}
    means = []
    for b in range(1, 6):
        ser = pd.Series(bucket_rets[b], dtype=float)
        m = float(ser.mean()) if len(ser) else float("nan")
        means.append(m)
        summary[f"Q{b}"] = {
            "label": "low_growth" if b == 1 else ("high_growth" if b == 5 else "mid"),
            "n_rebalances": int(len(ser)),
            "mean_fwd_gross": None if math.isnan(m) else round(m, 6),
        }
    valid = [(i + 1, m) for i, m in enumerate(means) if not math.isnan(m)]
    mono = False
    rho = None
    if len(valid) >= 4:
        xs = np.array([v[0] for v in valid], dtype=float)
        ys = np.array([v[1] for v in valid], dtype=float)
        if ys.std() > 0:
            rho = float(pd.Series(xs).corr(pd.Series(ys), method="spearman"))
            mono = bool(rho > 0.6 and means[-1] > means[0])
        else:
            rho = 0.0
    summary["spearman_bucket_vs_return"] = None if rho is None else round(rho, 4)
    summary["broadly_monotonic"] = bool(mono)
    summary["high_minus_low"] = (
        None if any(math.isnan(m) for m in (means[0], means[-1]))
        else round(means[-1] - means[0], 6)
    )
    return summary


def _audit_pit_yoy(fundamentals: pd.DataFrame, yoy: pd.DataFrame) -> dict:
    df = fundamentals.copy()
    df["available_at"] = pd.to_datetime(df["available_at"])
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df[df["period"].astype(str).str.lower().eq("quarterly")].copy()
    df = df.dropna(subset=["basic_eps", "period_end", "available_at"])
    df["_consol_score"] = (
        df["consolidated"].astype(str).str.lower().str.startswith("consolid").astype(int)
    )
    filings = df.sort_values(["symbol", "period_end", "available_at"])
    leak_rows = []
    kept = (
        filings.sort_values(["symbol", "period_end", "_consol_score", "available_at"])
        .drop_duplicates(["symbol", "period_end"], keep="last")
    )
    for sym, g in kept.groupby("symbol"):
        g = g.sort_values("period_end")
        eps = g.set_index("period_end")["basic_eps"]
        avail = g.set_index("period_end")["available_at"]
        for pe, e in eps.items():
            target = pe - pd.DateOffset(years=1)
            diffs = (eps.index.to_series() - target).abs()
            hit = diffs[diffs <= pd.Timedelta(days=40)]
            if hit.empty:
                continue
            prev_pe = hit.idxmin()
            prev_avail = avail.loc[prev_pe]
            cur_avail = avail.loc[pe]
            if pd.Timestamp(prev_avail) > pd.Timestamp(cur_avail):
                leak_rows.append({
                    "symbol": sym,
                    "period_end": str(pe.date()),
                    "current_available_at": str(pd.Timestamp(cur_avail).date()),
                    "prev_period_end": str(prev_pe.date()),
                    "prev_available_at": str(pd.Timestamp(prev_avail).date()),
                })
    n_period = filings.groupby(["symbol", "period_end"]).size()
    n_amended = int((n_period > 1).sum())
    dup_same_day = int(
        filings.duplicated(["symbol", "period_end", "available_at"], keep=False).sum()
    )
    material = len(leak_rows) > 0
    frac = (len(leak_rows) / max(len(yoy), 1))
    return {
        "restatement_lookahead_rows": len(leak_rows),
        "yoy_rows_total": int(len(yoy)),
        "leak_fraction_of_yoy": round(frac, 6),
        "material_pit_issue": bool(material and frac >= 0.01),
        "any_lookahead": bool(material),
        "amended_period_ends": n_amended,
        "duplicate_same_available_at_rows": dup_same_day,
        "examples": leak_rows[:10],
        "note": (
            "Parent _yoy_eps_map keeps last filing per period_end then tags growth "
            "with current available_at; if prior-year EPS was restated later than "
            "current filing, prev_eps can leak backward."
        ),
    }


def _placebo_premature(panel: D.FundPanel, yoy: pd.DataFrame) -> dict:
    """Visibility at period_end (before true AVAILABLE_AT)."""
    fake = yoy.copy()
    fake["available_at"] = pd.to_datetime(fake["period_end"])
    days = D.trading_days(panel.closes)
    window = D.period_mask(days, D.DISCOVERY_START, D.CONFIRM_END)
    scores = D.latest_asof_frame(
        fake, value_col="yoy_eps_growth", asof_col="available_at",
        dates=days, symbols=list(panel.closes.columns),
    )
    ev = _eval_partition(scores, panel.closes, window)
    return {
        "id": "PLACEBO_PREMATURE_PERIOD_END",
        "summary": {k: v for k, v in ev.items() if k not in ("gross_series", "ew_series")},
        "suspicious_if": "mean_net clearly positive and similar to true-signal edge",
        "flag_suspicious": bool(ev["mean_net"] > 0.003 and ev["n"] >= 30),
    }


def _placebo_pre_release(panel: D.FundPanel, yoy: pd.DataFrame) -> dict:
    """Growth known at AVAILABLE_AT should not predict the prior 21-session return."""
    days = D.trading_days(panel.closes)
    closes = panel.closes
    rows = []
    y = yoy.copy()
    y["symbol"] = y["symbol"].astype(str).str.upper()
    y["available_at"] = pd.to_datetime(y["available_at"])
    for _, r in y.iterrows():
        avail = pd.Timestamp(r["available_at"]).normalize()
        if avail < pd.Timestamp(D.DISCOVERY_START) or avail > pd.Timestamp(D.CONFIRM_END):
            continue
        prior = days[days <= avail]
        if len(prior) < D.HOLD + 1:
            continue
        end = prior[-1]
        start_idx = days.get_indexer([end])[0] - D.HOLD
        if start_idx < 0:
            continue
        start = days[start_idx]
        sym = r["symbol"]
        if sym not in closes.columns:
            continue
        p0 = closes.at[start, sym]
        p1 = closes.at[end, sym]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue
        rows.append({
            "available_at": avail,
            "symbol": sym,
            "growth": float(r["yoy_eps_growth"]),
            "pre_ret": float(p1 / p0 - 1.0),
        })
    if len(rows) < 50:
        return {
            "id": "PLACEBO_PRE_RELEASE_WINDOW",
            "summary": {"n": len(rows), "mean_net": None},
            "flag_suspicious": False,
            "note": "insufficient pre-release sample",
        }
    df = pd.DataFrame(rows)
    n = max(1, int(len(df) * D.Q))
    long = df.nlargest(n, "growth")["pre_ret"]
    short = df.nsmallest(n, "growth")["pre_ret"]
    df["ym"] = df["available_at"].dt.to_period("M")
    monthly = []
    for _, g in df.groupby("ym"):
        if len(g) < 20:
            continue
        nn = max(1, int(len(g) * D.Q))
        monthly.append(
            float(g.nlargest(nn, "growth")["pre_ret"].mean()
                  - g.nsmallest(nn, "growth")["pre_ret"].mean())
        )
    gross = pd.Series(monthly, dtype=float)
    summary = _pack_summary(gross) if len(gross) else {
        "n": 0, "mean_gross": 0.0, "mean_net": 0.0, "direction": "ZERO"
    }
    flag = bool(summary.get("mean_net", 0) > 0.003 and summary.get("n", 0) >= 8)
    return {
        "id": "PLACEBO_PRE_RELEASE_WINDOW",
        "summary": summary,
        "overall_ls_pre_ret": round(float(long.mean() - short.mean()), 6),
        "flag_suspicious": flag,
        "suspicious_if": "growth predicts returns BEFORE AVAILABLE_AT",
    }


def _incrementality(scores: pd.DataFrame, closes: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    mom = M.cross_sectional_momentum_scores(closes, lookback=60)
    resid = scores.copy() * np.nan
    for dt in scores.index.intersection(mom.index):
        s = scores.loc[dt].dropna()
        m = mom.loc[dt].reindex(s.index).dropna()
        s = s.reindex(m.index).dropna()
        if len(s) < 30:
            continue
        x = m.values.astype(float)
        y = s.values.astype(float)
        x = (x - x.mean())
        denom = float((x * x).sum())
        beta = 0.0 if denom <= 1e-12 else float((x * (y - y.mean())).sum() / denom)
        resid.loc[dt, s.index] = y - (beta * m.loc[s.index].values + y.mean())

    raw = _eval_partition(scores, closes, dates)
    orth = _eval_partition(resid, closes, dates)
    mom_ls = _eval_partition(mom.reindex(scores.index), closes, dates)

    def _strip(ev: dict) -> dict:
        return {k: v for k, v in ev.items() if k not in ("gross_series", "ew_series")}

    incremental = bool(
        orth["mean_net"] > 0
        and orth["n"] >= 30
        and orth["mean_net"] >= 0.4 * max(raw["mean_net"], 1e-9)
    )
    return {
        "raw_growth": _strip(raw),
        "momentum_60d_ls": _strip(mom_ls),
        "growth_residualized_vs_mom60": _strip(orth),
        "contains_incremental_info": incremental,
        "note": "Diagnostic only — not a new factor; PE/quality not used as optimizers.",
    }


def _liquidity_diag(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict:
    reb = D.rebalance_dates(dates, D.REB_CS)
    adv = volumes.rolling(21, min_periods=5).mean()
    dvol = adv * closes

    thin_shares = []
    selected_adv = []
    univ_adv = []
    n_long = []
    n_short = []
    for dt in reb:
        if dt not in scores.index or dt not in dvol.index:
            continue
        s = scores.loc[dt].dropna()
        if len(s) < 10:
            continue
        n = max(1, int(len(s) * D.Q))
        long = list(s.nlargest(n).index)
        short = list(s.nsmallest(n).index)
        selected = long + short
        dv = dvol.loc[dt].reindex(s.index).dropna()
        if len(dv) < 10:
            continue
        p20 = float(dv.quantile(0.2))
        sel_dv = dvol.loc[dt].reindex(selected).dropna()
        thin = int((sel_dv < p20).sum()) if len(sel_dv) else 0
        thin_shares.append(thin / max(len(sel_dv), 1))
        selected_adv.extend([float(x) for x in sel_dv.values])
        univ_adv.extend([float(x) for x in dv.values])
        n_long.append(len(long))
        n_short.append(len(short))

    med_sel = float(np.median(selected_adv)) if selected_adv else None
    med_univ = float(np.median(univ_adv)) if univ_adv else None
    mean_thin = float(np.mean(thin_shares)) if thin_shares else None
    return {
        "adv_window": 21,
        "median_selected_dollar_volume_proxy": med_sel,
        "median_universe_dollar_volume_proxy": med_univ,
        "selected_vs_universe_median_ratio": (
            None if not (med_sel and med_univ) else round(med_sel / med_univ, 4)
        ),
        "mean_share_of_selected_in_bottom_adv_quintile": (
            None if mean_thin is None else round(mean_thin, 4)
        ),
        "median_long_count": int(np.median(n_long)) if n_long else 0,
        "median_short_count": int(np.median(n_short)) if n_short else 0,
        "capacity_estimate": None,
        "capacity_note": (
            "No scientifically supportable capacity estimate from current data "
            "(ADV proxy only; no depth/impact model)."
        ),
        "liquidity_concern": bool(mean_thin is not None and mean_thin > 0.35),
    }


def _concentration_diag(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict:
    fwd = M.forward_returns(closes, D.HOLD)
    reb = D.rebalance_dates(dates, D.REB_CS)
    gross, _ = D.long_short_from_scores(scores, fwd, reb, invert=False)
    members = _cohort_members(scores, fwd, reb)

    if len(gross) == 0:
        return {"empty": True}
    pos = gross.clip(lower=0)
    total_pos = float(pos.sum()) if float(pos.sum()) > 0 else float(gross.abs().sum())
    top5_reb = gross.nlargest(5)
    share_top5_reb = float(top5_reb.clip(lower=0).sum() / total_pos) if total_pos else 0.0

    mdf = pd.DataFrame(members)
    if mdf.empty:
        name_share = None
        top_names = []
    else:
        by_name = mdf.groupby("symbol")["contrib"].sum().sort_values(ascending=False)
        pos_names = by_name.clip(lower=0)
        tot = float(pos_names.sum()) if float(pos_names.sum()) > 0 else float(by_name.abs().sum())
        top5_names = by_name.head(5)
        name_share = float(top5_names.clip(lower=0).sum() / tot) if tot else 0.0
        top_names = [
            {"symbol": str(i), "contrib_sum": round(float(v), 6)}
            for i, v in top5_names.items()
        ]

    return {
        "n_rebalances": int(len(gross)),
        "mean_ls_gross": round(float(gross.mean()), 6),
        "median_ls_gross": round(float(gross.median()), 6),
        "mean_vs_median_gap": round(float(gross.mean() - gross.median()), 6),
        "share_positive_pnl_from_top5_rebalances": round(share_top5_reb, 4),
        "top5_rebalances": [
            {"date": str(i.date()), "ls_gross": round(float(v), 6)}
            for i, v in top5_reb.items()
        ],
        "share_positive_contrib_from_top5_names": (
            None if name_share is None else round(name_share, 4)
        ),
        "top5_names": top_names,
        "lottery_like": bool(
            share_top5_reb > 0.55
            or (name_share is not None and name_share > 0.40)
            or (float(gross.mean()) > 0 and float(gross.median()) <= 0)
        ),
    }


def _decide_verdict(bundle: dict) -> tuple[str, str, list[str]]:
    reasons: list[str] = []
    repro_ok = bool(bundle["reproduction"]["pass"])
    pit = bundle["pit_audit"]
    pit_bad = bool(pit.get("material_pit_issue") or (
        pit.get("any_lookahead") and pit.get("leak_fraction_of_yoy", 0) >= 0.01
    ))
    placebo_bad = any(p.get("flag_suspicious") for p in bundle["placebos"])
    cost = bundle["cost_robustness"]
    base_net = float(cost["scenarios"]["0.32"]["mean_net"])
    high_net = float(cost["scenarios"]["0.50"]["mean_net"])
    fragile = bool(base_net > 0 and high_net <= 0)
    be = cost.get("break_even_round_trip_pct_points")
    if be is not None and be < 0.40:
        fragile = True
        reasons.append(f"break-even cost {be:.3f} pct pts is tight vs base 0.32")

    conc = bundle["concentration"]
    concentrated = bool(conc.get("lottery_like"))

    subs = bundle["subperiods"]
    dirs = [s["direction"] for s in subs.values()]
    pos_count = sum(1 for d in dirs if d == "POSITIVE")
    time_ok = pos_count >= 3
    if pos_count <= 1:
        reasons.append("effect concentrated in <=1 subperiod")

    mono_ok = bool(bundle["monotonicity"].get("broadly_monotonic"))
    liq_bad = bool(bundle["liquidity"].get("liquidity_concern"))

    if not repro_ok:
        return "FAILED_ROBUSTNESS", "RECORD_EVIDENCE_NO_TUNING", [
            "Exact reproduction of EXP-FUND-03 failed — branch must not advance."
        ]

    # Restatement / AVAILABLE_AT construction leakage is a hard scientific blocker.
    if pit_bad:
        return "DATA_CONCERN", "RECORD_EVIDENCE_NO_TUNING", [
            "Material AVAILABLE_AT / restatement look-ahead detected in YoY construction."
        ]

    # Placebo success is a leakage WARNING (F9). Blocks ROBUST_CONFIRMED; alone it
    # is not a restatement-style DATA_CONCERN when the PIT audit is clean.
    if placebo_bad:
        reasons.append(
            "Placebo/negative control looked suspiciously strong "
            "(possible anticipation or construction issue)."
        )

    if not time_ok and pos_count <= 1:
        return "FAILED_ROBUSTNESS", "RECORD_EVIDENCE_NO_TUNING", reasons

    if fragile and concentrated:
        return "CONFIRMED_BUT_FRAGILE", "RECORD_EVIDENCE_NO_TUNING", [
            "Edge dies under modestly higher costs and is observation-concentrated."
        ] + reasons

    if fragile:
        return "CONFIRMED_BUT_FRAGILE", "RECORD_EVIDENCE_NO_TUNING", [
            "Economic edge does not survive preregistered higher-cost scenarios."
        ] + reasons

    if concentrated:
        return "CONFIRMED_BUT_CONCENTRATED", "RECORD_EVIDENCE_NO_TUNING", [
            "Result depends disproportionately on few rebalances/names."
        ] + reasons

    needs = []
    if placebo_bad:
        needs.append("placebo controls not clean — cannot clear robust bar")
    if not time_ok:
        needs.append("subperiod direction not broad enough")
    if not mono_ok:
        needs.append("cross-sectional monotonicity weak")
    if liq_bad:
        needs.append("selected book skews toward thin ADV names")
    if base_net <= 0:
        needs.append("base-cost net edge non-positive on pooled window")
    if high_net <= 0:
        needs.append("0.50 pct-pt cost scenario destroys edge")
    if not bundle["incrementality"].get("contains_incremental_info"):
        needs.append("little incremental information vs 60d momentum")

    if needs:
        if base_net > 0 and time_ok and not fragile:
            return "INCONCLUSIVE_FOLLOWUP", "RECORD_EVIDENCE_NO_TUNING", needs
        return "FAILED_ROBUSTNESS", "RECORD_EVIDENCE_NO_TUNING", needs

    return (
        "ROBUST_CONFIRMED",
        "DESIGN_PAPER_SHADOW_POLICY_EXPERIMENT",
        ["All preregistered robustness gates cleared; still NOT live-authorized."],
    )


def _write_report(bundle: dict) -> None:
    v = bundle["final_verdict"]
    next_a = bundle["next_action"]
    plain = bundle["plain"]

    lines = [
        "# EXP-FUND-03-FOLLOWUP — Earnings Growth Follow-up Validation",
        "",
        "> Scientific follow-up only. Does **not** overwrite EXP-FUND-03.",
        "> Production unchanged. No ML. No Phase B. Not a live trading authorization.",
        "",
        "## WHAT WE ALREADY KNEW",
        "",
        "Earnings growth was the first QuantTerm idea to pass both discovery and an "
        "independent historical confirmation.",
        "",
        "## WHAT WE CHECKED NOW",
        "",
        "We checked whether that result was broad, repeatable, practical after costs, "
        "and dependent on only a few lucky stocks.",
        "",
        "## WHAT HAPPENED",
        "",
        plain["what_happened"],
        "",
        "## WHAT QUANTTERM WILL DO",
        "",
        plain["what_quantterm_will_do"],
        "",
        "---",
        "",
        "## Technical evidence",
        "",
        f"- Follow-up experiment ID: `{EXPERIMENT_ID}`",
        f"- Parent: `{PARENT_ID}` (hypothesis `{bundle['parent']['hypothesis_id']}`, "
        f"result hash `{bundle['parent']['result_hash']}`)",
        f"- Follow-up hypothesis ID: `{bundle['hypothesis_id']}`",
        f"- Foundation package: `{D.FOUNDATION_ID}`",
        f"- Parent OHLCV snapshot: `{D.OHLCV_ID}`",
        f"- Frozen protocol: `docs/overhaul/EXP_FUND_03_FOLLOWUP_FROZEN_PROTOCOL.json`",
        f"- Reproduction: **{'PASS' if bundle['reproduction']['pass'] else 'FAIL'}**",
        f"- Final follow-up verdict: **{v}**",
        f"- Next action: `{next_a}`",
        f"- Production authority: `False`",
        f"- Git SHA: `{bundle['git_sha']}`",
        f"- Result hash: `{bundle['result_hash']}`",
        "",
        "### F1 — Frozen confirmed effect",
        "",
        "Parent definition, AVAILABLE_AT treatment, universe, quintile L/S, reb=5, "
        "hold=21, CNC costs, and confirmation metrics are frozen in the follow-up "
        "protocol. Parent EXP-FUND-03 registry entry was **not** overwritten.",
        "",
        "### F2 — Reproduction",
        "",
        "```json",
        json.dumps(_jsonable(bundle["reproduction"]), indent=2),
        "```",
        "",
        "### F3 — Subperiod stability",
        "",
        "```json",
        json.dumps(_jsonable(bundle["subperiods"]), indent=2),
        "```",
        "",
        "### F4 — Cross-sectional monotonicity",
        "",
        "```json",
        json.dumps(_jsonable(bundle["monotonicity"]), indent=2),
        "```",
        "",
        "### F5 — Cost / turnover robustness",
        "",
        "```json",
        json.dumps(_jsonable(bundle["cost_robustness"]), indent=2),
        "```",
        "",
        "### F6 — Liquidity / implementability",
        "",
        "```json",
        json.dumps(_jsonable(bundle["liquidity"]), indent=2),
        "```",
        "",
        "### F7 — Concentration",
        "",
        "```json",
        json.dumps(_jsonable(bundle["concentration"]), indent=2),
        "```",
        "",
        "### F8 — Fundamental-data / AVAILABLE_AT audit",
        "",
        "```json",
        json.dumps(_jsonable(bundle["pit_audit"]), indent=2),
        "```",
        "",
        "### F9 — Placebo / negative controls",
        "",
        "```json",
        json.dumps(_jsonable(bundle["placebos"]), indent=2),
        "```",
        "",
        "### F10 — Benchmark incrementality",
        "",
        "```json",
        json.dumps(_jsonable(bundle["incrementality"]), indent=2),
        "```",
        "",
        "### Verdict rationale",
        "",
    ]
    for r in bundle["verdict_reasons"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "### Status card",
        "",
        "| Field | Value |",
        "|---|---|",
        "| ORIGINAL RESULT | CONFIRMED (EXP-FUND-03; net≈+0.71% discovery) |",
        f"| REPRODUCIBLE? | {'YES' if bundle['reproduction']['pass'] else 'NO'} |",
        f"| TIME-STABLE? | {bundle['status_card']['time_stable']} |",
        f"| MONOTONIC? | {bundle['status_card']['monotonic']} |",
        f"| COST-ROBUST? | {bundle['status_card']['cost_robust']} |",
        f"| LIQUIDITY ACCEPTABLE? | {bundle['status_card']['liquidity_ok']} |",
        f"| CONCENTRATION ACCEPTABLE? | {bundle['status_card']['concentration_ok']} |",
        f"| PIT/AVAILABLE_AT CLEAN? | {bundle['status_card']['pit_clean']} |",
        f"| PLACEBO CLEAN? | {bundle['status_card']['placebo_clean']} |",
        f"| FINAL FOLLOW-UP VERDICT | {v} |",
        f"| NEXT ACTION | {next_a} |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN_FU.read_text())
    assert frozen["frozen_before_outcome_inspection"] is True
    assert frozen["do_not_alter_parent"] is True
    assert frozen["parent_experiment_id"] == PARENT_ID

    parent_cycle = json.loads(CYCLE_RESULT.read_text())
    parent = next(
        r for r in parent_cycle["results"] if r["experiment_id"] == PARENT_ID
    )

    panel = D.load_panel()
    yoy = D._yoy_eps_map(panel.fundamentals)
    days = D.trading_days(panel.closes)
    scores = D.latest_asof_frame(
        yoy, value_col="yoy_eps_growth", asof_col="available_at",
        dates=days, symbols=list(panel.closes.columns),
    )

    hid = prereg.preregister(
        experiment_id=EXPERIMENT_ID,
        hypothesis=(
            "The CONFIRMED EXP-FUND-03 earnings-growth effect remains robust, "
            "economically meaningful, implementable, cost-tolerant, and stable "
            "enough to justify designing a future paper/shadow policy experiment."
        ),
        null_hypothesis=(
            "Follow-up diagnostics show fragility, concentration, PIT concern, "
            "irreproducibility, or economic non-viability; no paper/shadow design."
        ),
        success_criteria={
            # Follow-up hypothesis succeeds only if robustness clears.
            "reproduction_pass": {"eq": 1},
            "robust_confirmed": {"eq": 1},
            "live_behaviour_changed": {"eq": 0},
        },
        data_window={
            "foundation_id": D.FOUNDATION_ID,
            "ohlcv_snapshot_id": D.OHLCV_ID,
            "parent_experiment": PARENT_ID,
            "parent_hypothesis_id": parent["hypothesis_id"],
            "window": f"{D.DISCOVERY_START}→{D.CONFIRM_END}",
        },
        protocol={
            "type": "FOLLOWUP_ROBUSTNESS",
            "frozen_protocol": str(FROZEN_FU.relative_to(REPO_ROOT)),
            "no_ml": True,
            "no_signal_tuning": True,
            "production_authority": False,
        },
        seed=42,
        code_hash="exp_fund_03_followup_v1",
    )

    disc_days = D.period_mask(days, D.DISCOVERY_START, D.DISCOVERY_END)
    conf_days = D.period_mask(days, D.CONFIRM_START, D.CONFIRM_END)
    disc = _eval_partition(scores, panel.closes, disc_days)
    conf = _eval_partition(scores, panel.closes, conf_days)
    c_disc = _compare_to_parent(
        disc, parent["discovery"], label="discovery", frozen=frozen
    )
    c_conf = _compare_to_parent(
        conf, parent["confirmation"], label="confirmation", frozen=frozen
    )
    repro_pass = bool(c_disc["ok"] and c_conf["ok"])
    reproduction = {
        "pass": repro_pass,
        "package": D.FOUNDATION_ID,
        "ohlcv_snapshot": D.OHLCV_ID,
        "discovery": c_disc,
        "confirmation": c_conf,
        "same_config": {
            "reb": D.REB_CS,
            "hold": D.HOLD,
            "q": D.Q,
            "cost_drag": D.cost_drag(),
            "n_trials": D.N_TRIALS,
        },
    }

    if not repro_pass:
        verdict = "FAILED_ROBUSTNESS"
        next_action = "RECORD_EVIDENCE_NO_TUNING"
        reasons = ["Exact reproduction failed — STOP per mission rules."]
        bundle = {
            "experiment_id": EXPERIMENT_ID,
            "hypothesis_id": hid,
            "parent": {
                "experiment_id": PARENT_ID,
                "hypothesis_id": parent["hypothesis_id"],
                "result_hash": parent["result_hash"],
                "final_verdict": parent["final_verdict"],
            },
            "reproduction": reproduction,
            "subperiods": {},
            "monotonicity": {},
            "cost_robustness": {},
            "liquidity": {},
            "concentration": {},
            "pit_audit": {},
            "placebos": [],
            "incrementality": {},
            "final_verdict": verdict,
            "next_action": next_action,
            "verdict_reasons": reasons,
            "status_card": {
                "time_stable": "N/A",
                "monotonic": "N/A",
                "cost_robust": "N/A",
                "liquidity_ok": "N/A",
                "concentration_ok": "N/A",
                "pit_clean": "N/A",
                "placebo_clean": "N/A",
            },
            "plain": {
                "what_happened": (
                    "We could not exactly reproduce the confirmed earnings-growth "
                    "result from the stored artifacts, so follow-up stopped."
                ),
                "what_quantterm_will_do": (
                    "Nothing further on this branch until reproducibility is restored. "
                    "No tuning. No live use."
                ),
            },
            "git_sha": _git_sha(),
            "result_hash": E.result_hash({"repro": False, "v": verdict}),
            "live_behaviour_changed": 0,
        }
        reg = prereg.record(hid, {
            "reproduction_pass": 0,
            "robust_confirmed": 0,
            "mean_net": disc["mean_net"],
            "live_behaviour_changed": 0,
        })
        bundle["registry_status"] = reg.get("status")
        prereg.remember_negative(
            f"{EXPERIMENT_ID} reproduction failed vs EXP-FUND-03 artifacts",
            signal="earnings_growth_followup",
            evidence_n=int(disc["n"]),
            notes="STOP — do not advance; do not tune.",
        )
        _write_report(bundle)
        (OUT_DIR / "followup_result.json").write_text(
            json.dumps(_jsonable(bundle), indent=2), encoding="utf-8"
        )
        return bundle

    subperiods = {}
    for sp in frozen["diagnostics_preregistered"]["subperiods_chronological"]:
        dsub = D.period_mask(days, sp["start"], sp["end"])
        ev = _eval_partition(scores, panel.closes, dsub)
        subperiods[sp["id"]] = {
            "start": sp["start"],
            "end": sp["end"],
            "gross_effect": ev["mean_gross"],
            "net_effect": ev["mean_net"],
            "sample_size": ev["n"],
            "turnover_assumption": D.TURNOVER,
            "drawdown": ev["max_drawdown"],
            "effect_direction": ev["direction"],
            "direction": ev["direction"],
            "hit_rate": ev["hit_rate"],
        }

    full_days = D.period_mask(days, D.DISCOVERY_START, D.CONFIRM_END)
    pooled = _eval_partition(scores, panel.closes, full_days)
    mono = _monotonicity(scores, panel.closes, full_days)

    scenarios = {}
    for c_pts in frozen["diagnostics_preregistered"]["cost_scenarios_round_trip_pct_points"]:
        drag = M.cost_drag(D.TURNOVER, float(c_pts))
        ev = _eval_partition(scores, panel.closes, full_days, cost_drag=drag)
        scenarios[f"{c_pts:.2f}"] = {
            "round_trip_pct_points": c_pts,
            "cost_drag": drag,
            "mean_gross": ev["mean_gross"],
            "mean_net": ev["mean_net"],
            "n": ev["n"],
            "direction": ev["direction"],
        }
    be = None
    if D.TURNOVER > 0 and pooled["mean_gross"] is not None:
        be = round(float(pooled["mean_gross"]) / D.TURNOVER * 100.0, 4)
    cost_robustness = {
        "gross_edge": pooled["mean_gross"],
        "base_cost_edge": scenarios["0.32"]["mean_net"],
        "higher_cost_edge_0.50": scenarios["0.50"]["mean_net"],
        "higher_cost_edge_0.75": scenarios["0.75"]["mean_net"],
        "higher_cost_edge_1.00": scenarios["1.00"]["mean_net"],
        "turnover": D.TURNOVER,
        "break_even_round_trip_pct_points": be,
        "scenarios": scenarios,
        "economically_fragile": bool(
            scenarios["0.32"]["mean_net"] > 0 and scenarios["0.50"]["mean_net"] <= 0
        ),
    }

    volumes = _load_volumes()
    liquidity = _liquidity_diag(scores, panel.closes, volumes, full_days)
    concentration = _concentration_diag(scores, panel.closes, full_days)
    pit_audit = _audit_pit_yoy(panel.fundamentals, yoy)
    placebos = [
        _placebo_premature(panel, yoy),
        _placebo_pre_release(panel, yoy),
    ]
    for p in placebos:
        if isinstance(p.get("summary"), dict):
            p["summary"] = {
                k: v for k, v in p["summary"].items()
                if k not in ("gross_series", "ew_series")
            }
    incrementality = _incrementality(scores, panel.closes, full_days)

    bundle = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": hid,
        "parent": {
            "experiment_id": PARENT_ID,
            "hypothesis_id": parent["hypothesis_id"],
            "result_hash": parent["result_hash"],
            "final_verdict": parent["final_verdict"],
            "registry_status": parent.get("registry_status"),
        },
        "reproduction": reproduction,
        "pooled_window": {
            k: v for k, v in pooled.items() if k not in ("gross_series", "ew_series")
        },
        "subperiods": subperiods,
        "monotonicity": mono,
        "cost_robustness": cost_robustness,
        "liquidity": liquidity,
        "concentration": concentration,
        "pit_audit": pit_audit,
        "placebos": placebos,
        "incrementality": incrementality,
        "git_sha": _git_sha(),
        "live_behaviour_changed": 0,
    }

    verdict, next_action, reasons = _decide_verdict(bundle)
    bundle["final_verdict"] = verdict
    bundle["next_action"] = next_action
    bundle["verdict_reasons"] = reasons
    bundle["result_hash"] = E.result_hash({
        "repro": True,
        "v": verdict,
        "net": pooled["mean_net"],
        "be": be,
    })

    pos_subs = sum(1 for s in subperiods.values() if s["direction"] == "POSITIVE")
    bundle["status_card"] = {
        "time_stable": "YES" if pos_subs >= 3 else ("PARTIAL" if pos_subs == 2 else "NO"),
        "monotonic": "YES" if mono.get("broadly_monotonic") else "NO",
        "cost_robust": (
            "YES" if (
                cost_robustness["base_cost_edge"] > 0
                and cost_robustness["higher_cost_edge_0.50"] > 0
                and not cost_robustness["economically_fragile"]
            ) else "NO"
        ),
        "liquidity_ok": "NO" if liquidity.get("liquidity_concern") else "YES",
        "concentration_ok": "NO" if concentration.get("lottery_like") else "YES",
        "pit_clean": (
            "NO" if (
                pit_audit.get("material_pit_issue")
                or (pit_audit.get("any_lookahead") and pit_audit.get("leak_fraction_of_yoy", 0) >= 0.01)
            ) else (
                "CAUTION" if pit_audit.get("any_lookahead") else "YES"
            )
        ),
        "placebo_clean": (
            "NO" if any(p.get("flag_suspicious") for p in placebos) else "YES"
        ),
    }

    if verdict == "ROBUST_CONFIRMED":
        happened = (
            "The confirmed earnings-growth effect reproduced cleanly and looked "
            "broad enough through time, costs, and controls to justify designing a "
            "future paper/shadow policy test — not live trading."
        )
        will = (
            "Next action is DESIGN_PAPER_SHADOW_POLICY_EXPERIMENT only. "
            "Brain, rankings, sizing, and execution stay unchanged."
        )
    elif verdict == "CONFIRMED_BUT_FRAGILE":
        happened = (
            "The original confirmation still reproduces, but the economic edge looks "
            "fragile once realistic implementation friction rises."
        )
        will = (
            "Record the evidence. Do not tune the growth definition. Do not go live."
        )
    elif verdict == "CONFIRMED_BUT_CONCENTRATED":
        happened = (
            "The effect still reproduces, but too much of the result comes from a "
            "small set of dates or names to treat it as a broad, stable edge."
        )
        will = "Record the evidence. No tuning. No live use."
    elif verdict == "DATA_CONCERN":
        happened = (
            "Follow-up found a data-integrity or leakage concern that blocks treating "
            "the confirmed result as scientifically clean for implementation design."
        )
        will = (
            "Stop advancement on this branch until the PIT/data issue is resolved. "
            "Do not tune the factor to paper over it."
        )
    elif verdict == "FAILED_ROBUSTNESS":
        happened = (
            "The original confirmation remains on record, but follow-up robustness "
            "checks did not support treating the effect as stable enough to advance."
        )
        will = "Record the evidence. No retuning. No live use. No Phase B."
    else:
        happened = (
            "Follow-up checks were mixed: some supportive, some weak. Not enough to "
            "clear a robust-confirmed bar, and not a clean failure on one gate."
        )
        will = "Hold as inconclusive follow-up. No tuning. No live use."

    bundle["plain"] = {
        "what_happened": happened,
        "what_quantterm_will_do": will,
    }

    reg = prereg.record(hid, {
        "reproduction_pass": 1,
        "robust_confirmed": 1 if verdict == "ROBUST_CONFIRMED" else 0,
        "mean_net": pooled["mean_net"],
        "live_behaviour_changed": 0,
    })
    bundle["registry_status"] = reg.get("status")
    bundle["followup_verdict_recorded"] = verdict

    if verdict == "ROBUST_CONFIRMED":
        prereg.remember_watch(
            f"{EXPERIMENT_ID} {verdict}: parent EXP-FUND-03 remains CONFIRMED; "
            f"follow-up cleared robustness gates (pooled net={pooled['mean_net']})",
            signal="earnings_growth_followup",
            evidence_n=int(pooled["n"]),
            ev_r=float(pooled["mean_net"]),
            hypothesis_id=hid,
            notes="NOT production-authorized. Next=DESIGN_PAPER_SHADOW_POLICY_EXPERIMENT only.",
        )
    elif verdict in {"CONFIRMED_BUT_FRAGILE", "CONFIRMED_BUT_CONCENTRATED", "INCONCLUSIVE_FOLLOWUP"}:
        prereg.remember_watch(
            f"{EXPERIMENT_ID} {verdict}: parent CONFIRMED but follow-up limits advancement",
            signal="earnings_growth_followup",
            evidence_n=int(pooled["n"]),
            ev_r=float(pooled["mean_net"]),
            hypothesis_id=hid,
            notes="No tuning. No live use. " + "; ".join(reasons)[:300],
        )
    else:
        prereg.remember_negative(
            f"{EXPERIMENT_ID} {verdict}: follow-up blocked advancement",
            signal="earnings_growth_followup",
            evidence_n=int(pooled["n"]),
            notes="; ".join(reasons)[:400],
        )

    _write_report(bundle)
    slim = _jsonable(bundle)
    (OUT_DIR / "followup_result.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8"
    )
    return bundle


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "reproduction_pass": result["reproduction"]["pass"],
        "final_verdict": result["final_verdict"],
        "next_action": result["next_action"],
        "status_card": result.get("status_card"),
        "report": str(REPORT_PATH),
    }, indent=2))
