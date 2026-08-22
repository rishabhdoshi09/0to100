"""EDGE-003 study — all T1 qualifiers, EW. Research only."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.edge001.calendar import (
    cost_fraction,
    every_n_sessions,
    every_other,
    holding_window,
    month_ends,
    quarter_ends,
)
from research.edge001.study import (
    OpenCache,
    _mean_open_return,
    _sessions_from_frames,
    ew_universe_periods,
    live_on_session,
    nifty_period_return,
    simulate_spec,
)
from research.edge003.constants import (
    CONF_END,
    DSR_N_TRIALS,
    LOG_DIR,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    PRIMARY_SIGNAL,
    PROTOCOL_ACTIVATED_IST,
    SLOPE_LOOKBACK,
    SMA_WINDOW,
    protocol_sha,
)
from research.edge003.trend import dist_above_sma, trend_flag
from research.sepa.universe_pit import FastInvestable, _asof_ns, load_store_frames
from research.sepa003.regime import build_index_level, classify_regime_level, regime_at
from research.sepa003.sector import load_sector_map_v1

log = logging.getLogger(__name__)


def ew_one_way_turnover(prev: list[str], picks: list[str]) -> float:
    """One-way turnover for a variable-size equal-weight book."""
    if not prev:
        return 1.0
    if not picks:
        return 1.0
    old_w = 1.0 / len(prev)
    new_w = 1.0 / len(picks)
    prev_set, pick_set = set(prev), set(picks)
    drift = 0.0
    for s in prev_set | pick_set:
        w0 = old_w if s in prev_set else 0.0
        w1 = new_w if s in pick_set else 0.0
        drift += abs(w1 - w0)
    return 0.5 * drift


def rank_one_date(fast: FastInvestable, t: date, sector_of: dict[str, str]) -> dict[str, Any] | None:
    snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
    investable = [s for s in snap.investable if live_on_session(fast, s, t)]
    rows = []
    for sym in investable:
        i = fast._pos.get(sym)
        if i is None:
            continue
        j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
        close = fast._close[i]
        t1 = trend_flag(close, j, SMA_WINDOW, SLOPE_LOOKBACK, True)
        t2 = trend_flag(close, j, SMA_WINDOW, SLOPE_LOOKBACK, False)
        t3 = trend_flag(close, j, 150, SLOPE_LOOKBACK, True)
        if t1 is None:
            continue
        dist = dist_above_sma(close, j, SMA_WINDOW)
        rows.append({
            "as_of": t.isoformat(),
            "symbol": sym,
            "sector": sector_of.get(sym, "UNKNOWN"),
            "t1": bool(t1),
            "t2": bool(t2) if t2 is not None else False,
            "t3": bool(t3) if t3 is not None else False,
            "score_T1": 1.0 if t1 else np.nan,
            "score_T2": 1.0 if t2 else np.nan,
            "score_T3": 1.0 if t3 else np.nan,
            "score_DIST": (dist if t1 and dist is not None else np.nan),
            "adv20": float(fast._turn[i][j]) if j >= 0 and np.isfinite(fast._turn[i][j]) else 0.0,
        })
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    return {
        "as_of": t.isoformat(),
        "candidate_count": len(snap.candidates),
        "investable_count": len(snap.investable),
        "ranked_count": int(len(df)),
        "n_t1": int(df["t1"].sum()),
        "n_t2": int(df["t2"].sum()),
        "n_t3": int(df["t3"].sum()),
        "t1_share": float(df["t1"].mean()),
        "t2_share": float(df["t2"].mean()),
        "t3_share": float(df["t3"].mean()),
        "stale_dropped": int(len(snap.investable) - len(investable)),
        "exclusions": {**dict(snap.exclusions), "stale_last_print": int(len(snap.investable) - len(investable))},
        "membership_hash": snap.membership_hash,
        "investable_hash": snap.investable_hash,
        "data_quality": "PIT_DEGRADED_LISTING_SECTOR",
        "rows": df,
    }


def simulate_all(
    cache,
    rank_by_date,
    dates,
    sessions,
    sidx,
    flag_col,
    label,
    rt_pct,
    sector_of,
) -> dict[str, Any]:
    periods, holdings, txns, prev = [], [], [], []
    for i, t in enumerate(dates):
        df = rank_by_date.get(t)
        if df is None or flag_col not in df.columns:
            continue
        sub = df.loc[df[flag_col] == True]  # noqa: E712
        picks = sub["symbol"].tolist()
        if len(picks) < 5:
            continue
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        added = [s for s in picks if s not in prev]
        removed = [s for s in prev if s not in picks]
        ow = ew_one_way_turnover(prev, picks)
        cost = cost_fraction(ow, rt_pct)
        gross, n_filled = _mean_open_return(cache, picks, entry, exit_d)
        if gross is None:
            continue
        sectors = [sector_of.get(s, "UNKNOWN") for s in picks]
        sec = pd.Series(sectors).value_counts(normalize=True)
        advs = [float(a) for a in sub["adv20"].tolist()] if "adv20" in sub.columns else []
        periods.append({
            "rebalance": t.isoformat(),
            "entry_session": entry.isoformat(),
            "exit_session": exit_d.isoformat(),
            "n_picks": len(picks),
            "n_filled": n_filled,
            "n_univ": int(len(df)),
            "qualifier_share": float(len(picks) / max(len(df), 1)),
            "gross": gross,
            "net": gross - cost,
            "one_way_turnover": ow,
            "cost": cost,
            "n_added": len(added),
            "n_removed": len(removed),
            "n_retained": len(picks) - len(added),
            "max_sector_weight": float(sec.max()) if len(sec) else 0.0,
            "top_sector": str(sec.idxmax()) if len(sec) else "",
            "median_adv20": float(np.median(advs)) if advs else 0.0,
            "min_adv20": float(min(advs)) if advs else 0.0,
            "picks": picks[:8],
            "advs": [float(np.median(advs))] if advs else [],
        })
        holdings.append({
            "rebalance": t.isoformat(),
            "spec": label,
            "n": len(picks),
            "n_univ": int(len(df)),
            "qualifier_share": float(len(picks) / max(len(df), 1)),
        })
        prev = picks
    return {"periods": periods, "holdings": holdings, "txns": txns, "label": label}


def run_study(*, artifacts=None) -> dict[str, Any]:
    artifacts = artifacts or LOG_DIR
    artifacts.mkdir(parents=True, exist_ok=True)
    frames = load_store_frames(min_bars=40)
    fast = FastInvestable(frames)
    cache = OpenCache(fast)
    sessions = _sessions_from_frames(frames)
    sidx = {d: i for i, d in enumerate(sessions)}
    sector_of = {str(k).upper(): str(v) for k, v in (load_sector_map_v1().get("map") or {}).items()}
    rt_pct = float(round_trip_cost_pct("CNC"))
    nifty_level, nifty_src = build_index_level(frames)
    official_level = None
    try:
        from data.index_store import get_index_ohlcv
        odf = get_index_ohlcv("^NSEI")
        if odf is not None and len(odf) >= 60:
            col = next((c for c in odf.columns if str(c).lower() == "close"), None)
            if col is not None:
                official_level = pd.to_numeric(odf[col], errors="coerce").dropna()
    except Exception:
        official_level = None
    regime_tbl = classify_regime_level(nifty_level) if nifty_level is not None and len(nifty_level) else None

    ends = [
        t for t in month_ends(sessions)
        if t <= date.fromisoformat(CONF_END) and sidx.get(t, -1) >= SMA_WINDOW + SLOPE_LOOKBACK
    ]
    log.info("EDGE-003 month-ends %s (%s → %s)", len(ends), ends[0] if ends else None, ends[-1] if ends else None)

    snapshots, rank_by_date = [], {}
    for t in ends:
        pack = rank_one_date(fast, t, sector_of)
        if pack is None:
            continue
        df = pack.pop("rows")
        rank_by_date[t] = df
        snapshots.append(pack)
        log.info("T1 %s n=%s share=%.2f", t, pack["n_t1"], pack["t1_share"])

    dates = sorted(rank_by_date)
    (artifacts / "universe_snapshots.json").write_text(json.dumps(snapshots, indent=2, default=str))

    start = dates[0] if dates else sessions[0]
    extra = [
        d for d in every_n_sessions(sessions, 20, start)
        if d not in rank_by_date and sidx.get(d, -1) >= SMA_WINDOW + SLOPE_LOOKBACK
        and d <= date.fromisoformat(CONF_END)
    ]
    for t in extra:
        pack = rank_one_date(fast, t, sector_of)
        if pack is not None:
            rank_by_date[t] = pack["rows"]
    four_w = [d for d in every_n_sessions(sessions, 20, start) if d in rank_by_date]

    results, holdings_all = {}, []
    for name, flag, dset in (
        ("primary_T1_all_monthly", "t1", dates),
        ("sens_T2_all_monthly", "t2", dates),
        ("sens_T3_all_monthly", "t3", dates),
        ("sens_T1_all_4week", "t1", four_w),
        ("sens_T1_all_2month", "t1", every_other(dates)),
        ("sens_T1_all_quarterly", "t1", quarter_ends(dates)),
    ):
        sim = simulate_all(cache, rank_by_date, dset, sessions, sidx, flag, name, rt_pct, sector_of)
        results[name] = sim["periods"]
        holdings_all.extend(sim["holdings"])

    top20 = simulate_spec(
        cache, rank_by_date, dates, sessions, sidx, "score_DIST", 20, "sens_T1_top20_dist", rt_pct, sector_of,
    )
    results["sens_T1_top20_dist"] = top20["periods"]

    if holdings_all:
        pd.DataFrame(holdings_all).to_csv(artifacts / "holdings_ledger.csv", index=False)
    compact = {}
    for k, periods in results.items():
        if k == "primary_T1_all_monthly":
            compact[k] = periods
        else:
            compact[k] = [{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")} for p in periods]
    (artifacts / "portfolio_periods.json").write_text(json.dumps(compact, indent=2))

    ew = ew_universe_periods(cache, rank_by_date, dates, sessions, sidx)
    (artifacts / "ew_universe.json").write_text(json.dumps(ew, indent=2))

    buckets = []
    for i, t in enumerate(dates):
        df = rank_by_date[t]
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        for flag, lab in ((True, "T1"), (False, "exT1")):
            names = df.loc[df["t1"] == flag, "symbol"].tolist()
            gross, n = _mean_open_return(cache, names, entry, exit_d)
            if gross is not None:
                buckets.append({
                    "rebalance": t.isoformat(),
                    "bucket": lab,
                    "n": n,
                    "mean": gross,
                    "share": float(len(names) / max(len(df), 1)),
                })
    (artifacts / "inclusion_buckets.json").write_text(json.dumps(buckets, indent=2))

    prim = results.get("primary_T1_all_monthly", [])
    bench = []
    for p in prim:
        entry = date.fromisoformat(p["entry_session"])
        exit_d = date.fromisoformat(p["exit_session"])
        br = nifty_period_return(nifty_level, entry, exit_d)
        off = nifty_period_return(official_level, entry, exit_d) if official_level is not None else None
        ew_row = next((x for x in ew if x["rebalance"] == p["rebalance"]), None)
        ewg = None if ew_row is None else ew_row["gross"]
        snap = next((s for s in snapshots if s["as_of"] == p["rebalance"]), {})
        bench.append({
            **{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
            "nifty": br,
            "official_nifty": off,
            "ew_universe": ewg,
            "excess_net_vs_nifty": (p["net"] - br) if br is not None else None,
            "excess_net_vs_ew": (p["net"] - ewg) if ewg is not None else None,
            "t1_share": snap.get("t1_share", p.get("qualifier_share")),
            "nifty_source": nifty_src,
        })
    (artifacts / "benchmark_comparison.json").write_text(json.dumps(bench, indent=2))

    regime_rows = []
    if regime_tbl is not None:
        for p in prim:
            regime_rows.append({
                **{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
                **regime_at(regime_tbl, p["rebalance"]),
            })
    (artifacts / "regime_periods.json").write_text(json.dumps(regime_rows, indent=2))

    manifest = {
        "experiment": "EDGE-003",
        "protocol_sha": protocol_sha(),
        "protocol_activated_ist": PROTOCOL_ACTIVATED_IST,
        "primary": {"signal": PRIMARY_SIGNAL, "book": "all_qualifiers", "rebalance": "monthly_last_session"},
        "rt_cost_pct": rt_pct,
        "n_month_ends_ranked": len(dates),
        "first_rank": dates[0].isoformat() if dates else None,
        "last_rank": dates[-1].isoformat() if dates else None,
        "n_primary_periods": len(prim),
        "avg_t1_share": float(np.mean([s.get("t1_share") or 0 for s in snapshots])) if snapshots else None,
        "avg_n_t1": float(np.mean([s.get("n_t1") or 0 for s in snapshots])) if snapshots else None,
        "nifty_source": nifty_src,
        "official_nifty_sessions": int(len(official_level)) if official_level is not None else 0,
        "dsr_n_trials": DSR_N_TRIALS,
        "feature002_untouched": True,
        "production_buy_untouched": True,
        "fill": "next_open",
        "stop": "none_scheduled_rebalance_only",
        "listing_pit": "PIT_DEGRADED_bhav_inferred_same_session_print",
        "sector_pit": "PIT_DEGRADED_contemporaneous_map",
        "parent_consumed": "FEATURE-001 Trend on scanner fires through 2026-07-23",
        "store_sessions": len(sessions),
        "store_first": sessions[0].isoformat() if sessions else None,
        "store_last": sessions[-1].isoformat() if sessions else None,
        "n_frames": len(frames),
    }
    (artifacts / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "n_primary": len(prim)}
