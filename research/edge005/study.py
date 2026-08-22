"""EDGE-005 study — 252d near-high Top20 monthly. Research only."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.edge001.calendar import every_n_sessions, every_other, month_ends, quarter_ends
from research.edge001.study import (
    OpenCache,
    _sessions_from_frames,
    bucket_returns,
    ew_universe_periods,
    live_on_session,
    nifty_period_return,
    simulate_spec,
)
from research.edge005.constants import (
    CONF_END,
    DSR_N_TRIALS,
    LOG_DIR,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    P1_LOOKBACK,
    P2_LOOKBACK,
    P3_LOOKBACK,
    PRIMARY_N,
    PRIMARY_RANKER,
    PROTOCOL_ACTIVATED_IST,
    protocol_sha,
)
from research.edge005.proximity import proximity_to_high
from research.sepa.universe_pit import FastInvestable, _asof_ns, load_store_frames
from research.sepa003.regime import build_index_level, classify_regime_level, regime_at
from research.sepa003.sector import load_sector_map_v1

log = logging.getLogger(__name__)

SPECS = {
    "primary_P1_top20_monthly": ("score_P1", PRIMARY_N),
    "sens_P1_top10_monthly": ("score_P1", 10),
    "sens_P1_top30_monthly": ("score_P1", 30),
    "sens_P2_top20_monthly": ("score_P2", PRIMARY_N),
    "sens_P3_top20_monthly": ("score_P3", PRIMARY_N),
    "diag_LAG_top20_monthly": ("score_LAG", PRIMARY_N),
}


def _px(fast: FastInvestable, symbol: str, t: date) -> dict[str, float | None]:
    i = fast._pos.get(symbol)
    if i is None:
        return {k: None for k in ("p1", "p2", "p3")}
    j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
    close = fast._close[i]
    return {
        "p1": proximity_to_high(close, j, P1_LOOKBACK),
        "p2": proximity_to_high(close, j, P2_LOOKBACK),
        "p3": proximity_to_high(close, j, P3_LOOKBACK),
    }


def _assign_ranks(df: pd.DataFrame, score_col: str, key: str) -> None:
    valid = df[score_col].notna()
    df[f"pct_{key}"] = np.nan
    df[f"decile_{key}"] = np.nan
    if int(valid.sum()) < 50:
        return
    df.loc[valid, f"pct_{key}"] = df.loc[valid, score_col].rank(pct=True, method="average")
    df.loc[valid, f"decile_{key}"] = np.ceil(df.loc[valid, f"pct_{key}"] * 10).clip(1, 10)


def rank_one_date(fast: FastInvestable, t: date, sector_of: dict[str, str]) -> dict[str, Any] | None:
    snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
    investable = [s for s in snap.investable if live_on_session(fast, s, t)]
    rows = []
    for sym in investable:
        vs = _px(fast, sym, t)
        if vs["p1"] is None:
            continue
        i = fast._pos[sym]
        j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
        p1 = float(vs["p1"])
        rows.append({
            "as_of": t.isoformat(),
            "symbol": sym,
            "sector": sector_of.get(sym, "UNKNOWN"),
            "p1": p1,
            "p2": vs["p2"],
            "p3": vs["p3"],
            "score_P1": p1,
            "score_P2": (np.nan if vs["p2"] is None else float(vs["p2"])),
            "score_P3": (np.nan if vs["p3"] is None else float(vs["p3"])),
            "score_LAG": -p1,
            "adv20": float(fast._turn[i][j]) if j >= 0 and np.isfinite(fast._turn[i][j]) else 0.0,
        })
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    _assign_ranks(df, "score_P1", "P1")
    _assign_ranks(df, "score_P2", "P2")
    _assign_ranks(df, "score_P3", "P3")
    return {
        "as_of": t.isoformat(),
        "candidate_count": len(snap.candidates),
        "investable_count": len(snap.investable),
        "ranked_count": int(len(df)),
        "stale_dropped": int(len(snap.investable) - len(investable)),
        "exclusions": {**dict(snap.exclusions), "stale_last_print": int(len(snap.investable) - len(investable))},
        "membership_hash": snap.membership_hash,
        "investable_hash": snap.investable_hash,
        "data_quality": "PIT_DEGRADED_LISTING_SECTOR",
        "rows": df,
    }


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

    ends = [t for t in month_ends(sessions) if t <= date.fromisoformat(CONF_END) and sidx.get(t, -1) >= P1_LOOKBACK]
    log.info("EDGE-005 month-ends %s (%s → %s)", len(ends), ends[0] if ends else None, ends[-1] if ends else None)

    snapshots, rank_by_date = [], {}
    for t in ends:
        pack = rank_one_date(fast, t, sector_of)
        if pack is None:
            continue
        df = pack.pop("rows")
        rank_by_date[t] = df
        snapshots.append(pack)
        log.info("ranked %s n=%s", t, len(df))

    dates = sorted(rank_by_date)
    (artifacts / "universe_snapshots.json").write_text(json.dumps(snapshots, indent=2, default=str))

    start = dates[0] if dates else sessions[0]
    extra = [d for d in every_n_sessions(sessions, 20, start)
             if d not in rank_by_date and sidx.get(d, -1) >= P1_LOOKBACK and d <= date.fromisoformat(CONF_END)]
    for t in extra:
        pack = rank_one_date(fast, t, sector_of)
        if pack is not None:
            rank_by_date[t] = pack["rows"]
    four_w = [d for d in every_n_sessions(sessions, 20, start) if d in rank_by_date]

    results = {}
    for name, (col, n) in SPECS.items():
        results[name] = simulate_spec(cache, rank_by_date, dates, sessions, sidx, col, n, name, rt_pct, sector_of)["periods"]
    for name, dset in (
        ("sens_P1_top20_4week", four_w),
        ("sens_P1_top20_2month", every_other(dates)),
        ("sens_P1_top20_quarterly", quarter_ends(dates)),
    ):
        results[name] = simulate_spec(cache, rank_by_date, dset, sessions, sidx, "score_P1", PRIMARY_N, name, rt_pct, sector_of)["periods"]

    compact = {}
    for k, periods in results.items():
        if k == "primary_P1_top20_monthly":
            compact[k] = periods
        else:
            compact[k] = [{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")} for p in periods]
    (artifacts / "portfolio_periods.json").write_text(json.dumps(compact, indent=2))

    ew = ew_universe_periods(cache, rank_by_date, dates, sessions, sidx)
    (artifacts / "ew_universe.json").write_text(json.dumps(ew, indent=2))
    dec = {
        "P1_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_P1", range(1, 11)),
        "P2_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_P2", range(1, 11)),
        "P3_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_P3", range(1, 11)),
    }
    (artifacts / "decile_returns.json").write_text(json.dumps(dec, indent=2))

    prim = results.get("primary_P1_top20_monthly", [])
    bench = []
    for p in prim:
        entry = date.fromisoformat(p["entry_session"])
        exit_d = date.fromisoformat(p["exit_session"])
        br = nifty_period_return(nifty_level, entry, exit_d)
        off = nifty_period_return(official_level, entry, exit_d) if official_level is not None else None
        ew_row = next((x for x in ew if x["rebalance"] == p["rebalance"]), None)
        ewg = None if ew_row is None else ew_row["gross"]
        bench.append({
            **{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
            "nifty": br, "official_nifty": off, "ew_universe": ewg,
            "excess_net_vs_nifty": (p["net"] - br) if br is not None else None,
            "excess_net_vs_ew": (p["net"] - ewg) if ewg is not None else None,
            "nifty_source": nifty_src,
        })
    (artifacts / "benchmark_comparison.json").write_text(json.dumps(bench, indent=2))

    regime_rows = []
    if regime_tbl is not None:
        for p in prim:
            regime_rows.append({**{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
                                **regime_at(regime_tbl, p["rebalance"])})
    (artifacts / "regime_periods.json").write_text(json.dumps(regime_rows, indent=2))

    manifest = {
        "experiment": "EDGE-005",
        "protocol_sha": protocol_sha(),
        "protocol_activated_ist": PROTOCOL_ACTIVATED_IST,
        "primary": {"ranker": PRIMARY_RANKER, "n": PRIMARY_N, "rebalance": "monthly_last_session"},
        "rt_cost_pct": rt_pct,
        "n_month_ends_ranked": len(dates),
        "first_rank": dates[0].isoformat() if dates else None,
        "last_rank": dates[-1].isoformat() if dates else None,
        "n_primary_periods": len(prim),
        "nifty_source": nifty_src,
        "official_nifty_sessions": int(len(official_level)) if official_level is not None else 0,
        "dsr_n_trials": DSR_N_TRIALS,
        "feature002_untouched": True,
        "production_buy_untouched": True,
        "fill": "next_open",
        "stop": "none_scheduled_rebalance_only",
        "listing_pit": "PIT_DEGRADED_bhav_inferred_same_session_print",
        "sector_pit": "PIT_DEGRADED_contemporaneous_map",
        "parent_consumed": "scanner_laggard_demote_and_FEATURE-001",
        "store_sessions": len(sessions),
        "store_first": sessions[0].isoformat() if sessions else None,
        "store_last": sessions[-1].isoformat() if sessions else None,
        "n_frames": len(frames),
    }
    (artifacts / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "n_primary": len(prim)}
