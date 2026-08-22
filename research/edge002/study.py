"""EDGE-002 study runner — research only. Does not touch FEATURE-002 or BUY."""
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
from research.edge002.constants import (
    CONF_END,
    DSR_N_TRIALS,
    LOG_DIR,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    PRIMARY_N,
    PRIMARY_RANKER,
    PROTOCOL_ACTIVATED_IST,
    V0_LOOKBACK,
    V1_LOOKBACK,
    V2_LOOKBACK,
    V3_LOOKBACK,
    protocol_sha,
)
from research.edge002.vol import realized_vol
from research.sepa.universe_pit import FastInvestable, _asof_ns, load_store_frames
from research.sepa003.regime import build_index_level, classify_regime_level, regime_at
from research.sepa003.sector import load_sector_map_v1

log = logging.getLogger(__name__)

SPECS = {
    "primary_V1_top20_monthly": ("v1", PRIMARY_N),
    "sens_V1_top10_monthly": ("v1", 10),
    "sens_V1_top30_monthly": ("v1", 30),
    "sens_V1_top50_monthly": ("v1", 50),
    "sens_V2_top20_monthly": ("v2", PRIMARY_N),
    "sens_V3_top20_monthly": ("v3", PRIMARY_N),
    "diag_V0_top20_monthly": ("v0", PRIMARY_N),
}


def _vols(fast: FastInvestable, symbol: str, t: date) -> dict[str, float | None]:
    i = fast._pos.get(symbol)
    if i is None:
        return {k: None for k in ("v0", "v1", "v2", "v3")}
    j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
    close = fast._close[i]
    return {
        "v0": realized_vol(close, j, V0_LOOKBACK),
        "v1": realized_vol(close, j, V1_LOOKBACK),
        "v2": realized_vol(close, j, V2_LOOKBACK),
        "v3": realized_vol(close, j, V3_LOOKBACK),
    }


def _assign_ranks(df: pd.DataFrame, vol_col: str, key: str) -> None:
    """High score = low vol. Decile 10 = lowest volatility."""
    score = f"score_{key}"
    df[score] = np.where(df[vol_col].notna(), -df[vol_col], np.nan)
    valid = df[score].notna()
    df[f"pct_{key}"] = np.nan
    df[f"decile_{key}"] = np.nan
    if int(valid.sum()) < 50:
        return
    df.loc[valid, f"pct_{key}"] = df.loc[valid, score].rank(pct=True, method="average")
    df.loc[valid, f"decile_{key}"] = np.ceil(df.loc[valid, f"pct_{key}"] * 10).clip(1, 10)


def rank_one_date(fast: FastInvestable, t: date, sector_of: dict[str, str]) -> dict[str, Any] | None:
    snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
    investable = [s for s in snap.investable if live_on_session(fast, s, t)]
    rows = []
    for sym in investable:
        vs = _vols(fast, sym, t)
        if vs["v1"] is None:
            continue
        i = fast._pos[sym]
        j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
        rows.append({
            "as_of": t.isoformat(),
            "symbol": sym,
            "sector": sector_of.get(sym, "UNKNOWN"),
            "v0": vs["v0"],
            "v1": vs["v1"],
            "v2": vs["v2"],
            "v3": vs["v3"],
            "score_V1": -float(vs["v1"]),
            "adv20": float(fast._turn[i][j]) if j >= 0 and np.isfinite(fast._turn[i][j]) else 0.0,
        })
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    _assign_ranks(df, "v0", "V0")
    _assign_ranks(df, "v1", "V1")
    _assign_ranks(df, "v2", "V2")
    _assign_ranks(df, "v3", "V3")
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

    ends = [t for t in month_ends(sessions) if t <= date.fromisoformat(CONF_END) and sidx.get(t, -1) >= V3_LOOKBACK]
    log.info("EDGE-002 month-ends %s (%s → %s)", len(ends), ends[0] if ends else None, ends[-1] if ends else None)

    snapshots, rank_by_date, slim = [], {}, []
    for t in ends:
        pack = rank_one_date(fast, t, sector_of)
        if pack is None:
            continue
        df = pack.pop("rows")
        rank_by_date[t] = df
        snapshots.append(pack)
        slim.append(df[["as_of", "symbol", "sector", "v0", "v1", "v2", "v3",
                        "score_V1", "pct_V1", "decile_V1", "decile_V0", "decile_V2", "decile_V3", "adv20"]])
        log.info("ranked %s n=%s", t, len(df))

    dates = sorted(rank_by_date)
    if slim:
        pd.concat(slim, ignore_index=True).to_csv(artifacts / "monthly_ranks.csv", index=False)
    (artifacts / "universe_snapshots.json").write_text(json.dumps(snapshots, indent=2, default=str))

    start = dates[0] if dates else sessions[0]
    extra = [d for d in every_n_sessions(sessions, 20, start)
             if d not in rank_by_date and sidx.get(d, -1) >= V3_LOOKBACK and d <= date.fromisoformat(CONF_END)]
    for t in extra:
        pack = rank_one_date(fast, t, sector_of)
        if pack is None:
            continue
        rank_by_date[t] = pack["rows"]
    four_w = [d for d in every_n_sessions(sessions, 20, start) if d in rank_by_date]
    two_m = every_other(dates)
    qtr = quarter_ends(dates)

    results, holdings_all, txns_all = {}, [], []
    colmap = {"v0": "score_V0", "v1": "score_V1", "v2": "score_V2", "v3": "score_V3"}
    # scores from _assign_ranks
    for name, (key, n) in SPECS.items():
        col = colmap[key]
        # ensure score_* exists
        for d0, df in rank_by_date.items():
            if col not in df.columns and key.upper() in ("V0", "V1", "V2", "V3"):
                pass
        sim = simulate_spec(cache, rank_by_date, dates, sessions, sidx, col, n, name, rt_pct, sector_of)
        results[name] = sim["periods"]
        holdings_all.extend(sim["holdings"])
        txns_all.extend(sim["txns"])

    for name, dset in (
        ("sens_V1_top20_4week", four_w),
        ("sens_V1_top20_2month", two_m),
        ("sens_V1_top20_quarterly", qtr),
    ):
        sim = simulate_spec(cache, rank_by_date, dset, sessions, sidx, "score_V1", PRIMARY_N, name, rt_pct, sector_of)
        results[name] = sim["periods"]
        holdings_all.extend(sim["holdings"])
        txns_all.extend(sim["txns"])

    pd.DataFrame(holdings_all).to_csv(artifacts / "holdings_ledger.csv", index=False)
    pd.DataFrame(txns_all).to_csv(artifacts / "transaction_ledger.csv", index=False)
    compact = {}
    for k, periods in results.items():
        if k == "primary_V1_top20_monthly":
            compact[k] = periods
        else:
            compact[k] = [{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")} for p in periods]
    (artifacts / "portfolio_periods.json").write_text(json.dumps(compact, indent=2))

    ew = ew_universe_periods(cache, rank_by_date, dates, sessions, sidx)
    (artifacts / "ew_universe.json").write_text(json.dumps(ew, indent=2))
    dec = {
        "V1_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_V1", range(1, 11)),
        "V0_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_V0", range(1, 11)),
        "V2_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_V2", range(1, 11)),
        "V3_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_V3", range(1, 11)),
    }
    (artifacts / "decile_returns.json").write_text(json.dumps(dec, indent=2))

    prim = results.get("primary_V1_top20_monthly", [])
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
            info = regime_at(regime_tbl, p["rebalance"])
            regime_rows.append({**{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")}, **info})
    (artifacts / "regime_periods.json").write_text(json.dumps(regime_rows, indent=2))

    manifest = {
        "experiment": "EDGE-002",
        "protocol_sha": protocol_sha(),
        "protocol_activated_ist": PROTOCOL_ACTIVATED_IST,
        "primary": {"ranker": PRIMARY_RANKER, "n": PRIMARY_N, "rebalance": "monthly_last_session"},
        "rt_cost_pct": rt_pct,
        "n_month_ends_ranked": len(dates),
        "first_rank": dates[0].isoformat() if dates else None,
        "last_rank": dates[-1].isoformat() if dates else None,
        "store_sessions": len(sessions),
        "store_first": sessions[0].isoformat() if sessions else None,
        "store_last": sessions[-1].isoformat() if sessions else None,
        "n_frames": len(frames),
        "dsr_n_trials": DSR_N_TRIALS,
        "n_primary_periods": len(prim),
        "nifty_source": nifty_src,
        "official_nifty_sessions": int(len(official_level)) if official_level is not None else 0,
        "feature002_untouched": True,
        "production_buy_untouched": True,
        "parent_inconclusive": "EXP-NEXT-02",
        "fill": "next_open",
        "stop": "none_scheduled_rebalance_only",
        "listing_pit": "PIT_DEGRADED_bhav_inferred_same_session_print",
        "sector_pit": "PIT_DEGRADED_contemporaneous_map",
    }
    (artifacts / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "n_primary": len(prim)}
