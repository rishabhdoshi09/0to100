"""EDGE-006 study — highest 20d ADV Top20 monthly. Research only."""
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
from research.edge006.constants import (
    CONF_END,
    DSR_N_TRIALS,
    LOG_DIR,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    PRIMARY_N,
    PRIMARY_RANKER,
    PROTOCOL_ACTIVATED_IST,
    protocol_sha,
)
from research.sepa.universe_pit import FastInvestable, _asof_ns, load_store_frames
from research.sepa003.regime import build_index_level, classify_regime_level, regime_at
from research.sepa003.sector import load_sector_map_v1

log = logging.getLogger(__name__)


def rank_one_date(fast: FastInvestable, t: date, sector_of: dict[str, str]) -> dict[str, Any] | None:
    snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
    investable = [s for s in snap.investable if live_on_session(fast, s, t)]
    rows = []
    for sym in investable:
        i = fast._pos.get(sym)
        if i is None:
            continue
        j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
        if j < 19:
            continue
        adv = float(fast._turn[i][j]) if j >= 0 else float("nan")
        if not np.isfinite(adv) or adv <= 0:
            continue
        rows.append({
            "as_of": t.isoformat(),
            "symbol": sym,
            "sector": sector_of.get(sym, "UNKNOWN"),
            "adv20": adv,
            "score_L1": adv,
            "score_L0": -adv,
        })
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    valid = df["score_L1"].notna()
    df["pct_L1"] = np.nan
    df["decile_L1"] = np.nan
    df.loc[valid, "pct_L1"] = df.loc[valid, "score_L1"].rank(pct=True, method="average")
    df.loc[valid, "decile_L1"] = np.ceil(df.loc[valid, "pct_L1"] * 10).clip(1, 10)
    return {
        "as_of": t.isoformat(),
        "candidate_count": len(snap.candidates),
        "investable_count": len(snap.investable),
        "ranked_count": int(len(df)),
        "stale_dropped": int(len(snap.investable) - len(investable)),
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
    regime_tbl = classify_regime_level(nifty_level) if nifty_level is not None and len(nifty_level) else None

    ends = [t for t in month_ends(sessions) if t <= date.fromisoformat(CONF_END) and sidx.get(t, -1) >= 20]
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
             if d not in rank_by_date and sidx.get(d, -1) >= 20 and d <= date.fromisoformat(CONF_END)]
    for t in extra:
        pack = rank_one_date(fast, t, sector_of)
        if pack is not None:
            rank_by_date[t] = pack["rows"]
    four_w = [d for d in every_n_sessions(sessions, 20, start) if d in rank_by_date]

    results = {
        "primary_L1_top20_monthly": simulate_spec(cache, rank_by_date, dates, sessions, sidx, "score_L1", 20, "primary_L1_top20_monthly", rt_pct, sector_of)["periods"],
        "sens_L1_top50_monthly": simulate_spec(cache, rank_by_date, dates, sessions, sidx, "score_L1", 50, "sens_L1_top50_monthly", rt_pct, sector_of)["periods"],
        "diag_L0_top20_monthly": simulate_spec(cache, rank_by_date, dates, sessions, sidx, "score_L0", 20, "diag_L0_top20_monthly", rt_pct, sector_of)["periods"],
        "sens_L1_top20_4week": simulate_spec(cache, rank_by_date, four_w, sessions, sidx, "score_L1", 20, "sens_L1_top20_4week", rt_pct, sector_of)["periods"],
        "sens_L1_top20_2month": simulate_spec(cache, rank_by_date, every_other(dates), sessions, sidx, "score_L1", 20, "sens_L1_top20_2month", rt_pct, sector_of)["periods"],
        "sens_L1_top20_quarterly": simulate_spec(cache, rank_by_date, quarter_ends(dates), sessions, sidx, "score_L1", 20, "sens_L1_top20_quarterly", rt_pct, sector_of)["periods"],
    }
    compact = {k: (v if k == "primary_L1_top20_monthly" else [{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")} for p in v]) for k, v in results.items()}
    (artifacts / "portfolio_periods.json").write_text(json.dumps(compact, indent=2))
    ew = ew_universe_periods(cache, rank_by_date, dates, sessions, sidx)
    (artifacts / "ew_universe.json").write_text(json.dumps(ew, indent=2))
    (artifacts / "decile_returns.json").write_text(json.dumps({
        "L1_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_L1", range(1, 11)),
    }, indent=2))

    prim = results["primary_L1_top20_monthly"]
    bench = []
    for p in prim:
        entry = date.fromisoformat(p["entry_session"])
        exit_d = date.fromisoformat(p["exit_session"])
        br = nifty_period_return(nifty_level, entry, exit_d)
        ew_row = next((x for x in ew if x["rebalance"] == p["rebalance"]), None)
        ewg = None if ew_row is None else ew_row["gross"]
        bench.append({
            **{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
            "nifty": br, "ew_universe": ewg,
            "excess_net_vs_nifty": (p["net"] - br) if br is not None else None,
            "excess_net_vs_ew": (p["net"] - ewg) if ewg is not None else None,
        })
    (artifacts / "benchmark_comparison.json").write_text(json.dumps(bench, indent=2))
    regime_rows = []
    if regime_tbl is not None:
        for p in prim:
            regime_rows.append({**{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")},
                                **regime_at(regime_tbl, p["rebalance"])})
    (artifacts / "regime_periods.json").write_text(json.dumps(regime_rows, indent=2))
    manifest = {
        "experiment": "EDGE-006",
        "protocol_sha": protocol_sha(),
        "protocol_activated_ist": PROTOCOL_ACTIVATED_IST,
        "primary": {"ranker": PRIMARY_RANKER, "n": PRIMARY_N, "rebalance": "monthly_last_session"},
        "rt_cost_pct": rt_pct,
        "n_month_ends_ranked": len(dates),
        "first_rank": dates[0].isoformat() if dates else None,
        "last_rank": dates[-1].isoformat() if dates else None,
        "n_primary_periods": len(prim),
        "nifty_source": nifty_src,
        "dsr_n_trials": DSR_N_TRIALS,
        "feature002_untouched": True,
        "production_buy_untouched": True,
        "listing_pit": "PIT_DEGRADED_bhav_inferred_same_session_print",
        "sector_pit": "PIT_DEGRADED_contemporaneous_map",
        "store_sessions": len(sessions),
        "store_first": sessions[0].isoformat() if sessions else None,
        "store_last": sessions[-1].isoformat() if sessions else None,
        "last_budget_slot": True,
    }
    (artifacts / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "n_primary": len(prim)}
