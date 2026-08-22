"""EDGE-001 study runner — research only. Does not touch FEATURE-002 or BUY."""
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
    one_way_turnover,
    quarter_ends,
)
from research.edge001.constants import (
    ADV_LOOKBACK,
    CONF_END,
    DSR_N_TRIALS,
    HORIZON_LOOKBACK,
    LOG_DIR,
    M1_LOOKBACK,
    MIN_PRICE,
    MIN_SESSIONS,
    MIN_TURNOVER,
    PRIMARY_N,
    PRIMARY_RANKER,
    PRIMARY_REBALANCE,
    PROTOCOL_ACTIVATED_IST,
    SKIP,
    protocol_sha,
)
from research.edge001.momentum import incl_momentum, skip_momentum
from research.sepa.config import SepaConfig
from research.sepa.universe_pit import FastInvestable, _asof_ns, load_store_frames
from research.sepa003.fastrs import FastRS
from research.sepa003.regime import build_index_level, classify_regime_level
from research.sepa003.sector import load_sector_map_v1

log = logging.getLogger(__name__)

SPECS_MONTHLY_SIZE = {
    "primary_M1_top20_monthly": ("M1", PRIMARY_N),
    "sens_M1_top10_monthly": ("M1", 10),
    "sens_M1_top30_monthly": ("M1", 30),
    "sens_M1_top50_monthly": ("M1", 50),
    "sens_M2_top20_monthly": ("M2", PRIMARY_N),
    "sens_M3_top20_monthly": ("M3", PRIMARY_N),
    "sens_M4_top20_monthly": ("M4", PRIMARY_N),
    "sens_H3_top20_monthly": ("H3", PRIMARY_N),
}


def _sessions_from_frames(frames: dict[str, pd.DataFrame]) -> list[date]:
    """Official bhav session files, not the union of every symbol's index.

    A renamed/delisted name can carry a stray date that is not an exchange
    session for the rest of the book. Using that date as T+1 drops fills.
    """
    try:
        from data.bhavcopy_store import _dates_on_disk
        days = list(_dates_on_disk() or [])
        if len(days) >= 200:
            return sorted(days)
    except Exception:
        pass
    found: set[date] = set()
    for df in frames.values():
        idx = pd.DatetimeIndex(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        found.update(d.date() for d in idx.normalize())
    return sorted(found)


def _bar_date(fast: FastInvestable, symbol: str, t: date) -> date | None:
    i = fast._pos.get(symbol)
    if i is None:
        return None
    j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
    if j < 0:
        return None
    return pd.Timestamp(int(fast._dates[i][j])).date()


def live_on_session(fast: FastInvestable, symbol: str, t: date) -> bool:
    """True only if the name has an official bar on T. Stale last prints do not count."""
    return _bar_date(fast, symbol, t) == t


class OpenCache:
    """Exact-date open lookup without rebuilding a DatetimeIndex per fill."""

    def __init__(self, fast: FastInvestable):
        self._ord: dict[str, np.ndarray] = {}
        self._px: dict[str, np.ndarray] = {}
        for sym in fast.symbols:
            df = fast.frame(sym)
            if df is None or df.empty or "open" not in df.columns:
                continue
            idx = pd.DatetimeIndex(df.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            self._ord[sym] = np.asarray([d.date().toordinal() for d in idx.normalize()], dtype=np.int32)
            self._px[sym] = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)

    def get(self, symbol: str, d: date) -> float | None:
        ords = self._ord.get(symbol)
        if ords is None or ords.size == 0:
            return None
        i = int(np.searchsorted(ords, d.toordinal(), side="left"))
        if i >= ords.size or int(ords[i]) != d.toordinal():
            return None
        px = float(self._px[symbol][i])
        return px if px == px and px > 0 else None


def _open_on(cache: OpenCache, symbol: str, d: date) -> float | None:
    return cache.get(symbol, d)


def _adv20(fast: FastInvestable, symbol: str, t: date) -> float:
    i = fast._pos.get(symbol)
    if i is None:
        return 0.0
    as_ns = _asof_ns(t)
    j = fast.loc_as_of(fast._dates[i], as_ns)
    if j < 0:
        return 0.0
    turn = fast._turn[i]
    v = float(turn[j]) if j < len(turn) else float("nan")
    return v if v == v else 0.0


def _close_on_fast(fast: FastInvestable, symbol: str, t: date) -> float | None:
    i = fast._pos.get(symbol)
    if i is None:
        return None
    j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
    if j < 0:
        return None
    px = float(fast._close[i][j])
    return px if px == px and px > 0 else None


def _scores_at(fast: FastInvestable, rs: FastRS, symbol: str, t: date) -> dict[str, float | None]:
    i = fast._pos.get(symbol)
    if i is None:
        return {k: None for k in ("M1", "M2", "M3", "M4", "H3")}
    j = fast.loc_as_of(fast._dates[i], _asof_ns(t))
    close = fast._close[i]
    return {
        "M1": skip_momentum(close, j, HORIZON_LOOKBACK["M1"], SKIP),
        "M2": skip_momentum(close, j, HORIZON_LOOKBACK["M2"], SKIP),
        "M4": skip_momentum(close, j, HORIZON_LOOKBACK["M4"], SKIP),
        "H3": incl_momentum(close, j, M1_LOOKBACK),
        "M3": None,  # filled from FastRS table
    }


def _assign_ranks(df: pd.DataFrame, col: str, key: str) -> None:
    valid = df[col].notna()
    df[f"pct_{key}"] = np.nan
    df[f"decile_{key}"] = np.nan
    df[f"quintile_{key}"] = np.nan
    if int(valid.sum()) < 50:
        return
    df.loc[valid, f"pct_{key}"] = df.loc[valid, col].rank(pct=True, method="average")
    df.loc[valid, f"decile_{key}"] = np.ceil(df.loc[valid, f"pct_{key}"] * 10).clip(1, 10)
    df.loc[valid, f"quintile_{key}"] = np.ceil(df.loc[valid, f"pct_{key}"] * 5).clip(1, 5)


def rank_one_date(
    fast: FastInvestable,
    rs: FastRS,
    t: date,
    sector_of: dict[str, str],
) -> dict[str, Any] | None:
    snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
    # FastInvestable.loc_as_of reuses a name's last print forever after
    # delist/rename. Require a same-session bar so MAGMA-2021 cannot rank in 2026.
    investable = [s for s in snap.investable if live_on_session(fast, s, t)]
    table = rs.table(t, investable) if investable else {"scores": {}}
    rows = []
    for sym in investable:
        sc = _scores_at(fast, rs, sym, t)
        sc["M3"] = table.get("scores", {}).get(sym)
        if sc["M1"] is None:
            continue
        rows.append({
            "as_of": t.isoformat(),
            "symbol": sym,
            "sector": sector_of.get(sym, "UNKNOWN"),
            "m1": sc["M1"],
            "m2": sc["M2"],
            "m3": sc["M3"],
            "m4": sc["M4"],
            "h3": sc["H3"],
            "adv20": _adv20(fast, sym, t),
            "close": _close_on_fast(fast, sym, t),
        })
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows)
    _assign_ranks(df, "m1", "M1")
    _assign_ranks(df, "m2", "M2")
    _assign_ranks(df, "m3", "M3")
    _assign_ranks(df, "m4", "M4")
    _assign_ranks(df, "h3", "H3")
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
        "universe_source": snap.source or "bhav_inferred",
        "rows": df,
    }


def _mean_open_return(
    cache: OpenCache,
    symbols: list[str],
    entry: date,
    exit_d: date,
) -> tuple[float | None, int]:
    rets = []
    for s in symbols:
        px0 = _open_on(cache, s, entry)
        px1 = _open_on(cache, s, exit_d)
        if px0 is None or px1 is None:
            continue
        rets.append(px1 / px0 - 1.0)
    if not rets:
        return None, 0
    return float(np.mean(rets)), len(rets)


def simulate_spec(
    cache: OpenCache,
    rank_by_date: dict[date, pd.DataFrame],
    dates: list[date],
    sessions: list[date],
    sidx: dict[date, int],
    score_col: str,
    n: int,
    label: str,
    rt_pct: float,
    sector_of: dict[str, str],
) -> dict[str, Any]:
    periods: list[dict] = []
    holdings: list[dict] = []
    txns: list[dict] = []
    prev: list[str] = []
    for i, t in enumerate(dates):
        df = rank_by_date.get(t)
        if df is None or score_col not in df.columns:
            continue
        ranked = df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        picks = ranked.head(int(n))["symbol"].tolist()
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        added = [s for s in picks if s not in prev]
        removed = [s for s in prev if s not in picks]
        retained = [s for s in picks if s in prev]
        ow = one_way_turnover(prev, picks, n)
        cost = cost_fraction(ow, rt_pct)
        gross, n_filled = _mean_open_return(cache, picks, entry, exit_d)
        if gross is None:
            continue
        sectors = [sector_of.get(s, "UNKNOWN") for s in picks]
        sec = pd.Series(sectors).value_counts(normalize=True)
        advs = []
        for s in picks:
            row = ranked.loc[ranked["symbol"] == s]
            advs.append(float(row["adv20"].iloc[0]) if not row.empty else 0.0)
        periods.append({
            "rebalance": t.isoformat(),
            "entry_session": entry.isoformat(),
            "exit_session": exit_d.isoformat(),
            "n_picks": len(picks),
            "n_filled": n_filled,
            "gross": gross,
            "net": gross - cost,
            "one_way_turnover": ow,
            "cost": cost,
            "n_added": len(added),
            "n_removed": len(removed),
            "n_retained": len(retained),
            "max_sector_weight": float(sec.max()) if len(sec) else 0.0,
            "top_sector": str(sec.idxmax()) if len(sec) else "",
            "median_adv20": float(np.median(advs)) if advs else 0.0,
            "min_adv20": float(min(advs)) if advs else 0.0,
            "picks": picks,
            "advs": advs,
        })
        for s in picks:
            holdings.append({
                "rebalance": t.isoformat(),
                "entry_session": entry.isoformat(),
                "symbol": s,
                "spec": label,
                "sector": sector_of.get(s, "UNKNOWN"),
            })
        for s in added:
            txns.append({"rebalance": t.isoformat(), "side": "BUY", "symbol": s, "spec": label})
        for s in removed:
            txns.append({"rebalance": t.isoformat(), "side": "SELL", "symbol": s, "spec": label})
        prev = picks
    return {"periods": periods, "holdings": holdings, "txns": txns, "label": label}


def bucket_returns(
    cache: OpenCache,
    rank_by_date: dict[date, pd.DataFrame],
    dates: list[date],
    sessions: list[date],
    sidx: dict[date, int],
    bucket_col: str,
    buckets: range,
) -> list[dict]:
    rows = []
    for i, t in enumerate(dates):
        df = rank_by_date.get(t)
        if df is None or bucket_col not in df.columns:
            continue
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        uni = []
        for s in df["symbol"]:
            px0 = _open_on(cache, s, entry)
            px1 = _open_on(cache, s, exit_d)
            if px0 and px1:
                uni.append(px1 / px0 - 1.0)
        uni_mean = float(np.mean(uni)) if uni else None
        for b in buckets:
            names = df.loc[df[bucket_col] == b, "symbol"].tolist()
            rets = []
            for s in names:
                px0 = _open_on(cache, s, entry)
                px1 = _open_on(cache, s, exit_d)
                if px0 and px1:
                    rets.append(px1 / px0 - 1.0)
            if not rets:
                continue
            rows.append({
                "rebalance": t.isoformat(),
                "bucket": int(b),
                "n": len(rets),
                "mean": float(np.mean(rets)),
                "median": float(np.median(rets)),
                "universe_mean": uni_mean,
                "excess_vs_universe": (float(np.mean(rets)) - uni_mean) if uni_mean is not None else None,
            })
    return rows


def ew_universe_periods(
    cache: OpenCache,
    rank_by_date: dict[date, pd.DataFrame],
    dates: list[date],
    sessions: list[date],
    sidx: dict[date, int],
) -> list[dict]:
    out = []
    for i, t in enumerate(dates):
        df = rank_by_date.get(t)
        if df is None:
            continue
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        gross, n = _mean_open_return(cache, df["symbol"].tolist(), entry, exit_d)
        if gross is None:
            continue
        out.append({
            "rebalance": t.isoformat(),
            "entry_session": entry.isoformat(),
            "exit_session": exit_d.isoformat(),
            "gross": gross,
            "n": n,
        })
    return out


def nifty_period_return(level: pd.Series, entry: date, exit_d: date) -> float | None:
    if level is None or level.empty:
        return None
    idx = pd.DatetimeIndex(level.index).tz_localize(None).normalize()
    s = pd.Series(pd.to_numeric(level, errors="coerce").to_numpy(), index=idx)
    e = s.loc[s.index <= pd.Timestamp(entry)]
    x = s.loc[s.index <= pd.Timestamp(exit_d)]
    if e.empty or x.empty:
        return None
    a, b = float(e.iloc[-1]), float(x.iloc[-1])
    if a <= 0:
        return None
    return b / a - 1.0


def prod_momentum_compare(
    fast: FastInvestable,
    cache: OpenCache,
    dates: list[date],
    sessions: list[date],
    sidx: dict[date, int],
    rt_pct: float,
) -> list[dict]:
    """Same monthly next-open hold; ranker = 5-session time-series return."""
    rows = []
    prev: list[str] = []
    for i, t in enumerate(dates):
        snap = fast.snapshot(t, min_price=MIN_PRICE, min_turnover=MIN_TURNOVER, min_sessions=MIN_SESSIONS)
        scored = []
        for s in snap.investable:
            if not live_on_session(fast, s, t):
                continue
            pos = fast._pos.get(s)
            if pos is None:
                continue
            j = fast.loc_as_of(fast._dates[pos], _asof_ns(t))
            r5 = incl_momentum(fast._close[pos], j, 5)
            if r5 is None:
                continue
            scored.append((s, r5))
        if len(scored) < 50:
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in scored[:PRIMARY_N]]
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        window = holding_window(sessions, t, nxt, sidx)
        if window is None:
            continue
        entry, exit_d = window
        ow = one_way_turnover(prev, picks, PRIMARY_N)
        gross, n_filled = _mean_open_return(cache, picks, entry, exit_d)
        if gross is None:
            continue
        rows.append({
            "rebalance": t.isoformat(),
            "gross": gross,
            "net": gross - cost_fraction(ow, rt_pct),
            "one_way_turnover": ow,
            "n_picks": len(picks),
            "n_filled": n_filled,
        })
        prev = picks
    return rows


def run_study(*, artifacts=None) -> dict[str, Any]:
    artifacts = artifacts or LOG_DIR
    artifacts.mkdir(parents=True, exist_ok=True)
    log.info("EDGE-001 loading store frames")
    frames = load_store_frames(min_bars=40)
    fast = FastInvestable(frames)
    cache = OpenCache(fast)
    rs = FastRS(fast, SepaConfig())
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

    ends = month_ends(sessions)
    ends = [t for t in ends if t <= date.fromisoformat(CONF_END)]
    ends = [t for t in ends if sidx.get(t, -1) >= M1_LOOKBACK]
    log.info("month-ends eligible: %s (%s → %s)", len(ends), ends[0] if ends else None, ends[-1] if ends else None)

    snapshots = []
    rank_by_date: dict[date, pd.DataFrame] = {}
    slim_parts = []
    for t in ends:
        pack = rank_one_date(fast, rs, t, sector_of)
        if pack is None:
            continue
        df = pack.pop("rows")
        rank_by_date[t] = df
        snapshots.append(pack)
        slim_parts.append(df[[
            "as_of", "symbol", "sector", "m1", "m2", "m3", "m4", "h3",
            "pct_M1", "decile_M1", "quintile_M1",
            "pct_M2", "decile_M2", "pct_M3", "decile_M3",
            "pct_M4", "decile_M4", "adv20",
        ]])
        log.info("ranked %s n=%s investable=%s", t, len(df), pack["investable_count"])

    dates = sorted(rank_by_date)
    if slim_parts:
        pd.concat(slim_parts, ignore_index=True).to_csv(artifacts / "monthly_ranks.csv", index=False)
    (artifacts / "universe_snapshots.json").write_text(json.dumps(snapshots, indent=2, default=str))

    start = dates[0] if dates else sessions[0]
    extra = [
        d for d in every_n_sessions(sessions, 20, start)
        if d not in rank_by_date and sidx.get(d, -1) >= M1_LOOKBACK and d <= date.fromisoformat(CONF_END)
    ]
    log.info("extra 4-week rank dates: %s", len(extra))
    for t in extra:
        pack = rank_one_date(fast, rs, t, sector_of)
        if pack is None:
            continue
        rank_by_date[t] = pack["rows"]

    four_w = [d for d in every_n_sessions(sessions, 20, start) if d in rank_by_date]
    two_m = every_other(dates)
    qtr = quarter_ends(dates)

    results: dict[str, list] = {}
    holdings_all: list[dict] = []
    txns_all: list[dict] = []
    for name, (key, n) in SPECS_MONTHLY_SIZE.items():
        col = {"M1": "m1", "M2": "m2", "M3": "m3", "M4": "m4", "H3": "h3"}[key]
        sim = simulate_spec(cache, rank_by_date, dates, sessions, sidx, col, n, name, rt_pct, sector_of)
        results[name] = sim["periods"]
        holdings_all.extend(sim["holdings"])
        txns_all.extend(sim["txns"])

    for name, dset in (
        ("sens_M1_top20_4week", four_w),
        ("sens_M1_top20_2month", two_m),
        ("sens_M1_top20_quarterly", qtr),
    ):
        sim = simulate_spec(cache, rank_by_date, dset, sessions, sidx, "m1", PRIMARY_N, name, rt_pct, sector_of)
        results[name] = sim["periods"]
        holdings_all.extend(sim["holdings"])
        txns_all.extend(sim["txns"])

    pd.DataFrame(holdings_all).to_csv(artifacts / "holdings_ledger.csv", index=False)
    pd.DataFrame(txns_all).to_csv(artifacts / "transaction_ledger.csv", index=False)

    # Compact period JSON without pick lists for the big dump; keep picks on primary only
    compact = {}
    for k, periods in results.items():
        if k == "primary_M1_top20_monthly":
            compact[k] = periods
        else:
            compact[k] = [{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")} for p in periods]
    (artifacts / "portfolio_periods.json").write_text(json.dumps(compact, indent=2))

    ew = ew_universe_periods(cache, rank_by_date, dates, sessions, sidx)
    (artifacts / "ew_universe.json").write_text(json.dumps(ew, indent=2))

    dec = {
        "M1_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_M1", range(1, 11)),
        "M1_quintile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "quintile_M1", range(1, 6)),
        "M2_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_M2", range(1, 11)),
        "M3_decile": bucket_returns(cache, rank_by_date, dates, sessions, sidx, "decile_M3", range(1, 11)),
    }
    (artifacts / "decile_returns.json").write_text(json.dumps(dec, indent=2))

    prim = results.get("primary_M1_top20_monthly", [])
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
            "nifty": br,
            "official_nifty": off,
            "ew_universe": ewg,
            "excess_net_vs_nifty": (p["net"] - br) if br is not None else None,
            "excess_net_vs_official_nifty": (p["net"] - off) if off is not None else None,
            "excess_net_vs_ew": (p["net"] - ewg) if ewg is not None else None,
            "excess_gross_vs_nifty": (p["gross"] - br) if br is not None else None,
            "nifty_source": nifty_src,
        })
    (artifacts / "benchmark_comparison.json").write_text(json.dumps(bench, indent=2))

    mom = prod_momentum_compare(fast, cache, dates, sessions, sidx, rt_pct)
    (artifacts / "prod_momentum_compare.json").write_text(json.dumps(mom, indent=2))

    regime_rows = []
    if regime_tbl is not None:
        from research.sepa003.regime import regime_at
        for p in prim:
            info = regime_at(regime_tbl, p["rebalance"])
            regime_rows.append({**{kk: vv for kk, vv in p.items() if kk not in ("picks", "advs")}, **info})
    (artifacts / "regime_periods.json").write_text(json.dumps(regime_rows, indent=2))

    nifty500 = None
    try:
        from data.index_store import get_index_ohlcv
        nifty500 = get_index_ohlcv("^NSE500") or get_index_ohlcv("Nifty 500")
    except Exception:
        nifty500 = None

    manifest = {
        "experiment": "EDGE-001",
        "protocol_sha": protocol_sha(),
        "protocol_activated_ist": PROTOCOL_ACTIVATED_IST,
        "primary": {"ranker": PRIMARY_RANKER, "n": PRIMARY_N, "rebalance": PRIMARY_REBALANCE},
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
        "official_nifty_start": (
            str(pd.Timestamp(official_level.index.min()).date())
            if official_level is not None and len(official_level) else None
        ),
        "nifty500_available": bool(nifty500 is not None and len(nifty500)),
        "adv_lookback": ADV_LOOKBACK,
        "feature002_untouched": True,
        "production_buy_untouched": True,
        "ca_policy": "adjustment_on_read_plus_min_sessions; no exhaustive unresolved quarantine in EDGE-001",
        "listing_pit": "PIT_DEGRADED_bhav_inferred",
        "sector_pit": "PIT_DEGRADED_contemporaneous_map",
        "fill": "next_open",
        "stop": "none_scheduled_rebalance_only",
    }
    (artifacts / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "artifacts": str(artifacts), "n_primary": len(prim)}
