"""Fundamentals + events research cycle — shared data & eval (no ML)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from research.intelligence.data.pit_contract import (
    DOMAIN_EVENTS,
    DOMAIN_FUNDAMENTALS,
    DOMAIN_VALUATIONS,
    PitContract,
)
from research.intelligence.data.snapshot_store import SnapshotStore
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_next import eval_utils as E

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "logs" / "research_expansion" / "pit_foundation" / "packages" / "46ff79f58ee21c9e"
SNAP_ROOT = REPO_ROOT / "logs" / "research_expansion" / "snapshots"
FROZEN = REPO_ROOT / "docs" / "overhaul" / "EXP_FUND_CYCLE_FROZEN_PROTOCOLS.json"

FOUNDATION_ID = "46ff79f58ee21c9e"
OHLCV_ID = "2f683be0c73eaa33"
N_TRIALS = 4
COST_PRODUCT = "CNC"
TURNOVER = 1.0
HOLD = 21
Q = 0.2
REB_CS = 5
PE_CAP = 200.0

DISCOVERY_START = "2023-01-01"
DISCOVERY_END = "2024-06-30"
CONFIRM_START = "2024-07-01"
CONFIRM_END = "2025-03-18"


@dataclass
class FundPanel:
    closes: pd.DataFrame
    pit: PitContract
    foundation_id: str
    ohlcv_id: str
    events: pd.DataFrame
    fundamentals: pd.DataFrame
    valuations: pd.DataFrame
    frozen: dict


def load_frozen() -> dict:
    return json.loads(FROZEN.read_text())


def cost_pct() -> float:
    return float(round_trip_cost_pct(COST_PRODUCT))


def cost_drag() -> float:
    return M.cost_drag(TURNOVER, cost_pct())


def load_panel() -> FundPanel:
    frozen = load_frozen()
    assert frozen["frozen_before_outcome_inspection"] is True
    assert frozen["foundation_package_id"] == FOUNDATION_ID

    store = SnapshotStore(SNAP_ROOT)
    ok, fails = store.verify_snapshot(OHLCV_ID)
    if not ok:
        raise ValueError(f"OHLCV snapshot verify failed: {fails}")
    snap = store.open_snapshot(OHLCV_ID)
    man = dict(snap.manifest)
    if man.get("scoped_certification") not in {
        "SCOPED_RESEARCH_READY", "READY_FOR_SCIENTIFIC_RERUN"
    }:
        raise ValueError(f"unexpected ohlcv cert: {man.get('scoped_certification')}")

    by_sym: dict[str, dict[str, float]] = {}
    for r in snap._equity:
        by_sym.setdefault(str(r["symbol"]).upper(), {})[r["date"]] = float(r["close"])
    closes = pd.DataFrame({s: pd.Series(v) for s, v in by_sym.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()

    ev_path = PKG / "pit_events.json"
    fu_path = PKG / "pit_fundamentals.json"
    va_path = PKG / "pit_valuations.json"
    for p in (ev_path, fu_path, va_path):
        if not p.exists():
            raise FileNotFoundError(p)

    pit = PitContract.from_store(
        store, OHLCV_ID,
        events_path=ev_path,
        fundamentals_path=fu_path,
        valuations_path=va_path,
    )
    # PIT smoke
    mid = "2024-01-15"
    fr = pit.as_of(DOMAIN_FUNDAMENTALS, when=mid, symbol="RELIANCE")
    er = pit.as_of(DOMAIN_EVENTS, when=mid, symbol="RELIANCE")
    vr = pit.as_of(DOMAIN_VALUATIONS, when=mid, symbol="RELIANCE")
    if fr.status == "NOT_PIT_SAFE":
        raise ValueError("fundamentals domain not bound")
    if er.status != "READY":
        raise ValueError(f"events domain not READY: {er.status}")

    events = pd.DataFrame(json.loads(ev_path.read_text())["rows"])
    fundamentals = pd.DataFrame(json.loads(fu_path.read_text())["rows"])
    valuations = pd.DataFrame(json.loads(va_path.read_text())["rows"])

    return FundPanel(
        closes=closes,
        pit=pit,
        foundation_id=FOUNDATION_ID,
        ohlcv_id=OHLCV_ID,
        events=events,
        fundamentals=fundamentals,
        valuations=valuations,
        frozen=frozen,
    )


def trading_days(closes: pd.DataFrame) -> pd.DatetimeIndex:
    return closes.index.sort_values()


def next_session(days: pd.DatetimeIndex, after: str) -> pd.Timestamp | None:
    """First session strictly after available_at date (conservative entry)."""
    ts = pd.Timestamp(str(after)[:10])
    later = days[days > ts]
    if len(later) == 0:
        return None
    return later[0]


def period_mask(days: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    return days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def _yoy_eps_map(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Build YoY EPS growth keyed by symbol + available_at for quarterly rows."""
    df = fundamentals.copy()
    df["available_at"] = pd.to_datetime(df["available_at"])
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df[df["period"].astype(str).str.lower().eq("quarterly")].copy()
    df = df.dropna(subset=["basic_eps", "period_end", "available_at"])
    # Prefer consolidated when duplicate symbol+period_end
    df["_consol_score"] = df["consolidated"].astype(str).str.lower().str.startswith("consolid").astype(int)
    df = df.sort_values(["symbol", "period_end", "_consol_score", "available_at"])
    df = df.drop_duplicates(["symbol", "period_end"], keep="last")

    rows = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("period_end")
        eps = g.set_index("period_end")["basic_eps"]
        avail = g.set_index("period_end")["available_at"]
        for pe, e in eps.items():
            # prior year same quarter ≈ 365d earlier period_end match within 40d
            target = pe - pd.DateOffset(years=1)
            # find closest period_end within 40 days
            diffs = (eps.index.to_series() - target).abs()
            hit = diffs[diffs <= pd.Timedelta(days=40)]
            if hit.empty:
                continue
            prev_pe = hit.idxmin()
            prev = float(eps.loc[prev_pe])
            cur = float(e)
            denom = max(abs(prev), 0.01)
            growth = (cur - prev) / denom
            # available only when current row is public
            rows.append({
                "symbol": sym,
                "period_end": pe,
                "available_at": avail.loc[pe],
                "basic_eps": cur,
                "prev_eps": prev,
                "yoy_eps_growth": growth,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["available_at"] = pd.to_datetime(out["available_at"])
    return out.sort_values(["symbol", "available_at"])


def latest_asof_frame(
    panel_df: pd.DataFrame,
    *,
    value_col: str,
    asof_col: str,
    dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    """Point-in-time cross-section: for each date, latest value with asof_col <= date."""
    df = panel_df.dropna(subset=[value_col, asof_col, "symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df[asof_col] = pd.to_datetime(df[asof_col])
    df = df.sort_values([asof_col])
    # pivot time series per symbol of (asof -> value), then asof reindex
    out = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for sym, g in df.groupby("symbol"):
        if sym not in out.columns:
            continue
        s = g.drop_duplicates(asof_col, keep="last").set_index(asof_col)[value_col].astype(float)
        s = s.sort_index()
        # reindex to calendar with forward-fill of known values, then mask to dates
        aligned = s.reindex(s.index.union(dates)).sort_index().ffill()
        out[sym] = aligned.reindex(dates)
        # ensure no look-ahead: values only where asof <= date (ffill on union is OK if we start from asof index)
    return out


def pack_eval(gross: pd.Series) -> dict:
    net = gross - cost_drag()
    pack = E.pack_stream(net, n_trials=N_TRIALS)
    pack["mean_gross"] = round(float(gross.mean()) if len(gross) else 0.0, 6)
    pack["mean_net"] = round(float(net.mean()) if len(net) else 0.0, 6)
    pack["cost_drag"] = round(cost_drag(), 6)
    # override pack_stream cost_drag to our frozen drag
    verdict = E.map_discovery_verdict(pack["verdict"], mean_net=pack["mean_net"], fdr_ok=True)
    if pack["mean_net"] <= 0:
        verdict = "FAIL"
    elif pack["verdict"] == "PROMOTE" and pack["mean_net"] > 0:
        verdict = "PASS"
    elif pack["mean_net"] > 0 and pack["verdict"] in {"UNDERPOWERED", "INCONCLUSIVE"}:
        verdict = "INCONCLUSIVE"
    elif pack["verdict"] == "REJECT":
        verdict = "FAIL"
    return {"gross": gross, "net": net, "pack": pack, "verdict": verdict}


def long_short_from_scores(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    invert: bool = False,
    top_q: float = Q,
) -> tuple[pd.Series, pd.Series]:
    """Returns (long_short_gross, ew_benchmark_gross) on formation dates."""
    ls = []
    ew = []
    idx = []
    for dt in dates:
        if dt not in scores.index or dt not in fwd.index:
            continue
        s = scores.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 10:
            continue
        n = max(1, int(len(s) * top_q))
        if invert:
            long = s.nsmallest(n).index
            short = s.nlargest(n).index
        else:
            long = s.nlargest(n).index
            short = s.nsmallest(n).index
        ls.append(float(f.loc[long].mean() - f.loc[short].mean()))
        ew.append(float(f.mean()))
        idx.append(dt)
    return (
        pd.Series(ls, index=pd.Index(idx), dtype=float),
        pd.Series(ew, index=pd.Index(idx), dtype=float),
    )


def rebalance_dates(days: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    return days[::every]


def remember(final: str, experiment_id: str, hid: str, discovery: dict, signal: str) -> None:
    n = int(discovery["pack"]["n"])
    net = float(discovery["pack"]["mean_net"])
    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        prereg.remember_negative(
            f"{experiment_id} {final}: net={net}",
            signal=signal,
            evidence_n=n,
            notes="No ML rescue. No production use.",
        )
    elif final in {"INCONCLUSIVE", "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION", "CONFIRMED"}:
        prereg.remember_watch(
            f"{experiment_id} {final} net={net}",
            signal=signal,
            evidence_n=n,
            ev_r=net,
            hypothesis_id=hid,
            notes="Not production-authorized." if final == "CONFIRMED" else "No tuning.",
        )


def next_action_for(final: str) -> str:
    if final == "CONFIRMED":
        return "ELIGIBLE_FOR_FOLLOWUP_RESEARCH"
    if final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        return "WAIT_FOR_INDEPENDENT_EVIDENCE"
    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        return "REJECT_CLOSE_BRANCH"
    if final == "INCONCLUSIVE":
        return "HOLD_NO_TUNING"
    if final == "BLOCKED":
        return "UNBLOCK_DATA"
    return "HOLD_NO_TUNING"


def plain_sections(question: str, final: str) -> dict[str, str]:
    if final == "CONFIRMED":
        happened = "The effect appeared in discovery and held up in an untouched later period after costs."
        means = "The idea looks real on this certified history — not approved for live trading."
        will = "Eligible for follow-up research only. QuantTerm will not trade on it yet."
    elif final == "FAILED_CONFIRMATION":
        happened = "It looked helpful at first, but failed when checked on later untouched data."
        means = "The idea does not clear QuantTerm's confirmation bar."
        will = "Nothing. QuantTerm will not use this idea."
    elif final == "FAIL":
        happened = "After realistic costs, the evidence rejected the idea under the frozen rules."
        means = "The hypothesis failed on this dataset."
        will = "Nothing. QuantTerm will not use this idea."
    elif final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        happened = "The first out-of-sample window looked promising after costs, but independent confirmation is not yet strong enough."
        means = "Promising, not proven."
        will = "Wait for more independent evidence. No live use. No tuning."
    else:
        happened = "The evidence was mixed or too weak to decide cleanly under the frozen rules."
        means = "Still uncertain."
        will = "No tuning. No live use."
    return {
        "what_we_tested": question,
        "what_happened": happened,
        "what_it_means": means,
        "what_quantterm_will_do": will,
    }
