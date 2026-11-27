"""Bounded, offline NSE selection comparison. Discovery history only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from core.costs import round_trip_cost_pct
from core.evaluation_spec import HORIZON_SESSIONS, simulate_trade
from gauntlet.validator import scoped_equity_frames
from research.overlap_inference import overlap_robust_test

# The existing 12–1 experiment reserves Nov 2025 onward.
START, END = date(2025, 1, 1), date(2025, 11, 1)
SLOTS, STOP, TARGET, BREAKEVEN_PCT = 5, 0.03, 0.05, 2.0
CLASSES = ("target", "stop", "expiry")
SEED = 21457


def _trade(frame, days, cost):
    """Fill at the next official open and mark the independent sleeve daily."""
    available = []
    for day in days:
        if day not in frame.index:
            break
        available.append(day)
    if not available:
        return None, [1.0] * len(days), "missing_entry"
    bars = frame.loc[available]
    entry = bars.open.iloc[0]
    target = entry * (1 + TARGET)
    sim = simulate_trade(entry, entry * (1 - STOP), target, bars.high.to_numpy(),
                         bars.low.to_numpy(), bars.close.to_numpy(), opens=bars.open.to_numpy(),
                         be_pct=BREAKEVEN_PCT)
    marks = [float(c / entry - cost) for c in bars.close.iloc[:sim.exit_offset]]
    if len(available) < len(days) and sim.outcome == "FLAT":
        # An unresolved holding is marked down to zero at the first missing bar.
        marks.append(float(bars.close.iloc[-1] / entry - cost))
        return None, marks + [0.0] * (len(days) - len(marks)), "censored_after_entry"
    net = float(sim.exit_price / entry - 1 - cost)
    marks.extend([1 + net] * (len(days) - len(marks)))
    cause = {"WIN": "target", "LOSS": "stop", "SCRATCH": "stop", "FLAT": "expiry"}[sim.outcome]
    return net, marks, cause


def _rank(rows, field, allowed=lambda row: True):
    return sorted((s for s, row in rows.items() if allowed(row)), key=lambda s: (-rows[s][field], s))


def _rules(rows, market_above_average, market_return):
    scanner = _rank(rows, "score", lambda r: r["signals"])
    return {
        "scanner": scanner,
        "liquidity": _rank(rows, "turnover"),
        "relative_strength": _rank(rows, "r30", lambda r: r["r30"] > market_return and r["uptrend"]),
        "trend_pullback": _rank(rows, "r30", lambda r: "PULLBACK_SUPPORT" in r["signals"]),
        "mean_reversion": _rank(rows, "reversion", lambda r: r["uptrend"] and r["reversion"] >= .03),
        "volume_breakout": _rank(rows, "score", lambda r: r["signals"] & {"BREAKOUT_52W", "BREAKOUT_RES"}),
        "sector_leadership": _rank(rows, "r30", lambda r: r["sector_edge"] > 0 and r["uptrend"]),
        "cross_sectional_momentum": _rank(rows, "momentum"),
        "scanner_risk_on": scanner if market_above_average else [],
        "scanner_confluence": _rank(rows, "score", lambda r: len(r["signals"]) >= 2),
    }


def _features(frames, days, index, scanner, sectors, market_return):
    scanner._nifty_ret30 = market_return
    rows = {}
    day = days[index]
    ordinal = {d: i for i, d in enumerate(days)}
    for symbol, frame in frames.items():
        history = frame.loc[:day].tail(250)
        if len(history) < 60 or history.index[-1] != day:
            continue
        dates = [ordinal[d] for d in history.index]
        last_gap = max((i for i in range(1, len(dates)) if dates[i] != dates[i-1] + 1), default=0)
        history = history.iloc[last_gap:]
        if len(history) < 60:
            continue
        c = history.close.to_numpy()
        turnover = float((history.close * history.volume).tail(20).mean())
        if c[-1] < 20 or turnover < 10_000_000:
            continue
        signal = scanner._analyze(symbol, history)
        rows[symbol] = {
            "signals": set(signal.signals) if signal else set(), "score": signal.score if signal else 0,
            "r30": c[-1] / c[-31] - 1, "reversion": 1 - c[-1] / c[-6],
            "momentum": c[-6] / c[-60] - 1, "uptrend": c[-1] > c[-50:].mean(),
            "turnover": turnover, "sector_edge": 0.0,
        }
    groups = defaultdict(list)
    for symbol, row in rows.items():
        if sectors.get(symbol):
            groups[sectors[symbol]].append(row["r30"])
    for symbol, row in rows.items():
        peers = groups.get(sectors.get(symbol))
        if peers and len(peers) >= 4:
            row["sector_edge"] = ((sum(peers) - row["r30"]) / (len(peers) - 1)
                                  - market_return / 100)
    return rows


def _summary(records, split, horizon):
    causes, earlier = Counter(), Counter()
    nav, peak, dd = 1.0, 1.0, 0.0
    net, scores = [], []
    paired, random_pairs = [], []
    for i, row in enumerate(records):
        initial = nav
        for mark in row["path"]:
            nav = initial * mark
            peak = max(peak, nav)
            dd = max(dd, (peak - nav) / peak)
        for outcome, value in row["trades"]:
            causes[outcome] += 1
            if value is None:
                continue
            net.append(value)
            if i < split:
                earlier[outcome] += 1
            else:
                n = sum(earlier.values())
                if n:
                    scores.append(sum(((earlier[c] + 0.5) / (n + 1.5) - (outcome == c))**2 for c in CLASSES))
        paired.append(row["path"][-1] - row["market_path"][-1])
        random_pairs.append(row["path"][-1] - row["random_path"][-1])
    infer = overlap_robust_test({row["ordinal"]: (d, 1) for row, d in zip(records, paired)}, horizon)
    random_infer = overlap_robust_test(
        {row["ordinal"]: (d, 1) for row, d in zip(records, random_pairs)}, horizon)
    market = np.array([row["market_path"][-1] - 1 for row in records])
    portfolio = np.array([row["path"][-1] - 1 for row in records])
    market_variance = np.var(market)
    resolved = sum(causes[c] for c in CLASSES)
    late = {c: causes[c] - earlier[c] for c in CLASSES}
    return {
        "cycles": len(records), "active_cycles": sum(bool(r["trades"]) for r in records),
        "opportunities": sum(r["opportunities"] for r in records), "selected": sum(causes.values()),
        "resolved": resolved, "counts": dict(causes), "mean_net_pct_per_resolved": float(np.mean(net))*100 if net else None,
        "net_r_per_resolved": float(np.mean(net))/STOP if net else None,
        "return_pct": (nav - 1)*100, "max_drawdown_pct": dd*100,
        "matched_market_beta_descriptive": (
            float(np.mean((market - market.mean()) * (portfolio - portfolio.mean())) / market_variance)
            if market_variance > 0 else None),
        "matched_market_excess_pct_per_cycle": float(np.mean(paired))*100,
        "matched_random_excess_pct_per_cycle": float(np.mean(random_pairs))*100,
        "matched_market_lower_95_pct_per_cycle": (
            infer["lower_95_one_sided"]*100 if infer["lower_95_one_sided"] is not None else None),
        "matched_random_lower_95_pct_per_cycle": (
            random_infer["lower_95_one_sided"]*100 if random_infer["lower_95_one_sided"] is not None else None),
        "inference_status": infer["status"],
        "early_exit_forecast": {c: (earlier[c]+.5)/(sum(earlier.values())+1.5) for c in CLASSES}
                               if earlier else {},
        "later_exit_rate": {c: late[c]/sum(late.values()) for c in CLASSES} if sum(late.values()) else {},
        "later_brier": float(np.mean(scores)) if scores else None,
    }


def _path(trades, horizon, slots=SLOTS):
    if not trades:
        return [1.0] * horizon
    return (1 + np.sum([np.asarray(t[1]) - 1 for t in trades], axis=0) / slots).tolist()


def run(root, symbols):
    from research.intelligence.data.nse_calendar import covered_years, documented_special_sessions, is_session, load_manifest, weekday_holiday_digest
    from scan.sector_heat import _load_map
    from scan.unified_scanner import UnifiedScanner

    from core.live_volume import on_live_volume

    if not root.is_dir() or on_live_volume(root):
        raise ValueError("Specify an isolated research archive")
    logs = root.resolve() / "logs"
    from data.nse_universe import _is_non_stock_fund

    stocks = sorted(set(symbols))
    if not stocks or any(_is_non_stock_fund(stock) for stock in stocks):
        raise ValueError("Supply an explicit NSE operating-company cohort")
    required = [logs / name for name in (
        "nse_calendar_manifest.json", "nse_holidays.json", "ca_events.as-traded.json")]
    if any(not path.resolve().is_relative_to(root.resolve()) for path in required):
        raise ValueError("Metadata path escapes the isolated root")
    manifest = load_manifest(logs / "nse_calendar_manifest.json")
    holidays = set(json.loads((logs/"nse_holidays.json").read_text()))
    if START.year not in covered_years(holidays, manifest):
        raise ValueError("Verified 2025 calendar required")
    special = documented_special_sessions(manifest)
    dates = [START + timedelta(days=i) for i in range((END-START).days)]
    expected = [d for d in dates if is_session(d, holidays) or d.isoformat() in special]
    action_file = logs / "ca_events.as-traded.json"
    from data.corporate_actions import load_events
    actions = load_events(action_file)
    if not actions:
        raise ValueError("Verified corporate actions required")
    inputs = {}
    for d in expected:
        p = logs / "bhav" / d.strftime("%d%m%Y.csv")
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("Archive path escapes the isolated root")
        inputs[d] = p.read_bytes() if p.is_file() else None
    frames, validation = scoped_equity_frames(inputs.items(), stocks, actions)
    dates = pd.DatetimeIndex(expected)
    market_prices = pd.DataFrame({symbol: f.close for symbol,f in frames.items()}).reindex(dates)
    market_daily = market_prices.pct_change(fill_method=None).mean(axis=1)
    market_index = (1 + market_daily.fillna(0)).cumprod()
    benchmark30 = market_index.pct_change(30)
    risk_on = market_index > market_index.rolling(50).mean()
    scanner = UnifiedScanner(evidence_gate=False, calibration_snapshot={"multipliers": {}})
    sectors = _load_map()
    horizon = HORIZON_SESSIONS
    fee = round_trip_cost_pct("CNC") / 100
    if horizon != 10 or not np.isclose(fee, 0.0032):
        raise ValueError("Use the frozen 10-session rule and 0.32% costs")
    rng = random.Random(SEED)
    strategies = defaultdict(list)
    attribution = defaultdict(list)
    combinations = defaultdict(list)
    conditions = Counter()
    for i in range(59, len(dates) - horizon, horizon):
        rows = _features(frames, dates, i, scanner, sectors, benchmark30.iloc[i]*100)
        if len(rows) < SLOTS:
            continue
        conditions["risk_on" if risk_on.iloc[i] else "risk_off"] += 1
        rules = _rules(rows, risk_on.iloc[i], benchmark30.iloc[i])
        forward = dates[i+1:i+horizon+1]
        trades = {s: _trade(frames[s], forward, fee) for s in rows}
        market_path = np.array(_path(list(trades.values()), horizon, len(trades)))
        random_names = rng.sample(sorted(rows), SLOTS)
        alternatives = {"random": random_names}
        for label, selected in {**rules, **alternatives}.items():
            pick = selected[:SLOTS]
            matched = market_path * (len(pick)/SLOTS) + 1 - len(pick)/SLOTS
            strategies[label].append({
                "day": dates[i].date().isoformat(), "ordinal": i,
                "opportunities": len(rows) if label == "random" else len(selected),
                "trades": [(trades[s][2], trades[s][0]) for s in pick],
                "path": _path([trades[s] for s in pick], horizon), "market_path": matched.tolist(),
                "random_path": _path([trades[s] for s in random_names[:len(pick)]], horizon),
            })
        resolved = [t[0] for t in trades.values() if t[0] is not None]
        if not resolved:
            continue
        same_date_mean = float(np.mean(resolved))
        for symbol, row in rows.items():
            value, _, _ = trades[symbol]
            if value is None:
                continue
            for signal in row["signals"]:
                attribution[signal].append((value, value - same_date_mean))
            if len(row["signals"]) > 1:
                combinations["+".join(sorted(row["signals"]))].append((value, value - same_date_mean))
    count = len(strategies["scanner"])
    if count < 2:
        raise ValueError("Too few eligible decision cycles")
    split = count // 2
    return {
        "discovery_only": True, "reserved_start": END.isoformat(), "symbols": stocks,
        "archive_sha256": {d.isoformat(): sha256(b).hexdigest() if b else None for d,b in inputs.items()},
        "actions_sha256": sha256(action_file.read_bytes()).hexdigest(), "validation": validation,
        "calendar_holiday_digest": weekday_holiday_digest(holidays, START.year)[1],
        "risk_regime_cycles": dict(conditions), "sector_mapped_symbols": sum(s in sectors for s in stocks),
        "calibration_early_cycles": split,
        "strategies": {label: _summary(records, split, horizon) for label, records in strategies.items()},
        "cyclic": dict(strategies),
        "signal_attribution": {k: {"n":len(v), "net_pct":float(np.mean(v, axis=0)[0])*100,
                                   "matched_market_excess_pct":float(np.mean(v, axis=0)[1])*100} for k,v in sorted(attribution.items())},
        "combination_attribution": {k: {"n":len(v), "net_pct":float(np.mean(v, axis=0)[0])*100,
                                   "matched_market_excess_pct":float(np.mean(v, axis=0)[1])*100}
                                    for k,v in sorted(combinations.items()) if len(v) >= 20},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stocks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    stocks = args.stocks.read_text().strip().split(",")
    result = run(args.root, [s.strip().upper() for s in stocks if s.strip()])
    with args.out.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
