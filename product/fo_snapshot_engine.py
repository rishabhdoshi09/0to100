"""Compose one point-in-time NSE F&O opportunity from trusted snapshots.

No network and no broker mutation occurs here. Data acquisition is deliberately
outside this module so replay/live callers can prove exactly which snapshot was
used.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import pandas as pd

from data.nfo_market import (
    equity_fo_universe,
    futures_oi_features,
    nearest_future,
    option_instruments,
    quote_to_option_contract,
)
from product.fo_evidence import fo_context_key
from product.fo_features import build_fo_features
from product.fo_options_pipeline import evaluate_fo_opportunity


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _spread_bps(quote: Mapping[str, Any]) -> float:
    depth = quote.get("depth")
    if not isinstance(depth, Mapping):
        return 0.0
    buys = depth.get("buy")
    sells = depth.get("sell")
    if not isinstance(buys, Sequence) or not buys or not isinstance(sells, Sequence) or not sells:
        return 0.0
    bid = _f(buys[0].get("price")) if isinstance(buys[0], Mapping) else 0.0
    ask = _f(sells[0].get("price")) if isinstance(sells[0], Mapping) else 0.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4 if bid > 0 and ask >= bid and mid > 0 else 0.0


def evaluate_fo_snapshot(
    *,
    symbol: str,
    direction: str,
    daily_bars: pd.DataFrame,
    nfo_instruments: Sequence[Mapping[str, Any]],
    underlying_quote: Mapping[str, Any],
    futures_quote: Mapping[str, Any],
    previous_futures_price: float | None,
    previous_futures_oi: float | None,
    option_quotes: Mapping[str, Mapping[str, Any]],
    benchmark_20d_return_pct: float,
    nifty_change_pct: float,
    sector_relative_strength_pct: float,
    iv_percentile: float | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Evaluate one direction from pre-captured point-in-time data."""
    symbol = str(symbol or "").upper()
    direction = str(direction or "").upper()
    universe = set(equity_fo_universe(nfo_instruments))
    future = nearest_future(nfo_instruments, symbol, as_of=as_of)
    current_future_price = _f(futures_quote.get("last_price"))
    current_future_oi = _f(futures_quote.get("oi"))
    oi = futures_oi_features(
        current_price=current_future_price,
        previous_price=_f(previous_futures_price) if previous_futures_price is not None else 0.0,
        current_oi=current_future_oi,
        previous_oi=_f(previous_futures_oi) if previous_futures_oi is not None else 0.0,
    )

    vwap = _f(underlying_quote.get("vwap") or underlying_quote.get("average_price"))
    features = build_fo_features(
        symbol=symbol,
        daily_bars=daily_bars,
        direction=direction,
        intraday_vwap=vwap,
        benchmark_20d_return_pct=benchmark_20d_return_pct,
        nifty_change_pct=nifty_change_pct,
        sector_relative_strength_pct=sector_relative_strength_pct,
        futures_price_change_pct=oi["futures_price_change_pct"],
        futures_oi_change_pct=oi["futures_oi_change_pct"],
        is_fo=symbol in universe and future is not None,
        underlying_spread_bps=_spread_bps(underlying_quote),
    )

    spot = _f(underlying_quote.get("last_price"), _f(features.get("price")))
    contracts = []
    for meta in option_instruments(nfo_instruments, symbol, as_of=as_of, max_expiries=2):
        tradingsymbol = str(meta.get("tradingsymbol") or "")
        quote = option_quotes.get(tradingsymbol)
        if not isinstance(quote, Mapping):
            continue
        contracts.append(
            quote_to_option_contract(meta, quote, spot=spot, as_of=as_of)
        )

    result = evaluate_fo_opportunity(
        underlying_features=features,
        direction=direction,
        option_contracts=contracts,
        iv_percentile=iv_percentile,
    )
    result["data_provenance"] = {
        "underlying": "POINT_IN_TIME_SUPPLIED_QUOTE",
        "daily_bars": "POINT_IN_TIME_SUPPLIED_BARS",
        "futures": "NFO_CURRENT_PLUS_PREVIOUS_OI",
        "options": "NFO_QUOTE_DEPTH",
        "iv": "IMPLIED_FROM_MARKET_QUOTE",
    }
    result["futures_contract"] = dict(future or {})

    selected = result.get("selected_contract")
    if isinstance(selected, dict):
        selected["context_key"] = fo_context_key(
            direction=direction,
            futures_oi_state=str(result["setup"].get("futures_oi_state") or "NEUTRAL"),
            rvol=_f(features.get("rvol")),
            adx=_f(features.get("adx")),
            delta=_f(selected.get("delta")),
            dte=int(_f(selected.get("dte"))),
            iv_percentile=iv_percentile,
        )
    return result


def evaluate_fo_snapshot_auto(
    *,
    symbol: str,
    daily_bars: pd.DataFrame,
    nfo_instruments: Sequence[Mapping[str, Any]],
    underlying_quote: Mapping[str, Any],
    futures_quote: Mapping[str, Any],
    previous_futures_price: float | None,
    previous_futures_oi: float | None,
    option_quotes: Mapping[str, Mapping[str, Any]],
    benchmark_20d_return_pct: float,
    nifty_change_pct: float,
    sector_relative_strength_pct: float,
    iv_percentile: float | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Evaluate LONG and SHORT without forcing a direction when neither qualifies."""
    rows = [
        evaluate_fo_snapshot(
            symbol=symbol,
            direction=direction,
            daily_bars=daily_bars,
            nfo_instruments=nfo_instruments,
            underlying_quote=underlying_quote,
            futures_quote=futures_quote,
            previous_futures_price=previous_futures_price,
            previous_futures_oi=previous_futures_oi,
            option_quotes=option_quotes,
            benchmark_20d_return_pct=benchmark_20d_return_pct,
            nifty_change_pct=nifty_change_pct,
            sector_relative_strength_pct=sector_relative_strength_pct,
            iv_percentile=iv_percentile,
            as_of=as_of,
        )
        for direction in ("LONG", "SHORT")
    ]
    candidates = [
        row for row in rows
        if row.get("decision") == "PAPER_OPTION_CANDIDATE"
    ]
    candidates.sort(
        key=lambda row: (
            float((row.get("setup") or {}).get("score") or 0.0),
            float((row.get("selected_contract") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    return {
        "symbol": str(symbol or "").upper(),
        "decision": "PAPER_OPTION_CANDIDATE" if candidates else "NO_TRADE",
        "selected": candidates[0] if candidates else None,
        "directions": rows,
        "paper_only": True,
        "live_execution_allowed": False,
    }
