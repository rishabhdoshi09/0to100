"""Supervised NSE F&O directional scan using existing QuantTerm data stores.

The expensive derivatives reads occur only after every mapped F&O underlying
passes through a cheap point-in-time breakout/breakdown prefilter. This module
is market-data only and returns paper candidates; it never mutates a broker.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import pandas as pd

from data.nfo_market import (
    candidate_option_instruments,
    previous_future_close_oi,
    read_market_quotes,
    read_nfo_quotes,
)
from product.fo_snapshot_engine import evaluate_fo_snapshot_auto


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _daily_prefilter(frame: pd.DataFrame | None, *, lookback: int = 20) -> dict[str, Any]:
    if frame is None or len(frame) < max(60, lookback + 2):
        return {"eligible": False, "reason": "INSUFFICIENT_HISTORY"}
    try:
        close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
        high = pd.to_numeric(frame["high"], errors="coerce").astype(float)
        low = pd.to_numeric(frame["low"], errors="coerce").astype(float)
        volume = pd.to_numeric(frame["volume"], errors="coerce").astype(float)
    except Exception:
        return {"eligible": False, "reason": "INVALID_OHLCV_SCHEMA"}
    if any(series.isna().iloc[-1] for series in (close, high, low, volume)):
        return {"eligible": False, "reason": "LATEST_BAR_INVALID"}
    price = float(close.iloc[-1])
    prior_high = float(high.iloc[-(lookback + 1):-1].max())
    prior_low = float(low.iloc[-(lookback + 1):-1].min())
    avg_vol = float(volume.iloc[-21:-1].mean())
    rvol = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 0.0
    turnover_crore = float((close * volume).iloc[-20:].mean() / 1e7)
    directions: list[str] = []
    if price > prior_high:
        directions.append("LONG")
    if price < prior_low:
        directions.append("SHORT")
    if not directions:
        return {
            "eligible": False,
            "reason": "NO_CONFIRMED_20D_BREAK",
            "price": round(price, 4),
            "rvol": round(rvol, 4),
        }
    if rvol < 1.0:
        return {
            "eligible": False,
            "reason": "BREAK_WITHOUT_VOLUME_CONFIRMATION",
            "price": round(price, 4),
            "rvol": round(rvol, 4),
        }
    if turnover_crore < 5.0:
        return {
            "eligible": False,
            "reason": "UNDERLYING_LIQUIDITY_TOO_LOW",
            "price": round(price, 4),
            "rvol": round(rvol, 4),
        }
    clearance = max(
        (price - prior_high) / prior_high * 100.0 if price > prior_high and prior_high > 0 else 0.0,
        (prior_low - price) / prior_low * 100.0 if price < prior_low and prior_low > 0 else 0.0,
    )
    return {
        "eligible": True,
        "reason": "DEEP_EVIDENCE_REQUIRED",
        "directions": directions,
        "price": round(price, 4),
        "rvol": round(rvol, 4),
        "clearance_pct": round(clearance, 4),
        "priority": round(clearance * max(rvol, 1.0), 6),
    }


def _index_context() -> dict[str, float | None]:
    try:
        from data.index_store import get_index_ohlcv
        frame = get_index_ohlcv("^NSEI")
    except Exception:
        frame = None
    if frame is None or len(frame) < 21:
        return {"return_20d_pct": None, "return_5d_pct": None, "last_close": None}
    close_col = next((c for c in frame.columns if str(c).lower() == "close"), None)
    if close_col is None:
        return {"return_20d_pct": None, "return_5d_pct": None, "last_close": None}
    close = pd.to_numeric(frame[close_col], errors="coerce").dropna().astype(float)
    if len(close) < 21 or close.iloc[-21] <= 0 or close.iloc[-6] <= 0:
        return {"return_20d_pct": None, "return_5d_pct": None, "last_close": None}
    return {
        "return_20d_pct": (float(close.iloc[-1]) / float(close.iloc[-21]) - 1.0) * 100.0,
        "return_5d_pct": (float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0) * 100.0,
        "last_close": float(close.iloc[-1]),
    }


def _sector_strength_map(nifty_5d_pct: float | None) -> dict[str, float]:
    try:
        from scan.sector_heat import sector_performance
        rows = sector_performance()
    except Exception:
        rows = []
    benchmark = _f(nifty_5d_pct)
    return {
        str(row.get("sector") or ""): _f(row.get("chg_5d")) - benchmark
        for row in rows
        if str(row.get("sector") or "")
    }


def _sector_for(symbol: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(symbol) or "")
    except Exception:
        return ""


def _nifty_change_from_quote(quote: Mapping[str, Any] | None) -> float | None:
    quote = quote or {}
    last = _f(quote.get("last_price"))
    ohlc = quote.get("ohlc")
    prev_close = _f(ohlc.get("close")) if isinstance(ohlc, Mapping) else 0.0
    if last > 0 and prev_close > 0:
        return (last / prev_close - 1.0) * 100.0
    return None


def run_fo_directional_scan(
    *,
    report,
    instrument_rows: Sequence[Mapping[str, Any]],
    client,
    as_of: date,
    history_getter=None,
) -> dict[str, Any]:
    """Run one complete read-only F&O directional scan."""
    history_session: dict[str, Any] = {}
    if history_getter is None:
        from scan.bulk_fetcher import adopt_ready_store, get_cached

        # F&O discovery must see the current session before applying breakout
        # gates. The generic cash scanner deliberately overlays live bars in a
        # background thread for latency, but doing that here creates a race:
        # the prefilter can read yesterday's EOD frame before today's bar lands
        # and silently miss an intraday breakout. Keep this lane synchronous
        # and fail closed when current-session data cannot be established.
        adopted = int(adopt_ready_store(overlay_live=False) or 0)
        if adopted < 200:
            return {
                "available": False,
                "status": "BLOCKED",
                "code": "FNO_HISTORY_STORE_UNAVAILABLE",
                "universe_size": len(list(getattr(report, "underlyings", ()) or ())),
                "candidates": [],
                "paper_only": True,
                "live_execution_allowed": False,
            }
        try:
            from data.nse_live import live_session_ready

            history_session = dict(live_session_ready(apply=True) or {})
        except Exception as exc:
            history_session = {
                "ready": False,
                "source": "",
                "session_date": "",
                "reason": f"{type(exc).__name__}: {exc}"[:200],
            }
        if not bool(history_session.get("ready")):
            return {
                "available": False,
                "status": "BLOCKED",
                "code": "FNO_CURRENT_SESSION_UNAVAILABLE",
                "universe_size": len(list(getattr(report, "underlyings", ()) or ())),
                "history_session": history_session,
                "candidates": [],
                "paper_only": True,
                "live_execution_allowed": False,
            }
        history_getter = get_cached

    universe = list(getattr(report, "underlyings", ()) or ())
    considered: list[dict[str, Any]] = []
    deep: list[tuple[Any, pd.DataFrame, dict[str, Any]]] = []
    for item in universe:
        symbol = str(getattr(item, "symbol", "") or "").upper()
        try:
            frame = history_getter(symbol)
        except Exception as exc:
            considered.append({"symbol": symbol, "stage": "history", "reason": f"HISTORY_ERROR:{type(exc).__name__}"})
            continue
        pre = _daily_prefilter(frame)
        considered.append({"symbol": symbol, "stage": "prefilter", **pre})
        if pre.get("eligible"):
            deep.append((item, frame, pre))

    deep.sort(key=lambda row: float(row[2].get("priority") or 0.0), reverse=True)
    if not deep:
        return {
            "available": True,
            "status": "READY",
            "as_of": as_of.isoformat(),
            "universe_size": len(universe),
            "prefilter_passed": 0,
            "deep_evaluated": 0,
            "candidate_count": 0,
            "decision": "NO_ELIGIBLE_TRADE",
            "candidates": [],
            "decisions": [],
            "deep_failures": [],
            "considered": considered,
            "quote_scope": {"deep_underlyings": 0, "option_contracts_requested": 0},
            "history_session": history_session,
            "paper_only": True,
            "live_execution_allowed": False,
            "probability_claim": None,
        }

    index = _index_context()
    nifty_20 = index.get("return_20d_pct")
    nifty_5 = index.get("return_5d_pct")
    if nifty_20 is None:
        return {
            "available": False,
            "status": "BLOCKED",
            "code": "NIFTY_HISTORY_UNAVAILABLE",
            "universe_size": len(universe),
            "considered": considered,
            "candidates": [],
            "paper_only": True,
            "live_execution_allowed": False,
        }

    quote_keys = ["NSE:NIFTY 50"]
    for item, _, _ in deep:
        quote_keys.append(f"NSE:{getattr(item, 'symbol', '')}")
        quote_keys.append(f"NFO:{getattr(item, 'future_symbol', '')}")
    quotes = read_market_quotes(quote_keys, client=client)
    nifty_change = _nifty_change_from_quote(quotes.get("NSE:NIFTY 50"))
    if nifty_change is None:
        return {
            "available": False,
            "status": "BLOCKED",
            "code": "CURRENT_NIFTY_DIRECTION_UNAVAILABLE",
            "universe_size": len(universe),
            "considered": considered,
            "candidates": [],
            "paper_only": True,
            "live_execution_allowed": False,
        }
    sector_strengths = _sector_strength_map(nifty_5)

    option_meta_by_symbol: dict[str, list[dict[str, Any]]] = {}
    option_symbols: list[str] = []
    nfo_rows = [
        dict(row) for row in instrument_rows
        if str(row.get("exchange") or "").upper() == "NFO"
        or str(row.get("segment") or "").upper().startswith("NFO")
    ]
    for item, _, pre in deep:
        symbol = str(getattr(item, "symbol", "") or "").upper()
        uq = quotes.get(f"NSE:{symbol}") or {}
        spot = _f(uq.get("last_price"), _f(pre.get("price")))
        metas = candidate_option_instruments(
            nfo_rows, symbol, spot=spot, as_of=as_of, max_expiries=2,
        )
        option_meta_by_symbol[symbol] = metas
        option_symbols.extend(str(row.get("tradingsymbol") or "") for row in metas)
    option_quotes = read_nfo_quotes(option_symbols, client=client)

    decisions: list[dict[str, Any]] = []
    deep_failures: list[dict[str, Any]] = []
    for item, frame, pre in deep:
        symbol = str(getattr(item, "symbol", "") or "").upper()
        future_symbol = str(getattr(item, "future_symbol", "") or "")
        future_token = int(getattr(item, "instrument_token", 0) or 0)
        uq = quotes.get(f"NSE:{symbol}") or {}
        fq = quotes.get(f"NFO:{future_symbol}") or {}
        previous = previous_future_close_oi(
            future_token, as_of=as_of, client=client,
        )
        if not previous:
            deep_failures.append({
                "symbol": symbol,
                "stage": "futures_oi",
                "reason": "PREVIOUS_FUTURES_OI_UNAVAILABLE",
            })
            continue

        subset = {
            str(meta.get("tradingsymbol") or ""): option_quotes.get(
                str(meta.get("tradingsymbol") or ""), {}
            )
            for meta in option_meta_by_symbol.get(symbol, [])
        }
        sector = _sector_for(symbol)
        sector_strength = sector_strengths.get(sector)
        if sector_strength is None:
            deep_failures.append({
                "symbol": symbol,
                "stage": "sector",
                "reason": "SECTOR_STRENGTH_UNAVAILABLE",
                "sector": sector,
            })
            continue
        try:
            result = evaluate_fo_snapshot_auto(
                symbol=symbol,
                daily_bars=frame,
                nfo_instruments=nfo_rows,
                underlying_quote=uq,
                futures_quote=fq,
                previous_futures_price=float(previous["close"]),
                previous_futures_oi=float(previous["oi"]),
                option_quotes=subset,
                benchmark_20d_return_pct=float(nifty_20),
                nifty_change_pct=float(nifty_change),
                sector_relative_strength_pct=float(sector_strength),
                iv_percentile=None,
                as_of=as_of,
            )
        except Exception as exc:
            deep_failures.append({
                "symbol": symbol,
                "stage": "deep_evaluation",
                "reason": f"EVALUATION_ERROR:{type(exc).__name__}",
            })
            continue
        result["sector"] = sector
        result["prefilter"] = pre
        result["previous_future"] = previous
        decisions.append(result)

    candidates = [
        row["selected"] for row in decisions
        if row.get("decision") == "PAPER_OPTION_CANDIDATE"
        and isinstance(row.get("selected"), Mapping)
    ]
    candidates.sort(
        key=lambda row: (
            float((row.get("setup") or {}).get("score") or 0.0),
            float((row.get("selected_contract") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    return {
        "available": True,
        "status": "READY",
        "as_of": as_of.isoformat(),
        "universe_size": len(universe),
        "prefilter_passed": len(deep),
        "deep_evaluated": len(decisions),
        "candidate_count": len(candidates),
        "decision": "PAPER_CANDIDATES" if candidates else "NO_ELIGIBLE_TRADE",
        "candidates": candidates,
        "decisions": decisions,
        "deep_failures": deep_failures,
        "considered": considered,
        "market_context": {
            "nifty_change_pct": round(float(nifty_change), 4),
            "nifty_20d_return_pct": round(float(nifty_20), 4),
            "nifty_5d_return_pct": round(float(nifty_5), 4) if nifty_5 is not None else None,
        },
        "quote_scope": {
            "deep_underlyings": len(deep),
            "option_contracts_requested": len(option_symbols),
        },
        "history_session": history_session,
        "paper_only": True,
        "live_execution_allowed": False,
        "probability_claim": None,
    }
