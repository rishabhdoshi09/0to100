"""
ScreenerEngine — filter the entire NSE universe by fundamentals,
technicals, and ensemble ML signal.

Filter order (fast → slow to minimise network calls):
  1. Technicals  (SQLite cache / yfinance — fast)
  2. Fundamentals (screener.in SQLite cache — medium; skipped if not cached and
                   --no-scrape is set)
  3. Ensemble ML  (in-process model inference — optional)

First run warms the technicals cache for all symbols (~200 s with 6 threads).
Subsequent runs are instant (SQLite only).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from logger import get_logger
from screener.universe import StockUniverseFetcher
from screener.technicals import TechnicalsCache

log = get_logger(__name__)

_universe = StockUniverseFetcher()
_tech     = TechnicalsCache()


# ── Fundamental extractor ──────────────────────────────────────────────────────

def _parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[₹,% ]", "", str(val))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_fundamentals(data: Dict) -> Dict:
    """Pull key ratios, debt, and promoter holding out of deep-fundamentals dict."""
    result: Dict[str, Any] = {}

    # Build a lower-cased lookup from key_ratios list
    kr: Dict[str, Any] = {}
    for item in data.get("key_ratios", []):
        key = str(item.get("name", "")).lower().strip()
        kr[key] = item.get("value")

    def _first(*keys):
        for k in keys:
            if k in kr:
                return _parse_float(kr[k])
        return None

    result["pe"]             = _first("stock p/e", "p/e", "price to earning", "pe")
    result["roe"]            = _first("roe", "return on equity")
    result["roce"]           = _first("roce", "return on capital employed")
    result["dividend_yield"] = _first("dividend yield", "div yield")
    result["market_cap_cr"]  = _first("market cap", "mkt cap", "market capitalization")

    # Debt-to-equity: try key ratios first, then derive from balance sheet
    de = _first("debt to equity", "d/e ratio", "debt / equity", "debt/equity")
    if de is not None:
        result["debt_to_equity"] = de
    else:
        borrowings = equity = None
        for row in data.get("balance_sheet", []):
            label = str(row.get("", row.get("row_label", ""))).lower()
            vals  = [v for k, v in row.items()
                     if k not in ("", "row_label") and v is not None]
            latest = _parse_float(vals[0]) if vals else None
            if "borrowing" in label and latest is not None:
                borrowings = latest
            if ("equity capital" in label or "total equity" in label) and latest is not None:
                equity = latest
        if borrowings is not None and equity and equity > 0:
            result["debt_to_equity"] = round(borrowings / equity, 2)
        else:
            result["debt_to_equity"] = None

    # Promoter holding — most recent column in shareholding table
    result["promoter_holding"] = None
    for row in data.get("shareholding", []):
        label = str(row.get("", row.get("row_label", ""))).lower()
        if "promoter" in label and "pledge" not in label:
            vals = [v for k, v in row.items()
                    if k not in ("", "row_label") and v is not None]
            if vals:
                result["promoter_holding"] = _parse_float(vals[0])
            break

    return result


# ── Screener Engine ────────────────────────────────────────────────────────────

class ScreenerEngine:
    """
    Screen the NSE universe by any combination of fundamental, technical,
    and ML-signal filters.

    Usage
    -----
    from screener.engine import ScreenerEngine
    df = ScreenerEngine().screen_by_ratios(pe_max=20, roe_min=15, rsi_max=35, limit=20)
    print(df.to_string(index=False))
    """

    def screen_by_ratios(
        self,
        pe_max:               Optional[float] = None,
        roe_min:              Optional[float] = None,
        debt_max:             Optional[float] = None,
        market_cap_min_cr:    Optional[float] = None,
        promoter_holding_min: Optional[float] = None,
        dividend_yield_min:   Optional[float] = None,
        rsi_max:              Optional[float] = None,
        rsi_min:              Optional[float] = None,
        volume_spike_min:     Optional[float] = None,
        price_above_sma_days: Optional[int]   = None,
        price_below_sma_days: Optional[int]   = None,
        ensemble_signal:      Optional[str]   = None,
        symbols:              Optional[List[str]] = None,
        limit:                int = 50,
        scrape_missing_fundamentals: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of stocks matching ALL supplied filters.

        Parameters
        ----------
        scrape_missing_fundamentals
            If True, fetch screener.in for symbols not yet in cache.
            If False (default), skip symbols whose fundamentals aren't cached.
            Set to True for a thorough (but slow) first-time scan.
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw):
                return it

        # ── Step 1: resolve symbol list ────────────────────────────────────
        if symbols:
            all_syms = [s.upper() for s in symbols]
        else:
            print("Loading NSE universe …")
            all_syms = _universe.get_all_symbols()
        print(f"Universe: {len(all_syms):,} symbols")

        # ── Step 2: warm technicals cache (skips already-fresh entries) ────
        _has_tech_filter = any([
            rsi_max, rsi_min, volume_spike_min,
            price_above_sma_days, price_below_sma_days,
        ])
        print(f"Updating technicals cache …")
        _tech.bulk_update(all_syms, skip_fresh=True, show_progress=True)

        # ── Step 3: technical filter ───────────────────────────────────────
        tech_pass: List[tuple] = []
        for sym in all_syms:
            t = _tech.get(sym)
            if t is None:
                continue
            if rsi_max  is not None and (t["rsi_14"]       is None or t["rsi_14"]       > rsi_max):
                continue
            if rsi_min  is not None and (t["rsi_14"]       is None or t["rsi_14"]       < rsi_min):
                continue
            if volume_spike_min is not None and (t["volume_ratio"] is None or t["volume_ratio"] < volume_spike_min):
                continue
            if price_above_sma_days == 20 and t["sma_20"] and t["current_price"]:
                if t["current_price"] <= t["sma_20"]:
                    continue
            if price_above_sma_days == 50 and t["sma_50"] and t["current_price"]:
                if t["current_price"] <= t["sma_50"]:
                    continue
            if price_below_sma_days == 20 and t["sma_20"] and t["current_price"]:
                if t["current_price"] >= t["sma_20"]:
                    continue
            if price_below_sma_days == 50 and t["sma_50"] and t["current_price"]:
                if t["current_price"] >= t["sma_50"]:
                    continue
            tech_pass.append((sym, t))

        print(f"After technical filter: {len(tech_pass):,} stocks")

        # ── Step 4: fundamental filter ─────────────────────────────────────
        _has_fund_filter = any([
            pe_max, roe_min, debt_max, market_cap_min_cr,
            promoter_holding_min, dividend_yield_min,
        ])

        from fundamentals.fetcher import get_deep_fundamentals

        results: List[Dict] = []

        for sym, tech in _tqdm(tech_pass, desc="Fundamental filter", unit="sym", ncols=80):
            # --- fundamentals ---
            fund: Dict = {}
            if _has_fund_filter or ensemble_signal:
                try:
                    raw = get_deep_fundamentals(
                        sym,
                        force_refresh=False,  # always cache-first
                    )
                    fund = _extract_fundamentals(raw)
                except Exception:
                    if scrape_missing_fundamentals:
                        pass  # already attempted above
                    else:
                        # Skip if no cached fundamentals and we're not scraping
                        if _has_fund_filter:
                            continue

            if _has_fund_filter:
                if pe_max is not None:
                    v = fund.get("pe")
                    if v is None or v > pe_max:
                        continue
                if roe_min is not None:
                    v = fund.get("roe")
                    if v is None or v < roe_min:
                        continue
                if debt_max is not None:
                    v = fund.get("debt_to_equity")
                    if v is None or v > debt_max:
                        continue
                if market_cap_min_cr is not None:
                    v = fund.get("market_cap_cr")
                    if v is None or v < market_cap_min_cr:
                        continue
                if promoter_holding_min is not None:
                    v = fund.get("promoter_holding")
                    if v is None or v < promoter_holding_min:
                        continue
                if dividend_yield_min is not None:
                    v = fund.get("dividend_yield")
                    if v is None or v < dividend_yield_min:
                        continue

            # --- ensemble signal filter ---
            sig_label = "N/A"
            if ensemble_signal:
                try:
                    from ml.ensemble_signal import EnsembleSignalGenerator
                    import yfinance as yf
                    df_yf = yf.download(
                        f"{sym}.NS", period="1y", interval="1d",
                        auto_adjust=True, progress=False, timeout=15,
                    )
                    if not df_yf.empty:
                        if isinstance(df_yf.columns, pd.MultiIndex):
                            df_yf = df_yf.droplevel(1, axis=1)
                        df_yf.columns = [c.lower() for c in df_yf.columns]
                        sig = EnsembleSignalGenerator().generate_signal(df_yf, sym)
                        sig_label = sig.get("action", "HOLD")
                        if sig_label != ensemble_signal.upper():
                            continue
                except Exception as exc:
                    log.debug("ensemble_signal_failed", symbol=sym, error=str(exc)[:60])
                    if ensemble_signal:
                        continue

            results.append({
                "symbol":           sym,
                "price":            tech.get("current_price"),
                "rsi_14":           tech.get("rsi_14"),
                "volume_ratio":     tech.get("volume_ratio"),
                "sma_20":           tech.get("sma_20"),
                "sma_50":           tech.get("sma_50"),
                "pe":               fund.get("pe"),
                "roe":              fund.get("roe"),
                "roce":             fund.get("roce"),
                "debt_to_equity":   fund.get("debt_to_equity"),
                "market_cap_cr":    fund.get("market_cap_cr"),
                "promoter_holding": fund.get("promoter_holding"),
                "dividend_yield":   fund.get("dividend_yield"),
                "signal":           sig_label,
            })

            if len(results) >= limit:
                break

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        # Format numerics
        for col in ["price", "sma_20", "sma_50", "pe", "roe", "roce",
                    "debt_to_equity", "market_cap_cr", "promoter_holding", "dividend_yield"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

        log.info("screener_complete", results=len(df))
        return df.head(limit)
