"""DeepSeek-driven 30-min pre-screener.

Pulls the universe (200 NSE liquid stocks) from ``config.yaml``, fetches
daily OHLCV via yfinance in chunks, computes the 5 indicators required
by the screener prompt (SMA20, SMA50, RSI14, vol_ratio, ATR14), asks
DeepSeek to rank them, persists the top-N to SQLite ``screener_results``.

Falls back to a deterministic technical ranking when DeepSeek is offline.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import yaml

from sq_ai.backend.data_fetcher import fetch_yf_history
from sq_ai.backend.llm_clients import DeepSeekClient
from sq_ai.portfolio.tracker import PortfolioTracker
from sq_ai.signals.composite_signal import atr, rsi


CONFIG_PATH = os.environ.get(
    "SQ_CONFIG_PATH", str(Path(__file__).resolve().parents[2] / "config.yaml")
)
IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))


# ── config ───────────────────────────────────────────────────────────────────
def load_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        return {"universe": [], "max_universe_size": 200,
                "screener_top_n": 10, "screener_interval_minutes": 30,
                "decision_interval_minutes": 5,
                "ensemble_threshold_percent": 10.0}
    with p.open() as f:
        return yaml.safe_load(f) or {}


# ── indicators per symbol ────────────────────────────────────────────────────
def compute_screener_features(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 60:
        raise ValueError("need >=60 bars")
    close = df["close"]
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = float(df["volume"].iloc[-1] / vol_avg) if vol_avg else 1.0
    atr14 = float(atr(df, 14).iloc[-1])
    return {
        "price": float(close.iloc[-1]),
        "sma_20": sma20,
        "sma_50": sma50,
        "rsi": rsi14,
        "volume_ratio": vol_ratio,
        "atr": atr14,
    }


# ── prompt assembly ──────────────────────────────────────────────────────────
SCREENER_SYSTEM = (
    "You are a disciplined technical screener for Indian equities. "
    "Output ONLY valid JSON, no prose."
)

SCREENER_USER_TMPL = """You are a technical screener. Rank these {n} stocks by bullish momentum.
For each stock you receive: symbol, price, SMA20, SMA50, RSI, volume_ratio, ATR.
A stock is bullish when SMA20 > SMA50, RSI is in [40,70], volume_ratio > 1.
Return ONLY this JSON:
{{"ranked_tickers": ["SYM1","SYM2",...top_{top_n}],
  "reasons": {{"SYM1": "<=80 chars", ...}}}}

DATA:
{rows}
"""


def build_prompt(features_per_sym: list[dict], top_n: int = 10) -> str:
    rows = []
    for f in features_per_sym:
        rows.append(
            f"- {f['symbol']}: price={f['price']:.2f} sma20={f['sma_20']:.2f} "
            f"sma50={f['sma_50']:.2f} rsi={f['rsi']:.1f} "
            f"vol_ratio={f['volume_ratio']:.2f} atr={f['atr']:.2f}"
        )
    return SCREENER_USER_TMPL.format(
        n=len(features_per_sym), top_n=top_n, rows="\n".join(rows),
    )


# ── deterministic fallback ranking ──────────────────────────────────────────
def fallback_rank(features: list[dict], top_n: int = 10) -> dict[str, Any]:
    scored = []
    for f in features:
        score = 0.0
        if f["sma_20"] > f["sma_50"] * 1.005:
            score += 1.0
        elif f["sma_20"] < f["sma_50"] * 0.995:
            score -= 0.5
        rsi_v = f["rsi"]
        if 40 <= rsi_v <= 70:
            score += 0.5
        elif rsi_v >= 75 or rsi_v <= 25:
            score -= 0.5
        score += min(max(f["volume_ratio"] - 1, 0), 1.0) * 0.5
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]
    return {
        "ranked_tickers": [f["symbol"] for _, f in top],
        "reasons": {
            f["symbol"]: (
                f"sma20{'>' if f['sma_20'] > f['sma_50'] else '<'}sma50, "
                f"rsi={f['rsi']:.0f}, vol_x{f['volume_ratio']:.1f}"
            )
            for _, f in top
        },
        "_fallback": True,
    }


# ── JSON parsing ────────────────────────────────────────────────────────────
def parse_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if "ranked_tickers" not in obj:
            return None
        return obj
    except json.JSONDecodeError:
        return None


# ── main entry-point ────────────────────────────────────────────────────────
class Screener:
    def __init__(self,
                 tracker: PortfolioTracker | None = None,
                 client: DeepSeekClient | None = None,
                 config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.tracker = tracker or PortfolioTracker()
        self.client = client or DeepSeekClient()

    # --------------------------------------------------------------- helpers
    def _gather_features(self) -> list[dict]:
        from sq_ai.backend.universe import get_active_universe
        max_n = int(self.cfg.get("max_universe_size", 500))
        universe = get_active_universe(
            max_symbols=max_n,
            tracker=self.tracker,
            fallback_yaml=self.cfg.get("universe") or [],
        )
        feats: list[dict] = []
        for sym in universe:
            try:
                df = fetch_yf_history(sym, period="6mo", interval="1d")
                if len(df) < 60:
                    continue
                f = compute_screener_features(df)
                f["symbol"] = sym
                feats.append(f)
            except Exception:
                continue
        return feats

    # --------------------------------------------------------------- run
    def run(self, gather_fn=None) -> dict[str, Any]:
        ts = datetime.now(IST)
        features = (gather_fn or self._gather_features)()
        top_n = int(self.cfg.get("screener_top_n", 10))
        if not features:
            return {"date": ts.date().isoformat(), "ranked": [], "note": "no data"}

        prompt = build_prompt(features, top_n=top_n)
        raw = self.client.generate(prompt, max_tokens=600, temperature=0.2,
                                   system=SCREENER_SYSTEM)
        parsed = parse_json(raw) or fallback_rank(features, top_n=top_n)

        feat_by_sym = {f["symbol"]: f for f in features}
        ranked = []
        for sym in parsed.get("ranked_tickers", [])[:top_n]:
            if sym not in feat_by_sym:
                continue
            ranked.append({
                "symbol": sym,
                "score": float(feat_by_sym[sym]["sma_20"] - feat_by_sym[sym]["sma_50"]),
                "reasoning": (parsed.get("reasons") or {}).get(sym, ""),
            })

        date_str = ts.date().isoformat()
        if ranked:
            self.tracker.save_screener(date_str, ranked)
        return {
            "date": date_str,
            "ranked": ranked,
            "used_deepseek": bool(raw) and not parsed.get("_fallback"),
            "n_universe": len(features),
        }


__all__ = ["Screener", "compute_screener_features", "build_prompt",
           "fallback_rank", "parse_json", "load_config"]
