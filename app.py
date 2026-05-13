"""
QUANTTERM — Professional Algorithmic Trading Terminal
Clean 5-tab layout: Home | Terminal | Research | AlgoLab | Tools
"""

import logging
import os
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ── Lightweight structured logger ────────────────────────────────────────────
_LOG_DIR = os.environ.get("DEVBLOOM_LOG_DIR", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_handler = logging.FileHandler(
    os.path.join(_LOG_DIR, "streamlit_app.log"), encoding="utf-8"
)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger = logging.getLogger("quantterm.app")
if not logger.handlers:
    logger.addHandler(_log_handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
logger.info("quantterm_start pid=%s", os.getpid())

load_dotenv()

# ── Core data / feature imports ───────────────────────────────────────────────
from config import Settings
from data.historical import HistoricalDataFetcher
from data.instruments import InstrumentManager
from data.kite_client import KiteClient
from features.indicators import IndicatorEngine
from features.market_structure import is_recent_swing_breakout
from features.volume_profile import VolumeProfile
from paper_trading import (
    close_position,
    get_closed_positions,
    get_equity_curve,
    get_open_positions,
    get_trading_summary,
    init_db,
    open_position,
)
from signals.conviction import ConvictionScorer
from signals.profiles import PROFILES
from signals.trade_setup import compute_trade_setup

# ── UI modules ────────────────────────────────────────────────────────────────
from ui.theme import DEVBLOOM_CSS, COMMAND_PALETTE_JS
from ui.heatmap import render_heatmap, render_macro_strip
from ui.watchlist import render_watchlist
from ui.alert_inbox import render_alert_inbox
from ui.macro import render_macro_dashboard
from ui.copilot import render_copilot_sidebar, render_copilot_inline
from ai.dual_llm_service import get_service as _get_svc
from ui.order_pad import (
    render_order_pad,
    render_position_monitor,
    render_equity_curve,
    render_backtest_bridge,
)
from ui.algolab import render_algolab
from ui.journal import render_journal, log_trade_to_journal
from ui.anomaly_scanner import render_anomaly_scanner
from charting.multi_tf import render_multi_tf_grid
from ui.agent_dashboard import render_agent_dashboard
from ui.memory_vault import render_memory_vault
from ui.earnings_page import render_earnings_page
from ui.news_feed import render_news_feed
from ui.homepage import render_homepage
from ui.scanner import render_scanner
from ui.alerts_page import render_alerts_page
from ui.options_page import render_options_page
from ui.fii_dii_page import render_fii_dii_page
from ui.vcp_page import render_vcp_page

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QUANTTERM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DEVBLOOM_CSS, unsafe_allow_html=True)

# ── Settings ──────────────────────────────────────────────────────────────────
settings = Settings()


# ── Cached client initialisation ─────────────────────────────────────────────
@st.cache_resource
def init_clients():
    kite    = KiteClient()
    im      = InstrumentManager()
    fetcher = HistoricalDataFetcher(kite, im)
    ie      = IndicatorEngine()
    vp      = VolumeProfile()
    return kite, im, fetcher, ie, vp


kite, im, fetcher, ie, vp = init_clients()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_all_equity_symbols():
    out = {}
    for sym, meta in im._meta_map.items():
        ex  = meta.get("exchange", "")
        seg = meta.get("segment", "")
        it  = meta.get("instrument_type", "")
        if ex in ("NSE", "BSE") and seg == ex and it == "EQ":
            if not sym[0].isdigit() and "-" not in sym and len(sym) <= 10:
                out[sym] = meta.get("companyName", sym)
    return out or {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy",
        "HDFCBANK": "HDFC Bank",
    }


@st.cache_data(ttl=300)
def fetch_historical(symbol, days=250):
    to_d   = datetime.now().strftime("%Y-%m-%d")
    from_d = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
    df = fetcher.fetch(symbol, from_d, to_d, interval="day")
    if df is None or len(df) == 0:
        from_d = (datetime.now() - timedelta(days=days + 100)).strftime("%Y-%m-%d")
        df = fetcher.fetch(symbol, from_d, to_d, interval="day")
    if df is None or len(df) == 0:
        try:
            ticker = (
                symbol
                if symbol.endswith(".NS") or symbol.startswith("^")
                else symbol + ".NS"
            )
            raw = yf.download(
                ticker, period=f"{days}d", interval="1d",
                progress=False, auto_adjust=True,
            )
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                df = raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            pass
    return df


@st.cache_data(ttl=3600)
def get_indices_data():
    names = {
        "Nifty 50": "NIFTY 50",
        "Bank Nifty": "NIFTY BANK",
        "Nifty IT": "NIFTY IT",
        "Nifty Pharma": "NIFTY PHARMA",
        "Nifty FMCG": "NIFTY FMCG",
    }
    out = {}
    for name, sym in names.items():
        try:
            df = fetch_historical(sym, days=5)
            if df is not None and len(df) >= 2:
                last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
                out[name] = {"price": last, "change": (last - prev) / prev * 100}
            elif df is not None and len(df) == 1:
                out[name] = {"price": df["close"].iloc[-1], "change": 0.0}
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600)
def get_global_indices():
    tickers = {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC",
        "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225",
        "Hang Seng": "^HSI", "Shanghai": "000001.SS",
    }
    out = {}
    for name, t in tickers.items():
        try:
            h = yf.Ticker(t).history(period="2d")
            if len(h) >= 2:
                _hc = h["Close"] if "Close" in h.columns else h[h.columns[3]]
                last, prev = float(_hc.iloc[-1]), float(_hc.iloc[-2])
                out[name] = {"price": last, "change": (last - prev) / prev * 100}
        except Exception:
            continue
    return out


@st.cache_data(ttl=86400)
def get_market_cap(symbol):
    try:
        info = yf.Ticker(symbol + ".NS").info
        mc = info.get("marketCap", 0)
        return mc / 1e7 if mc else 0
    except Exception:
        return 0


def categorize_by_mcap(symbol):
    mc = get_market_cap(symbol)
    if mc >= 20000: return "Largecap"
    if mc >= 5000:  return "Midcap"
    if mc > 0:      return "Smallcap"
    return "Unknown"


def get_symbols_by_market_cap(cap_filter, max_symbols=500):
    all_syms = list(get_all_equity_symbols().keys())
    if cap_filter == "All":
        return all_syms[:max_symbols]
    out = []
    for sym in all_syms:
        mc = get_market_cap(sym)
        if   mc >= 20000 and cap_filter == "Largecap":          out.append(sym)
        elif 5000 <= mc < 20000 and cap_filter == "Midcap":     out.append(sym)
        elif 0 < mc < 5000 and cap_filter == "Smallcap":        out.append(sym)
        if len(out) >= max_symbols:
            break
    return out


def market_status():
    now = datetime.now()
    t   = now.time()
    ot  = datetime.strptime("09:15", "%H:%M").time()
    ct  = datetime.strptime("15:30", "%H:%M").time()
    if now.weekday() >= 5:
        return "🔴 Closed (Weekend)", "#fee2e2"
    if t < ot:
        mins = int((datetime.combine(now.date(), ot) - now).total_seconds() // 60)
        return f"🟡 Pre-Market · Opens in {mins}m", "#fef9c3"
    if t > ct:
        return "🔴 Closed (After Hours)", "#fee2e2"
    return "🟢 Market Open", "#dcfce7"


# ── Memory helpers ─────────────────────────────────────────────────────────────
_MEMORY_FILE = "trading_memory.json"


def load_memory():
    if os.path.exists(_MEMORY_FILE):
        with open(_MEMORY_FILE) as f:
            return json.load(f)
    return {}


def save_memory(mem):
    with open(_MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)


def update_memory(symbol, decision, confidence, price):
    mem = load_memory()
    mem.setdefault(symbol, []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "decision": decision,
        "confidence": confidence,
        "price": price,
        "actual_return": "pending",
    })
    mem[symbol] = mem[symbol][-10:]
    save_memory(mem)


def get_recent_memory(symbol, limit=3):
    return load_memory().get(symbol, [])[-limit:]


# ── DeepSeek API helper ────────────────────────────────────────────────────────
def call_deepseek(prompt, system="You are a financial analyst."):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 800,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" not in data:
            err = data.get("error", {})
            return f"DeepSeek error: {err.get('message', resp.text[:200])}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek unavailable: {e}"


# ── Scanner helpers ────────────────────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def fetch_and_test(symbol, test_func, days=150):
    try:
        df = fetch_historical(symbol, days=days)
        if df is not None and len(df) >= days - 20:
            return symbol, test_func(df)
    except Exception:
        pass
    return symbol, False


def is_breakout_candidate(df):
    if df is None or len(df) < 120:
        return False
    close     = df["close"]
    base_high = close.rolling(90).max().iloc[-1]
    base_low  = close.rolling(90).min().iloc[-1]
    if (base_high - base_low) == 0:
        return False
    near_top   = (close.iloc[-1] - base_low) / (base_high - base_low) > 0.8
    volume_dry = df["volume"].iloc[-20:].mean() < 0.7 * df["volume"].rolling(120).mean().iloc[-1]
    if not is_recent_swing_breakout(df, lookback=3):
        return False
    vol_spike = df["volume"].iloc[-1] > 1.5 * df["volume"].rolling(20).mean().iloc[-1]
    ema10     = close.ewm(span=10).mean()
    ema20     = close.ewm(span=20).mean()
    ema50     = close.ewm(span=50).mean()
    ema200    = close.ewm(span=200).mean()
    ema_bull  = ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
    consec_up = (close.diff().iloc[-5:] > 0).all()
    return near_top and volume_dry and vol_spike and ema_bull and consec_up


def is_momentum_breakout(df):
    if df is None or len(df) < 50:
        return False
    close, volume = df["close"], df["volume"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if close.iloc[-1] <= sma20.iloc[-1] or close.iloc[-1] <= sma50.iloc[-1]:
        return False
    if sma20.iloc[-1] <= sma50.iloc[-1]:
        return False
    mom_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0
    if mom_5d <= 5:
        return False
    if volume.iloc[-1] < 1.5 * volume.iloc[-21:-1].mean():
        return False
    if close.iloc[-1] < df["high"].rolling(20).max().iloc[-1]:
        return False
    rsi = compute_rsi(close)
    return 60 <= rsi.iloc[-1] <= 80


def is_buzzing(df):
    if df is None or len(df) < 20:
        return False
    latest        = df.iloc[-1]
    close, open_  = latest["close"], latest["open"]
    volume        = latest["volume"]
    if close <= open_:
        return False
    if (close - open_) / open_ * 100 < 3.0:
        return False
    if volume < 1.5 * df["volume"].iloc[-21:-1].mean():
        return False
    return compute_rsi(df["close"]).iloc[-1] >= 60


def find_buzzing_stocks(symbols, limit=20, workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_test, s, is_buzzing, 30) for s in symbols[:500]]
        for fut in as_completed(futures):
            sym, passed = fut.result()
            if passed:
                df = fetch_historical(sym, days=30)
                if df is not None and len(df) >= 2:
                    close, open_ = df["close"].iloc[-1], df["open"].iloc[-1]
                    chg   = (close - open_) / open_ * 100
                    vol_r = df["volume"].iloc[-1] / df["volume"].iloc[-21:-1].mean()
                    rsi   = compute_rsi(df["close"]).iloc[-1]
                    results.append((sym, chg, vol_r, rsi))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


# ── Screener.in scrapers ───────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def scrape_screener_shareholding(symbol):
    url     = f"https://www.screener.in/company/{symbol.upper()}/"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
    try:
        resp = requests.Session().get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, None, None, None, f"HTTP {resp.status_code}"
        soup  = BeautifulSoup(resp.text, "html.parser")
        sh    = soup.find("section", id="shareholding")
        table = sh.find("table", class_="data-table") if sh and sh.name != "table" else sh
        if not table:
            return None, None, None, None, "No shareholding table."
        thead = table.find("thead")
        if not thead:
            return None, None, None, None, "No header row."
        quarters = [th.text.strip() for th in thead.find_all("th")][1:]
        p_data, f_data, d_data = [], [], []
        for row in table.find_all("tr"):
            cells  = row.find_all("td")
            if len(cells) < 2:
                continue
            label  = cells[0].text.strip()
            values = []
            for cell in cells[1:]:
                try:
                    values.append(float(cell.text.strip().replace("%", "")))
                except Exception:
                    values.append(None)
            if "Promoter" in label:                         p_data = values
            elif "Foreign" in label or "FII" in label:     f_data = values
            elif "Domestic" in label or "DII" in label:    d_data = values
        if not any([p_data, f_data, d_data]):
            return None, None, None, None, "Could not parse shareholding."
        return quarters, p_data, f_data, d_data, None
    except Exception as e:
        return None, None, None, None, str(e)


@st.cache_data(ttl=43200)
def scrape_screener_concall(symbol):
    url = f"https://www.screener.in/company/{symbol.upper()}/"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}"
        soup  = BeautifulSoup(resp.text, "html.parser")
        sec   = soup.find("section", id="concall")
        if not sec:
            return "No concall transcript found."
        paras = sec.find_all("p")
        text  = " ".join(p.get_text(strip=True) for p in paras) if paras else sec.get_text(" ", strip=True)
        return text[:1500] if len(text) >= 50 else "Concall too short or unavailable."
    except Exception as e:
        return f"Error: {e}"


@st.cache_data(ttl=1800)
def get_moneycontrol_premarket():
    url  = "https://www.moneycontrol.com/pre-market/"
    hdrs = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
    try:
        resp = requests.Session().get(url, headers=hdrs, timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        div  = soup.find("div", class_="premarket_data") or soup.find("div", {"id": "premarket"})
        if div:
            return div.get_text("\n", strip=True)[:3000], None
        kws = ["S&P", "Dow", "Nasdaq", "Gift Nifty", "Gold", "Crude", "Asian", "pre-market"]
        relevant = [
            l.strip() for l in soup.get_text("\n").split("\n")
            if any(k in l for k in kws)
        ]
        return ("\n".join(relevant[:30]) or None), (None if relevant else "No data")
    except Exception as e:
        return None, str(e)


# ── Verdict/verdict badge helper ──────────────────────────────────────────────
def _verdict_badge(direction):
    label = {1: "BUY", -1: "SELL", 0: "HOLD"}[direction]
    css   = {1: "buy",  -1: "sell",  0: "hold"}[direction]
    return label, css


def get_stock_change_kite(symbol):
    try:
        df = fetch_historical(symbol, days=5)
        if df is not None and len(df) >= 2:
            return (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
    except Exception:
        pass
    return None


def macro_regime_allocation():
    try:
        def _pct(h):
            c = h["Close"] if "Close" in h.columns else h.iloc[:, 3]
            return (float(c.iloc[-1]) - float(c.iloc[0])) / float(c.iloc[0])

        def _chg(h):
            c = h["Close"] if "Close" in h.columns else h.iloc[:, 3]
            return float(c.iloc[-1]) - float(c.iloc[0])

        nifty_r = _pct(yf.Ticker("^NSEI").history(period="1mo"))
        gold_r  = _pct(yf.Ticker("GC=F").history(period="1mo"))
        yld_chg = _chg(yf.Ticker("^TNX").history(period="1mo"))
        score   = nifty_r - gold_r - yld_chg / 100
        if score > 0.02:
            return "🟢 Risk ON – favour equities", "Equities 70% | Gold 15% | Bonds 15%"
        if score < -0.02:
            return "🔴 Risk OFF – raise cash", "Cash 40% | Gold 30% | Bonds 20% | Equities 10%"
        return "🟡 Neutral – mixed signals", "Equities 40% | Gold 25% | Bonds 25% | Cash 10%"
    except Exception:
        return "⚠️ Regime unavailable", "N/A"


def generate_swot(symbol):
    try:
        info   = yf.Ticker(symbol + ".NS").info
        sector = info.get("sector", "N/A")
        pe     = info.get("trailingPE", "N/A")
        roe    = info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else "N/A"
        mc_cr  = info.get("marketCap", 0) / 1e7
    except Exception:
        sector = pe = roe = mc_cr = "N/A"
    return call_deepseek(
        f"SWOT for {symbol} (NSE). Sector:{sector} P/E:{pe} ROE:{roe} MCap:₹{mc_cr}Cr.\n"
        "Format: **Strengths** / **Weaknesses** / **Opportunities** / **Threats** — 2-3 bullet points each.",
        system="You are an equity research analyst. Output ONLY bullet points.",
    )


def swarm_consensus(symbol, news_text):
    personas = [
        ("🟢 Bullish Trader",   "Aggressive growth, follows momentum."),
        ("🔴 Pension Fund",      "Conservative, dividends only, avoids volatility."),
        ("📊 Quant Chartist",    "Price patterns, MAs, volume. Ignores fundamentals."),
        ("💼 Value Investor",    "P/E, P/B, ROE. Buys below intrinsic value."),
        ("🌍 Macro Hedge Fund",  "Global cues, rates, currency, sector rotation."),
    ]
    results = []
    for name, style in personas:
        resp    = call_deepseek(
            f"You are {name}. {style}\nNews about {symbol}: \"{news_text}\"\n"
            "Output ONE line: 'VERDICT: BUY/SELL/HOLD (confidence: XX)' then one-line REASON.",
            system="Professional investor. Two lines only.",
        )
        verdict, confidence, reason = "HOLD", 50, "—"
        if resp:
            for line in resp.split("\n"):
                if "VERDICT:" in line.upper():
                    parts   = line.split("VERDICT:")[1].strip().split()
                    verdict = parts[0].upper() if parts else "HOLD"
                    try:
                        confidence = int(line.split("confidence:")[1].strip().split()[0])
                    except Exception:
                        pass
                elif "REASON:" in line.upper():
                    reason = line.split("REASON:")[1].strip()
        results.append({"persona": name, "verdict": verdict, "confidence": confidence, "reason": reason})
    return results


def generate_auto_pulse():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return "DeepSeek API key missing."
    indices = get_indices_data()
    nifty   = indices.get("Nifty 50", {"price": 0, "change": 0})
    sp500   = get_global_indices().get("S&P 500", {"price": 0, "change": 0})
    lcs     = get_symbols_by_market_cap("Largecap", 30)
    moves   = [(s, get_stock_change_kite(s)) for s in lcs]
    moves   = [(s, c) for s, c in moves if c is not None]
    gainers = sorted(moves, key=lambda x: x[1], reverse=True)[:3]
    losers  = sorted(moves, key=lambda x: x[1])[:3]
    prompt  = (
        f"Date: {datetime.now().strftime('%d-%m-%Y')}\n"
        f"Nifty 50: {nifty['price']:.1f} ({nifty['change']:+.2f}%)\n"
        f"Top Gainers: {', '.join(f'{s}({c:+.2f}%)' for s, c in gainers)}\n"
        f"Top Losers:  {', '.join(f'{s}({c:+.2f}%)' for s, c in losers)}\n"
        f"S&P 500: {sp500['price']:.1f} ({sp500['change']:+.2f}%)\n"
        "Write Daily Street Pulse as bullet points: Market Overview, Top Gainers/Losers, Global Cues, Technical Outlook."
    )
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 800,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" not in data:
            return "Error generating pulse."
        return data["choices"][0]["message"]["content"]
    except Exception:
        return "Error generating pulse."


def get_stock_verdict(symbol):
    df = fetch_historical(symbol, days=250)
    if df is None or len(df) < 50:
        return {"error": f"Insufficient data for {symbol}"}
    indicators        = ie.compute(df, symbol)
    conviction_result = ConvictionScorer().score(indicators)
    _dir = {"BUY": 1, "SELL": -1, "HOLD": 0}[conviction_result.verdict]
    _comp = conviction_result.components
    _attribution = {
        "factor":   (_comp.get("trend", 0.5) + _comp.get("rsi", 0.5) + _comp.get("momentum", 0.5)) / 3,
        "ml":       _comp.get("ml", 0.5),
        "regime":   _comp.get("regime", 0.5),
        "volume":   _comp.get("volume", 0.5),
        "combined": conviction_result.score / 100 * _dir,
    }
    latest_price = df["close"].iloc[-1]
    last_date    = df.index[-1].strftime("%Y-%m-%d")
    try:
        info = yf.Ticker(symbol + ".NS").info
        fundamentals = {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "ROE (%)":   info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else "N/A",
        }
    except Exception:
        fundamentals = {"P/E Ratio": "N/A", "ROE (%)": "N/A"}

    past      = get_recent_memory(symbol)
    past_text = (
        "\n".join(
            f"- {p['date']}: {p['decision']} (conf {p['confidence']:.0f}%) → {p.get('actual_return','pending')}%"
            for p in past
        )
        if past else "- No past trades."
    )
    rsi = indicators.get("rsi_14", 50)
    zsc = indicators.get("zscore_20", 0)
    mom = indicators.get("momentum_5d_pct", 0)
    vol = indicators.get("volume_ratio", 1)
    debate = call_deepseek(
        f"You are a trading debate moderator. Data for {symbol} (₹{latest_price}):\n"
        f"- RSI: {rsi:.1f}  Z-Score: {zsc:.2f}  5d-momentum: {mom:.2f}%  Volume: {vol:.2f}x\n"
        f"- P/E: {fundamentals['P/E Ratio']}  ROE: {fundamentals['ROE (%)']}%\n\n"
        "Format as bullet points only:\n**Bull Case** - 2 points\n**Bear Case** - 2 points\n"
        "**Verdict** - BUY/SELL/HOLD (confidence: XX) + one-line reason",
        system="You are a professional trading debate moderator. Use ONLY bullet points.",
    )
    return {
        "price":       latest_price,
        "signal":      conviction_result.score / 100 * _dir,
        "direction":   _dir,
        "confidence":  conviction_result.score,
        "attribution": _attribution,
        "conviction":  conviction_result,
        "indicators":  indicators,
        "df":          df,
        "last_date":   last_date,
        "debate":      debate,
        "past_memory": past_text,
        "fundamentals": fundamentals,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:.75rem 0 .5rem'>"
        "<span style='font-size:.6rem;color:#4a5568;text-transform:uppercase;"
        "letter-spacing:.15em;font-family:JetBrains Mono,monospace'>AlgoTrading Terminal</span><br>"
        "<span style='font-size:1.5rem;color:#00d4ff;font-weight:800;"
        "letter-spacing:.04em;font-family:JetBrains Mono,monospace'>⚡ QUANTTERM</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Navigation ─────────────────────────────────────────────────────────
    _nav_page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "⚡  Terminal", "🔬  Research", "🧬  AlgoLab", "🛠️  Tools"],
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    _page = _nav_page.split("  ", 1)[-1].strip()  # "Dashboard", "Terminal", …

    st.divider()

    # ── Status strip ───────────────────────────────────────────────────────
    _ms, _mc = market_status()
    _paper = os.getenv("SQ_PAPER_TRADING", "true").lower() == "true"
    _ds_key_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
    st.markdown(
        f"<div style='background:{_mc}22;border:1px solid {_mc}55;border-radius:8px;"
        f"padding:.35rem .75rem;font-size:.78rem;font-weight:600;"
        f"font-family:JetBrains Mono,monospace;color:{_mc}'>{_ms}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='display:flex;gap:.5rem;margin-top:.4rem;flex-wrap:wrap'>"
        f"<span style='font-size:.65rem;color:#8892a4'>🕐 {datetime.now().strftime('%H:%M')}</span>"
        f"<span style='font-size:.65rem;color:{'#4a5568' if _paper else '#00d4a0'}'>{'📄 Paper' if _paper else '💸 Live'}</span>"
        f"<span style='font-size:.65rem;color:{'#00d4a0' if _ds_key_ok else '#ff4466'}'>{'🟢 DeepSeek' if _ds_key_ok else '🔴 No API Key'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Universe (collapsed by default) ───────────────────────────────────
    with st.expander("📋 Universe", expanded=False):
        _default_universe = "\n".join(settings.symbol_list)
        _universe_input = st.text_area(
            "Symbols (one per line)",
            value=_default_universe,
            height=100,
            key="universe_input",
            label_visibility="collapsed",
        )
        universe = [s.strip().upper() for s in _universe_input.split("\n") if s.strip()]
        if not universe:
            universe = settings.symbol_list
        _paper_toggle = st.toggle("📄 Paper Trading", value=_paper, key="paper_trading_toggle")

    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    # ── Auto-refresh ───────────────────────────────────────────────────────
    _auto = st.checkbox("⏱ Auto-refresh (30s)", key="auto_refresh_enabled")
    if _auto:
        try:
            from streamlit_autorefresh import st_autorefresh
            _ticks = st_autorefresh(interval=30_000, key="quantterm_autorefresh")
            if _ticks and _ticks != st.session_state.get("_last_auto_tick"):
                st.session_state["_last_auto_tick"] = _ticks
                st.cache_data.clear()
        except Exception:
            pass

    st.divider()

    # ── Co-Pilot ───────────────────────────────────────────────────────────
    render_copilot_sidebar(context={
        "symbol": st.session_state.get("last_selected", ""),
        "indicators": st.session_state.get("last_verdict", {}).get("indicators", {}),
    })


# ── Shared symbol data (built once) ──────────────────────────────────────────
symbol_map  = get_all_equity_symbols()
symbol_list = sorted(symbol_map.keys())

# ── Route to active page ──────────────────────────────────────────────────────
if "sidebar_nav" not in st.session_state:
    st.session_state["sidebar_nav"] = "🏠  Dashboard"
_page = st.session_state.get("sidebar_nav", "🏠  Dashboard").split("  ", 1)[-1].strip()

# Handle session-state navigation from watchlist / homepage buttons
if st.session_state.get("active_tab") == "terminal":
    st.session_state["sidebar_nav"] = "⚡  Terminal"
    st.session_state.pop("active_tab", None)
    _page = "Terminal"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if _page == "Dashboard":
    render_homepage(universe)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "Terminal":
    # ── Top bar: symbol picker + timeframe ───────────────────────────────
    _tb_c1, _tb_c2, _tb_c3, _tb_c4 = st.columns([3, 2, 1, 1])
    with _tb_c1:
        _chart_sym = st.selectbox(
            "Symbol",
            options=symbol_list,
            index=symbol_list.index(st.session_state.get("terminal_symbol", "RELIANCE"))
                  if st.session_state.get("terminal_symbol", "RELIANCE") in symbol_list else 0,
            format_func=lambda x: f"{x}  ·  {symbol_map.get(x, '')}",
            key="terminal_chart_sym",
            label_visibility="collapsed",
        )
    with _tb_c2:
        _tf = st.radio(
            "Timeframe", ["5m", "15m", "1h", "1D"],
            horizontal=True, key="terminal_tf", label_visibility="collapsed",
        )
    with _tb_c3:
        _load_chart = st.button("📈 Load Chart", key="terminal_chart_load",
                                type="primary", width='stretch')
    with _tb_c4:
        _run_analysis = st.button("🤖 Analyse", key="terminal_run_analysis",
                                  width='stretch')

    # ── Two-panel layout: chart (65%) + signal panel (35%) ───────────────
    col_chart, col_signal = st.columns([65, 35])

    with col_chart:
        if _load_chart or st.session_state.get("terminal_chart_loaded_sym") == _chart_sym:
            _PERIOD_MAP = {"5m": ("5d", "5m"), "15m": ("5d", "15m"),
                           "1h": ("1mo", "60m"), "1D": ("1y", "1d")}
            _period, _interval = _PERIOD_MAP.get(_tf, ("3mo", "1d"))
            with st.spinner(f"Fetching {_chart_sym}…"):
                try:
                    from charting.engine import SmartChart as _SC
                    _df = yf.download(f"{_chart_sym}.NS", period=_period, interval=_interval,
                                      auto_adjust=True, progress=False)
                    if _df.empty:
                        st.error(f"No data for {_chart_sym}.NS")
                    else:
                        if isinstance(_df.columns, pd.MultiIndex):
                            _df.columns = [c[0].lower() for c in _df.columns]
                        else:
                            _df.columns = [c.lower() for c in _df.columns]
                        _df = _df[["open", "high", "low", "close", "volume"]].dropna()
                        _fig = _SC().build(_df, symbol=_chart_sym, show_vp=False)
                        st.plotly_chart(_fig, width='stretch', key="terminal_main_chart")
                        st.session_state["terminal_chart_loaded_sym"] = _chart_sym
                except Exception as _ce:
                    st.error(f"Chart error: {_ce}")
        else:
            st.markdown(
                "<div style='height:460px;display:flex;align-items:center;justify-content:center;"
                "background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);"
                "border-radius:12px;color:#4a5568;font-size:.9rem;flex-direction:column;gap:.5rem'>"
                "<span style='font-size:2rem'>📈</span>"
                "Select a symbol and click <b style='color:#00d4ff'>Load Chart</b>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Open positions strip ──────────────────────────────────────────
        init_db()
        _op = get_open_positions()
        if not _op.empty:
            st.markdown(
                "<div style='margin-top:.75rem;font-size:.7rem;color:#8892a4;"
                "text-transform:uppercase;letter-spacing:.08em'>Open Positions</div>",
                unsafe_allow_html=True,
            )
            _pos_cols = st.columns(min(len(_op), 4))
            for _pi, (_, _row) in enumerate(_op.iterrows()):
                with _pos_cols[_pi % 4]:
                    _dc = "#00d4a0" if _row["direction"] == "BUY" else "#ff4466"
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.03);border:1px solid {_dc}33;"
                        f"border-left:3px solid {_dc};border-radius:8px;padding:.4rem .6rem'>"
                        f"<span style='font-size:.82rem;font-weight:700;color:#e8eaf0'>{_row['symbol']}</span>"
                        f"<span style='font-size:.72rem;color:#8892a4;margin-left:.4rem'>@ ₹{_row['entry_price']:.0f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    with col_signal:
        # ── Run analysis ──────────────────────────────────────────────────
        if _run_analysis:
            with st.spinner(f"Analysing {_chart_sym}…"):
                _v = get_stock_verdict(_chart_sym)
            st.session_state["last_verdict"]  = _v
            st.session_state["last_selected"] = _chart_sym

        _v = st.session_state.get("last_verdict", {})
        _active_sym = st.session_state.get("last_selected", "")

        if _v and "error" not in _v and _active_sym == _chart_sym:
            _lbl, _css = _verdict_badge(_v["direction"])
            # Signal badge
            st.markdown(
                f"<div class='recommendation {_css}' style='font-size:1.3rem;"
                f"padding:.75rem;border-radius:10px;margin-bottom:.5rem'>"
                f"{_lbl}</div>",
                unsafe_allow_html=True,
            )
            st.progress(int(_v["confidence"]), text=f"Confidence: {_v['confidence']:.0f}%")

            # Price + indicators
            _ind = _v.get("indicators", {})
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("Price",    f"₹{_v.get('price', 0):,.0f}")
            _m2.metric("RSI",      f"{_ind.get('rsi_14', '—')}")
            _m3.metric("Momentum", f"{_ind.get('momentum_5d_pct', 0):.1f}%")

            with st.expander("Reasoning", expanded=True):
                st.caption(_v.get("debate") or "—")

            # DeepSeek signal
            if st.button("⚡ DeepSeek Signal", key="terminal_ds_final", width='stretch'):
                with st.spinner("Running V3 → R1 pipeline…"):
                    _sig = _get_svc().signal(
                        f"Symbol: {_chart_sym} | Price: ₹{_v['price']:.2f} | "
                        f"Conviction: {_v['confidence']:.0f}%\n{(_v.get('debate') or '')[:400]}",
                        _chart_sym,
                    )
                if _sig:
                    _, _scss = _verdict_badge({"BUY": 1, "SELL": -1, "HOLD": 0}.get(_sig.action, 0))
                    st.markdown(
                        f"{_get_svc().badge(_sig.llm_decision_maker)}<br>"
                        f"<div class='recommendation {_scss}' style='margin-top:.4rem'>"
                        f"{_sig.action}  ·  {_sig.confidence * 100:.0f}%</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(_sig.reasoning)

            st.divider()
            render_order_pad(_chart_sym, _v.get("price", 0.0), _ind)

        else:
            # Empty state
            st.markdown(
                "<div style='text-align:center;padding:2rem 1rem;color:#4a5568'>"
                "<div style='font-size:2.5rem;margin-bottom:.5rem'>🤖</div>"
                "<div style='font-size:.85rem'>Click <b style='color:#00d4ff'>Analyse</b><br>"
                "to run DeepSeek V3 → R1 signal</div></div>",
                unsafe_allow_html=True,
            )

        # ── Portfolio summary ─────────────────────────────────────────────
        st.divider()
        _summ = get_trading_summary()
        _pnl  = _summ.get("total_pnl", 0)
        _pc   = "#00d4a0" if _pnl >= 0 else "#ff4466"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);"
            f"border-radius:10px;padding:.6rem .9rem'>"
            f"<div><div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em'>Today P&L</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{_pc};font-family:JetBrains Mono,monospace'>₹{_pnl:,.0f}</div></div>"
            f"<div style='text-align:right'>"
            f"<div style='font-size:.72rem;color:#8892a4'>Win rate</div>"
            f"<div style='font-size:.9rem;font-weight:600;color:#e8eaf0'>{_summ.get('win_rate', 0):.0f}%</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🛑 Kill Switch", use_container_width=True, type="primary", key="terminal_kill"):
            st.error("⚠️ Kill switch activated.")
            st.session_state["kill_switch"] = True


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "Research":
    _r1, _r2, _r3, _r4, _r5, _r6, _r7 = st.tabs([
        "📈 Charts",
        "📊 Fundamentals",
        "📐 Multi-TF",
        "🔥 Heatmap",
        "🎯 Options",
        "🌊 FII/DII",
        "🎯 VCP Scanner",
    ])

    # ── Charts ─────────────────────────────────────────────────────────────
    with _r1:
        st.subheader("📈 Technical Analysis")
        _charts_sym = st.selectbox(
            "Symbol",
            symbol_list,
            format_func=lambda x: f"{x} – {symbol_map.get(x, x)}",
            key="research_charts_sym",
        )

        with st.expander("🕐 Multi-Timeframe Grid (8 views)", expanded=False):
            _mtf_sel = st.session_state.get("last_selected")
            if _mtf_sel:
                render_multi_tf_grid(_mtf_sel)
            else:
                st.info("Click **Analyze** in Terminal first to load multi-timeframe charts.")

        with st.expander("🔍 Anomaly Scanner", expanded=False):
            render_anomaly_scanner()

        st.divider()
        if st.button("Load Chart", key="research_chart_load"):
            with st.spinner(f"Fetching {_charts_sym}…"):
                _df_r = fetch_historical(_charts_sym, days=150)
            if _df_r is not None and len(_df_r) > 50:
                _ind_r = ie.compute(_df_r, _charts_sym)
                _df_r["EMA10"]  = _df_r["close"].ewm(span=10).mean()
                _df_r["EMA20"]  = _df_r["close"].ewm(span=20).mean()
                _df_r["EMA50"]  = _df_r["close"].ewm(span=50).mean()
                _df_r["EMA200"] = _df_r["close"].ewm(span=200).mean()

                _cn = _df_r["close"].iloc[-1]
                _e10, _e20, _e50, _e200 = (
                    _df_r["EMA10"].iloc[-1], _df_r["EMA20"].iloc[-1],
                    _df_r["EMA50"].iloc[-1], _df_r["EMA200"].iloc[-1],
                )
                _msgs = []
                if _cn > _e10 > _e20 > _e50 > _e200:
                    _msgs.append("🟢 Above all EMAs — strong uptrend.")
                elif _cn > _e10 and _cn > _e20:
                    _msgs.append("🟡 Above short-term EMAs — short-term bullish.")
                else:
                    _msgs.append("🔴 Below EMA10 — weak momentum.")
                if _e10 > _e20 > _e50:
                    _msgs.append("📈 Golden cross alignment (10>20>50).")
                elif _e10 < _e20 < _e50:
                    _msgs.append("📉 Death cross — trend is down.")
                _msgs.append(f"📊 {((_cn - _e200) / _e200 * 100):.1f}% from 200 EMA.")
                st.info("\n".join(_msgs))

                _fig_r = go.Figure()
                _fig_r.add_trace(go.Candlestick(
                    x=_df_r.index, open=_df_r["open"], high=_df_r["high"],
                    low=_df_r["low"], close=_df_r["close"], name="Price",
                ))
                for span, color, name in [(10, "#22c55e", "EMA10"), (20, "#eab308", "EMA20"),
                                           (50, "#a855f7", "EMA50"), (200, "#ef4444", "EMA200")]:
                    _fig_r.add_trace(go.Scatter(
                        x=_df_r.index, y=_df_r[f"EMA{span}"],
                        mode="lines", line=dict(color=color, width=1), name=name,
                    ))
                if st.checkbox("Show VWAP", key="research_vwap"):
                    _tp = (_df_r["high"] + _df_r["low"] + _df_r["close"]) / 3
                    _fig_r.add_trace(go.Scatter(
                        x=_df_r.index,
                        y=(_tp * _df_r["volume"]).cumsum() / _df_r["volume"].cumsum(),
                        mode="lines", line=dict(color="#3b82f6", width=1.5, dash="dot"),
                        name="VWAP",
                    ))
                _fig_r.update_layout(
                    title=f"{_charts_sym} — Candlestick + EMAs",
                    height=580, xaxis_title="Date", yaxis_title="Price (₹)",
                )
                st.plotly_chart(_fig_r, width='stretch')

                _rsi_v = _ind_r.get("rsi_14", 50)
                _zsc_v = _ind_r.get("zscore_20", 0)
                _mom_v = _ind_r.get("momentum_5d_pct", 0)
                _vol_v = _ind_r.get("volume_ratio", 1)
                _tp2   = (_df_r["high"] + _df_r["low"] + _df_r["close"]) / 3
                _vwap_v = ((_tp2 * _df_r["volume"]).cumsum() / _df_r["volume"].cumsum()).iloc[-1]
                st.dataframe(
                    pd.DataFrame({
                        "Indicator":  ["RSI(14)", "Z-Score(20)", "Momentum 5d", "Volume Ratio", "VWAP"],
                        "Value":      [f"{_rsi_v:.1f}", f"{_zsc_v:.2f}", f"{_mom_v:.2f}%", f"{_vol_v:.2f}x", f"₹{_vwap_v:.2f}"],
                        "Signal":     ["Oversold<30/OB>70", "Extreme<-2|>2", "Bullish>2%", "High>1.5x", "Below=discount"],
                    }),
                    hide_index=True,
                    width='stretch',
                )
            else:
                st.warning("Not enough data.")

        # Professional Charts (SmartChart)
        st.divider()
        st.subheader("📊 Professional Charts")
        _pc_col1, _pc_col2, _pc_col3 = st.columns([2, 2, 3])
        with _pc_col1:
            _pc_sym = st.text_input("Symbol", value="RELIANCE", key="pc_symbol").upper().strip()
        with _pc_col2:
            _pc_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5d", "1d"], index=1, key="pc_period")
        with _pc_col3:
            _pc_vp_bins = st.slider("Volume Profile bins", 20, 80, 40, 5, key="pc_vp_bins")
        _pc_show_vp = st.checkbox("Show Volume Profile sidebar", value=True, key="pc_show_vp")

        with st.expander("📖 Chart Legend", expanded=False):
            _l1, _l2, _l3 = st.columns(3)
            with _l1:
                st.markdown(
                    "**🟠 VWAP** — Volume-weighted fair price\n\n"
                    "**🔵 Bollinger Bands** — Price channels ±2σ\n\n"
                    "**🟣 Donchian Channels** — 20-period high/low"
                )
            with _l2:
                st.markdown(
                    "**🟡 POC** — Highest-volume price level\n\n"
                    "**🟢 Value Area** — 70% of all volume\n\n"
                    "**📊 Volume bars** — Green=up, Red=down"
                )
            with _l3:
                st.markdown(
                    "**RSI > 70** — Overbought\n\n"
                    "**RSI < 30** — Oversold\n\n"
                    "**MACD cross ↑** — Bullish shift"
                )

        with st.expander("❓ Explain an indicator", expanded=False):
            from charting.explanations import list_all as _pc_list_all, explain as _pc_explain
            _ind_list = _pc_list_all()
            _sel_ind = st.selectbox(
                "Choose indicator",
                _ind_list,
                format_func=lambda k: k.replace("_", " ").title(),
                key="pc_explain_select",
            )
            if st.button("Explain", key="pc_explain_btn"):
                st.session_state["pc_explain_text"] = _pc_explain(_sel_ind)
            if "pc_explain_text" in st.session_state:
                st.code(st.session_state["pc_explain_text"], language=None)

        if st.button("🔄 Load Charts", key="pc_load", type="primary") or st.session_state.get("pc_data_loaded"):
            _PERIOD_INTERVAL = {
                "1d": "5m", "5d": "30m",
                "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1d",
            }
            _pc_interval = _PERIOD_INTERVAL.get(_pc_period, "1d")
            _pc_ticker   = f"{_pc_sym}.NS"
            try:
                from charting.engine import SmartChart as _SmartChart
                from charting.footprint import FootprintAnalyzer as _FPA
                from charting.liquidity import LiquidityHeatmap as _LiqHM

                with st.spinner(f"Fetching {_pc_ticker}…"):
                    _pc_df = yf.download(
                        _pc_ticker, period=_pc_period, interval=_pc_interval,
                        auto_adjust=True, progress=False,
                    )
                if _pc_df.empty:
                    st.error(f"No data for {_pc_ticker}")
                else:
                    if isinstance(_pc_df.columns, pd.MultiIndex):
                        _pc_df.columns = [c[0].lower() for c in _pc_df.columns]
                    else:
                        _pc_df.columns = [c.lower() for c in _pc_df.columns]
                    _pc_df = _pc_df[["open", "high", "low", "close", "volume"]].dropna()
                    st.session_state["pc_data_loaded"] = True
                    st.caption(f"✅ {len(_pc_df)} bars · {_pc_sym} latest close: ₹{_pc_df['close'].iloc[-1]:,.2f}")

                    _pct1, _pct2, _pct3 = st.tabs([
                        "📈 Main Chart",
                        "🦶 Footprint & Order Flow",
                        "💧 Liquidity Heatmap",
                    ])
                    with _pct1:
                        st.plotly_chart(
                            _SmartChart().build(_pc_df, symbol=_pc_sym, show_vp=_pc_show_vp, vp_bins=_pc_vp_bins),
                            width='stretch',
                        )
                    with _pct2:
                        st.plotly_chart(_FPA().build_figure(_pc_df, symbol=_pc_sym), width='stretch')
                        st.info(
                            "⭐ Green star = ask imbalance (bullish pressure)  \n"
                            "⭐ Red star = bid imbalance (bearish pressure)  \n"
                            "⚠️ Simulated from OHLCV — real tick data requires Zerodha WebSocket."
                        )
                    with _pct3:
                        _cp  = float(_pc_df["close"].iloc[-1])
                        _book = _LiqHM().simulate_book(_cp)
                        st.plotly_chart(_LiqHM().build_figure(_book, symbol=_pc_sym), width='stretch')
                        st.info(
                            "🟢 Green bars = pending buy orders (support)  \n"
                            "🔴 Red bars = pending sell orders (resistance)  \n"
                            "⚠️ Simulated — live order book requires Zerodha WebSocket."
                        )
            except Exception as _pce:
                st.error(f"Chart error: {_pce}")
        else:
            st.info("Enter a symbol and click **Load Charts**.")

    # ── Fundamentals ───────────────────────────────────────────────────────
    with _r2:
        st.subheader("📊 Fundamental Deep Dive")
        _fund_tabs = st.tabs(["📋 Quick Fundamentals", "🔬 Deep Fundamentals (screener.in)", "📊 Ownership"])

        with _fund_tabs[0]:
            _f_sym = st.selectbox(
                "Symbol", symbol_list, key="fund_sym_r",
                format_func=lambda x: f"{x} – {symbol_map.get(x, x)}",
            )
            if st.button("Load Fundamentals", key="fund_load_r"):
                with st.spinner("Fetching…"):
                    _f_info = yf.Ticker(_f_sym + ".NS").info
                st.write(
                    f"**{_f_info.get('longName', _f_sym)}** | "
                    f"{_f_info.get('sector', 'N/A')} → {_f_info.get('industry', 'N/A')}"
                )
                st.write(_f_info.get("longBusinessSummary", "")[:500] + "…")
                _fc1, _fc2, _fc3, _fc4 = st.columns(4)
                _fc1.metric("P/E Ratio",  f"{_f_info.get('trailingPE', 'N/A')}")
                _fc2.metric("Market Cap", f"₹{(_f_info.get('marketCap', 0) / 1e9):.1f}B")
                _fc3.metric("ROE %",      f"{(_f_info.get('returnOnEquity', 0) or 0) * 100:.1f}%")
                _fc4.metric("Div Yield",  f"{(_f_info.get('dividendYield', 0) or 0) * 100:.2f}%")
                st.subheader("📞 Concall Summary")
                st.markdown(scrape_screener_concall(_f_sym))

            st.divider()
            st.subheader("🧮 Conviction Score")
            _dt_c1, _dt_c2 = st.columns([1, 2])
            with _dt_c1:
                _dt_profile = st.selectbox("Trader Profile", list(PROFILES.keys()), key="dt_p_r")
                _dt_capital = st.number_input("Capital (₹)", value=100_000, step=10_000, key="dt_cap_r")
                _dt_sym     = st.selectbox(
                    "Symbol", symbol_list, key="dt_sym_r",
                    format_func=lambda x: f"{x} – {symbol_map.get(x, x)}",
                )
                if st.button("▶ Compute", key="dt_go_r", width='stretch'):
                    with st.spinner("Computing…"):
                        _dt_df = fetch_historical(_dt_sym, days=100)
                    if _dt_df is not None and len(_dt_df) >= 30:
                        _dt_ind    = ie.compute(_dt_df, _dt_sym)
                        _dt_scorer = ConvictionScorer(PROFILES[_dt_profile])
                        _dt_res    = _dt_scorer.score(_dt_ind)
                        _dt_atr    = _dt_ind.get("atr_14", _dt_df["close"].iloc[-1] * 0.015)
                        _dt_price  = _dt_df["close"].iloc[-1]
                        _dt_setup  = (
                            compute_trade_setup(_dt_sym, _dt_res.verdict, _dt_price, _dt_atr, _dt_capital)
                            if _dt_res.verdict in ("BUY", "SELL") else None
                        )
                        st.session_state.update({
                            "dt_res_r": _dt_res, "dt_setup_r": _dt_setup,
                            "dt_price_r": _dt_price, "dt_ind_r": _dt_ind, "dt_sym_r_val": _dt_sym,
                        })
                    else:
                        st.warning("Not enough data.")

            if "dt_res_r" in st.session_state:
                with _dt_c2:
                    _r_r  = st.session_state["dt_res_r"]
                    _lbl_r, _css_r = _verdict_badge({"BUY": 1, "SELL": -1, "HOLD": 0}.get(_r_r.verdict, 0))
                    _m1, _m2, _m3 = st.columns(3)
                    _m1.metric("Conviction", f"{_r_r.score:.1f}/100")
                    _m2.markdown(
                        f"<div class='recommendation {_css_r}' style='font-size:1.1rem'>{_lbl_r}</div>",
                        unsafe_allow_html=True,
                    )
                    _m3.metric("Gates", "✅ Pass" if _r_r.gates_passed else f"❌ {len(_r_r.gate_failures)} failed")

                    if not _r_r.gates_passed:
                        for _gf in _r_r.gate_failures:
                            st.warning(f"Gate: {_gf}")

                    _s_r = st.session_state.get("dt_setup_r")
                    if _s_r:
                        st.markdown("#### 📐 ATR Trade Setup")
                        _s1, _s2, _s3, _s4, _s5 = st.columns(5)
                        _s1.metric("Entry",  f"₹{_s_r.entry}")
                        _s2.metric("Stop",   f"₹{_s_r.stop}")
                        _s3.metric("Target", f"₹{_s_r.target}")
                        _s4.metric("Qty",    str(_s_r.quantity))
                        _s5.metric("R:R",    f"{_s_r.rr_ratio}×")

                    _comp_r = _r_r.components
                    _fb = go.Figure(go.Bar(
                        x=list(_comp_r.values()), y=list(_comp_r.keys()),
                        orientation="h",
                        marker_color=["#22c55e" if v >= 0.5 else "#ef4444" for v in _comp_r.values()],
                    ))
                    _fb.update_layout(title="Conviction Components (0-1)", height=250,
                                      xaxis_range=[0, 1], margin=dict(t=30, b=10))
                    st.plotly_chart(_fb, width='stretch')

                    if st.button("🤖 DeepSeek Analysis", key="dt_ds_r", width='stretch'):
                        _ds_ctx = (
                            f"Symbol: {st.session_state.get('dt_sym_r_val')} | "
                            f"Price: ₹{st.session_state.get('dt_price_r', 0):.2f} | "
                            f"Conviction: {_r_r.score:.0f}/100 | Verdict: {_r_r.verdict}\n"
                            f"Components: {json.dumps({k: round(v, 2) for k, v in _comp_r.items()})}"
                        )
                        with st.spinner("Running DeepSeek…"):
                            _cop = _get_svc().signal(_ds_ctx, st.session_state.get("dt_sym_r_val", ""))
                        if _cop:
                            _, _ccss = _verdict_badge({"BUY": 1, "SELL": -1, "HOLD": 0}.get(_cop.action, 0))
                            st.markdown(
                                f"{_get_svc().badge(_cop.llm_decision_maker)} "
                                f"<div class='recommendation {_ccss}'>{_cop.action} · {_cop.confidence * 100:.0f}%</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Reasoning:** {_cop.reasoning}")

        with _fund_tabs[1]:
            st.subheader("🔬 Deep Fundamentals (screener.in)")
            st.caption("Key Ratios, P&L, Balance Sheet, Quarterly, Shareholding, Cash Flow. Cached 24h.")
            _df_sym = st.text_input("NSE Symbol", value="BEL", key="df_sym_r").upper().strip()
            _df_force = st.checkbox("Force refresh", key="df_force_r")
            if st.button("📥 Load", key="df_load_r") and _df_sym:
                with st.spinner(f"Fetching {_df_sym}…"):
                    try:
                        from fundamentals.fetcher import get_deep_fundamentals
                        _fd = get_deep_fundamentals(_df_sym, force_refresh=_df_force)
                        _about = _fd.get("about", "")
                        _meta  = _fd.get("metadata", {})
                        st.markdown(
                            f"**{_df_sym}** · "
                            f"{'Consolidated' if _meta.get('consolidated') else 'Standalone'} · "
                            f"[screener.in]({_fd.get('url', '#')})"
                        )
                        if _about:
                            with st.expander("About"):
                                st.write(_about)
                        _kr = _fd.get("key_ratios", [])
                        if _kr:
                            st.markdown("### 📊 Key Ratios")
                            _kr_cols = st.columns(min(len(_kr), 5))
                            for _i, _ratio in enumerate(_kr[:10]):
                                with _kr_cols[_i % 5]:
                                    st.metric(_ratio.get("name", ""), _ratio.get("value", "—"))
                        _sec_defs = [
                            ("📈 P&L", "profit_loss"), ("🏦 Balance Sheet", "balance_sheet"),
                            ("📅 Quarterly", "quarterly_results"), ("👥 Shareholding", "shareholding"),
                            ("💵 Cash Flow", "cash_flow"), ("🏢 Peers", "peer_comparison"),
                        ]
                        _fund_sec_tabs = st.tabs([s[0] for s in _sec_defs])
                        for _tab_obj, (_, _sk) in zip(_fund_sec_tabs, _sec_defs):
                            with _tab_obj:
                                _rows = _fd.get(_sk, [])
                                if not _rows:
                                    st.info("No data.")
                                else:
                                    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
                    except ValueError as _ve:
                        st.error(f"Symbol not found: {_ve}")
                    except Exception as _ex:
                        st.error(f"Failed: {_ex}")

        with _fund_tabs[2]:
            st.subheader("📊 Shareholding & Ownership")
            _own_sym = st.selectbox(
                "Symbol", symbol_list, key="own_sym",
                format_func=lambda x: f"{x} – {symbol_map.get(x, x)}",
            )
            if st.button("Load Ownership Data", key="own_load"):
                with st.spinner():
                    _info_o = yf.Ticker(_own_sym + ".NS").info
                st.write(
                    f"**{_info_o.get('longName', _own_sym)}** | "
                    f"{_info_o.get('sector', 'N/A')} → {_info_o.get('industry', 'N/A')}"
                )
                _qtr, _p, _fi, _di, _err = scrape_screener_shareholding(_own_sym)
                if _err:
                    st.warning(_err)
                else:
                    if _p:
                        st.metric("Promoter Holding", f"{_p[-1]:.2f}%")
                    if _qtr and _fi and _di:
                        _mn   = min(len(_qtr), len(_fi), len(_di))
                        _td   = pd.DataFrame({
                            "Quarter": _qtr[:_mn], "FII (%)": _fi[:_mn], "DII (%)": _di[:_mn],
                        }).dropna()
                        if not _td.empty:
                            _fig_fi = px.line(
                                _td, x="Quarter", y=["FII (%)", "DII (%)"],
                                markers=True,
                                color_discrete_map={"FII (%)": "orange", "DII (%)": "green"},
                            )
                            st.plotly_chart(_fig_fi, width='stretch')
                st.subheader("📞 Concall Summary")
                st.markdown(scrape_screener_concall(_own_sym))

    # ── Multi-TF ───────────────────────────────────────────────────────────
    with _r3:
        st.subheader("📐 Multi-Timeframe Signal Aligner")
        st.caption("Checks signal direction across 5-min, 15-min, 1-h, and daily timeframes.")
        _mtf_sym = st.selectbox("Symbol", options=symbol_list, key="mtf_sym_r")
        _mtf_tfs = st.multiselect(
            "Timeframes", ["5min", "15min", "1h", "1d"],
            default=["5min", "15min", "1h"], key="mtf_tfs_r",
        )
        if st.button("🔍 Run Multi-Timeframe Grid", key="mtf_run_r"):
            render_multi_tf_grid(_mtf_sym)
            with st.spinner("Running signal alignment…"):
                try:
                    from analysis.multi_timeframe import MultiTimeframeAligner
                    _aligner = MultiTimeframeAligner(fetcher=fetcher)
                    _result  = _aligner.align(_mtf_sym, _mtf_tfs)
                    _score   = _result.get("alignment_score", 0.0)
                    _cons    = _result.get("consensus_action", "HOLD")
                    _colour  = {"BUY": "#2e7d32", "SELL": "#c62828", "HOLD": "#e65100"}.get(_cons, "#555")
                    _ma, _mb = st.columns([1, 2])
                    with _ma:
                        st.markdown(
                            f"<div style='background:#fff;border-radius:16px;padding:1.5rem;"
                            f"text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);border:1px solid #eef2f6'>"
                            f"<p style='margin:0;font-size:.85rem;color:#666'>Alignment Score</p>"
                            f"<p style='margin:.25rem 0;font-size:2.5rem;font-weight:700;color:{_colour}'>{_score:+.2f}</p>"
                            f"<p style='margin:0;font-size:1rem;font-weight:600;color:{_colour}'>{_cons}</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with _mb:
                        _tf_data = _result.get("timeframes", {})
                        if _tf_data:
                            st.dataframe(
                                pd.DataFrame([
                                    {"Timeframe": _tf, "Action": v.get("action", "?"),
                                     "Confidence": f"{v.get('confidence', 0):.1%}"}
                                    for _tf, v in _tf_data.items()
                                ]),
                                hide_index=True, width='stretch',
                            )
                except Exception as _mte:
                    st.error(f"Error: {_mte}")

    # ── Heatmap ────────────────────────────────────────────────────────────
    with _r4:
        render_macro_strip()
        st.divider()
        render_heatmap()

    with _r5:
        render_options_page()

    # ── FII/DII ────────────────────────────────────────────────────────────
    with _r6:
        render_fii_dii_page()

    with _r7:
        render_vcp_page(universe)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALGOLAB
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "AlgoLab":
    _al1, _al2, _al3 = st.tabs([
        "💻 Strategy Editor",
        "🧪 Backtest",
        "📊 Walk-Forward",
    ])

    with _al1:
        render_algolab(fetcher=fetcher)

    with _al2:
        st.subheader("🧪 Backtest")
        render_backtest_bridge(
            st.session_state.get("last_selected", "RELIANCE"),
            fetcher=fetcher,
            ie=ie,
        )

    with _al3:
        st.subheader("📊 Walk-Forward Validator")
        _wf_sym = st.selectbox("Symbol", symbol_list, key="wf_sym",
                                format_func=lambda x: f"{x} – {symbol_map.get(x, x)}")
        if st.button("Run Walk-Forward", key="wf_run"):
            with st.spinner("Running walk-forward validation…"):
                try:
                    from backtest.walk_forward import WalkForwardValidator
                    _wfv    = WalkForwardValidator(fetcher=fetcher, ie=ie)
                    _wf_res = _wfv.run(_wf_sym)
                    if _wf_res.get("error"):
                        st.error(_wf_res["error"])
                    else:
                        _wf_c1, _wf_c2, _wf_c3 = st.columns(3)
                        _wf_c1.metric("In-Sample Sharpe",  f"{_wf_res.get('is_sharpe', 'N/A'):.2f}")
                        _wf_c2.metric("OOS Sharpe",         f"{_wf_res.get('oos_sharpe', 'N/A'):.2f}")
                        _wf_c3.metric("Degradation",        f"{_wf_res.get('degradation_pct', 'N/A'):.1f}%")
                        st.dataframe(
                            pd.DataFrame(_wf_res.get("folds", [])),
                            hide_index=True, width='stretch',
                        )
                except Exception as _wfe:
                    st.error(f"Walk-forward error: {_wfe}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TOOLS
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "Tools":
    _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9 = st.tabs([
        "📄 Daily Report",
        "📰 News",
        "🔔 Alerts",
        "📊 Screener",
        "📚 Journal",
        "🌍 Macro",
        "🤖 Agents",
        "🧠 Memory",
        "🎙️ Earnings",
        "📊 Signal Tracker",
    ])

    # ── Daily Street Pulse Report ──────────────────────────────────────────
    with _t0:
        st.markdown(
            "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
            "font-size:1.2rem;letter-spacing:2px;margin-bottom:.25rem'>"
            "📄 DAILY STREET PULSE</h2>"
            "<p style='color:#4a5568;font-size:.75rem;margin-bottom:1.2rem'>"
            "Auto-generated daily market report · DeepSeek V3 analysis · "
            "Nifty snapshot · Top movers · Breakout picks</p>",
            unsafe_allow_html=True,
        )

        _rp_col1, _rp_col2, _rp_col3 = st.columns([2, 2, 3])
        with _rp_col1:
            _gen_report = st.button(
                "⚡ Generate Today's Report",
                key="gen_daily_report",
                type="primary",
                width='stretch',
            )
        with _rp_col2:
            _rp_format = st.radio(
                "Format", ["PDF", "HTML"],
                horizontal=True, key="report_format",
            )
        with _rp_col3:
            st.caption(
                "Fetches live data from Kite/yfinance + runs DeepSeek analysis. "
                "Takes ~30-60 seconds."
            )

        if _gen_report:
            with st.spinner("Fetching market data and running DeepSeek analysis…"):
                try:
                    from reports.daily_pulse import build_report_data, generate_pdf, generate_html_bytes, render_html
                    _rdata = build_report_data()
                    st.success(f"Report generated for {_rdata['date']}")

                    if _rp_format == "PDF":
                        try:
                            _pdf_bytes = generate_pdf(_rdata)
                            st.download_button(
                                label="📥 Download PDF",
                                data=_pdf_bytes,
                                file_name=f"daily_street_pulse_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key="download_pdf_btn",
                            )
                        except Exception as _pdf_err:
                            st.warning(f"PDF generation failed ({_pdf_err}), downloading HTML instead.")
                            _html_bytes = generate_html_bytes(_rdata)
                            st.download_button(
                                label="📥 Download HTML",
                                data=_html_bytes,
                                file_name=f"daily_street_pulse_{datetime.now().strftime('%Y%m%d')}.html",
                                mime="text/html",
                                key="download_html_fallback_btn",
                            )
                    else:
                        _html_bytes = generate_html_bytes(_rdata)
                        st.download_button(
                            label="📥 Download HTML",
                            data=_html_bytes,
                            file_name=f"daily_street_pulse_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                            key="download_html_btn",
                        )

                    # Preview in app
                    with st.expander("👁 Preview Report", expanded=True):
                        st.components.v1.html(render_html(_rdata), height=900, scrolling=True)

                except Exception as _re:
                    st.error(f"Report generation failed: {_re}")


    # ── News ───────────────────────────────────────────────────────────────
    with _t1:
        render_news_feed()

    # ── Alerts ─────────────────────────────────────────────────────────────
    with _t2:
        _al_sub1, _al_sub2 = st.tabs(["🔔 Telegram Alerts", "📥 Alert Inbox"])
        with _al_sub1:
            render_alerts_page()
        with _al_sub2:
            render_alert_inbox()

    # ── Screener ───────────────────────────────────────────────────────────
    with _t3:
        _sc_tabs = st.tabs(["🚀 Full NSE Screener", "⚡ Quick Screener", "🔗 Correlation", "🌅 Pre-Market", "🌍 Global"])

        with _sc_tabs[0]:
            st.subheader("🔎 NSE Stock Screener")
            st.caption("Filter ~2000 stocks by fundamentals, technicals, and ensemble ML signal.")
            with st.expander("📐 Fundamental Filters", expanded=True):
                _sf1, _sf2, _sf3 = st.columns(3)
                with _sf1:
                    _sc_pe_max  = st.number_input("P/E ≤",           min_value=0.0, max_value=500.0, value=0.0, step=1.0,   key="sc_pe_max")
                    _sc_roe_min = st.number_input("ROE ≥ %",         min_value=0.0, max_value=100.0, value=0.0, step=1.0,   key="sc_roe_min")
                with _sf2:
                    _sc_debt_max = st.number_input("Debt/Equity ≤",   min_value=0.0, max_value=50.0,  value=0.0, step=0.1,  key="sc_debt_max")
                    _sc_mcap_min = st.number_input("Market Cap ≥ ₹Cr",min_value=0.0, max_value=1e7,   value=0.0, step=100.0, key="sc_mcap_min")
                with _sf3:
                    _sc_prom_min = st.number_input("Promoter Holding ≥ %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_prom_min")
                    _sc_div_min  = st.number_input("Dividend Yield ≥ %",   min_value=0.0, max_value=20.0,  value=0.0, step=0.1, key="sc_div_min")
            with st.expander("📊 Technical Filters", expanded=True):
                _st1, _st2, _st3 = st.columns(3)
                with _st1:
                    _sc_rsi_max = st.number_input("RSI ≤ (oversold)",  min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_max")
                    _sc_rsi_min = st.number_input("RSI ≥ (overbought)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_min")
                with _st2:
                    _sc_vol_spike = st.number_input("Volume spike ≥ ×", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="sc_vol_spike")
                    _sc_above_sma = st.selectbox("Price above SMA", [None, 20, 50], key="sc_above_sma")
                with _st3:
                    _sc_below_sma = st.selectbox("Price below SMA", [None, 20, 50], key="sc_below_sma")
                    _sc_signal    = st.selectbox("Ensemble signal",  [None, "BUY", "SELL", "HOLD"], key="sc_signal")
            _sc_lim_c, _sc_scrape_c = st.columns([3, 2])
            with _sc_lim_c:
                _sc_limit = st.slider("Max results", 5, 200, 50, 5, key="sc_limit")
            with _sc_scrape_c:
                _sc_scrape = st.checkbox("Scrape missing fundamentals (slow)", value=False, key="sc_scrape")

            if st.button("🚀 Run Screener", key="sc_run", type="primary"):
                def _nz(v):
                    return v if v and v > 0 else None
                _sc_filters = dict(
                    pe_max=_nz(_sc_pe_max), roe_min=_nz(_sc_roe_min), debt_max=_nz(_sc_debt_max),
                    market_cap_min_cr=_nz(_sc_mcap_min), promoter_holding_min=_nz(_sc_prom_min),
                    dividend_yield_min=_nz(_sc_div_min), rsi_max=_nz(_sc_rsi_max), rsi_min=_nz(_sc_rsi_min),
                    volume_spike_min=_nz(_sc_vol_spike), price_above_sma_days=_sc_above_sma,
                    price_below_sma_days=_sc_below_sma, ensemble_signal=_sc_signal,
                    limit=_sc_limit, scrape_missing_fundamentals=_sc_scrape,
                )
                _active = {k: v for k, v in _sc_filters.items() if v is not None and v is not False}
                if not _active:
                    st.warning("Set at least one filter before running.")
                else:
                    _t0_sc   = time.time()
                    _prog_sc = st.progress(0, text="Loading NSE universe…")
                    try:
                        from screener.engine import ScreenerEngine as _SE
                        _prog_sc.progress(20, text="Updating technicals cache…")
                        _sc_df = _SE().screen_by_ratios(**_sc_filters)
                        _prog_sc.progress(100, text="Done!")
                        _elapsed = round(time.time() - _t0_sc, 1)
                        if _sc_df.empty:
                            st.info("No stocks match the current filters.")
                        else:
                            st.success(f"Found **{len(_sc_df)}** stocks in {_elapsed}s")
                            st.dataframe(_sc_df, width='stretch')
                            st.download_button(
                                "⬇️ Download CSV",
                                data=_sc_df.to_csv(index=False).encode(),
                                file_name="screener_results.csv",
                                mime="text/csv",
                            )
                    except Exception as _sce:
                        _prog_sc.empty()
                        st.error(f"Screener error: {_sce}")

        with _sc_tabs[1]:
            st.subheader("⚡ Quick Screener")
            st.caption("Momentum & breakout scanner with fundamental filters.")
            with st.expander("📐 Fundamental Filters", expanded=True):
                _qsf1, _qsf2, _qsf3 = st.columns(3)
                with _qsf1:
                    _qsc_pe_max  = st.number_input("P/E ≤",           min_value=0.0, max_value=500.0, value=0.0, step=1.0,  key="sc_pe_max2")
                    _qsc_roe_min = st.number_input("ROE ≥ %",         min_value=0.0, max_value=100.0, value=0.0, step=1.0,  key="sc_roe_min2")
                with _qsf2:
                    _qsc_debt_max = st.number_input("Debt/Equity ≤",  min_value=0.0, max_value=50.0,  value=0.0, step=0.1,  key="sc_debt_max2")
                    _qsc_mcap_min = st.number_input("Market Cap ≥ ₹Cr",min_value=0.0,max_value=1e7,   value=0.0, step=100.0, key="sc_mcap_min2")
                with _qsf3:
                    _qsc_prom_min = st.number_input("Promoter Holding ≥ %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_prom_min2")
                    _qsc_div_min  = st.number_input("Dividend Yield ≥ %",   min_value=0.0, max_value=20.0,  value=0.0, step=0.1, key="sc_div_min2")
            with st.expander("📊 Technical Filters", expanded=True):
                _qst1, _qst2, _qst3 = st.columns(3)
                with _qst1:
                    _qsc_rsi_max  = st.number_input("RSI ≤", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_max2")
                    _qsc_rsi_min2 = st.number_input("RSI ≥", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_min2")
                with _qst2:
                    _qsc_vol_spike2 = st.number_input("Volume spike ≥ ×", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="sc_vol_spike2")
                    _qsc_above_sma2 = st.checkbox("Above 50-day SMA", key="sc_above_sma2")
                with _qst3:
                    _qsc_ml_min2 = st.number_input("ML conviction ≥", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="sc_ml_min2")
            if st.button("🔎 Run Quick Screener", key="screener_run2", width='stretch'):
                with st.spinner("Screening…"):
                    try:
                        from screener.engine import ScreenerEngine
                        _qsc_eng = ScreenerEngine()
                        _qsc_res, _qsc_err = _qsc_eng.run(
                            pe_max=_qsc_pe_max or None, roe_min=_qsc_roe_min or None,
                            debt_equity_max=_qsc_debt_max or None, mcap_min_cr=_qsc_mcap_min or None,
                            promoter_min=_qsc_prom_min or None, div_yield_min=_qsc_div_min or None,
                            rsi_max=_qsc_rsi_max or None, rsi_min=_qsc_rsi_min2 or None,
                            vol_spike_min=_qsc_vol_spike2 or None, above_sma50=_qsc_above_sma2 or None,
                            conviction_min=_qsc_ml_min2 or None,
                        )
                        if _qsc_err:
                            st.error(f"Screener error: {_qsc_err}")
                        elif _qsc_res is not None and not _qsc_res.empty:
                            st.success(f"Found {len(_qsc_res)} stocks")
                            st.dataframe(_qsc_res, use_container_width=True, hide_index=True)
                        else:
                            st.info("No stocks matched.")
                    except Exception as _qsce:
                        st.error(f"Screener unavailable: {_qsce}")

            st.divider()
            st.subheader("🤖 Daily Pulse")
            if st.button("Generate Daily Pulse", key="pulse_btn2"):
                with st.spinner("Generating…"):
                    _pulse = generate_auto_pulse()
                st.markdown(f"<div class='devbloom-card'>{_pulse}</div>", unsafe_allow_html=True)

            st.subheader("📊 Buzzing Stocks")
            if st.button("Find Buzzing Stocks", key="buzz_btn2"):
                _syms = list(get_all_equity_symbols().keys())
                with st.spinner("Scanning for buzz…"):
                    _buzz = find_buzzing_stocks(_syms, limit=20)
                if _buzz:
                    st.dataframe(pd.DataFrame(_buzz, columns=["Symbol", "Change%", "Vol Ratio", "RSI"]),
                                 width='stretch')
                else:
                    st.info("No buzzing stocks found.")

        with _sc_tabs[2]:
            st.subheader("🔗 Correlation Matrix Heatmap")
            st.caption("Pairwise return correlations. Pairs >0.7 highlighted.")
            _corr_syms = st.multiselect(
                "Select up to 20 symbols",
                options=symbol_list,
                default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"][:min(5, len(symbol_list))],
                max_selections=20, key="corr_syms",
            )
            _corr_days = st.slider("Lookback days", 20, 252, 60, key="corr_days")
            if st.button("📊 Compute Correlations", key="corr_run") and _corr_syms:
                with st.spinner("Computing…"):
                    try:
                        from analysis.correlation import CorrelationAnalyzer
                        _ca      = CorrelationAnalyzer(fetcher=fetcher)
                        _summary = _ca.summary(_corr_syms, lookback_days=_corr_days)
                        if _summary.get("error"):
                            st.warning(f"Error: {_summary['error']}")
                        else:
                            _corr_df = pd.DataFrame(_summary["matrix"])
                            _cfig = px.imshow(
                                _corr_df, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                                title=f"Return Correlation — {_corr_days}d", text_auto=".2f",
                            )
                            _cfig.update_layout(height=500)
                            st.plotly_chart(_cfig, width='stretch')
                            _hp = _summary.get("high_corr_pairs", [])
                            if _hp:
                                st.markdown("**Highly correlated pairs (>0.7):**")
                                st.dataframe(pd.DataFrame(_hp), hide_index=True, width='stretch')
                    except Exception as _coe:
                        st.error(f"Error: {_coe}")

        with _sc_tabs[3]:
            st.header("🌅 Pre-Market Report")
            if st.button("Fetch Moneycontrol"):
                with st.spinner():
                    _report, _err = get_moneycontrol_premarket()
                if _report:
                    st.markdown(f"<div class='card'>{_report}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Failed: {_err}")
                    for _n, _v in get_global_indices().items():
                        st.write(f"{_n}: {_v['price']:.1f} ({_v['change']:+.2f}%)")

        with _sc_tabs[4]:
            st.subheader("🌍 Global Indices")
            if st.button("Load Global Indices", key="gl_load"):
                _gl = get_global_indices()
                if _gl:
                    _gc = st.columns(3)
                    for _i, (_name, _val) in enumerate(_gl.items()):
                        _gc[_i % 3].metric(
                            _name, f"{_val['price']:.1f}", f"{_val['change']:+.2f}%",
                            delta_color="normal" if _val["change"] >= 0 else "inverse",
                        )
                else:
                    st.info("Unavailable.")

    # ── Journal ────────────────────────────────────────────────────────────
    with _t4:
        render_journal()

    # ── Macro ──────────────────────────────────────────────────────────────
    with _t5:
        st.subheader("🌍 Macro Dashboard")
        render_macro_strip()
        st.divider()
        _mac_l, _mac_r = st.columns([3, 2])
        with _mac_l:
            render_heatmap()
        with _mac_r:
            render_macro_dashboard()

        st.divider()
        st.subheader("🌡️ Market Regime Detector")
        if st.button("🔄 Detect Current Regime", key="regime_run"):
            with st.spinner("Analysing market regime…"):
                try:
                    from analysis.regime_detector import RegimeDetector
                    _rd = RegimeDetector(fetcher=fetcher)
                    _rg = _rd.detect()
                    _rg_regime = _rg.get("regime", "UNKNOWN")
                    _rg_colour = (
                        "#c62828" if "HIGH_VOL" in _rg_regime else
                        "#2e7d32" if "LOW_VOL" in _rg_regime else "#e65100"
                    )
                    st.markdown(
                        f"<div style='background:{_rg_colour}11;border:2px solid {_rg_colour};"
                        f"border-radius:16px;padding:1.25rem;text-align:center;margin-bottom:1rem'>"
                        f"<span style='font-size:1.4rem;font-weight:700;color:{_rg_colour}'>{_rg_regime}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if _rg.get("expiry_warning"):
                        st.warning("⚠️ F&O expiry within 3 days — expect elevated volatility.")
                    _rc1, _rc2, _rc3 = st.columns(3)
                    _rc1.metric("Hurst Exponent", f"{_rg.get('hurst') or 'N/A'}")
                    _rc2.metric("ADX",             f"{_rg.get('adx') or 'N/A'}")
                    _rc3.metric("India VIX",       f"{_rg.get('india_vix') or 'N/A'}")
                    st.info(f"**Recommended strategy:** {_rg.get('recommended_strategy', '')}")
                except Exception as _rge:
                    st.error(f"Error: {_rge}")

        st.divider()
        st.subheader("🏦 Quant Hedge Fund Models")
        _qhf_sym = st.selectbox("Symbol", symbol_list, key="qhf_sym",
                                 format_func=lambda x: f"{x} – {symbol_map.get(x, x)}")
        if st.button("🚀 Run All Quant Models", key="qhf_run"):
            with st.spinner("Fetching & computing…"):
                _qhf_df = fetch_historical(_qhf_sym, days=300)
            if _qhf_df is None or len(_qhf_df) < 60:
                st.error("Need at least 60 days of data.")
            else:
                _close = _qhf_df["close"]
                if len(_close) >= 200:
                    _ret  = (_close.iloc[-1] - _close.iloc[-200]) / _close.iloc[-200]
                    _dv   = _close.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)
                    _msig = float(np.clip(_ret / _dv, -1, 1)) if _dv else 0.0
                    _mv   = "🟢 BUY" if _msig > 0.3 else ("🔴 SELL" if _msig < -0.3 else "⚪ HOLD")
                else:
                    _msig, _mv = 0.0, "Insufficient (need 200d)"
                _rv   = _close.pct_change().dropna().iloc[-60:].std() * np.sqrt(252)
                _mult = float(np.clip(0.20 / _rv, 0.2, 3.0)) if _rv else 1.0
                _vv   = "📈 Increase size" if _mult > 1.2 else ("📉 Reduce size" if _mult < 0.8 else "✅ Normal size")
                _regime, _alloc = macro_regime_allocation()
                st.markdown(
                    f"- **Momentum Signal:** `{_msig:.2f}` → {_mv}\n"
                    f"- **Vol Targeting Mult:** `{_mult:.2f}x` → {_vv}\n"
                    f"- **Macro Regime:** {_regime} — {_alloc}"
                )

        st.divider()
        st.subheader("🔮 What-If Trade Simulator")
        with st.form("whatif_form"):
            _wi1, _wi2 = st.columns(2)
            with _wi1:
                _wi_sym  = st.selectbox("Symbol", options=symbol_list, key="wi_sym_sel")
                _wi_qty  = st.number_input("Quantity", min_value=1, value=100, step=1)
            with _wi2:
                _wi_price = st.number_input("Entry Price (₹)", min_value=1.0, value=1000.0, step=1.0)
                _wi_days  = st.number_input("Holding Days",    min_value=1, max_value=60, value=5, step=1)
            _wi_submit = st.form_submit_button("🔮 Simulate")
        if _wi_submit:
            with st.spinner("Running simulation…"):
                try:
                    from simulator.whatif import WhatIfSimulator
                    _wi_r = WhatIfSimulator(fetcher=fetcher).simulate(_wi_sym, _wi_qty, _wi_price, _wi_days)
                    if _wi_r.get("error"):
                        st.warning(_wi_r["error"])
                    else:
                        _w1, _w2, _w3, _w4 = st.columns(4)
                        _w1.metric("Prob. Profit",   f"{_wi_r.get('prob_profit', 0):.0%}")
                        _w2.metric("Prob. Loss >2%", f"{_wi_r.get('prob_loss_gt_2pct', 0):.0%}")
                        _w3.metric("Expected Return",f"{_wi_r.get('expected_return_pct', 0):.2f}%")
                        _w4.metric("99% VaR",        f"₹{_wi_r.get('var_99', 0):,.0f}")
                        st.warning("⚠️ Past distributions don't guarantee future results.")
                except Exception as _wie:
                    st.error(f"Error: {_wie}")

        st.divider()
        st.subheader("⚠️ Risk Metrics")
        _rm_sym = st.selectbox("Symbol", options=symbol_list, key="rm_sym",
                                format_func=lambda x: f"{x} – {symbol_map.get(x, x)}")
        if st.button("📊 Compute Risk Metrics", key="rm_run"):
            with st.spinner("Computing…"):
                try:
                    from analytics.risk_metrics import RiskMetrics
                    _rm_m = RiskMetrics(fetcher=fetcher).compute(_rm_sym)
                    if _rm_m.get("error"):
                        st.warning(_rm_m["error"])
                    else:
                        _rs = _rm_m.get("risk_score", 5)
                        _sc = "#c62828" if _rs >= 7 else "#e65100" if _rs >= 5 else "#2e7d32"
                        st.markdown(
                            f"<div style='background:{_sc}11;border:2px solid {_sc};border-radius:12px;"
                            f"padding:1rem;text-align:center;margin-bottom:1rem'>"
                            f"<span style='font-size:.9rem;color:#666'>Risk Score</span><br>"
                            f"<span style='font-size:2.5rem;font-weight:700;color:{_sc}'>{_rs}/10</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            pd.DataFrame([
                                ("Annualised Return",    f"{_rm_m.get('annualised_return_pct', 'N/A')}%"),
                                ("Annualised Volatility",f"{_rm_m.get('annualised_volatility_pct', 'N/A')}%"),
                                ("Sharpe Ratio",         _rm_m.get("sharpe_ratio", "N/A")),
                                ("Sortino Ratio",        _rm_m.get("sortino_ratio", "N/A")),
                                ("Calmar Ratio",         _rm_m.get("calmar_ratio", "N/A")),
                                ("Max Drawdown",         f"{_rm_m.get('max_drawdown_pct', 'N/A')}%"),
                                ("VaR 95%",              _rm_m.get("var_95", "N/A")),
                                ("VaR 99%",              _rm_m.get("var_99", "N/A")),
                                ("CVaR 95%",             _rm_m.get("cvar_95", "N/A")),
                                ("Beta vs Nifty",        _rm_m.get("beta", "N/A")),
                            ], columns=["Metric", "Value"]),
                            hide_index=True, width='stretch',
                        )
                except Exception as _rme:
                    st.error(f"Error: {_rme}")

    # ── Agents ─────────────────────────────────────────────────────────────
    with _t6:
        _ag_tabs = st.tabs(["🤖 Multi-Agent", "🔎 Decision Terminal", "🧠 Ensemble ML", "⚡ Co-Pilot"])

        with _ag_tabs[0]:
            render_agent_dashboard()

        with _ag_tabs[1]:
            st.subheader("🔎 Decision Terminal — Swarm Intelligence")
            st.caption("5 investor personas debate the news and reach a consensus.")
            _sw_sym = st.selectbox("Symbol", symbol_list, key="sw_sym",
                                   format_func=lambda x: f"{x} – {symbol_map.get(x, x)}")
            _news_input = st.text_area("News / event", height=80, placeholder="HDFC Life profit up 4%…")
            if st.button("Run Swarm", key="sw_run"):
                if not _news_input:
                    st.warning("Enter some news.")
                else:
                    with st.spinner("Asking 5 personas…"):
                        _sw_results = swarm_consensus(_sw_sym, _news_input)
                    for _sw_r in _sw_results:
                        _ca, _cb = st.columns([3, 7])
                        with _ca:
                            _, _sw_css = _verdict_badge({"BUY": 1, "SELL": -1, "HOLD": 0}.get(_sw_r["verdict"], 0))
                            st.markdown(f"**{_sw_r['persona']}**")
                            st.markdown(
                                f"<div class='recommendation {_sw_css}' style='font-size:1rem;padding:.4rem'>"
                                f"{_sw_r['verdict']} {_sw_r['confidence']}%</div>",
                                unsafe_allow_html=True,
                            )
                        with _cb:
                            st.caption(_sw_r["reason"])
                    _buy_ = sum(1 for r in _sw_results if r["verdict"] == "BUY")
                    _sell_ = sum(1 for r in _sw_results if r["verdict"] == "SELL")
                    _hold_ = sum(1 for r in _sw_results if r["verdict"] == "HOLD")
                    st.info(f"Consensus → BUY: {_buy_} | SELL: {_sell_} | HOLD: {_hold_}")

        with _ag_tabs[2]:
            st.subheader("🧠 Ensemble ML Signal")
            st.caption("XGBoost + LightGBM (+ optional Chronos-Bolt) weighted-vote signal.")
            _ens_sym = st.selectbox("Symbol", options=symbol_list, key="ens_sym",
                                    format_func=lambda x: f"{x} – {symbol_map.get(x, x)}")
            if st.button("🤖 Generate Ensemble Signal", key="ens_run"):
                with st.spinner("Training / loading models…"):
                    try:
                        from ml.ensemble_signal import EnsembleSignalGenerator
                        _df_ens = fetch_historical(_ens_sym, days=400)
                        if _df_ens is None or _df_ens.empty:
                            st.warning("No data available.")
                        else:
                            _sig = EnsembleSignalGenerator().generate_signal(_df_ens, _ens_sym)
                            _act = _sig.get("action", "HOLD")
                            _conf = _sig.get("confidence", 0)
                            _ec = {"BUY": "#2e7d32", "SELL": "#c62828", "HOLD": "#e65100"}.get(_act, "#555")
                            _ea, _eb = st.columns(2)
                            _ea.markdown(
                                f"<div class='recommendation {_act.lower()}'>{_act}</div>"
                                f"<p style='text-align:center;margin-top:.5rem'>Confidence: <b>{_conf:.1%}</b></p>",
                                unsafe_allow_html=True,
                            )
                            with _eb:
                                st.write("**Reasoning:**", _sig.get("reasoning", ""))
                                _details = _sig.get("ensemble_details", {})
                                if _details:
                                    st.dataframe(
                                        pd.DataFrame([
                                            {"Model": m, "Action": v.get("action", "?"),
                                             "Confidence": f"{v.get('confidence', 0):.1%}", "Weight": v.get("weight", "")}
                                            for m, v in _details.items() if isinstance(v, dict)
                                        ]),
                                        hide_index=True,
                                    )
                    except Exception as _ene:
                        st.error(f"Error: {_ene}")

        with _ag_tabs[3]:
            render_copilot_inline(context={
                "symbol":     st.session_state.get("last_selected", ""),
                "price":      st.session_state.get("last_verdict", {}).get("price", 0),
                "indicators": st.session_state.get("last_verdict", {}).get("indicators", {}),
            })

    # ── Memory ─────────────────────────────────────────────────────────────
    with _t7:
        render_memory_vault()

    # ── Earnings ───────────────────────────────────────────────────────────
    with _t8:
        render_earnings_page()

    # ── Signal Tracker ─────────────────────────────────────────────────────
    with _t9:
        from ui.signal_tracker_page import render_signal_tracker
        render_signal_tracker()


# ── Paper trading dashboard (accessible from any tab via sidebar session) ────
# Initialise DB on every load so paper_trading tables exist
init_db()

# ── Preserve devbloom_chat session state key for history compatibility ────────
if "devbloom_chat" not in st.session_state:
    st.session_state["devbloom_chat"] = []
