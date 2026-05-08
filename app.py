import os
import json
import time
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

from data.historical import HistoricalDataFetcher
from data.instruments import InstrumentManager
from data.kite_client import KiteClient
from features.indicators import IndicatorEngine
from features.market_structure import is_recent_swing_breakout
from features.volume_profile import VolumeProfile
from llm.claude_client import ClaudeClient
from paper_trading import (
    close_position,
    get_closed_positions,
    get_equity_curve,
    get_open_positions,
    get_trading_summary,
    init_db,
    open_position,
)
from sq_ai.signals.conviction import ConvictionScorer
from sq_ai.signals.profiles import PROFILES
from sq_ai.signals.trade_setup import compute_trade_setup

load_dotenv()

# ── DevBloom UI modules ───────────────────────────────────────────────────────
from ui.theme import DEVBLOOM_CSS, COMMAND_PALETTE_JS
from ui.heatmap import render_heatmap, render_macro_strip
from ui.watchlist import render_watchlist
from ui.alert_inbox import render_alert_inbox
from ui.macro import render_macro_dashboard
from ui.copilot import render_copilot_sidebar, render_copilot_inline
from ai.dual_llm_service import get_service as _get_svc
from ui.order_pad import render_order_pad, render_position_monitor, render_equity_curve, render_backtest_bridge
from ui.algolab import render_algolab
from ui.journal import render_journal, log_trade_to_journal
from ui.anomaly_scanner import render_anomaly_scanner
from charting.multi_tf import render_multi_tf_grid

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevBloom Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DEVBLOOM_CSS, unsafe_allow_html=True)
try:
    st.components.v1.html(COMMAND_PALETTE_JS, height=1)
except Exception:
    pass

# ── Cached resource init ───────────────────────────────────────────────────────
@st.cache_resource
def init_clients():
    kite    = KiteClient()
    im      = InstrumentManager()
    fetcher = HistoricalDataFetcher(kite, im)
    ie      = IndicatorEngine()
    vp      = VolumeProfile()
    claude  = ClaudeClient()
    return kite, im, fetcher, ie, vp, claude

kite, im, fetcher, ie, vp, _claude = init_clients()

# ── Symbol universe ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_all_equity_symbols():
    out = {}
    for sym, meta in im._meta_map.items():
        ex, seg, it = meta.get('exchange',''), meta.get('segment',''), meta.get('instrument_type','')
        if ex in ('NSE','BSE') and seg == ex and it == 'EQ':
            if not sym[0].isdigit() and '-' not in sym and len(sym) <= 10:
                out[sym] = meta.get('companyName', sym)
    return out or {"RELIANCE":"Reliance Industries","TCS":"Tata Consultancy","HDFCBANK":"HDFC Bank"}

# ── Historical OHLCV (5-min cache) ────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_historical(symbol, days=250):
    # Try Kite first
    to_d   = datetime.now().strftime("%Y-%m-%d")
    from_d = (datetime.now() - timedelta(days=days+30)).strftime("%Y-%m-%d")
    df = fetcher.fetch(symbol, from_d, to_d, interval="day")
    if df is None or len(df) == 0:
        from_d = (datetime.now() - timedelta(days=days+100)).strftime("%Y-%m-%d")
        df = fetcher.fetch(symbol, from_d, to_d, interval="day")

    # Fallback to yfinance when Kite is unavailable
    if df is None or len(df) == 0:
        try:
            ticker = symbol if symbol.endswith(".NS") or symbol.startswith("^") else symbol + ".NS"
            raw = yf.download(ticker, period=f"{days}d", interval="1d",
                              progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0].lower() for c in raw.columns]
                else:
                    raw.columns = [c.lower() for c in raw.columns]
                df = raw[["open","high","low","close","volume"]].dropna()
        except Exception:
            pass
    return df

# ── Index data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_indices_data():
    names = {"Nifty 50":"NIFTY 50","Bank Nifty":"NIFTY BANK",
             "Nifty IT":"NIFTY IT","Nifty Pharma":"NIFTY PHARMA","Nifty FMCG":"NIFTY FMCG"}
    out = {}
    for name, sym in names.items():
        try:
            df = fetch_historical(sym, days=5)
            if df is not None and len(df) >= 2:
                last, prev = df['close'].iloc[-1], df['close'].iloc[-2]
                out[name] = {"price": last, "change": (last-prev)/prev*100}
            elif df is not None and len(df) == 1:
                out[name] = {"price": df['close'].iloc[-1], "change": 0.0}
        except Exception:
            continue
    return out

@st.cache_data(ttl=3600)
def get_global_indices():
    tickers = {"S&P 500":"^GSPC","Dow Jones":"^DJI","Nasdaq":"^IXIC",
               "FTSE 100":"^FTSE","DAX":"^GDAXI","Nikkei 225":"^N225",
               "Hang Seng":"^HSI","Shanghai":"000001.SS"}
    out = {}
    for name, t in tickers.items():
        try:
            h = yf.Ticker(t).history(period="2d")
            if len(h) >= 2:
                _hc = h["Close"] if "Close" in h.columns else h[h.columns[3]]
                last, prev = float(_hc.iloc[-1]), float(_hc.iloc[-2])
                out[name] = {"price": last, "change": (last-prev)/prev*100}
        except Exception:
            continue
    return out

# ── Market cap helpers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_market_cap(symbol):
    try:
        info = yf.Ticker(symbol+".NS").info
        mc = info.get('marketCap', 0)
        return mc/1e7 if mc else 0
    except Exception:
        return 0

def categorize_by_mcap(symbol):
    mc = get_market_cap(symbol)
    if mc >= 20000: return "Largecap"  # noqa: E701,E702,E741
    if mc >= 5000:  return "Midcap"  # noqa: E701,E702,E741
    if mc > 0:      return "Smallcap"  # noqa: E701,E702,E741
    return "Unknown"

def get_symbols_by_market_cap(cap_filter, max_symbols=500):
    all_syms = list(get_all_equity_symbols().keys())
    if cap_filter == "All":
        return all_syms[:max_symbols]
    out = []
    for sym in all_syms:
        mc = get_market_cap(sym)
        if   mc >= 20000 and cap_filter == "Largecap": out.append(sym)  # noqa: E701,E702,E741
        elif 5000 <= mc < 20000 and cap_filter == "Midcap": out.append(sym)  # noqa: E701,E702,E741
        elif 0 < mc < 5000 and cap_filter == "Smallcap": out.append(sym)  # noqa: E701,E702,E741
        if len(out) >= max_symbols: break  # noqa: E701,E702,E741
    return out

# ── Market status ──────────────────────────────────────────────────────────────
def market_status():
    now = datetime.now()
    t   = now.time()
    ot  = datetime.strptime("09:15","%H:%M").time()
    ct  = datetime.strptime("15:30","%H:%M").time()
    if now.weekday() >= 5:
        return "🔴 Closed (Weekend)", "#fee2e2"
    if t < ot:
        mins = int((datetime.combine(now.date(), ot)-now).total_seconds()//60)
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
    with open(_MEMORY_FILE, 'w') as f:
        json.dump(mem, f, indent=2)

def update_memory(symbol, decision, confidence, price):
    mem = load_memory()
    mem.setdefault(symbol, []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "decision": decision, "confidence": confidence,
        "price": price, "actual_return": "pending"
    })
    mem[symbol] = mem[symbol][-10:]
    save_memory(mem)

def get_recent_memory(symbol, limit=3):
    return load_memory().get(symbol, [])[-limit:]

# ── DeepSeek helper ────────────────────────────────────────────────────────────
def call_deepseek(prompt, system="You are a financial analyst."):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model":"deepseek-chat",
                  "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
                  "temperature":0.3, "max_tokens":800},
            timeout=30
        )
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# ── Core verdict engine ────────────────────────────────────────────────────────
def get_stock_verdict(symbol):
    df = fetch_historical(symbol, days=250)
    if df is None or len(df) < 50:
        return {"error": f"Insufficient data for {symbol}"}
    indicators        = ie.compute(df, symbol)
    conviction_result = ConvictionScorer().score(indicators)
    _dir              = {"BUY": 1, "SELL": -1, "HOLD": 0}[conviction_result.verdict]
    _comp             = conviction_result.components
    _attribution      = {
        "factor":   (_comp.get("trend", 0.5) + _comp.get("rsi", 0.5) + _comp.get("momentum", 0.5)) / 3,
        "ml":       _comp.get("ml", 0.5),
        "regime":   _comp.get("regime", 0.5),
        "volume":   _comp.get("volume", 0.5),
        "combined": conviction_result.score / 100 * _dir,
    }

    latest_price = df['close'].iloc[-1]
    last_date    = df.index[-1].strftime('%Y-%m-%d')

    try:
        info = yf.Ticker(symbol+".NS").info
        fundamentals = {
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "ROE (%)":   info.get('returnOnEquity', 0)*100 if info.get('returnOnEquity') else 'N/A'
        }
    except Exception:
        fundamentals = {"P/E Ratio": "N/A", "ROE (%)": "N/A"}

    past = get_recent_memory(symbol)
    past_text = "\n".join(
        [f"- {p['date']}: {p['decision']} (conf {p['confidence']:.0f}%) → {p.get('actual_return','pending')}%"
         for p in past]
    ) if past else "- No past trades."

    # DeepSeek bull/bear debate
    rsi = indicators.get('rsi_14', 50)
    zsc = indicators.get('zscore_20', 0)
    mom = indicators.get('momentum_5d_pct', 0)
    vol = indicators.get('volume_ratio', 1)
    debate = call_deepseek(f"""
You are a trading debate moderator. Data for {symbol} (₹{latest_price}):
- RSI: {rsi:.1f}  Z-Score: {zsc:.2f}  5d-momentum: {mom:.2f}%  Volume: {vol:.2f}x
- P/E: {fundamentals['P/E Ratio']}  ROE: {fundamentals['ROE (%)']}%

Format as bullet points only:
**Bull Case** - 2 points
**Bear Case**  - 2 points
**Verdict** - BUY/SELL/HOLD (confidence: XX) + one-line reason
""", system="You are a professional trading debate moderator. Use ONLY bullet points.")

    return {
        "price":      latest_price,
        "signal":     conviction_result.score / 100 * _dir,
        "direction":  _dir,
        "confidence": conviction_result.score,
        "attribution": _attribution,
        "conviction": conviction_result,
        "indicators": indicators,
        "df":         df,
        "last_date":  last_date,
        "debate":     debate,
        "past_memory":past_text,
        "fundamentals": fundamentals,
    }

# ── Scanner helpers ────────────────────────────────────────────────────────────
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.where(delta>0, 0).rolling(period).mean()
    loss  = (-delta.where(delta<0, 0)).rolling(period).mean()
    return 100 - (100/(1+gain/loss))

def fetch_and_test(symbol, test_func, days=150):
    try:
        df = fetch_historical(symbol, days=days)
        if df is not None and len(df) >= days-20:
            return symbol, test_func(df)
    except Exception:
        pass
    return symbol, False

def is_breakout_candidate(df):
    if df is None or len(df) < 120: return False  # noqa: E701,E702,E741
    close = df['close']
    base_high = close.rolling(90).max().iloc[-1]
    base_low  = close.rolling(90).min().iloc[-1]
    if (base_high - base_low) == 0: return False  # noqa: E701,E702,E741
    near_top = (close.iloc[-1]-base_low)/(base_high-base_low) > 0.8
    volume_dry = df['volume'].iloc[-20:].mean() < 0.7*df['volume'].rolling(120).mean().iloc[-1]
    if not is_recent_swing_breakout(df, lookback=3): return False  # noqa: E701,E702,E741
    vol_spike = df['volume'].iloc[-1] > 1.5*df['volume'].rolling(20).mean().iloc[-1]
    ema10  = close.ewm(span=10).mean()
    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    ema_bull = ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
    consec_up = (close.diff().iloc[-5:] > 0).all()
    return near_top and volume_dry and vol_spike and ema_bull and consec_up

def is_momentum_breakout(df):
    if df is None or len(df) < 50: return False  # noqa: E701,E702,E741
    close, volume = df['close'], df['volume']
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if close.iloc[-1] <= sma20.iloc[-1] or close.iloc[-1] <= sma50.iloc[-1]: return False  # noqa: E701,E702,E741
    if sma20.iloc[-1] <= sma50.iloc[-1]: return False  # noqa: E701,E702,E741
    mom_5d = (close.iloc[-1]-close.iloc[-6])/close.iloc[-6]*100 if len(close)>=6 else 0
    if mom_5d <= 5: return False  # noqa: E701,E702,E741
    if volume.iloc[-1] < 1.5*volume.iloc[-21:-1].mean(): return False  # noqa: E701,E702,E741
    if close.iloc[-1] < df['high'].rolling(20).max().iloc[-1]: return False  # noqa: E701,E702,E741
    rsi = compute_rsi(close)
    return 60 <= rsi.iloc[-1] <= 80

def is_buzzing(df):
    if df is None or len(df) < 20: return False  # noqa: E701,E702,E741
    latest = df.iloc[-1]
    close, open_, volume = latest['close'], latest['open'], latest['volume']
    if close <= open_: return False  # noqa: E701,E702,E741
    if (close-open_)/open_*100 < 3.0: return False  # noqa: E701,E702,E741
    if volume < 1.5*df['volume'].iloc[-21:-1].mean(): return False  # noqa: E701,E702,E741
    return compute_rsi(df['close']).iloc[-1] >= 60

def scan_parallel(symbols, test_func, days, cap_filter, limit=500, workers=8):
    categorized = {"Largecap":[], "Midcap":[], "Smallcap":[], "Unknown":[]}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_test, s, test_func, days) for s in symbols[:limit]]
        for fut in as_completed(futures):
            sym, passed = fut.result()
            if passed:
                cat = categorize_by_mcap(sym)
                if cat in cap_filter or "All" in cap_filter:
                    categorized[cat].append(sym)
    return categorized

def find_buzzing_stocks(symbols, limit=20, workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_test, s, is_buzzing, 30) for s in symbols[:500]]
        for fut in as_completed(futures):
            sym, passed = fut.result()
            if passed:
                df = fetch_historical(sym, days=30)
                if df is not None and len(df) >= 2:
                    close, open_ = df['close'].iloc[-1], df['open'].iloc[-1]
                    chg   = (close-open_)/open_*100
                    vol_r = df['volume'].iloc[-1]/df['volume'].iloc[-21:-1].mean()
                    rsi   = compute_rsi(df['close']).iloc[-1]
                    results.append((sym, chg, vol_r, rsi))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]

# ── Screener.in scrapers ───────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def scrape_screener_shareholding(symbol):
    url = f"https://www.screener.in/company/{symbol.upper()}/"
    headers = {"User-Agent":"Mozilla/5.0","Accept":"text/html"}
    try:
        resp = requests.Session().get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None,None,None,None,f"HTTP {resp.status_code}"
        soup  = BeautifulSoup(resp.text, 'html.parser')
        sh    = soup.find('section', id='shareholding')
        table = sh.find('table', class_='data-table') if sh and sh.name != 'table' else sh
        if not table:
            return None,None,None,None,"No shareholding table."
        thead = table.find('thead')
        if not thead:
            return None,None,None,None,"No header row."
        quarters = [th.text.strip() for th in thead.find_all('th')][1:]
        p_data, f_data, d_data = [], [], []
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 2: continue  # noqa: E701,E702,E741
            label  = cells[0].text.strip()
            values = []
            for cell in cells[1:]:
                try:   values.append(float(cell.text.strip().replace('%','')))  # noqa: E701,E702,E741
                except Exception: values.append(None)  # noqa: E701,E702,E741
            if 'Promoter' in label: p_data = values  # noqa: E701,E702,E741
            elif 'Foreign' in label or 'FII' in label: f_data = values  # noqa: E701,E702,E741
            elif 'Domestic' in label or 'DII' in label: d_data = values  # noqa: E701,E702,E741
        if not any([p_data, f_data, d_data]):
            return None,None,None,None,"Could not parse shareholding."
        return quarters, p_data, f_data, d_data, None
    except Exception as e:
        return None,None,None,None,str(e)

@st.cache_data(ttl=43200)
def scrape_screener_concall(symbol):
    url = f"https://www.screener.in/company/{symbol.upper()}/"
    try:
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, 'html.parser')
        sec  = soup.find('section', id='concall')
        if not sec:
            return "No concall transcript found."
        paras = sec.find_all('p')
        text  = ' '.join(p.get_text(strip=True) for p in paras) if paras else sec.get_text(' ', strip=True)
        return text[:1500] if len(text) >= 50 else "Concall too short or unavailable."
    except Exception as e:
        return f"Error: {e}"

@st.cache_data(ttl=1800)
def get_moneycontrol_premarket():
    url  = "https://www.moneycontrol.com/pre-market/"
    hdrs = {"User-Agent":"Mozilla/5.0","Accept":"text/html"}
    try:
        resp = requests.Session().get(url, headers=hdrs, timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, 'html.parser')
        div  = soup.find('div', class_='premarket_data') or soup.find('div', {'id':'premarket'})
        if div:
            return div.get_text('\n', strip=True)[:3000], None
        kws = ['S&P','Dow','Nasdaq','Gift Nifty','Gold','Crude','Asian','pre-market']
        relevant = [l.strip() for l in soup.get_text('\n').split('\n')  # noqa: E701,E702,E741
                    if any(k in l for k in kws)]
        return ('\n'.join(relevant[:30]) or None), (None if relevant else "No data")
    except Exception as e:
        return None, str(e)

# ── DeepSeek market pulse ──────────────────────────────────────────────────────
def generate_auto_pulse():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return "DeepSeek API key missing."  # noqa: E701,E702,E741
    indices = get_indices_data()
    nifty   = indices.get("Nifty 50", {"price":0,"change":0})
    sp500   = get_global_indices().get("S&P 500", {"price":0,"change":0})
    lcs     = get_symbols_by_market_cap("Largecap", 30)
    moves   = [(s, get_stock_change_kite(s)) for s in lcs]
    moves   = [(s,c) for s,c in moves if c is not None]
    gainers = sorted(moves, key=lambda x: x[1], reverse=True)[:3]
    losers  = sorted(moves, key=lambda x: x[1])[:3]
    prompt  = f"""Date: {datetime.now().strftime('%d-%m-%Y')}
Nifty 50: {nifty['price']:.1f} ({nifty['change']:+.2f}%)
Top Gainers: {', '.join(f"{s}({c:+.2f}%)" for s,c in gainers)}
Top Losers:  {', '.join(f"{s}({c:+.2f}%)" for s,c in losers)}
S&P 500: {sp500['price']:.1f} ({sp500['change']:+.2f}%)
Write Daily Street Pulse as bullet points: Market Overview, Top Gainers/Losers, Global Cues, Technical Outlook."""
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
                  "temperature":0.4,"max_tokens":800},
            timeout=30
        )
        return resp.json()['choices'][0]['message']['content']
    except Exception:
        return "Error generating pulse."

def get_stock_change_kite(symbol):
    try:
        df = fetch_historical(symbol, days=5)
        if df is not None and len(df) >= 2:
            return (df['close'].iloc[-1]-df['close'].iloc[-2])/df['close'].iloc[-2]*100
    except Exception:
        pass
    return None

def macro_regime_allocation():
    try:
        def _pct(h): c = h["Close"] if "Close" in h.columns else h.iloc[:, 3]; return (float(c.iloc[-1])-float(c.iloc[0]))/float(c.iloc[0])
        def _chg(h): c = h["Close"] if "Close" in h.columns else h.iloc[:, 3]; return float(c.iloc[-1])-float(c.iloc[0])
        nifty_r = _pct(yf.Ticker("^NSEI").history(period="1mo"))
        gold_r  = _pct(yf.Ticker("GC=F").history(period="1mo"))
        yld_chg = _chg(yf.Ticker("^TNX").history(period="1mo"))
        score   = nifty_r - gold_r - yld_chg/100
        if score > 0.02:
            return "🟢 Risk ON – favour equities", "Equities 70% | Gold 15% | Bonds 15%"
        if score < -0.02:
            return "🔴 Risk OFF – raise cash",      "Cash 40% | Gold 30% | Bonds 20% | Equities 10%"
        return "🟡 Neutral – mixed signals",         "Equities 40% | Gold 25% | Bonds 25% | Cash 10%"
    except Exception:
        return "⚠️ Regime unavailable", "N/A"

def generate_swot(symbol):
    try:
        info  = yf.Ticker(symbol+".NS").info
        sector = info.get('sector','N/A')
        pe     = info.get('trailingPE','N/A')
        roe    = info.get('returnOnEquity',0)*100 if info.get('returnOnEquity') else 'N/A'
        mc_cr  = info.get('marketCap',0)/1e7
    except Exception:
        sector = pe = roe = mc_cr = 'N/A'
    return call_deepseek(f"""SWOT for {symbol} (NSE). Sector:{sector} P/E:{pe} ROE:{roe} MCap:₹{mc_cr}Cr.
Format: **Strengths** / **Weaknesses** / **Opportunities** / **Threats** — 2-3 bullet points each.""",
    system="You are an equity research analyst. Output ONLY bullet points.")

def swarm_consensus(symbol, news_text):
    personas = [
        ("🟢 Bullish Trader",       "Aggressive growth, follows momentum."),
        ("🔴 Pension Fund",          "Conservative, dividends only, avoids volatility."),
        ("📊 Quant Chartist",        "Price patterns, MAs, volume. Ignores fundamentals."),
        ("💼 Value Investor",        "P/E, P/B, ROE. Buys below intrinsic value."),
        ("🌍 Macro Hedge Fund",      "Global cues, rates, currency, sector rotation."),
    ]
    results = []
    for name, style in personas:
        resp = call_deepseek(
            f"You are {name}. {style}\nNews about {symbol}: \"{news_text}\"\n"
            "Output ONE line: 'VERDICT: BUY/SELL/HOLD (confidence: XX)' then one-line REASON.",
            system="Professional investor. Two lines only."
        )
        verdict, confidence, reason = "HOLD", 50, "—"
        if resp:
            for line in resp.split('\n'):
                if 'VERDICT:' in line.upper():
                    parts = line.split('VERDICT:')[1].strip().split()
                    verdict = parts[0].upper() if parts else "HOLD"
                    try: confidence = int(line.split('confidence:')[1].strip().split()[0])  # noqa: E701,E702,E741
                    except Exception: pass  # noqa: E701,E702,E741
                elif 'REASON:' in line.upper():
                    reason = line.split('REASON:')[1].strip()
        results.append({"persona":name, "verdict":verdict, "confidence":confidence, "reason":reason})
    return results

# ── UI Helpers ─────────────────────────────────────────────────────────────────
def _index_card(name, price, change, currency="₹"):
    color = "#16a34a" if change >= 0 else "#dc2626"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"background:#f8fafc;border-radius:10px;padding:.5rem 1rem;margin-bottom:.4rem;"
        f"border-left:4px solid {color}'>"
        f"<span style='font-weight:600'>{name}</span>"
        f"<span>{currency}{price:.1f} &nbsp;"
        f"<span style='color:{color};font-weight:700'>{arrow} {abs(change):.2f}%</span></span>"
        f"</div>", unsafe_allow_html=True
    )

def _verdict_badge(direction):
    label = {1:"BUY", -1:"SELL", 0:"HOLD"}[direction]
    css   = {1:"buy",  -1:"sell",  0:"hold"}[direction]
    return label, css

def get_heatmap_data(cap, max_symbols=200):
    symbols = get_symbols_by_market_cap(cap, max_symbols)
    results = []
    with st.spinner(f"Fetching {len(symbols)} stocks…"):
        for sym in symbols:
            chg = get_stock_change_kite(sym)
            if chg is not None:
                results.append({"Stock": sym, "Change %": chg})
            time.sleep(0.02)
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Prism Quant")
st.caption("Kite · DeepSeek · Claude · Conviction Scoring · Paper Trading")

# Status banner
_ms, _mc = market_status()
_paper   = os.getenv("SQ_PAPER_TRADING","true").lower() == "true"
_b1, _b2, _b3 = st.columns(3)
_b1.markdown(f"<div style='background:{_mc};border-radius:10px;padding:.5rem 1rem;text-align:center;font-weight:600'>{_ms}</div>", unsafe_allow_html=True)
_b2.markdown(f"<div style='background:{'#dbeafe' if _paper else '#fef3c7'};border-radius:10px;padding:.5rem 1rem;text-align:center;font-weight:600'>{'📄 Paper Trading' if _paper else '💸 Live Trading'}</div>", unsafe_allow_html=True)
_b3.markdown(f"<div style='background:#f1f5f9;border-radius:10px;padding:.5rem 1rem;text-align:center;font-weight:600'>🕐 {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
st.write("")

symbol_map  = get_all_equity_symbols()
symbol_list = sorted(symbol_map.keys())

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh All Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    if st.checkbox("⏱ Auto-refresh (30s)"):
        _cd = st.empty()
        for _i in range(30, 0, -1):
            _cd.caption(f"Refreshing in {_i}s…")
            time.sleep(1)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.header("🔍 Stock Picker")
    selected = st.selectbox("Symbol", symbol_list, format_func=lambda x: f"{x} – {symbol_map[x]}")
    analyze  = st.button("🚀 Analyze", width="stretch")

    st.divider()
    st.header("🔥 Buzzing Stocks")
    buzz_min = st.slider("Min change %", 1.0, 10.0, 3.0, 0.5)
    if st.button("Scan Now"):
        with st.spinner("Scanning…"):
            buzzing = find_buzzing_stocks(list(symbol_map.keys()), limit=20)
        if buzzing:
            for sym, chg, vol_r, rsi in buzzing:
                st.write(f"🟢 **{sym}** +{chg:.1f}% · {vol_r:.1f}x vol · RSI {rsi:.0f}")
        else:
            st.info("None found.")

    st.divider()
    st.header("📊 SWOT")
    if st.button("Generate SWOT"):
        with st.spinner():
            st.session_state.swot = generate_swot(selected)
    if "swot" in st.session_state:
        st.markdown(st.session_state.swot)

# ── VERDICT CARD (after Analyze click) ────────────────────────────────────────
if analyze and selected:
    with st.spinner(f"Analysing {selected}…"):
        verdict = get_stock_verdict(selected)
    st.session_state["last_verdict"]  = verdict
    st.session_state["last_selected"] = selected

if "last_verdict" in st.session_state:
    verdict   = st.session_state["last_verdict"]
    _selected = st.session_state.get("last_selected", selected)

    if "error" in verdict:
        st.error(verdict["error"])
    else:
        cv = verdict["conviction"]
        dir_text, css = _verdict_badge(verdict['direction'])

        st.subheader(f"📊 {_selected} – {symbol_map.get(_selected,'')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price",          f"₹{verdict['price']:.2f}")
        c2.metric("Signal",         f"{verdict['signal']:.3f}")
        c3.metric("ML Confidence",  f"{verdict['confidence']:.1f}%")
        c4.metric("Conviction",     f"{cv.score:.0f}/100")
        c5.markdown(f"<div class='recommendation {css}'>{dir_text}</div>", unsafe_allow_html=True)

        if not cv.gates_passed:
            st.warning("⚠️ Gate failures: " + " | ".join(cv.gate_failures))

        st.caption(f"Data as of {verdict['last_date']}")

        col_l, col_r = st.columns(2)
        with col_l:
            with st.expander("📝 DeepSeek Bull/Bear Debate"):
                st.markdown(verdict['debate'] or "_DeepSeek key not set._")
            with st.expander("🔧 Signal Attribution"):
                st.json(verdict['attribution'])
            with st.expander("🎯 Conviction Breakdown"):
                comp = cv.components
                cdf  = pd.DataFrame({"Component": list(comp.keys()),
                                     "Score (0-1)": [round(v,3) for v in comp.values()]})
                st.dataframe(cdf, hide_index=True, width="stretch")
                st.caption(f"Profile: Conservative · Verdict: **{cv.verdict}**")

        with col_r:
            with st.expander("📊 Volume Profile"):
                if len(verdict['df']) >= 50:
                    prof = vp.compute(verdict['df'])
                    px_  = verdict['price']
                    loc  = ('Above' if px_ > prof['vah'] else
                            'Inside' if prof['val'] <= px_ <= prof['vah'] else 'Below')
                    st.markdown(f"""
- **POC:** ₹{prof['poc']:.2f}
- **Value Area:** ₹{prof['val']:.2f} – ₹{prof['vah']:.2f}
- **Price Location:** {loc} Value Area
- **HVN walls:** {', '.join(f'₹{h:.2f}' for h in prof['hvns'][:4]) or 'None'}
- **LVN gaps:** {', '.join(f'₹{lvn:.2f}' for lvn in prof['lvns'][:4]) or 'None'}
""")
            with st.expander("🤖 Dual-LLM Final Signal (DeepSeek → Claude)"):
                if st.button("Ask DeepSeek → Claude", key="claude_btn"):
                    with st.spinner("Running dual-LLM pipeline…"):
                        _sig = _get_svc().signal(
                            f"Symbol: {_selected} | Price: ₹{verdict['price']:.2f} | "
                            f"Signal: {verdict['signal']:.3f} | Confidence: {verdict['confidence']:.1f}% | "
                            f"Conviction: {cv.score:.0f}/100 | Direction: {dir_text}\n\n"
                            f"DeepSeek analysis:\n{(verdict['debate'] or '')[:600]}",
                            _selected,
                        )
                    if _sig:
                        _act = _sig.action
                        _cc  = _sig.confidence
                        _, _scss = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(_act,0))
                        st.markdown(
                            f"{_get_svc().badge(_sig.llm_decision_maker)} "
                            f"<div class='recommendation {_scss}'>{_act} · {_cc*100:.0f}%</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Reasoning:** {_sig.reasoning}")
                    else:
                        st.warning("No signal returned.")
            with st.expander("🧠 Trade Memory"):
                st.markdown(verdict['past_memory'])

        update_memory(_selected, dir_text, verdict['confidence'], verdict['price'])

# ── TABS ───────────────────────────────────────────────────────────────────────
# ── DevBloom Terminal — 9-section navigation ──────────────────────────────────
st.sidebar.markdown(
    "<div style='padding:.75rem 0 .25rem'>"
    "<span style='font-size:.6rem;color:#8892a4;text-transform:uppercase;letter-spacing:.1em'>DevBloom Terminal</span><br>"
    "<span style='font-size:1.25rem;color:#00d4ff;font-weight:700;letter-spacing:.02em'>⚡ v1.0</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Ctrl+K  command palette")

# Co-Pilot sidebar widget
render_copilot_sidebar(context={
    "symbol": selected if "selected" in dir() else "",
    "indicators": {},
})

tabs = st.tabs([
    "🏠 Command Center",      # 0 — Home Dashboard
    "📈 Charts",              # 1 — Technical Analysis Suite
    "📊 Fundamentals",        # 2 — Fundamental Deep Dive
    "⚡ Co-Pilot",            # 3 — AI Co-Pilot Dev
    "⚙️ Execution",           # 4 — Execution & Risk Cockpit
    "🧬 AlgoLab",             # 5 — Code Cave
    "📓 Journal",             # 6 — Journaling & Performance
    "🔬 Screener",            # 7 — Stock Screener (existing)
    "🔎 Decision",            # 8 — Legacy Decision Terminal
])
# Extend with hidden placeholders so legacy tab[N] references (N≥9) don't IndexError
tabs = list(tabs) + [st.empty() for _ in range(14)]

# ── Tab 0: Command Center ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>⚡ Command Center</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.2rem 0 1rem'>Global pulse · Watchlist · Alerts · Macro</p>",
        unsafe_allow_html=True,
    )

    # ── Macro strip ──────────────────────────────────────────────────────────
    render_macro_strip()
    st.divider()

    # ── Main grid: Heatmap + Watchlist | Alerts + Macro ──────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### 🗺️ Sector Heatmap")
        render_heatmap()
        st.markdown("##### 📋 Watchlist")
        render_watchlist()

    with right:
        st.markdown("##### 🚨 Alert Inbox")
        render_alert_inbox()
        st.markdown("##### 📊 Macro Dashboard")
        render_macro_dashboard()

# ── Tab 1: Technical Analysis Suite ──────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>📈 Technical Analysis Suite</h2>",
        unsafe_allow_html=True,
    )
    _charts_sym = selected

    # Multi-timeframe grid
    with st.expander("🕐 Multi-Timeframe Grid (8 views)", expanded=False):
        if _charts_sym:
            render_multi_tf_grid(_charts_sym)
        else:
            st.info("Select a symbol in the sidebar to view multi-timeframe charts.")

    # Anomaly scanner
    with st.expander("🔍 Anomaly Scanner", expanded=False):
        render_anomaly_scanner()

    st.divider()
    if "last_verdict" in st.session_state and "error" not in st.session_state["last_verdict"]:
        _sel = st.session_state.get("last_selected", selected)
        df   = fetch_historical(_sel, days=150)
        if df is not None and len(df) > 50:
            ind = ie.compute(df, _sel)
            df['EMA10']  = df['close'].ewm(span=10).mean()
            df['EMA20']  = df['close'].ewm(span=20).mean()
            df['EMA50']  = df['close'].ewm(span=50).mean()
            df['EMA200'] = df['close'].ewm(span=200).mean()
            close_now = df['close'].iloc[-1]
            e10,e20,e50,e200 = df['EMA10'].iloc[-1],df['EMA20'].iloc[-1],df['EMA50'].iloc[-1],df['EMA200'].iloc[-1]
            msgs = []
            if close_now > e10 > e20 > e50 > e200: msgs.append("🟢 Above all EMAs — strong uptrend.")  # noqa: E701,E702,E741
            elif close_now > e10 and close_now > e20: msgs.append("🟡 Above short-term EMAs — short-term bullish.")  # noqa: E701,E702,E741
            else: msgs.append("🔴 Below EMA10 — weak momentum.")  # noqa: E701,E702,E741
            if e10 > e20 > e50: msgs.append("📈 Golden cross alignment (10>20>50).")  # noqa: E701,E702,E741
            elif e10 < e20 < e50: msgs.append("📉 Death cross — trend is down.")  # noqa: E701,E702,E741
            msgs.append(f"📊 {((close_now-e200)/e200*100):.1f}% from 200 EMA.")
            st.info("\n".join(msgs))
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'],name="Price"))
            for span, color, name in [(10,'#22c55e','EMA10'),(20,'#eab308','EMA20'),(50,'#a855f7','EMA50'),(200,'#ef4444','EMA200')]:
                fig.add_trace(go.Scatter(x=df.index,y=df[f'EMA{span}'],mode='lines',line=dict(color=color,width=1),name=name))
            if st.checkbox("Show VWAP"):
                tp = (df['high']+df['low']+df['close'])/3
                fig.add_trace(go.Scatter(x=df.index,y=(tp*df['volume']).cumsum()/df['volume'].cumsum(),
                                         mode='lines',line=dict(color='#3b82f6',width=1.5,dash='dot'),name='VWAP'))
            fig.update_layout(title=f"{_sel} — Candlestick + EMAs", height=580,
                              xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, width="stretch")

            rsi_v = ind.get('rsi_14',50); zsc_v=ind.get('zscore_20',0); mom_v=ind.get('momentum_5d_pct',0); vol_v=ind.get('volume_ratio',1)  # noqa: E701,E702,E741
            tp2 = (df['high']+df['low']+df['close'])/3
            vwap_v = ((tp2*df['volume']).cumsum()/df['volume'].cumsum()).iloc[-1]
            st.dataframe(pd.DataFrame({
                "Indicator":["RSI(14)","Z-Score(20)","Momentum 5d","Volume Ratio","VWAP"],
                "Value":[f"{rsi_v:.1f}",f"{zsc_v:.2f}",f"{mom_v:.2f}%",f"{vol_v:.2f}x",f"₹{vwap_v:.2f}"],
                "Signal":["Oversold<30/OB>70","Extreme<-2|>2","Bullish>2%","High>1.5x","Below=discount"],
            }), hide_index=True, width="stretch")
    else:
        st.info("Select a stock and click Analyze first.")

# ── Tab 2: Fundamentals Deep Dive ────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>📊 Fundamental Deep Dive</h2>",
        unsafe_allow_html=True,
    )
    f_sym = st.selectbox("Symbol", symbol_list, key="fund_sym2",
                          format_func=lambda x: f"{x} – {symbol_map[x]}")
    if st.button("🔍 Load Fundamentals", key="fund_load2"):
        with st.spinner("Fetching…"):
            _f_info = yf.Ticker(f_sym+".NS").info
        st.subheader("🏢 Overview")
        st.write(f"**{_f_info.get('longName',f_sym)}** | {_f_info.get('sector','N/A')} → {_f_info.get('industry','N/A')}")
        st.write(_f_info.get('longBusinessSummary','')[:500] + "…")

        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        _fc1.metric("P/E Ratio",   f"{_f_info.get('trailingPE','N/A')}")
        _fc2.metric("Market Cap",  f"₹{(_f_info.get('marketCap',0)/1e9):.1f}B")
        _fc3.metric("ROE %",       f"{(_f_info.get('returnOnEquity',0) or 0)*100:.1f}%")
        _fc4.metric("Div Yield",   f"{(_f_info.get('dividendYield',0) or 0)*100:.2f}%")

        st.subheader("📞 Concall Summary")
        _cc = scrape_screener_concall(f_sym)
        st.markdown(_cc)

    st.divider()
    st.markdown("#### 🧮 Conviction Score (Decision Terminal)")
    dt_c1, dt_c2 = st.columns([1,2])
    with dt_c1:
        dt_profile = st.selectbox("Trader Profile", list(PROFILES.keys()), key="dt_p")
        dt_capital = st.number_input("Capital (₹)", value=100_000, step=10_000, key="dt_cap")
        dt_sym     = st.selectbox("Symbol", symbol_list, key="dt_sym",
                                   format_func=lambda x: f"{x} – {symbol_map[x]}")
        run_dt = st.button("▶ Run", width="stretch", key="dt_go")

    if run_dt:
        with st.spinner("Computing…"):
            _df = fetch_historical(dt_sym, days=100)
        if _df is not None and len(_df) >= 30:
            _ind    = ie.compute(_df, dt_sym)
            _scorer = ConvictionScorer(PROFILES[dt_profile])
            _res    = _scorer.score(_ind)
            _atr    = _ind.get("atr_14", _df['close'].iloc[-1]*0.015)
            _price  = _df['close'].iloc[-1]
            _setup  = compute_trade_setup(dt_sym, _res.verdict, _price, _atr, dt_capital) \
                      if _res.verdict in ("BUY","SELL") else None
            st.session_state["dt_res"]      = _res
            st.session_state["dt_setup"]    = _setup
            st.session_state["dt_price"]    = _price
            st.session_state["dt_ind"]      = _ind
            st.session_state["dt_sym_val"]  = dt_sym
        else:
            st.warning("Not enough data.")

    if "dt_res" in st.session_state:
        r       = st.session_state["dt_res"]
        setup   = st.session_state["dt_setup"]
        dt_price= st.session_state["dt_price"]
        _lbl, _css = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(r.verdict, 0))

        with dt_c2:
            m1,m2,m3 = st.columns(3)
            m1.metric("Conviction Score", f"{r.score:.1f}/100")
            m2.markdown(f"<div class='recommendation {_css}' style='font-size:1.1rem'>{_lbl}</div>",
                        unsafe_allow_html=True)
            m3.metric("Gates", "✅ All pass" if r.gates_passed else f"❌ {len(r.gate_failures)} failed")

            if not r.gates_passed:
                for gf in r.gate_failures:
                    st.warning(f"Gate: {gf}")

            if setup:
                st.markdown("#### 📐 ATR Trade Setup")
                s1,s2,s3,s4,s5 = st.columns(5)
                s1.metric("Entry",  f"₹{setup.entry}")
                s2.metric("Stop",   f"₹{setup.stop}")
                s3.metric("Target", f"₹{setup.target}")
                s4.metric("Qty",    str(setup.quantity))
                s5.metric("R:R",    f"{setup.rr_ratio}×")
                st.caption(f"Risk ₹{setup.risk_amount:,.0f} · Reward ₹{setup.reward_amount:,.0f}")
            else:
                st.info("No trade setup — HOLD signal.")

            # Component bar chart
            comp = r.components
            fig_bar = go.Figure(go.Bar(
                x=list(comp.values()),
                y=list(comp.keys()),
                orientation='h',
                marker_color=['#22c55e' if v>=0.5 else '#ef4444' for v in comp.values()]
            ))
            fig_bar.update_layout(title="Conviction Components (0-1)", height=250,
                                  xaxis_range=[0,1], margin=dict(t=30,b=10))
            st.plotly_chart(fig_bar, width="stretch")

            # Dual-LLM final opinion for Decision Terminal
            if st.button("🤖 Get Dual-LLM Opinion (DeepSeek → Claude)", key="dt_claude"):
                _dt_sym = st.session_state.get('dt_sym_val', dt_sym)
                with st.spinner("Running DeepSeek → Claude pipeline…"):
                    _c_ctx = (
                        f"Symbol: {_dt_sym} | Price: ₹{dt_price:.2f} | "
                        f"Conviction: {r.score:.0f}/100 | Verdict: {r.verdict} | "
                        f"Gates passed: {r.gates_passed}\n"
                        f"Components: {json.dumps({k:round(v,2) for k,v in comp.items()})}"
                    )
                    _cop = _get_svc().signal(_c_ctx, _dt_sym)
                if _cop:
                    _ca = _cop.action
                    _, _ccss = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(_ca,0))
                    st.markdown(
                        f"{_get_svc().badge(_cop.llm_decision_maker)} "
                        f"<div class='recommendation {_ccss}'>{_ca} · {_cop.confidence*100:.0f}%</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Reasoning:** {_cop.reasoning}")

# ── Tab 3: AI Co-Pilot ───────────────────────────────────────────────────────
with tabs[3]:
    render_copilot_inline(context={
        "symbol": selected,
        "price":  st.session_state.get("last_verdict", {}).get("price", 0),
        "indicators": st.session_state.get("last_verdict", {}).get("indicators", {}),
    })

# ── Tab 4: Execution & Risk Cockpit ──────────────────────────────────────────
with tabs[4]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>⚙️ Execution & Risk Cockpit</h2>",
        unsafe_allow_html=True,
    )
    _exec_tabs = st.tabs(["Order Pad", "Position Monitor", "Backtest Bridge", "Paper Trading"])

    with _exec_tabs[0]:
        _exec_sym   = selected or ""
        _exec_price = st.session_state.get("last_verdict", {}).get("price", 0.0)
        _exec_ind   = st.session_state.get("last_verdict", {}).get("indicators", {})
        render_order_pad(_exec_sym, _exec_price, _exec_ind)

    with _exec_tabs[1]:
        render_position_monitor()
        st.divider()
        render_equity_curve()

    with _exec_tabs[2]:
        render_backtest_bridge(selected or "", fetcher=fetcher, ie=ie)

    with _exec_tabs[3]:
        st.header("📋 Paper Trading Dashboard")
        init_db()
        summ = get_trading_summary()
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total P&L",   f"₹{summ['total_pnl']:,.2f}")
        m2.metric("Win Rate",    f"{summ['win_rate']:.1f}%")
        m3.metric("Trades",      str(summ['num_trades']))
        m4.metric("Best/Worst",  f"₹{summ['best_trade']:,.0f} / ₹{summ['worst_trade']:,.0f}")
        eq_df2 = get_equity_curve()
        if not eq_df2.empty:
            fig_eq2 = go.Figure(go.Scatter(x=eq_df2['date'], y=eq_df2['equity'],
                                           mode='lines+markers', fill='tozeroy',
                                           line=dict(color='#00d4ff', width=2),
                                           fillcolor='rgba(0,212,255,.06)'))
            fig_eq2.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(8,12,28,.6)", margin=dict(t=0,b=0))
            st.plotly_chart(fig_eq2, width="stretch")
        pt_c1, pt_c2 = st.columns(2)
        with pt_c1:
            st.subheader("🟢 Open Positions")
            op = get_open_positions()
            if not op.empty:
                st.dataframe(op[['symbol','entry_date','entry_price','quantity','direction']],
                             hide_index=True, width="stretch")
            else:
                st.info("No open positions.")
        with pt_c2:
            st.subheader("📜 Trade History")
            cp = get_closed_positions()
            if not cp.empty:
                st.dataframe(cp[['symbol','entry_date','exit_date','entry_price','exit_price','pnl']],
                             hide_index=True, width="stretch")
            else:
                st.info("No closed trades.")

# ── Tab 5: AlgoLab ───────────────────────────────────────────────────────────
with tabs[5]:
    render_algolab(fetcher=fetcher)

# ── Tab 6: Journal & Analytics ───────────────────────────────────────────────
with tabs[6]:
    render_journal()

# ── Tab 7: Screener ──────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>🔬 Stock Screener</h2>",
        unsafe_allow_html=True,
    )
    # ── Screener content (moved from old tabs[22]) ────────────────────────────
    st.caption(
        "Filter the entire NSE universe by fundamentals, technicals, and ensemble ML signal."
    )
    with st.expander("📐 Fundamental Filters", expanded=True):
        _sf1, _sf2, _sf3 = st.columns(3)
        with _sf1:
            _sc_pe_max   = st.number_input("P/E ≤",           min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="sc_pe_max2")
            _sc_roe_min  = st.number_input("ROE ≥ %",         min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_roe_min2")
        with _sf2:
            _sc_debt_max = st.number_input("Debt/Equity ≤",   min_value=0.0, max_value=50.0,  value=0.0, step=0.1, key="sc_debt_max2")
            _sc_mcap_min = st.number_input("Market Cap ≥ ₹Cr",min_value=0.0, max_value=1e7,   value=0.0, step=100.0, key="sc_mcap_min2")
        with _sf3:
            _sc_prom_min = st.number_input("Promoter Holding ≥ %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_prom_min2")
            _sc_div_min  = st.number_input("Dividend Yield ≥ %",   min_value=0.0, max_value=20.0,  value=0.0, step=0.1, key="sc_div_min2")
    with st.expander("📊 Technical Filters", expanded=True):
        _st1, _st2, _st3 = st.columns(3)
        with _st1:
            _sc_rsi_max   = st.number_input("RSI ≤", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_max2")
            _sc_rsi_min2  = st.number_input("RSI ≥", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_min2")
        with _st2:
            _sc_vol_spike2 = st.number_input("Volume spike ≥ ×", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="sc_vol_spike2")
            _sc_above_sma2 = st.checkbox("Above 50-day SMA", key="sc_above_sma2")
        with _st3:
            _sc_ml_min2  = st.number_input("ML conviction ≥", min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="sc_ml_min2")
    if st.button("🔎 Run Screener", key="screener_run2", width="stretch"):
        with st.spinner("Screening NSE universe…"):
            try:
                from screener.engine import ScreenerEngine
                _sc_eng2 = ScreenerEngine()
                _sc_res2, _sc_err2 = _sc_eng2.run(
                    pe_max=_sc_pe_max or None,
                    roe_min=_sc_roe_min or None,
                    debt_equity_max=_sc_debt_max or None,
                    mcap_min_cr=_sc_mcap_min or None,
                    promoter_min=_sc_prom_min or None,
                    div_yield_min=_sc_div_min or None,
                    rsi_max=_sc_rsi_max or None,
                    rsi_min=_sc_rsi_min2 or None,
                    vol_spike_min=_sc_vol_spike2 or None,
                    above_sma50=_sc_above_sma2 or None,
                    conviction_min=_sc_ml_min2 or None,
                )
                if _sc_err2:
                    st.error(f"Screener error: {_sc_err2}")
                elif _sc_res2 is not None and not _sc_res2.empty:
                    st.success(f"Found {len(_sc_res2)} stocks")
                    st.dataframe(_sc_res2, width="stretch", hide_index=True)
                else:
                    st.info("No stocks matched the filters.")
            except Exception as _sc_exc2:
                st.error(f"Screener unavailable: {_sc_exc2}")

    st.divider()
    st.subheader("🤖 Daily Pulse + Buzz Scanner")
    if st.button("Generate Daily Pulse", key="pulse_btn2"):
        with st.spinner("Generating…"):
            pulse = generate_auto_pulse()
        st.markdown(f"<div class='devbloom-card'>{pulse}</div>", unsafe_allow_html=True)

    st.subheader("📊 Buzzing Stocks")
    if st.button("Find Buzzing Stocks", key="buzz_btn2"):
        syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning for buzz…"):
            buzz = find_buzzing_stocks(syms, limit=20)
        if buzz:
            st.dataframe(pd.DataFrame(buzz), width="stretch")
        else:
            st.info("No buzzing stocks found.")

# ── Tab 8: Decision Terminal (legacy + Swarm) ─────────────────────────────────
with tabs[8]:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:1.4rem;margin:0'>🔎 Decision Terminal</h2>",
        unsafe_allow_html=True,
    )
    st.subheader("🐟 Swarm Intelligence — 5 Personas")
    news_input = st.text_area("News / event", height=80,
                               placeholder="HDFC Life profit up 4%…")
    if st.button("Run Swarm"):
        if not news_input:
            st.warning("Enter some news.")
        else:
            with st.spinner("Asking 5 personas…"):
                sw_results = swarm_consensus(selected, news_input)
            for r in sw_results:
                col_a, col_b = st.columns([3,7])
                with col_a:
                    _, css = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(r['verdict'],0))
                    st.markdown(f"**{r['persona']}**")
                    st.markdown(f"<div class='recommendation {css}' style='font-size:1rem;padding:.4rem'>"
                                f"{r['verdict']} {r['confidence']}%</div>", unsafe_allow_html=True)
                with col_b:
                    st.caption(r['reason'])
            buy_  = sum(1 for r in sw_results if r['verdict']=="BUY")
            sell_ = sum(1 for r in sw_results if r['verdict']=="SELL")
            hold_ = sum(1 for r in sw_results if r['verdict']=="HOLD")
            st.info(f"Consensus → BUY: {buy_} | SELL: {sell_} | HOLD: {hold_}")

# ── Tab 9 onwards: Legacy content rendered in hidden containers ────────────────
with tabs[9]:
    st.subheader("🌍 Global Indices")
    gl = get_global_indices()
    if gl:
        gc = st.columns(3)
        for i, (name, val) in enumerate(gl.items()):
            gc[i%3].metric(name, f"{val['price']:.1f}", f"{val['change']:+.2f}%",
                           delta_color="normal" if val['change']>=0 else "inverse")
    else:
        st.info("Unavailable.")

# ── Tab 10: Pre-Market Report ──────────────────────────────────────────────────
with tabs[10]:
    st.header("🌅 Pre-Market Report")
    if st.button("Fetch Moneycontrol"):
        with st.spinner():
            report, err = get_moneycontrol_premarket()
        if report:
            st.markdown(f"<div class='card'>{report}</div>", unsafe_allow_html=True)
        else:
            st.error(f"Failed: {err}")
            st.subheader("Raw Global Cues")
            for n,v in get_global_indices().items():
                st.write(f"{n}: {v['price']:.1f} ({v['change']:+.2f}%)")

# ── Tab 11: Quant Hedge Fund ───────────────────────────────────────────────────
with tabs[11]:
    st.header("🏦 Quant Hedge Fund")
    if st.button("🚀 Run All Quant Models"):
        with st.spinner("Fetching & computing…"):
            df = fetch_historical(selected, days=300)
        if df is None or len(df) < 60:
            st.error("Need at least 60 days of data.")
        else:
            close = df['close']
            # Time-series momentum
            if len(close) >= 200:
                ret     = (close.iloc[-1]-close.iloc[-200])/close.iloc[-200]
                dv      = close.pct_change().rolling(60).std().iloc[-1]*np.sqrt(252)
                msig    = float(np.clip(ret/dv, -1, 1)) if dv else 0
                mv      = "🟢 BUY" if msig>0.3 else ("🔴 SELL" if msig<-0.3 else "⚪ HOLD")
            else:
                msig, mv = 0.0, "Insufficient (need 200d)"

            # Volatility targeting
            rv   = close.pct_change().dropna().iloc[-60:].std()*np.sqrt(252)
            mult = float(np.clip(0.20/rv, 0.2, 3.0)) if rv else 1.0
            vv   = "📈 Increase size" if mult>1.2 else ("📉 Reduce size" if mult<0.8 else "✅ Normal size")

            # Pair trade (best largecap correlation)
            best_pair, best_corr, close2 = None, 0, None
            for sym in get_symbols_by_market_cap("Largecap", 30):
                if sym == selected: continue  # noqa: E701,E702,E741
                df2 = fetch_historical(sym, days=250)
                if df2 is None or len(df2)<100: continue  # noqa: E701,E702,E741
                common = close.index.intersection(df2['close'].index)
                if len(common)<100: continue  # noqa: E701,E702,E741
                corr = close.loc[common].corr(df2['close'].loc[common])
                if abs(corr) > abs(best_corr):
                    best_corr, best_pair, close2 = corr, sym, df2['close']
            if best_pair and close2 is not None:
                common = close.index.intersection(close2.index)
                ratio  = close.loc[common]/close2.loc[common]
                zscore = (ratio.iloc[-1]-ratio.mean())/ratio.std()
                pv     = (f"🔴 SELL {selected}/BUY {best_pair}" if zscore>2 else
                          f"🟢 BUY {selected}/SELL {best_pair}" if zscore<-2 else
                          f"⚪ HOLD (z={zscore:.2f})")
            else:
                pv = "No correlated pair found."

            regime, alloc = macro_regime_allocation()
            st.markdown(f"""
- **Momentum Signal:** `{msig:.2f}` → {mv}
- **Vol Targeting Mult:** `{mult:.2f}x` → {vv}
- **Pair Trade:** {pv}
- **Macro Regime:** {regime} — {alloc}
""")

# ── Tab 12: Fundamentals & Ownership ─────────────────────────────────────────
with tabs[12]:
    st.header("📊 Fundamentals & Ownership")
    f_sym = st.selectbox("Symbol", symbol_list, key="fund_sym",
                          format_func=lambda x: f"{x} – {symbol_map[x]}")
    if st.button("Load", key="fund_load"):
        with st.spinner():
            info = yf.Ticker(f_sym+".NS").info
        st.subheader("🏢 Overview")
        st.write(f"**{info.get('longName',f_sym)}** | {info.get('sector','N/A')} → {info.get('industry','N/A')}")
        st.write(info.get('longBusinessSummary','')[:500] + "…")

        st.subheader("📈 Shareholding (Screener.in)")
        qtr,p,fi,di,err = scrape_screener_shareholding(f_sym)
        if err:
            st.warning(err)
        else:
            if p: st.metric("Promoter Holding", f"{p[-1]:.2f}%")  # noqa: E701,E702,E741
            if qtr and fi and di:
                mn = min(len(qtr),len(fi),len(di))
                td = pd.DataFrame({"Quarter":qtr[:mn],"FII (%)":fi[:mn],"DII (%)":di[:mn]}).dropna()
                if not td.empty:
                    fig_fi = px.line(td, x="Quarter", y=["FII (%)","DII (%)"],
                                     markers=True, color_discrete_map={"FII (%)":"orange","DII (%)":"green"})
                    st.plotly_chart(fig_fi, width="stretch")

        st.subheader("📞 Concall Summary")
        cc = scrape_screener_concall(f_sym)
        st.markdown(cc)

# ── Tab 13: Paper Trading ──────────────────────────────────────────────────────
with tabs[13]:
    st.header("📋 Paper Trading Dashboard")
    init_db()

    # Summary metrics
    summ = get_trading_summary()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total P&L",   f"₹{summ['total_pnl']:,.2f}",
              delta=f"₹{summ['total_pnl']:,.2f}",
              delta_color="normal" if summ['total_pnl']>=0 else "inverse")
    m2.metric("Win Rate",    f"{summ['win_rate']:.1f}%")
    m3.metric("Trades",      str(summ['num_trades']))
    m4.metric("Best/Worst",  f"₹{summ['best_trade']:,.0f} / ₹{summ['worst_trade']:,.0f}")

    # Equity curve
    eq_df = get_equity_curve()
    if not eq_df.empty:
        st.subheader("📈 Equity Curve")
        fig_eq = go.Figure(go.Scatter(
            x=eq_df['date'], y=eq_df['equity'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#2563eb', width=2),
            fillcolor='rgba(37,99,235,.08)'
        ))
        fig_eq.update_layout(height=260, margin=dict(t=10,b=10),
                             xaxis_title="Date", yaxis_title="Equity (₹)")
        st.plotly_chart(fig_eq, width="stretch")

    st.divider()
    pt_c1, pt_c2 = st.columns(2)

    # Open positions
    with pt_c1:
        st.subheader("🟢 Open Positions")
        op = get_open_positions()
        if not op.empty:
            def _cd(v): return "color:#16a34a;font-weight:700" if v=="BUY" else "color:#dc2626;font-weight:700"
            st.dataframe(op[['symbol','entry_date','entry_price','quantity','direction']]
                         .style.applymap(_cd, subset=['direction']),
                         hide_index=True, width="stretch")
        else:
            st.info("No open positions.")

    # Closed trades
    with pt_c2:
        st.subheader("📜 Trade History")
        cp = get_closed_positions()
        if not cp.empty:
            cp2 = cp[['symbol','entry_date','exit_date','entry_price','exit_price','quantity','pnl']].copy()
            cp2['pnl'] = cp2['pnl'].round(2)
            def _cpnl(v): return "color:#16a34a;font-weight:700" if v>0 else ("color:#dc2626;font-weight:700" if v<0 else "")
            st.dataframe(cp2.style.applymap(_cpnl, subset=['pnl']),
                         hide_index=True, width="stretch")
        else:
            st.info("No closed trades.")

    st.divider()
    st.subheader("⚡ Execute Paper Trade")
    pt_sym = st.selectbox("Symbol", symbol_list, key="pt_sym",
                           format_func=lambda x: f"{x} – {symbol_map[x]}")
    if st.button("▶ Run Signal & Trade", width="stretch"):
        with st.spinner(f"Analysing {pt_sym}…"):
            v2 = get_stock_verdict(pt_sym)
        if "error" in v2:
            st.error(v2["error"])
        else:
            dl, css2 = _verdict_badge(v2['direction'])
            st.markdown(f"<div class='recommendation {css2}'>{dl} · {v2['confidence']:.1f}%</div>",
                        unsafe_allow_html=True)
            open_pos = get_open_positions()
            if v2['direction'] == 1:
                if open_pos[open_pos['symbol']==pt_sym].empty:
                    open_position(pt_sym, v2['price'], 100, "BUY", datetime.now().strftime("%Y-%m-%d"))
                    st.success(f"BUY opened: {pt_sym} @ ₹{v2['price']:.2f}")
                else:
                    st.info("Position already open.")
            elif v2['direction'] == -1:
                closed = False
                for _, row in open_pos.iterrows():
                    if row['symbol'] == pt_sym:
                        close_position(row['id'], v2['price'], datetime.now().strftime("%Y-%m-%d"))
                        st.success(f"Position closed: {pt_sym} @ ₹{v2['price']:.2f}")
                        closed = True
                if not closed:
                    st.info("No open position to close.")
            else:
                st.info("HOLD — no action.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 14 — Multi-Timeframe Signal Aligner
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[14]:
    st.subheader("📐 Multi-Timeframe Signal Aligner")
    st.caption("Checks XGBoost signal direction across 5-min, 15-min, 1-h, and daily timeframes.")

    mtf_sym = st.selectbox("Symbol", options=list(get_all_equity_symbols().keys()), key="mtf_sym")
    mtf_tfs = st.multiselect("Timeframes", ["5min", "15min", "1h", "1d"], default=["5min", "15min", "1h"], key="mtf_tfs")

    if st.button("🔍 Run Multi-Timeframe Analysis", key="mtf_run"):
        with st.spinner("Fetching data and running signals…"):
            try:
                from analysis.multi_timeframe import MultiTimeframeAligner
                aligner = MultiTimeframeAligner(fetcher=fetcher)
                result = aligner.align(mtf_sym, mtf_tfs)

                score = result.get("alignment_score", 0.0)
                consensus = result.get("consensus_action", "HOLD")
                colour = {"BUY": "#2e7d32", "SELL": "#c62828", "HOLD": "#e65100"}.get(consensus, "#555")

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"""
                    <div style="background:#fff;border-radius:16px;padding:1.5rem;text-align:center;
                                box-shadow:0 2px 8px rgba(0,0,0,.05);border:1px solid #eef2f6">
                      <p style="margin:0;font-size:.85rem;color:#666">Alignment Score</p>
                      <p style="margin:.25rem 0;font-size:2.5rem;font-weight:700;color:{colour}">{score:+.2f}</p>
                      <p style="margin:0;font-size:1rem;font-weight:600;color:{colour}">{consensus}</p>
                    </div>""", unsafe_allow_html=True)

                with col_b:
                    tf_data = result.get("timeframes", {})
                    if tf_data:
                        import pandas as pd
                        rows = [
                            {"Timeframe": tf, "Action": v.get("action","?"), "Confidence": f"{v.get('confidence',0):.1%}"}
                            for tf, v in tf_data.items()
                        ]
                        df_tf = pd.DataFrame(rows)
                        st.dataframe(df_tf, hide_index=True, width="stretch")

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 15 — Market Regime Detector
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[15]:
    st.subheader("🌡️ Market Regime Detector")
    st.caption("Combines Hurst exponent, ADX, India VIX, and F&O expiry proximity.")

    if st.button("🔄 Detect Current Regime", key="regime_run"):
        with st.spinner("Analysing market regime…"):
            try:
                from analysis.regime_detector import RegimeDetector
                rd = RegimeDetector(fetcher=fetcher)
                r = rd.detect()

                regime = r.get("regime", "UNKNOWN")
                vix_label = r.get("vix_label", "?")
                colour = "#c62828" if "HIGH_VOL" in regime else "#2e7d32" if "LOW_VOL" in regime else "#e65100"

                st.markdown(f"""
                <div style="background:{colour}11;border:2px solid {colour};border-radius:16px;
                            padding:1.25rem;text-align:center;margin-bottom:1rem">
                  <span style="font-size:1.4rem;font-weight:700;color:{colour}">{regime}</span>
                </div>""", unsafe_allow_html=True)

                if r.get("expiry_warning"):
                    st.warning("⚠️ F&O expiry within 3 days — expect elevated volatility and thin liquidity.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Hurst Exponent", f"{r.get('hurst') or 'N/A'}", help="<0.45=Mean Rev, >0.55=Trending")
                c2.metric("ADX", f"{r.get('adx') or 'N/A'}", help=">25=Trending, <20=Sideways")
                c3.metric("India VIX", f"{r.get('india_vix') or 'N/A'}", help="<13=Low, >18=High")

                st.info(f"**Recommended strategy:** {r.get('recommended_strategy','')}")

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 16 — Ensemble ML Signal
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[16]:
    st.subheader("🧠 Ensemble ML Signal")
    st.caption("XGBoost + LightGBM (+ optional Chronos-Bolt) weighted-vote signal.")

    ens_sym = st.selectbox("Symbol", options=list(get_all_equity_symbols().keys()), key="ens_sym")

    if st.button("🤖 Generate Ensemble Signal", key="ens_run"):
        with st.spinner("Training / loading models and generating signal…"):
            try:
                from ml.ensemble_signal import EnsembleSignalGenerator
                gen = EnsembleSignalGenerator()
                df_ens = fetch_historical(ens_sym, days=400)
                if df_ens is None or df_ens.empty:
                    st.warning("No data available.")
                else:
                    sig = gen.generate_signal(df_ens, ens_sym)
                    action = sig.get("action", "HOLD")
                    conf = sig.get("confidence", 0)
                    colour = {"BUY": "#2e7d32", "SELL": "#c62828", "HOLD": "#e65100"}.get(action, "#555")

                    c1, c2 = st.columns(2)
                    c1.markdown(f"""
                    <div class="recommendation {action.lower()}">{action}</div>
                    <p style="text-align:center;margin-top:.5rem">Confidence: <b>{conf:.1%}</b></p>
                    """, unsafe_allow_html=True)
                    with c2:
                        st.write("**Reasoning:**", sig.get("reasoning", ""))
                        details = sig.get("ensemble_details", {})
                        if details:
                            rows = [
                                {"Model": m, "Action": v.get("action","?"), "Confidence": f"{v.get('confidence',0):.1%}", "Weight": v.get("weight","")}
                                for m, v in details.items() if isinstance(v, dict)
                            ]
                            import pandas as pd
                            if rows:
                                st.dataframe(pd.DataFrame(rows), hide_index=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 17 — Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[17]:
    st.subheader("⚠️ Risk Metrics")
    st.caption("VaR, CVaR, Sortino, Calmar, Beta vs Nifty, Max Drawdown — last 2 years of daily data.")

    rm_sym = st.selectbox("Symbol", options=list(get_all_equity_symbols().keys()), key="rm_sym")

    if st.button("📊 Compute Risk Metrics", key="rm_run"):
        with st.spinner("Computing risk metrics…"):
            try:
                from analytics.risk_metrics import RiskMetrics
                rm = RiskMetrics(fetcher=fetcher)
                m = rm.compute(rm_sym)

                if m.get("error"):
                    st.warning(f"Could not compute metrics: {m['error']}")
                else:
                    risk_score = m.get("risk_score", 5)
                    score_colour = "#c62828" if risk_score >= 7 else "#e65100" if risk_score >= 5 else "#2e7d32"

                    st.markdown(f"""
                    <div style="background:{score_colour}11;border:2px solid {score_colour};
                                border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem">
                      <span style="font-size:1rem;color:#666">Risk Score</span><br>
                      <span style="font-size:2.5rem;font-weight:700;color:{score_colour}">{risk_score}/10</span>
                    </div>""", unsafe_allow_html=True)

                    import pandas as pd
                    rows = [
                        ("Annualised Return", f"{m.get('annualised_return_pct','N/A')}%"),
                        ("Annualised Volatility", f"{m.get('annualised_volatility_pct','N/A')}%"),
                        ("Sharpe Ratio", m.get('sharpe_ratio','N/A')),
                        ("Sortino Ratio", m.get('sortino_ratio','N/A')),
                        ("Calmar Ratio", m.get('calmar_ratio','N/A')),
                        ("Max Drawdown", f"{m.get('max_drawdown_pct','N/A')}%"),
                        ("VaR 95%", m.get('var_95','N/A')),
                        ("VaR 99%", m.get('var_99','N/A')),
                        ("CVaR 95%", m.get('cvar_95','N/A')),
                        ("CVaR 99%", m.get('cvar_99','N/A')),
                        ("Beta vs Nifty", m.get('beta','N/A')),
                    ]
                    df_rm = pd.DataFrame(rows, columns=["Metric", "Value"])
                    st.dataframe(df_rm, hide_index=True, width="stretch")

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 18 — What-If Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[18]:
    st.subheader("🔮 What-If Trade Simulator")
    st.caption("Estimates trade outcomes using historical rolling-window distributions.")

    with st.form("whatif_form"):
        wi_col1, wi_col2 = st.columns(2)
        with wi_col1:
            wi_sym = st.selectbox("Symbol", options=list(get_all_equity_symbols().keys()), key="wi_sym_sel")
            wi_qty = st.number_input("Quantity (shares)", min_value=1, value=100, step=1)
        with wi_col2:
            wi_price = st.number_input("Entry Price (₹)", min_value=1.0, value=1000.0, step=1.0)
            wi_days = st.number_input("Holding Days", min_value=1, max_value=60, value=5, step=1)
        submitted = st.form_submit_button("🔮 Simulate")

    if submitted:
        with st.spinner("Running simulation…"):
            try:
                from simulator.whatif import WhatIfSimulator
                sim = WhatIfSimulator(fetcher=fetcher)
                r = sim.simulate(wi_sym, wi_qty, wi_price, wi_days)

                if r.get("error"):
                    st.warning(f"Could not simulate: {r['error']}")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Prob. Profit", f"{r.get('prob_profit',0):.0%}")
                    c2.metric("Prob. Loss >2%", f"{r.get('prob_loss_gt_2pct',0):.0%}")
                    c3.metric("Expected Return", f"{r.get('expected_return_pct',0):.2f}%")
                    c4.metric("99% VaR", f"₹{r.get('var_99',0):,.0f}", help="Worst-case loss at 99% confidence")

                    st.caption(f"Based on {r.get('simulation_samples',0)} historical windows · "
                               f"Current price: ₹{r.get('current_price','N/A')}")
                    st.warning("⚠️ Past return distributions do not guarantee future results. "
                               "This is for educational purposes only — not financial advice.")

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 19 — Correlation Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[19]:
    st.subheader("🔗 Correlation Matrix Heatmap")
    st.caption("Pairwise return correlations. Pairs >0.7 highlighted in red.")

    all_syms = list(get_all_equity_symbols().keys())
    corr_syms = st.multiselect(
        "Select up to 20 symbols",
        options=all_syms,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"][:min(5, len(all_syms))],
        max_selections=20,
        key="corr_syms",
    )
    corr_days = st.slider("Lookback days", 20, 252, 60, key="corr_days")

    if st.button("📊 Compute Correlations", key="corr_run") and corr_syms:
        with st.spinner("Fetching prices and computing correlations…"):
            try:
                from analysis.correlation import CorrelationAnalyzer
                ca = CorrelationAnalyzer(fetcher=fetcher)
                summary = ca.summary(corr_syms, lookback_days=corr_days)

                if summary.get("error"):
                    st.warning(f"Could not compute: {summary['error']}")
                else:
                    import plotly.express as px
                    import pandas as pd
                    corr_df = pd.DataFrame(summary["matrix"])

                    fig = px.imshow(
                        corr_df,
                        color_continuous_scale="RdYlGn",
                        zmin=-1, zmax=1,
                        title=f"Return Correlation — {corr_days}d",
                        text_auto=".2f",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")

                    high_pairs = summary.get("high_corr_pairs", [])
                    if high_pairs:
                        st.markdown("**Highly correlated pairs (>0.7):**")
                        hp_df = pd.DataFrame(high_pairs)
                        st.dataframe(hp_df, hide_index=True, width="stretch")

                    avg_c = summary.get("avg_correlation", {})
                    if avg_c:
                        st.markdown("**Average correlation to universe:**")
                        avg_df = pd.DataFrame(avg_c.items(), columns=["Symbol", "Avg Correlation"]).sort_values("Avg Correlation", ascending=False)
                        st.dataframe(avg_df, hide_index=True, width="stretch")

            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 20 — Deep Fundamentals (screener.in full data)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[20]:
    st.subheader("🔬 Deep Fundamentals")
    st.caption("Full screener.in data: Key Ratios, P&L, Balance Sheet, Quarterly, Shareholding, Cash Flow. Cached 24h.")

    df_sym_input = st.text_input("NSE Symbol", value="BEL", key="df_sym_in").upper().strip()
    df_col1, df_col2 = st.columns([1, 1])
    with df_col1:
        df_force = st.checkbox("Force refresh (ignore cache)", key="df_force")
    with df_col2:
        df_load_btn = st.button("📥 Load Fundamentals", key="df_load")

    if df_load_btn and df_sym_input:
        with st.spinner(f"Fetching fundamentals for {df_sym_input}…"):
            try:
                from fundamentals.fetcher import get_deep_fundamentals
                fund_data = get_deep_fundamentals(df_sym_input, force_refresh=df_force)

                # Company header
                about = fund_data.get("about", "")
                meta  = fund_data.get("metadata", {})
                st.markdown(f"**{df_sym_input}** · {'Consolidated' if meta.get('consolidated') else 'Standalone'} · "
                            f"[screener.in]({fund_data.get('url','#')})")
                if about:
                    with st.expander("About the company"):
                        st.write(about)

                # ── Key Ratios ────────────────────────────────────────────
                key_ratios = fund_data.get("key_ratios", [])
                if key_ratios:
                    st.markdown("### 📊 Key Ratios")
                    ratio_cols = st.columns(min(len(key_ratios), 5))
                    for i, ratio in enumerate(key_ratios[:10]):
                        with ratio_cols[i % 5]:
                            st.metric(
                                label=ratio.get("name", ""),
                                value=ratio.get("value", "—"),
                            )

                # ── Section tabs ──────────────────────────────────────────
                _section_defs = [
                    ("📈 P&L",           "profit_loss"),
                    ("🏦 Balance Sheet",  "balance_sheet"),
                    ("📅 Quarterly",      "quarterly_results"),
                    ("👥 Shareholding",   "shareholding"),
                    ("💵 Cash Flow",      "cash_flow"),
                    ("🏢 Peers",          "peer_comparison"),
                ]
                fund_tabs = st.tabs([s[0] for s in _section_defs])

                for tab_obj, (_, section_key) in zip(fund_tabs, _section_defs):
                    with tab_obj:
                        rows = fund_data.get(section_key, [])
                        if not rows:
                            st.info("No data available for this section.")
                            continue
                        import pandas as pd
                        df_sec = pd.DataFrame(rows)
                        st.dataframe(
                            df_sec,
                            width="stretch",
                            hide_index=True,
                        )
                        st.caption(f"{len(rows)} rows · Data from screener.in")

            except ValueError as ve:
                st.error(f"Symbol not found: {ve}")
            except Exception as ex:
                st.error(f"Failed to load fundamentals: {ex}")

# ── TAB 21 — Professional Charts ──────────────────────────────────────────────
with tabs[21]:
    st.subheader("📊 Professional Charts")
    st.caption(
        "Candlestick · VWAP · Bollinger Bands · Donchian Channels · "
        "Volume Profile · RSI · MACD · ATR · Footprint · Liquidity Heatmap"
    )

    # ── sidebar controls ──────────────────────────────────────────────────
    pc_col1, pc_col2, pc_col3 = st.columns([2, 2, 3])
    with pc_col1:
        pc_symbol = st.text_input(
            "Symbol", value="RELIANCE", key="pc_symbol",
            help="NSE symbol — uses yfinance (.NS suffix added automatically)"
        ).upper().strip()
    with pc_col2:
        pc_period = st.selectbox(
            "Period", ["1mo", "3mo", "6mo", "1y", "5d", "1d"],
            index=1, key="pc_period"
        )
    with pc_col3:
        pc_vp_bins = st.slider("Volume Profile bins", 20, 80, 40, 5, key="pc_vp_bins")

    pc_show_vp = st.checkbox("Show Volume Profile sidebar", value=True, key="pc_show_vp")
    pc_load = st.button("🔄 Load Charts", key="pc_load", type="primary")

    # ── quick legend ──────────────────────────────────────────────────────
    with st.expander("📖 Chart Legend — what each indicator means", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**On the price chart**
🟠 **VWAP** — Today's volume-weighted fair price
🔵 **Bollinger Bands** — Price channels ±2σ from SMA-20
🟣 **Donchian Channels** — 20-period high/low range
🟡 **POC** — Price with highest volume (magnet level)
🟢 **Value Area** — 70% of all volume traded here
📊 **Volume bars** — Green = up candle, Red = down candle
""")
        with col2:
            st.markdown("""
**RSI (Relative Strength Index)**
🔴 RSI > 70 — Overbought (may pull back)
🟢 RSI < 30 — Oversold (may bounce)
⚪ RSI 30–70 — Neutral zone

**MACD**
↑ MACD cross above signal — Bullish momentum shift
↓ MACD cross below signal — Bearish momentum shift
🟩 Green histogram — Buying pressure building
🟥 Red histogram — Selling pressure building
""")
        with col3:
            st.markdown("""
**ATR (Average True Range)**
📏 Measures volatility — higher ATR = bigger swings
Use to set stop-loss distance (e.g. 1.5× ATR below entry)

**Volume Profile (right sidebar)**
Wide bars = heavy trading at that price (strong support/resistance)
Narrow bars = light trading (price moved through quickly)

**Footprint chart**
Shows buy vs sell volume per candle
⬆ Imbalance = aggressive buyers dominating
""")

    # ── explain panel ─────────────────────────────────────────────────────
    with st.expander("❓ Explain an indicator (plain language)", expanded=False):
        from charting.explanations import list_all as _pc_list_all, explain as _pc_explain
        _indicator_list = _pc_list_all()
        _selected_ind = st.selectbox(
            "Choose an indicator",
            _indicator_list,
            format_func=lambda k: k.replace("_", " ").title(),
            key="pc_explain_select",
        )
        if st.button("Explain this to me", key="pc_explain_btn"):
            st.session_state["pc_explain_text"] = _pc_explain(_selected_ind)
        if "pc_explain_text" in st.session_state:
            st.code(st.session_state["pc_explain_text"], language=None)

    # ── data fetch & charts ───────────────────────────────────────────────
    if pc_load or st.session_state.get("pc_data_loaded"):
        try:
            import yfinance as _yf
            import pandas as _pd
            from charting.engine import SmartChart as _SmartChart
            from charting.footprint import FootprintAnalyzer as _FootprintAnalyzer
            from charting.liquidity import LiquidityHeatmap as _LiqHeatmap

            _PERIOD_INTERVAL = {
                "1d": "5m", "5d": "30m",
                "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1d",
            }
            _interval = _PERIOD_INTERVAL.get(pc_period, "1d")
            _ticker = f"{pc_symbol}.NS"

            with st.spinner(f"Fetching {_ticker}  period={pc_period}  interval={_interval} …"):
                _df = _yf.download(
                    _ticker, period=pc_period, interval=_interval,
                    auto_adjust=True, progress=False
                )

            if _df.empty:
                st.error(f"No data for {_ticker}. Check the symbol spelling.")
            else:
                # Flatten MultiIndex columns (yfinance may return them)
                if isinstance(_df.columns, _pd.MultiIndex):
                    _df.columns = [c[0].lower() for c in _df.columns]
                else:
                    _df.columns = [c.lower() for c in _df.columns]
                _df = _df[["open", "high", "low", "close", "volume"]].dropna()

                st.session_state["pc_data_loaded"] = True
                st.caption(
                    f"✅ {len(_df)} bars loaded for **{pc_symbol}** "
                    f"(latest close: ₹{_df['close'].iloc[-1]:,.2f})"
                )

                # ── Chart sub-tabs ────────────────────────────────────────
                _ct1, _ct2, _ct3 = st.tabs([
                    "📈 Main Chart (VWAP · BB · Donchian · Volume Profile)",
                    "🦶 Footprint & Order Flow",
                    "💧 Liquidity Heatmap",
                ])

                with _ct1:
                    _smart_fig = _SmartChart().build(
                        _df, symbol=pc_symbol,
                        show_vp=pc_show_vp,
                        vp_bins=pc_vp_bins,
                    )
                    st.plotly_chart(_smart_fig, width="stretch")

                    # Quick indicator legend
                    _leg_col1, _leg_col2, _leg_col3 = st.columns(3)
                    with _leg_col1:
                        st.markdown(
                            "**🟠 VWAP** — Today's volume-weighted fair price\n\n"
                            "**🔵 Bollinger Bands** — Price channels ±2σ from SMA-20\n\n"
                            "**🟣 Donchian Channels** — 20-period high/low range"
                        )
                    with _leg_col2:
                        st.markdown(
                            "**🟡 POC** — Price with highest volume (magnet level)\n\n"
                            "**🟢 Value Area** — 70% of all volume traded here\n\n"
                            "**📊 Volume bars** — Green = up candle, Red = down candle"
                        )
                    with _leg_col3:
                        st.markdown(
                            "**RSI > 70** — Overbought (may pull back)\n\n"
                            "**RSI < 30** — Oversold (may bounce)\n\n"
                            "**MACD cross ↑** — Bullish momentum shift"
                        )

                with _ct2:
                    _fp_fig = _FootprintAnalyzer().build_figure(_df, symbol=pc_symbol)
                    st.plotly_chart(_fp_fig, width="stretch")
                    st.info(
                        "⭐ **Green star above candle** = ask imbalance (buyers ≥ 3× sellers at that level — bullish pressure)  \n"
                        "⭐ **Red star below candle** = bid imbalance (sellers ≥ 3× buyers — bearish pressure)  \n"
                        "📊 **Cumulative Delta** = running total of (ask vol − bid vol). Rising = buyers winning; falling = sellers winning.  \n"
                        "🔍 *Hover over any candle to see bid vol, ask vol, and delta.*  \n"
                        "⚠️ *Simulated from OHLCV data — real tick data requires Zerodha WebSocket.*"
                    )

                with _ct3:
                    _current_price = float(_df["close"].iloc[-1])
                    _book = _LiqHeatmap().simulate_book(_current_price)
                    _liq_fig = _LiqHeatmap().build_figure(_book, symbol=pc_symbol)
                    st.plotly_chart(_liq_fig, width="stretch")
                    st.info(
                        "🟢 **Green bars (left)** = pending buy orders (bids) — act as support  \n"
                        "🔴 **Red bars (right)** = pending sell orders (asks) — act as resistance  \n"
                        "**Bigger/brighter bar** = larger order cluster = stronger support/resistance  \n"
                        "⚠️ *Simulated — live order book requires Zerodha WebSocket market depth.*"
                    )

        except Exception as _pc_err:
            st.error(f"Chart error: {_pc_err}")
            st.exception(_pc_err)
    else:
        st.info("Enter a symbol and click **Load Charts** to begin.")

# ── TAB 22 — Stock Screener ────────────────────────────────────────────────────
with tabs[22]:
    st.subheader("🔎 NSE Stock Screener")
    st.caption(
        "Filter the entire NSE universe (~2000 stocks) by fundamentals, "
        "technicals, and ensemble ML signal. "
        "Technical data is cached locally; fundamental data is fetched from screener.in."
    )

    # ── Filter controls ───────────────────────────────────────────────────
    with st.expander("📐 Fundamental Filters", expanded=True):
        _sf1, _sf2, _sf3 = st.columns(3)
        with _sf1:
            _sc_pe_max   = st.number_input("P/E ≤",           min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="sc_pe_max",   help="Max P/E ratio (0 = no filter)")
            _sc_roe_min  = st.number_input("ROE ≥ %",         min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_roe_min",  help="Min Return on Equity %")
        with _sf2:
            _sc_debt_max = st.number_input("Debt/Equity ≤",   min_value=0.0, max_value=50.0,  value=0.0, step=0.1, key="sc_debt_max", help="Max Debt-to-Equity (0 = no filter)")
            _sc_mcap_min = st.number_input("Market Cap ≥ ₹Cr",min_value=0.0, max_value=1e7,   value=0.0, step=100.0, key="sc_mcap_min")
        with _sf3:
            _sc_prom_min = st.number_input("Promoter Holding ≥ %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_prom_min")
            _sc_div_min  = st.number_input("Dividend Yield ≥ %",   min_value=0.0, max_value=20.0,  value=0.0, step=0.1, key="sc_div_min")

    with st.expander("📊 Technical Filters", expanded=True):
        _st1, _st2, _st3 = st.columns(3)
        with _st1:
            _sc_rsi_max = st.number_input("RSI ≤ (oversold)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_max")
            _sc_rsi_min = st.number_input("RSI ≥ (overbought)",min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sc_rsi_min")
        with _st2:
            _sc_vol_spike = st.number_input("Volume spike ≥ ×", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="sc_vol_spike", help="Volume ratio vs 30-day avg")
            _sc_above_sma = st.selectbox("Price above SMA", [None, 20, 50], key="sc_above_sma")
        with _st3:
            _sc_below_sma = st.selectbox("Price below SMA", [None, 20, 50], key="sc_below_sma")
            _sc_signal    = st.selectbox("Ensemble signal", [None, "BUY", "SELL", "HOLD"], key="sc_signal")

    _sc_lim_col, _sc_scrape_col = st.columns([3, 2])
    with _sc_lim_col:
        _sc_limit = st.slider("Max results", 5, 200, 50, 5, key="sc_limit")
    with _sc_scrape_col:
        _sc_scrape = st.checkbox(
            "Scrape missing fundamentals (slow, 1st run)",
            value=False, key="sc_scrape",
            help="If unchecked, stocks without cached fundamentals are skipped"
        )

    _sc_run = st.button("🚀 Run Screener", key="sc_run", type="primary")

    if _sc_run:
        # Convert 0 → None for "no filter"
        def _nz(v):
            return v if v and v > 0 else None

        _sc_filters = dict(
            pe_max               = _nz(_sc_pe_max),
            roe_min              = _nz(_sc_roe_min),
            debt_max             = _nz(_sc_debt_max),
            market_cap_min_cr    = _nz(_sc_mcap_min),
            promoter_holding_min = _nz(_sc_prom_min),
            dividend_yield_min   = _nz(_sc_div_min),
            rsi_max              = _nz(_sc_rsi_max),
            rsi_min              = _nz(_sc_rsi_min),
            volume_spike_min     = _nz(_sc_vol_spike),
            price_above_sma_days = _sc_above_sma,
            price_below_sma_days = _sc_below_sma,
            ensemble_signal      = _sc_signal,
            limit                = _sc_limit,
            scrape_missing_fundamentals = _sc_scrape,
        )

        active_filters = {k: v for k, v in _sc_filters.items() if v is not None and v is not False}
        if not active_filters:
            st.warning("Set at least one filter before running.")
        else:
            import time as _time
            _t0 = _time.time()
            _prog = st.progress(0, text="Loading NSE universe …")
            try:
                from screener.engine import ScreenerEngine as _SE
                _prog.progress(20, text="Updating technicals cache (may take a few minutes on first run) …")
                _sc_df = _SE().screen_by_ratios(**_sc_filters)
                _prog.progress(100, text="Done!")
                _elapsed = round(_time.time() - _t0, 1)

                if _sc_df.empty:
                    st.info("No stocks match the current filters.")
                else:
                    st.success(f"Found **{len(_sc_df)}** stocks in {_elapsed}s")
                    st.dataframe(_sc_df, width="stretch")

                    _csv_bytes = _sc_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download CSV",
                        data=_csv_bytes,
                        file_name="screener_results.csv",
                        mime="text/csv",
                    )

            except Exception as _sc_err:
                _prog.empty()
                st.error(f"Screener error: {_sc_err}")
                st.exception(_sc_err)
