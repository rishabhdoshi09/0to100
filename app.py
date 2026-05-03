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
from signals.composite_signal import CompositeSignal
from sq_ai.signals.conviction import ConvictionScorer
from sq_ai.signals.profiles import PROFILES
from sq_ai.signals.trade_setup import compute_trade_setup

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prism Quant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp { background-color: #f8fafc; font-family: 'Inter', system-ui, sans-serif; }
  .card  { background: #fff; border-radius: 20px; padding: 1.25rem; margin-bottom: 1.5rem;
           box-shadow: 0 4px 12px rgba(0,0,0,.03); border: 1px solid #eef2f6; }
  .recommendation { font-size: 1.5rem; font-weight: 700; text-align: center;
                    padding: .75rem; border-radius: 16px; margin-top: .25rem; }
  .buy  { background: linear-gradient(135deg,#e6f7e6,#d0f0d0); color:#1e5a1e; border-left:4px solid #2e7d32; }
  .sell { background: linear-gradient(135deg,#ffe6e5,#ffd6d5); color:#b71c1c; border-left:4px solid #c62828; }
  .hold { background: linear-gradient(135deg,#fff8e1,#ffeecc); color:#e65100; border-left:4px solid #f57c00; }
  .stButton button { background:#1e2a3e; color:white; border-radius:12px; border:none;
                     padding:.5rem 1.25rem; font-weight:500; transition:all .2s; }
  .stButton button:hover { background:#2c3e5c; transform:translateY(-1px); }
  .stTabs [data-baseweb="tab-list"] { gap:.5rem; background:#fff; padding:.5rem;
                                      border-radius:20px; border:1px solid #eef2f6; }
  .stTabs [data-baseweb="tab"]       { border-radius:16px; padding:.5rem 1rem; font-weight:500; }
  .stTabs [aria-selected="true"]     { background:#1e2a3e; color:white !important; }
  @media(max-width:768px){
    .stTabs [data-baseweb="tab-list"]{ flex-wrap:nowrap; overflow-x:auto; }
    .stColumn{ width:100% !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Cached resource init ───────────────────────────────────────────────────────
@st.cache_resource
def init_clients():
    kite    = KiteClient()
    im      = InstrumentManager()
    fetcher = HistoricalDataFetcher(kite, im)
    ie      = IndicatorEngine()
    vp      = VolumeProfile()
    cs      = CompositeSignal()
    claude  = ClaudeClient()
    return kite, im, fetcher, ie, vp, cs, claude

kite, im, fetcher, ie, vp, cs, _claude = init_clients()

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
    to_d   = datetime.now().strftime("%Y-%m-%d")
    from_d = (datetime.now() - timedelta(days=days+30)).strftime("%Y-%m-%d")
    df = fetcher.fetch(symbol, from_d, to_d, interval="day")
    if df is None or len(df) == 0:
        from_d = (datetime.now() - timedelta(days=days+100)).strftime("%Y-%m-%d")
        df = fetcher.fetch(symbol, from_d, to_d, interval="day")
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
                last, prev = h['Close'].iloc[-1], h['Close'].iloc[-2]
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
    indicators   = ie.compute(df, symbol)
    signal_result = cs.compute(indicators, llm_signal=None, regime=1)

    # Conviction score (new layer)
    conviction_result = ConvictionScorer().score(indicators)

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
        "signal":     signal_result['signal'],
        "direction":  signal_result['direction'],
        "confidence": signal_result['confidence'],
        "attribution":signal_result['attribution'],
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
        nifty_r = (lambda h: (h['Close'].iloc[-1]-h['Close'].iloc[0])/h['Close'].iloc[0])(yf.Ticker("^NSEI").history(period="1mo"))
        gold_r  = (lambda h: (h['Close'].iloc[-1]-h['Close'].iloc[0])/h['Close'].iloc[0])(yf.Ticker("GC=F").history(period="1mo"))
        yld_chg = (lambda h: h['Close'].iloc[-1]-h['Close'].iloc[0])(yf.Ticker("^TNX").history(period="1mo"))
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
    if st.button("🔄 Refresh All Data", use_container_width=True):
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
    analyze  = st.button("🚀 Analyze", use_container_width=True)

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
                st.dataframe(cdf, hide_index=True, use_container_width=True)
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
            with st.expander("🤖 Claude Final Signal"):
                if not _claude.available:
                    st.info("Set ANTHROPIC_API_KEY in .env to enable Claude.")
                elif st.button("Ask Claude", key="claude_btn"):
                    with st.spinner("Calling Claude…"):
                        _sig = _claude.get_signal(
                            f"Symbol: {_selected} | Price: ₹{verdict['price']:.2f} | "
                            f"Signal: {verdict['signal']:.3f} | Confidence: {verdict['confidence']:.1f}% | "
                            f"Conviction: {cv.score:.0f}/100 | Direction: {dir_text}\n\n"
                            f"DeepSeek analysis:\n{(verdict['debate'] or '')[:600]}"
                        )
                    if _sig:
                        _act = _sig.get("action","HOLD")
                        _cc  = _sig.get("confidence",0)
                        _, _scss = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(_act,0))
                        st.markdown(f"<div class='recommendation {_scss}'>{_act} · {_cc*100:.0f}%</div>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**Reasoning:** {_sig.get('reasoning','')}")
                        st.caption(f"Sentiment: {_sig.get('sentiment_score',0):.2f}")
                    else:
                        st.warning("No signal returned.")
            with st.expander("🧠 Trade Memory"):
                st.markdown(verdict['past_memory'])

        update_memory(_selected, dir_text, verdict['confidence'], verdict['price'])

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Market Dashboard",
    "📈 Charts & EMAs",
    "🎯 Decision Terminal",
    "🔍 Stock Categories",
    "🔥 Breakout Scanner",
    "🚀 Momentum Scanner",
    "🗺️ Heatmap",
    "🤖 Daily Pulse",
    "🐟 Swarm Intel",
    "🌍 Global Markets",
    "📰 Pre-Market",
    "🏦 Quant Hedge Fund",
    "📊 Fundamentals",
    "📋 Paper Trading",
])

# ── Tab 0: Market Dashboard ────────────────────────────────────────────────────
with tabs[0]:
    t0c1, t0c2 = st.columns(2)
    with t0c1:
        st.subheader("🇮🇳 Indian Indices")
        for n, v in get_indices_data().items():
            _index_card(n, v['price'], v['change'])
    with t0c2:
        st.subheader("🌏 Global Indices")
        for n, v in get_global_indices().items():
            _index_card(n, v['price'], v['change'], currency="")

# ── Tab 1: Charts & EMAs ───────────────────────────────────────────────────────
with tabs[1]:
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
            st.plotly_chart(fig, use_container_width=True)

            rsi_v = ind.get('rsi_14',50); zsc_v=ind.get('zscore_20',0); mom_v=ind.get('momentum_5d_pct',0); vol_v=ind.get('volume_ratio',1)  # noqa: E701,E702,E741
            tp2 = (df['high']+df['low']+df['close'])/3
            vwap_v = ((tp2*df['volume']).cumsum()/df['volume'].cumsum()).iloc[-1]
            st.dataframe(pd.DataFrame({
                "Indicator":["RSI(14)","Z-Score(20)","Momentum 5d","Volume Ratio","VWAP"],
                "Value":[f"{rsi_v:.1f}",f"{zsc_v:.2f}",f"{mom_v:.2f}%",f"{vol_v:.2f}x",f"₹{vwap_v:.2f}"],
                "Signal":["Oversold<30/OB>70","Extreme<-2|>2","Bullish>2%","High>1.5x","Below=discount"],
            }), hide_index=True, use_container_width=True)
    else:
        st.info("Select a stock and click Analyze first.")

# ── Tab 2: Decision Terminal ───────────────────────────────────────────────────
with tabs[2]:
    st.header("🎯 Decision Terminal")
    st.caption("Conviction score · ATR trade setup · Claude final opinion")

    dt_c1, dt_c2 = st.columns([1,2])
    with dt_c1:
        dt_profile = st.selectbox("Trader Profile", list(PROFILES.keys()), key="dt_p")
        dt_capital = st.number_input("Capital (₹)", value=100_000, step=10_000, key="dt_cap")
        dt_sym     = st.selectbox("Symbol", symbol_list, key="dt_sym",
                                   format_func=lambda x: f"{x} – {symbol_map[x]}")
        run_dt = st.button("▶ Run", use_container_width=True, key="dt_go")

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
            st.session_state["dt_res"]   = _res
            st.session_state["dt_setup"] = _setup
            st.session_state["dt_price"] = _price
            st.session_state["dt_ind"]   = _ind
            st.session_state["dt_sym"]   = dt_sym
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
            st.plotly_chart(fig_bar, use_container_width=True)

            # Claude final opinion for Decision Terminal
            if _claude.available:
                if st.button("🤖 Get Claude Opinion", key="dt_claude"):
                    with st.spinner("Asking Claude…"):
                        _c_ctx = (
                            f"Symbol: {st.session_state['dt_sym']} | Price: ₹{dt_price:.2f} | "
                            f"Conviction: {r.score:.0f}/100 | Verdict: {r.verdict} | "
                            f"Gates passed: {r.gates_passed}\n"
                            f"Components: {json.dumps({k:round(v,2) for k,v in comp.items()})}"
                        )
                        _cop = _claude.get_signal(_c_ctx)
                    if _cop:
                        _ca = _cop.get("action","HOLD")
                        _, _ccss = _verdict_badge({"BUY":1,"SELL":-1,"HOLD":0}.get(_ca,0))
                        st.markdown(f"<div class='recommendation {_ccss}'>"
                                    f"Claude: {_ca} · {_cop.get('confidence',0)*100:.0f}%</div>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**Reasoning:** {_cop.get('reasoning','')}")
            else:
                st.caption("Add ANTHROPIC_API_KEY to .env to enable Claude here.")

# ── Tab 3: Stock Categories ────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🔍 Stock Categorization")
    max_scan = st.slider("Stocks to scan", 100, 1000, 300, 100)
    if st.button("Run Scan"):
        all_syms = list(get_all_equity_symbols().keys())
        cats     = defaultdict(list)
        prog     = st.progress(0)
        for i, sym in enumerate(all_syms[:max_scan]):
            prog.progress((i+1)/max_scan)
            try:
                df = fetch_historical(sym, days=100)
                if df is None or len(df) < 50: continue  # noqa: E701,E702,E741
                close, vol = df['close'], df['volume']
                dm   = (close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100
                e10  = close.ewm(span=10).mean()
                e50  = close.ewm(span=50).mean()
                if dm > 15 and vol.iloc[-1] > 2*vol.iloc[-20:-1].mean():
                    cats["Buzzing Stock"].append(sym)
                elif close.iloc[-1]>e10.iloc[-1] and close.iloc[-2]<e10.iloc[-2] and vol.iloc[-1]>vol.iloc[-2]:
                    cats["Gaining Strength"].append(sym)
                elif close.iloc[-1]<e50.iloc[-1] and close.iloc[-2]>e50.iloc[-2]:
                    cats["Losing Momentum"].append(sym)
            except Exception:
                continue
        prog.empty()
        for cat, syms in cats.items():
            st.subheader(cat)
            st.write(", ".join(syms[:20]) + ("…" if len(syms)>20 else ""))
        if not cats:
            st.info("No stocks matched criteria today.")

# ── Tab 4: Breakout Scanner ────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🔥 Breakout Scanner")
    cap_filter_bo = st.multiselect("Market cap", ["Largecap","Midcap","Smallcap","All"], default=["All"])
    if st.button("Run Breakout Scan"):
        syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning…"):
            result = scan_parallel(syms, is_breakout_candidate, 120, cap_filter_bo)
        found = False
        for cap, s in result.items():
            if cap != "Unknown" and s:
                st.write(f"**{cap}:** {', '.join(s)}")
                found = True
        if not found: st.info("No breakouts found.")  # noqa: E701,E702,E741

# ── Tab 5: Momentum Scanner ────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🚀 Momentum Breakout Scanner")
    cap_filter_mo = st.multiselect("Market cap", ["Largecap","Midcap","Smallcap","All"], default=["All"], key="mo_cap")
    if st.button("Run Momentum Scan"):
        syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning…"):
            result = scan_parallel(syms, is_momentum_breakout, 90, cap_filter_mo)
        found = False
        for cap, s in result.items():
            if cap != "Unknown" and s:
                st.write(f"**{cap}:** {', '.join(s)}")
                found = True
        if not found: st.info("No momentum breakouts found.")  # noqa: E701,E702,E741

# ── Tab 6: Heatmap ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("🗺️ Market Heatmap")
    cap_hm  = st.radio("Market cap", ["Largecap","Midcap","Smallcap","All"], horizontal=True)
    max_hm  = st.slider("Max symbols", 50, 500, 200, 50)
    if st.button("Refresh Heatmap"):
        df_heat = get_heatmap_data(cap_hm, max_hm)
        if not df_heat.empty:
            fig = px.treemap(df_heat, path=['Stock'], values='Change %', color='Change %',
                             color_continuous_scale=['red','yellow','green'],
                             title=f"{cap_hm} — Daily % Change")
            fig.update_traces(textinfo="label+value", textposition="middle center")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data. Check Kite connection.")

# ── Tab 7: Daily Pulse ─────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("🤖 Daily Street Pulse (DeepSeek)")
    if st.button("Generate Pulse"):
        with st.spinner("Generating…"):
            pulse = generate_auto_pulse()
        st.markdown(f"<div class='card'>{pulse}</div>", unsafe_allow_html=True)

# ── Tab 8: Swarm Intelligence ──────────────────────────────────────────────────
with tabs[8]:
    st.header("🐟 Swarm Intelligence — 5 Personas")
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

# ── Tab 9: Global Markets ──────────────────────────────────────────────────────
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
                    st.plotly_chart(fig_fi, use_container_width=True)

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
        st.plotly_chart(fig_eq, use_container_width=True)

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
                         hide_index=True, use_container_width=True)
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
                         hide_index=True, use_container_width=True)
        else:
            st.info("No closed trades.")

    st.divider()
    st.subheader("⚡ Execute Paper Trade")
    pt_sym = st.selectbox("Symbol", symbol_list, key="pt_sym",
                           format_func=lambda x: f"{x} – {symbol_map[x]}")
    if st.button("▶ Run Signal & Trade", use_container_width=True):
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
