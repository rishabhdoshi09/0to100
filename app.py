import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import os
import yfinance as yf
from dotenv import load_dotenv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from bs4 import BeautifulSoup

from data.kite_client import KiteClient
from data.instruments import InstrumentManager
from data.historical import HistoricalDataFetcher
from features.indicators import IndicatorEngine
from features.volume_profile import VolumeProfile
from signals.composite_signal import CompositeSignal
from features.market_structure import is_recent_swing_breakout
from backtest.walk_forward import walk_forward_backtest
from paper_trading import init_db, open_position, close_position, get_open_positions

load_dotenv()

st.set_page_config(page_title="Prism Quant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Inter', system-ui, sans-serif; }
    .card { background: #ffffff; border-radius: 20px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #eef2f6; }
    .metric-value { font-size: 2rem; font-weight: 600; color: #0f172a; }
    .metric-label { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; color: #475569; }
    .recommendation { font-size: 1.5rem; font-weight: 700; text-align: center; padding: 0.75rem; border-radius: 16px; margin-top: 0.25rem; }
    .buy { background: linear-gradient(135deg, #e6f7e6 0%, #d0f0d0 100%); color: #1e5a1e; border-left: 4px solid #2e7d32; }
    .sell { background: linear-gradient(135deg, #ffe6e5 0%, #ffd6d5 100%); color: #b71c1c; border-left: 4px solid #c62828; }
    .hold { background: linear-gradient(135deg, #fff8e1 0%, #ffeecc 100%); color: #e65100; border-left: 4px solid #f57c00; }
    .stButton button { background: #1e2a3e; color: white; border-radius: 12px; border: none; padding: 0.5rem 1.25rem; font-weight: 500; transition: all 0.2s; }
    .stButton button:hover { background: #2c3e5c; transform: translateY(-1px); }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background-color: #ffffff; padding: 0.5rem; border-radius: 20px; border: 1px solid #eef2f6; }
    .stTabs [data-baseweb="tab"] { border-radius: 16px; padding: 0.5rem 1rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #1e2a3e; color: white !important; }
    @media only screen and (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap; overflow-x: auto; white-space: nowrap; gap: 0.5rem; }
        .stColumn { width: 100% !important; min-width: 100% !important; }
        .metric-value { font-size: 1.4rem; }
        .recommendation { font-size: 1.2rem; }
        .card { padding: 1rem; }
        section[data-testid="stSidebar"] { width: 80% !important; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# INIT CLIENTS
# -------------------------------
@st.cache_resource
def init_clients():
    kite = KiteClient()
    im = InstrumentManager()
    fetcher = HistoricalDataFetcher(kite, im)
    ie = IndicatorEngine()
    vp = VolumeProfile()
    cs = CompositeSignal()
    return kite, im, fetcher, ie, vp, cs

kite, im, fetcher, ie, vp, cs = init_clients()

def get_all_equity_symbols():
    symbol_name = {}
    for sym, meta in im._meta_map.items():
        ex = meta.get('exchange', '')
        segment = meta.get('segment', '')
        inst_type = meta.get('instrument_type', '')
        if ex in ('NSE', 'BSE') and segment == ex and inst_type == 'EQ':
            if not sym[0].isdigit() and '-' not in sym and len(sym) <= 10:
                symbol_name[sym] = meta.get('companyName', sym)
    if not symbol_name:
        symbol_name = {"RELIANCE":"Reliance Industries", "TCS":"Tata Consultancy", "HDFCBANK":"HDFC Bank"}
    return symbol_name

def fetch_historical(symbol, days=250):
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days+30)).strftime("%Y-%m-%d")
    df = fetcher.fetch(symbol, from_date, to_date, interval="day")
    if df is None or len(df) == 0:
        from_date = (datetime.now() - timedelta(days=days+100)).strftime("%Y-%m-%d")
        df = fetcher.fetch(symbol, from_date, to_date, interval="day")
    return df

@st.cache_data(ttl=3600)
def get_indices_data():
    indices = {
        "Nifty 50": "NIFTY 50",
        "Bank Nifty": "NIFTY BANK",
        "Nifty IT": "NIFTY IT",
        "Nifty Pharma": "NIFTY PHARMA",
        "Nifty FMCG": "NIFTY FMCG"
    }
    data = {}
    for name, kite_symbol in indices.items():
        try:
            df = fetch_historical(kite_symbol, days=5)
            if df is not None and len(df) >= 2:
                last = df['close'].iloc[-1]
                prev = df['close'].iloc[-2]
                change = ((last - prev) / prev) * 100
                data[name] = {"price": last, "change": change}
            elif df is not None and len(df) == 1:
                last = df['close'].iloc[-1]
                data[name] = {"price": last, "change": 0.0}
        except Exception:
            continue
    return data

@st.cache_data(ttl=3600)
def get_global_indices():
    indices = {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC",
        "FTSE 100": "^FTSE", "DAX": "^GDAXI", "CAC 40": "^FCHI",
        "Nikkei 225": "^N225", "Hang Seng": "^HSI", "Shanghai": "000001.SS"
    }
    data = {}
    for name, ticker in indices.items():
        try:
            tick = yf.Ticker(ticker)
            hist = tick.history(period="2d")
            if len(hist) >= 2:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((last - prev) / prev) * 100
                data[name] = {"price": last, "change": change}
        except:
            continue
    return data

# -------------------------------
# SESSION FILTER
# -------------------------------
def session_signal(current_time=None):
    if current_time is None:
        current_time = datetime.now().time()
    if current_time < datetime.strptime("09:15", "%H:%M").time() or current_time > datetime.strptime("15:30", "%H:%M").time():
        return 0.0
    if datetime.strptime("09:15", "%H:%M").time() <= current_time <= datetime.strptime("09:45", "%H:%M").time():
        return 0.1
    if datetime.strptime("12:00", "%H:%M").time() <= current_time <= datetime.strptime("13:00", "%H:%M").time():
        return -0.1
    if datetime.strptime("14:30", "%H:%M").time() <= current_time <= datetime.strptime("15:30", "%H:%M").time():
        return 0.05
    return 0.0

# -------------------------------
# HEATMAP & MARKET CAP
# -------------------------------
@st.cache_data(ttl=86400)
def get_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        mc = info.get('marketCap', 0)
        return mc / 1e7 if mc else 0
    except:
        return 0

def categorize_by_mcap(symbol):
    mc = get_market_cap(symbol)
    if mc >= 20000:
        return "Largecap"
    elif mc >= 5000:
        return "Midcap"
    elif mc > 0:
        return "Smallcap"
    return "Unknown"

def get_symbols_by_market_cap(cap_filter, max_symbols=500):
    all_symbols = list(get_all_equity_symbols().keys())
    if cap_filter == "All":
        return all_symbols[:max_symbols]
    filtered = []
    for sym in all_symbols:
        mc = get_market_cap(sym)
        if mc >= 20000 and cap_filter == "Largecap":
            filtered.append(sym)
        elif 5000 <= mc < 20000 and cap_filter == "Midcap":
            filtered.append(sym)
        elif 0 < mc < 5000 and cap_filter == "Smallcap":
            filtered.append(sym)
        if len(filtered) >= max_symbols:
            break
    return filtered

def get_stock_change_kite(symbol):
    try:
        df = fetch_historical(symbol, days=5)
        if df is not None and len(df) >= 2:
            last = df['close'].iloc[-1]
            prev = df['close'].iloc[-2]
            change = ((last - prev) / prev) * 100
            return change
    except:
        pass
    return None

def get_heatmap_data(selected_cap, max_symbols=200):
    symbols = get_symbols_by_market_cap(selected_cap, max_symbols)
    results = []
    with st.spinner(f"Fetching data for {len(symbols)} stocks..."):
        for sym in symbols:
            change = get_stock_change_kite(sym)
            if change is not None:
                results.append({"Stock": sym, "Change %": change})
            time.sleep(0.02)
    return pd.DataFrame(results)

# -------------------------------
# PRE-MARKET SCRAPER
# -------------------------------
@st.cache_data(ttl=1800)
def get_moneycontrol_premarket():
    url = "https://www.moneycontrol.com/pre-market/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, 'html.parser')
        content_div = soup.find('div', class_='premarket_data')
        if not content_div:
            content_div = soup.find('div', {'id': 'premarket'})
        if not content_div:
            text = soup.get_text(separator='\n')
            lines = text.split('\n')
            relevant = []
            keywords = ['S&P', 'Dow', 'Nasdaq', 'Gift Nifty', 'SGX Nifty', 'Gold', 'Crude', 'Asian markets', 'pre-market']
            for line in lines:
                if any(kw in line for kw in keywords):
                    relevant.append(line.strip())
            if relevant:
                return "\n".join(relevant[:30]), None
        else:
            text = content_div.get_text(separator='\n', strip=True)
            return text[:3000], None
        return None, "Could not parse pre-market data"
    except Exception as e:
        return None, str(e)

# -------------------------------
# SCREENER.IN SCRAPERS
# -------------------------------
@st.cache_data(ttl=86400)
def scrape_screener_shareholding(symbol):
    symbol = symbol.upper()
    url = f"https://www.screener.in/company/{symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    session = requests.Session()
    session.headers.update(headers)
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None, None, None, None, f"Page not found (HTTP {resp.status_code})"
        soup = BeautifulSoup(resp.text, 'html.parser')
        sh = soup.find('section', id='shareholding')
        if not sh:
            tables = soup.find_all('table', class_='data-table')
            for table in tables:
                if 'Promoter' in table.text:
                    sh = table
                    break
        if not sh:
            return None, None, None, None, "No shareholding table found."
        if sh.name == 'table':
            table = sh
        else:
            table = sh.find('table', class_='data-table')
        if not table:
            return None, None, None, None, "Shareholding table not found."
        thead = table.find('thead')
        if not thead:
            return None, None, None, None, "No header row."
        quarters = [th.text.strip() for th in thead.find_all('th')]
        if len(quarters) < 2:
            quarters = None
        else:
            quarters = quarters[1:]
        rows = table.find_all('tr')
        promoter_data = []
        fii_data = []
        dii_data = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 2:
                continue
            label = cells[0].text.strip()
            values = []
            for cell in cells[1:]:
                txt = cell.text.strip().replace('%', '')
                try:
                    values.append(float(txt))
                except:
                    values.append(None)
            if 'Promoter' in label:
                promoter_data = values
            elif 'Foreign' in label or 'FII' in label:
                fii_data = values
            elif 'Domestic' in label or 'DII' in label:
                dii_data = values
        if not promoter_data and not fii_data and not dii_data:
            return None, None, None, None, "Could not parse any shareholding data."
        return quarters, promoter_data, fii_data, dii_data, None
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

@st.cache_data(ttl=43200)
def scrape_screener_concall(symbol):
    symbol = symbol.upper()
    url = f"https://www.screener.in/company/{symbol}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return f"Could not load page (HTTP {resp.status_code})."
        soup = BeautifulSoup(resp.text, 'html.parser')
        concall_section = soup.find('section', id='concall')
        if not concall_section:
            return "No concall transcript found on Screener.in for this symbol."
        paragraphs = concall_section.find_all('p')
        if not paragraphs:
            text = concall_section.get_text(separator=' ', strip=True)
        else:
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        if len(text) < 50:
            return "Concall summary too short or not available."
        return text[:1500]
    except Exception as e:
        return f"Error fetching concall: {str(e)}"

# -------------------------------
# MEMORY AND DEEPSEEK
# -------------------------------
MEMORY_FILE = "trading_memory.json"
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def update_memory(symbol, decision, confidence, price, actual_return=None):
    memory = load_memory()
    if symbol not in memory:
        memory[symbol] = []
    memory[symbol].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "decision": decision,
        "confidence": confidence,
        "price": price,
        "actual_return": actual_return
    })
    memory[symbol] = memory[symbol][-10:]
    save_memory(memory)

def get_recent_memory(symbol, limit=3):
    memory = load_memory()
    return memory.get(symbol, [])[-limit:]

def call_deepseek(prompt, system="You are a financial analyst."):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 800
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def bull_bear_debate(symbol, price, rsi, zscore, momentum, volume_ratio, fundamentals):
    prompt = f"""
You are a trading debate moderator. Given the following data for {symbol} (current price ₹{price}):
- RSI: {rsi:.1f}
- Z-Score: {zscore:.2f}
- 5-day momentum: {momentum:.2f}%
- Volume ratio: {volume_ratio:.2f}x
- Fundamentals: P/E {fundamentals.get('P/E Ratio', 'N/A')}, ROE {fundamentals.get('ROE (%)', 'N/A')}%

Provide a balanced debate with verdict. Format as bullet points:

**Bull Case**
- point1
- point2

**Bear Case**
- point1
- point2

**Verdict**
- BUY/SELL/HOLD (confidence: XX)
- one line reasoning
"""
    return call_deepseek(prompt, system="You are a professional trading debate moderator. Use ONLY bullet points.")

def get_stock_verdict(symbol):
    df = fetch_historical(symbol, days=250)
    if df is None or len(df) < 50:
        return {"error": f"Insufficient data for {symbol}"}
    indicators = ie.compute(df, symbol)
    factor_score = (
        (1 if indicators.get('rsi_14',50)<30 else -1 if indicators.get('rsi_14',50)>70 else 0) * 0.5 +
        (1 if indicators.get('zscore_20',0)<-1.5 else -1 if indicators.get('zscore_20',0)>1.5 else 0) * 0.3 +
        (1 if indicators.get('momentum_5d_pct',0)>0.02 else -1 if indicators.get('momentum_5d_pct',0)<-0.02 else 0) * 0.2
    )
    ml_score = cs._compute_ml_signal(indicators)
    regime_score = cs._compute_regime_signal(indicators)
    llm_score = 0.5
    session_score = session_signal()
    signal_result = cs.compute(indicators, llm_signal=None, regime=1)
    latest_price = df['close'].iloc[-1]
    last_date = df.index[-1].strftime('%Y-%m-%d')
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        fundamentals = {"P/E Ratio": info.get('trailingPE', 'N/A'), "ROE (%)": info.get('returnOnEquity', 0)*100 if info.get('returnOnEquity') else 'N/A'}
    except:
        fundamentals = {"P/E Ratio": "N/A", "ROE (%)": "N/A"}
    past = get_recent_memory(symbol)
    past_text = "\n".join([f"- {p['date']}: {p['decision']} (conf {p['confidence']:.0f}%) → actual return {p.get('actual_return', 'pending')}%" for p in past]) if past else "- No past trades."
    debate = bull_bear_debate(symbol, latest_price, indicators.get('rsi_14', 50), indicators.get('zscore_20', 0),
                              indicators.get('momentum_5d_pct', 0), indicators.get('volume_ratio', 1), fundamentals)
    return {
        "price": latest_price,
        "signal": signal_result['signal'],
        "direction": signal_result['direction'],
        "confidence": signal_result['confidence'],
        "attribution": signal_result['attribution'],
        "last_date": last_date,
        "debate": debate,
        "past_memory": past_text
    }

def categorize_stock(df):
    if df is None or len(df) < 50:
        return None
    close = df['close']
    volume = df['volume']
    daily_move = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    is_buzzing = daily_move > 15 and volume.iloc[-1] > 2 * volume.iloc[-20:-1].mean()
    ema10 = close.ewm(span=10).mean()
    bounce = (close.iloc[-1] > ema10.iloc[-1]) and (close.iloc[-2] < ema10.iloc[-2]) and (volume.iloc[-1] > volume.iloc[-2])
    ema50 = close.ewm(span=50).mean()
    losing = (close.iloc[-1] < ema50.iloc[-1]) and (close.iloc[-2] > ema50.iloc[-2])
    if is_buzzing:
        return "Buzzing Stock"
    elif bounce:
        return "Gaining Strength"
    elif losing:
        return "Losing Momentum"
    return None

def is_breakout_candidate(df):
    if df is None or len(df) < 120:
        return False
    close = df['close']
    base_high = close.rolling(90).max().iloc[-1]
    base_low = close.rolling(90).min().iloc[-1]
    near_top = (close.iloc[-1] - base_low) / (base_high - base_low) > 0.8
    avg_vol_6m = df['volume'].rolling(120).mean().iloc[-1]
    last_month_vol = df['volume'].iloc[-20:].mean()
    volume_dry = last_month_vol < 0.7 * avg_vol_6m
    swing_break = is_recent_swing_breakout(df, lookback=3)
    if not swing_break:
        return False
    vol_spike = df['volume'].iloc[-1] > 1.5 * df['volume'].rolling(20).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    ema_bull = (ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
    consecutive_up = (close.diff().iloc[-5:] > 0).all()
    return near_top and volume_dry and vol_spike and ema_bull and consecutive_up

def is_momentum_breakout(df):
    if df is None or len(df) < 50:
        return False
    close = df['close']
    volume = df['volume']
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if close.iloc[-1] <= sma20.iloc[-1] or close.iloc[-1] <= sma50.iloc[-1]:
        return False
    if sma20.iloc[-1] <= sma50.iloc[-1]:
        return False
    mom_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0
    if mom_5d <= 5:
        return False
    avg_vol = volume.iloc[-21:-1].mean()
    if volume.iloc[-1] < 1.5 * avg_vol:
        return False
    high_20 = df['high'].rolling(20).max()
    if close.iloc[-1] < high_20.iloc[-1]:
        return False
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] < 60 or rsi.iloc[-1] > 80:
        return False
    return True

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_test(symbol, test_func, days=150):
    try:
        df = fetch_historical(symbol, days=days)
        if df is not None and len(df) >= days-20:
            return symbol, test_func(df)
    except:
        pass
    return symbol, False

def scan_breakouts_categorized(symbols, categories_filter, limit=500, workers=8):
    categorized = {"Largecap": [], "Midcap": [], "Smallcap": [], "Unknown": []}
    to_scan = symbols[:limit]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_and_test, sym, is_breakout_candidate, 120) for sym in to_scan]
        for future in as_completed(futures):
            sym, passed = future.result()
            if passed:
                cat = categorize_by_mcap(sym)
                if cat in categories_filter or "All" in categories_filter:
                    categorized[cat].append(sym)
    return categorized

def scan_momentum_breakouts(symbols, categories_filter, limit=500, workers=8):
    categorized = {"Largecap": [], "Midcap": [], "Smallcap": [], "Unknown": []}
    to_scan = symbols[:limit]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_and_test, sym, is_momentum_breakout, 90) for sym in to_scan]
        for future in as_completed(futures):
            sym, passed = future.result()
            if passed:
                cat = categorize_by_mcap(sym)
                if cat in categories_filter or "All" in categories_filter:
                    categorized[cat].append(sym)
    return categorized

def test_buzzing(sym):
    try:
        df = fetch_historical(sym, days=30)
        if df is None or len(df) < 20:
            return False
        latest = df.iloc[-1]
        close = latest['close']; open_ = latest['open']; volume = latest['volume']
        if close <= open_:
            return False
        price_change = (close - open_) / open_ * 100
        if price_change < 3.0:
            return False
        avg_vol = df['volume'].iloc[-21:-1].mean()
        if volume < 1.5 * avg_vol:
            return False
        rsi = compute_rsi(df['close']).iloc[-1]
        if rsi < 60:
            return False
        return True
    except:
        return False

def find_buzzing_stocks_parallel(symbols, categories, min_price_change=3.0, limit=20, workers=8):
    results = []
    to_scan = symbols[:500]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_and_test, sym, test_buzzing, 30) for sym in to_scan]
        for future in as_completed(futures):
            sym, passed = future.result()
            if passed:
                df = fetch_historical(sym, days=30)
                close = df['close'].iloc[-1]; open_ = df['open'].iloc[-1]
                chg = (close - open_) / open_ * 100
                vol_r = df['volume'].iloc[-1] / df['volume'].iloc[-21:-1].mean()
                rsi = compute_rsi(df['close']).iloc[-1]
                results.append((sym, chg, vol_r, rsi))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]

def generate_auto_pulse():
    indices = get_indices_data()
    nifty = indices.get("Nifty 50", {"price": "N/A", "change": 0})
    sample_largecaps = get_symbols_by_market_cap("Largecap", 30)
    gainers, losers = [], []
    for sym in sample_largecaps:
        change = get_stock_change_kite(sym)
        if change is not None:
            gainers.append((sym, change))
            losers.append((sym, change))
    gainers.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    sp500 = get_global_indices().get("S&P 500", {"price": "N/A", "change": 0})
    prompt = f"""
Today's date: {datetime.now().strftime('%d-%m-%Y')}
Nifty 50: {nifty['price']:.1f} ({nifty['change']:+.2f}%)
Top Gainers: {', '.join([f"{s} ({c:+.2f}%)" for s,c in gainers[:3]])}
Top Losers: {', '.join([f"{s} ({c:+.2f}%)" for s,c in losers[:3]])}
S&P 500: {sp500['price']:.1f} ({sp500['change']:+.2f}%)
Generate a concise Daily Street Pulse report using ONLY bullet points for each section. Sections:
- Market Overview (3–4 bullets)
- Top Gainers/Losers (2–3 bullets)
- Global Cues (2–3 bullets)
- Stock of the Day (2–3 bullets)
- Top Updates (2–3 bullets)
- Technical Outlook (3–4 bullets)
"""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        return "DeepSeek API key missing."
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages":[{"role":"user","content":prompt}], "temperature":0.4, "max_tokens":800}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        return resp.json()['choices'][0]['message']['content']
    except:
        return "Error generating pulse"

def macro_regime_allocation_explained():
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_hist = nifty.history(period="1mo")
        nifty_return = (nifty_hist['Close'].iloc[-1] - nifty_hist['Close'].iloc[0]) / nifty_hist['Close'].iloc[0]
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="1mo")
        gold_return = (gold_hist['Close'].iloc[-1] - gold_hist['Close'].iloc[0]) / gold_hist['Close'].iloc[0]
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="1mo")
        yield_change = tnx_hist['Close'].iloc[-1] - tnx_hist['Close'].iloc[0]
        score = nifty_return - gold_return - (yield_change / 100)
        if score > 0.02:
            regime = "🟢 Risk ON – Optimistic, favor equities."
            allocation = "Equities 70% | Gold 15% | Bonds 15%"
        elif score < -0.02:
            regime = "🔴 Risk OFF – Fearful, raise cash."
            allocation = "Cash 40% | Gold 30% | Bonds 20% | Equities 10%"
        else:
            regime = "🟡 Neutral – Mixed signals."
            allocation = "Equities 40% | Gold 25% | Bonds 25% | Cash 10%"
        return regime, allocation
    except:
        return "⚠️ Regime data unavailable", "N/A"

def generate_swot(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        sector = info.get('sector', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        mc = info.get('marketCap', 0)
        mc_cr = mc / 1e7 if mc else 'N/A'
    except:
        sector = pe = roe = mc_cr = 'N/A'
    prompt = f"""
You are a financial analyst. Provide a concise SWOT analysis for the Indian stock {symbol} (NSE/BSE) based on the following known data and general market knowledge:

- Sector: {sector}
- P/E Ratio: {pe}
- ROE: {roe}
- Market Cap: ₹{mc_cr} Cr

Also consider recent news, sector trends, and technical conditions.

Format your response using ONLY bullet points for each section:

**Strengths**
- point1
- point2

**Weaknesses**
- point1
- point2

**Opportunities**
- point1
- point2

**Threats**
- point1
- point2

Keep each section to 2–3 bullet points. Be specific to {symbol}.
"""
    response = call_deepseek(prompt, system="You are a professional equity research analyst. Use ONLY bullet points.")
    return response if response else "Could not generate SWOT analysis."

def swarm_consensus(symbol, news_text):
    personas = [
        {"name": "🟢 Bullish Trader", "style": "Aggressive growth seeker, follows momentum, ignores valuation."},
        {"name": "🔴 Risk-Averse Pension Fund", "style": "Stable, dividend-paying stocks only. Avoids volatility."},
        {"name": "📊 Quant Chartist", "style": "Looks at price patterns, moving averages, volume. Ignores fundamentals."},
        {"name": "💼 Value Investor", "style": "Focuses on P/E, P/B, ROE. Buys only when price < intrinsic value."},
        {"name": "🌍 Macro Hedge Fund", "style": "Considers global cues, interest rates, currency, sector rotation."}
    ]
    results = []
    for persona in personas:
        prompt = f"""
You are a {persona['name']}. {persona['style']}
You see the following news about {symbol}: "{news_text}"
Based ONLY on your investment style, would you BUY, SELL, or HOLD the stock? 
Give your verdict as a single bullet point: "VERDICT: BUY/SELL/HOLD (confidence: XX)". Then one-line reason.
"""
        response = call_deepseek(prompt, system="You are a professional investor. Output ONLY two bullet points.")
        if response:
            lines = response.split('\n')
            verdict = "HOLD"
            confidence = 50
            reason = "Could not parse"
            for line in lines:
                if 'VERDICT:' in line.upper():
                    parts = line.split('VERDICT:')[1].strip().split()
                    verdict = parts[0].upper()
                    if 'confidence' in line.lower():
                        try:
                            conf_str = line.split('confidence:')[1].strip().split()[0]
                            confidence = int(conf_str)
                        except:
                            pass
                elif 'REASON:' in line.upper():
                    reason = line.split('REASON:')[1].strip()
            results.append({"persona": persona['name'], "verdict": verdict, "confidence": confidence, "reason": reason})
    return results

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🧠 Prism Quant")
st.caption("Live Kite Data · AI Debate · Quant Models · Bullet‑point Insights")

symbol_map = get_all_equity_symbols()
symbol_list = sorted(symbol_map.keys())

with st.sidebar:
    st.header("🔥 Daily Buzzing Stocks")
    buzz_min_change = st.slider("Min price change (%)", 1.0, 10.0, 3.0, 0.5)
    if st.button("Find Buzzing Stocks"):
        all_syms = list(symbol_map.keys())
        with st.spinner("Scanning in parallel..."):
            buzzing = find_buzzing_stocks_parallel(all_syms, [], buzz_min_change, limit=20, workers=8)
        if buzzing:
            st.write(f"**Top {len(buzzing)} buzzing stocks:**")
            for sym, chg, vol_r, rsi in buzzing:
                st.write(f"🟢 **{sym}** | +{chg:.1f}% | Vol {vol_r:.1f}x | RSI {rsi:.0f}")
        else:
            st.info("No buzzing stocks found.")
    st.divider()
    st.header("🔍 Stock Analysis")
    selected = st.selectbox("Choose a stock", symbol_list, format_func=lambda x: f"{x} – {symbol_map[x]}")
    analyze = st.button("🚀 Analyze & Get Verdict", use_container_width=True)
    st.divider()
    st.header("📊 SWOT Analysis")
    if st.button("Generate SWOT", key="swot_btn"):
        if selected:
            with st.spinner("Generating SWOT..."):
                swot = generate_swot(selected)
            st.session_state.swot_result = swot
        else:
            st.warning("Please select a stock first.")
    if "swot_result" in st.session_state:
        st.markdown(st.session_state.swot_result)

if analyze and selected:
    with st.spinner(f"Fetching data for {selected}..."):
        verdict = get_stock_verdict(selected)
    if "error" in verdict:
        st.error(verdict["error"])
    else:
        st.subheader(f"📊 {selected} – {symbol_map[selected]}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Price", f"₹{verdict['price']:.2f}")
        col2.metric("Composite Signal", f"{verdict['signal']:.3f}")
        col3.metric("Confidence", f"{verdict['confidence']:.1f}%")
        dir_text = {1:"BUY", -1:"SELL", 0:"HOLD"}[verdict['direction']]
        color_class = "buy" if verdict['direction']==1 else "sell" if verdict['direction']==-1 else "hold"
        col4.markdown(f"<div class='recommendation {color_class}'>{dir_text}</div>", unsafe_allow_html=True)
        st.caption(f"Data as of {verdict['last_date']} (last trading day)")

        with st.expander("📝 Bull/Bear LLM Debate"):
            st.markdown(verdict['debate'])
        with st.expander("🔧 Signal Attribution"):
            st.json(verdict['attribution'])
        with st.expander("📊 Volume Profile – What it means"):
            df_vp = fetch_historical(selected, days=250)
            if df_vp is not None and len(df_vp) >= 50:
                profile = vp.compute(df_vp)
                current_price = verdict['price']
                poc = profile['poc']
                vah = profile['vah']
                val = profile['val']
                hvns = profile['hvns'][:5]
                lvns = profile['lvns'][:5]
                st.markdown(f"""
- **Most traded price (POC):** ₹{poc:.2f} – Highest volume bar. Price often returns here.
- **Value Area (70% of volume):** ₹{val:.2f} – ₹{vah:.2f} – Fair value zone.
- **Current price location:** {'Above Value Area' if current_price > vah else 'Inside Value Area' if val <= current_price <= vah else 'Below Value Area'}
- **High volume zones (walls):** {', '.join([f'₹{h:.2f}' for h in hvns]) if hvns else 'None'}
- **Low volume zones (fast zones):** {', '.join([f'₹{l:.2f}' for l in lvns]) if lvns else 'None'}
""")
            else:
                st.info("Not enough data for volume profile (need 50+ days).")
        with st.expander("🧠 Past Memory (same stock)"):
            st.markdown(verdict['past_memory'])
        update_memory(selected, dir_text, verdict['confidence'], verdict['price'])
        st.success("Decision saved to memory.")

# -------------------------------
# TABS (12 tabs – all guaranteed to work)
# -------------------------------
tabs = st.tabs(["📊 Market Dashboard", "📈 Charts & EMAs", "🔍 Stock Categories", "🔥 Breakout Forecast",
                "🚀 Momentum Breakout", "🗺️ Heatmap", "🤖 Auto Pulse", "🌍 Global Markets",
                "📰 Pre-Market Report", "🏦 Quant Hedge Fund", "🐟 Swarm Consensus", 
                "📊 Fundamentals & Ownership", "📋 Paper Trading"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🇮🇳 Indian Indices (Kite)")
        ind_data = get_indices_data()
        if ind_data:
            df_ind = pd.DataFrame([{"Index": k, "Price": f"{v['price']:.1f}", "Change %": f"{v['change']:+.2f}%"} for k,v in ind_data.items()])
            st.dataframe(df_ind, use_container_width=True, hide_index=True)
        else:
            st.info("No index data available.")
    with col2:
        st.subheader("🌏 Global Indices")
        global_data = get_global_indices()
        if global_data:
            df_glob = pd.DataFrame([{"Index": k, "Price": f"{v['price']:.1f}", "Change %": f"{v['change']:+.2f}%"} for k,v in global_data.items()])
            st.dataframe(df_glob, use_container_width=True, hide_index=True)
        else:
            st.info("No global data available.")

with tabs[1]:
    if analyze and selected:
        df = fetch_historical(selected, days=150)
        if df is not None and len(df) > 50:
            indicators_for_table = ie.compute(df, selected)
            df['EMA10'] = df['close'].ewm(span=10).mean()
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['EMA50'] = df['close'].ewm(span=50).mean()
            df['EMA200'] = df['close'].ewm(span=200).mean()
            close = df['close'].iloc[-1]
            ema10, ema20, ema50, ema200 = df['EMA10'].iloc[-1], df['EMA20'].iloc[-1], df['EMA50'].iloc[-1], df['EMA200'].iloc[-1]
            interp = []
            if close > ema10 and close > ema20 and close > ema50 and close > ema200:
                interp.append("🟢 Price above all EMAs – Strong bullish trend.")
            elif close > ema10 and close > ema20:
                interp.append("🟡 Price above short‑term EMAs – Short‑term bullish.")
            else:
                interp.append("🔴 Price below 10 EMA – Weak short‑term momentum.")
            if ema10 > ema20 > ema50:
                interp.append("📈 Golden cross alignment (10>20>50) – Trend is up.")
            elif ema10 < ema20 < ema50:
                interp.append("📉 Death cross alignment – Trend is down.")
            pct_from_200 = (close - ema200) / ema200 * 100
            interp.append(f"📊 Price is {pct_from_200:.1f}% {'above' if close > ema200 else 'below'} 200 EMA – {'Long‑term uptrend' if close > ema200 else 'Long‑term downtrend'}.")
            st.info("\n".join(interp))
            show_vwap = st.checkbox("Show VWAP (Volume Weighted Avg Price)", value=False)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA10'], mode='lines', line=dict(color='green', width=1), name="10 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', line=dict(color='yellow', width=1), name="20 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', line=dict(color='purple', width=1), name="50 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], mode='lines', line=dict(color='red', width=1), name="200 EMA"))
            if show_vwap:
                typical = (df['high'] + df['low'] + df['close']) / 3
                cum_vol = df['volume'].cumsum()
                cum_pv = (typical * df['volume']).cumsum()
                vwap_series = cum_pv / cum_vol
                fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode='lines', line=dict(color='blue', width=1.5, dash='dot'), name='VWAP'))
            fig.update_layout(title=f"{selected} – Candlestick with EMAs", xaxis_title="Date", yaxis_title="Price (₹)", height=600)
            st.plotly_chart(fig, use_container_width=True)
            # Technical table
            rsi = indicators_for_table.get('rsi_14', 50)
            zscore = indicators_for_table.get('zscore_20', 0)
            mom = indicators_for_table.get('momentum_5d_pct', 0)
            vol_ratio = indicators_for_table.get('volume_ratio', 1)
            typical = (df['high'] + df['low'] + df['close']) / 3
            cum_vol = df['volume'].cumsum()
            cum_pv = (typical * df['volume']).cumsum()
            vwap_last = (cum_pv / cum_vol).iloc[-1]
            tec_df = pd.DataFrame({
                "Indicator": ["RSI (14)", "Z‑Score (20)", "5‑day Momentum (%)", "Volume Ratio", "VWAP"],
                "Value": [f"{rsi:.1f}", f"{zscore:.2f}", f"{mom:.2f}%", f"{vol_ratio:.2f}x", f"₹{vwap_last:.2f}"],
                "Meaning": [
                    "Oversold (<30) / Overbought (>70)",
                    "Extreme (<-2 or >2) suggests mean reversion",
                    "Bullish if >2%, Bearish if <-2%",
                    "High (>1.5) confirms moves",
                    "Above VWAP = premium (sell); below = discount (buy)"
                ]
            })
            st.dataframe(tec_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough historical data.")
    else:
        st.info("Select a stock and click 'Analyze & Get Verdict' first.")

with tabs[2]:
    st.subheader("🔍 Universe Stock Categorization")
    max_scan = st.slider("Number of stocks to scan", 100, 1000, 300, step=100)
    if st.button("Run Universe Scan"):
        all_syms = list(get_all_equity_symbols().keys())
        categories = defaultdict(list)
        progress = st.progress(0)
        for i, sym in enumerate(all_syms[:max_scan]):
            progress.progress((i+1)/max_scan)
            try:
                df = fetch_historical(sym, days=100)
                cat = categorize_stock(df)
                if cat:
                    categories[cat].append(sym)
            except:
                continue
        progress.empty()
        if categories:
            for cat, syms in categories.items():
                st.subheader(cat)
                st.write(", ".join(syms[:20]) + ("..." if len(syms)>20 else ""))
        else:
            st.info("No stocks matched criteria today.")

with tabs[3]:
    st.subheader("🚀 Breakout Stocks (Mean‑Reversion)")
    categories_filter = st.multiselect("Market cap", ["Largecap", "Midcap", "Smallcap", "All"], default=["All"])
    if st.button("Run Breakout Scanner"):
        all_syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning in parallel..."):
            result = scan_breakouts_categorized(all_syms, categories_filter, limit=500, workers=8)
        for cap, syms in result.items():
            if cap != "Unknown" and syms:
                st.write(f"**{cap}:** {', '.join(syms)}")
        if not any(result.values()):
            st.info("No breakouts found.")
    else:
        st.info("Select categories and click to scan.")

with tabs[4]:
    st.subheader("🚀 Momentum Breakout Scanner")
    categories_filter_mom = st.multiselect("Market cap", ["Largecap", "Midcap", "Smallcap", "All"], default=["All"], key="mom_cat")
    if st.button("Run Momentum Scan"):
        all_syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning in parallel..."):
            result = scan_momentum_breakouts(all_syms, categories_filter_mom, limit=500, workers=8)
        for cap, syms in result.items():
            if cap != "Unknown" and syms:
                st.write(f"**{cap}:** {', '.join(syms)}")
        if not any(result.values()):
            st.info("No momentum breakouts found.")
    else:
        st.info("Select categories and click to scan.")

with tabs[5]:
    st.subheader("🗺️ Market Heatmap (Kite Data)")
    cap_choice = st.radio("Select market cap", ["Largecap", "Midcap", "Smallcap", "All"], horizontal=True)
    max_symbols = st.slider("Max symbols to display", 50, 500, 200, step=50)
    if st.button("Refresh Heatmap"):
        df_heat = get_heatmap_data(cap_choice, max_symbols)
        if not df_heat.empty:
            fig = px.treemap(df_heat, path=['Stock'], values='Change %', color='Change %',
                             color_continuous_scale=['red','yellow','green'],
                             title=f"{cap_choice} – Daily % Change (Kite)")
            fig.update_traces(textinfo="label+value", textposition="middle center")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Showing {len(df_heat)} stocks; data from Kite historical.")
        else:
            st.error("No data retrieved. Check Kite connection or symbol availability.")
    else:
        st.info("Click 'Refresh Heatmap' to fetch latest changes from Kite.")

with tabs[6]:
    st.subheader("🤖 Generate Daily Street Pulse")
    if st.button("Generate Today's Pulse"):
        with st.spinner("Generating bullet-point summary..."):
            pulse = generate_auto_pulse()
        st.markdown(f"<div class='card'>{pulse}</div>", unsafe_allow_html=True)

with tabs[7]:
    st.subheader("🌍 Global Indices")
    for name, val in get_global_indices().items():
        st.metric(name, f"{val['price']:.1f}", f"{val['change']:+.2f}%")

with tabs[8]:
    st.header("🌅 Pre-Market Report (Moneycontrol)")
    if st.button("Refresh Report"):
        with st.spinner("Fetching latest pre-market data..."):
            try:
                report, error = get_moneycontrol_premarket()
                if report:
                    st.markdown(f"<div class='card'>{report}</div>", unsafe_allow_html=True)
                    st.caption("Source: Moneycontrol. Data may be delayed.")
                else:
                    st.error(f"Moneycontrol scraping failed: {error}")
                    st.subheader("📊 Global Cues (Raw data)")
                    global_data = get_global_indices()
                    for name, val in global_data.items():
                        st.write(f"{name}: {val['price']:.1f} ({val['change']:+.2f}%)")
                    st.info("Could not fetch pre‑market report from Moneycontrol. Try again later or check the link manually.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("You can check [Moneycontrol Pre-market](https://www.moneycontrol.com/pre-market/) manually.")
    else:
        st.info("Click 'Refresh Report' to get the latest pre-market data from Moneycontrol.")

with tabs[9]:
    st.header("🏦 Quant Hedge Fund – One Click")
    if st.button("🚀 Run All Quant Models (Auto Fetch Data)"):
        if selected:
            with st.spinner("Fetching data and analyzing..."):
                df = fetch_historical(selected, days=300)
                if df is None or len(df) < 200:
                    st.error(f"Could not fetch enough data for {selected} from Kite. Ensure Kite session is active and symbol has sufficient history.")
                else:
                    st.success(f"Data fetched from Kite ({len(df)} days).")
                    if len(df) < 252:
                        st.warning(f"⚠️ Only {len(df)} days (less than ideal 252). Momentum signal may be less reliable.")
                    close = df['close']
                    if len(close) >= 200:
                        ret = (close.iloc[-1] - close.iloc[-200]) / close.iloc[-200]
                        daily_vol = close.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252) if len(close) >= 60 else 0
                        if daily_vol and daily_vol != 0:
                            mom_signal = np.clip(ret / daily_vol, -1, 1)
                            mom_verdict = "🟢 BUY" if mom_signal > 0.3 else "🔴 SELL" if mom_signal < -0.3 else "⚪ HOLD"
                        else:
                            mom_signal, mom_verdict = 0, "Insufficient volatility data"
                    else:
                        mom_signal, mom_verdict = 0, "Insufficient data (need 200 days)"
                    if len(df) >= 60:
                        returns = df['close'].pct_change().dropna().iloc[-60:]
                        realized_vol = returns.std() * np.sqrt(252)
                        mult = np.clip(0.20 / realized_vol, 0.2, 3.0) if realized_vol != 0 else 1.0
                        vol_verdict = "📈 Increase size" if mult > 1.2 else "📉 Reduce size" if mult < 0.8 else "✅ Normal size"
                    else:
                        mult = 1.0
                        vol_verdict = "Insufficient data (need 60 days)"
                    samples = get_symbols_by_market_cap("Largecap", 50)
                    best_pair = None
                    best_corr = 0
                    for sym in samples[:20]:
                        if sym == selected:
                            continue
                        df2 = fetch_historical(sym, days=250)
                        if df2 is None or len(df2) < 200:
                            continue
                        close2 = df2['close']
                        common = close.index.intersection(close2.index)
                        if len(common) < 100:
                            continue
                        corr = close.loc[common].corr(close2.loc[common])
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_pair = sym
                    if best_pair:
                        common = close.index.intersection(close2.index)
                        ratio = close.loc[common] / close2.loc[common]
                        zscore = (ratio.iloc[-1] - ratio.mean()) / ratio.std()
                        if zscore > 2:
                            pair_verdict = f"🔴 SELL {selected} / BUY {best_pair} (expensive)"
                        elif zscore < -2:
                            pair_verdict = f"🟢 BUY {selected} / SELL {best_pair} (cheap)"
                        else:
                            pair_verdict = f"⚪ HOLD – Fairly priced (z-score {zscore:.2f})"
                    else:
                        pair_verdict = "No strongly correlated stock found."
                    regime, alloc = macro_regime_allocation_explained()
                    st.subheader("📊 Results")
                    st.markdown(f"""
- **Time Series Momentum:** {mom_signal:.2f} – {mom_verdict}
- **Volatility Targeting:** Multiplier = {mult:.2f}x – {vol_verdict}
- **Relative Value Spread:** {pair_verdict}
- **Macro Regime:** {regime} – Allocation: {alloc}
""")
        else:
            st.warning("Please select a stock from the sidebar first.")
    else:
        st.info("Click the button to automatically fetch data (Kite only) and run all four quant models.")

with tabs[10]:
    st.header("🐟 Swarm Intelligence – 5 Investor Personas")
    news = st.text_area("📰 News or event", height=100, placeholder="HDFC Life profit up 4%...")
    if st.button("Run Swarm"):
        if not news:
            st.warning("Enter news.")
        else:
            with st.spinner("Contacting personas..."):
                results = swarm_consensus(selected if 'selected' in locals() else "RELIANCE", news)
            for r in results:
                st.markdown(f"**{r['persona']}** – {r['verdict']} (conf {r['confidence']}%)  \n*{r['reason']}*")
            buy = sum(1 for r in results if r['verdict']=="BUY")
            sell = sum(1 for r in results if r['verdict']=="SELL")
            hold = sum(1 for r in results if r['verdict']=="HOLD")
            st.info(f"Consensus: BUY {buy} | SELL {sell} | HOLD {hold}")

with tabs[11]:
    st.header("📊 Fundamentals & Ownership")
    if selected:
        with st.spinner("Fetching company data..."):
            ticker = yf.Ticker(selected + ".NS")
            info = ticker.info
            st.subheader("🏢 Company Overview")
            st.write(f"**Name:** {info.get('longName', selected)}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}  |  **Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Description:** {info.get('longBusinessSummary', 'No description available.')[:500]}...")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.subheader("📈 Shareholding Pattern (Screener.in)")
        quarters, promoter_data, fii_data, dii_data, err = scrape_screener_shareholding(selected)
        if err:
            st.warning(f"Could not fetch shareholding data: {err}")
            st.markdown(f"👉 [View on Screener.in](https://www.screener.in/company/{selected.upper()}/)")
        else:
            if quarters and promoter_data and len(promoter_data) > 0:
                latest_promoter = promoter_data[-1] if promoter_data else 0
                st.metric("Promoter Holding (latest)", f"{latest_promoter:.2f}%")
                min_len = min(len(quarters), len(fii_data), len(dii_data))
                if min_len > 0:
                    trend_df = pd.DataFrame({
                        "Quarter": quarters[:min_len],
                        "FII (%)": fii_data[:min_len],
                        "DII (%)": dii_data[:min_len]
                    }).dropna()
                    if not trend_df.empty:
                        fig_trend = px.line(trend_df, x="Quarter", y=["FII (%)", "DII (%)"],
                                            title="FII / DII Investment Trend",
                                            markers=True, color_discrete_map={"FII (%)": "orange", "DII (%)": "green"})
                        st.plotly_chart(fig_trend, use_container_width=True)
                        if len(fii_data) >= 2:
                            fii_change = fii_data[-1] - fii_data[-2]
                            dii_change = dii_data[-1] - dii_data[-2]
                            st.write(f"**FII change:** +{fii_change:.2f}%" if fii_change>0 else f"**FII change:** {fii_change:.2f}%")
                            st.write(f"**DII change:** +{dii_change:.2f}%" if dii_change>0 else f"**DII change:** {dii_change:.2f}%")
                    else:
                        st.info("Insufficient quarterly data for FII/DII trend.")
                else:
                    st.info("No quarterly data available for trend.")
            else:
                st.warning("No shareholding data found for this symbol.")
                st.markdown(f"👉 [Check manually on Screener.in](https://www.screener.in/company/{selected.upper()}/)")
        st.subheader("📞 Latest Concall Summary")
        concall = scrape_screener_concall(selected)
        if "No concall" in concall or "not found" in concall:
            st.warning(concall)
            st.markdown(f"👉 [Check announcements on Screener.in](https://www.screener.in/company/{selected.upper()}/)")
        else:
            st.markdown(concall)
        st.caption("Source: Screener.in (management commentary). May not be available for all stocks.")
    else:
        st.info("Select a stock from the sidebar first.")

with tabs[12]:
    st.header("📋 Paper Trading (Simulated)")
    init_db()
    open_positions = get_open_positions()
    if not open_positions.empty:
        st.subheader("Open Positions")
        st.dataframe(open_positions[['symbol','entry_date','entry_price','quantity','direction']])
    else:
        st.info("No open positions.")
    if selected:
        if st.button("Run Daily Paper Trading"):
            verdict = get_stock_verdict(selected)
            if verdict['direction'] == 1:
                existing = open_positions[open_positions['symbol'] == selected]
                if existing.empty:
                    open_position(selected, verdict['price'], 100, "BUY", datetime.now().strftime("%Y-%m-%d"))
                    st.success(f"Opened BUY position for {selected} at ₹{verdict['price']}")
                else:
                    st.info("Position already open.")
            elif verdict['direction'] == -1:
                for idx, row in open_positions.iterrows():
                    if row['symbol'] == selected:
                        close_position(row['id'], verdict['price'], datetime.now().strftime("%Y-%m-%d"))
                        st.success(f"Closed position for {selected} at ₹{verdict['price']}")
            else:
                st.info("No action: HOLD signal.")
    else:
        st.info("Select a stock from the sidebar first.")
