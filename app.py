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
import io
import PyPDF2
import json
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.kite_client import KiteClient
from data.instruments import InstrumentManager
from data.historical import HistoricalDataFetcher
from features.indicators import IndicatorEngine
from features.volume_profile import VolumeProfile
from signals.composite_signal import CompositeSignal

load_dotenv()

st.set_page_config(page_title="Prism Quant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f5f7fb; }
    .card { background-color: white; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e6e9f0; }
    .metric-value { font-size: 2rem; font-weight: 600; color: #1e2a3e; }
    .metric-label { font-size: 0.85rem; color: #5a6e8a; text-transform: uppercase; }
    .recommendation { font-size: 2rem; font-weight: bold; text-align: center; padding: 0.5rem; border-radius: 12px; margin-top: 0.5rem; }
    .buy { background-color: #e6f7e6; color: #2e7d32; border-left: 4px solid #2e7d32; }
    .sell { background-color: #ffe6e5; color: #c62828; border-left: 4px solid #c62828; }
    .hold { background-color: #fff8e1; color: #f57c00; border-left: 4px solid #f57c00; }
    .stButton button { background-color: #1e2a3e; color: white; border-radius: 8px; border: none; padding: 0.5rem 1.5rem; }
    .stButton button:hover { background-color: #2c3e5c; }
    .daily-pulse { background-color: #ffffff; border-radius: 16px; padding: 1.5rem; border: 1px solid #e0e5ec; font-family: 'Segoe UI', sans-serif; line-height: 1.5; }
    .debate { background-color: #fef9e6; border-left: 4px solid #ff9800; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .hedge-card { background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 5px solid #1e2a3e; }
</style>
""", unsafe_allow_html=True)

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
        symbol_name = {"RELIANCE":"Reliance Ind", "TCS":"TCS", "HDFCBANK":"HDFC Bank"}
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

@st.cache_data(ttl=3600)
def get_commodities():
    comm = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Copper": "HG=F"}
    data = {}
    for name, ticker in comm.items():
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

def get_nifty50_constituents():
    return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC", "AXISBANK", "BAJFINANCE",
            "WIPRO", "LT", "SUNPHARMA", "TITAN", "MARUTI", "NTPC", "POWERGRID",
            "ADANIPORTS", "ONGC", "TATAMOTORS", "NESTLE", "M&M", "TECHM",
            "JSWSTEEL", "COALINDIA", "BAJAJFINSV", "GRASIM", "EICHERMOT",
            "HDFCLIFE", "DRREDDY", "BRITANNIA", "CIPLA", "DIVISLAB", "APOLLOHOSP",
            "SBILIFE", "HINDALCO", "BPCL", "TATASTEEL", "SHREECEM", "HEROMOTOCO",
            "INDUSINDBK", "UPL", "BAJAJ-AUTO", "ADANIENT"]

@st.cache_data(ttl=300)
def get_nifty50_heatmap():
    rows = []
    for sym in get_nifty50_constituents():
        try:
            ltp = kite.ltp(f"NSE:{sym}")[f"NSE:{sym}"]['last_price']
            df = fetch_historical(sym, days=2)
            if df is not None and len(df) >= 2:
                prev = df['close'].iloc[-2]
                change = ((ltp - prev) / prev) * 100
                rows.append({"Stock": sym, "Change %": change})
        except:
            continue
    return pd.DataFrame(rows)

# ---------- Market Cap helper ----------
@st.cache_data(ttl=86400)
def get_market_cap(symbol):
    """Return market cap in crores (₹ Cr)."""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        mc = info.get('marketCap', 0)
        if mc:
            return mc / 1e7   # convert to crores
    except:
        pass
    return 0

def categorize_by_mcap(symbol):
    mc = get_market_cap(symbol)
    if mc >= 20000:
        return "Largecap"
    elif mc >= 5000:
        return "Midcap"
    elif mc > 0:
        return "Smallcap"
    else:
        return "Unknown"

# ---------- Memory & LLM calls (same as before) ----------
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
    if symbol not in memory:
        return []
    return memory[symbol][-limit:]

def call_deepseek(prompt, system="You are a financial analyst."):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
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

First, argue the BULL case (reasons to BUY). Then argue the BEAR case (reasons to SELL). Finally give a balanced verdict: BUY, SELL, or HOLD, with a confidence score (0-100). Format your response as:

**Bull Case**: ...
**Bear Case**: ...
**Verdict**: BUY/SELL/HOLD (confidence: XX)
**Reasoning**: ...
"""
    response = call_deepseek(prompt, system="You are a professional trading debate moderator.")
    return response

def get_stock_verdict(symbol):
    df = fetch_historical(symbol, days=250)
    if df is None or len(df) < 50:
        return {"error": f"Insufficient data for {symbol} (need 50+ days)"}
    indicators = ie.compute(df, symbol)
    signal_result = cs.compute(indicators, llm_signal=None, regime=1)
    latest_price = df['close'].iloc[-1]
    last_date = df.index[-1].strftime('%Y-%m-%d')
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        fundamentals = {
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "ROE (%)": info.get('returnOnEquity', 0)*100 if info.get('returnOnEquity') else 'N/A'
        }
    except:
        fundamentals = {"P/E Ratio": "N/A", "ROE (%)": "N/A"}
    past = get_recent_memory(symbol)
    past_text = "\n".join([f"- {p['date']}: {p['decision']} (conf {p['confidence']:.0f}%) → actual return {p.get('actual_return', 'pending')}%" for p in past]) if past else "No past trades."
    debate = bull_bear_debate(
        symbol, latest_price,
        indicators.get('rsi_14', 50),
        indicators.get('zscore_20', 0),
        indicators.get('momentum_5d_pct', 0),
        indicators.get('volume_ratio', 1),
        fundamentals
    )
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
    high_20 = df['high'].rolling(20).max()
    breakout_high = df['high'].iloc[-1] > high_20.iloc[-2]
    vol_spike = df['volume'].iloc[-1] > 1.5 * df['volume'].rolling(20).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    ema_bull = (ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
    consecutive_up = (close.diff().iloc[-5:] > 0).all()
    return near_top and volume_dry and breakout_high and vol_spike and ema_bull and consecutive_up

def scan_breakouts_categorized(symbols, limit=500):
    """Return dict with keys Largecap, Midcap, Smallcap containing lists of breakout symbols."""
    categorized = {"Largecap": [], "Midcap": [], "Smallcap": [], "Unknown": []}
    for sym in symbols[:limit]:
        try:
            df = fetch_historical(sym, days=200)
            if df is not None and len(df) >= 120 and is_breakout_candidate(df):
                cat = categorize_by_mcap(sym)
                categorized[cat].append(sym)
        except:
            continue
    return categorized

def generate_auto_pulse():
    indices = get_indices_data()
    nifty = indices.get("Nifty 50", {"price": "N/A", "change": 0})
    constituents = get_nifty50_constituents()
    gainers, losers = [], []
    for sym in constituents[:20]:
        try:
            ltp = kite.ltp(f"NSE:{sym}")[f"NSE:{sym}"]['last_price']
            df = fetch_historical(sym, days=2)
            if df is not None and len(df) >= 2:
                prev = df['close'].iloc[-2]
                change = ((ltp - prev) / prev) * 100
                gainers.append((sym, change))
                losers.append((sym, change))
        except:
            continue
    gainers.sort(key=lambda x: x[1], reverse=True)
    losers.sort(key=lambda x: x[1])
    all_syms = list(get_all_equity_symbols().keys())
    buzzing = []
    for sym in all_syms[:200]:
        try:
            df = fetch_historical(sym, days=50)
            if df is not None and len(df) >= 30:
                close = df['close']
                daily_move = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                avg_vol = df['volume'].iloc[-21:-1].mean()
                vol_spike = df['volume'].iloc[-1] > 2 * avg_vol
                if daily_move > 15 and vol_spike:
                    buzzing.append(sym)
                    if len(buzzing) >= 3: break
        except:
            continue
    gaining = []
    for sym in all_syms[:200]:
        try:
            df = fetch_historical(sym, days=50)
            if df is not None and len(df) >= 30:
                close = df['close']
                ema10 = close.ewm(span=10).mean()
                volume = df['volume']
                if (close.iloc[-1] > ema10.iloc[-1] and close.iloc[-2] < ema10.iloc[-2]) and volume.iloc[-1] > volume.iloc[-2]:
                    gaining.append(sym)
                    if len(gaining) >= 3: break
        except:
            continue
    losing = []
    for sym in all_syms[:200]:
        try:
            df = fetch_historical(sym, days=100)
            if df is not None and len(df) >= 50:
                close = df['close']
                ema50 = close.ewm(span=50).mean()
                if close.iloc[-1] < ema50.iloc[-1] and close.iloc[-2] > ema50.iloc[-2]:
                    losing.append(sym)
                    if len(losing) >= 3: break
        except:
            continue
    sp500 = get_global_indices().get("S&P 500", {"price": "N/A", "change": 0})
    prompt = f"""
    Today's date: {datetime.now().strftime('%d-%m-%Y')}
    Nifty 50: {nifty['price']:.1f} ({nifty['change']:+.2f}%)
    Top Gainers: {', '.join([f"{s} ({c:+.2f}%)" for s,c in gainers[:3]])}
    Top Losers: {', '.join([f"{s} ({c:+.2f}%)" for s,c in losers[:3]])}
    Buzzing Stock: {buzzing[0] if buzzing else 'None'}
    Gaining strength: {', '.join(gaining) if gaining else 'None'}
    Losing momentum: {', '.join(losing) if losing else 'None'}
    S&P 500: {sp500['price']:.1f} ({sp500['change']:+.2f}%)
    Generate a concise Daily Street Pulse report with sections: Market Overview, Top Gainers/Losers, Global Cues, Stock of the Day, Top Updates, Technical Outlook. Use bullet points.
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
    except Exception as e:
        return f"Error: {e}"

# ========== QUANT HEDGE FUND MODELS (simplified explanations) ==========
def time_series_momentum_explained(symbol):
    df = fetch_historical(symbol, days=300)
    if df is None or len(df) < 252:
        return None, "Needs at least one year of data."
    prices = df['close']
    ret = (prices.iloc[-1] - prices.iloc[-252]) / prices.iloc[-252]
    daily_vol = prices.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)
    if daily_vol == 0:
        return 0, "No volatility data."
    signal = ret / daily_vol
    signal = np.clip(signal, -1, 1)
    if signal > 0.3:
        verdict = "🟢 BUY – Strong upward momentum with reliable trend."
    elif signal < -0.3:
        verdict = "🔴 SELL – Strong downward momentum."
    else:
        verdict = "⚪ HOLD – Momentum is weak or mixed."
    return signal, verdict

def volatility_targeting_explained(symbol):
    df = fetch_historical(symbol, days=150)
    if df is None or len(df) < 60:
        return 1.0, "Not enough data."
    returns = df['close'].pct_change().dropna().iloc[-60:]
    realized_vol = returns.std() * np.sqrt(252)
    target_vol = 0.20
    if realized_vol == 0:
        return 1.0, "No volatility data."
    multiplier = target_vol / realized_vol
    multiplier = np.clip(multiplier, 0.2, 3.0)
    if multiplier > 1.2:
        verdict = f"📈 Increase position size (x{multiplier:.1f}) – low recent volatility."
    elif multiplier < 0.8:
        verdict = f"📉 Reduce position size (x{multiplier:.1f}) – high volatility."
    else:
        verdict = f"✅ Normal position size (x{multiplier:.1f})"
    return multiplier, verdict

def relative_value_spread_explained(symbol):
    partners = get_nifty50_constituents()
    best_pair = None
    best_corr = 0
    df_self = fetch_historical(symbol, days=252)
    if df_self is None or len(df_self) < 200:
        return (None, None), "Insufficient data for analysis."
    returns_self = df_self['close'].pct_change().dropna().iloc[-200:]
    for sym in partners[:20]:
        if sym == symbol:
            continue
        df_other = fetch_historical(sym, days=252)
        if df_other is None or len(df_other) < 200:
            continue
        returns_other = df_other['close'].pct_change().dropna().iloc[-200:]
        common = returns_self.index.intersection(returns_other.index)
        if len(common) < 100:
            continue
        corr = returns_self.loc[common].corr(returns_other.loc[common])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_pair = sym
    if best_pair is None:
        return (None, None), "No strongly correlated stock found."
    df1 = fetch_historical(symbol, days=252)
    df2 = fetch_historical(best_pair, days=252)
    merged = pd.merge(df1[['close']], df2[['close']], left_index=True, right_index=True, suffixes=('_1', '_2')).dropna()
    ratio = merged['close_1'] / merged['close_2']
    zscore = (ratio.iloc[-1] - ratio.mean()) / ratio.std()
    if zscore > 2:
        verdict = f"🔴 SELL {symbol} / BUY {best_pair} – {symbol} is expensive relative to its historical relationship."
    elif zscore < -2:
        verdict = f"🟢 BUY {symbol} / SELL {best_pair} – {symbol} is cheap relative to its pair."
    else:
        verdict = f"⚪ HOLD – The pair is fairly priced (z-score {zscore:.2f})."
    return (best_pair, zscore), verdict

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

# ========== LIGHTWEIGHT SWARM CONSENSUS (MiroFish style) ==========
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
Give your answer as:
VERDICT: BUY/SELL/HOLD
CONFIDENCE: number between 0 and 100
REASON: one short sentence.
"""
        response = call_deepseek(prompt, system="You are a professional investor with a clear bias.")
        if response:
            verdict = "HOLD"
            confidence = 50
            reason = "Could not parse"
            for line in response.split('\n'):
                if 'VERDICT:' in line:
                    verdict = line.split('VERDICT:')[1].strip().split()[0].upper()
                if 'CONFIDENCE:' in line:
                    try: confidence = int(line.split('CONFIDENCE:')[1].strip().split()[0])
                    except: pass
                if 'REASON:' in line:
                    reason = line.split('REASON:')[1].strip()
            results.append({"persona": persona['name'], "verdict": verdict, "confidence": confidence, "reason": reason})
    return results

# ---------- MAIN UI ----------
st.title("🧠 Prism Quant – Multi‑Agent Trading Desk")
st.caption("Live Kite Data | Bull/Bear LLM Debate | Composite Signal | Volume Profile | Auto Pulse | Quant Hedge Fund | Swarm Consensus")

symbol_map = get_all_equity_symbols()
symbol_list = sorted(symbol_map.keys())

with st.sidebar:
    st.header("🔍 Stock Analysis")
    selected = st.selectbox("Choose a stock", symbol_list, format_func=lambda x: f"{x} – {symbol_map[x]}")
    analyze = st.button("🚀 Analyze & Get Verdict", use_container_width=True)

if analyze and selected:
    with st.spinner(f"Fetching data and running multi‑agent analysis for {selected}..."):
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

        with st.expander("📝 Bull/Bear LLM Debate (DeepSeek)"):
            st.markdown(f"<div class='debate'>{verdict['debate']}</div>", unsafe_allow_html=True)

        with st.expander("🔧 Signal Attribution (weights: factor 0.1, ml 0.7, regime 0.1, llm 0.1)"):
            st.json(verdict['attribution'])

        with st.expander("📊 Volume Profile (last 250 days)"):
            df_vp = fetch_historical(selected, days=250)
            if df_vp is not None and len(df_vp) >= 50:
                profile = vp.compute(df_vp)
                st.write(f"**📍 Point of Control (POC):** ₹{profile['poc']:.2f} – most traded price, acts like a magnet")
                st.write(f"**📦 Value Area:** ₹{profile['val']:.2f} – ₹{profile['vah']:.2f} (70% of volume)")
                if profile['hvns']:
                    st.write(f"**🔴 High Volume Nodes (support/resistance):** {', '.join([f'₹{h:.2f}' for h in profile['hvns'][:5]])}")
                if profile['lvns']:
                    st.write(f"**🟢 Low Volume Nodes (breakout zones):** {', '.join([f'₹{l:.2f}' for l in profile['lvns'][:5]])}")
                bins = profile.get('bins', [])
                vol = profile.get('volume_profile', [])
                if bins and vol:
                    mids = [(bins[i]+bins[i+1])/2 for i in range(len(vol))]
                    fig_vp = go.Figure(go.Bar(x=vol, y=mids, orientation='h', marker_color='#4a90e2'))
                    fig_vp.add_hline(y=profile['poc'], line_dash="dash", line_color="#e67e22", annotation_text="POC")
                    fig_vp.update_layout(height=350, margin=dict(l=0, r=0), xaxis_title="Volume", yaxis_title="Price (₹)")
                    st.plotly_chart(fig_vp, use_container_width=True)
            else:
                st.info("Not enough data for volume profile (need 50+ days).")

        with st.expander("🧠 Past Memory (same stock)"):
            st.markdown(verdict['past_memory'])

        update_memory(selected, dir_text, verdict['confidence'], verdict['price'], actual_return=None)
        st.success("Decision saved to memory (actual return can be updated later).")

# Tabs for other features (9 tabs)
tabs = st.tabs(["📊 Market Dashboard", "📈 Charts & EMAs", "🔍 Stock Categories", "🔥 Breakout Forecast", "🗺️ Heatmap", "🤖 Auto Pulse", "🌍 Global Markets", "🏦 Quant Hedge Fund", "🐟 Swarm Consensus"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🇮🇳 Indian Indices (Kite)")
        ind_data = get_indices_data()
        if ind_data:
            df_ind = pd.DataFrame([{"Index": k, "Price": f"{v['price']:.1f}", "Change %": f"{v['change']:+.2f}%"} for k,v in ind_data.items()])
            st.dataframe(df_ind, width='stretch', hide_index=True)
    with col2:
        st.subheader("🌏 Global Indices")
        global_data = get_global_indices()
        if global_data:
            df_glob = pd.DataFrame([{"Index": k, "Price": f"{v['price']:.1f}", "Change %": f"{v['change']:+.2f}%"} for k,v in global_data.items()])
            st.dataframe(df_glob, width='stretch', hide_index=True)
    st.subheader("📦 Commodities")
    comm_data = get_commodities()
    if comm_data:
        df_comm = pd.DataFrame([{"Commodity": k, "Price": f"{v['price']:.2f}", "Change %": f"{v['change']:+.2f}%"} for k,v in comm_data.items()])
        st.dataframe(df_comm, width='stretch', hide_index=True)

with tabs[1]:
    if analyze and selected:
        df = fetch_historical(selected, days=150)
        if df is not None and len(df) > 50:
            df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA10'], mode='lines', line=dict(color='green', width=1), name="10 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', line=dict(color='yellow', width=1), name="20 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', line=dict(color='purple', width=1), name="50 EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], mode='lines', line=dict(color='red', width=1), name="200 EMA"))
            fig.update_layout(title=f"{selected} – Candlestick with EMAs", xaxis_title="Date", yaxis_title="Price (₹)", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough historical data for chart.")
    else:
        st.info("Select a stock from the sidebar and click 'Analyze & Get Verdict' first.")

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
            st.info("No stocks matched the criteria today.")

# ---------- BREAKOUT FORECAST – Categorized by Market Cap ----------
with tabs[3]:
    st.subheader("🚀 Breakout Stocks for Tomorrow (Categorized by Market Cap)")
    if st.button("Run Breakout Scanner"):
        all_syms = list(get_all_equity_symbols().keys())
        with st.spinner("Scanning for breakout setups and fetching market caps..."):
            result = scan_breakouts_categorized(all_syms, limit=500)
        # Show results in three expanders
        with st.expander("🏢 Largecap (Market Cap ≥ ₹20,000 Cr)", expanded=True):
            if result["Largecap"]:
                st.write(", ".join(result["Largecap"]))
            else:
                st.info("No largecap breakout candidates found.")
        with st.expander("📈 Midcap (₹5,000 Cr – ₹20,000 Cr)", expanded=True):
            if result["Midcap"]:
                st.write(", ".join(result["Midcap"]))
            else:
                st.info("No midcap breakout candidates found.")
        with st.expander("🔬 Smallcap (₹ < 5,000 Cr)", expanded=True):
            if result["Smallcap"]:
                st.write(", ".join(result["Smallcap"]))
            else:
                st.info("No smallcap breakout candidates found.")
        if result["Unknown"]:
            with st.expander("❓ Unknown (Market cap not available)", expanded=False):
                st.write(", ".join(result["Unknown"]))
    else:
        st.info("Click the button to scan the universe for breakout patterns.")

with tabs[4]:
    st.subheader("🗺️ Nifty 50 Heatmap (Kite LTP)")
    df_heat = get_nifty50_heatmap()
    if not df_heat.empty:
        fig = px.treemap(df_heat, path=['Stock'], values='Change %', color='Change %',
                         color_continuous_scale=['red', 'yellow', 'green'], title="% Change")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Heatmap data not available (Kite LTP fetch failed).")

with tabs[5]:
    st.subheader("🤖 Generate Daily Street Pulse (Live Kite Data)")
    if st.button("Generate Today's Pulse"):
        with st.spinner("Collecting live market data and generating report..."):
            pulse = generate_auto_pulse()
        st.markdown(f"<div class='daily-pulse'>{pulse}</div>", unsafe_allow_html=True)

with tabs[6]:
    st.subheader("🌍 Global Indices & Commodities")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Indices**")
        for name, val in get_global_indices().items():
            st.metric(name, f"{val['price']:.1f}", f"{val['change']:+.2f}%")
    with colB:
        st.write("**Commodities**")
        for name, val in get_commodities().items():
            st.metric(name, f"{val['price']:.2f}", f"{val['change']:+.2f}%")

with tabs[7]:
    st.header("🏦 Quant Hedge Fund Models")
    st.markdown("Four institutional models explained in simple language.")
    if selected:
        with st.expander("📈 1. Time Series Momentum – Trending or not?"):
            sig, expl = time_series_momentum_explained(selected)
            if sig is None:
                st.warning(expl)
            else:
                st.metric("Momentum Score", f"{sig:.2f}")
                st.write(expl)
        with st.expander("🎯 2. Volatility Targeting – How big to bet?"):
            mult, expl = volatility_targeting_explained(selected)
            st.metric("Position Size Multiplier", f"{mult:.2f}x")
            st.write(expl)
        with st.expander("🔄 3. Relative Value Spread – Cheap or expensive?"):
            pair_info, verdict = relative_value_spread_explained(selected)
            if pair_info[0] is None:
                st.warning(verdict)
            else:
                st.write(f"**Most correlated stock:** {pair_info[0]} (z-score: {pair_info[1]:.2f})")
                st.write(verdict)
        with st.expander("🌍 4. Macro Regime – Big picture"):
            regime, alloc = macro_regime_allocation_explained()
            st.write(regime)
            st.write(f"**Recommended Portfolio:** {alloc}")
    else:
        st.info("Select a stock from the sidebar first.")

with tabs[8]:
    st.header("🐟 Swarm Intelligence – 5 Investor Personas")
    st.markdown("We simulate five different investors (bull, risk‑averse, quant, value, macro) to see how they react to a news event.")
    news_input = st.text_area("📰 News or event (e.g., from Daily Pulse)", height=100,
                              placeholder="HDFC Life profit up 4% on policy renewals...")
    if st.button("Run Swarm Simulation"):
        if not news_input:
            st.warning("Please enter a news snippet.")
        else:
            with st.spinner("Contacting 5 investor personas via DeepSeek..."):
                results = swarm_consensus(selected if 'selected' in locals() else "RELIANCE", news_input)
            if results:
                for res in results:
                    color = "🟢" if res['verdict'] == "BUY" else "🔴" if res['verdict'] == "SELL" else "🟡"
                    st.markdown(f"### {res['persona']} {color}")
                    st.write(f"**Verdict:** {res['verdict']}  |  **Confidence:** {res['confidence']}%")
                    st.write(f"*{res['reason']}*")
                    st.divider()
                buy_count = sum(1 for r in results if r['verdict'] == "BUY")
                sell_count = sum(1 for r in results if r['verdict'] == "SELL")
                hold_count = sum(1 for r in results if r['verdict'] == "HOLD")
                st.subheader("🤝 Swarm Consensus")
                st.write(f"**BUY:** {buy_count}  |  **SELL:** {sell_count}  |  **HOLD:** {hold_count}")
                if buy_count > sell_count and buy_count > hold_count:
                    st.success("The swarm leans **BUY** – most personas see opportunity.")
                elif sell_count > buy_count and sell_count > hold_count:
                    st.error("The swarm leans **SELL** – caution is advised.")
                else:
                    st.warning("The swarm is **divided** – no strong consensus.")
            else:
                st.error("Failed to get responses from personas. Check API key.")
