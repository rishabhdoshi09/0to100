"""
Daily Street Pulse — auto-generated daily market report.

Format inspired by professional equity research daily notes:
  · Market snapshot (Nifty, BankNifty, Smallcap, Metal)
  · Top gainers / losers from Nifty 50
  · Global cues (US, Asia, Commodities)
  · Buzzing stock of the day (highest momentum)
  · Stock gaining strength + losing momentum
  · Breakout picks for tomorrow
  · DeepSeek V3 analysis for each stock

Output: HTML (in-memory) + optional PDF via weasyprint.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

from logger import get_logger

log = get_logger(__name__)

# ── Index tickers ──────────────────────────────────────────────────────────────
_INDIA_INDICES = {
    "Nifty 50":       "^NSEI",
    "Bank Nifty":     "^NSEBANK",
    "Nifty Smallcap": "^CNXSC",
    "Nifty Metal":    "CNXMETAL.NS",
    "Nifty IT":       "^CNXIT",
    "Nifty FMCG":     "CNXFMCG.NS",
}
_GLOBAL_INDICES = {
    "S&P 500":     "^GSPC",
    "Dow Jones":   "^DJI",
    "Nasdaq":      "^IXIC",
    "FTSE 100":    "^FTSE",
    "Nikkei 225":  "^N225",
    "Hang Seng":   "^HSI",
    "Shanghai":    "000001.SS",
}
_COMMODITIES = {
    "Gold":    "GC=F",
    "Silver":  "SI=F",
    "Crude":   "CL=F",
    "Nat Gas": "NG=F",
    "DXY":     "DX-Y.NYB",
}

_NIFTY50_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
    "BAJFINANCE","WIPRO","SUNPHARMA","ULTRACEMCO","NTPC","POWERGRID","TECHM",
    "HCLTECH","ADANIENT","ADANIPORTS","JSWSTEEL","TATAMOTORS","TATASTEEL",
    "COALINDIA","ONGC","BAJAJFINSV","CIPLA","DIVISLAB","DRREDDY","EICHERMOT",
    "GRASIM","HEROMOTOCO","HINDALCO","INDUSINDBK","MM","NESTLEIND","SBILIFE",
    "SHRIRAMFIN","TATACONSUM","APOLLOHOSP","BPCL","BRITANNIA","HDFCLIFE",
    "BAJAJ-AUTO","UPL",
]


def _fetch_index_data(ticker: str) -> dict:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist) >= 2:
            last  = float(hist["Close"].iloc[-1])
            prev  = float(hist["Close"].iloc[-2])
            chg   = (last - prev) / prev * 100
            vol   = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
            return {"price": last, "change": chg, "volume": vol}
    except Exception:
        pass
    return {"price": 0.0, "change": 0.0, "volume": 0}


def _fetch_nifty50_movers() -> tuple[list, list]:
    """Returns (top5_gainers, top5_losers) as list of dicts."""
    results = []
    for sym in _NIFTY50_SYMBOLS[:30]:  # limit to avoid timeout
        try:
            info = yf.Ticker(f"{sym}.NS").fast_info
            price = float(getattr(info, "last_price", 0) or 0)
            prev  = float(getattr(info, "previous_close", price) or price)
            if prev:
                chg = (price - prev) / prev * 100
                results.append({"symbol": sym, "price": price, "change": chg})
        except Exception:
            pass
    results.sort(key=lambda x: x["change"], reverse=True)
    return results[:5], results[-5:][::-1]


def _get_ema_position(ticker: str) -> str:
    """Returns a one-line EMA status."""
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if len(df) < 50:
            return "Insufficient data"
        c = df["Close"]
        e10, e20, e50, e200 = (
            float(c.ewm(span=10).mean().iloc[-1]),
            float(c.ewm(span=20).mean().iloc[-1]),
            float(c.ewm(span=50).mean().iloc[-1]),
            float(c.ewm(span=200).mean().iloc[-1]),
        )
        last = float(c.iloc[-1])
        if last > e10 > e20 > e50 > e200:
            return "Above all EMAs — strong uptrend 🟢"
        elif last > e20 and last > e50:
            return "Above 20 & 50 EMA — bullish structure 🟡"
        elif last > e200:
            return "Above 200 EMA — long-term bullish 🟡"
        else:
            return "Below 200 EMA — bearish trend 🔴"
    except Exception:
        return "—"


def _deepseek_analysis(prompt: str, max_tokens: int = 200) -> str:
    """Quick V3 analysis — returns plain text, empty string on failure."""
    try:
        from ai.dual_llm_service import get_service
        text, _, _ = get_service().ask(prompt, max_tokens=max_tokens)
        return text.strip()
    except Exception:
        return ""


def build_report_data() -> dict:
    """
    Fetches all data needed for the report.
    Returns a dict consumed by render_html().
    """
    today = datetime.now().strftime("%d %b %Y, %A")
    log.info("daily_pulse_building", date=today)

    # ── India indices ─────────────────────────────────────────────────────
    india = {name: _fetch_index_data(t) for name, t in _INDIA_INDICES.items()}

    # ── Global ────────────────────────────────────────────────────────────
    global_idx  = {name: _fetch_index_data(t) for name, t in _GLOBAL_INDICES.items()}
    commodities = {name: _fetch_index_data(t) for name, t in _COMMODITIES.items()}

    # ── Nifty 50 movers ───────────────────────────────────────────────────
    gainers, losers = _fetch_nifty50_movers()

    # ── Momentum stocks (from scanner) ───────────────────────────────────
    try:
        from screener.momentum_scanner import MomentumScanner
        scanner  = MomentumScanner(max_workers=8)
        momentum = scanner.scan_momentum(_NIFTY50_SYMBOLS, top_n=5)
        breakouts = scanner.scan_breakouts(_NIFTY50_SYMBOLS, top_n=5)
    except Exception as exc:
        log.warning("scanner_failed", error=str(exc))
        momentum, breakouts = [], []

    # ── EMA positions for top indices ─────────────────────────────────────
    nifty_ema    = _get_ema_position("^NSEI")
    banknifty_ema = _get_ema_position("^NSEBANK")
    sp500_ema    = _get_ema_position("^GSPC")

    # ── DeepSeek market commentary ────────────────────────────────────────
    nifty_data = india.get("Nifty 50", {})
    nifty_chg  = nifty_data.get("change", 0)
    nifty_px   = nifty_data.get("price", 0)

    market_summary = _deepseek_analysis(
        f"Nifty 50 today: ₹{nifty_px:,.0f}, change {nifty_chg:+.2f}%. "
        f"EMA status: {nifty_ema}. "
        "Write a 3-line market summary in trader language. Be specific and concise.",
        max_tokens=150,
    ) or f"Nifty closed {'up' if nifty_chg >= 0 else 'down'} {abs(nifty_chg):.2f}% at {nifty_px:,.0f}."

    # ── Analysis for top 3 stocks ─────────────────────────────────────────
    stock_analyses = []
    for ms in momentum[:3]:
        analysis = _deepseek_analysis(
            f"Stock: {ms.symbol} | Price: ₹{ms.price:,.0f} | "
            f"Change: {ms.change_pct:+.1f}% | RSI: {ms.rsi:.0f} | "
            f"Volume ratio: {ms.volume_ratio:.1f}x | Signal: {ms.signal}\n"
            "Write 2 bullet points on why this stock is in focus today. "
            "Be specific: mention price levels, pattern, volume. No fluff.",
            max_tokens=120,
        )
        stock_analyses.append({
            "symbol": ms.symbol,
            "price": ms.price,
            "change": ms.change_pct,
            "rsi": ms.rsi,
            "volume_ratio": ms.volume_ratio,
            "signal": ms.signal,
            "analysis": analysis,
            "ema": _get_ema_position(f"{ms.symbol}.NS"),
        })

    return {
        "date":            today,
        "india":           india,
        "global":          global_idx,
        "commodities":     commodities,
        "gainers":         gainers,
        "losers":          losers,
        "momentum":        momentum,
        "breakouts":       breakouts,
        "nifty_ema":       nifty_ema,
        "banknifty_ema":   banknifty_ema,
        "sp500_ema":       sp500_ema,
        "market_summary":  market_summary,
        "stock_analyses":  stock_analyses,
    }


def render_html(data: dict) -> str:
    """Renders the report as a dark-themed HTML string."""

    def _chg_color(v): return "#00d4a0" if v >= 0 else "#ff4466"
    def _chg_arrow(v): return "▲" if v >= 0 else "▼"
    def _idx_row(name, d):
        c = _chg_color(d["change"])
        return (
            f"<tr><td style='color:#e8eaf0;font-weight:600'>{name}</td>"
            f"<td style='color:#c9d1e0;text-align:right'>{d['price']:,.1f}</td>"
            f"<td style='color:{c};text-align:right;font-weight:700'>"
            f"{_chg_arrow(d['change'])} {abs(d['change']):.2f}%</td></tr>"
        )

    india_rows    = "".join(_idx_row(n, d) for n, d in data["india"].items())
    global_rows   = "".join(_idx_row(n, d) for n, d in data["global"].items())
    commodity_rows = "".join(_idx_row(n, d) for n, d in data["commodities"].items())

    def _mover_cards(stocks, label, color):
        cards = ""
        for s in stocks:
            c = _chg_color(s["change"])
            cards += (
                f"<div style='background:rgba(255,255,255,.04);border:1px solid {color}33;"
                f"border-left:3px solid {color};border-radius:8px;padding:.5rem .75rem;"
                f"margin:.3rem 0'>"
                f"<span style='color:#e8eaf0;font-weight:700;font-size:.82rem'>{s['symbol']}</span>"
                f"<span style='color:#8892a4;font-size:.72rem;margin-left:.5rem'>₹{s['price']:,.1f}</span>"
                f"<span style='color:{c};font-weight:700;font-size:.78rem;float:right'>"
                f"{_chg_arrow(s['change'])} {abs(s['change']):.1f}%</span></div>"
            )
        return cards

    gainer_cards = _mover_cards(data["gainers"], "Gainers", "#00d4a0")
    loser_cards  = _mover_cards(data["losers"],  "Losers",  "#ff4466")

    # Stock analysis cards
    stock_cards = ""
    icons = ["🔥", "⚡", "📈"]
    for i, s in enumerate(data["stock_analyses"]):
        sig_color = {"BUY": "#00d4a0", "WATCH": "#f59e0b", "NEUTRAL": "#8892a4"}.get(s["signal"], "#8892a4")
        analysis_html = s["analysis"].replace("\n", "<br>") if s["analysis"] else "Analysis unavailable."
        stock_cards += f"""
        <div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
             border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem'>
            <span style='color:#00d4ff;font-size:1.05rem;font-weight:800;letter-spacing:.05em'>
              {icons[i]} {s['symbol']}</span>
            <span style='color:{sig_color};font-size:.7rem;font-weight:700;background:rgba(255,255,255,.05);
              padding:.2rem .6rem;border-radius:6px;border:1px solid {sig_color}44'>{s['signal']}</span>
          </div>
          <div style='display:flex;gap:1.5rem;margin-bottom:.6rem;flex-wrap:wrap'>
            <span style='font-size:.78rem'><span style='color:#8892a4'>Price </span>
              <span style='color:#e8eaf0;font-weight:700'>₹{s['price']:,.0f}</span></span>
            <span style='font-size:.78rem'><span style='color:#8892a4'>Change </span>
              <span style='color:{_chg_color(s["change"])};font-weight:700'>
              {_chg_arrow(s["change"])} {abs(s["change"]):.1f}%</span></span>
            <span style='font-size:.78rem'><span style='color:#8892a4'>RSI </span>
              <span style='color:#e8eaf0'>{s['rsi']:.0f}</span></span>
            <span style='font-size:.78rem'><span style='color:#8892a4'>Vol </span>
              <span style='color:#e8eaf0'>{s['volume_ratio']:.1f}x avg</span></span>
          </div>
          <div style='font-size:.72rem;color:#8892a4;margin-bottom:.4rem'>{s['ema']}</div>
          <div style='font-size:.78rem;color:#c9d1e0;line-height:1.7;border-top:1px solid rgba(255,255,255,.06);
               padding-top:.5rem'>{analysis_html}</div>
        </div>"""

    # Breakout picks
    breakout_rows = ""
    for b in data["breakouts"]:
        conf_color = "#00d4a0" if b.confidence >= 70 else "#f59e0b"
        breakout_rows += (
            f"<tr>"
            f"<td style='color:#00d4ff;font-weight:700'>{b.symbol}</td>"
            f"<td style='color:#e8eaf0'>₹{b.price:,.0f}</td>"
            f"<td style='color:#f97316;font-size:.72rem'>{b.breakout_type.replace('_',' ')}</td>"
            f"<td style='color:{conf_color};font-weight:700'>{b.confidence:.0f}%</td>"
            f"</tr>"
        )

    # Nifty summary
    nifty = data["india"].get("Nifty 50", {})
    nifty_color = _chg_color(nifty.get("change", 0))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #060b18;
    color: #e8eaf0;
    font-family: Inter, system-ui, sans-serif;
    padding: 2rem;
    max-width: 960px;
    margin: 0 auto;
  }}
  h2 {{ color: #00d4ff; font-size: .85rem; letter-spacing: .12em;
       text-transform: uppercase; margin: 1.5rem 0 .75rem;
       border-bottom: 1px solid rgba(0,212,255,.2); padding-bottom: .35rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .78rem; }}
  td, th {{ padding: .35rem .5rem; border-bottom: 1px solid rgba(255,255,255,.04); }}
  th {{ color: #4a5568; font-size: .62rem; text-transform: uppercase;
       letter-spacing: .08em; font-weight: 600; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }}
  .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }}
  .card {{ background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
           border-radius: 12px; padding: .9rem 1.1rem; }}
  .tag {{ display: inline-block; font-size: .6rem; font-weight: 700;
          padding: .15rem .5rem; border-radius: 4px; letter-spacing: .06em;
          text-transform: uppercase; margin-right: .3rem; }}
  @media print {{ body {{ background: white; color: #111; }} }}
</style>
</head>
<body>

<!-- HEADER -->
<div style='text-align:center;margin-bottom:2rem;padding:1.5rem;
     background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(0,212,255,.02));
     border:1px solid rgba(0,212,255,.2);border-radius:16px'>
  <div style='font-size:.7rem;color:#4a5568;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.3rem'>
    Daily Market Report</div>
  <div style='font-size:2rem;font-weight:800;color:#00d4ff;letter-spacing:.05em'>
    Daily Street Pulse</div>
  <div style='font-size:.88rem;color:#8892a4;margin-top:.3rem'>{data["date"]}</div>
  <div style='margin-top:.9rem;font-size:.82rem;color:#c9d1e0;max-width:600px;
       margin-left:auto;margin-right:auto;line-height:1.7'>
    {data["market_summary"]}
  </div>
</div>

<!-- MARKET SNAPSHOT -->
<h2>📊 Market Snapshot — India</h2>
<div class='three-col' style='margin-bottom:1rem'>
  <div class='card' style='border-left:3px solid {nifty_color}'>
    <div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em'>Nifty 50</div>
    <div style='font-size:1.4rem;font-weight:800;color:#e8eaf0;margin:.2rem 0'>
      {nifty.get("price", 0):,.0f}</div>
    <div style='font-size:.82rem;color:{nifty_color};font-weight:700'>
      {_chg_arrow(nifty.get("change",0))} {abs(nifty.get("change",0)):.2f}%</div>
    <div style='font-size:.65rem;color:#8892a4;margin-top:.4rem'>{data["nifty_ema"]}</div>
  </div>
  <div class='card' style='border-left:3px solid {_chg_color(data["india"].get("Bank Nifty",{}).get("change",0))}'>
    <div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em'>Bank Nifty</div>
    <div style='font-size:1.4rem;font-weight:800;color:#e8eaf0;margin:.2rem 0'>
      {data["india"].get("Bank Nifty",{}).get("price",0):,.0f}</div>
    <div style='font-size:.82rem;color:{_chg_color(data["india"].get("Bank Nifty",{}).get("change",0))};font-weight:700'>
      {_chg_arrow(data["india"].get("Bank Nifty",{}).get("change",0))}
      {abs(data["india"].get("Bank Nifty",{}).get("change",0)):.2f}%</div>
    <div style='font-size:.65rem;color:#8892a4;margin-top:.4rem'>{data["banknifty_ema"]}</div>
  </div>
  <div class='card' style='border-left:3px solid {_chg_color(data["india"].get("Nifty Smallcap",{}).get("change",0))}'>
    <div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em'>Nifty Smallcap</div>
    <div style='font-size:1.4rem;font-weight:800;color:#e8eaf0;margin:.2rem 0'>
      {data["india"].get("Nifty Smallcap",{}).get("price",0):,.0f}</div>
    <div style='font-size:.82rem;color:{_chg_color(data["india"].get("Nifty Smallcap",{}).get("change",0))};font-weight:700'>
      {_chg_arrow(data["india"].get("Nifty Smallcap",{}).get("change",0))}
      {abs(data["india"].get("Nifty Smallcap",{}).get("change",0)):.2f}%</div>
  </div>
</div>

<div class='card' style='margin-bottom:1.5rem'>
<table>
  <tr><th>Index</th><th style='text-align:right'>Level</th><th style='text-align:right'>Change</th></tr>
  {india_rows}
</table>
</div>

<!-- GAINERS / LOSERS -->
<h2>🏆 Top Gainers &amp; Losers — Nifty 50</h2>
<div class='two-col' style='margin-bottom:1.5rem'>
  <div>
    <div style='font-size:.65rem;color:#00d4a0;font-weight:700;text-transform:uppercase;
         letter-spacing:.1em;margin-bottom:.4rem'>Top Gainers</div>
    {gainer_cards}
  </div>
  <div>
    <div style='font-size:.65rem;color:#ff4466;font-weight:700;text-transform:uppercase;
         letter-spacing:.1em;margin-bottom:.4rem'>Top Losers</div>
    {loser_cards}
  </div>
</div>

<!-- STOCKS IN FOCUS -->
<h2>🔥 Stocks in Focus — DeepSeek Analysis</h2>
{stock_cards}

<!-- BREAKOUT PICKS -->
<h2>💥 Breakout Picks for Tomorrow</h2>
<div class='card' style='margin-bottom:1.5rem'>
<table>
  <tr><th>Symbol</th><th>Price</th><th>Pattern</th><th style='text-align:right'>Confidence</th></tr>
  {breakout_rows or "<tr><td colspan='4' style='color:#4a5568;text-align:center'>No breakouts detected</td></tr>"}
</table>
</div>

<!-- GLOBAL CUES -->
<h2>🌐 Global Cues</h2>
<div class='two-col' style='margin-bottom:1.5rem'>
  <div class='card'>
    <div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem'>
      US &amp; Europe</div>
    <table>
      <tr><th>Index</th><th style='text-align:right'>Level</th><th style='text-align:right'>Change</th></tr>
      {"".join(_idx_row(n,d) for n,d in list(data["global"].items())[:5])}
    </table>
  </div>
  <div class='card'>
    <div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem'>
      Commodities &amp; Asia</div>
    <table>
      <tr><th>Name</th><th style='text-align:right'>Price</th><th style='text-align:right'>Change</th></tr>
      {"".join(_idx_row(n,d) for n,d in data["commodities"].items())}
      {"".join(_idx_row(n,d) for n,d in list(data["global"].items())[5:])}
    </table>
  </div>
</div>

<!-- FOOTER -->
<div style='text-align:center;padding:1rem;border-top:1px solid rgba(255,255,255,.05);
     margin-top:2rem;font-size:.62rem;color:#4a5568;letter-spacing:.06em'>
  QUANTTERM · Generated {data["date"]} · AI analysis by DeepSeek V3 ·
  Data: Kite Connect / yfinance · For educational purposes only — not investment advice
</div>

</body>
</html>"""


def generate_pdf(data: dict) -> bytes:
    """Convert HTML report to PDF bytes using weasyprint."""
    html = render_html(data)
    from weasyprint import HTML as WP
    return WP(string=html).write_pdf()


def generate_html_bytes(data: dict) -> bytes:
    return render_html(data).encode("utf-8")
