"""Daily PDF market report.

Runs at 17:30 IST every weekday (cron via APScheduler in
``backend/scheduler.py`` — added below).  Pulls market snapshot, top
movers, sector heatmap, screener winners, asks Claude (via ``ClaudeClient``)
for a 250-word narrative, renders the PDF with ``reportlab``, persists
the metadata into ``reports``.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from sq_ai.backend.cache import cached
from sq_ai.backend.data_fetcher import fetch_yf_history
from sq_ai.backend.llm_clients import ClaudeClient
from sq_ai.portfolio.tracker import PortfolioTracker


IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))
INDEX_TICKERS = {
    "Nifty 50":        "^NSEI",
    "Nifty Smallcap":  "^NSESMCP100",      # may not exist; fallback handled
    "Bank Nifty":      "^NSEBANK",
    "S&P 500":         "^GSPC",
    "Nasdaq":          "^IXIC",
}
SECTORS = {
    "IT":     "^CNXIT",
    "Bank":   "^CNXBANK",
    "Pharma": "^CNXPHARMA",
    "Auto":   "^CNXAUTO",
    "Metal":  "^CNXMETAL",
    "FMCG":   "^CNXFMCG",
    "Energy": "^CNXENERGY",
}
DEFAULT_REPORTS_DIR = str(Path(__file__).resolve().parents[2] / "reports")


def _safe_history(sym: str, period: str = "3mo"):
    try:
        df = fetch_yf_history(sym, period=period, interval="1d")
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _emas(series, periods=(10, 20, 50, 200)) -> dict[str, float]:
    out = {}
    for p in periods:
        try:
            out[f"ema_{p}"] = float(series.ewm(span=p, adjust=False).mean().iloc[-1])
        except Exception:
            out[f"ema_{p}"] = None
    return out


def market_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"indices": [], "sectors": []}
    for name, sym in INDEX_TICKERS.items():
        df = _safe_history(sym)
        if df is None or len(df) < 5:
            continue
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        snap["indices"].append({
            "name": name, "symbol": sym,
            "price": last, "change_pct": (last - prev) / prev * 100 if prev else 0,
            **_emas(df["close"]),
        })
    for sec, sym in SECTORS.items():
        df = _safe_history(sym)
        if df is None or len(df) < 5:
            continue
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        snap["sectors"].append({
            "sector": sec, "change_pct": (last - prev) / prev * 100 if prev else 0,
        })
    return snap


def top_movers(symbols: list[str], top_n: int = 5) -> dict[str, list[dict]]:
    rows = []
    for s in symbols:
        df = _safe_history(s, period="1mo")
        if df is None or len(df) < 2:
            continue
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        rows.append({"symbol": s, "price": last,
                     "change_pct": (last - prev) / prev * 100 if prev else 0})
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return {"gainers": rows[:top_n], "losers": rows[-top_n:][::-1]}


@cached("daily_narrative", ttl_seconds=43200)        # 12 h
def _narrative(snapshot: dict, gainers: list[dict], losers: list[dict],
               claude: ClaudeClient | None = None) -> str:
    claude = claude or ClaudeClient()
    if not claude.available:
        # deterministic fallback
        idx_lines = "; ".join(
            f"{i['name']} {i['change_pct']:+.2f}%" for i in snapshot["indices"]
        )
        return (
            f"Markets today: {idx_lines}. "
            f"Top gainer: {gainers[0]['symbol'] if gainers else '–'} "
            f"({gainers[0]['change_pct']:+.2f}%). "
            f"Top loser: {losers[0]['symbol'] if losers else '–'} "
            f"({losers[0]['change_pct']:+.2f}%). "
            "Risk-on tone moderate; trade selectively, respect stops."
        )
    prompt = (
        "Write a crisp 220-word professional market wrap for Indian retail "
        "traders. Use the data provided. No bullet points, two paragraphs.\n\n"
        f"Indices: {snapshot['indices']}\n"
        f"Sectors: {snapshot['sectors']}\n"
        f"Top gainers: {gainers}\nTop losers: {losers}\n"
    )
    return claude.generate(prompt, max_tokens=400, temperature=0.4) or ""


def render_pdf(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.lib.pagesizes import A4  # noqa: WPS433
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
        from reportlab.lib import colors
    except Exception:                              # pragma: no cover
        # Bare-minimum text fallback
        out_path.write_text(_to_text(payload))
        return out_path

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="Daily Market Report")
    flow = []

    flow.append(Paragraph(
        f"<b>sq_ai Daily Market Report</b> – {payload['date']}",
        styles["Title"],
    ))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<b>Indices</b>", styles["Heading2"]))
    if payload["snapshot"]["indices"]:
        rows = [["Index", "Price", "Δ %", "EMA20", "EMA50"]]
        for i in payload["snapshot"]["indices"]:
            rows.append([i["name"], f"{i['price']:.2f}",
                         f"{i['change_pct']:+.2f}%",
                         f"{i.get('ema_20', 0) or 0:.2f}",
                         f"{i.get('ema_50', 0) or 0:.2f}"])
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161b22")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        flow.append(t)

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<b>Sector heatmap</b>", styles["Heading2"]))
    if payload["snapshot"]["sectors"]:
        rows = [["Sector", "Δ %"]]
        for s in payload["snapshot"]["sectors"]:
            rows.append([s["sector"], f"{s['change_pct']:+.2f}%"])
        flow.append(Table(rows, hAlign="LEFT"))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<b>Top gainers</b>", styles["Heading2"]))
    if payload["movers"]["gainers"]:
        rows = [["Symbol", "Price", "Δ %"]]
        for r in payload["movers"]["gainers"]:
            rows.append([r["symbol"], f"{r['price']:.2f}",
                         f"{r['change_pct']:+.2f}%"])
        flow.append(Table(rows, hAlign="LEFT"))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<b>Top losers</b>", styles["Heading2"]))
    if payload["movers"]["losers"]:
        rows = [["Symbol", "Price", "Δ %"]]
        for r in payload["movers"]["losers"]:
            rows.append([r["symbol"], f"{r['price']:.2f}",
                         f"{r['change_pct']:+.2f}%"])
        flow.append(Table(rows, hAlign="LEFT"))

    flow.append(Spacer(1, 18))
    flow.append(Paragraph("<b>Market wrap</b>", styles["Heading2"]))
    flow.append(Paragraph(payload["narrative"].replace("\n", "<br/>"),
                          styles["BodyText"]))

    doc.build(flow)
    return out_path


def _to_text(payload: dict) -> str:
    lines = [f"sq_ai Daily Market Report – {payload['date']}", "=" * 60, ""]
    for i in payload["snapshot"]["indices"]:
        lines.append(f"{i['name']:20} {i['price']:>10.2f}  {i['change_pct']:+.2f}%")
    lines.append("")
    lines.append("Sector heatmap:")
    for s in payload["snapshot"]["sectors"]:
        lines.append(f"  {s['sector']:10} {s['change_pct']:+.2f}%")
    lines.append("")
    lines.append("Top gainers:")
    for r in payload["movers"]["gainers"]:
        lines.append(f"  {r['symbol']:15} {r['price']:>10.2f} {r['change_pct']:+.2f}%")
    lines.append("Top losers:")
    for r in payload["movers"]["losers"]:
        lines.append(f"  {r['symbol']:15} {r['price']:>10.2f} {r['change_pct']:+.2f}%")
    lines.append("")
    lines.append("Market wrap:")
    lines.append(payload["narrative"])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
class ReportGenerator:
    def __init__(self,
                 tracker: PortfolioTracker | None = None,
                 reports_dir: str | None = None,
                 watchlist_symbols: list[str] | None = None) -> None:
        self.tracker = tracker or PortfolioTracker()
        self.reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
        self.watchlist_symbols = watchlist_symbols or [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS",
            "HINDUNILVR.NS", "BHARTIARTL.NS", "MARUTI.NS", "TITAN.NS",
            "BAJFINANCE.NS",
        ]

    def generate(self) -> dict[str, Any]:
        ts = datetime.now(IST)
        snap = market_snapshot()
        movers = top_movers(self.watchlist_symbols, top_n=5)
        narrative = _narrative(snap, movers["gainers"], movers["losers"])
        payload = {
            "date": ts.strftime("%Y-%m-%d"),
            "snapshot": snap, "movers": movers, "narrative": narrative,
        }
        filename = f"daily_report_{ts.strftime('%Y%m%d_%H%M')}.pdf"
        out = render_pdf(payload, self.reports_dir / filename)
        self.tracker.report_record(out.name, summary=narrative[:200])
        return {"filename": out.name, "path": str(out),
                "narrative_preview": narrative[:300]}


__all__ = ["ReportGenerator", "market_snapshot", "top_movers",
           "render_pdf", "DEFAULT_REPORTS_DIR"]
