"""Professional PDF rendering for QuantTerm research dossiers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DARK_GREEN = colors.HexColor("#073D35")
TEAL = colors.HexColor("#36D9C0")
PALE = colors.HexColor("#EEF5F1")
INK = colors.HexColor("#15231F")
MUTED = colors.HexColor("#64746E")
AMBER = colors.HexColor("#A46516")
WHITE = colors.white


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("QTTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=DARK_GREEN, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ParagraphStyle("QTSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5, leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
        "h1": ParagraphStyle("QTH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=21, textColor=DARK_GREEN, spaceBefore=5, spaceAfter=8),
        "h2": ParagraphStyle("QTH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=DARK_GREEN, spaceBefore=6, spaceAfter=5),
        "body": ParagraphStyle("QTBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("QTSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=10, textColor=MUTED),
        "cover_label": ParagraphStyle("QTCoverLabel", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED, alignment=TA_CENTER),
        "cover_value": ParagraphStyle("QTCoverValue", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=DARK_GREEN, alignment=TA_CENTER),
        "table_head": ParagraphStyle("QTTableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=WHITE),
        "table_body": ParagraphStyle("QTTableBody", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.3, textColor=INK),
        "bullet": ParagraphStyle("QTBullet", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.4, textColor=INK, leftIndent=10, firstLineIndent=-7, bulletIndent=1, spaceAfter=3),
    }


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value in (None, ""):
        return "Not available"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _p(text: Any, style) -> Paragraph:
    value = str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(value, style)


def _bullets(items: Iterable[Any], styles, *, empty: str = "No recorded evidence.") -> list[Any]:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        values = [empty]
    return [_p(f"- {item}", styles["bullet"]) for item in values]


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(DARK_GREEN)
    canvas.rect(0, height - 13 * mm, width, 13 * mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(18 * mm, height - 8.5 * mm, "QUANTTERM RESEARCH")
    canvas.setFont("Helvetica", 6.8)
    canvas.drawRightString(width - 18 * mm, height - 8.5 * mm, "Evidence first - current decision aid")
    canvas.setStrokeColor(colors.HexColor("#CFC8B8"))
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(18 * mm, 8 * mm, "QuantTerm Research - not investment advice")
    canvas.drawRightString(width - 18 * mm, 8 * mm, str(doc.page))
    canvas.restoreState()


def _metric_cards(cards: list[tuple[str, str]], styles) -> Table:
    values = [_p(value, styles["cover_value"]) for label, value in cards]
    labels = [_p(label.upper(), styles["cover_label"]) for label, value in cards]
    table = Table([values, labels], colWidths=[40 * mm] * len(cards), rowHeights=[14 * mm, 7 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8D1C3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8D1C3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _callout(label: str, text: str, styles, tone=TEAL) -> Table:
    table = Table([[_p(label.upper(), styles["table_head"]), _p(text, styles["body"])]], colWidths=[34 * mm, 137 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), DARK_GREEN),
        ("BACKGROUND", (1, 0), (1, 0), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, tone),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _data_table(headers: list[str], rows: list[list[Any]], styles, widths=None) -> Table:
    data = [[_p(item, styles["table_head"]) for item in headers]]
    data.extend([[_p(item, styles["table_body"]) for item in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFCFCF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _chart_image(frame: Any) -> Image | None:
    if frame is None or len(frame) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = frame.tail(260).copy()
        close = data["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=140)
        ax.plot(data.index, close, linewidth=1.6, label="Close")
        ax.plot(data.index, ema20, linewidth=1.0, label="EMA 20")
        ax.plot(data.index, ema50, linewidth=1.0, label="EMA 50")
        ax.set_title("Official NSE price history")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        image = Image(buf, width=171 * mm, height=66 * mm)
        image._qt_buffer = buf
        return image
    except Exception:
        return None


def _cover(dossier: Mapping[str, Any], styles) -> list[Any]:
    price = dossier.get("price", {})
    long_term = dossier.get("long_term", {})
    story: list[Any] = [Spacer(1, 12 * mm)]
    brand = Table([[
        _p("QUANTTERM", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=22, textColor=WHITE, alignment=TA_CENTER)),
        _p("PROFESSIONAL EQUITY RESEARCH", ParagraphStyle("BrandSub", fontName="Helvetica", fontSize=8, textColor=WHITE, alignment=TA_CENTER)),
    ]], colWidths=[70 * mm, 101 * mm], rowHeights=[22 * mm])
    brand.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0, DARK_GREEN),
    ]))
    story += [brand, Spacer(1, 16 * mm), _p(dossier.get("company"), styles["title"]), _p(f"{dossier.get('symbol')} - Equity Research Dossier", styles["subtitle"])]
    story.append(_p("A source-traced brief covering price structure, current fundamentals, market context, curated events, risks and unresolved research gaps.", styles["subtitle"]))
    cards = [
        ("Classification", str(dossier.get("classification", "UNCLASSIFIED")).replace("_", " ")),
        ("Current price", f"INR {_fmt(price.get('latest_price'), 2)}" if price.get("latest_price") is not None else "Not available"),
        ("Combined score", _fmt(long_term.get("combined_score"), 1)),
        ("Evidence coverage", f"{dossier.get('coverage_pct', 0)}%"),
    ]
    story += [Spacer(1, 8 * mm), _metric_cards(cards, styles), Spacer(1, 10 * mm)]
    thesis = dossier.get("thesis", [])
    story.append(_callout("Report frame", thesis[0] if thesis else "Insufficient traced evidence for a positive thesis.", styles))
    story += [Spacer(1, 9 * mm), _p(f"Sector: {dossier.get('sector', 'Unclassified')}", styles["body"]), _p(f"Generated: {dossier.get('generated_at', '')}", styles["small"])]
    return story


def _fundamental_rows(fundamentals: Mapping[str, Any]) -> list[list[str]]:
    labels = {
        "market_cap": ("Market capitalisation", "INR Cr"),
        "pe": ("P/E", "x"),
        "roe": ("ROE", "%"),
        "roce": ("ROCE", "%"),
        "sales_growth_3y": ("Sales CAGR - 3Y", "%"),
        "profit_growth_3y": ("Profit CAGR - 3Y", "%"),
        "debt_to_equity": ("Debt / equity", "x"),
        "interest_coverage": ("Interest coverage", "x"),
        "cfo_to_pat": ("CFO / PAT", "x"),
        "promoter_holding": ("Promoter holding", "%"),
        "promoter_pledge": ("Promoter pledge", "%"),
        "fii_holding": ("FII holding", "%"),
        "dii_holding": ("DII holding", "%"),
    }
    rows = []
    for key, (label, unit) in labels.items():
        if key in fundamentals and fundamentals.get(key) not in (None, ""):
            rows.append([label, _fmt(fundamentals.get(key), 2), unit, "Current snapshot"])
    return rows


def render_equity_pdf(dossier: Mapping[str, Any], output: str | Path) -> Path:
    styles = _styles()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm, title=f"{dossier.get('symbol')} QuantTerm Research")
    story: list[Any] = _cover(dossier, styles)

    story += [PageBreak(), _p("Research frame", styles["h1"]), _p("What QuantTerm knows - and what it refuses to invent", styles["subtitle"])]
    story.append(_callout("Primary thesis", dossier.get("thesis", [""])[0], styles))
    story += [Spacer(1, 5 * mm), _p("Why it qualified", styles["h2"]), *_bullets(dossier.get("thesis", []), styles)]
    story += [_p("Current risks", styles["h2"]), *_bullets(dossier.get("risks", []), styles)]
    story += [_p("Open research items", styles["h2"]), *_bullets(dossier.get("open_items", []), styles, empty="No open items recorded.")]

    story += [PageBreak(), _p("Price structure and market context", styles["h1"])]
    chart = _chart_image(dossier.get("_frame"))
    if chart is not None:
        story += [chart, Spacer(1, 4 * mm)]
    else:
        story.append(_callout("Chart status", "Official price history is unavailable; no synthetic chart was drawn.", styles, tone=AMBER))
    price = dossier.get("price", {})
    price_rows = [
        ["Latest price", f"INR {_fmt(price.get('latest_price'), 2)}", "Latest session", str(price.get("latest_date") or "Not available")],
        ["1 month return", _fmt(price.get("return_1m_pct"), 2, "%"), "3 month return", _fmt(price.get("return_3m_pct"), 2, "%")],
        ["6 month return", _fmt(price.get("return_6m_pct"), 2, "%"), "12 month return", _fmt(price.get("return_12m_pct"), 2, "%")],
        ["52 week high", f"INR {_fmt(price.get('high_52w'), 2)}", "Distance from high", _fmt(price.get("from_high_pct"), 2, "%")],
    ]
    story += [_data_table(["Metric", "Value", "Metric", "Value"], price_rows, styles, [34 * mm, 50 * mm, 38 * mm, 49 * mm]), Spacer(1, 5 * mm)]
    market = dossier.get("market", {})
    story += [_callout("Market posture", str(market.get("trade_stance") or market.get("summary") or "Market context unavailable."), styles), Spacer(1, 4 * mm), _p("Technical evidence", styles["h2"]), *_bullets(dossier.get("technical_evidence", []), styles)]

    story += [PageBreak(), _p("Financial quality and valuation", styles["h1"])]
    fundamentals = dossier.get("fundamentals", {})
    frows = _fundamental_rows(fundamentals)
    if frows:
        story.append(_data_table(["Metric", "Value", "Unit", "Evidence status"], frows, styles, [58 * mm, 31 * mm, 24 * mm, 58 * mm]))
    else:
        story.append(_callout("Data gap", "Current fundamental metrics are not available. This section is intentionally blank rather than estimated.", styles, tone=AMBER))
    story += [Spacer(1, 5 * mm), _p("Recorded quality factors", styles["h2"]), *_bullets(dossier.get("quality_factors", []), styles)]
    long_term = dossier.get("long_term", {})
    score_rows = [
        ["Technical score", _fmt(long_term.get("technical_score"), 1), "Fundamental score", _fmt(long_term.get("fundamental_score"), 1)],
        ["Combined score", _fmt(long_term.get("combined_score"), 1), "Fundamental coverage", _fmt((long_term.get("fundamental_coverage") or 0) * 100, 1, "%")],
        ["Classification", str(long_term.get("classification") or "Not available").replace("_", " "), "Timing", str(long_term.get("timing") or "Not available").replace("_", " ")],
    ]
    story += [Spacer(1, 5 * mm), _data_table(["Field", "Value", "Field", "Value"], score_rows, styles, [39 * mm, 45 * mm, 43 * mm, 44 * mm])]

    story += [PageBreak(), _p("Management, filings and event evidence", styles["h1"])]
    evidence = dossier.get("management_evidence", [])
    if evidence:
        rows = [[item.get("published_at", "")[:10], item.get("source", ""), item.get("headline", ""), item.get("why_it_matters", "")] for item in evidence]
        story.append(_data_table(["Date", "Source", "Event", "Why it matters"], rows, styles, [22 * mm, 31 * mm, 52 * mm, 66 * mm]))
    else:
        story.append(_callout("Evidence gap", "No traced management commentary, filing or company-linked event is currently available. QuantTerm will not manufacture quotations.", styles, tone=AMBER))

    story += [PageBreak(), _p("Ownership and derivatives context", styles["h1"])]
    ownership_rows = [
        ["Promoter holding", _fmt(fundamentals.get("promoter_holding"), 2, "%"), "Promoter pledge", _fmt(fundamentals.get("promoter_pledge"), 2, "%")],
        ["FII holding", _fmt(fundamentals.get("fii_holding"), 2, "%"), "DII holding", _fmt(fundamentals.get("dii_holding"), 2, "%")],
    ]
    story += [_data_table(["Ownership field", "Value", "Ownership field", "Value"], ownership_rows, styles, [43 * mm, 42 * mm, 43 * mm, 43 * mm]), Spacer(1, 5 * mm)]
    fno = dossier.get("fno", {})
    if fno:
        fno_rows = [[fno.get("future_symbol", ""), fno.get("expiry", ""), fno.get("lot_size", ""), fno.get("contract_count", "")]]
        story += [_p("Current F&O metadata", styles["h2"]), _data_table(["Nearest future", "Expiry", "Lot size", "Contracts"], fno_rows, styles, [55 * mm, 40 * mm, 35 * mm, 41 * mm])]
    else:
        story.append(_callout("F&O status", "No current mapped stock-futures contract is available for this symbol.", styles, tone=AMBER))
    story += [Spacer(1, 5 * mm), _callout("Institutional evidence rule", "Quarterly FII/DII changes must come from traced exchange/shareholding data. Absence is disclosed; it is never inferred from price or volume.", styles)]

    story += [PageBreak(), _p("Why QuantTerm shortlisted it", styles["h1"]), *_bullets(dossier.get("thesis", []), styles)]
    story += [_p("What can break the thesis", styles["h2"]), *_bullets(dossier.get("risks", []), styles)]
    story += [Spacer(1, 6 * mm), _callout("Decision rule", "A professional report organises evidence. It does not convert incomplete evidence into certainty or a guaranteed trade.", styles)]

    story += [PageBreak(), _p("Source ledger and research note", styles["h1"])]
    source_rows = [[item.get("name", ""), item.get("status", ""), item.get("timestamp", ""), "Yes" if item.get("point_in_time") else "No", item.get("note", "")] for item in dossier.get("sources", [])]
    story.append(_data_table(["Source", "Status", "Timestamp", "PIT", "Note"], source_rows, styles, [39 * mm, 24 * mm, 33 * mm, 13 * mm, 62 * mm]))
    story += [Spacer(1, 6 * mm), _p("Open items before circulation", styles["h2"]), *_bullets(dossier.get("open_items", []), styles)]
    story += [Spacer(1, 5 * mm), _callout("Research note", dossier.get("disclaimer", ""), styles)]

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


def render_basket_pdf(basket: Mapping[str, Any], output: str | Path) -> Path:
    styles = _styles()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm, title=basket.get("title", "QuantTerm Basket Report"))
    companies = list(basket.get("companies", []) or [])
    story: list[Any] = [Spacer(1, 12 * mm)]
    story += [_p(basket.get("title", "QuantTerm Research Basket"), styles["title"]), _p(basket.get("subtitle", ""), styles["subtitle"])]
    story.append(_metric_cards([
        ("Companies", str(len(companies))),
        ("Sectors", str(len({item.get('sector') for item in companies if item.get('sector')}))),
        ("Generated", str(basket.get("generated_at", ""))[:10]),
        ("Framework", "Evidence first"),
    ], styles))
    story += [Spacer(1, 8 * mm), _callout("Report frame", "A current research basket assembled from the Long-Term shortlist, official price history, current fundamentals, curated events and explicit data gaps.", styles)]

    for dossier in companies:
        story += [PageBreak(), _p(f"{dossier.get('company')} ({dossier.get('symbol')})", styles["h1"]), _p(f"{dossier.get('sector')} - {str(dossier.get('classification')).replace('_', ' ')}", styles["subtitle"])]
        price = dossier.get("price", {})
        long_term = dossier.get("long_term", {})
        story.append(_metric_cards([
            ("Price", f"INR {_fmt(price.get('latest_price'), 2)}" if price.get("latest_price") is not None else "N/A"),
            ("Combined score", _fmt(long_term.get("combined_score"), 1)),
            ("Coverage", f"{dossier.get('coverage_pct', 0)}%"),
            ("52W distance", _fmt(price.get("from_high_pct"), 1, "%")),
        ], styles))
        story += [Spacer(1, 5 * mm), _p("Why it qualified", styles["h2"]), *_bullets(dossier.get("thesis", [])[:6], styles), _p("Risks", styles["h2"]), *_bullets(dossier.get("risks", [])[:5], styles)]
        frows = _fundamental_rows(dossier.get("fundamentals", {}))[:8]
        if frows:
            story += [Spacer(1, 4 * mm), _data_table(["Metric", "Value", "Unit", "Evidence status"], frows, styles, [58 * mm, 31 * mm, 24 * mm, 58 * mm])]
        story += [PageBreak(), _p(f"{dossier.get('symbol')}: price and event evidence", styles["h1"])]
        chart = _chart_image(dossier.get("_frame"))
        if chart:
            story.append(chart)
        events = dossier.get("management_evidence", [])[:6]
        if events:
            rows = [[item.get("published_at", "")[:10], item.get("headline", ""), item.get("why_it_matters", "")] for item in events]
            story += [Spacer(1, 5 * mm), _data_table(["Date", "Event", "Why it matters"], rows, styles, [24 * mm, 63 * mm, 84 * mm])]
        else:
            story += [Spacer(1, 5 * mm), _callout("Event evidence", "No traced company-linked filing or management event is currently available.", styles, tone=AMBER)]

    story += [PageBreak(), _p("Cross-company synthesis", styles["h1"])]
    synth_rows = [[item.get("company"), item.get("sector"), str(item.get("classification")).replace("_", " "), (item.get("thesis") or ["No thesis"])[0]] for item in companies]
    story.append(_data_table(["Company", "Sector", "Current class", "Primary evidence"], synth_rows, styles, [39 * mm, 30 * mm, 36 * mm, 66 * mm]))
    story += [Spacer(1, 6 * mm), _p("Common quality signals", styles["h2"]), *_bullets(basket.get("common_quality", []), styles), _p("Common risks", styles["h2"]), *_bullets(basket.get("common_risks", []), styles)]
    story += [_p("Open items before circulation", styles["h2"]), *_bullets(basket.get("open_items", [])[:15], styles), Spacer(1, 5 * mm), _callout("Research note", basket.get("disclaimer", ""), styles)]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output
