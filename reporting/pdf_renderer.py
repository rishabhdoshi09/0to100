"""Professional PDF rendering for QuantTerm research dossiers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
RED = colors.HexColor("#A83B3B")
WHITE = colors.white


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("QTTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=DARK_GREEN, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ParagraphStyle("QTSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5, leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
        "h1": ParagraphStyle("QTH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=21, textColor=DARK_GREEN, spaceBefore=5, spaceAfter=8),
        "h2": ParagraphStyle("QTH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=DARK_GREEN, spaceBefore=6, spaceAfter=5),
        "body": ParagraphStyle("QTBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("QTSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.6, textColor=MUTED),
        "cover_label": ParagraphStyle("QTCoverLabel", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED, alignment=TA_CENTER),
        "cover_value": ParagraphStyle("QTCoverValue", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=DARK_GREEN, alignment=TA_CENTER),
        "table_head": ParagraphStyle("QTTableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.1, leading=8.8, textColor=WHITE),
        "table_body": ParagraphStyle("QTTableBody", parent=base["BodyText"], fontName="Helvetica", fontSize=6.9, leading=8.8, textColor=INK),
        "bullet": ParagraphStyle("QTBullet", parent=base["BodyText"], fontName="Helvetica", fontSize=8.7, leading=12.2, textColor=INK, leftIndent=10, firstLineIndent=-7, bulletIndent=1, spaceAfter=3),
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
    canvas.drawRightString(width - 18 * mm, height - 8.5 * mm, "Evidence first - dates and gaps disclosed")
    canvas.setStrokeColor(colors.HexColor("#CFC8B8"))
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(18 * mm, 8 * mm, "QuantTerm Research - not investment advice")
    canvas.drawRightString(width - 18 * mm, 8 * mm, str(doc.page))
    canvas.restoreState()


def _metric_cards(cards: list[tuple[str, str]], styles) -> Table:
    count = max(1, len(cards))
    width = 160 * mm / count
    values = [_p(value, styles["cover_value"]) for _, value in cards]
    labels = [_p(label.upper(), styles["cover_label"]) for label, _ in cards]
    table = Table([values, labels], colWidths=[width] * count, rowHeights=[14 * mm, 7 * mm])
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
        _p("EQUITY EVIDENCE BRIEF", ParagraphStyle("BrandSub", fontName="Helvetica", fontSize=8, textColor=WHITE, alignment=TA_CENTER)),
    ]], colWidths=[70 * mm, 101 * mm], rowHeights=[22 * mm])
    brand.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0, DARK_GREEN),
    ]))
    story += [brand, Spacer(1, 16 * mm), _p(dossier.get("company"), styles["title"]), _p(f"{dossier.get('symbol')} - source-traced research dossier", styles["subtitle"])]
    story.append(_p("Price, financial, ownership, event and uploaded evidence are dated separately. Missing inputs are accompanied by official retrieval and upload instructions.", styles["subtitle"]))
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
        "market_cap_cr": ("Market capitalisation", "INR Cr"),
        "market_cap": ("Market capitalisation", "INR Cr"),
        "pe": ("P/E", "x"), "roe": ("ROE", "%"), "roce": ("ROCE", "%"),
        "sales_growth_3y": ("Sales CAGR - 3Y", "%"), "profit_growth_3y": ("Profit CAGR - 3Y", "%"),
        "debt_to_equity": ("Debt / equity", "x"), "interest_coverage": ("Interest coverage", "x"),
        "cfo_to_pat": ("CFO / PAT", "x"), "promoter_holding": ("Promoter holding", "%"),
        "promoter_pledge": ("Promoter pledge", "%"), "fii_holding": ("FII holding", "%"),
        "dii_holding": ("DII holding", "%"),
    }
    rows = []
    seen = set()
    for key, (label, unit) in labels.items():
        if label in seen:
            continue
        if key in fundamentals and fundamentals.get(key) not in (None, ""):
            rows.append([label, _fmt(fundamentals.get(key), 2), unit, "Current snapshot"])
            seen.add(label)
    return rows


def _normalised_table(rows: Sequence[Mapping[str, Any]], styles, *, limit_rows: int = 12) -> Table | None:
    rows = list(rows or [])[:limit_rows]
    if not rows:
        return None
    periods: list[str] = []
    for row in rows:
        for key in (row.get("values") or {}).keys():
            if key not in periods:
                periods.append(str(key))
    periods = periods[-5:]
    headers = ["Metric", *periods]
    body = []
    for row in rows:
        values = row.get("values") or {}
        body.append([row.get("label") or "—", *[_fmt(values.get(period), 2) if values.get(period) is not None else "—" for period in periods]])
    widths = [52 * mm] + [119 * mm / max(1, len(periods))] * len(periods)
    return _data_table(headers, body, styles, widths)


def _structured_table(rows: Sequence[Mapping[str, Any]], styles, preferred: Sequence[str], *, limit_rows: int = 12) -> Table | None:
    rows = list(rows or [])[:limit_rows]
    if not rows:
        return None
    keys = [key for key in preferred if any(str(row.get(key, "")).strip() for row in rows)]
    if not keys:
        keys = list(rows[0].keys())[:6]
    body = [[row.get(key, "") for key in keys] for row in rows]
    widths = [171 * mm / max(1, len(keys))] * len(keys)
    return _data_table([key.replace("_", " ").title() for key in keys], body, styles, widths)


def _as_of_callout(label: str, as_of: Any, styles) -> Table:
    return _callout(label, f"Data as of: {as_of or 'UNKNOWN - treat this section as undated'}", styles, tone=AMBER if not as_of else TEAL)


def render_equity_pdf(dossier: Mapping[str, Any], output: str | Path) -> Path:
    styles = _styles()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm, title=f"{dossier.get('symbol')} QuantTerm Research")
    story: list[Any] = _cover(dossier, styles)

    story += [PageBreak(), _p("Evidence coverage", styles["h1"]), _p("Every section is weighted and dated separately", styles["subtitle"])]
    coverage_rows = [[item.get("key", "").replace("_", " ").title(), item.get("weight"), item.get("status"), item.get("as_of") or "Unknown", item.get("age_days") if item.get("age_days") is not None else "—"] for item in dossier.get("section_coverage", [])]
    story.append(_data_table(["Section", "Weight", "Status", "As of", "Age days"], coverage_rows, styles, [52 * mm, 18 * mm, 25 * mm, 49 * mm, 27 * mm]))
    story += [Spacer(1, 5 * mm), _p("Primary thesis", styles["h2"]), *_bullets(dossier.get("thesis", []), styles), _p("Current risks", styles["h2"]), *_bullets(dossier.get("risks", []), styles)]

    story += [PageBreak(), _p("Business profile", styles["h1"])]
    story.append(_as_of_callout("Fundamental source", dossier.get("deep_fundamentals_fetched_at"), styles))
    about = str(dossier.get("company_about") or "").strip()
    if about:
        story += [Spacer(1, 5 * mm), _p(about, styles["body"])]
    else:
        story += [Spacer(1, 5 * mm), _callout("Data gap", "No traced business description is available. Use the Research Data workspace to open the annual-report source or upload a business-profile template.", styles, tone=AMBER)]
    segments = dossier.get("business_segments", [])
    segment_table = _structured_table(segments, styles, ("period_end", "segment", "revenue_cr", "revenue_mix_pct", "growth_pct", "driver"))
    if segment_table:
        story += [_p("Business and segment mix", styles["h2"]), segment_table]
    else:
        story += [_p("Business and segment mix", styles["h2"]), _callout("Missing", "Segment mix has not been extracted or uploaded. No estimated mix is shown.", styles, tone=AMBER)]

    story += [PageBreak(), _p("Price structure and market context", styles["h1"])]
    story.append(_as_of_callout("Official price history", dossier.get("price", {}).get("latest_date"), styles))
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
    story.append(_as_of_callout("Deep fundamentals", dossier.get("deep_fundamentals_fetched_at"), styles))
    fundamentals = dossier.get("fundamentals", {})
    frows = _fundamental_rows(fundamentals)
    if frows:
        story.append(_data_table(["Metric", "Value", "Unit", "Evidence status"], frows, styles, [58 * mm, 31 * mm, 24 * mm, 58 * mm]))
    else:
        story.append(_callout("Data gap", "Current fundamental metrics are unavailable. This section is intentionally blank rather than estimated.", styles, tone=AMBER))
    story += [Spacer(1, 5 * mm), _p("Recorded quality factors", styles["h2"]), *_bullets(dossier.get("quality_factors", []), styles)]
    long_term = dossier.get("long_term", {})
    score_rows = [
        ["Technical score", _fmt(long_term.get("technical_score"), 1), "Fundamental score", _fmt(long_term.get("fundamental_score"), 1)],
        ["Combined score", _fmt(long_term.get("combined_score"), 1), "Fundamental coverage", _fmt((long_term.get("fundamental_coverage") or 0) * 100, 1, "%")],
        ["Classification", str(long_term.get("classification") or "Not available").replace("_", " "), "Timing", str(long_term.get("timing") or "Not available").replace("_", " ")],
    ]
    story += [Spacer(1, 5 * mm), _data_table(["Field", "Value", "Field", "Value"], score_rows, styles, [39 * mm, 45 * mm, 43 * mm, 44 * mm])]

    financial = dossier.get("financial_tables", {})
    for title, key in (("Quarterly results", "quarterly_results"), ("Annual profit and loss", "profit_loss"), ("Balance sheet", "balance_sheet"), ("Cash flow", "cash_flow")):
        table = _normalised_table(financial.get(key, []), styles)
        story += [PageBreak(), _p(title, styles["h1"]), _as_of_callout(title, financial.get("as_of"), styles)]
        if table:
            story += [Spacer(1, 4 * mm), table]
        else:
            story += [Spacer(1, 4 * mm), _callout("Data gap", f"No traced {title.lower()} table is available. Download the exchange result/annual report or upload the structured financial-history template.", styles, tone=AMBER)]
    uploaded_financial = _structured_table(financial.get("uploaded", []), styles, ("period_end", "period_type", "revenue_cr", "ebitda_cr", "ebitda_margin_pct", "pat_cr"))
    if uploaded_financial:
        story += [PageBreak(), _p("User-supplied structured financial history", styles["h1"]), uploaded_financial]

    story += [PageBreak(), _p("Management, filings and guidance", styles["h1"])]
    evidence = dossier.get("management_evidence", [])
    if evidence:
        rows = [[item.get("published_at", "")[:10], item.get("speaker") or item.get("source", ""), item.get("headline", ""), item.get("why_it_matters", "")] for item in evidence]
        story.append(_data_table(["Date", "Speaker/source", "Event", "Evidence"], rows, styles, [22 * mm, 35 * mm, 48 * mm, 66 * mm]))
    else:
        story.append(_callout("Evidence gap", "No traced management transcript, filing or company-linked event is available. QuantTerm will not manufacture quotations.", styles, tone=AMBER))
    guidance_table = _structured_table(dossier.get("order_book_guidance", []), styles, ("as_of_date", "metric", "value", "unit", "period", "management_wording"))
    if guidance_table:
        story += [Spacer(1, 5 * mm), _p("Order book and forward guidance", styles["h2"]), guidance_table]
    else:
        story += [Spacer(1, 5 * mm), _callout("Guidance gap", "No structured order-book or forward-guidance evidence has been attached.", styles, tone=AMBER)]

    story += [PageBreak(), _p("Ownership and derivatives context", styles["h1"])]
    story.append(_as_of_callout("Shareholding evidence", dossier.get("deep_fundamentals_fetched_at"), styles))
    ownership_rows = [
        ["Promoter holding", _fmt(fundamentals.get("promoter_holding"), 2, "%"), "Promoter pledge", _fmt(fundamentals.get("promoter_pledge"), 2, "%")],
        ["FII holding", _fmt(fundamentals.get("fii_holding"), 2, "%"), "DII holding", _fmt(fundamentals.get("dii_holding"), 2, "%")],
    ]
    story += [_data_table(["Ownership field", "Value", "Ownership field", "Value"], ownership_rows, styles, [43 * mm, 42 * mm, 43 * mm, 43 * mm]), Spacer(1, 5 * mm)]
    share_table = _normalised_table(dossier.get("shareholding_history", []), styles, limit_rows=18)
    if share_table:
        story += [_p("Shareholding series", styles["h2"]), share_table]
    else:
        story += [_callout("Ownership gap", "Quarterly shareholding history is unavailable. Use the NSE shareholding link or upload the QuantTerm shareholding CSV template.", styles, tone=AMBER)]
    fno = dossier.get("fno", {})
    if fno:
        fno_rows = [[fno.get("future_symbol", ""), fno.get("expiry", ""), fno.get("lot_size", ""), fno.get("contract_count", "")]]
        story += [Spacer(1, 5 * mm), _p("Current F&O metadata", styles["h2"]), _data_table(["Nearest future", "Expiry", "Lot size", "Contracts"], fno_rows, styles, [55 * mm, 40 * mm, 35 * mm, 41 * mm])]
    else:
        story += [Spacer(1, 5 * mm), _callout("F&O status", "No current mapped stock-futures contract is available for this symbol.", styles, tone=AMBER)]

    story += [PageBreak(), _p("Why QuantTerm shortlisted it", styles["h1"]), *_bullets(dossier.get("thesis", []), styles)]
    story += [_p("What can break the thesis", styles["h2"]), *_bullets(dossier.get("risks", []), styles)]
    story += [Spacer(1, 6 * mm), _callout("Decision rule", "A professional report organises evidence. It does not convert incomplete evidence into certainty or a guaranteed trade.", styles)]

    story += [PageBreak(), _p("Source ledger", styles["h1"])]
    source_rows = [[item.get("name", ""), item.get("status", ""), item.get("as_of", "") or "Unknown", "Yes" if item.get("point_in_time") else "No", item.get("note", "")] for item in dossier.get("sources", [])]
    story.append(_data_table(["Source", "Status", "As of", "PIT", "Note"], source_rows, styles, [37 * mm, 25 * mm, 31 * mm, 13 * mm, 65 * mm]))

    story += [PageBreak(), _p("How to complete missing research", styles["h1"]), _p("Official source, instructions, accepted files and strict date status", styles["subtitle"])]
    req_rows = []
    for item in dossier.get("evidence_requirements", {}).get("requirements", []):
        links = item.get("links", [])
        first = links[0].get("url", "") if links else ""
        req_rows.append([
            item.get("label", ""), item.get("status", ""), item.get("as_of") or "Unknown",
            item.get("instructions", ""), first,
        ])
    story.append(_data_table(["Dataset", "Status", "As of", "What to do", "Primary source"], req_rows, styles, [30 * mm, 20 * mm, 24 * mm, 58 * mm, 39 * mm]))
    story += [Spacer(1, 6 * mm), _p("Open items before circulation", styles["h2"]), *_bullets(dossier.get("open_items", []), styles), Spacer(1, 5 * mm), _callout("Research note", dossier.get("disclaimer", ""), styles)]

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


def render_basket_pdf(basket: Mapping[str, Any], output: str | Path) -> Path:
    styles = _styles()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm, title=basket.get("title", "QuantTerm Basket Report"))
    companies = list(basket.get("companies", []) or [])
    story: list[Any] = [Spacer(1, 12 * mm), _p(basket.get("title", "QuantTerm Research Basket"), styles["title"]), _p(basket.get("subtitle", ""), styles["subtitle"])]
    story.append(_metric_cards([
        ("Companies", str(len(companies))),
        ("Sectors", str(len({item.get('sector') for item in companies if item.get('sector')}))),
        ("Generated", str(basket.get("generated_at", ""))[:10]),
        ("Framework", "Evidence first"),
    ], styles))
    story += [Spacer(1, 8 * mm), _callout("Report frame", "A current research basket assembled from persisted evidence. Every company carries its own coverage score, source dates and missing-data instructions.", styles)]

    for dossier in companies:
        story += [PageBreak(), _p(f"{dossier.get('company')} ({dossier.get('symbol')})", styles["h1"]), _p(f"{dossier.get('sector')} - {str(dossier.get('classification')).replace('_', ' ')}", styles["subtitle"])]
        price = dossier.get("price", {})
        long_term = dossier.get("long_term", {})
        story.append(_metric_cards([
            ("Price", f"INR {_fmt(price.get('latest_price'), 2)}" if price.get("latest_price") is not None else "N/A"),
            ("Combined score", _fmt(long_term.get("combined_score"), 1)),
            ("Coverage", f"{dossier.get('coverage_pct', 0)}%"),
            ("Price as of", str(price.get("latest_date") or "Unknown")),
        ], styles))
        if dossier.get("company_about"):
            story += [Spacer(1, 4 * mm), _p(dossier.get("company_about"), styles["body"])]
        story += [Spacer(1, 5 * mm), _p("Why it qualified", styles["h2"]), *_bullets(dossier.get("thesis", [])[:6], styles), _p("Risks", styles["h2"]), *_bullets(dossier.get("risks", [])[:5], styles)]
        frows = _fundamental_rows(dossier.get("fundamentals", {}))[:8]
        if frows:
            story += [Spacer(1, 4 * mm), _data_table(["Metric", "Value", "Unit", "Evidence status"], frows, styles, [58 * mm, 31 * mm, 24 * mm, 58 * mm])]
        chart = _chart_image(dossier.get("_frame"))
        if chart:
            story += [PageBreak(), _p(f"{dossier.get('symbol')}: price and event evidence", styles["h1"]), chart]
        events = dossier.get("management_evidence", [])[:6]
        if events:
            rows = [[item.get("published_at", "")[:10], item.get("headline", ""), item.get("why_it_matters", "")] for item in events]
            story += [Spacer(1, 5 * mm), _data_table(["Date", "Event", "Why it matters"], rows, styles, [24 * mm, 63 * mm, 84 * mm])]
        else:
            story += [Spacer(1, 5 * mm), _callout("Event evidence", "No traced company-linked filing or management event is currently available.", styles, tone=AMBER)]

    story += [PageBreak(), _p("Cross-company synthesis", styles["h1"])]
    synth_rows = [[item.get("company"), item.get("sector"), str(item.get("classification")).replace("_", " "), f"{item.get('coverage_pct', 0)}%", (item.get("thesis") or ["No thesis"])[0]] for item in companies]
    story.append(_data_table(["Company", "Sector", "Current class", "Coverage", "Primary evidence"], synth_rows, styles, [34 * mm, 27 * mm, 33 * mm, 20 * mm, 57 * mm]))
    story += [Spacer(1, 6 * mm), _p("Common quality signals", styles["h2"]), *_bullets(basket.get("common_quality", []), styles), _p("Common risks", styles["h2"]), *_bullets(basket.get("common_risks", []), styles)]
    story += [_p("Open items before circulation", styles["h2"]), *_bullets(basket.get("open_items", [])[:20], styles), Spacer(1, 5 * mm), _callout("Research note", basket.get("disclaimer", ""), styles)]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output
