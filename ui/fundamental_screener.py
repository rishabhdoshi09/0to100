"""Stock Screener — find stocks by fundamentals. No jargon."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

# ── Colour palette (matches DevBloom theme) ───────────────────────────────────
_CYAN   = "#00d4ff"
_GREEN  = "#00d4a0"
_AMBER  = "#f59e0b"
_RED    = "#ff4b4b"
_WHITE  = "#e8eaf0"
_MUTED  = "#8892a4"
_CARD   = "rgba(255,255,255,0.04)"
_BORDER = "rgba(255,255,255,0.08)"

# ── Preset filter definitions ─────────────────────────────────────────────────
_PRESETS = {
    "💎 Quality Growth": {
        "pe_max":        30.0,
        "roe_min":       20.0,
        "de_max":         0.5,
        "promoter_min":  50.0,
        "mcap_min":       0.0,
        "div_yield_min":  0.0,
    },
    "🏦 Large Cap Value": {
        "pe_max":        20.0,
        "roe_min":       15.0,
        "de_max":         5.0,
        "promoter_min":   0.0,
        "mcap_min":   10_000.0,
        "div_yield_min":  0.0,
    },
    "💰 High Dividend": {
        "pe_max":        25.0,
        "roe_min":        0.0,
        "de_max":         5.0,
        "promoter_min":   0.0,
        "mcap_min":       0.0,
        "div_yield_min":  2.0,
    },
}

# ── Data extraction helpers ───────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").replace("%", "").replace("₹", "").strip())
    except Exception:
        return default


def _find_ratio(ratios: list, *keywords: str) -> Optional[float]:
    """Search key_ratios list for the first entry whose name contains any keyword."""
    if not ratios:
        return None
    kws = [k.lower() for k in keywords]
    for r in ratios:
        name = str(r.get("name", "")).lower()
        if any(kw in name for kw in kws):
            return _safe_float(r.get("value"), default=None)  # type: ignore[arg-type]
    return None


def _find_shareholding_promoter(shareholding: list) -> Optional[float]:
    """Extract latest promoter holding % from shareholding table rows."""
    if not shareholding:
        return None
    for row in shareholding:
        label = str(list(row.values())[0] if row else "").lower()
        if "promoter" in label:
            # Last numeric column is most recent quarter
            vals = [v for v in list(row.values())[1:] if isinstance(v, (int, float))]
            if vals:
                return float(vals[-1])
    return None


def _get_fundamentals_yf(symbol: str) -> Dict[str, Any]:
    """Fallback: pull fundamentals from yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(f"{symbol}.NS").info
        mc_raw = info.get("marketCap", 0) or 0
        rev_curr = info.get("totalRevenue", 0) or 0
        rev_prev = 0.0  # yfinance doesn't give prior-year revenue directly
        revenue_growth = _safe_float(info.get("revenueGrowth", 0)) * 100  # fraction → %

        return {
            "pe_ratio":        _safe_float(info.get("trailingPE") or info.get("forwardPE")),
            "pb_ratio":        _safe_float(info.get("priceToBook")),
            "roe":             _safe_float(info.get("returnOnEquity", 0)) * 100,
            "debt_to_equity":  _safe_float(info.get("debtToEquity", 0)) / 100,  # yf gives 0-100 scale
            "revenue_growth":  revenue_growth,
            "promoter_holding": 0.0,     # not in yfinance
            "market_cap_cr":   mc_raw / 1e7 if mc_raw else 0.0,
            "dividend_yield":  _safe_float(info.get("dividendYield", 0)) * 100,
            "eps_growth":      _safe_float(info.get("earningsGrowth", 0)) * 100,
            "current_price":   _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            "_source":         "yfinance",
        }
    except Exception:
        return {"_source": "error"}


def _extract_from_screener(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map screener_deep raw data → flat fundamentals dict."""
    ratios     = data.get("key_ratios", [])
    pl         = data.get("profit_loss", [])
    bs         = data.get("balance_sheet", [])
    shareholding = data.get("shareholding", [])

    pe    = _find_ratio(ratios, "p/e", "pe ratio", "price/earning", "price to earning")
    pb    = _find_ratio(ratios, "p/b", "pb ratio", "price/book", "price to book")
    roe   = _find_ratio(ratios, "roe", "return on equity")
    de    = _find_ratio(ratios, "debt/equity", "debt to equity", "d/e")
    dy    = _find_ratio(ratios, "dividend yield", "div yield")
    mcap  = _find_ratio(ratios, "market cap", "mkt cap", "mcap")
    eps   = _find_ratio(ratios, "eps", "earning per share")
    price = _find_ratio(ratios, "current price", "price", "cmp", "stock p/e")

    # Revenue growth: compare most recent two years from profit_loss
    rev_growth = 0.0
    if pl:
        rev_rows = [r for r in pl if "sales" in str(list(r.values())[0] if r else "").lower()
                    or "revenue" in str(list(r.values())[0] if r else "").lower()]
        if rev_rows:
            vals = [v for v in list(rev_rows[0].values())[1:] if isinstance(v, (int, float))]
            if len(vals) >= 2 and vals[-2] and vals[-2] != 0:
                rev_growth = ((vals[-1] - vals[-2]) / abs(vals[-2])) * 100

    # EPS growth from P&L
    eps_growth = 0.0
    if pl and eps is None:
        eps_rows = [r for r in pl if "eps" in str(list(r.values())[0] if r else "").lower()]
        if eps_rows:
            vals = [v for v in list(eps_rows[0].values())[1:] if isinstance(v, (int, float))]
            if len(vals) >= 2 and vals[-2] and vals[-2] != 0:
                eps_growth = ((vals[-1] - vals[-2]) / abs(vals[-2])) * 100
            eps = vals[-1] if vals else None

    promoter = _find_shareholding_promoter(shareholding)

    return {
        "pe_ratio":         pe,
        "pb_ratio":         pb,
        "roe":              roe,
        "debt_to_equity":   de,
        "revenue_growth":   rev_growth,
        "promoter_holding": promoter,
        "market_cap_cr":    mcap,
        "dividend_yield":   dy,
        "eps_growth":       eps_growth,
        "current_price":    price,
        "_source":          "screener.in",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fetch fundamentals for one symbol — screener.in preferred, yfinance fallback."""
    try:
        from fundamentals.fetcher import get_deep_fundamentals
        raw = get_deep_fundamentals(symbol)
        result = _extract_from_screener(raw)
        # If screener gave us almost nothing, supplement with yfinance
        useful = sum(1 for k, v in result.items() if not k.startswith("_") and v and v != 0.0)
        if useful < 3:
            yf_data = _get_fundamentals_yf(symbol)
            for k, v in yf_data.items():
                if not k.startswith("_") and (result.get(k) is None or result.get(k) == 0.0):
                    result[k] = v
        return result
    except Exception:
        return _get_fundamentals_yf(symbol)


# ── Quality score ─────────────────────────────────────────────────────────────

def _quality_score(f: Dict[str, Any]) -> float:
    """
    Simple composite score (0-100) for sorting.
    ROE 40% + Promoter 20% + (1/PE * 10) 30% + Revenue growth 10%
    """
    try:
        roe      = _safe_float(f.get("roe", 0))
        promoter = _safe_float(f.get("promoter_holding", 0))
        pe       = _safe_float(f.get("pe_ratio", 0))
        rev_g    = _safe_float(f.get("revenue_growth", 0))

        pe_score = (1.0 / pe * 10) if pe and pe > 0 else 0.0
        pe_score = min(pe_score, 1.0)   # cap at 1

        score = (
            min(roe / 40.0, 1.0)       * 40.0
            + min(promoter / 75.0, 1.0) * 20.0
            + pe_score                  * 30.0
            + min(max(rev_g, 0) / 30.0, 1.0) * 10.0
        )
        return round(score, 1)
    except Exception:
        return 0.0


# ── Filter logic ──────────────────────────────────────────────────────────────

def _passes_filters(
    f: Dict[str, Any],
    pe_max: float,
    roe_min: float,
    de_max: float,
    promoter_min: float,
    mcap_min: float,
    div_yield_min: float,
    only_profitable: bool,
) -> tuple[bool, str]:
    """Returns (passes, label) where label is 'pass', 'borderline', or 'fail'."""
    pe       = _safe_float(f.get("pe_ratio"))
    roe      = _safe_float(f.get("roe"))
    de       = _safe_float(f.get("debt_to_equity"))
    promoter = _safe_float(f.get("promoter_holding"))
    mcap     = _safe_float(f.get("market_cap_cr"))
    dy       = _safe_float(f.get("dividend_yield"))

    failures = 0
    borderlines = 0

    if pe and pe > pe_max:
        diff = pe - pe_max
        if diff > pe_max * 0.1:
            failures += 1
        else:
            borderlines += 1

    if roe_min > 0 and roe < roe_min:
        diff = roe_min - roe
        if diff > roe_min * 0.15:
            failures += 1
        else:
            borderlines += 1

    if de > de_max:
        diff = de - de_max
        if diff > de_max * 0.15:
            failures += 1
        else:
            borderlines += 1

    if promoter_min > 0 and promoter < promoter_min:
        diff = promoter_min - promoter
        if diff > 5:
            failures += 1
        else:
            borderlines += 1

    if mcap_min > 0 and mcap < mcap_min:
        failures += 1

    if div_yield_min > 0 and dy < div_yield_min:
        failures += 1

    if only_profitable and roe <= 0:
        failures += 1

    if failures > 0:
        return False, "fail"
    if borderlines > 0:
        return True, "borderline"
    return True, "pass"


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _na(val: Any, fmt: str = ".1f", suffix: str = "") -> str:
    if val is None or val == 0.0:
        return "N/A"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except Exception:
        return str(val)


def _plain_english_card(symbol: str, f: Dict[str, Any], tech_setup: Optional[str]) -> None:
    """Render the expandable plain-English card for one stock."""
    pe    = f.get("pe_ratio")
    pb    = f.get("pb_ratio")
    roe   = f.get("roe")
    de    = f.get("debt_to_equity")
    prom  = f.get("promoter_holding")
    rev_g = f.get("revenue_growth")
    dy    = f.get("dividend_yield")
    eps_g = f.get("eps_growth")
    mcap  = f.get("market_cap_cr")
    price = f.get("current_price")

    lines = []

    if roe is not None:
        lines.append(
            f"**Earns ₹{_safe_float(roe):.0f} for every ₹100 invested (ROE)**"
            + (" — excellent!" if _safe_float(roe) >= 20 else " — decent" if _safe_float(roe) >= 12 else " — below average")
        )

    if pe is not None:
        pe_f = _safe_float(pe)
        lines.append(
            f"**Price/Earnings (PE): {pe_f:.1f}×**"
            + f" — you're paying ₹{pe_f:.0f} for every ₹1 the company earns"
            + (" — looks cheap!" if pe_f < 15 else " — reasonable" if pe_f < 30 else " — expensive")
        )

    if de is not None:
        de_f = _safe_float(de)
        lines.append(
            f"**Debt/Equity: {de_f:.2f}**"
            + (" — very low debt ✅" if de_f < 0.3 else " — manageable debt" if de_f < 1.0 else " — high debt ⚠️")
        )

    if prom is not None:
        prom_f = _safe_float(prom)
        lines.append(
            f"**Promoters own {prom_f:.1f}% of the company**"
            + (" — high confidence" if prom_f >= 55 else " — moderate" if prom_f >= 35 else " — low promoter skin-in-game ⚠️")
        )

    if rev_g:
        rev_f = _safe_float(rev_g)
        lines.append(
            f"**Revenue grew {rev_f:+.1f}% last year**"
            + (" — strong growth 🚀" if rev_f > 20 else " — healthy" if rev_f > 5 else " — sluggish")
        )

    if dy and _safe_float(dy) > 0:
        lines.append(f"**Dividend yield: {_safe_float(dy):.2f}%** — regular income for shareholders")

    if mcap:
        mcap_f = _safe_float(mcap)
        bucket = (
            "Large-cap" if mcap_f >= 20_000
            else "Mid-cap" if mcap_f >= 5_000
            else "Small-cap" if mcap_f >= 500
            else "Micro-cap"
        )
        lines.append(f"**Market size: ₹{mcap_f:,.0f} Cr ({bucket})**")

    if tech_setup:
        lines.append(f"**Chart pattern detected:** {tech_setup}")

    if not lines:
        lines.append("Fundamental data not available — try refreshing or check NSE/BSE directly.")

    for line in lines:
        st.markdown(f"- {line}")


# ── Main render function ──────────────────────────────────────────────────────

def render_fundamental_screener(universe: list[str]) -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;letter-spacing:2px;margin:0'>🔎 Stock Screener</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.3rem 0 .8rem'>"
        "Find good stocks using simple, plain-English filters &nbsp;·&nbsp; "
        "No finance jargon &nbsp;·&nbsp; Data from Screener.in / Yahoo Finance</p>",
        unsafe_allow_html=True,
    )

    # ── Preset buttons ────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{_MUTED};font-size:.8rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem'>"
        f"Quick Presets</p>",
        unsafe_allow_html=True,
    )

    preset_cols = st.columns(len(_PRESETS))
    chosen_preset: Optional[str] = None
    for col, label in zip(preset_cols, _PRESETS):
        with col:
            if st.button(label, key=f"preset_{label}", use_container_width=True):
                chosen_preset = label

    # Store preset in session so sliders auto-update
    if chosen_preset:
        st.session_state["_fs_preset"] = chosen_preset

    active_preset = _PRESETS.get(st.session_state.get("_fs_preset", ""), {})

    st.divider()

    # ── Filter panel ──────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{_MUTED};font-size:.8rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem'>"
        f"Filters</p>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        pe_max = st.slider(
            "Max PE Ratio (Price vs Earnings)",
            min_value=0, max_value=100,
            value=int(active_preset.get("pe_max", 50)),
            step=1,
            key="fs_pe_max",
            help="Lower = cheaper stock relative to earnings. Under 20 is often considered value territory.",
        )
        roe_min = st.slider(
            "Min Return on Equity %",
            min_value=0, max_value=40,
            value=int(active_preset.get("roe_min", 10)),
            step=1,
            key="fs_roe_min",
            help="How much profit the company earns per ₹100 of shareholder money. Higher is better.",
        )

    with f2:
        de_max = st.slider(
            "Max Debt/Equity",
            min_value=0.0, max_value=5.0,
            value=float(active_preset.get("de_max", 2.0)),
            step=0.1,
            key="fs_de_max",
            help="How much the company has borrowed vs what shareholders own. Lower is safer.",
        )
        promoter_min = st.slider(
            "Min Promoter Holding %",
            min_value=0, max_value=75,
            value=int(active_preset.get("promoter_min", 30)),
            step=5,
            key="fs_promoter_min",
            help="% of company owned by founders/promoters. Higher means founders have more skin in the game.",
        )

    with f3:
        mcap_min = st.number_input(
            "Min Market Cap (₹ Cr)",
            min_value=0.0,
            value=float(active_preset.get("mcap_min", 0.0)),
            step=1000.0,
            key="fs_mcap_min",
            help="Filter by company size. 500 Cr = small-cap, 5000 Cr = mid-cap, 20000 Cr = large-cap.",
        )
        div_yield_min = st.number_input(
            "Min Dividend Yield %",
            min_value=0.0, max_value=20.0,
            value=float(active_preset.get("div_yield_min", 0.0)),
            step=0.5,
            key="fs_div_yield_min",
            help="Annual dividend as % of share price. Useful for income-seeking investors.",
        )

    ch1, ch2 = st.columns(2)
    with ch1:
        only_profitable = st.checkbox(
            "Only profitable companies",
            value=True,
            key="fs_only_profitable",
            help="Exclude companies with negative ROE (i.e. those losing money).",
        )
    with ch2:
        show_technical = st.checkbox(
            "Also show technical setup (chart patterns)",
            value=True,
            key="fs_show_technical",
            help="If enabled, runs a quick chart pattern scan for stocks that pass the fundamental filters.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scan button ───────────────────────────────────────────────────────────
    scan_clicked = st.button(
        "🔍 Find Stocks",
        type="primary",
        key="fs_scan",
        use_container_width=False,
    )

    if not scan_clicked:
        st.markdown(
            f"<div style='color:{_MUTED};font-size:.85rem;padding:.8rem 0'>"
            f"Set your filters above, then click <strong>Find Stocks</strong> to screen the universe.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Fetch fundamentals ────────────────────────────────────────────────────
    progress = st.progress(0, text="Fetching fundamentals…")
    results: List[Dict[str, Any]] = []

    for idx, symbol in enumerate(universe):
        progress.progress(
            (idx + 1) / len(universe),
            text=f"Analysing {symbol} ({idx+1}/{len(universe)})…",
        )
        f = _fetch_fundamentals(symbol)
        f["symbol"] = symbol
        passes, quality_label = _passes_filters(
            f,
            pe_max=pe_max,
            roe_min=roe_min,
            de_max=de_max,
            promoter_min=promoter_min,
            mcap_min=mcap_min,
            div_yield_min=div_yield_min,
            only_profitable=only_profitable,
        )
        if passes:
            f["_quality_label"] = quality_label
            f["_score"]         = _quality_score(f)
            results.append(f)

    progress.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    if not results:
        st.markdown(
            f"<div style='background:rgba(255,75,75,0.08);border:1px solid rgba(255,75,75,0.2);"
            f"border-radius:12px;padding:1rem 1.2rem;color:{_WHITE};font-size:.88rem'>"
            f"No stocks matched all your filters. Try relaxing one or two criteria — "
            f"e.g. increase Max PE or reduce Min ROE.</div>",
            unsafe_allow_html=True,
        )
        return

    # Sort by quality score descending
    results.sort(key=lambda x: x.get("_score", 0), reverse=True)

    st.markdown(
        f"<p style='color:{_GREEN};font-size:.85rem;font-weight:600;margin-bottom:.6rem'>"
        f"✅ Found {len(results)} stock{'s' if len(results) != 1 else ''} matching your filters</p>",
        unsafe_allow_html=True,
    )

    # ── Optional technical setup fetch ────────────────────────────────────────
    tech_setups: Dict[str, str] = {}
    if show_technical:
        matching_symbols = [r["symbol"] for r in results]
        try:
            from screener.momentum_scanner import MomentumScanner
            scanner = MomentumScanner()
            scanned = scanner.scan_breakouts(matching_symbols, top_n=len(matching_symbols))
            for s in scanned:
                sym = getattr(s, "symbol", None)
                pat = getattr(s, "pattern", None) or getattr(s, "breakout_type", None)
                sig = getattr(s, "signal", None)
                if sym:
                    parts = []
                    if pat:
                        parts.append(str(pat).replace("_", " ").title())
                    if sig:
                        parts.append(str(sig))
                    tech_setups[sym] = " · ".join(parts) if parts else "No pattern"
        except Exception:
            pass   # technical data is optional

    # ── Summary table ─────────────────────────────────────────────────────────
    import pandas as pd

    table_rows = []
    for r in results:
        label  = r.get("_quality_label", "pass")
        score  = r.get("_score", 0)
        sym    = r.get("symbol", "")
        tech   = tech_setups.get(sym, "—") if show_technical else "—"

        table_rows.append({
            "Symbol":       sym,
            "Price (₹)":    _na(r.get("current_price"), ".0f"),
            "PE":           _na(r.get("pe_ratio"), ".1f"),
            "ROE %":        _na(r.get("roe"), ".1f"),
            "Debt/Eq":      _na(r.get("debt_to_equity"), ".2f"),
            "Promoter %":   _na(r.get("promoter_holding"), ".1f"),
            "Mkt Cap (Cr)": _na(r.get("market_cap_cr"), ",.0f"),
            "Quality Score": f"{score:.0f}",
            "Technical":    tech,
            "_label":       label,
        })

    df = pd.DataFrame(table_rows)

    def _row_color(row):
        lbl = row.get("_label", "pass")
        if lbl == "pass":
            color = "rgba(0,212,160,0.07)"
        else:
            color = "rgba(245,158,11,0.07)"
        return [f"background-color:{color}"] * len(row)

    display_df = df.drop(columns=["_label"])

    st.dataframe(
        display_df.style.apply(_row_color, axis=1),
        hide_index=True,
        use_container_width=True,
        key="fs_results_table",
    )

    st.divider()

    # ── Per-stock expandable cards ────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{_MUTED};font-size:.8rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem'>"
        f"Detailed View — click a company to expand</p>",
        unsafe_allow_html=True,
    )

    for r in results:
        sym   = r.get("symbol", "")
        score = r.get("_score", 0)
        label = r.get("_quality_label", "pass")
        tech  = tech_setups.get(sym) if show_technical else None

        badge_color = _GREEN if label == "pass" else _AMBER
        badge_text  = "Strong match" if label == "pass" else "Borderline"

        header_html = (
            f"<span style='font-weight:700;color:{_WHITE}'>{sym}</span>"
            f" &nbsp;"
            f"<span style='background:{badge_color}22;border:1px solid {badge_color}55;"
            f"border-radius:5px;padding:1px 7px;font-size:.72rem;color:{badge_color}'>"
            f"{badge_text}</span>"
            f" &nbsp;"
            f"<span style='color:{_MUTED};font-size:.78rem'>Score: {score:.0f}/100</span>"
        )

        with st.expander(f"{sym} — {badge_text} (Score: {score:.0f})", expanded=False):
            st.markdown(
                f"<div style='font-size:.78rem;color:{_MUTED};margin-bottom:.6rem'>"
                f"Source: {r.get('_source', 'N/A')}</div>",
                unsafe_allow_html=True,
            )
            _plain_english_card(sym, r, tech)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{_MUTED};font-size:.72rem;margin-top:1rem'>"
        f"Fundamental data sourced from Screener.in and Yahoo Finance. "
        f"Quality scores are a simple heuristic — not investment advice. "
        f"Always do your own research before investing.</p>",
        unsafe_allow_html=True,
    )
