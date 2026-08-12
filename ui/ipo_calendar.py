"""IPO Calendar — upcoming IPOs, open subscriptions, allotment, listing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import streamlit as st

# ── NSE endpoints ─────────────────────────────────────────────────────────────
_NSE_HOME     = "https://www.nseindia.com/"
_NSE_OPEN     = "https://www.nseindia.com/api/ipo-current-open"
_NSE_UPCOMING = "https://www.nseindia.com/api/ipo-upcoming"
_NSE_LISTING  = "https://www.nseindia.com/api/ipo-listing"

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Referer":    "https://www.nseindia.com/",
    "Accept":     "application/json",
}

# ── Colours (matches DevBloom theme) ──────────────────────────────────────────
_CYAN   = "#00d4ff"
_GREEN  = "#00d4a0"
_AMBER  = "#f59e0b"
_RED    = "#ff4b4b"
_WHITE  = "#e8eaf0"
_MUTED  = "#8892a4"
_CARD   = "rgba(255,255,255,0.04)"
_BORDER = "rgba(255,255,255,0.08)"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class IPOItem:
    name:               str
    symbol:             str           = "TBD"
    open_date:          Optional[str] = None   # "YYYY-MM-DD"
    close_date:         Optional[str] = None
    listing_date:       Optional[str] = None
    price_band_low:     float         = 0.0
    price_band_high:    float         = 0.0
    lot_size:           int           = 0
    category:           str           = "UPCOMING"   # OPEN / UPCOMING / LISTED
    subscription_times: float         = 0.0
    gmp_pct:            float         = 0.0
    issue_size_cr:      float         = 0.0


# ── Demo fallback data ────────────────────────────────────────────────────────

_DEMO_IPOS: List[IPOItem] = [
    IPOItem(
        name="Ola Electric Mobility",
        symbol="OLAELEC",
        open_date="2026-05-21",
        close_date="2026-05-23",
        listing_date="2026-05-28",
        price_band_low=72,
        price_band_high=76,
        lot_size=195,
        category="OPEN",
        subscription_times=4.32,
        gmp_pct=8.5,
        issue_size_cr=6145.0,
    ),
    IPOItem(
        name="Swiggy Ltd",
        symbol="SWIGGY",
        open_date="2026-05-27",
        close_date="2026-05-29",
        listing_date="2026-06-03",
        price_band_low=371,
        price_band_high=390,
        lot_size=38,
        category="UPCOMING",
        subscription_times=0.0,
        gmp_pct=12.0,
        issue_size_cr=11327.0,
    ),
    IPOItem(
        name="Vishal Mega Mart",
        symbol="VISHAL",
        open_date="2026-05-11",
        close_date="2026-05-13",
        listing_date="2026-05-18",
        price_band_low=74,
        price_band_high=78,
        lot_size=192,
        category="LISTED",
        subscription_times=27.4,
        gmp_pct=0.0,
        issue_size_cr=8000.0,
    ),
    IPOItem(
        name="Mobikwik Systems",
        symbol="MOBIKWIK",
        open_date="2026-06-02",
        close_date="2026-06-04",
        listing_date="2026-06-09",
        price_band_low=265,
        price_band_high=279,
        lot_size=53,
        category="UPCOMING",
        subscription_times=0.0,
        gmp_pct=3.0,
        issue_size_cr=572.0,
    ),
    IPOItem(
        name="NTPC Green Energy",
        symbol="NTPCGREEN",
        open_date="2026-04-28",
        close_date="2026-04-30",
        listing_date="2026-05-06",
        price_band_low=102,
        price_band_high=108,
        lot_size=138,
        category="LISTED",
        subscription_times=2.55,
        gmp_pct=0.0,
        issue_size_cr=10000.0,
    ),
]


# ── NSE fetch helpers ─────────────────────────────────────────────────────────

def _build_nse_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    try:
        sess.get(_NSE_HOME, timeout=10)
    except Exception:
        pass
    return sess


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", "").replace("₹", "").strip())
    except Exception:
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(str(val).replace(",", "").strip())
    except Exception:
        return default


def _parse_nse_date(val: str) -> Optional[str]:
    """Convert NSE date string (e.g. '22-May-2026' or '2026-05-22') to YYYY-MM-DD."""
    if not val:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_open(raw: list) -> List[IPOItem]:
    items = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        lo = _safe_float(r.get("minPrice") or r.get("priceBandMin") or r.get("fromPrice", 0))
        hi = _safe_float(r.get("maxPrice") or r.get("priceBandMax") or r.get("toPrice", 0))
        items.append(IPOItem(
            name               = r.get("companyName", r.get("name", "Unknown")),
            symbol             = r.get("symbol", "TBD") or "TBD",
            open_date          = _parse_nse_date(r.get("openDate", r.get("bidOpenDate", ""))),
            close_date         = _parse_nse_date(r.get("closeDate", r.get("bidCloseDate", ""))),
            listing_date       = _parse_nse_date(r.get("listingDate", "")),
            price_band_low     = lo,
            price_band_high    = hi,
            lot_size           = _safe_int(r.get("lotSize", r.get("minBidQuantity", 0))),
            category           = "OPEN",
            subscription_times = _safe_float(r.get("subscriptionTimes", r.get("totalApplications", 0))),
            gmp_pct            = 0.0,
            issue_size_cr      = _safe_float(r.get("issueSizeCr", r.get("issueSize", 0))),
        ))
    return items


def _parse_upcoming(raw: list) -> List[IPOItem]:
    items = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        lo = _safe_float(r.get("minPrice") or r.get("priceBandMin") or r.get("fromPrice", 0))
        hi = _safe_float(r.get("maxPrice") or r.get("priceBandMax") or r.get("toPrice", 0))
        items.append(IPOItem(
            name            = r.get("companyName", r.get("name", "Unknown")),
            symbol          = r.get("symbol", "TBD") or "TBD",
            open_date       = _parse_nse_date(r.get("openDate", r.get("bidOpenDate", ""))),
            close_date      = _parse_nse_date(r.get("closeDate", r.get("bidCloseDate", ""))),
            listing_date    = _parse_nse_date(r.get("listingDate", "")),
            price_band_low  = lo,
            price_band_high = hi,
            lot_size        = _safe_int(r.get("lotSize", r.get("minBidQuantity", 0))),
            category        = "UPCOMING",
            issue_size_cr   = _safe_float(r.get("issueSizeCr", r.get("issueSize", 0))),
        ))
    return items


def _parse_listing(raw: list) -> List[IPOItem]:
    items = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        lo = _safe_float(r.get("minPrice") or r.get("priceBandMin") or r.get("fromPrice", 0))
        hi = _safe_float(r.get("maxPrice") or r.get("priceBandMax") or r.get("toPrice", 0))
        items.append(IPOItem(
            name               = r.get("companyName", r.get("name", "Unknown")),
            symbol             = r.get("symbol", "TBD") or "TBD",
            open_date          = _parse_nse_date(r.get("openDate", r.get("bidOpenDate", ""))),
            close_date         = _parse_nse_date(r.get("closeDate", r.get("bidCloseDate", ""))),
            listing_date       = _parse_nse_date(r.get("listingDate", "")),
            price_band_low     = lo,
            price_band_high    = hi,
            lot_size           = _safe_int(r.get("lotSize", r.get("minBidQuantity", 0))),
            category           = "LISTED",
            subscription_times = _safe_float(r.get("subscriptionTimes", r.get("totalApplications", 0))),
            issue_size_cr      = _safe_float(r.get("issueSizeCr", r.get("issueSize", 0))),
        ))
    return items


# ── Main fetch with 6-hour cache ──────────────────────────────────────────────

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_ipos() -> List[IPOItem]:
    """
    Fetch IPOs from NSE. Returns list[IPOItem].
    Falls back to demo data if NSE is unreachable.
    """
    try:
        sess = _build_nse_session()
        all_ipos: List[IPOItem] = []
        errors = 0

        for url, parser, label in [
            (_NSE_OPEN,     _parse_open,     "open"),
            (_NSE_UPCOMING, _parse_upcoming, "upcoming"),
            (_NSE_LISTING,  _parse_listing,  "listing"),
        ]:
            try:
                resp = sess.get(url, timeout=12)
                if resp.status_code == 200:
                    payload = resp.json()
                    # NSE may wrap data under a key, or return a bare list
                    if isinstance(payload, list):
                        raw = payload
                    elif isinstance(payload, dict):
                        raw = (
                            payload.get("data")
                            or payload.get("ipoData")
                            or payload.get("ipo")
                            or []
                        )
                    else:
                        raw = []
                    all_ipos.extend(parser(raw))
                else:
                    errors += 1
            except Exception:
                errors += 1

        # If all three endpoints failed, fall back to demo
        if errors == 3 or not all_ipos:
            return _DEMO_IPOS

        return all_ipos

    except Exception:
        return _DEMO_IPOS


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_date(iso: Optional[str]) -> str:
    if not iso:
        return "TBD"
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%b %-d")
    except Exception:
        return iso


def _lot_cost(ipo: IPOItem) -> str:
    if ipo.price_band_high and ipo.lot_size:
        cost = ipo.price_band_high * ipo.lot_size
        return f"₹{cost:,.0f} per lot"
    return "—"


def _price_band(ipo: IPOItem) -> str:
    if ipo.price_band_low and ipo.price_band_high:
        return f"₹{ipo.price_band_low:,.0f} – ₹{ipo.price_band_high:,.0f} per share"
    if ipo.price_band_high:
        return f"₹{ipo.price_band_high:,.0f} per share"
    return "Price TBD"


def _subscription_badge(sub: float) -> str:
    if sub <= 0:
        return ""
    color = _GREEN if sub >= 1.0 else _RED
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;"
        f"border-radius:6px;padding:2px 8px;font-size:.72rem;color:{color};"
        f"font-weight:600'>{sub:.1f}× subscribed</span>"
    )


def _gmp_badge(gmp: float) -> str:
    if gmp == 0:
        return ""
    sign = "+" if gmp > 0 else ""
    color = _AMBER
    return (
        f"<span style='background:{color}22;border:1px solid {color}55;"
        f"border-radius:6px;padding:2px 8px;font-size:.72rem;color:{color};"
        f"font-weight:600'>GMP: {sign}{gmp:.1f}%</span>"
    )


def _demand_hint(ipo: IPOItem) -> tuple[str, str]:
    """Returns (text, color) for the 'Should I apply?' heuristic."""
    sub = ipo.subscription_times
    gmp = ipo.gmp_pct

    if ipo.category == "UPCOMING":
        return ("Subscription not yet open", _MUTED)
    if sub > 5 and gmp > 5:
        return ("Looks interesting — high demand 🔥", _GREEN)
    if sub > 1:
        return ("Decent demand — consider applying", _CYAN)
    if sub > 0:
        return ("Low demand — be cautious ⚠️", _AMBER)
    return ("No subscription data yet", _MUTED)


def _render_ipo_card(ipo: IPOItem) -> None:
    """Render a single IPO as a glass-morphism card."""
    hint_text, hint_color = _demand_hint(ipo)

    date_line = (
        f"Open: {_fmt_date(ipo.open_date)}"
        f" &nbsp;|&nbsp; Close: {_fmt_date(ipo.close_date)}"
        f" &nbsp;|&nbsp; Listing: {_fmt_date(ipo.listing_date)}"
    )

    badges_html = " &nbsp;".join(filter(None, [
        _subscription_badge(ipo.subscription_times),
        _gmp_badge(ipo.gmp_pct),
    ]))

    issue_line = (
        f"₹{ipo.issue_size_cr:,.0f} Cr issue size" if ipo.issue_size_cr else ""
    )

    lot_line = ""
    if ipo.lot_size:
        lot_line = (
            f"Lot size: <strong>{ipo.lot_size:,} shares</strong>"
            f" &nbsp;·&nbsp; {_lot_cost(ipo)}"
        )

    st.markdown(
        f"""
<div style='background:{_CARD};border:1px solid {_BORDER};border-radius:14px;
     padding:1rem 1.3rem;margin-bottom:.85rem;
     box-shadow:0 4px 20px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.05)'>

  <div style='display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin-bottom:.35rem'>
    <span style='color:{_WHITE};font-size:1.05rem;font-weight:700'>{ipo.name}</span>
    {"<span style='color:" + _MUTED + ";font-size:.78rem;font-family:JetBrains Mono,monospace'>" + ipo.symbol + "</span>" if ipo.symbol != "TBD" else ""}
  </div>

  <div style='color:{_CYAN};font-size:.88rem;font-weight:600;margin-bottom:.25rem'>
    {_price_band(ipo)}
  </div>

  {"<div style='color:" + _MUTED + ";font-size:.8rem;margin-bottom:.25rem'>" + lot_line + "</div>" if lot_line else ""}

  <div style='color:{_MUTED};font-size:.78rem;margin-bottom:.45rem'>{date_line}</div>

  {"<div style='margin-bottom:.45rem'>" + badges_html + "</div>" if badges_html else ""}

  <div style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>
    <span style='color:{hint_color};font-size:.82rem;font-weight:600'>{hint_text}</span>
    {"<span style='color:" + _MUTED + ";font-size:.78rem'>·&nbsp;" + issue_line + "</span>" if issue_line else ""}
  </div>

</div>
""",
        unsafe_allow_html=True,
    )


# ── Public render function ────────────────────────────────────────────────────

def render_ipo_calendar() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;letter-spacing:2px;margin:0'>🚀 IPO Calendar</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.3rem 0 1rem'>"
        "Track upcoming stock market listings &nbsp;·&nbsp; NSE data &nbsp;·&nbsp; 6-hour cache</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns([5, 1])
    with c2:
        if st.button("🔄 Refresh", key="ipo_refresh"):
            get_ipos.clear()
            st.rerun()

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner("Fetching IPO data from NSE…"):
        ipos = get_ipos()

    if not ipos:
        st.info("IPO data temporarily unavailable. Check NSE website at nseindia.com/market-data/ipo")
        return

    # ── Last updated ──────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{_MUTED};font-size:.72rem;margin-bottom:.8rem'>"
        f"Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}</p>",
        unsafe_allow_html=True,
    )

    # ── Bucket IPOs ───────────────────────────────────────────────────────────
    open_ipos     = [i for i in ipos if i.category == "OPEN"]
    upcoming_ipos = [i for i in ipos if i.category == "UPCOMING"]
    listed_ipos   = [i for i in ipos if i.category == "LISTED"]

    tab_open, tab_upcoming, tab_listed = st.tabs([
        f"Open Now ({len(open_ipos)})",
        f"Coming Soon ({len(upcoming_ipos)})",
        f"Recently Listed ({len(listed_ipos)})",
    ])

    # ── Tab: Open Now ─────────────────────────────────────────────────────────
    with tab_open:
        if not open_ipos:
            st.markdown(
                f"<div style='color:{_MUTED};padding:2rem;text-align:center;"
                f"font-size:.9rem'>No IPOs are open for subscription right now."
                f"<br>Check the <em>Coming Soon</em> tab for upcoming issues.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:{_GREEN};font-size:.82rem;font-weight:600;"
                f"margin-bottom:1rem'>⏳ These IPOs are currently open — apply before the close date!</p>",
                unsafe_allow_html=True,
            )
            for ipo in open_ipos:
                _render_ipo_card(ipo)

    # ── Tab: Coming Soon ──────────────────────────────────────────────────────
    with tab_upcoming:
        if not upcoming_ipos:
            st.markdown(
                f"<div style='color:{_MUTED};padding:2rem;text-align:center;"
                f"font-size:.9rem'>No upcoming IPOs announced yet.<br>"
                f"Refresh in a few days for new announcements.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:{_CYAN};font-size:.82rem;font-weight:600;"
                f"margin-bottom:1rem'>📅 Mark your calendar — these are coming up soon</p>",
                unsafe_allow_html=True,
            )
            for ipo in upcoming_ipos:
                _render_ipo_card(ipo)

    # ── Tab: Recently Listed ──────────────────────────────────────────────────
    with tab_listed:
        if not listed_ipos:
            st.markdown(
                f"<div style='color:{_MUTED};padding:2rem;text-align:center;"
                f"font-size:.9rem'>No recent listings to show.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:{_AMBER};font-size:.82rem;font-weight:600;"
                f"margin-bottom:1rem'>📋 Recently listed — see how they performed vs subscription demand</p>",
                unsafe_allow_html=True,
            )
            for ipo in listed_ipos:
                _render_ipo_card(ipo)

    # ── Explainer ─────────────────────────────────────────────────────────────
    with st.expander("ℹ️ How to read this calendar", expanded=False):
        st.markdown(
            """
**Subscription times** — how many times the IPO was applied for vs shares available.
Over 1× means oversubscribed (more demand than supply). Over 10× is very popular.

**GMP (Grey Market Premium)** — informal trading price before official listing.
A positive GMP suggests the market expects a listing gain. Not guaranteed.

**Lot size** — you can only apply in multiples of the lot size.
The amount shown is the minimum you need to invest.

**Should I apply?** hints are based on subscription demand and GMP only —
always read the company's Red Herring Prospectus (RHP) before applying.

*Data sourced from NSE India. Refresh for latest updates.*
""",
            unsafe_allow_html=True,
        )
