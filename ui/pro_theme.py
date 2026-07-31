"""QuantTerm professional visual system for the Streamlit product shell."""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


_CSS = r"""
<style>
:root {
  --qt-bg: #060a12;
  --qt-bg-elevated: #0a101c;
  --qt-panel: #0d1523;
  --qt-panel-2: #101b2c;
  --qt-border: rgba(137, 170, 210, 0.16);
  --qt-border-strong: rgba(65, 214, 255, 0.34);
  --qt-text: #eef5ff;
  --qt-muted: #8593aa;
  --qt-cyan: #36d9ff;
  --qt-cyan-soft: rgba(54, 217, 255, 0.12);
  --qt-green: #55e68a;
  --qt-amber: #f5bd55;
  --qt-red: #ff6978;
  --qt-purple: #9a86ff;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(circle at 78% -10%, rgba(54,217,255,.07), transparent 28rem),
    linear-gradient(180deg, #060a12 0%, #070c15 100%) !important;
  color: var(--qt-text);
}

[data-testid="stHeader"] { background: rgba(6,10,18,.78); }
[data-testid="stToolbar"] { right: 1.2rem; }
[data-testid="stMainBlockContainer"] {
  max-width: 1540px;
  padding-top: 1.15rem;
  padding-bottom: 4rem;
  padding-left: 1.55rem;
  padding-right: 1.55rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080d17 0%, #070b13 100%);
  border-right: 1px solid var(--qt-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: .6rem; }
[data-testid="stSidebarNav"] { padding-top: .45rem; }
[data-testid="stSidebarNav"] span { font-size: .92rem; }
[data-testid="stSidebarNav"] li > div {
  border-radius: 10px;
  margin: 2px 8px;
  min-height: 42px;
}
[data-testid="stSidebarNav"] li > div:hover { background: rgba(54,217,255,.07); }
[data-testid="stSidebarNav"] li [aria-current="page"] {
  color: var(--qt-cyan) !important;
  background: linear-gradient(90deg, rgba(54,217,255,.14), rgba(54,217,255,.04)) !important;
  box-shadow: inset 2px 0 0 var(--qt-cyan);
}

.qt-brand {
  margin: .25rem .65rem 1rem;
  padding: .8rem .7rem 1rem;
  border-bottom: 1px solid var(--qt-border);
}
.qt-brand-row { display:flex; gap:.72rem; align-items:center; }
.qt-logo {
  width: 34px; height: 34px; display:grid; place-items:center;
  border: 1px solid var(--qt-border-strong); border-radius: 10px;
  background: linear-gradient(145deg, rgba(54,217,255,.18), rgba(154,134,255,.12));
  color: var(--qt-cyan); font-weight: 800; letter-spacing:-.08em;
}
.qt-brand-name { color:var(--qt-text); font-weight:750; letter-spacing:.06em; font-size:.98rem; }
.qt-brand-sub { color:var(--qt-muted); font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; margin-top:.15rem; }
.qt-brand-mode {
  display:inline-flex; align-items:center; gap:.4rem; margin-top:.72rem;
  color:var(--qt-green); font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
}
.qt-brand-dot { width:7px; height:7px; border-radius:999px; background:var(--qt-green); box-shadow:0 0 10px rgba(85,230,138,.45); }

.qt-page-head {
  display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
  padding:.45rem 0 1rem; border-bottom:1px solid var(--qt-border); margin-bottom:1rem;
}
.qt-eyebrow { color:var(--qt-cyan); text-transform:uppercase; letter-spacing:.13em; font-size:.68rem; font-weight:700; }
.qt-title { font-size:1.55rem; line-height:1.15; font-weight:720; color:var(--qt-text); margin:.22rem 0 .25rem; letter-spacing:-.02em; }
.qt-subtitle { color:var(--qt-muted); font-size:.86rem; max-width:760px; }
.qt-badges { display:flex; flex-wrap:wrap; gap:.45rem; justify-content:flex-end; }
.qt-badge {
  display:inline-flex; align-items:center; gap:.38rem; padding:.34rem .55rem;
  border:1px solid var(--qt-border); border-radius:999px; background:rgba(13,21,35,.8);
  color:#b7c4d8; font-size:.7rem; white-space:nowrap;
}
.qt-badge-dot { width:6px; height:6px; border-radius:999px; background:var(--qt-cyan); }
.qt-badge.good .qt-badge-dot { background:var(--qt-green); }
.qt-badge.warn .qt-badge-dot { background:var(--qt-amber); }
.qt-badge.bad .qt-badge-dot { background:var(--qt-red); }

.qt-metric {
  min-height:116px; padding:.95rem 1rem; border:1px solid var(--qt-border);
  border-radius:13px; background:linear-gradient(180deg, rgba(16,27,44,.92), rgba(10,17,29,.92));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
.qt-metric-label { color:var(--qt-muted); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; }
.qt-metric-value { color:var(--qt-text); font-size:1.46rem; font-weight:720; margin:.4rem 0 .25rem; letter-spacing:-.025em; }
.qt-metric-detail { color:#9eabc0; font-size:.72rem; line-height:1.35; }
.qt-metric.good { border-color:rgba(85,230,138,.22); }
.qt-metric.good .qt-metric-value { color:var(--qt-green); }
.qt-metric.warn { border-color:rgba(245,189,85,.23); }
.qt-metric.warn .qt-metric-value { color:var(--qt-amber); }
.qt-metric.bad { border-color:rgba(255,105,120,.24); }
.qt-metric.bad .qt-metric-value { color:var(--qt-red); }
.qt-metric.accent { border-color:var(--qt-border-strong); }
.qt-metric.accent .qt-metric-value { color:var(--qt-cyan); }

.qt-section-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin:1.25rem 0 .65rem; }
.qt-section-title { color:var(--qt-text); font-size:.92rem; font-weight:700; letter-spacing:.02em; text-transform:uppercase; }
.qt-section-sub { color:var(--qt-muted); font-size:.72rem; margin-top:.18rem; }

.qt-panel {
  border:1px solid var(--qt-border); border-radius:13px;
  background:linear-gradient(180deg, rgba(13,21,35,.95), rgba(9,15,25,.95));
  padding:1rem;
}
.qt-kicker { color:var(--qt-cyan); font-size:.69rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
.qt-panel-title { color:var(--qt-text); font-size:1rem; font-weight:700; margin:.3rem 0 .45rem; }
.qt-panel-copy { color:var(--qt-muted); font-size:.78rem; line-height:1.48; }
.qt-divider { border-top:1px solid var(--qt-border); margin:.9rem 0; }
.qt-insight { display:flex; gap:.6rem; color:#b9c7d9; font-size:.78rem; line-height:1.45; margin:.55rem 0; }
.qt-insight:before { content:'◆'; color:var(--qt-cyan); font-size:.55rem; margin-top:.28rem; }
.qt-plan-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin-top:.75rem; }
.qt-plan-item { padding:.7rem; border:1px solid var(--qt-border); border-radius:10px; background:rgba(7,12,21,.55); }
.qt-plan-label { color:var(--qt-muted); font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; }
.qt-plan-value { color:var(--qt-text); font-weight:680; margin-top:.22rem; }

div[data-testid="stMetric"] {
  border:1px solid var(--qt-border); border-radius:12px; padding:.75rem .85rem;
  background:linear-gradient(180deg, rgba(16,27,44,.9), rgba(10,17,29,.9));
}
div[data-testid="stMetricLabel"] p { color:var(--qt-muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
div[data-testid="stMetricValue"] { color:var(--qt-text); }

.stButton > button, .stLinkButton > a {
  border-radius:9px !important; border:1px solid rgba(54,217,255,.28) !important;
  background:linear-gradient(180deg, rgba(23,48,67,.95), rgba(12,30,44,.95)) !important;
  color:#dff8ff !important; min-height:40px; font-weight:650;
}
.stButton > button:hover, .stLinkButton > a:hover {
  border-color:var(--qt-cyan) !important; color:white !important;
}
button[kind="primary"] {
  background:linear-gradient(135deg, #16bde3, #358bff) !important;
  border-color:transparent !important; color:#031019 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--qt-border); }
[data-testid="stTabs"] button { border-radius:8px 8px 0 0; color:var(--qt-muted); }
[data-testid="stTabs"] button[aria-selected="true"] { color:var(--qt-cyan); background:var(--qt-cyan-soft); }

[data-testid="stDataFrame"] {
  border:1px solid var(--qt-border); border-radius:12px; overflow:hidden;
  background:rgba(8,14,24,.82);
}
[data-testid="stExpander"] { border:1px solid var(--qt-border) !important; border-radius:11px !important; background:rgba(9,15,25,.7); }
[data-testid="stAlert"] { border-radius:10px; }
hr { border-color:var(--qt-border) !important; }

@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding-left:.8rem; padding-right:.8rem; }
  .qt-page-head { align-items:flex-start; flex-direction:column; }
  .qt-badges { justify-content:flex-start; }
  .qt-plan-grid { grid-template-columns:1fr; }
}
</style>
"""


def apply_pro_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="qt-brand">
          <div class="qt-brand-row">
            <div class="qt-logo">Q</div>
            <div>
              <div class="qt-brand-name">QUANTTERM</div>
              <div class="qt-brand-sub">Professional Retail Quant</div>
            </div>
          </div>
          <div class="qt-brand-mode"><span class="qt-brand-dot"></span> Evidence-first mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str = "QuantTerm Workspace",
    badges: Iterable[tuple[str, str]] = (),
) -> None:
    badge_html = "".join(
        f'<span class="qt-badge {html.escape(tone)}"><span class="qt-badge-dot"></span>{html.escape(label)}</span>'
        for label, tone in badges
    )
    st.markdown(
        f"""
        <div class="qt-page-head">
          <div>
            <div class="qt-eyebrow">{html.escape(eyebrow)}</div>
            <div class="qt-title">{html.escape(title)}</div>
            <div class="qt-subtitle">{html.escape(subtitle)}</div>
          </div>
          <div class="qt-badges">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "", *, tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="qt-metric {html.escape(tone)}">
          <div class="qt-metric-label">{html.escape(label)}</div>
          <div class="qt-metric-value">{html.escape(value)}</div>
          <div class="qt-metric-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="qt-section-head">
          <div>
            <div class="qt-section-title">{html.escape(title)}</div>
            <div class="qt-section-sub">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_panel(title: str, insights: Iterable[str], *, kicker: str = "System Intelligence") -> None:
    rows = "".join(f'<div class="qt-insight">{html.escape(str(item))}</div>' for item in insights)
    st.markdown(
        f"""
        <div class="qt-panel">
          <div class="qt-kicker">{html.escape(kicker)}</div>
          <div class="qt-panel-title">{html.escape(title)}</div>
          {rows or '<div class="qt-panel-copy">No current insight is available.</div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_panel(
    title: str,
    subtitle: str,
    *,
    reasons: Iterable[str] = (),
    risks: Iterable[str] = (),
    plan: Iterable[tuple[str, str]] = (),
) -> None:
    reason_html = "".join(f'<div class="qt-insight">{html.escape(str(item))}</div>' for item in reasons)
    risk_html = "".join(
        f'<div class="qt-insight" style="--qt-cyan: var(--qt-amber)">{html.escape(str(item))}</div>'
        for item in risks
    )
    plan_html = "".join(
        f'<div class="qt-plan-item"><div class="qt-plan-label">{html.escape(label)}</div>'
        f'<div class="qt-plan-value">{html.escape(value)}</div></div>'
        for label, value in plan
    )
    st.markdown(
        f"""
        <div class="qt-panel">
          <div class="qt-kicker">Evidence Pack</div>
          <div class="qt-panel-title">{html.escape(title)}</div>
          <div class="qt-panel-copy">{html.escape(subtitle)}</div>
          <div class="qt-divider"></div>
          {reason_html or '<div class="qt-panel-copy">No positive evidence recorded.</div>'}
          {('<div class="qt-divider"></div>' + risk_html) if risk_html else ''}
          {('<div class="qt-plan-grid">' + plan_html + '</div>') if plan_html else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
