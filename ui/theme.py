"""QuantTerm design system — one refined dark theme + small render helpers.

`DEVBLOOM_CSS` is injected once in app.py and styles every page. The palette and helper
functions below are the single source of truth for the look, so pages stay consistent
instead of hand-rolling inline styles. Kept intentionally restrained — an institutional
desk tool, not a neon arcade.
"""
from __future__ import annotations

import streamlit as st

# ── palette (single source of truth) ────────────────────────────────────────────
ACCENT = "#38bdf8"   # confident, restrained cyan — used sparingly
TEAL   = "#2dd4bf"
GREEN  = "#34d399"   # profit / healthy
RED    = "#fb7185"   # loss / down
AMBER  = "#fbbf24"   # caution
TEXT   = "#e6e9ef"   # primary text
MUTED  = "#8b94a7"   # secondary text
DIM    = "#525c6e"   # tertiary / eyebrow
BASE   = "#0a0e17"   # app background
PANEL  = "rgba(255,255,255,0.025)"
BORDER = "rgba(255,255,255,0.07)"

# backwards-compat aliases (older pages import these names)
CYAN, WHITE, NAVY, CARD = ACCENT, TEXT, BASE, PANEL

DEVBLOOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
:root{
  --qt-accent:#38bdf8; --qt-teal:#2dd4bf; --qt-green:#34d399; --qt-red:#fb7185;
  --qt-amber:#fbbf24; --qt-text:#e6e9ef; --qt-muted:#8b94a7; --qt-dim:#525c6e;
  --qt-base:#0a0e17; --qt-panel:rgba(255,255,255,0.025); --qt-border:rgba(255,255,255,0.07);
  --qt-radius:14px;
}

/* ─── Base ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .stApp {
  background:
     radial-gradient(1200px 600px at 15% -10%, rgba(56,189,248,0.06), transparent 60%),
     radial-gradient(1000px 500px at 100% 0%, rgba(45,212,191,0.05), transparent 55%),
     #0a0e17 !important;
  color: var(--qt-text) !important;
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
[data-testid="stMainBlockContainer"]{ padding-top: 2.4rem !important; max-width: 1500px; }
[data-testid="stHeader"]{ background: transparent !important; }

/* tighter, quieter default headings */
h1,h2,h3,h4{ letter-spacing:-0.01em !important; font-weight:750 !important; color:var(--qt-text)!important; }
h1{ font-size:1.7rem !important; } h2{ font-size:1.32rem !important; } h3{ font-size:1.08rem !important; }
a, a:visited { color: var(--qt-accent) !important; text-decoration: none; }
hr, [data-testid="stDivider"]{ border-color: var(--qt-border) !important; opacity:.7; }

/* ─── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.012)) !important;
  border-right: 1px solid var(--qt-border) !important;
  backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] * { color: var(--qt-text); }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:.5rem; }

/* ─── Design-system utility classes ───────────────────────── */
.qt-eyebrow{ font-size:.62rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--qt-dim); font-weight:700; font-family:'JetBrains Mono',monospace; }
.qt-section{ margin:1.4rem 0 .7rem; }
.qt-section .t{ font-size:1.06rem; font-weight:750; color:var(--qt-text); }
.qt-section .s{ font-size:.8rem; color:var(--qt-muted); margin-top:.1rem; }

.qt-card{ background:var(--qt-panel); border:1px solid var(--qt-border);
  border-radius:var(--qt-radius); padding:1.05rem 1.2rem; }
.qt-card.tight{ padding:.8rem .95rem; }
.qt-grid{ display:grid; gap:.7rem; }

.qt-stat{ background:var(--qt-panel); border:1px solid var(--qt-border);
  border-radius:var(--qt-radius); padding:.85rem 1rem; }
.qt-stat .k{ font-size:.64rem; letter-spacing:.11em; text-transform:uppercase;
  color:var(--qt-muted); font-weight:700; }
.qt-stat .v{ font-size:1.5rem; font-weight:760; margin-top:.15rem;
  font-family:'JetBrains Mono',monospace; letter-spacing:-.02em; }
.qt-stat .d{ font-size:.72rem; color:var(--qt-muted); margin-top:.1rem; }

.qt-pill{ display:inline-flex; align-items:center; gap:.35rem; font-size:.7rem;
  font-weight:650; padding:.2rem .6rem; border-radius:999px;
  border:1px solid var(--qt-border); background:rgba(255,255,255,0.03); color:var(--qt-muted); }
.qt-dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }
.qt-dot.ok{ background:var(--qt-green); box-shadow:0 0 8px rgba(52,211,153,.7); }
.qt-dot.warn{ background:var(--qt-amber); box-shadow:0 0 8px rgba(251,191,36,.6); }
.qt-dot.bad{ background:var(--qt-red); box-shadow:0 0 8px rgba(251,113,133,.6); }
.qt-dot.off{ background:var(--qt-dim); }

/* reasoning / activity stream (the "what's happening inside" feed) */
.qt-think{ border-left:2px solid var(--qt-border); padding:.15rem 0 .15rem .9rem;
  margin:.15rem 0; }
.qt-think .kind{ font-size:.6rem; letter-spacing:.1em; text-transform:uppercase;
  font-weight:700; font-family:'JetBrains Mono',monospace; color:var(--qt-dim); }
.qt-think .txt{ font-size:.86rem; color:var(--qt-text); line-height:1.35; }
.qt-think.OBSERVE{ border-color:#3b82f6; } .qt-think.REASON{ border-color:var(--qt-teal); }
.qt-think.DECIDE{ border-color:var(--qt-amber); } .qt-think.PROPOSE{ border-color:var(--qt-accent); }
.qt-think.CONCLUDE{ border-color:var(--qt-green); }

/* ─── Metric chips ────────────────────────────────────────── */
[data-testid="stMetric"]{
  background:var(--qt-panel) !important; border:1px solid var(--qt-border) !important;
  border-radius:var(--qt-radius) !important; padding:.8rem 1rem !important; }
[data-testid="stMetricLabel"]{ color:var(--qt-muted) !important; font-size:.66rem !important;
  text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
[data-testid="stMetricValue"]{ color:var(--qt-text) !important;
  font-family:'JetBrains Mono',monospace !important; font-weight:720 !important; }

/* ─── Buttons — quiet default, clear primary ──────────────── */
.stButton>button{
  background:rgba(255,255,255,0.04) !important; color:var(--qt-text) !important;
  border:1px solid var(--qt-border) !important; border-radius:10px !important;
  font-weight:600 !important; transition:all .16s ease !important; }
.stButton>button:hover{ border-color:rgba(56,189,248,0.5) !important;
  background:rgba(56,189,248,0.08) !important; color:#bfe9ff !important; }
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#38bdf8,#2dd4bf) !important; color:#04121b !important;
  border:none !important; font-weight:750 !important; }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.07); box-shadow:0 6px 20px rgba(56,189,248,.28) !important; }

/* ─── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  background:var(--qt-panel) !important; border:1px solid var(--qt-border) !important;
  border-radius:12px !important; padding:4px !important; gap:2px !important; }
[data-testid="stTabs"] [data-baseweb="tab"]{ color:var(--qt-muted) !important;
  border-radius:9px !important; font-size:.82rem !important; font-weight:600 !important;
  padding:.4rem .9rem !important; }
[data-testid="stTabs"] [aria-selected="true"]{
  background:rgba(56,189,248,0.12) !important; color:var(--qt-accent) !important; }

/* ─── Inputs ──────────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"]>div{
  background:rgba(255,255,255,0.04) !important; border:1px solid var(--qt-border) !important;
  border-radius:9px !important; color:var(--qt-text) !important; }

/* ─── Cards: expander, dataframe, alerts ──────────────────── */
[data-testid="stExpander"]{ background:var(--qt-panel) !important;
  border:1px solid var(--qt-border) !important; border-radius:12px !important; }
[data-testid="stExpander"] summary{ color:var(--qt-muted) !important; font-weight:600; }
[data-testid="stDataFrame"]{ border-radius:12px; overflow:hidden; border:1px solid var(--qt-border); }
[data-testid="stDataFrame"] th{ background:#111827 !important; color:var(--qt-muted) !important;
  font-size:.68rem !important; text-transform:uppercase; letter-spacing:.05em; }
[data-testid="stDataFrame"] td{ background:rgba(255,255,255,0.015) !important; color:var(--qt-text) !important; }
[data-testid="stAlert"]{ border-radius:12px !important; border:1px solid var(--qt-border) !important; }
[data-testid="stCaptionContainer"], .stCaption{ color:var(--qt-muted) !important; }

/* ─── Verdict badges (kept for existing pages) ────────────── */
.db-buy,.buy,.recommendation.buy{ background:linear-gradient(135deg,rgba(52,211,153,.16),rgba(52,211,153,.05));
  color:var(--qt-green); border:1px solid rgba(52,211,153,.35); border-radius:10px;
  font-weight:720; text-align:center; padding:.55rem 1rem; }
.db-sell,.sell,.recommendation.sell{ background:linear-gradient(135deg,rgba(251,113,133,.16),rgba(251,113,133,.05));
  color:var(--qt-red); border:1px solid rgba(251,113,133,.35); border-radius:10px;
  font-weight:720; text-align:center; padding:.55rem 1rem; }
.db-hold,.hold,.recommendation.hold{ background:linear-gradient(135deg,rgba(251,191,36,.14),rgba(251,191,36,.04));
  color:var(--qt-amber); border:1px solid rgba(251,191,36,.3); border-radius:10px;
  font-weight:720; text-align:center; padding:.55rem 1rem; }
.recommendation{ font-size:1.1rem; font-weight:720; text-align:center; padding:.6rem; border-radius:10px; margin-top:.25rem; }
.devbloom-card{ background:var(--qt-panel); border:1px solid var(--qt-border);
  border-radius:var(--qt-radius); padding:1.1rem 1.3rem; margin-bottom:1rem; }

/* command palette overlay (unchanged behaviour) */
#devbloom-palette-overlay{ display:none; position:fixed; inset:0; z-index:9999;
  background:rgba(0,0,0,0.6); backdrop-filter:blur(6px); align-items:flex-start;
  justify-content:center; padding-top:12vh; }
#devbloom-palette-overlay.open{ display:flex; }
#devbloom-palette-box{ background:rgba(12,17,29,0.97); border:1px solid rgba(56,189,248,0.35);
  border-radius:16px; width:min(640px,90vw); box-shadow:0 24px 80px rgba(0,0,0,0.7); overflow:hidden; }
#devbloom-palette-input{ width:100%; padding:1rem 1.25rem; background:transparent; border:none;
  border-bottom:1px solid var(--qt-border); color:var(--qt-text); font-size:1rem;
  font-family:'JetBrains Mono',monospace; outline:none; }
#devbloom-palette-results{ max-height:320px; overflow-y:auto; }
.palette-item{ padding:.65rem 1.25rem; cursor:pointer; font-family:'JetBrains Mono',monospace;
  font-size:.85rem; color:var(--qt-muted); border-bottom:1px solid rgba(255,255,255,0.04);
  display:flex; align-items:center; gap:.75rem; }
.palette-item:hover,.palette-item.active{ background:rgba(56,189,248,0.08); color:var(--qt-accent); }
.palette-item .badge{ font-size:.65rem; padding:.15rem .4rem; border-radius:4px;
  background:rgba(56,189,248,0.12); color:var(--qt-accent); flex-shrink:0; }

/* ─── Sidebar brand + Reco desk nav ───────────────────────── */
[data-testid="stSidebar"] > div:first-child::before{
  content:"QUANTTERM  ·  NSE DESK";
  display:block; font-family:'JetBrains Mono',monospace;
  font-size:.62rem; letter-spacing:.18em; color:var(--qt-dim);
  padding:1.05rem 1.05rem .15rem; font-weight:700;
}
[data-testid="stSidebarNav"] ul{ padding-top:.35rem; }
[data-testid="stSidebarNav"] li a{
  border-radius:10px !important; padding:.45rem .7rem !important;
  font-weight:600 !important; letter-spacing:.01em;
}
[data-testid="stSidebarNav"] li a[aria-current="page"]{
  background:rgba(56,189,248,0.12) !important; color:var(--qt-accent) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"]{ display:none; }

/* ─── Reco / Moneycontrol setup cards ─────────────────────── */
.reco-strip{
  background:rgba(13,20,33,.92); border:1px solid var(--qt-border);
  border-radius:12px; padding:.85rem 1.1rem; margin-bottom:.9rem;
}
.reco-card{
  background:rgba(13,20,33,.92); border:1px solid #1e293b;
  border-radius:12px; padding:1.05rem 1.2rem; margin-bottom:.75rem;
}
.reco-card .row{ display:flex; justify-content:space-between; align-items:center; }
.reco-card .sym{ font-family:'JetBrains Mono',monospace; font-weight:700;
  font-size:1.02rem; color:var(--qt-text); }
.reco-card .co{ font-size:.72rem; color:var(--qt-muted); margin-top:.12rem; }
.reco-card .px{ font-size:.9rem; font-weight:650; color:var(--qt-text); margin:.35rem 0 .2rem; }
.reco-card .lv{ font-size:.75rem; color:var(--qt-muted); font-family:'JetBrains Mono',monospace; }
.reco-card .why{ font-size:.78rem; color:#c9d1d9; margin-top:.4rem; line-height:1.35; }
.reco-badge{ font-size:.8rem; font-weight:750; }
.reco-badge.buy{ color:var(--qt-green); }
.reco-badge.watch{ color:var(--qt-amber); }
.reco-badge.wait{ color:#fb923c; }
.reco-badge.avoid{ color:var(--qt-red); }
.reco-how{
  background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.18);
  border-radius:12px; padding:1rem 1.15rem; margin:1rem 0 1.2rem;
}
.reco-how ol{ margin:.35rem 0 0 1.1rem; padding:0; color:var(--qt-text); }
.reco-how li{ margin:.22rem 0; font-size:.88rem; line-height:1.4; }
.reco-how .k{ color:var(--qt-accent); font-weight:700; }

/* scrollbars */
::-webkit-scrollbar{ width:7px; height:7px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:rgba(255,255,255,0.12); border-radius:4px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(255,255,255,0.2); }
</style>
"""


# ── render helpers (keep pages consistent, no inline-style sprawl) ───────────────

def section(title: str, sub: str = "", eyebrow: str = "") -> None:
    """A consistent section header: optional eyebrow label + title + subtitle."""
    eb = f"<div class='qt-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    sb = f"<div class='s'>{sub}</div>" if sub else ""
    st.markdown(f"<div class='qt-section'>{eb}<div class='t'>{title}</div>{sb}</div>",
                unsafe_allow_html=True)


def stat_grid(items: list[dict], cols: int = 4) -> None:
    """Render a responsive row of stat cards. Each item: {k, v, d?, color?}."""
    cells = []
    for it in items:
        color = it.get("color", "var(--qt-text)")
        d = f"<div class='d'>{it['d']}</div>" if it.get("d") else ""
        cells.append(f"<div class='qt-stat'><div class='k'>{it['k']}</div>"
                     f"<div class='v' style='color:{color}'>{it['v']}</div>{d}</div>")
    st.markdown(
        f"<div class='qt-grid' style='grid-template-columns:repeat({cols},1fr)'>"
        + "".join(cells) + "</div>", unsafe_allow_html=True)


def pill(text: str, status: str = "off") -> str:
    """Return a status pill html string. status ∈ ok|warn|bad|off."""
    return (f"<span class='qt-pill'><span class='qt-dot {status}'></span>{text}</span>")


def pill_row(pills: list[str]) -> None:
    st.markdown("<div style='display:flex;gap:.45rem;flex-wrap:wrap;margin:.2rem 0'>"
                + "".join(pills) + "</div>", unsafe_allow_html=True)


COMMAND_PALETTE_JS = """
<div id="devbloom-palette-overlay">
  <div id="devbloom-palette-box">
    <input id="devbloom-palette-input" placeholder="⌘ Type a command or symbol…" autocomplete="off" />
    <div id="devbloom-palette-results"></div>
  </div>
</div>

<script>
(function(){
  const COMMANDS = [
    {cmd:"/home",        label:"Go to Command Center",            badge:"NAV"},
    {cmd:"/charts",      label:"Open Technical Analysis Suite",   badge:"NAV"},
    {cmd:"/fundamentals",label:"Open Fundamental Deep Dive",      badge:"NAV"},
    {cmd:"/copilot",     label:"Open AI Co-Pilot Dev",            badge:"AI"},
    {cmd:"/execution",   label:"Open Execution & Risk Cockpit",   badge:"NAV"},
    {cmd:"/algolab",     label:"Open AlgoLab (Code Cave)",        badge:"NAV"},
    {cmd:"/journal",     label:"Open Journaling & Analytics",     badge:"NAV"},
    {cmd:"/screener",    label:"Open Stock Screener",             badge:"SCAN"},
    {cmd:"/backtest",    label:"Run Backtest Bridge",             badge:"RUN"},
    {cmd:"/why",         label:"/why [symbol] — Ask AI why it's moving", badge:"AI"},
    {cmd:"/model",       label:"/model [symbol] — Load DCF model",badge:"AI"},
    {cmd:"/anomaly",     label:"Scan for Anomalies (z>3)",        badge:"SCAN"},
  ];

  const overlay = document.getElementById('devbloom-palette-overlay');
  const input   = document.getElementById('devbloom-palette-input');
  const results = document.getElementById('devbloom-palette-results');
  let activeIdx = 0;

  function open(){  overlay.classList.add('open');  input.value=''; render(''); input.focus(); }
  function close(){ overlay.classList.remove('open'); }

  function render(q){
    const hits = q ? COMMANDS.filter(c=>c.cmd.includes(q.toLowerCase())||c.label.toLowerCase().includes(q.toLowerCase())) : COMMANDS;
    activeIdx = 0;
    results.innerHTML = hits.map((c,i)=>
      `<div class="palette-item${i===0?' active':''}" data-cmd="${c.cmd}">
         <span class="badge">${c.badge}</span>${c.cmd} — ${c.label}
       </div>`
    ).join('');
    results.querySelectorAll('.palette-item').forEach((el,i)=>{
      el.addEventListener('click',()=>{ dispatch(hits[i].cmd); close(); });
    });
  }

  function dispatch(cmd){
    const url = new URL(window.parent.location.href);
    url.searchParams.set('palette_cmd', cmd);
    window.parent.history.replaceState(null,'', url.toString());
    const btn = window.parent.document.querySelector('[data-testid="baseButton-secondary"][aria-label="palette-trigger"]');
    if(btn) btn.click();
  }

  input.addEventListener('input', ()=> render(input.value));
  input.addEventListener('keydown', e=>{
    const items = results.querySelectorAll('.palette-item');
    if(e.key==='ArrowDown'){ e.preventDefault(); items[activeIdx]?.classList.remove('active'); activeIdx=Math.min(activeIdx+1,items.length-1); items[activeIdx]?.classList.add('active'); }
    if(e.key==='ArrowUp'){   e.preventDefault(); items[activeIdx]?.classList.remove('active'); activeIdx=Math.max(activeIdx-1,0); items[activeIdx]?.classList.add('active'); }
    if(e.key==='Enter'){ const c=results.querySelectorAll('.palette-item')[activeIdx]; if(c) c.click(); close(); }
    if(e.key==='Escape') close();
  });
  overlay.addEventListener('click', e=>{ if(e.target===overlay) close(); });

  window.addEventListener('keydown', e=>{
    if((e.metaKey||e.ctrlKey) && e.key==='k'){ e.preventDefault(); open(); return; }
    if(e.key===' ' && !e.metaKey && !e.ctrlKey && !e.altKey){
      const tag = document.activeElement?.tagName?.toLowerCase();
      const editable = document.activeElement?.isContentEditable;
      if(tag !== 'input' && tag !== 'textarea' && tag !== 'select' && !editable){
        e.preventDefault();
        const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] [data-baseweb="tab"]');
        if(tabs[3]) tabs[3].click();
      }
    }
  }, true);
})();
</script>
"""
