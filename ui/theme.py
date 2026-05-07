"""DevBloom Terminal — dark glass-morphism theme constants and CSS injection."""
from __future__ import annotations

# Neon accent palette
CYAN   = "#00d4ff"   # buy-side
AMBER  = "#ffb800"   # sell-side
GREEN  = "#00ff88"   # profit
RED    = "#ff4466"   # loss
WHITE  = "#e8eaf0"   # neutral text
NAVY   = "#0a0e1a"   # base background
CARD   = "rgba(255,255,255,0.04)"
BORDER = "rgba(255,255,255,0.08)"

DEVBLOOM_CSS = """
<style>
/* ─── Base ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: #0a0e1a !important;
  color: #e8eaf0 !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}

/* ─── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.03) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

/* ─── Cards / glass panels ───────────────────────────────── */
.devbloom-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ─── Metric chips ───────────────────────────────────────── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="metric-container"] label { color: #8892a4 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8eaf0 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ─── Verdict badges ─────────────────────────────────────── */
.db-buy  { background: linear-gradient(135deg,rgba(0,212,255,.15),rgba(0,212,255,.05));
           color:#00d4ff; border:1px solid rgba(0,212,255,.35); border-radius:10px;
           font-size:1.4rem; font-weight:700; text-align:center; padding:.6rem 1rem; }
.db-sell { background: linear-gradient(135deg,rgba(255,68,102,.15),rgba(255,68,102,.05));
           color:#ff4466; border:1px solid rgba(255,68,102,.35); border-radius:10px;
           font-size:1.4rem; font-weight:700; text-align:center; padding:.6rem 1rem; }
.db-hold { background: linear-gradient(135deg,rgba(255,184,0,.12),rgba(255,184,0,.04));
           color:#ffb800; border:1px solid rgba(255,184,0,.3); border-radius:10px;
           font-size:1.4rem; font-weight:700; text-align:center; padding:.6rem 1rem; }

/* ─── Tabs ───────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
  padding: 4px !important;
  gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  color: #8892a4 !important;
  border-radius: 10px !important;
  font-size: .82rem !important;
  font-weight: 500 !important;
  padding: .4rem .9rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: rgba(0,212,255,0.12) !important;
  color: #00d4ff !important;
  border: 1px solid rgba(0,212,255,0.25) !important;
}

/* ─── Buttons ────────────────────────────────────────────── */
.stButton button {
  background: rgba(0,212,255,0.1) !important;
  color: #00d4ff !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all .2s !important;
}
.stButton button:hover {
  background: rgba(0,212,255,0.2) !important;
  box-shadow: 0 0 16px rgba(0,212,255,0.25) !important;
  transform: translateY(-1px) !important;
}

/* ─── Inputs / selects ───────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 8px !important;
  color: #e8eaf0 !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ─── Dataframe ──────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] th { background:#111827 !important; color:#8892a4 !important; font-size:.72rem !important; text-transform:uppercase; }
[data-testid="stDataFrame"] td { background:rgba(255,255,255,0.02) !important; color:#e8eaf0 !important; }

/* ─── Expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #8892a4 !important; }

/* ─── Command palette overlay ────────────────────────────── */
#devbloom-palette-overlay {
  display: none;
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(6px);
  align-items: flex-start;
  justify-content: center;
  padding-top: 12vh;
}
#devbloom-palette-overlay.open { display: flex; }
#devbloom-palette-box {
  background: rgba(15,20,40,0.96);
  border: 1px solid rgba(0,212,255,0.35);
  border-radius: 16px;
  width: min(640px, 90vw);
  box-shadow: 0 24px 80px rgba(0,0,0,0.7), 0 0 0 1px rgba(0,212,255,0.1);
  overflow: hidden;
}
#devbloom-palette-input {
  width: 100%; padding: 1rem 1.25rem;
  background: transparent; border: none;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  color: #e8eaf0; font-size: 1rem;
  font-family: 'JetBrains Mono', monospace;
  outline: none;
}
#devbloom-palette-results { max-height: 320px; overflow-y: auto; }
.palette-item {
  padding: .65rem 1.25rem; cursor: pointer;
  font-family: 'JetBrains Mono', monospace; font-size: .85rem;
  color: #8892a4; border-bottom: 1px solid rgba(255,255,255,0.04);
  display: flex; align-items: center; gap: .75rem;
}
.palette-item:hover, .palette-item.active { background: rgba(0,212,255,0.08); color: #00d4ff; }
.palette-item .badge {
  font-size: .65rem; padding: .15rem .4rem; border-radius: 4px;
  background: rgba(0,212,255,0.12); color: #00d4ff; flex-shrink: 0;
}

/* ─── Anomaly ticker tape ────────────────────────────────── */
.db-ticker {
  background: rgba(255,184,0,0.08);
  border: 1px solid rgba(255,184,0,0.2);
  border-radius: 8px; padding: .4rem .8rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .78rem; color: #ffb800;
  display: inline-block; margin: .2rem;
}

/* ─── Verdict badge aliases (backwards compat with existing app code) ─────── */
.recommendation { font-size: 1.1rem; font-weight: 700; text-align: center;
                  padding: .6rem; border-radius: 10px; margin-top: .25rem; }
.buy, .recommendation.buy {
  background: linear-gradient(135deg,rgba(0,212,255,.15),rgba(0,212,255,.05));
  color:#00d4ff; border:1px solid rgba(0,212,255,.35); border-radius:10px; }
.sell, .recommendation.sell {
  background: linear-gradient(135deg,rgba(255,68,102,.15),rgba(255,68,102,.05));
  color:#ff4466; border:1px solid rgba(255,68,102,.35); border-radius:10px; }
.hold, .recommendation.hold {
  background: linear-gradient(135deg,rgba(255,184,0,.12),rgba(255,184,0,.04));
  color:#ffb800; border:1px solid rgba(255,184,0,.3); border-radius:10px; }

/* ─── Scrollbars ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
</style>
"""

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
    // Write to Streamlit via query param so Python can pick it up on rerun
    const url = new URL(window.parent.location.href);
    url.searchParams.set('palette_cmd', cmd);
    window.parent.history.replaceState(null,'', url.toString());
    // Trigger a Streamlit rerun by clicking the hidden dummy button
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

  // Global shortcuts: Cmd/Ctrl+K → command palette
  //                    Space (when not in a text input) → jump to Co-Pilot tab
  window.addEventListener('keydown', e=>{
    if((e.metaKey||e.ctrlKey) && e.key==='k'){ e.preventDefault(); open(); return; }

    // Space → Co-Pilot: only when focus is NOT in an input / textarea / contenteditable
    if(e.key===' ' && !e.metaKey && !e.ctrlKey && !e.altKey){
      const tag = document.activeElement?.tagName?.toLowerCase();
      const editable = document.activeElement?.isContentEditable;
      if(tag !== 'input' && tag !== 'textarea' && tag !== 'select' && !editable){
        e.preventDefault();
        // Click the "⚡ Dev" tab (tab index 3 in the Streamlit tab bar)
        const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] [data-baseweb="tab"]');
        if(tabs[3]) tabs[3].click();
      }
    }
  }, true);
})();
</script>
"""
