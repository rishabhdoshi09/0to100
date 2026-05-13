"""
Regime Bar — renders a compact persistent strip at the top of every page.
Shows: Market Regime | Regime Score | Nifty | VIX | Breadth | Sector Leader
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def _get_regime_cached() -> dict:
    """Cache regime for 15 min — regime doesn't change minute to minute."""
    try:
        from analytics.regime_engine import compute_regime
        r = compute_regime()
        return {
            "regime": r.regime,
            "regime_score": r.regime_score,
            "emoji": r.emoji,
            "nifty_price": r.nifty_price,
            "nifty_change_pct": r.nifty_change_pct,
            "vix": r.vix,
            "vix_state": r.vix_state,
            "breadth": r.breadth,
            "sector_leader": r.sector_leader,
            "quality_multiplier": r.quality_multiplier,
            "timestamp": r.timestamp,
            "atr_regime": r.atr_regime,
        }
    except Exception:
        return {
            "regime": "UNKNOWN", "regime_score": 50.0, "emoji": "⚪",
            "nifty_price": 0.0, "nifty_change_pct": 0.0,
            "vix": 16.0, "vix_state": "NORMAL",
            "breadth": "NEUTRAL", "sector_leader": "N/A",
            "quality_multiplier": 1.0, "timestamp": "--",
            "atr_regime": "STABLE",
        }


def get_regime() -> dict:
    """Returns current regime dict (cached 15 min)."""
    return _get_regime_cached()


def render_regime_bar() -> None:
    """
    Renders a compact horizontal regime strip.
    Call this at the top of any page that should show market context.
    """
    r = get_regime()

    regime_colors = {
        "BULL_TREND":   ("#00d4a0", "#002a1f"),
        "EXPANSION":    ("#00d4ff", "#002030"),
        "CHOPPY":       ("#f59e0b", "#2a1f00"),
        "DISTRIBUTION": ("#fb923c", "#2a1000"),
        "BEAR":         ("#ff4b4b", "#2a0000"),
        "UNKNOWN":      ("#8892a4", "#111827"),
    }
    fg, bg = regime_colors.get(r["regime"], ("#8892a4", "#111827"))

    nifty_chg   = r["nifty_change_pct"]
    chg_color   = "#00d4a0" if nifty_chg >= 0 else "#ff4b4b"
    chg_arrow   = "▲" if nifty_chg >= 0 else "▼"
    breadth_col = {"STRONG": "#00d4a0", "NEUTRAL": "#f59e0b", "WEAK": "#ff4b4b"}.get(r["breadth"], "#8892a4")
    vix_col     = {"LOW": "#00d4a0", "NORMAL": "#f59e0b", "HIGH": "#ff4b4b"}.get(r["vix_state"], "#8892a4")

    score_bar_w = int(r["regime_score"])

    st.markdown(
        f"""
        <div style='background:{bg};border:1px solid {fg}44;border-radius:10px;
                    padding:8px 16px;margin-bottom:12px;display:flex;
                    align-items:center;gap:20px;flex-wrap:wrap;font-family:JetBrains Mono,monospace'>

          <!-- Regime badge -->
          <div style='display:flex;flex-direction:column;gap:2px;min-width:120px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Market Regime
            </span>
            <span style='font-size:.85rem;color:{fg};font-weight:800;letter-spacing:.05em'>
              {r["emoji"]} {r["regime"].replace("_", " ")}
            </span>
          </div>

          <!-- Regime score bar -->
          <div style='display:flex;flex-direction:column;gap:3px;min-width:100px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Regime Score
            </span>
            <div style='height:5px;background:#1e293b;border-radius:3px;width:100px'>
              <div style='height:5px;width:{score_bar_w}px;background:{fg};border-radius:3px'></div>
            </div>
            <span style='font-size:.65rem;color:{fg}'>{r["regime_score"]:.0f}/100</span>
          </div>

          <!-- Nifty -->
          <div style='display:flex;flex-direction:column;gap:2px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Nifty 50
            </span>
            <span style='font-size:.82rem;color:#e8eaf0;font-weight:700'>
              {r["nifty_price"]:,.0f}
              <span style='font-size:.7rem;color:{chg_color}'> {chg_arrow}{abs(nifty_chg):.2f}%</span>
            </span>
          </div>

          <!-- VIX -->
          <div style='display:flex;flex-direction:column;gap:2px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              India VIX
            </span>
            <span style='font-size:.82rem;color:{vix_col};font-weight:700'>
              {r["vix"]:.1f}
              <span style='font-size:.65rem;color:{vix_col};opacity:.8'> {r["vix_state"]}</span>
            </span>
          </div>

          <!-- Breadth -->
          <div style='display:flex;flex-direction:column;gap:2px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Breadth
            </span>
            <span style='font-size:.82rem;color:{breadth_col};font-weight:700'>
              {r["breadth"]}
            </span>
          </div>

          <!-- Sector Leader -->
          <div style='display:flex;flex-direction:column;gap:2px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Sector Leader
            </span>
            <span style='font-size:.82rem;color:#00d4ff;font-weight:700'>
              {r["sector_leader"]}
            </span>
          </div>

          <!-- ATR / Volatility -->
          <div style='display:flex;flex-direction:column;gap:2px'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              ATR Regime
            </span>
            <span style='font-size:.75rem;color:#8892a4;font-weight:600'>
              {r["atr_regime"]}
            </span>
          </div>

          <!-- Quality multiplier -->
          <div style='display:flex;flex-direction:column;gap:2px;margin-left:auto'>
            <span style='font-size:.58rem;color:#8892a4;letter-spacing:.1em;text-transform:uppercase'>
              Setup Qualifier
            </span>
            <span style='font-size:.75rem;color:{fg};font-weight:700'>
              ×{r["quality_multiplier"]:.2f}
            </span>
          </div>

          <!-- Timestamp -->
          <div style='font-size:.6rem;color:#4a5568'>
            as of {r["timestamp"]}
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )
