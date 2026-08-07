"""
Memory Vault — Streamlit page to view, search, and manage Co-Pilot memories.
Rendered as a sub-tab inside the 🤖 Agents top-level tab.
"""

from __future__ import annotations

import streamlit as st
from ai.mem0_store import get_memory


def render_memory_vault() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;margin:0'>🧠 Memory Vault</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.2rem 0 1rem'>"
        "Persistent Co-Pilot memories — survive browser refresh · SQLite-backed</p>",
        unsafe_allow_html=True,
    )

    mem = get_memory()
    total = mem.count()

    # ── Stats strip ──────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Memories", total)
    s2.metric("Categories", len({m["category"] for m in mem.get_all()}) if total else 0)
    s3.metric("Storage", "SQLite local")

    st.divider()

    # ── Manual add ───────────────────────────────────────────────────────────
    with st.expander("➕ Add a memory manually"):
        mc1, mc2 = st.columns([4, 1])
        new_mem_text = mc1.text_area(
            "Memory content",
            placeholder="e.g. I prefer high-volatility momentum stocks over value plays",
            key="vault_new_mem",
            label_visibility="collapsed",
        )
        new_mem_cat = mc2.selectbox(
            "Category",
            ["preference", "insight", "watchlist", "rule", "general"],
            key="vault_new_cat",
        )
        if st.button("💾 Save memory", key="vault_save_new"):
            if new_mem_text.strip():
                mem.add(new_mem_text.strip(), category=new_mem_cat)
                st.success("Memory saved!")
                st.rerun()

    # ── Search ───────────────────────────────────────────────────────────────
    search_q = st.text_input(
        "Search memories",
        placeholder="e.g. RELIANCE, momentum, risk limit…",
        key="vault_search",
    )

    # ── Display ──────────────────────────────────────────────────────────────
    if search_q.strip():
        memories = mem.search(search_q.strip(), limit=20)
        st.caption(f"Found {len(memories)} result(s) for '{search_q}'")
    else:
        memories = mem.get_all()

    if not memories:
        st.info(
            "No memories stored yet. Chat with the Co-Pilot — exchanges are "
            "auto-saved, or click **💾 Remember** after any AI response."
        )
        return

    # Category colour map
    _cat_colors = {
        "preference": "#00d4a0",
        "insight": "#00d4ff",
        "watchlist": "#ffb800",
        "rule": "#ff4b4b",
        "exchange": "#8892a4",
        "general": "#c8cfe0",
    }

    for m in memories:
        cat = m.get("category", "general")
        color = _cat_colors.get(cat, "#c8cfe0")
        col_content, col_del = st.columns([9, 1])
        with col_content:
            st.markdown(
                f"<div style='background:rgba(255,255,255,.03);border-left:3px solid {color};"
                f"border-radius:0 6px 6px 0;padding:.5rem .8rem;margin:.2rem 0'>"
                f"<span style='color:{color};font-size:.65rem;text-transform:uppercase;"
                f"font-weight:700;letter-spacing:.06em'>{cat}</span> "
                f"<span style='color:#8892a4;font-size:.65rem'>{m['created_at'][:16]}</span><br>"
                f"<span style='color:#e8eaf0;font-size:.83rem;line-height:1.6'>{m['content']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_del:
            st.write("")
            if st.button("🗑", key=f"vault_del_{m['id']}", help="Delete this memory"):
                mem.delete(m["id"])
                st.rerun()

    st.divider()
    if st.button("🗑 Clear ALL memories", key="vault_clear_all", type="secondary"):
        mem.clear_all()
        st.success("All memories cleared.")
        st.rerun()
