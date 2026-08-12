"""
🗂️ Historical Data Setup — the layman-friendly Streamlit page.

Thin renderer over `research.momentum_breakout.data_setup` (all logic + validation live
there, pure and tested). It lets a local user hand QuantTerm real NSE history, checks it,
saves it into the existing canonical stores, shows research readiness, and runs the
UNCHANGED frozen EXP-006 test only when the gate allows.

It contains NO order actions of any kind (paper, live, GTT, Telegram, broker) and cannot
change EXP-006 logic. Historical research cannot place broker orders.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from research.momentum_breakout import data_setup as D


def _readiness_banner(r: dict) -> None:
    colour = {"green": "#00d4a0", "amber": "#f59e0b", "red": "#ff4b4b"}[r["color"]]
    st.markdown(
        f"<div style='background:{colour}18;border:1px solid {colour}55;"
        f"border-left:5px solid {colour};border-radius:12px;padding:.9rem 1.1rem;"
        f"margin:.5rem 0'><div style='font-size:1.15rem;font-weight:800;color:{colour}'>"
        f"{r['label']}</div></div>", unsafe_allow_html=True)
    for reason in r["reasons"]:
        st.markdown(f"- {reason}")


def _status_panel(v) -> None:
    def yn(b): return "✅ found" if b else "❌ missing"
    st.markdown("#### What we found")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"- Daily price data: {yn(v.price_data)}")
        st.markdown(f"- Nifty benchmark: {yn(v.benchmark)}")
        st.markdown(f"- Corporate-action file: {yn(v.corporate_actions)}")
        st.markdown(f"- Listing/delisting history: {yn(v.universe_history)}")
        st.markdown(f"- Delivery data: {yn(v.delivery_data)}")
    with c2:
        st.markdown(f"- First market date: **{v.first_date or '—'}**")
        st.markdown(f"- Last market date: **{v.last_date or '—'}**")
        st.markdown(f"- Files: **{v.file_count}** · Stocks: **{v.symbol_count}**")
        st.markdown(f"- Price rows: **{v.row_count:,}**")
        st.markdown(f"- Prices adjusted: **{v.adjustment_status}**")
    if v.blockers:
        st.markdown("#### What needs fixing")
        for b in v.blockers:
            st.markdown(f"- ⚠️ {b}")
    with st.expander("Technical details"):
        st.json(v.as_dict())


def render_data_setup() -> None:
    st.markdown("## 🗂️ Historical Data Setup")
    st.caption("Give QuantTerm real NSE history so it can test ideas honestly. "
               "This page only manages data — **historical research cannot place broker "
               "orders.**")
    st.info("What is this? A place to load past market data, check it, save it, and — "
            "only when the data is good enough — run the frozen research test (EXP-006). "
            "It never trades.")

    stg_key = "_hds_staging"
    method = st.radio("How do you want to provide the data?",
                      ["Upload files", "Use an existing folder on this computer"],
                      key="hds_method")

    # ── 1. input ──
    if method == "Upload files":
        st.markdown("Upload a **ZIP package** *or* individual files: daily price files "
                    "(`.csv`), Nifty index files (`.csv`), `ca_events.json`, "
                    "`universe_history.json`. Tables in `.md` are read too; `.pdf` is "
                    "best-effort and rejected if it can't be validated (export to CSV).")
        ups = st.file_uploader("Choose file(s)", type=["zip", "csv", "json", "md", "pdf"],
                               accept_multiple_files=True, key="hds_files")
        if ups and st.button("Check these files", key="hds_check_files"):
            dest = Path(tempfile.mkdtemp(prefix="hds_"))
            extracted, rejected = [], []
            loose = []
            for f in ups:
                if f.name.lower().endswith(".zip"):
                    r = D.safe_extract_zip(f, dest)
                    extracted += r.extracted; rejected += r.rejected
                else:
                    loose.append((f.name, f.getvalue()))
            if loose:
                r = D.ingest_files(loose, dest)
                extracted += r.extracted; rejected += r.rejected
            if not extracted:
                st.error("Could not read a usable dataset from those files.")
                for n, why in rejected:
                    st.markdown(f"- `{n}` — {why}")
            else:
                st.session_state[stg_key] = str(dest)
                st.success(f"Read {len(extracted)} data part(s).")
                if rejected:
                    with st.expander(f"{len(rejected)} file(s) skipped"):
                        for n, why in rejected:
                            st.markdown(f"- `{n}` — {why}")
    else:
        folder = st.text_input("Full path to your data folder", key="hds_folder")
        if folder and st.button("Check this folder", key="hds_check_folder"):
            if Path(folder).exists():
                st.session_state[stg_key] = folder
            else:
                st.error("That folder does not exist on this computer.")

    staging = st.session_state.get(stg_key)
    if not staging:
        st.stop()

    # ── 2–3. validate + preview ──
    v = D.validate_dataset(staging)
    r = D.readiness(v)
    _readiness_banner(r)
    _status_panel(v)

    # ── 4–5. save + snapshot (overwrite protection) ──
    st.markdown("#### Save this data")
    if r["color"] == "red":
        st.warning("This data can't be used yet — fix the items above first.")
    else:
        mode = st.radio("If a dataset already exists:",
                        ["Create a new dataset (don't overwrite)",
                         "Replace the existing dataset", "Cancel"],
                        key="hds_savemode")
        mode_map = {"Create a new dataset (don't overwrite)": "new",
                    "Replace the existing dataset": "replace", "Cancel": "cancel"}
        if st.button("💾 Save data", key="hds_save"):
            try:
                res = D.save_into_canonical(staging, mode=mode_map[mode])
                if res["status"] == "cancelled":
                    st.info("Cancelled — nothing was changed.")
                else:
                    D.materialize()
                    snap = D.dataset_snapshot(staging, v)
                    st.session_state["_hds_snapshot"] = snap
                    st.success(f"Saved. Exact data version (snapshot): "
                               f"`{snap['snapshot_id']}`")
            except D.OverwriteRefused as e:
                st.error(str(e))

    snap = st.session_state.get("_hds_snapshot")
    if snap:
        with st.expander("Technical details — dataset snapshot"):
            st.json(snap)

    # ── 6–7. readiness + run ──
    st.markdown("#### Run the research test")
    st.caption("Historical research cannot place broker orders. This only studies the past.")
    if not r["can_run"]:
        st.button("Run EXP-006 Historical Test", disabled=True,
                  help="Disabled: the data readiness check is red. " +
                       (r["reasons"][0] if r["reasons"] else ""))
    elif not snap:
        st.button("Run EXP-006 Historical Test", disabled=True,
                  help="Save the data first.")
    else:
        if st.button("▶️ Run EXP-006 Historical Test", key="hds_run", type="primary"):
            with st.spinner("Running the frozen research test on your data…"):
                try:
                    out = D.run_exp006(r)
                    st.session_state["_hds_result"] = out
                except Exception as e:
                    st.error(f"Could not run the test: {e}")

    # ── 8. results ──
    out = st.session_state.get("_hds_result")
    if out:
        _render_result(out)


def _render_result(out: dict) -> None:
    from core import simple_language as SL   # optional plain-language verdict text
    verdict = out["verdict"]["verdict"]
    st.markdown("### Result")
    colour = {"PASS": "#00d4a0", "FAIL": "#ff4b4b", "INCONCLUSIVE": "#f59e0b"}.get(
        verdict, "#38bdf8")
    st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:{colour}'>"
                f"{verdict}</div>", unsafe_allow_html=True)
    try:
        st.write(SL.verdict_meaning(verdict)["plain"])
    except Exception:
        pass
    st.markdown("- Unit tests are **not** market evidence.")
    st.markdown("- Synthetic (made-up) data is **not** market evidence.")
    st.markdown("- **INCONCLUSIVE does not mean PASS.**")
    st.markdown("- **DATA_UNAVAILABLE does not mean the strategy failed** — it means the "
                "data could not honestly judge it.")
    st.caption(f"Saved to: `{out['out_dir']}`")
    with st.expander("Technical results & artifact paths"):
        st.json(out["verdict"])
