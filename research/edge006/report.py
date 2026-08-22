"""EDGE-006 markdown dossiers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from research.edge001.report import _md_table, _num, _pct
from research.edge006.constants import OUT_DIR


def write_all(stats: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    p, d, m = stats.get("primary") or {}, stats.get("decision") or {}, stats.get("manifest") or {}
    dec, inf, crash, u = (stats.get("deciles") or {}).get("L1") or {}, stats.get("inference") or {}, stats.get("crash") or {}, stats.get("universe") or {}
    (out / "EDGE_006_DATA_INTEGRITY.md").write_text(
        "# EDGE-006 — Data Integrity\n\n"
        "ADV is `FastInvestable._turn` (20-session mean close×volume, bars ≤ T).\n\n"
        f"- Sessions `{m.get('store_sessions')}` ({m.get('store_first')} → {m.get('store_last')})\n"
        f"- Ranked month-ends `{m.get('n_month_ends_ranked')}` ({m.get('first_rank')} → {m.get('last_rank')})\n"
        f"- Primary periods `{m.get('n_primary_periods')}`\n"
        f"- Mean ranked `{_num(u.get('avg_ranked'), 1)}`\n"
        f"- Protocol SHA `{m.get('protocol_sha')}`\n"
    )
    rows = [[f"D{t['decile']}", t["n_observations"], _pct(t["mean"]), _pct(t["median"]),
             _pct(t["excess_vs_universe"])] for t in dec.get("table") or []]
    (out / "EDGE_006_DECILES.md").write_text(
        "# EDGE-006 — ADV Deciles\n\nD10 = **highest** 20d ADV.\n\n"
        f"Spearman `{_num(dec.get('spearman'))}`. D10−D1 `{_pct(dec.get('d10_minus_d1'))}`.\n\n"
        + (_md_table(["Decile", "n obs", "Mean", "Median", "Excess vs U"], rows) if rows else "") + "\n"
    )
    metrics = [
        ["CAGR net", _pct(p.get("cagr_net"))], ["EW CAGR", _pct(p.get("ew_cagr"))],
        ["Excess vs EW", _pct(p.get("excess_cagr_ew"))], ["Excess vs Nifty", _pct(p.get("excess_cagr_nifty"))],
        ["Sharpe", _num(p.get("sharpe"))], ["Max DD", _pct(p.get("max_dd"))],
        ["TO/year", _pct(p.get("turnover_per_year"))], ["Cost drag/year", _pct(p.get("cost_drag_per_year"))],
        ["Avg names", _num(p.get("avg_names"), 1)], ["Hit vs EW", _pct(p.get("hit_vs_ew"))],
    ]
    yn, yg = p.get("by_year_net") or {}, p.get("by_year_gross") or {}
    sens = stats.get("sensitivities") or {}
    srows = [[k, v.get("n"), _pct(v.get("cagr_net")), _num(v.get("sharpe"))] for k, v in sorted(sens.items())]
    (out / "EDGE_006_PORTFOLIO_RESULTS.md").write_text(
        "# EDGE-006 — Portfolio Results\n\n" + _md_table(["Metric", "L1 Top20"], metrics)
        + "\n\n" + _md_table(["Year", "Net", "Gross"], [[y, _pct(yn.get(y)), _pct(yg.get(y))] for y in sorted(set(yn)|set(yg))])
        + "\n\n" + _md_table(["Spec", "n", "CAGR net", "Sharpe"], srows)
        + "\n\nFormula excess vs EW: " + ", ".join(f"{k}={_pct(v)}" for k, v in (stats.get("formula_excess_ew") or {}).items()) + ".\n"
    )
    (out / "EDGE_006_TURNOVER_COSTS.md").write_text(
        f"# EDGE-006 — Turnover and Costs\n\nTO/year `{_pct(p.get('turnover_per_year'))}`. "
        f"Drag `{_pct(p.get('cost_drag_per_year'))}`. Gross → net `{_pct(p.get('cagr_gross'))}` → `{_pct(p.get('cagr_net'))}`.\n"
    )
    cap = stats.get("capacity") or {}
    crows = [[f"₹{int(k):,}", v.get("flagged_positions"), _pct(v.get("flagged_share"))] for k, v in (cap.get("capitals") or {}).items()]
    (out / "EDGE_006_CAPACITY.md").write_text("# EDGE-006 — Capacity\n\n" + (_md_table(["Capital", "Flagged", "Share"], crows) if crows else "") + "\n")
    blocks = stats.get("blocks") or {}
    brows = [[name, (blocks.get(name) or {}).get("n"), _pct((blocks.get(name) or {}).get("cagr_net")),
              _pct((blocks.get(name) or {}).get("excess_cagr_ew")), _pct((blocks.get(name) or {}).get("excess_cagr_nifty"))]
             for name in ("development", "validation", "confirmation")]
    (out / "EDGE_006_WALK_FORWARD.md").write_text(
        "# EDGE-006 — Walk-Forward\n\n" + _md_table(["Block", "n", "CAGR net", "Ex EW", "Ex Nifty"], brows) + "\n"
    )
    reg = stats.get("regime") or {}
    rrows = [[k, (reg.get(k) or {}).get("months", 0), _pct((reg.get(k) or {}).get("mean"))] for k in ("bull", "sideways", "correction", "bear")]
    (out / "EDGE_006_REGIME_ANALYSIS.md").write_text("# EDGE-006 — Regime\n\n" + _md_table(["Regime", "Months", "Mean net"], rrows) + "\n")
    (out / "EDGE_006_RESULTS.md").write_text(
        f"# EDGE-006 — Results\n\n**Classification: `{d.get('label')}`**\n\n"
        f"Net CAGR {_pct(p.get('cagr_net'))} vs EW {_pct(p.get('ew_cagr'))} (excess {_pct(p.get('excess_cagr_ew'))}). "
        f"Spearman {_num(dec.get('spearman'))}. Harness `{(inf.get('harness_excess_ew') or {}).get('verdict')}`.\n"
        f"Failures: `{d.get('failures')}`\n"
    )
    (out / "EDGE_006_DECISION.md").write_text(
        f"# EDGE-006 — Decision\n\n## `{d.get('label')}`\n\n"
        f"- Failures: {d.get('failures')}\n- Notes: {d.get('notes')}\n"
        f"- Later excess vs EW: {_pct(d.get('later_excess_ew'))}\n"
        f"- Live authorised: `{d.get('live_trading_authorised')}`\n"
        "Last budget slot. Proceed to program synthesis. No production changes.\n"
    )
    return {p.stem: str(p) for p in out.glob("EDGE_006_*.md")}
