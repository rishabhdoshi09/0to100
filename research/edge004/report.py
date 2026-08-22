"""EDGE-004 markdown dossiers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from research.edge001.report import _md_table, _num, _pct
from research.edge004.constants import OUT_DIR


def write_all(stats: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    p = stats.get("primary") or {}
    d = stats.get("decision") or {}
    m = stats.get("manifest") or {}
    dec = (stats.get("deciles") or {}).get("R1") or {}
    inf = stats.get("inference") or {}
    crash = stats.get("crash") or {}
    u = stats.get("universe") or {}

    (out / "EDGE_004_DATA_INTEGRITY.md").write_text(
        "# EDGE-004 — Data Integrity\n\n"
        "Same store and PIT screens as EDGE-001/002/003, including `live_on_session`. "
        "R1 uses closes ≤ T only (`incl_momentum`).\n\n"
        f"- Sessions `{m.get('store_sessions')}` ({m.get('store_first')} → {m.get('store_last')})\n"
        f"- Ranked month-ends `{m.get('n_month_ends_ranked')}` "
        f"({m.get('first_rank')} → {m.get('last_rank')})\n"
        f"- Primary periods `{m.get('n_primary_periods')}`\n"
        f"- Mean ranked `{_num(u.get('avg_ranked'), 1)}` of investable `{_num(u.get('avg_investable'), 1)}`\n"
        f"- Listing `{m.get('listing_pit')}`; sector `{m.get('sector_pit')}`\n"
        f"- Nifty source `{m.get('nifty_source')}`\n"
        f"- Protocol SHA `{m.get('protocol_sha')}`\n"
        f"- Parent fail `{m.get('parent_fail')}` (29-name 1/3/5d reversal)\n"
    )

    rows = []
    for t in dec.get("table") or []:
        rows.append([
            f"D{t['decile']}", t["n_observations"], _pct(t["mean"]), _pct(t["median"]),
            _pct(t["excess_vs_universe"]), f"[{_pct(t['ci_lower'])}, {_pct(t['ci_upper'])}]",
        ])
    (out / "EDGE_004_DECILES.md").write_text(
        "# EDGE-004 — Prior-return Deciles\n\n"
        "D10 = **lowest** 21-session return (score = −R1). Forward = next-open to next rebalance open.\n\n"
        f"Spearman `{_num(dec.get('spearman'))}`. D10−D1 `{_pct(dec.get('d10_minus_d1'))}`.\n\n"
        + (_md_table(["Decile", "n obs", "Mean", "Median", "Excess vs U", "CI"], rows) if rows else "")
        + "\n"
    )

    metrics = [
        ["Years", _num(p.get("years"), 2)], ["Rebalances", p.get("n")],
        ["CAGR gross", _pct(p.get("cagr_gross"))], ["CAGR net", _pct(p.get("cagr_net"))],
        ["EW CAGR", _pct(p.get("ew_cagr"))], ["Nifty proxy CAGR", _pct(p.get("nifty_cagr"))],
        ["Excess vs EW", _pct(p.get("excess_cagr_ew"))], ["Excess vs Nifty", _pct(p.get("excess_cagr_nifty"))],
        ["Vol", _pct(p.get("vol"))], ["Sharpe", _num(p.get("sharpe"))], ["Sortino", _num(p.get("sortino"))],
        ["Max DD", _pct(p.get("max_dd"))], ["Calmar", _num(p.get("calmar"))],
        ["Win months", _pct(p.get("win_months"))], ["TO/year", _pct(p.get("turnover_per_year"))],
        ["Cost drag/year", _pct(p.get("cost_drag_per_year"))],
        ["Worst month", _pct(p.get("worst_month"))], ["Best month", _pct(p.get("best_month"))],
        ["Avg names", _num(p.get("avg_names"), 1)], ["Hit vs EW", _pct(p.get("hit_vs_ew"))],
    ]
    yn, yg = p.get("by_year_net") or {}, p.get("by_year_gross") or {}
    sens = stats.get("sensitivities") or {}
    srows = [[k, v.get("n"), _pct(v.get("cagr_net")), _num(v.get("sharpe")), _pct(v.get("max_dd"))]
             for k, v in sorted(sens.items())]
    (out / "EDGE_004_PORTFOLIO_RESULTS.md").write_text(
        "# EDGE-004 — Portfolio Results (R1 21d losers, Top20 monthly)\n\n"
        + _md_table(["Metric", "R1 Top20 Monthly"], metrics) + "\n\n## By year\n\n"
        + _md_table(["Year", "Net", "Gross"], [[y, _pct(yn.get(y)), _pct(yg.get(y))] for y in sorted(set(yn) | set(yg))])
        + "\n\n## Sensitivities (not for winner-picking)\n\n"
        + _md_table(["Spec", "n", "CAGR net", "Sharpe", "Max DD"], srows)
        + "\n\n2-month/quarterly CAGRs use the monthly annualizer and are **not** comparable.\n"
        + "Formula excess vs EW: "
        + ", ".join(f"{k}={_pct(v)}" for k, v in (stats.get("formula_excess_ew") or {}).items()) + ".\n"
    )
    (out / "EDGE_004_TURNOVER_COSTS.md").write_text(
        "# EDGE-004 — Turnover and Costs\n\n"
        f"One-way TO/year `{_pct(p.get('turnover_per_year'))}`. "
        f"Cost drag/year `{_pct(p.get('cost_drag_per_year'))}`. "
        f"Gross → net `{_pct(p.get('cagr_gross'))}` → `{_pct(p.get('cagr_net'))}`.\n"
    )
    cap = stats.get("capacity") or {}
    crows = [[f"₹{int(k):,}", v.get("flagged_positions"), _pct(v.get("flagged_share"))]
             for k, v in (cap.get("capitals") or {}).items()]
    (out / "EDGE_004_CAPACITY.md").write_text(
        "# EDGE-004 — Capacity\n\nFlag if position ≥ 5% of 20d ADV.\n\n"
        + (_md_table(["Capital", "Flagged", "Share"], crows) if crows else "") + "\n"
    )
    blocks = stats.get("blocks") or {}
    brows = []
    for name in ("development", "validation", "confirmation"):
        b = blocks.get(name) or {}
        brows.append([name, b.get("n"), _pct(b.get("cagr_net")), _pct(b.get("excess_cagr_ew")),
                      _pct(b.get("excess_cagr_nifty")), _num(b.get("sharpe"))])
    (out / "EDGE_004_WALK_FORWARD.md").write_text(
        "# EDGE-004 — Walk-Forward\n\n"
        "Blocks frozen before results. R1/Top20/monthly not retuned.\n\n"
        + _md_table(["Block", "n", "CAGR net", "Ex EW", "Ex Nifty", "Sharpe"], brows) + "\n"
    )
    reg = stats.get("regime") or {}
    rrows = [[k, (reg.get(k) or {}).get("months", 0), _pct((reg.get(k) or {}).get("mean")),
              _pct((reg.get(k) or {}).get("cagr"))] for k in ("bull", "sideways", "correction", "bear")]
    (out / "EDGE_004_REGIME_ANALYSIS.md").write_text(
        "# EDGE-004 — Regime (descriptive only)\n\nNo regime gate.\n\n"
        + _md_table(["Regime", "Months", "Mean net", "CAGR-like"], rrows) + "\n"
    )
    (out / "EDGE_004_RESULTS.md").write_text(
        "# EDGE-004 — Results\n\n"
        f"**Classification: `{d.get('label')}`**\n\n"
        f"Net CAGR {_pct(p.get('cagr_net'))} vs EW {_pct(p.get('ew_cagr'))} "
        f"(excess {_pct(p.get('excess_cagr_ew'))}). "
        f"Sharpe {_num(p.get('sharpe'))}, max DD {_pct(p.get('max_dd'))}. "
        f"Worst month {_pct(p.get('worst_month'))} ({(crash.get('worst_month') or {}).get('rebalance')}). "
        f"Decile Spearman {_num(dec.get('spearman'))}. D10−D1 {_pct(dec.get('d10_minus_d1'))}.\n\n"
        f"Inference: {inf.get('excess_ew')}\n"
        f"Harness: `{(inf.get('harness_excess_ew') or {}).get('verdict')}` — "
        f"{(inf.get('harness_excess_ew') or {}).get('insight')}\n\n"
        f"Failures: `{d.get('failures')}`\n"
    )
    (out / "EDGE_004_DECISION.md").write_text(
        "# EDGE-004 — Decision\n\n"
        f"## `{d.get('label')}`\n\n"
        "None of the labels authorise paper, live, FEATURE-002, or production BUY changes.\n\n"
        f"- Failures: {d.get('failures')}\n- Notes: {d.get('notes')}\n"
        f"- Later excess vs EW: {_pct(d.get('later_excess_ew'))}\n"
        f"- Live authorised: `{d.get('live_trading_authorised')}`\n"
        "Do not rescue with 10-session lookback, winner book, or stops inside EDGE-004.\n"
    )
    return {p.stem: str(p) for p in out.glob("EDGE_004_*.md")}
