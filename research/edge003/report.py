"""EDGE-003 markdown dossiers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from research.edge001.report import _md_table, _num, _pct
from research.edge003.constants import OUT_DIR


def write_all(stats: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    p = stats.get("primary") or {}
    d = stats.get("decision") or {}
    m = stats.get("manifest") or {}
    inf = stats.get("inference") or {}
    crash = stats.get("crash") or {}
    u = stats.get("universe") or {}
    incl = stats.get("inclusion") or {}

    (out / "EDGE_003_DATA_INTEGRITY.md").write_text(
        "# EDGE-003 — Data Integrity\n\n"
        "Same store and PIT screens as EDGE-001/002, including `live_on_session` "
        "(no stale last prints). SMA windows use closes ≤ T only.\n\n"
        f"- Sessions `{m.get('store_sessions')}` ({m.get('store_first')} → {m.get('store_last')})\n"
        f"- Ranked month-ends `{m.get('n_month_ends_ranked')}` "
        f"({m.get('first_rank')} → {m.get('last_rank')})\n"
        f"- Primary periods `{m.get('n_primary_periods')}`\n"
        f"- Mean ranked `{_num(u.get('avg_ranked'), 1)}` of investable `{_num(u.get('avg_investable'), 1)}`\n"
        f"- Mean T1 names `{_num(u.get('avg_n_t1'), 1)}` share `{_pct(u.get('avg_t1_share'))}`\n"
        f"- Listing `{m.get('listing_pit')}`; sector `{m.get('sector_pit')}`\n"
        f"- Nifty source `{m.get('nifty_source')}`\n"
        f"- Protocol SHA `{m.get('protocol_sha')}`\n"
        f"- Parent consumed `{m.get('parent_consumed')}`\n"
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
        ["Avg names", _num(p.get("avg_names"), 1)], ["Mean T1 share", _pct(p.get("mean_qualifier_share"))],
        ["Hit vs EW", _pct(p.get("hit_vs_ew"))],
    ]
    yn, yg = p.get("by_year_net") or {}, p.get("by_year_gross") or {}
    sens = stats.get("sensitivities") or {}
    srows = []
    for k, v in sorted(sens.items()):
        cagr = v.get("cagr_net_calendar") if v.get("annualization") == "calendar_span" else v.get("cagr_net")
        srows.append([k, v.get("n"), _pct(cagr), _num(v.get("sharpe")), _pct(v.get("max_dd")), v.get("annualization") or "monthly_12"])
    (out / "EDGE_003_PORTFOLIO_RESULTS.md").write_text(
        "# EDGE-003 — Portfolio Results (T1 all-qualifiers monthly)\n\n"
        + _md_table(["Metric", "T1 All Monthly"], metrics) + "\n\n## By year\n\n"
        + _md_table(["Year", "Net", "Gross"], [[y, _pct(yn.get(y)), _pct(yg.get(y))] for y in sorted(set(yn) | set(yg))])
        + "\n\n## Sensitivities (not for winner-picking)\n\n"
        + _md_table(["Spec", "n", "CAGR net", "Sharpe", "Max DD", "Ann."], srows)
        + "\n\n2-month and quarterly CAGRs use calendar span, not 12/year "
        "(a monthly annualizer would inflate them). Formula excess vs EW is "
        "only computed for monthly-aligned books.\n\nFormula excess vs EW: "
        + ", ".join(f"{k}={_pct(v)}" for k, v in (stats.get("formula_excess_ew") or {}).items()) + ".\n"
    )

    (out / "EDGE_003_INCLUSION.md").write_text(
        "# EDGE-003 — Inclusion vs Excluded (H2)\n\n"
        "Forward return = next-open to next rebalance open. T1 = included; exT1 = investable fail.\n\n"
        f"- Mean T1 share `{_pct(incl.get('mean_t1_share'))}` (n T1 `{_num(incl.get('t1_n_mean'), 1)}`, "
        f"n exT1 `{_num(incl.get('ext1_n_mean'), 1)}`)\n"
        f"- Mean T1 next-month `{_pct(incl.get('t1_mean'))}` vs excluded `{_pct(incl.get('ext1_mean'))}` "
        f"(spread `{_pct(incl.get('t1_minus_ext1'))}`)\n"
        f"- Months T1 > exT1 `{_pct(incl.get('t1_beats_ext1_share'))}`\n"
        f"- Share vs T1−exT1 Spearman `{_num(incl.get('share_vs_spread_spearman'))}`\n"
        "If share ≈ 100% and excess ≈ 0 the filter has no content.\n"
    )

    (out / "EDGE_003_TURNOVER_COSTS.md").write_text(
        "# EDGE-003 — Turnover and Costs\n\n"
        "Variable-N equal-weight one-way TO = ½ Σ |w_new − w_old|.\n\n"
        f"One-way TO/year `{_pct(p.get('turnover_per_year'))}`. "
        f"Cost drag/year `{_pct(p.get('cost_drag_per_year'))}`. "
        f"Gross → net `{_pct(p.get('cagr_gross'))}` → `{_pct(p.get('cagr_net'))}`.\n"
    )
    cap = stats.get("capacity") or {}
    crows = [[f"₹{int(k):,}", v.get("flagged_rebalances"), _pct(v.get("flagged_share"))]
             for k, v in (cap.get("capitals") or {}).items()]
    (out / "EDGE_003_CAPACITY.md").write_text(
        "# EDGE-003 — Capacity\n\n"
        "Flag a rebalance if equal-weight clip (capital / N) ≥ 5% of median 20d ADV. "
        "Large N makes retail clips small.\n\n"
        + (_md_table(["Capital", "Flagged rebalances", "Share"], crows) if crows else "") + "\n"
    )
    blocks = stats.get("blocks") or {}
    brows = []
    for name in ("development", "validation", "confirmation"):
        b = blocks.get(name) or {}
        brows.append([name, b.get("n"), _pct(b.get("cagr_net")), _pct(b.get("excess_cagr_ew")),
                      _pct(b.get("excess_cagr_nifty")), _pct(b.get("mean_qualifier_share")),
                      _num(b.get("sharpe"))])
    (out / "EDGE_003_WALK_FORWARD.md").write_text(
        "# EDGE-003 — Walk-Forward\n\n"
        "Blocks frozen before results. T1 / all-qualifiers / monthly not retuned.\n\n"
        + _md_table(["Block", "n", "CAGR net", "Ex EW", "Ex Nifty", "T1 share", "Sharpe"], brows) + "\n"
    )
    reg = stats.get("regime") or {}
    rrows = [[k, (reg.get(k) or {}).get("months", 0), _pct((reg.get(k) or {}).get("mean")),
              _pct((reg.get(k) or {}).get("cagr"))] for k in ("bull", "sideways", "correction", "bear")]
    (out / "EDGE_003_REGIME_ANALYSIS.md").write_text(
        "# EDGE-003 — Regime (descriptive only)\n\nNo regime gate.\n\n"
        + _md_table(["Regime", "Months", "Mean net", "CAGR-like"], rrows) + "\n"
    )
    (out / "EDGE_003_RESULTS.md").write_text(
        "# EDGE-003 — Results\n\n"
        f"**Classification: `{d.get('label')}`**\n\n"
        f"Net CAGR {_pct(p.get('cagr_net'))} vs EW {_pct(p.get('ew_cagr'))} "
        f"(excess {_pct(p.get('excess_cagr_ew'))}). "
        f"Sharpe {_num(p.get('sharpe'))}, max DD {_pct(p.get('max_dd'))}. "
        f"Mean T1 share {_pct(p.get('mean_qualifier_share'))} (N `{_num(p.get('avg_names'), 1)}`). "
        f"Included minus excluded {_pct(incl.get('t1_minus_ext1'))}. "
        f"Worst month {_pct(p.get('worst_month'))} ({(crash.get('worst_month') or {}).get('rebalance')}).\n\n"
        f"Inference: {inf.get('excess_ew')}\n"
        f"Harness: `{(inf.get('harness_excess_ew') or {}).get('verdict')}` — "
        f"{(inf.get('harness_excess_ew') or {}).get('insight')}\n\n"
        f"Failures: `{d.get('failures')}`\n"
    )
    (out / "EDGE_003_DECISION.md").write_text(
        "# EDGE-003 — Decision\n\n"
        f"## `{d.get('label')}`\n\n"
        "None of the labels authorise paper, live, FEATURE-002, or production BUY changes.\n\n"
        f"- Failures: {d.get('failures')}\n- Notes: {d.get('notes')}\n"
        f"- Later excess vs EW: {_pct(d.get('later_excess_ew'))}\n"
        f"- Live authorised: `{d.get('live_trading_authorised')}`\n"
        "Do not rescue with Top-20 distance rank, a shorter SMA, or a regime gate inside EDGE-003.\n"
    )
    return {p.stem: str(p) for p in out.glob("EDGE_003_*.md")}
