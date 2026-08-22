"""Write EDGE-001 markdown dossiers from computed stats. Research only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from research.edge001.constants import OUT_DIR


def _pct(x, digits=2) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{100.0 * v:.{digits}f}%"


def _num(x, digits=3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_data_integrity(stats: dict[str, Any], path: Path) -> None:
    m = stats.get("manifest") or {}
    u = stats.get("universe") or {}
    path.write_text(
        "# EDGE-001 — Data Integrity\n\n"
        "EDGE-001 reuses official bhav OHLCV via `load_store_frames` → `get_ohlcv` "
        "(corporate-action adjustment **on read**). The store is not rewritten.\n\n"
        f"- Store sessions: `{m.get('store_sessions')}` "
        f"({m.get('store_first')} → {m.get('store_last')})\n"
        f"- Frames loaded: `{m.get('n_frames')}`\n"
        f"- Month-ends ranked: `{m.get('n_month_ends_ranked')}` "
        f"({m.get('first_rank')} → {m.get('last_rank')})\n"
        f"- Primary complete holding periods: `{m.get('n_primary_periods')}`\n"
        f"- Mean candidates / investable / ranked: "
        f"{_num(u.get('avg_candidates'), 1)} / {_num(u.get('avg_investable'), 1)} / "
        f"{_num(u.get('avg_ranked'), 1)}\n"
        f"- Listing identity: **{m.get('listing_pit')}** — membership is bars present ≤ T, "
        "not an official historical listing file.\n"
        f"- Sector map: **{m.get('sector_pit')}** — today’s NIFTY 500 comment map applied "
        "historically. Concentration tables are descriptive only.\n"
        f"- CA policy: {m.get('ca_policy')}. Resolved official events are already in the "
        "adjusted read path. EDGE-001 does **not** re-run the SEPA exhaustive unresolved "
        "gap audit; residual phantom gaps remain a PIT limitation.\n"
        f"- Nifty source: `{m.get('nifty_source')}` "
        f"(Nifty 500 official local series available: `{m.get('nifty500_available')}`).\n"
        f"- Fill: `{m.get('fill')}` (no same-close). Stop: `{m.get('stop')}`.\n"
        f"- Cost model: `core.costs.round_trip_cost_pct('CNC')` = "
        f"{_num(m.get('rt_cost_pct'), 2)} percent points per round-trip, applied to "
        "one-way turnover (`cost = one_way × rt_pct / 100`).\n"
        f"- Protocol SHA: `{m.get('protocol_sha')}` activated "
        f"{m.get('protocol_activated_ist')}.\n\n"
        "## Honest limitations\n\n"
        "1. Survivorship: names that never appear in the 2019–2026 store cannot enter. "
        "Delisted names **with** historical bars remain as-of investable — that is the "
        "correct direction — but the store is not a complete exchange membership tape.\n"
        "2. FEATURE-001 / SEPA already mined this window. Confirmation 2025–2026 is "
        "held-out **for this protocol**, not philosophically pristine lifetime OOS.\n"
        "3. Index period returns use the local Nifty close series (entry date ≤ t ≤ exit "
        "date), not a traded futures roll. Broad alternative is equal-weight investable.\n"
        "4. Missing next-open drops the name from that month’s equal-weight (no invented fill).\n"
    )


def write_deciles(stats: dict[str, Any], path: Path) -> None:
    d = (stats.get("deciles") or {}).get("M1") or {}
    rows = []
    for t in d.get("table") or []:
        rows.append([
            f"D{t['decile']} ({'weakest' if t['decile']==1 else 'strongest' if t['decile']==10 else ''})".replace(" ()", ""),
            t["n_observations"],
            _pct(t["mean"]),
            _pct(t["median"]),
            _pct(t["excess_vs_universe"]),
            f"[{_pct(t['ci_lower'])}, {_pct(t['ci_upper'])}]",
        ])
    body = (
        "# EDGE-001 — Decile Monotonicity\n\n"
        "Each month the investable universe is ranked on **M1 12-1** and split into "
        "deciles. Forward return is next-session open → next rebalance’s next open. "
        "Inference is clustered by rebalance (the monthly mean is the observation).\n\n"
        f"Spearman(decile, mean next-month return) = **{_num(d.get('spearman'))}**. "
        f"D10 − D1 = **{_pct(d.get('d10_minus_d1'))}**. "
        f"D10-only flag = `{d.get('d10_only')}`.\n\n"
    )
    if rows:
        body += _md_table(
            ["Momentum Decile", "n observations", "Mean next-month return", "Median", "Excess vs universe", "CI"],
            rows,
        ) + "\n\n"
    body += (
        "Primary evidence question: does return generally improve as momentum rank improves?\n\n"
        "Year-by-year decile means (monthly average, not compounded):\n\n"
    )
    byy = d.get("by_year") or {}
    if byy:
        headers = ["Year"] + [f"D{i}" for i in range(1, 11)]
        yrows = []
        for y, m in byy.items():
            yrows.append([y] + [_pct(m.get(i) if isinstance(m.get(str(i)), float) else m.get(i)) for i in range(1, 11)])
        body += _md_table(headers, yrows) + "\n"
    path.write_text(body)


def write_portfolio(stats: dict[str, Any], path: Path) -> None:
    p = stats.get("primary") or {}
    m = stats.get("manifest") or {}
    rows = [
        ["Metric", "M1 Top20 Monthly"],
        ["Years", _num(p.get("years"), 2)],
        ["Rebalances", p.get("n")],
        ["CAGR gross", _pct(p.get("cagr_gross"))],
        ["CAGR net", _pct(p.get("cagr_net"))],
        ["Benchmark CAGR (Nifty)", _pct(p.get("nifty_cagr"))],
        ["Benchmark CAGR (EW universe)", _pct(p.get("ew_cagr"))],
        ["Excess CAGR vs Nifty", _pct(p.get("excess_cagr_nifty"))],
        ["Excess CAGR vs EW", _pct(p.get("excess_cagr_ew"))],
        ["Volatility", _pct(p.get("vol"))],
        ["Sharpe", _num(p.get("sharpe"))],
        ["Sortino", _num(p.get("sortino"))],
        ["Max drawdown", _pct(p.get("max_dd"))],
        ["Calmar", _num(p.get("calmar"))],
        ["Win months", _pct(p.get("win_months"))],
        ["Turnover/year (one-way)", _pct(p.get("turnover_per_year"))],
        ["Cost drag/year", _pct(p.get("cost_drag_per_year"))],
        ["Worst month", _pct(p.get("worst_month"))],
        ["Best month", _pct(p.get("best_month"))],
        ["Avg names filled", _num(p.get("avg_names"), 1)],
        ["Beta vs Nifty", _num(p.get("beta_nifty"))],
        ["Monthly hit vs Nifty", _pct(p.get("hit_vs_nifty"))],
        ["Monthly hit vs EW", _pct(p.get("hit_vs_ew"))],
    ]
    body = (
        "# EDGE-001 — Portfolio Results (primary = M1 Top20 monthly)\n\n"
        f"Protocol SHA `{m.get('protocol_sha')}`. Fill = next open. No stop. "
        "Gross and net both shown. Top 20 was locked before looking at later blocks.\n\n"
        + _md_table(["Metric", "M1 Top20 Monthly"], [[a, b] for a, b in rows[1:]])
        + "\n\n## By-year net / gross\n\n"
    )
    yn, yg = p.get("by_year_net") or {}, p.get("by_year_gross") or {}
    body += _md_table(
        ["Year", "Net", "Gross"],
        [[y, _pct(yn.get(y)), _pct(yg.get(y))] for y in sorted(set(yn) | set(yg))],
    )
    body += "\n\n## Size / cadence / horizon sensitivities (net CAGR, not for winner-picking)\n\n"
    sens = stats.get("sensitivities") or {}
    srows = []
    for k, v in sorted(sens.items()):
        srows.append([k, v.get("n"), _pct(v.get("cagr_net")), _num(v.get("sharpe")), _pct(v.get("max_dd")), _pct(v.get("turnover_per_year"))])
    body += _md_table(["Spec", "n", "CAGR net", "Sharpe", "Max DD", "TO/year"], srows)
    body += (
        "\n\nFormula excess CAGR vs EW (net): "
        + ", ".join(f"{k}={_pct(v)}" for k, v in (stats.get("formula_excess_ew") or {}).items())
        + ".\n"
    )
    path.write_text(body)


def write_turnover(stats: dict[str, Any], path: Path) -> None:
    p = stats.get("primary") or {}
    path.write_text(
        "# EDGE-001 — Turnover and Costs\n\n"
        "Primary book is equal-weight Top 20. One-way turnover = "
        "`(|added| + |removed|) / (2N)`, 100% on the first deploy. "
        f"CNC round-trip from `core.costs` is applied as "
        f"`one_way × rt_pct / 100`.\n\n"
        f"- Average one-way turnover per year: **{_pct(p.get('turnover_per_year'))}**\n"
        f"- Average cost drag per year: **{_pct(p.get('cost_drag_per_year'))}**\n"
        f"- CAGR gross → net: {_pct(p.get('cagr_gross'))} → {_pct(p.get('cagr_net'))}\n"
        f"- Cost vs EW: the edge must survive this drag. "
        f"Failure flag `costs_destroy_edge` is evaluated in the decision file.\n\n"
        "Per-rebalance added / removed / retained / cost live in "
        "`logs/edge001/portfolio_periods.json` and `transaction_ledger.csv`.\n"
    )


def write_capacity(stats: dict[str, Any], path: Path) -> None:
    cap = stats.get("capacity") or {}
    rows = []
    for k, v in (cap.get("capitals") or {}).items():
        rows.append([f"₹{int(k):,}", v.get("flagged_positions"), _pct(v.get("flagged_share"))])
    path.write_text(
        "# EDGE-001 — Retail Capacity\n\n"
        "Equal-weight position = capital / N. Participation = position / 20-day ADV "
        f"as of the rebalance close. Flag if participation ≥ {_pct(cap.get('participation_flag'))}. "
        "No institutional impact model.\n\n"
        + (_md_table(["Capital", "Flagged positions", "Share of name-months"], rows) if rows else "_no capacity rows_")
        + "\n\nADV is trailing 20-session INR turnover from the as-of bar. "
        "Names with missing ADV are flagged (cannot certify executability).\n"
    )


def write_walk_forward(stats: dict[str, Any], path: Path) -> None:
    blocks = stats.get("blocks") or {}
    rows = []
    for name in ("development", "validation", "confirmation"):
        b = blocks.get(name) or {}
        rows.append([
            name, b.get("n"), _pct(b.get("cagr_net")), _pct(b.get("excess_cagr_ew")),
            _pct(b.get("excess_cagr_nifty")), _num(b.get("sharpe")), _pct(b.get("max_dd")),
        ])
    path.write_text(
        "# EDGE-001 — Walk-Forward\n\n"
        "Blocks were frozen in `EDGE_001_RESEARCH_PROTOCOL.md` before backtests. "
        "M1 / Top 20 / monthly were **not** changed after opening validation or confirmation.\n\n"
        "| Block | Rebalance dates | Role |\n|---|---|---|\n"
        "| Warm-up | until 252 sessions exist | no claim |\n"
        "| Development | first valid rebalance → 2022-12-31 | specification lock |\n"
        "| Validation | 2023-01-01 → 2024-12-31 | robustness |\n"
        "| Confirmation | 2025-01-01 → 2026-08-21 | confirmation |\n\n"
        + _md_table(["Block", "n", "CAGR net", "Excess vs EW", "Excess vs Nifty", "Sharpe", "Max DD"], rows)
        + "\n\n2019–2026 was already mined by SEPA / FEATURE-001. Confirmation is held-out "
        "for **this** protocol only. No period is philosophically pristine.\n"
    )


def write_regime(stats: dict[str, Any], path: Path) -> None:
    reg = stats.get("regime") or {}
    rows = []
    for k in ("bull", "sideways", "correction", "bear", "unknown"):
        b = reg.get(k) or {}
        rows.append([k, b.get("months", 0), _pct(b.get("mean")), _pct(b.get("cagr")), _pct(b.get("max_dd"))])
    path.write_text(
        "# EDGE-001 — Regime Attribution (descriptive only)\n\n"
        "PIT labels from `research/sepa003/regime.py` (`classify_regime_level` + `regime_at`) "
        "on the official Nifty series (or EW Nifty-50 proxy). "
        "**H4 is not a gate.** No regime filter is added after seeing this table. "
        "A regime-conditioned strategy would be a new experiment.\n\n"
        + _md_table(["Regime", "Months", "Mean monthly net", "CAGR-like", "Max DD in bucket"], rows)
        + "\n"
    )


def write_results(stats: dict[str, Any], path: Path) -> None:
    p = stats.get("primary") or {}
    d = stats.get("decision") or {}
    inf = stats.get("inference") or {}
    mom = stats.get("prod_momentum") or {}
    crash = stats.get("crash") or {}
    dec = (stats.get("deciles") or {}).get("M1") or {}
    path.write_text(
        "# EDGE-001 — Results\n\n"
        f"**Classification: `{d.get('label')}`**\n\n"
        "No classification authorises paper, live, FEATURE-002, or production BUY changes.\n\n"
        "## Primary (M1 Top20 monthly, net)\n\n"
        f"- CAGR net {_pct(p.get('cagr_net'))} vs Nifty {_pct(p.get('nifty_cagr'))} "
        f"(excess {_pct(p.get('excess_cagr_nifty'))}) and vs EW {_pct(p.get('ew_cagr'))} "
        f"(excess {_pct(p.get('excess_cagr_ew'))})\n"
        f"- Sharpe {_num(p.get('sharpe'))}, Sortino {_num(p.get('sortino'))}, "
        f"max DD {_pct(p.get('max_dd'))}, Calmar {_num(p.get('calmar'))}\n"
        f"- Worst month {_pct(p.get('worst_month'))} "
        f"({(crash.get('worst_month') or {}).get('rebalance')}), "
        f"worst quarter {_pct((crash.get('worst_quarter') or {}).get('ret'))} "
        f"from {(crash.get('worst_quarter') or {}).get('start')}\n"
        f"- Decile Spearman {_num(dec.get('spearman'))}, D10−D1 {_pct(dec.get('d10_minus_d1'))}\n\n"
        "## Inference on monthly net excess vs EW\n\n"
        f"{inf.get('excess_ew')}\n\n"
        f"Harness: `{((inf.get('harness_excess_ew') or {}).get('verdict'))}` — "
        f"{((inf.get('harness_excess_ew') or {}).get('insight'))}\n\n"
        "## Production MOMENTUM comparison\n\n"
        "Production `MOMENTUM` is 5-day time-series + RSI + volume on scanner cards. "
        "EDGE-001 is 12-1 cross-sectional, monthly, no stop. The comparison **reuses "
        "EDGE-001’s next-open monthly hold** and ranks the PIT universe on 5-session "
        "return so expectancy is not mixed with the scanner’s 10–20 day ticket.\n\n"
        f"- CS M1 net CAGR {_pct(p.get('cagr_net'))} vs 5d-TS Top20 net CAGR {_pct(mom.get('cagr_net'))}\n"
        f"- Excess vs EW: M1 {_pct(p.get('excess_cagr_ew'))} vs 5d-TS {_pct(mom.get('excess_cagr_ew'))}\n"
        f"- Turnover/year: M1 {_pct(p.get('turnover_per_year'))} vs 5d-TS {_pct(mom.get('turnover_per_year'))}\n\n"
        "They are **not the same phenomenon**. A 5-day TS sort is closer to short-horizon "
        "reversal/continuation than to 12-1 relative strength.\n\n"
        f"Failure flags: `{d.get('failures')}`\n"
    )


def write_decision(stats: dict[str, Any], path: Path) -> None:
    d = stats.get("decision") or {}
    path.write_text(
        "# EDGE-001 — Decision\n\n"
        f"## `{d.get('label')}`\n\n"
        "Exactly one label is allowed. None of them authorise live trading, paper "
        "autopilot, FEATURE-002 changes, or production BUY edits.\n\n"
        f"- Failures: {d.get('failures')}\n"
        f"- Notes: {d.get('notes')}\n"
        f"- Later-block mean excess vs EW: {_pct(d.get('later_excess_ew'))}\n"
        f"- Later-block mean excess vs Nifty: {_pct(d.get('later_excess_nifty'))}\n"
        f"- Live authorised: `{d.get('live_trading_authorised')}`\n"
        f"- Paper authorised: `{d.get('paper_trading_authorised')}`\n"
        f"- FEATURE-002 change authorised: `{d.get('feature002_change_authorised')}`\n\n"
        "If the label is REJECT or RESEARCH-ONLY, do **not** rescue the hypothesis "
        "with stops, sector caps, news, or AI inside this milestone. "
        "A stop overlay would be EDGE-002 only after a surviving primary effect.\n"
    )


def write_all(stats: dict[str, Any], out_dir: Path | None = None) -> dict[str, str]:
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "data_integrity": (write_data_integrity, out / "EDGE_001_DATA_INTEGRITY.md"),
        "deciles": (write_deciles, out / "EDGE_001_DECILES.md"),
        "portfolio": (write_portfolio, out / "EDGE_001_PORTFOLIO_RESULTS.md"),
        "turnover": (write_turnover, out / "EDGE_001_TURNOVER_COSTS.md"),
        "capacity": (write_capacity, out / "EDGE_001_CAPACITY.md"),
        "walk_forward": (write_walk_forward, out / "EDGE_001_WALK_FORWARD.md"),
        "regime": (write_regime, out / "EDGE_001_REGIME_ANALYSIS.md"),
        "results": (write_results, out / "EDGE_001_RESULTS.md"),
        "decision": (write_decision, out / "EDGE_001_DECISION.md"),
    }
    written = {}
    for k, (fn, p) in mapping.items():
        fn(stats, p)
        written[k] = str(p)
    return written
