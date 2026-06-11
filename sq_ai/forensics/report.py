"""
Terminal report renderer — dark institutional theme via Rich.

Renders: institutional score card, per-layer gauges, fraud probability meter,
red flag timeline, full metric explanations, and the executive verdict.
"""

from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from forensics.analyzer import AnalysisReport
from forensics.models import LayerResult, Severity, Verdict
from forensics import knowledge_graph as kg


def _grade_color(grade: str) -> str:
    if grade.startswith("A"):
        return "bold green"
    if grade.startswith("B"):
        return "green"
    if grade == "C":
        return "yellow"
    return "bold red"

GAUGE_WIDTH = 24

SEV_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}

VERDICT_STYLE = {
    Verdict.STRONG_CANDIDATE: "bold black on green",
    Verdict.WATCHLIST: "bold black on bright_green",
    Verdict.NEUTRAL: "bold black on yellow",
    Verdict.CAUTION: "bold white on dark_orange",
    Verdict.HIGH_RISK: "bold white on red",
    Verdict.AVOID: "bold white on dark_red",
}


def _score_color(s: Optional[float]) -> str:
    if s is None:
        return "dim"
    if s >= 70:
        return "green"
    if s >= 50:
        return "yellow"
    return "red"


def _gauge(score: Optional[float], width: int = GAUGE_WIDTH) -> Text:
    if score is None:
        return Text("─" * width + "  n/a", style="dim")
    filled = round(score / 100 * width)
    t = Text()
    t.append("█" * filled, style=_score_color(score))
    t.append("░" * (width - filled), style="grey23")
    t.append(f" {score:5.1f}", style=f"bold {_score_color(score)}")
    return t


def _layer_panel(L: LayerResult) -> Panel:
    body = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    body.add_column(style="bold bright_white", min_width=26)
    body.add_column()
    for m in L.metrics:
        body.add_row(m.name, Text(m.display, style=_score_color(m.score)
                                  if m.score is not None else "bright_cyan"))
    rows = [body]
    if L.notes:
        rows.append(Text("\n".join(f"· {n}" for n in L.notes), style="dim italic"))
    title = Text.assemble((f" {L.layer} ", "bold bright_white"))
    return Panel(Group(*rows), title=title, subtitle=_gauge(L.score),
                 subtitle_align="right", border_style="grey39", box=box.HEAVY)


def render(report: AnalysisReport, console: Optional[Console] = None,
           explain: bool = False) -> None:
    c = console or Console()
    comp = report.composite

    # ── Header / Institutional Score Card ─────────────────────────────────────
    header = Table(box=None, show_header=False, expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    score_txt = (f"{comp.score:.0f}/100" if comp.score is not None else "N/A")
    score_detail = Text.assemble(
        ("INSTITUTIONAL QUALITY SCORE\n", "dim"),
        (score_txt, f"bold {_score_color(comp.score)} underline"))
    if comp.raw_score is not None and abs(comp.evidence_discount) >= 0.05:
        score_detail.append(
            f"\nraw {comp.raw_score:.0f} · evidence discount "
            f"{-comp.evidence_discount:+.1f}", style="dim")
    if comp.data_reliability is not None:
        score_detail.append(
            f"\ndata reliability {comp.data_reliability:.0f}/100", style="dim")
    header.add_row(
        Text.assemble((f"{report.company}\n", "bold bright_white"),
                      (f"{report.ticker}", "dim")),
        score_detail,
    )
    c.print(Panel(header, title=" QUANT RED FLAG ANALYST™ — INSTITUTIONAL MODULE ",
                  border_style="bright_blue", box=box.DOUBLE))

    # Component breakdown with gauges
    grid = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=False)
    grid.add_column("COMPONENT", min_width=22)
    grid.add_column("WEIGHT", justify="right")
    grid.add_column("SCORE")
    for label, w, s in comp.components:
        grid.add_row(label, f"{w:.0%}", _gauge(s))
    c.print(Panel(grid, title=" SCORE CARD ", border_style="grey39", box=box.HEAVY,
                  subtitle=f" data coverage {comp.coverage:.0%} · "
                           f"confidence {comp.confidence} ",
                  subtitle_align="right"))

    # ── Explainability graph: what moved the score off neutral 50 ────────────
    if comp.contributions:
        wf = Table(box=None, show_header=False, padding=(0, 1))
        wf.add_column(min_width=22)
        wf.add_column(justify="right", min_width=7)
        wf.add_column()
        max_abs = max(abs(v) for _, v in comp.contributions) or 1
        for label, v in comp.contributions:
            bar = "█" * max(1, round(abs(v) / max_abs * 16))
            style = "green" if v >= 0 else "red"
            wf.add_row(label, Text(f"{v:+.1f}", style=f"bold {style}"),
                       Text(bar, style=style))
        c.print(Panel(wf, title=" SCORE ATTRIBUTION — points vs neutral 50 ",
                      border_style="grey39", box=box.HEAVY))

    # ── Institutional Accumulation Score (microstructure) ────────────────────
    micro = report.layers.get("microstructure")
    if micro and micro.score is not None:
        acc = Table(box=None, show_header=False, padding=(0, 2))
        acc.add_column(style="bold", min_width=30)
        acc.add_column()
        acc.add_row("Institutional Accumulation Score", _gauge(micro.score))
        c.print(Panel(acc, title=" MARKET MICROSTRUCTURE ",
                      border_style="grey39", box=box.HEAVY))

    # ── Blow-up similarity ────────────────────────────────────────────────────
    bl = report.layers.get("blowup")
    if bl and bl.extras.get("matches"):
        bt = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        bt.add_column("HISTORICAL FAILURE", min_width=18)
        bt.add_column("SIMILARITY", min_width=30)
        bt.add_column("DRIVEN BY")
        for name, sim, drivers in bl.extras["matches"]:
            style = "bold red" if sim >= 0.6 else ("yellow" if sim >= 0.4 else "green")
            bar = "█" * round(sim * 20) + "░" * (20 - round(sim * 20))
            bt.add_row(name,
                       Text(f"{bar} {sim:.0%}", style=style),
                       Text(", ".join(drivers) or "—", style="dim"))
        c.print(Panel(bt, title=" BLOW-UP SIMILARITY ENGINE ",
                      border_style="red" if bl.extras["matches"][0][1] >= 0.6
                      else "grey39", box=box.HEAVY))

    # ── Fraud probability meter ────────────────────────────────────────────────
    fraud = report.layers.get("fraud")
    if fraud and fraud.extras:
        m = fraud.extras.get("m_score")
        z = fraud.extras.get("z_score")
        f = fraud.extras.get("f_score")
        meter = Table(box=None, show_header=False, padding=(0, 2))
        meter.add_column(style="bold", min_width=18)
        meter.add_column()
        if m is not None:
            risk = "MANIPULATOR PROFILE" if m > -1.78 else (
                "elevated" if m > -2.22 else "low")
            style = "bold red" if m > -1.78 else ("yellow" if m > -2.22 else "green")
            meter.add_row("Fraud Risk", Text(f"M-Score {m:.2f} → {risk}", style=style))
        if z is not None:
            zone = "DISTRESS" if z < 1.81 else ("grey" if z < 2.99 else "safe")
            style = "bold red" if z < 1.81 else ("yellow" if z < 2.99 else "green")
            meter.add_row("Distress Risk", Text(f"Z-Score {z:.2f} → {zone}", style=style))
        if f is not None:
            checks = fraud.extras.get("f_checks", 9)
            style = "green" if f / max(checks, 1) >= 0.66 else (
                "yellow" if f / max(checks, 1) >= 0.4 else "red")
            meter.add_row("Financial Strength",
                          Text(f"Piotroski F {f}/{checks}", style=style))
        c.print(Panel(meter, title=" FRAUD PROBABILITY METER ",
                      border_style="grey39", box=box.HEAVY))

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    if report.graph and report.graph.nodes:
        g = report.graph
        kg_lines = kg.render_text(g)
        kg_text = Text()
        for line in kg_lines:
            kg_text.append(line + "\n", style="bright_cyan" if "→" in line else "dim")
        c.print(Panel(kg_text,
                      title=f" CAUSAL SIGNAL GRAPH — {len(g.nodes)} active nodes ",
                      border_style="bright_cyan", box=box.HEAVY))

    # ── Data Coverage Report ──────────────────────────────────────────────────
    if report.coverage:
        cov = report.coverage
        ct = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=False)
        ct.add_column("DIMENSION", min_width=18)
        ct.add_column("COVERAGE", min_width=28)
        ct.add_column("MISSING", min_width=20)
        for dim in cov.dimensions:
            missing_str = ", ".join(dim.missing[:3]) if dim.missing else "—"
            ct.add_row(dim.dimension, _gauge(dim.score, 20),
                       Text(missing_str, style="dim red" if dim.missing else "dim"))
        c.print(Panel(ct,
                      title=Text.assemble(
                          " DATA RELIABILITY FRAMEWORK  ",
                          ("Overall Coverage: ", "dim"),
                          (f"{cov.overall:.0f}/100 ", _score_color(cov.overall))),
                      border_style="grey39", box=box.HEAVY))

    # ── Capital Allocation panel ───────────────────────────────────────────────
    cap = report.layers.get("capital_allocation")
    if cap and (cap.metrics or cap.notes):
        grade = cap.extras.get("grade", "N/A")
        grade_style = _grade_color(grade)
        c.print(Panel(
            Group(*[_layer_panel(cap)]),
            title=Text.assemble(" CAPITAL ALLOCATION  ", ("Grade: ", "dim"),
                                (f" {grade} ", grade_style + " bold")),
            subtitle=_gauge(cap.score), subtitle_align="right",
            border_style="grey39", box=box.HEAVY))

    # ── Layer panels ──────────────────────────────────────────────────────────
    order = ["forensics", "quant", "fraud", "governance", "smart_money",
             "valuation", "altdata", "microstructure", "promoter", "auditor",
             "credibility"]
    for key in order:
        if key in report.layers and (report.layers[key].metrics
                                     or report.layers[key].notes):
            c.print(_layer_panel(report.layers[key]))

    # ── Evidence Locker ───────────────────────────────────────────────────────
    auditable = [fl for fl in report.flags if fl.evidence_rows]
    if auditable:
        el = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        el.add_column("FLAG", min_width=30)
        el.add_column("AUDITABLE EVIDENCE")
        el.add_column("SOURCE", min_width=14)
        for fl in auditable:
            ev_table = Table(box=None, show_header=False, padding=(0, 1))
            ev_table.add_column(style="dim", min_width=28)
            ev_table.add_column(style="bold bright_white")
            for lbl, vval in fl.evidence_rows:
                ev_table.add_row(lbl, vval)
            rel = fl.reliability
            rel_style = ("green" if rel >= 0.9 else
                         "yellow" if rel >= 0.7 else "red")
            el.add_row(
                Text.assemble((fl.title + "\n", "bold"),
                              (fl.severity.value, SEV_STYLE[fl.severity]),
                              ("  ·  evidence weight ", "dim"),
                              (f"{rel:.0%}", rel_style)),
                ev_table,
                Text("\n".join(fl.sources or ["—"]), style="dim"))
        c.print(Panel(el, title=f" EVIDENCE LOCKER — {len(auditable)} auditable flags ",
                      border_style="bright_blue", box=box.HEAVY))

    # ── Corporate Event Timeline ──────────────────────────────────────────────
    if report.timeline:
        et = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        et.add_column("WHEN", min_width=10)
        et.add_column("CATEGORY", min_width=10)
        et.add_column("EVENT")
        sev_txt = {"CRITICAL": "bold white on red", "HIGH": "bold red",
                   "MEDIUM": "yellow", "LOW": "cyan", "INFO": "dim"}
        for ev in report.timeline:
            et.add_row(ev.date, Text(ev.category, style="bright_cyan"),
                       Text(ev.description,
                            style=sev_txt.get(ev.severity, "white")))
        c.print(Panel(et, title=f" CORPORATE EVENT TIMELINE — "
                                f"{len(report.timeline)} events ",
                      border_style="grey39", box=box.HEAVY))

    # ── Red flag timeline ─────────────────────────────────────────────────────
    if report.flags:
        tl = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        tl.add_column("PERIOD", min_width=8)
        tl.add_column("SEVERITY", min_width=9)
        tl.add_column("RED FLAG")
        for fl in report.flags:
            tl.add_row(fl.period or "—",
                       Text(fl.severity.value, style=SEV_STYLE[fl.severity]),
                       Text.assemble((f"{fl.title}\n", "bold"),
                                     (f"Evidence: {fl.evidence}\n", ""),
                                     (f"Why it matters: {fl.why_it_matters}\n", "dim"),
                                     (f"Precedent: {fl.precedent}", "dim italic")))
        c.print(Panel(tl, title=f" RED FLAG TIMELINE — {len(report.flags)} DETECTED ",
                      border_style="red", box=box.HEAVY))
    else:
        c.print(Panel(Text("No red flags detected across all engines.",
                           style="green"), title=" RED FLAGS ",
                      border_style="green", box=box.HEAVY))

    # ── Metric explanations (--explain) ───────────────────────────────────────
    if explain:
        ex = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        ex.add_column("METRIC", min_width=22)
        ex.add_column("EXPLANATION")
        for key in order:
            for m in report.layers.get(key, LayerResult(layer="", score=None)).metrics:
                ex.add_row(
                    Text(m.name, style="bold"),
                    Text.assemble(("What: ", "bold dim"), f"{m.what}\n",
                                  ("Why: ", "bold dim"), f"{m.why}\n",
                                  ("Good: ", "bold dim"), f"{m.good}\n",
                                  ("Here: ", "bold dim"), m.implication or "—"))
        c.print(Panel(ex, title=" METRIC GLOSSARY ", border_style="grey39",
                      box=box.HEAVY))

    # ── Position Sizing ───────────────────────────────────────────────────────
    if report.sizing:
        sz = report.sizing
        bucket_style = {
            "Core Compounder": "bold black on green",
            "High Conviction": "bold black on bright_green",
            "Standard": "bold black on yellow",
            "Speculative": "yellow",
            "Tracking": "dim",
            "Avoid": "bold white on red",
        }.get(sz.bucket, "white")
        sz_rows = Table(box=None, show_header=False, padding=(0, 2))
        sz_rows.add_column(style="bold", min_width=26)
        sz_rows.add_column()
        sz_rows.add_row("Bucket", Text(f" {sz.bucket} ", style=bucket_style))
        if sz.bucket != "Avoid":
            sz_rows.add_row(
                "Position Range",
                Text(f"{sz.weight_lo}% – {sz.weight_hi}%", style="bright_white"))
            sz_rows.add_row(
                "Suggested Weight",
                Text(f"{sz.suggested_weight:.1f}%", style="bold bright_white"))
        if sz.hard_gate:
            sz_rows.add_row("Hard Gate", Text(sz.hard_gate, style="bold red"))
        if sz.soft_cap:
            sz_rows.add_row("Soft Cap", Text(sz.soft_cap, style="yellow"))
        sz_rows.add_row("Sizing Score", _gauge(sz.sizing_score))
        for k, v in sz.inputs.items():
            sz_rows.add_row(k, Text(v, style="dim"))
        if sz.reasoning:
            sz_rows.add_row("Logic", Text(" → ".join(sz.reasoning), style="dim italic"))
        c.print(Panel(sz_rows, title=" POSITION SIZING ENGINE ",
                      border_style="bright_blue", box=box.DOUBLE))

    # ── Scenario Stress Testing ───────────────────────────────────────────────
    if report.stress:
        outcome_style = {
            "Safe": "bold green", "Stressed": "yellow",
            "High Risk": "bold red", "Critical": "bold white on red",
        }
        st = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        st.add_column("SCENARIO", min_width=18)
        st.add_column("SHOCKS", min_width=26)
        st.add_column("KEY METRICS")
        st.add_column("SURVIVAL", min_width=12)
        st.add_column("OUTCOME", min_width=10)
        for sc in report.stress:
            shocks = ", ".join(filter(None, [
                f"Rev {sc.revenue_shock:+.0%}" if sc.revenue_shock else "",
                f"Margin {sc.margin_shock:+.0%}pp" if sc.margin_shock else "",
                f"Rate +{sc.interest_shock:.0%}" if sc.interest_shock else "",
                f"CFO {sc.cfo_shock:+.0%}" if sc.cfo_shock else "",
            ]))
            metrics_txt = "\n".join(f"{lbl}: {val}" for lbl, val in sc.details
                                    if lbl != "Survival Probability")
            st.add_row(
                sc.name, shocks, metrics_txt,
                Text(f"{sc.survival_probability:.0f}%",
                     style=outcome_style.get(sc.outcome, "white")),
                Text(sc.outcome, style=outcome_style.get(sc.outcome, "white")),
            )
        c.print(Panel(st, title=" SCENARIO STRESS TESTING ",
                      border_style="grey39", box=box.HEAVY))

    # ── AI Investment Committee ───────────────────────────────────────────────
    if report.committee:
        cm = report.committee
        vt = Table(box=box.SIMPLE_HEAD, header_style="bold dim", expand=True)
        vt.add_column("ANALYST", min_width=16)
        vt.add_column("VOTE", min_width=6)
        vt.add_column("RATIONALE")
        vote_style = {"Buy": "bold green", "Hold": "yellow", "Sell": "bold red"}
        for v in cm.votes:
            vt.add_row(Text.assemble((f"{v.analyst}\n", "bold"),
                                     (v.mandate, "dim italic")),
                       Text(v.vote.upper(), style=vote_style[v.vote]),
                       v.rationale)
        tally = "  ·  ".join(f"{k} {n}" for k, n in cm.tally.items())
        c.print(Panel(vt, title=" AI INVESTMENT COMMITTEE ",
                      subtitle=Text.assemble(
                          (f" {tally}  →  CONSENSUS: ", "dim"),
                          (f" {cm.consensus.upper()} ", vote_style[cm.consensus]),
                          (" ",)),
                      subtitle_align="right",
                      border_style="bright_blue", box=box.HEAVY))

    # ── Evidence Chain: why this verdict ──────────────────────────────────────
    if report.graph and report.graph.chains:
        ev_lines = kg.explain_verdict(report.graph, report.flags,
                                      comp.verdict.value)
        ev_text = Text()
        for line in ev_lines:
            if "VERDICT:" in line and "→" not in line:
                ev_text.append(line + "\n", style="bold bright_white")
            elif "→" in line:
                ev_text.append(line + "\n", style="bright_cyan")
            else:
                ev_text.append(line + "\n", style="dim")
        c.print(Panel(ev_text, title=" WHY THIS VERDICT — EVIDENCE CHAIN ",
                      border_style="bright_blue", box=box.HEAVY))

    # ── Verdict ───────────────────────────────────────────────────────────────
    parts = []
    if comp.strengths:
        parts.append(Text("STRENGTHS", style="bold green"))
        parts.extend(Text(f"  ▲ {s}", style="green") for s in comp.strengths)
    if comp.risks:
        parts.append(Text("RISKS", style="bold red"))
        parts.extend(Text(f"  ▼ {r}", style="red") for r in comp.risks)
    if comp.hidden_signals:
        parts.append(Text("HIDDEN SIGNALS", style="bold bright_cyan"))
        parts.extend(Text(f"  ◆ {h}", style="bright_cyan")
                     for h in comp.hidden_signals)
    parts.append(Text())
    parts.append(Text.assemble(
        ("INSTITUTIONAL VERDICT: ", "bold"),
        (f" {comp.verdict.value} ", VERDICT_STYLE[comp.verdict]),
        (f"   confidence: {comp.confidence}", "dim")))
    c.print(Panel(Group(*parts), title=" VERDICT ENGINE ",
                  border_style="bright_blue", box=box.DOUBLE))
    c.print(Text("Analysis identifies quality, risk and manipulation signals — "
                 "it does not predict prices. Not investment advice.",
                 style="dim italic"))


def render_replay(result, console: Optional[Console] = None,
                  full: bool = False) -> None:
    """Render a Research Replay result: the as-of verdict vs what followed."""
    c = console or Console()
    rep = result.report
    comp = rep.composite

    if full:
        render(rep, console=c)

    body = Table(box=None, show_header=False, padding=(0, 2))
    body.add_column(style="bold", min_width=26)
    body.add_column()
    body.add_row("As-of Date", Text(result.as_of.isoformat(), style="bright_white"))
    score_txt = f"{comp.score:.0f}/100" if comp.score is not None else "N/A"
    body.add_row("IQS (then)", Text(score_txt, style=_score_color(comp.score)))
    body.add_row("Verdict (then)",
                 Text(f" {comp.verdict.value} ", style=VERDICT_STYLE[comp.verdict]))
    if rep.sizing:
        body.add_row("Position Bucket (then)", Text(rep.sizing.bucket))
    n_crit = sum(1 for f in rep.flags if f.severity == Severity.CRITICAL)
    n_high = sum(1 for f in rep.flags if f.severity == Severity.HIGH)
    body.add_row("Flags (then)",
                 Text(f"{len(rep.flags)} total · {n_crit} critical · {n_high} high"))

    body.add_row("", Text(""))
    for h in ("6m", "12m", "to date"):
        r = result.forward_returns.get(h)
        b = result.benchmark_returns.get(h)
        if r is None:
            continue
        style = "bold green" if r >= 0 else "bold red"
        line = Text(f"{r:+.1%}", style=style)
        if b is not None:
            line.append(f"   (benchmark {b:+.1%}, alpha {r - b:+.1%})", style="dim")
        body.add_row(f"What happened after ({h})", line)

    c.print(Panel(body,
                  title=f" RESEARCH REPLAY — {rep.company} @ "
                        f"{result.as_of.isoformat()} ",
                  border_style="bright_blue", box=box.DOUBLE))

    if result.caveats:
        c.print(Text("\n".join(f"· {cv}" for cv in result.caveats),
                     style="dim italic"))
