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
    header.add_row(
        Text.assemble((f"{report.company}\n", "bold bright_white"),
                      (f"{report.ticker}", "dim")),
        Text.assemble(("INSTITUTIONAL QUALITY SCORE\n", "dim"),
                      (score_txt, f"bold {_score_color(comp.score)} underline")),
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

    # ── Layer panels ──────────────────────────────────────────────────────────
    order = ["forensics", "quant", "fraud", "governance", "smart_money",
             "valuation", "altdata"]
    for key in order:
        if key in report.layers and (report.layers[key].metrics
                                     or report.layers[key].notes):
            c.print(_layer_panel(report.layers[key]))

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
