"""
Simple report — the layman's view.

The default output of `python main.py analyze`. One screen, plain English,
no jargon. The full institutional report still exists behind `--pro`;
this view answers the only four questions a normal person has:

    1. Is this a good business?
    2. What's wrong with it?
    3. What's good about it?
    4. Should I buy it?
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from forensics.analyzer import AnalysisReport
from forensics.models import Severity, Verdict

BAR_W = 20

# ── Plain-English translations ────────────────────────────────────────────────

VERDICT_PLAIN = {
    Verdict.STRONG_CANDIDATE: ("Looks like a quality business",
                               "Few warning signs. Worth researching further.",
                               "bold black on green"),
    Verdict.WATCHLIST: ("Decent, with some caveats",
                        "Mostly healthy, but keep an eye on the concerns below.",
                        "bold black on bright_green"),
    Verdict.NEUTRAL: ("Nothing special either way",
                      "No strong reasons to buy or to avoid.",
                      "bold black on yellow"),
    Verdict.CAUTION: ("Be careful with this one",
                      "Several warning signs. Understand them before investing.",
                      "bold white on dark_orange"),
    Verdict.HIGH_RISK: ("Risky — serious warning signs",
                        "Patterns here often come before real trouble.",
                        "bold white on red"),
    Verdict.AVOID: ("Best avoided",
                    "Multiple serious warning signs of the kind seen before "
                    "big collapses.",
                    "bold white on dark_red"),
}

BUCKET_PLAIN = {
    "Core Compounder": "Strong enough to be a meaningful long-term holding.",
    "High Conviction": "Could justify a sizeable position for believers.",
    "Standard": "Fine as a normal-sized position.",
    "Speculative": "Only with money you can afford to lose, in a small amount.",
    "Tracking": "Watch it; don't commit real money yet.",
    "Avoid": "Our analysis says: don't put money here.",
}

SEV_PLAIN = {
    Severity.CRITICAL: ("Serious", "bold white on red"),
    Severity.HIGH: ("Important", "bold red"),
    Severity.MEDIUM: ("Worth noting", "yellow"),
    Severity.LOW: ("Minor", "cyan"),
    Severity.INFO: ("FYI", "dim"),
}

# Flag titles → one-line layman explanations.
FLAG_PLAIN = {
    "Revenue growing while operating cash flow shrinks":
        "Sales are growing on paper, but actual cash coming in is falling.",
    "High accruals — earnings not backed by cash":
        "Reported profits aren't turning into real cash.",
    "Weak cash conversion":
        "The company is bad at turning profit into cash.",
    "Receivables growing unusually fast":
        "Customers owe more and more — sales may be booked before money arrives.",
    "Beneish M-Score in manipulator zone":
        "Its numbers statistically resemble companies that cooked their books.",
    "Altman Z-Score in distress zone":
        "Its finances look like companies that later went bankrupt.",
    "Empire Building Detector: scale over value":
        "Management is buying growth instead of creating value.",
    "Debt increasing rapidly":
        "The company is borrowing a lot more than before.",
    "Persistent negative free cash flow":
        "It spends more cash than it makes, year after year.",
    "Heavy insider selling":
        "The people running the company are selling their own shares.",
    "Margin deterioration":
        "It's keeping less profit from every rupee of sales.",
    "Goodwill concentration — write-down risk":
        "A large part of its 'assets' is just the premium paid for acquisitions.",
    "Share dilution":
        "It keeps issuing new shares, shrinking your slice of the pie.",
    "Dividend exceeds free cash flow":
        "It pays dividends with money it isn't actually earning.",
    "Smart money distribution divergence":
        "Big investors appear to be quietly selling.",
    "Clustered high-volume selling":
        "Unusually heavy selling has shown up in the trading data.",
    "Auditor changed or resigned":
        "Its auditor walked away — often a sign of disagreements.",
    "Audit fee spike":
        "Audit costs jumped, which usually means extra digging was needed.",
    "CARO observations flagged":
        "The auditor formally noted compliance problems.",
    "Qualified audit opinion":
        "The auditor would not fully vouch for the accounts.",
    "Going concern doubt raised by auditor":
        "The auditor doubts the company can survive as is.",
    "Promoter pledging elevated":
        "The founders have pawned a big chunk of their own shares.",
    "Management chronically over-promises":
        "Management keeps promising results it doesn't deliver.",
    "Elevated audit risk score":
        "Independent reviewers rate its audit oversight as weak.",
    "Weak Piotroski F-Score":
        "It fails most basic financial health checks.",
    "Inventory exploding relative to sales":
        "Unsold goods are piling up faster than sales.",
    "Statistical anomaly: outlier revenue quarter":
        "One quarter's sales look oddly out of line with the rest.",
}


def _plain_flag(title: str) -> str:
    if title.startswith("Blow-up profile match:"):
        case = title.split(":", 1)[1].strip()
        return f"Its overall profile resembles {case} before it collapsed."
    return FLAG_PLAIN.get(title, title)


def _bar(score: Optional[float], width: int = BAR_W) -> Text:
    if score is None:
        return Text("not enough data", style="dim italic")
    color = "green" if score >= 70 else "yellow" if score >= 45 else "red"
    filled = round(score / 100 * width)
    t = Text()
    t.append("━" * filled, style=color)
    t.append("━" * (width - filled), style="grey23")
    word = "good" if score >= 70 else "okay" if score >= 45 else "weak"
    t.append(f"  {word}", style=color)
    return t


def _health_rows(report: AnalysisReport) -> List[Tuple[str, Optional[float]]]:
    L = report.layers
    sc = lambda k: L[k].score if k in L else None
    return [
        ("Real profits (not just paper)", sc("forensics")),
        ("Honest-looking accounts", sc("fraud")),
        ("Stable, survivable stock", sc("quant")),
        ("Management quality", sc("capital_allocation")),
        ("Price you'd be paying", sc("valuation")),
    ]


def render_simple(report: AnalysisReport,
                  console: Optional[Console] = None) -> None:
    c = console or Console()
    comp = report.composite
    headline, meaning, style = VERDICT_PLAIN[comp.verdict]

    # ── 1. The verdict, big and unmissable ────────────────────────────────────
    head = Table(box=None, show_header=False, expand=True, padding=(0, 1))
    head.add_column(justify="left", ratio=3)
    head.add_column(justify="right", ratio=2)
    score_t = Text()
    if comp.score is not None:
        col = "green" if comp.score >= 70 else "yellow" if comp.score >= 45 else "red"
        score_t.append(f"{comp.score:.0f}", style=f"bold {col}")
        score_t.append(" / 100", style="dim")
    else:
        score_t.append("N/A", style="dim")
    head.add_row(
        Text.assemble((f"{report.company}\n", "bold bright_white"),
                      (f" {headline} ", style),
                      ("\n" + meaning, "dim")),
        Text.assemble(("QUALITY SCORE\n", "dim"), score_t),
    )
    c.print(Panel(Padding(head, (1, 1)), box=box.ROUNDED,
                  border_style="bright_blue",
                  title=f" {report.symbol} ", title_align="left"))

    # ── 2. Health check at a glance ───────────────────────────────────────────
    health = Table(box=None, show_header=False, padding=(0, 2))
    health.add_column(min_width=30)
    health.add_column()
    for label, score in _health_rows(report):
        health.add_row(Text(label), _bar(score))
    c.print(Panel(Padding(health, (0, 1)), title=" Health check ",
                  title_align="left", box=box.ROUNDED, border_style="grey39"))

    # ── 3. Top concerns, in plain words ──────────────────────────────────────
    serious = [f for f in report.flags
               if f.severity in (Severity.CRITICAL, Severity.HIGH)][:5]
    if serious:
        rows = []
        for f in serious:
            word, sev_style = SEV_PLAIN[f.severity]
            rows.append(Text.assemble(
                (f" {word} ", sev_style), ("  ", ""),
                (_plain_flag(f.title), "bright_white")))
        more = len([f for f in report.flags
                    if f.severity in (Severity.CRITICAL, Severity.HIGH)]) - len(serious)
        if more > 0:
            rows.append(Text(f"…and {more} more (run with --pro for everything)",
                             style="dim italic"))
        c.print(Panel(Padding(Group(*rows), (0, 1)),
                      title=" What worries us ", title_align="left",
                      box=box.ROUNDED, border_style="red"))
    else:
        c.print(Panel(Text("No serious warning signs found.", style="green"),
                      title=" What worries us ", title_align="left",
                      box=box.ROUNDED, border_style="green"))

    # ── 4. Good signs ─────────────────────────────────────────────────────────
    if comp.strengths:
        # strengths are "Name: value — implication"; the implication is the
        # part already written in plain English. Some high-scoring metrics
        # carry cautionary implications (e.g. leverage-inflated ROE) — those
        # don't belong under "looks good".
        _negative = ("inflated", "lower than", "risk", "weak", "erratic",
                     "aggressive", "not backed", "deteriorat")
        plain = [s.split(" — ", 1)[-1] for s in comp.strengths]
        rows = [Text(f"✓  {p}", style="green") for p in plain
                if not any(n in p.lower() for n in _negative)][:3]
        if rows:
            c.print(Panel(Padding(Group(*rows), (0, 1)),
                          title=" What looks good ", title_align="left",
                          box=box.ROUNDED, border_style="green"))

    # ── 5. The bottom line ────────────────────────────────────────────────────
    bottom = Text()
    if report.sizing:
        bottom.append(BUCKET_PLAIN.get(report.sizing.bucket, ""),
                      style="bold bright_white")
        if report.sizing.bucket not in ("Avoid", "Tracking"):
            bottom.append(
                f"  (suggested size: about {report.sizing.suggested_weight:.0f}% "
                "of a portfolio)", style="dim")
    if report.graph and report.graph.chains:
        bottom.append("\nWant the full reasoning? Run with ", style="dim")
        bottom.append("--pro", style="bold dim")
        bottom.append(" to see every number and how we got here.", style="dim")
    c.print(Panel(Padding(bottom, (0, 1)), title=" Bottom line ",
                  title_align="left", box=box.ROUNDED,
                  border_style="bright_blue"))

    c.print(Text("  This is automated analysis of public data — "
                 "not investment advice.", style="dim italic"))
