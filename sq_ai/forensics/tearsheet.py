"""
L15 — Institutional Tear Sheet, Executive One-Pager, and Social Card.

Three artifacts from a single function call. All rendered to strings so they
can be printed, saved to files, or passed to a front-end.

  generate(report)  →  TearSheetBundle(institutional, one_pager, social_card)

All three are plain text, copyable to any medium. The social card is designed
to travel on X, LinkedIn, WhatsApp, and Reddit without any image renderer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from forensics.analyzer import AnalysisReport
from forensics.models import Severity, Verdict


@dataclass
class TearSheetBundle:
    institutional: str   # full research-quality report (plain text)
    one_pager: str       # condensed decision format (plain text)
    social_card: str     # shareable card (copyable anywhere)


def _render_to_str(renderable, width: int = 100) -> str:
    buf = io.StringIO()
    con = Console(file=buf, width=width, force_terminal=False, no_color=True)
    con.print(renderable)
    return buf.getvalue()


def _flag_icon(sev: Severity) -> str:
    return {"CRITICAL": "⛔", "HIGH": "🔴", "MEDIUM": "🟡",
            "LOW": "🟢", "INFO": "⬜"}.get(sev.value, "•")


def _score_bar(s: Optional[float], w: int = 20) -> str:
    if s is None:
        return "—"
    filled = round(s / 100 * w)
    return "█" * filled + "░" * (w - filled) + f" {s:.0f}/100"


def _grade_color(grade: str) -> str:
    if grade.startswith("A"):
        return "bold green"
    if grade.startswith("B"):
        return "green"
    if grade == "C":
        return "yellow"
    return "bold red"


def _verdict_line(v: Verdict) -> str:
    return {
        Verdict.STRONG_CANDIDATE: "✅ STRONG CANDIDATE",
        Verdict.WATCHLIST: "✅ WATCHLIST",
        Verdict.NEUTRAL: "⚪ NEUTRAL",
        Verdict.CAUTION: "⚠️  CAUTION",
        Verdict.HIGH_RISK: "🔴 HIGH RISK",
        Verdict.AVOID: "⛔ AVOID",
    }.get(v, str(v.value))


# ── INSTITUTIONAL TEAR SHEET ─────────────────────────────────────────────────

def _institutional(report: AnalysisReport) -> str:
    c = report.composite
    cap = report.layers.get("capital_allocation")
    grade = cap.extras.get("grade", "N/A") if cap else "N/A"
    comm = report.committee
    blowup_layer = report.layers.get("blowup")
    top_match = blowup_layer.extras.get("matches", [None])[0] if blowup_layer else None
    today = date.today().strftime("%d %b %Y")

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  QUANT RED FLAG ANALYST™  |  INSTITUTIONAL TEAR SHEET")
    lines.append(f"  {report.company}  ({report.ticker})")
    lines.append(f"  Report date: {today}  |  Data: yfinance / public filings")
    lines.append("=" * 78)
    lines.append("")

    # Score card
    lines.append("INSTITUTIONAL QUALITY SCORE")
    lines.append(f"  {_score_bar(c.score, 40)}")
    lines.append(f"  Data coverage {c.coverage:.0%}  |  Confidence: {c.confidence}")
    lines.append("")
    lines.append(f"{'COMPONENT':<26} {'WEIGHT':>7}  {'SCORE':>26}")
    lines.append("  " + "-" * 64)
    for label, w, s in c.components:
        lines.append(f"  {label:<24} {w:>6.0%}  {_score_bar(s, 24)}")

    if c.contributions:
        lines.append("")
        lines.append("SCORE ATTRIBUTION  (pts vs neutral 50)")
        for lbl, v in c.contributions:
            bar = ("+" if v >= 0 else "") + "█" * max(1, round(abs(v) / 15 * 12))
            lines.append(f"  {lbl:<26}  {v:+6.1f}  {bar}")

    # Capital allocation
    lines.append("")
    lines.append("CAPITAL ALLOCATION")
    lines.append(f"  Grade: {grade}")
    if cap:
        roic = cap.extras.get("roic")
        inc_roic = cap.extras.get("inc_roic")
        emp = cap.extras.get("empire_signals", [])
        if roic is not None:
            lines.append(f"  ROIC:               {roic:.1%}")
        if inc_roic is not None:
            lines.append(f"  Incremental ROIC:   {inc_roic:.1%}")
        if emp:
            lines.append(f"  Empire signals:     {'; '.join(emp)}")

    # Committee
    if comm:
        lines.append("")
        lines.append("AI INVESTMENT COMMITTEE")
        for v in comm.votes:
            lines.append(f"  {v.analyst:<18}  {v.vote:<5}  {v.rationale}")
        lines.append(f"  {'CONSENSUS':<18}  {comm.consensus}")

    # Blow-up
    if top_match:
        name, sim, drivers = top_match
        lines.append("")
        lines.append("BLOW-UP SIMILARITY")
        lines.append(f"  Top match: {name}  ({sim:.0%} similar)")
        if drivers:
            lines.append(f"  Driven by: {', '.join(drivers)}")

    # Flags
    if report.flags:
        lines.append("")
        lines.append("RED FLAGS")
        for fl in report.flags:
            lines.append(f"  {_flag_icon(fl.severity)} [{fl.severity.value}] {fl.title}")
            lines.append(f"    Evidence: {fl.evidence}")
            if fl.evidence_rows:
                for lbl, val_ in fl.evidence_rows:
                    lines.append(f"      {lbl}: {val_}")
            lines.append(f"    Source: {', '.join(fl.sources) if fl.sources else 'see evidence'}")
            lines.append(f"    Precedent: {fl.precedent}")
            lines.append("")

    # Strengths & risks
    if c.strengths:
        lines.append("STRENGTHS")
        for s in c.strengths:
            lines.append(f"  ▲ {s}")
        lines.append("")
    if c.risks:
        lines.append("KEY RISKS")
        for r in c.risks:
            lines.append(f"  ▼ {r}")
        lines.append("")
    if c.hidden_signals:
        lines.append("HIDDEN SIGNALS")
        for h in c.hidden_signals:
            lines.append(f"  ◆ {h}")
        lines.append("")

    # Verdict
    lines.append("─" * 78)
    lines.append(f"  VERDICT:  {_verdict_line(c.verdict)}"
                 f"   (confidence: {c.confidence})")
    lines.append("─" * 78)
    lines.append("")
    lines.append("Analysis identifies quality, risk and manipulation signals.")
    lines.append("Not investment advice. Verify all figures against primary filings.")

    return "\n".join(lines)


# ── EXECUTIVE ONE-PAGER ───────────────────────────────────────────────────────

def _one_pager(report: AnalysisReport) -> str:
    c = report.composite
    cap = report.layers.get("capital_allocation")
    grade = cap.extras.get("grade", "N/A") if cap else "N/A"
    comm = report.committee
    blowup_layer = report.layers.get("blowup")
    top_match = blowup_layer.extras.get("matches", [None])[0] if blowup_layer else None
    top_risk = (report.flags[0].title if report.flags else "None identified")
    top_strength = (c.strengths[0].split(" — ")[0] if c.strengths else "—")

    lines: list[str] = []
    lines.append("━" * 56)
    lines.append(f"  {report.symbol}  |  EXECUTIVE ONE-PAGER")
    lines.append(f"  {report.company}")
    lines.append("━" * 56)
    lines.append(f"  IQ Score:           {c.score:.0f}/100  ({c.confidence} confidence)"
                 if c.score else "  IQ Score:           N/A")
    lines.append(f"  Verdict:            {c.verdict.value}")
    lines.append(f"  Capital Allocation: {grade}")
    if comm:
        lines.append(f"  Committee:          {comm.consensus}  "
                     f"({comm.tally['Buy']}B/{comm.tally['Hold']}H/{comm.tally['Sell']}S)")
    if top_match:
        name, sim, _ = top_match
        lines.append(f"  Blow-Up Similarity: {name}  {sim:.0%}")
    lines.append("")
    lines.append(f"  TOP RISK:     {top_risk}")
    lines.append(f"  TOP STRENGTH: {top_strength}")
    lines.append("")
    if report.flags:
        lines.append("  RED FLAGS:")
        for fl in report.flags[:4]:
            lines.append(f"   {_flag_icon(fl.severity)} {fl.title}")
    lines.append("")
    if c.hidden_signals:
        lines.append("  HIDDEN SIGNALS:")
        for h in c.hidden_signals[:3]:
            lines.append(f"   ◆ {h}")
    lines.append("━" * 56)
    lines.append("  Not investment advice.")

    return "\n".join(lines)


# ── SOCIAL CARD ───────────────────────────────────────────────────────────────

def _social_card(report: AnalysisReport) -> str:
    c = report.composite
    cap = report.layers.get("capital_allocation")
    grade = cap.extras.get("grade", "?") if cap else "?"
    comm = report.committee
    blowup_layer = report.layers.get("blowup")
    top_match = blowup_layer.extras.get("matches", [None])[0] if blowup_layer else None

    top_risk = report.flags[0].title if report.flags else "None detected"
    top_strength = (c.strengths[0].split(":")[0].strip() if c.strengths
                    else "Insufficient data")
    score_str = f"{c.score:.0f}/100" if c.score is not None else "N/A"
    consensus = comm.consensus if comm else "N/A"
    blow_str = (f"{top_match[0]}  {top_match[1]:.0%}"
                if top_match else "No significant match")

    lines: list[str] = []
    w = 48
    border = "╔" + "═" * (w - 2) + "╗"
    bottom = "╚" + "═" * (w - 2) + "╝"

    def row(text: str = "", pad: int = 2) -> str:
        content = (" " * pad) + text
        return "║" + content.ljust(w - 2) + "║"

    def divider() -> str:
        return "╠" + "═" * (w - 2) + "╣"

    lines.append(border)
    lines.append(row())
    lines.append(row(f"🏦  {report.symbol}  —  Forensic Analysis"))
    lines.append(row(f"    {report.company}"))
    lines.append(row())
    lines.append(divider())
    lines.append(row())
    lines.append(row(f"Institutional Quality Score:   {score_str}"))
    lines.append(row(f"Capital Allocation Grade:      {grade}"))
    lines.append(row(f"Committee Consensus:           {consensus}"))
    lines.append(row())
    lines.append(divider())
    lines.append(row())
    lines.append(row(f"🔴 Top Risk:"))
    lines.append(row(f"   {top_risk[:w - 6]}", pad=3))
    lines.append(row())
    lines.append(row(f"✅ Top Strength:"))
    lines.append(row(f"   {top_strength[:w - 6]}", pad=3))
    lines.append(row())
    lines.append(divider())
    lines.append(row())
    lines.append(row(f"🔬 Closest Blow-Up Match:"))
    lines.append(row(f"   {blow_str}"))
    lines.append(row())
    lines.append(divider())
    lines.append(row())
    lines.append(row(f"📊 Verdict:  {c.verdict.value}"))
    lines.append(row(f"   Confidence: {c.confidence}"))
    lines.append(row())
    lines.append(bottom)
    lines.append("")
    lines.append("Powered by Quant Red Flag Analyst™")
    lines.append("Not investment advice.")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def generate(report: AnalysisReport) -> TearSheetBundle:
    return TearSheetBundle(
        institutional=_institutional(report),
        one_pager=_one_pager(report),
        social_card=_social_card(report),
    )


def save(bundle: TearSheetBundle, prefix: str = "report") -> list[str]:
    """Save all three artifacts and return the file paths."""
    paths = []
    for attr, suffix in (
        ("institutional", "_tearsheet.txt"),
        ("one_pager", "_one_pager.txt"),
        ("social_card", "_social_card.txt"),
    ):
        path = f"{prefix}{suffix}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(getattr(bundle, attr))
        paths.append(path)
    return paths
