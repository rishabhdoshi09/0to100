"""
Knowledge Graph — Causal Signal Chains.

Signals are nodes; edges represent "A is evidence for / contributes to B."
The graph is static (all possible edges are coded here) but only the ACTIVE
subgraph (signals that actually fired in this analysis) is rendered.

This turns a flat list of 17 flags into "3 root causes generating 5 chains."
Root causes = nodes with no active incoming edges.
Terminal signals = the highest-severity downstream consequences.

Rendering produces a tiered ASCII layout: ROOT → INTERMEDIATE → TERMINAL.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Causal edge definitions ───────────────────────────────────────────────────
# (source_code, target_code, relationship_label)
EDGES: List[Tuple[str, str, str]] = [
    # Accounting manipulation cluster
    ("HIGH_ACCRUALS",       "WEAK_CASH_CONVERSION",  "accruals reduce future CFO"),
    ("HIGH_ACCRUALS",       "BENEISH_MANIPULATOR",   "TATA ratio drives M-Score"),
    ("RECEIVABLES_GAP",     "REV_CFO_DIVERGENCE",    "unpaid receivables deplete CFO"),
    ("RECEIVABLES_GAP",     "BENEISH_MANIPULATOR",   "DSRI ratio drives M-Score"),
    ("REV_CFO_DIVERGENCE",  "WEAK_CASH_CONVERSION",  "revenue not converting to cash"),
    ("MARGIN_DETERIORATION","WEAK_CASH_CONVERSION",  "lower margins compress cash"),
    ("WEAK_CASH_CONVERSION","NEGATIVE_FCF",          "poor conversion leads to cash burn"),

    # Balance sheet stress cluster
    ("NEGATIVE_FCF",        "DEBT_SURGE",            "FCF shortfall funded by debt"),
    ("NEGATIVE_FCF",        "ALTMAN_DISTRESS",        "cash burn raises distress risk"),
    ("DEBT_SURGE",          "ALTMAN_DISTRESS",        "leverage reduces Z-Score"),
    ("DEBT_SURGE",          "ROE_LEVERAGE_INFLATED",  "debt amplifies ROE artificially"),
    ("GOODWILL_RISK",       "EMPIRE_BUILDING",        "goodwill concentration signals acquisitions"),
    ("EMPIRE_BUILDING",     "ALTMAN_DISTRESS",        "value-destroying capex raises debt"),

    # Governance / promoter cluster
    ("AUDITOR_CHANGE",      "BENEISH_MANIPULATOR",   "auditor exit precedes disclosure"),
    ("PROMOTER_PLEDGE",     "INSIDER_SELLING",        "pledging creates forced-sale risk"),
    ("GOVERNANCE_WEAK",     "BENEISH_MANIPULATOR",   "weak oversight enables manipulation"),

    # Market structure cluster
    ("VOLUME_DISTRIBUTION", "INSIDER_SELLING",        "institutional exit leaves volume footprint"),

    # Terminal convergence
    ("BENEISH_MANIPULATOR", "BLOWUP_SIMILARITY",      "fraud profile matches historical cases"),
    ("ALTMAN_DISTRESS",     "BLOWUP_SIMILARITY",      "distress profile matches IL&FS/DHFL"),
    ("INSIDER_SELLING",     "BLOWUP_SIMILARITY",      "smart money exit precedes disclosure"),
    ("EMPIRE_BUILDING",     "BLOWUP_SIMILARITY",      "acquisition spiral matches GE/Carillion"),
]

# ── Flag title → signal code mapping ─────────────────────────────────────────
# Covers both prediction_ledger signal codes and graph-only codes.
TITLE_TO_CODE: Dict[str, str] = {
    "High accruals — earnings not backed by cash":          "HIGH_ACCRUALS",
    "Weak cash conversion":                                  "WEAK_CASH_CONVERSION",
    "Receivables growing unusually fast":                    "RECEIVABLES_GAP",
    "Revenue growing while operating cash flow shrinks":     "REV_CFO_DIVERGENCE",
    "Beneish M-Score in manipulator zone":                   "BENEISH_MANIPULATOR",
    "Altman Z-Score in distress zone":                       "ALTMAN_DISTRESS",
    "Empire Building Detector: scale over value":            "EMPIRE_BUILDING",
    "Debt increasing rapidly":                               "DEBT_SURGE",
    "Persistent negative free cash flow":                    "NEGATIVE_FCF",
    "Heavy insider selling":                                 "INSIDER_SELLING",
    "Margin deterioration":                                  "MARGIN_DETERIORATION",
    "Goodwill concentration — write-down risk":              "GOODWILL_RISK",
    "ROE inflated by leverage, not business quality":        "ROE_LEVERAGE_INFLATED",
    "Dividend exceeds free cash flow":                       "NEGATIVE_FCF",
    "Share dilution":                                        "INSIDER_SELLING",
    "Smart money distribution divergence":                   "VOLUME_DISTRIBUTION",
    "Clustered high-volume selling":                         "VOLUME_DISTRIBUTION",
    "Auditor changed or resigned":                           "AUDITOR_CHANGE",
    "Promoter pledging elevated":                            "PROMOTER_PLEDGE",
    "Elevated audit risk score":                             "GOVERNANCE_WEAK",
    "Weak Piotroski F-Score":                                "ALTMAN_DISTRESS",
    "Statistical anomaly: outlier revenue quarter":          "RECEIVABLES_GAP",
    "Inventory exploding relative to sales":                 "HIGH_ACCRUALS",
}

# Map blow-up similarity flags
for case in ("Enron", "Satyam", "DHFL", "IL&FS", "Luckin", "Wirecard",
             "Carillion", "Evergrande"):
    TITLE_TO_CODE[f"Blow-up profile match: {case}"] = "BLOWUP_SIMILARITY"

# Short labels for graph display
CODE_LABEL: Dict[str, str] = {
    "HIGH_ACCRUALS":        "HIGH ACCRUALS",
    "WEAK_CASH_CONVERSION": "WEAK CASH\nCONVERSION",
    "RECEIVABLES_GAP":      "RECEIVABLES\nGAP",
    "REV_CFO_DIVERGENCE":   "REV > CFO\nDIVERGENCE",
    "BENEISH_MANIPULATOR":  "BENEISH\nMANIPULATOR",
    "ALTMAN_DISTRESS":      "ALTMAN\nDISTRESS",
    "EMPIRE_BUILDING":      "EMPIRE\nBUILDING",
    "DEBT_SURGE":           "DEBT SURGE",
    "NEGATIVE_FCF":         "NEGATIVE FCF",
    "INSIDER_SELLING":      "INSIDER\nSELLING",
    "MARGIN_DETERIORATION": "MARGIN\nDETERIOR.",
    "GOODWILL_RISK":        "GOODWILL\nRISK",
    "ROE_LEVERAGE_INFLATED":"ROE LEVERAGE\nINFLATED",
    "AUDITOR_CHANGE":       "AUDITOR\nCHANGE",
    "PROMOTER_PLEDGE":      "PROMOTER\nPLEDGING",
    "GOVERNANCE_WEAK":      "GOVERNANCE\nWEAK",
    "VOLUME_DISTRIBUTION":  "VOL DISTRIB.",
    "BLOWUP_SIMILARITY":    "BLOW-UP\nSIMILARITY",
}


@dataclass
class GraphNode:
    code: str
    label: str
    severity: str        # CRITICAL / HIGH / MEDIUM / LOW / INFO
    flag_title: str


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, GraphNode]
    edges: List[Tuple[str, str, str]]    # (src, tgt, label)
    roots: List[str]                     # codes with no incoming edges
    terminals: List[str]                 # codes with no outgoing edges in active graph
    chains: List[List[str]]              # paths from root to terminal
    n_isolated: int                      # nodes not connected to any edge


def build(flags) -> KnowledgeGraph:
    """Build the active subgraph from the flags that fired."""
    from forensics.models import Severity

    # Map fired flags → signal codes
    active: Dict[str, GraphNode] = {}
    for fl in flags:
        code = TITLE_TO_CODE.get(fl.title)
        if code and code not in active:
            active[code] = GraphNode(
                code=code,
                label=CODE_LABEL.get(code, code),
                severity=fl.severity.value,
                flag_title=fl.title,
            )

    if not active:
        return KnowledgeGraph(nodes={}, edges=[], roots=[], terminals=[], chains=[], n_isolated=0)

    # Filter edges to active subgraph
    active_edges = [
        (s, t, rel) for s, t, rel in EDGES
        if s in active and t in active
    ]

    # Find roots (no incoming) and terminals (no outgoing) in active subgraph
    has_incoming = {t for _, t, _ in active_edges}
    has_outgoing = {s for s, _, _ in active_edges}
    roots = [c for c in active if c not in has_incoming]
    terminals = [c for c in active if c not in has_outgoing]

    # BFS chains from roots to terminals
    chains: List[List[str]] = []
    adj: Dict[str, List[str]] = defaultdict(list)
    for s, t, _ in active_edges:
        adj[s].append(t)

    for root in roots:
        queue: deque = deque([[root]])
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if not adj[curr]:
                if len(path) > 1:
                    chains.append(path)
            else:
                for nxt in adj[curr]:
                    if nxt not in path:  # no cycles
                        queue.append(path + [nxt])

    # Count isolated (no edges)
    n_isolated = sum(1 for c in active
                     if c not in has_incoming and c not in has_outgoing)

    return KnowledgeGraph(
        nodes=active, edges=active_edges,
        roots=roots, terminals=terminals,
        chains=chains, n_isolated=n_isolated,
    )


def render_text(g: KnowledgeGraph) -> List[str]:
    """Return lines of ASCII art for the knowledge graph."""
    if not g.nodes:
        return ["No active causal chains."]

    lines: List[str] = []
    sev_prefix = {
        "CRITICAL": "⛔", "HIGH": "🔴", "MEDIUM": "🟡",
        "LOW": "🟢", "INFO": "⬜",
    }

    def short(code: str) -> str:
        return CODE_LABEL.get(code, code).replace("\n", " ")

    def sev(code: str) -> str:
        node = g.nodes.get(code)
        return sev_prefix.get(node.severity, "•") if node else "•"

    if g.chains:
        lines.append(f"  {len(g.roots)} root cause(s) → "
                     f"{len(g.chains)} causal chain(s) → "
                     f"{len(g.terminals)} terminal signal(s)")
        lines.append("")
        for chain in g.chains:
            parts = [f"{sev(c)} {short(c)}" for c in chain]
            lines.append("  " + "  →  ".join(parts))
        lines.append("")

    if g.n_isolated:
        lines.append(f"  {g.n_isolated} isolated signal(s) (no causal chain):")
        for code, node in g.nodes.items():
            if (code not in {s for s, _, _ in g.edges}
                    and code not in {t for _, t, _ in g.edges}):
                lines.append(f"    {sev(code)} {short(code)}")

    return lines
