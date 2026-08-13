"""Paper vs live comparison scaffolding.

LIVE broker truth ingestion is prepared here but LIVE trading remains blocked.
Paper and live rows are never overwritten into each other.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from research.forward_evidence.sources import LIVE, PAPER_FORWARD


@dataclass
class PaperLiveComparison:
    policy_id: str
    paper_n: int = 0
    live_n: int = 0
    paper_fill_rate: float | None = None
    live_fill_rate: float | None = None
    paper_avg_slippage: float | None = None
    live_avg_slippage: float | None = None
    paper_expectancy_r: float | None = None
    live_expectancy_r: float | None = None
    paper_net_pnl: float | None = None
    live_net_pnl: float | None = None
    slippage_delta: float | None = None
    expectancy_delta: float | None = None
    plain_language: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def compare_policy(
    *,
    policy_id: str,
    paper_outcomes: list[dict],
    live_outcomes: list[dict] | None = None,
) -> PaperLiveComparison:
    live_outcomes = live_outcomes or []
    paper = [o for o in paper_outcomes if o.get("evidence_source") == PAPER_FORWARD
             or o.get("evidence_source") is None]
    live = [o for o in live_outcomes if o.get("evidence_source") == LIVE]

    def _exp(rows: list[dict]) -> float | None:
        if not rows:
            return None
        vals = [float(r.get("r_outcome", 0) or 0) for r in rows]
        return sum(vals) / len(vals)

    def _slip(rows: list[dict]) -> float | None:
        if not rows:
            return None
        vals = [float(r.get("slippage", 0) or 0) for r in rows]
        return sum(vals) / len(vals)

    def _pnl(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return sum(float(r.get("net_pnl", 0) or 0) for r in rows)

    pe, le = _exp(paper), _exp(live)
    ps, ls = _slip(paper), _slip(live)
    cmp_ = PaperLiveComparison(
        policy_id=policy_id,
        paper_n=len(paper),
        live_n=len(live),
        paper_fill_rate=1.0 if paper else None,
        live_fill_rate=None if not live else 1.0,
        paper_avg_slippage=ps,
        live_avg_slippage=ls,
        paper_expectancy_r=pe,
        live_expectancy_r=le,
        paper_net_pnl=_pnl(paper),
        live_net_pnl=_pnl(live),
        slippage_delta=(None if ps is None or ls is None else ls - ps),
        expectancy_delta=(None if pe is None or le is None else le - pe),
    )
    if not live:
        cmp_.plain_language = "NO LIVE EVIDENCE YET"
    elif pe is not None and le is not None and le < pe:
        slip_note = ""
        if cmp_.slippage_delta is not None and cmp_.slippage_delta > 0:
            slip_note = " mainly because actual slippage is higher"
        cmp_.plain_language = (
            f"Real trades are performing worse than the paper simulation{slip_note}."
        )
    elif pe is not None and le is not None:
        cmp_.plain_language = "Real trades are roughly in line with paper so far."
    else:
        cmp_.plain_language = "Live evidence exists but is too thin to compare cleanly."
    return cmp_


def side_by_side_table(comparisons: list[PaperLiveComparison]) -> list[dict[str, Any]]:
    rows = []
    for c in comparisons:
        rows.append({
            "Policy": c.policy_id,
            "PAPER trades": c.paper_n,
            "REAL trades": c.live_n if c.live_n else "NO LIVE EVIDENCE YET",
            "PAPER expectancy (R)": c.paper_expectancy_r,
            "REAL expectancy (R)": c.live_expectancy_r if c.live_n else "—",
            "Meaning": c.plain_language,
        })
    return rows
