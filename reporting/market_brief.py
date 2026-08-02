"""Themed market research brief — FII/DII flows, bulk deals, index options context."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from reporting.research_dossier import build_equity_dossier


def build_institutional_market_brief(
    *,
    days: int = 30,
    symbol_limit: int = 4,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Compose a research-style brief from persisted institutional + scan data only."""
    from data.fii_dii_store import workspace_payload

    inst = workspace_payload(days=days)
    cash = dict(inst.get("cash") or {})
    bulk_buys = list(inst.get("bulk_buy_symbols") or [])
    bulk_deals = list(inst.get("bulk_deals") or [])

    from product.scan_store import load_scan

    scan = load_scan() or {}
    scan_records = [dict(r) for r in (scan.get("records") or []) if isinstance(r, Mapping)]
    overlap = []
    for sym in bulk_buys[:20]:
        row = next((r for r in scan_records if str(r.get("symbol", "")).upper() == sym), None)
        if row:
            overlap.append(
                {
                    "symbol": sym,
                    "company": row.get("company") or sym,
                    "verdict": row.get("verdict") or row.get("status"),
                    "score": row.get("score"),
                    "sector": row.get("sector"),
                }
            )

    featured_symbols = [item["symbol"] for item in overlap[:symbol_limit]]
    if not featured_symbols and bulk_buys:
        featured_symbols = bulk_buys[:symbol_limit]

    company_dossiers = []
    for sym in featured_symbols:
        try:
            company_dossiers.append(build_equity_dossier(sym, scan_payload=scan))
        except Exception:
            continue

    totals = dict(cash.get("totals") or {})
    today = dict(cash.get("today") or {})
    derivatives = dict(inst.get("derivatives") or {})

    narrative_parts: list[str] = []
    if cash.get("available"):
        narrative_parts.append(
            f"Over the last {cash.get('sessions', 0)} session(s) in store, "
            f"FII net cash flow totalled ₹{totals.get('fii_net_cr', 0):,.0f} Cr "
            f"and DII ₹{totals.get('dii_net_cr', 0):,.0f} Cr."
        )
        if today.get("date"):
            narrative_parts.append(
                f"Latest session ({today.get('date')}): FII net ₹{today.get('fii_net', 0):,.0f} Cr, "
                f"DII net ₹{today.get('dii_net', 0):,.0f} Cr."
            )
    if inst.get("insight"):
        narrative_parts.append(str(inst.get("insight")))
    if bulk_buys:
        narrative_parts.append(
            f"Bulk-deal screen shows net institutional buying interest in "
            f"{len(bulk_buys)} symbol(s), including {', '.join(bulk_buys[:8])}."
        )
    if derivatives.get("total_net") is not None:
        narrative_parts.append(
            f"FII derivatives positioning (latest NSE snapshot): total net ₹{derivatives.get('total_net', 0):,.0f} Cr "
            f"(index futures ₹{derivatives.get('index_futures_net', 0):,.0f} Cr, "
            f"index options ₹{derivatives.get('index_options_net', 0):,.0f} Cr)."
        )

    try:
        from options.chain_fetch import chain_workspace

        nifty_opts = chain_workspace("NIFTY")
        if nifty_opts.get("available"):
            narrative_parts.append(
                f"NIFTY nearest-expiry PCR {nifty_opts.get('pcr')} · max pain {nifty_opts.get('max_pain')} · "
                f"{nifty_opts.get('bias')} ({nifty_opts.get('note')})."
            )
    except Exception:
        nifty_opts = {"available": False}

    if not narrative_parts:
        narrative_parts.append(
            "Institutional flow history is not yet backfilled. Run "
            "`python main.py fii-dii-backfill` to populate NSE cash-market data."
        )

    return {
        "schema_version": 1,
        "report_type": "INSTITUTIONAL_MARKET_BRIEF",
        "title": "What institutions are doing in Indian equities",
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "window_days": days,
        "institutional": inst,
        "nifty_options": nifty_opts if isinstance(nifty_opts, dict) else {"available": False},
        "bulk_buy_overlap": overlap,
        "featured_companies": company_dossiers,
        "narrative": " ".join(narrative_parts),
        "bulk_deals_sample": bulk_deals[:25],
        "disclaimer": (
            "Numbers are sourced from NSE public endpoints and QuantTerm persisted stores. "
            "Bulk deals reflect exchange disclosures for the current window only. "
            "This brief organises evidence; it is not investment advice."
        ),
    }


def generate_institutional_market_report(
    *,
    days: int = 30,
    symbol_limit: int = 4,
    report_dir: str | None = None,
) -> Any:
    from pathlib import Path

    from reporting.pdf_renderer import render_institutional_market_pdf
    from reporting.research_dossier import _report_path, DEFAULT_REPORT_DIR

    brief = build_institutional_market_brief(days=days, symbol_limit=symbol_limit)
    target = Path(report_dir or DEFAULT_REPORT_DIR)
    path = _report_path("institutional_market_brief", f"{days}d", target)
    render_institutional_market_pdf(brief, path)
    return path
