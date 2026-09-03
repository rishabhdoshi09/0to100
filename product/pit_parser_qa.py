"""Parser quality ledger.

Parsing success is not factual correctness. Samples compare extracted
facts against values read from the official document itself.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from product.pit_xbrl import PARSER_VERSION, parse_xbrl

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "product" / "parser_qa.jsonl"


def compare_facts(
    extracted: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    rel_tol: float = 0.015,
) -> dict[str, Any]:
    field_ok: dict[str, bool] = {}
    failures: list[str] = []
    ambiguous: list[str] = []
    for key, exp in expected.items():
        got = extracted.get(key)
        if got is None:
            field_ok[key] = False
            failures.append(key)
            continue
        try:
            g, e = float(got), float(exp)
        except (TypeError, ValueError):
            field_ok[key] = str(got) == str(exp)
            if not field_ok[key]:
                failures.append(key)
            continue
        if e == 0:
            ok = abs(g) < rel_tol
        else:
            ok = abs(g - e) / abs(e) <= rel_tol
        field_ok[key] = ok
        if not ok:
            failures.append(key)
        elif abs(g - e) / (abs(e) or 1) > rel_tol / 3:
            ambiguous.append(key)
    n = len(expected)
    n_ok = sum(1 for v in field_ok.values() if v)
    return {
        "parser_version": PARSER_VERSION,
        "n_expected": n,
        "n_matched": n_ok,
        "accuracy": round(n_ok / n, 4) if n else None,
        "field_ok": field_ok,
        "field_failures": failures,
        "ambiguous_values": ambiguous,
        "parsing_success_is_not_correctness": True,
    }


def validate_xbrl_text(
    xml_text: str,
    expected: Mapping[str, Any],
    *,
    symbol: str = "",
    source: str = "",
    persist: bool = False,
    path: Path | None = None,
) -> dict[str, Any]:
    parsed = parse_xbrl(xml_text)
    cmp = compare_facts(dict(parsed.get("facts") or {}), expected)
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "source": source,
        "parser_success": bool(parsed.get("ok")),
        "parser_confidence": parsed.get("confidence"),
        "unit_normalization": "INR_crore_and_INR_per_share",
        "revision_conflicts": [],
        **cmp,
        "extracted_n_fields": parsed.get("n_fields"),
        "period_end": parsed.get("period_end"),
        "board_date": parsed.get("board_date"),
    }
    if persist:
        target = path or LEDGER
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    return row
