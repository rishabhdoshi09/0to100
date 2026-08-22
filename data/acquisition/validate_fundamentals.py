"""Cross-source / reparse validation. Never silently repairs the ledger."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from data.acquisition.cache import write_manifest
from data.nse_results_ingest import parse_xbrl_metrics
from data.pit_fundamentals import _load_rows

COMPARE_FIELDS = (
    "revenue_from_operations",
    "profit_before_tax",
    "profit_after_tax",
    "basic_eps",
    "operating_profit",
)
TOL_ABS = {
    "revenue_from_operations": 1.0,
    "profit_before_tax": 1.0,
    "profit_after_tax": 1.0,
    "basic_eps": 0.05,
    "operating_profit": 1.0,
}


def _close(a, b, field: str) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb:
        return True
    return abs(fa - fb) <= TOL_ABS.get(field, 0.0) or (
        fb != 0 and abs(fa / fb - 1.0) <= 0.01
    )


def _second_parser(xml_bytes: bytes) -> dict[str, float]:
    """Independent pass: last non-empty mapped tag wins (vs first-wins)."""
    from xml.etree import ElementTree as ET
    from data.nse_results_ingest import _XBRL_MAP, _local_name

    root = ET.fromstring(xml_bytes)
    found: dict[str, float] = {}
    for el in root.iter():
        field = _XBRL_MAP.get(_local_name(el.tag))
        if not field:
            continue
        text = (el.text or "").strip()
        if not text:
            continue
        try:
            found[field] = float(text)
        except ValueError:
            continue
    return found


def validate_sample(*, sample_n: int = 80, seed: str = "qt-phase2-validation") -> dict[str, Any]:
    rows = [r for r in _load_rows() if r.get("xbrl_url") and r.get("raw_hash")]
    rows.sort(key=lambda r: (r.get("symbol") or "", r.get("period_end") or "", r.get("row_id") or ""))
    # Deterministic sample across the sorted ledger.
    if not rows:
        return {"sample_size": 0, "note": "empty ledger"}
    step = max(1, len(rows) // sample_n)
    sample = rows[::step][:sample_n]

    exact = 0
    tolerant = 0
    disagree = 0
    missing_raw = 0
    field_stats = {f: Counter() for f in COMPARE_FIELDS}
    disagreements: list[dict] = []
    causes = Counter()

    from data.nse_results_ingest import _xbrl_cache_path
    from pathlib import Path
    raw_dir = Path(__file__).resolve().parents[2] / "logs" / "acquisition" / "raw" / "xbrl"

    for row in sample:
        cache = _xbrl_cache_path(str(row.get("xbrl_url") or ""))
        blob = None
        if cache.exists():
            blob = cache.read_bytes()
        else:
            digest = str(row.get("raw_hash") or "")
            alt = raw_dir / f"{digest[:20]}.xml"
            if alt.exists():
                blob = alt.read_bytes()
        if not blob:
            missing_raw += 1
            causes["missing_raw_xbrl"] += 1
            continue
        first = parse_xbrl_metrics(blob)
        second = _second_parser(blob)
        row_exact = True
        row_tol = True
        for field in COMPARE_FIELDS:
            a = row.get(field)
            b = first.get(field)
            c = second.get(field)
            if a is None and b is None:
                field_stats[field]["both_missing"] += 1
                continue
            if a is not None and b is not None and float(a) == float(b):
                field_stats[field]["exact"] += 1
            elif _close(a, b, field):
                field_stats[field]["tolerance"] += 1
                row_exact = False
            else:
                field_stats[field]["disagree"] += 1
                row_exact = False
                row_tol = False
                disagreements.append({
                    "symbol": row.get("symbol"),
                    "period_end": row.get("period_end"),
                    "field": field,
                    "ledger": a,
                    "reparse_first": b,
                    "reparse_last": c,
                })
            if b is not None and c is not None and float(b) != float(c):
                causes["first_vs_last_tag_differs"] += 1
        if row_exact:
            exact += 1
        elif row_tol:
            tolerant += 1
        else:
            disagree += 1
            causes["value_mismatch"] += 1

    compared = exact + tolerant + disagree
    out = {
        "sample_size": len(sample),
        "compared_with_raw": compared,
        "missing_raw": missing_raw,
        "fields_compared": list(COMPARE_FIELDS),
        "exact_match_rate": (exact / compared) if compared else None,
        "tolerance_match_rate": ((exact + tolerant) / compared) if compared else None,
        "disagreement_rate": (disagree / compared) if compared else None,
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "common_causes": dict(causes),
        "unresolved_discrepancies": disagreements[:40],
        "unresolved_count": len(disagreements),
        "secondary_source": (
            "Same official XBRL instance, last-wins tag pass. "
            "Disagreement with last-wins is expected when FourD YTD shares "
            "the quarter's dates; parser prefers NSE OneD (current period). "
            "Ledger is not auto-repaired from last-wins."
        ),
        "parser_context_policy": "prefer_OneD_current_period",
        "seed": seed,
        "sample_hash": hashlib.sha256(
            "|".join(r.get("row_id") or "" for r in sample).encode()
        ).hexdigest()[:16],
    }
    write_manifest("fundamentals_validation", out)
    return out
