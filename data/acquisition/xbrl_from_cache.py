"""Build fundamentals from already-cached XBRL + results JSON (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from data.acquisition import PARSER_VERSION
from data.acquisition.cache import sha256_bytes
from data.acquisition.results_run import _enrich_fund_row
from data.nse_results_ingest import _xbrl_cache_path, parse_xbrl_metrics, select_xbrl_candidates
from data.pit_fundamentals import merge_fundamentals


def load_cached_result_rows() -> list[dict]:
    d = Path(__file__).resolve().parents[2] / "logs" / "acquisition" / "raw" / "results"
    rows: list[dict] = []
    if not d.exists():
        return rows
    for p in d.glob("*.json"):
        if p.name.endswith(".meta.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend(data)
    return rows


def materialize_from_cache(*, min_year: int = 2019) -> dict:
    raw = load_cached_result_rows()
    cands = select_xbrl_candidates(raw, min_period_end_year=min_year)
    built = []
    skipped = 0
    for raw_row in cands:
        url = str(raw_row.get("xbrl") or "")
        cache = _xbrl_cache_path(url)
        if not cache.exists() or cache.stat().st_size < 500:
            skipped += 1
            continue
        blob = cache.read_bytes()
        try:
            metrics = parse_xbrl_metrics(blob)
        except Exception:
            skipped += 1
            continue
        if not metrics:
            skipped += 1
            continue
        row = _enrich_fund_row(raw_row, metrics, sha256_bytes(blob))
        if row:
            built.append(row)
    status = merge_fundamentals(built, source="nse_xbrl_financial_results")
    status["cache_candidates"] = len(cands)
    status["cache_built"] = len(built)
    status["cache_missing"] = skipped
    status["parser_version"] = PARSER_VERSION
    return status
