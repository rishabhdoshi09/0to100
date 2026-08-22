"""Phase II ingest entrypoint. Network allowed here; not in EvidenceSnapshot."""
from __future__ import annotations

import argparse
import sys
from datetime import date


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", action="store_true")
    p.add_argument("--xbrl", action="store_true")
    p.add_argument("--universe", action="store_true")
    p.add_argument("--indices", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--xbrl-min-year", type=int, default=2019)
    p.add_argument("--xbrl-max", type=int, default=None)
    args = p.parse_args(argv)
    if args.all:
        args.results = args.xbrl = args.universe = args.indices = True

    raw = []
    if args.results or args.xbrl:
        from data.acquisition.results_run import ingest_results_metadata
        print("results_metadata_start", flush=True)
        m = ingest_results_metadata(start=date(2016, 1, 1), end=date(2026, 8, 21))
        print(
            "results_metadata_done",
            m.get("normalized_event_rows"),
            m.get("event_symbols"),
            m.get("raw_rows"),
            flush=True,
        )
        raw = m.get("raw") or []

    if args.xbrl:
        from data.acquisition.results_run import ingest_xbrl
        print("xbrl_start", len(raw), flush=True)
        x = ingest_xbrl(raw, workers=10, min_year=args.xbrl_min_year, max_files=args.xbrl_max)
        print("xbrl_done", x.get("normalized_row_count"), x.get("failed_objects"), flush=True)

    if args.universe:
        from data.acquisition.universe_run import ingest_official_identity_and_universe
        print("universe_start", flush=True)
        print(ingest_official_identity_and_universe(), flush=True)

    if args.indices:
        from data.acquisition.index_backfill import backfill
        print("index_start", flush=True)
        print(backfill(), flush=True)

    print("phase2_done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
