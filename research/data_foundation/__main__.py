"""python -m research.data_foundation — write catalog-adjacent reports (no network ingest)."""
from __future__ import annotations

import json

from research.data_foundation.manifest import build_manifest
from data.benchmarks import catalog
from data.ca_research import research_status
from data.listing_archive import universe_pit_class
from data.sector_map import coverage as sector_coverage


def main() -> int:
    print(json.dumps({
        "manifest": build_manifest(as_of="2026-08-21"),
        "benchmarks": catalog(),
        "ca": {k: research_status()[k] for k in (
            "events", "symbols", "ca_research_acceptable", "ca_complete", "label", "status",
        )},
        "universe": universe_pit_class(),
        "sector": sector_coverage(),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
