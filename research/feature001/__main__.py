"""python -m research.feature001"""
from __future__ import annotations

import argparse
import json

from research.feature001.analyze import run_analysis
from research.feature001.constants import OUT_DIR
from research.feature001.dataset import persist, replay
from research.feature001.report import write_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FEATURE-001 Trend/RS attribution (research only)")
    p.add_argument("--max-dates", type=int, default=None)
    args = p.parse_args(argv)
    payload = replay(max_dates=args.max_dates)
    persist(payload)
    stats = run_analysis(payload["events"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stat_path = OUT_DIR / "feature_001_stats.json"
    stat_path.write_text(json.dumps(stats, indent=2, default=str))
    write_all(stats, payload["meta"])
    print(
        f"FEATURE-001 done events={stats.get('n_events')} "
        f"trend={stats.get('final_trend')} rs={stats.get('final_rs')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
