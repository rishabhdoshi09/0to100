"""python -m research.edge001 — reconstruct + analyse. No orders."""
from __future__ import annotations

import json
import logging
import sys


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from research.edge001.analyze import analyse
    from research.edge001.report import write_all
    from research.edge001.study import run_study

    argv = list(argv or [])
    if argv[:1] == ["analyze"]:
        stats = analyse()
    else:
        print("EDGE-001 study start", flush=True)
        run_study()
        stats = analyse()
    reports = write_all(stats)
    decision = (stats.get("decision") or {}).get("label")
    print(json.dumps({
        "decision": decision,
        "n_primary": (stats.get("primary") or {}).get("n"),
        "reports": reports,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
