"""python -m research.edge006"""
from __future__ import annotations

import json
import logging
import sys


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from research.edge006.analyze import analyse
    from research.edge006.report import write_all
    from research.edge006.study import run_study

    argv = list(argv or [])
    if argv[:1] != ["analyze"]:
        print("EDGE-006 study start", flush=True)
        run_study()
    stats = analyse()
    reports = write_all(stats)
    print(json.dumps({"decision": (stats.get("decision") or {}).get("label"),
                      "n_primary": (stats.get("primary") or {}).get("n"),
                      "reports": reports}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
