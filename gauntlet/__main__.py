"""
E2 — the one command:  python -m gauntlet

Validate the data (abort on any failure), freeze the config, build the ledger,
run the battery, register the experiment, and print the committee report. Exit
code is non-zero on ABORT so a pipeline can gate on trustworthy data.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gauntlet",
                                 description="Run the historical gauntlet.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--factors", action="store_true",
                    help="also test factor-neutral alpha (requires factor data)")
    ap.add_argument("--skip-validation", action="store_true",
                    help="DANGER: run without the dataset gate (never for real evidence)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    from gauntlet.runner import run_gauntlet
    from gauntlet.report import build_report, to_markdown

    raw = run_gauntlet(seed=args.seed, factors_enabled=args.factors,
                       skip_validation=args.skip_validation)
    report = build_report(raw)

    if args.json:
        import json
        print(json.dumps(report, indent=2, default=str))
    else:
        print(to_markdown(report))

    if report.get("status") == "ABORTED":
        print("\n⛔ Gauntlet aborted — fix the dataset checks above, then rerun.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
