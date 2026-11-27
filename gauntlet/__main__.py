"""
E2 — the one command:  python -m gauntlet

Validate the data (abort on any failure), freeze the config, build the ledger,
run the battery, register the experiment, and print the committee report. Exit
code is non-zero on ABORT so a pipeline can gate on trustworthy data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    ap.add_argument("--geometry-checkpoint", type=Path, help="SQLite checkpoint for the local geometry study")
    ap.add_argument("--development-before", help="First date of the untouched evaluation period")
    ap.add_argument("--evaluation-before", help="Exclusive end date of the evaluation period")
    ap.add_argument("--research-start", help="First archive date for exploratory scoped equity research")
    ap.add_argument("--stocks", help="Comma-separated, explicitly named operating-company NSE symbols")
    args = ap.parse_args(argv)
    if args.geometry_checkpoint:
        if args.skip_validation or args.factors or not args.development_before or not args.evaluation_before:
            ap.error("geometry requires both date boundaries and dataset validation")
        from gauntlet.geometry import run

        if bool(args.research_start) != bool(args.stocks):
            ap.error("scoped research requires both --research-start and --stocks")
        report = run(args.geometry_checkpoint, cutoff=args.development_before, end=args.evaluation_before,
                     research_start=args.research_start,
                     stocks=args.stocks.split(",") if args.stocks else None)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report.get("status") == "ABORTED" else 0
    if args.development_before or args.evaluation_before or args.research_start or args.stocks:
        ap.error("date boundaries require --geometry-checkpoint")

    from gauntlet.runner import run_gauntlet
    from gauntlet.report import build_report, to_markdown

    raw = run_gauntlet(seed=args.seed, factors_enabled=args.factors,
                       skip_validation=args.skip_validation)
    report = build_report(raw)

    if args.json:
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
