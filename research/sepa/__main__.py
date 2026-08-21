"""python -m research.sepa  — research ablation / replay. No orders."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SEPA-001 research runner (no live orders)")
    p.add_argument("cmd", nargs="?", default="ablation", choices=("ablation", "replay", "eval"))
    p.add_argument("--max-symbols", type=int, default=80)
    p.add_argument("--sample-step", type=int, default=10)
    p.add_argument("--lookback", type=int, default=320)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--symbol", default="")
    p.add_argument("--as-of", default="")
    args = p.parse_args(argv)

    if args.cmd == "eval":
        from research.sepa import evaluate_sepa_eligibility
        if not args.symbol or not args.as_of:
            print("eval requires --symbol and --as-of", file=sys.stderr)
            return 2
        el = evaluate_sepa_eligibility(args.symbol, args.as_of)
        print(json.dumps(el.to_dict(), indent=2, default=str))
        return 0
    if args.cmd == "replay":
        from research.sepa.replay import format_replay, try_live_examples
        rows = try_live_examples()
        print(format_replay(rows) if rows else "No bhavcopy frames loaded.")
        return 0
    from research.sepa.ablation import persist, run_ablation
    payload = run_ablation(
        max_symbols=args.max_symbols,
        sample_step=args.sample_step,
        lookback_sessions=args.lookback,
        horizon=args.horizon,
    )
    path = persist(payload)
    print(json.dumps({"wrote": str(path), "sample": payload.get("sample"),
                      "pit": payload.get("pit"),
                      "variants": {k: {"n": v.get("n"), "expectancy_r": v.get("expectancy_r"),
                                       "harness": (v.get("harness") or {}).get("verdict")}
                                   for k, v in payload.get("variants", {}).items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
