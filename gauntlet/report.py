"""
E3 — Evidence Report.

Turns the runner's raw results into one reproducible document a committee could
sit with: for every strategy the full statistical picture, and up top the
assumptions and known limitations stated plainly. No spin — a FAIL is written as
a FAIL. Returned as a dict (machine-readable, persisted to logs/gauntlet/) and a
markdown rendering.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent.parent / "logs" / "gauntlet"

# The honest caveats that apply to EVERY historical gauntlet result. These are not
# boilerplate — each is a real reason a committee should discount the number.
ASSUMPTIONS = [
    "Costs are MODELED, not measured: ~0.22% delivery + 0.10% slippage round-trip. "
    "Slippage is an assumption (E0 evidence) until reconciled against real fills.",
    "Breakout fills are simulated AT the pivot; real breakouts can gap through it, "
    "so realised entries may be worse than modeled (optimistic bias).",
    "Benchmark alpha is measured vs a single index (Nifty). Factor-neutral alpha "
    "(size/value/momentum/low-vol) is only tested when factors are explicitly enabled.",
    "Trades sharing a signal are attributed to that signal; correlated trades are "
    "accounted for via effective-N and the block bootstrap, not assumed independent.",
]

LIMITATIONS = [
    "No capacity / market-impact model — results are gross of the slippage that "
    "grows with size; capacity must be measured before real capital.",
    "Regime coverage is only as broad as the index history; ≥2 regimes requires the "
    "window to span a real drawdown, not just a bull tape.",
    "This is historical (in-sample to the strategy design). Only forward paper "
    "trading (E4 evidence) tests true out-of-sample performance.",
]


def build_report(results: dict, assumptions=None, limitations=None) -> dict:
    """Assemble the committee report dict from a runner result."""
    if results.get("aborted"):
        return {"status": "ABORTED", "reason": results.get("reason"),
                "validation": results.get("validation"),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    strategies = results.get("strategies", {})
    tally = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
    for s in strategies.values():
        tally[s.get("verdict", "INCONCLUSIVE")] = tally.get(
            s.get("verdict", "INCONCLUSIVE"), 0) + 1
    report = {
        "status": "COMPLETE",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": results.get("experiment"),
        "freeze_hash": (results.get("freeze") or {}).get("hash"),
        "n_trades": results.get("n_trades"),
        "n_strategies_tested": len(strategies),
        "strategies_tried_for_deflation": results.get("n_trials"),
        "reality_check_p": results.get("reality_check_p"),
        "verdict_tally": tally,
        "strategies": strategies,
        "assumptions": assumptions or ASSUMPTIONS,
        "known_limitations": limitations or LIMITATIONS,
    }
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        eid = (results.get("experiment") or {}).get("experiment_id", "adhoc")
        (_DIR / f"report_{eid}.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    return report


def to_markdown(report: dict) -> str:
    if report.get("status") == "ABORTED":
        lines = ["# Historical Gauntlet — ABORTED", "",
                 f"**Reason:** {report.get('reason')}", "",
                 "## Failed dataset checks"]
        for c in (report.get("validation", {}) or {}).get("checks", []):
            mark = "✅" if c["ok"] else "❌"
            lines.append(f"- {mark} `{c['check']}` — {c['detail']}")
        return "\n".join(lines)
    t = report["verdict_tally"]
    lines = ["# Historical Gauntlet — Evidence Report", "",
             f"_Generated {report['generated_at']} · experiment "
             f"`{(report.get('experiment') or {}).get('experiment_id','?')}` · "
             f"git `{(report.get('experiment') or {}).get('git_commit','?')[:8]}` · "
             f"freeze `{report.get('freeze_hash')}`_", "",
             f"**Trades:** {report['n_trades']} · **Strategies:** "
             f"{report['n_strategies_tested']} · **Reality-Check p:** "
             f"{report.get('reality_check_p')}", "",
             f"## Verdicts — PASS {t['PASS']} · FAIL {t['FAIL']} · "
             f"INCONCLUSIVE {t['INCONCLUSIVE']}", ""]
    for name, s in sorted(report["strategies"].items(),
                          key=lambda kv: (kv[1]["verdict"] != "PASS", kv[0])):
        eq = s.get("equity", {})
        lines += [
            f"### {name} — **{s['verdict']}**",
            f"- n={s['n']} (effective {s['n_effective']}) · "
            f"expectancy {s['expectancy_r']:+}R · CI "
            f"[{s['ci_lower']:+}, {s['ci_upper']:+}]R "
            f"({'excludes 0' if s['ci_excludes_zero'] else 'includes 0'})",
            f"- PF {s['profit_factor']} · Sharpe {s['sharpe']} · "
            f"Deflated Sharpe {s['deflated_sharpe']} · p={s['p_value']:.4g} · "
            f"FDR-significant: {s.get('fdr_significant')}",
            f"- alpha {s['alpha']} · beta {s['beta']} · beats benchmark: "
            f"{s['beats_benchmark']} (tested: {s['benchmark_tested']})",
            f"- total {eq.get('total_R')}R · max DD {eq.get('max_drawdown_R')}R "
            f"({eq.get('max_drawdown_pct')}%) · modeled CAGR "
            f"{eq.get('modeled_cagr_pct')}%",
            f"- regimes: {s.get('regime_breakdown')}",
            f"- _{s.get('harness_insight','')}_", ""]
    lines += ["## Assumptions"]
    lines += [f"- {a}" for a in report["assumptions"]]
    lines += ["", "## Known limitations"]
    lines += [f"- {l}" for l in report["known_limitations"]]
    return "\n".join(lines)
